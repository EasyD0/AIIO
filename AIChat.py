import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Iterable, Protocol

from aiohttp import ClientConnectorError, ClientSession, ServerDisconnectedError

from AIIO import AIInput, AIOutput, AIOutput_location
from MyPyLib.FuncAnalysis import FuncAnalyser
from MyPyLib.LogSet import logSetup
from MyPyLib.Preprocessor import Preprocessor
from for_sqllite import sqlite_data, SQLiteServe
from info_for_llm_review import get_func_info
from prompt.prompt_chat import PromptGenerator
from utils import ProgressBar

logger = logSetup(__file__)


class Model(Enum):
    MiniMax = "GLM-5.3"
    Qwen = "Opus-4.8"


@dataclass
class ReviewContext:
    """走查上下文，封装依赖和配置"""

    reviewer_model: str = "GLM-5.3"
    checker_model: str = "GLM-5.3"
    session: ClientSession | None = None
    executor: ThreadPoolExecutor | None = None
    _owns_session: bool = False
    _owns_executor: bool = False

    # def __post_init__(self):
    #     if self.session is None:
    #         self.session = ClientSession()
    #         self._owns_session = True
    #     if self.executor is None:
    #         self.executor = ThreadPoolExecutor(max_workers=4)
    #         self._owns_executor = True

    async def __aenter__(self):
        if self.session is None:
            self.session = ClientSession()
            self._owns_session = True
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=4)
            self._owns_executor = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owns_executor and self.executor:
            self.executor.shutdown(wait=True)
        if self._owns_session and self.session is not None:
            await self.session.close()


class AIReviewer_async(Protocol):
    async def do(self, ai_input: AIInput) -> list[AIOutput]: ...


class AIChecker_async(Protocol):
    async def do(self, ai_input: AIInput, ai_output: AIOutput) -> AIOutput: ...


def str_similarity(x, y) -> float:
    """
    计算编辑距离相似度
    """
    x, y = x.strip(), y.strip()
    if x == y or x.lower() == y.lower():
        return 1.0

    m, n = len(x), len(y)
    max_len = max(m, n)

    # dp[i][j] 表示子串x[0:i] 和 子串 y[0:j]的距离
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    return lcs_len / max_len if max_len > 0 else 1.0


class _BaseChatYC:
    """
    走查类与检查类共用对话器
    """

    url = "https://openai/v1/chat/"
    tokenList = [
        r"Bearer sk-xxxx",
        r"Bearer sk-xxxx",
        r"Bearer sk-xxxx",
    ]

    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9",
        "content-type": "application/json",
    }

    # 共享状态，保证所有审查/检查实例全局轮换
    _token_idx = 0
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, session: ClientSession | None = None):
        self.session: ClientSession = session
        if self.session is None:
            self.session = ClientSession()

    async def get_next_token(self) -> str:
        async with self._lock:
            token = self.tokenList[_BaseChatYC._token_idx]
            _BaseChatYC._token_idx = (_BaseChatYC._token_idx + 1) % len(self.tokenList)
            return token

    async def requestAI(self, content: str) -> str | None:
        cur_token = await self.get_next_token()
        cur_headers = self.headers.copy()
        cur_headers["authorization"] = cur_token

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.6,
        }
        try:
            async with self.session.post(
                self.url, json=data, headers=cur_headers, timeout=2000
            ) as response:
                try:
                    response.raise_for_status()
                    if response.status == 200:
                        logger.debug("请求成功")
                    else:
                        logger.error(f"请求失败：{response.status}-{response.text}")
                        return None
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                except Exception as e:
                    logger.error(f"请求AI发生异常: {e}")

        except asyncio.TimeoutError:
            logger.error(f"请求AI连接超时: {e}")
        except ClientConnectorError:
            logger.error(f"AI连接失败: {e}")
        except ServerDisconnectedError as e:
            logger.error(f"AI服务器断开连接: {e}")
        except Exception as e:
            logger.error(f"请求AI发生异常: {type(e).__name__}: {e}")

        return None


class arequestChatYC(_BaseChatYC):
    """
    走查(AI审查)实现
    """

    def __init__(
        self,
        rule: str = "AI走查-数组越界",
        model: str = "GLM-5.3",
        session: ClientSession | None = None,
    ):
        super().__init__(session)
        self.model = model  # 默认模型
        self.rule = rule  # 默认规则为代码走查
        self.base_prompt = PromptGenerator.get_prompt_for_reviewer(rule)  # 基础提示词

    def _convert_input_to_string(self, ai_input: AIInput) -> str:
        """
        base_prompt 在外部提供
        """
        warnings.warn(
            "已弃用， 使用AIInput.to_string_for_request代替。",
            DeprecationWarning,
            stacklevel=2,
        )

        if isinstance(ai_input, AIInput):
            tmp_dict = ai_input.to_dict()
        else:
            tmp_dict = ai_input

        # 提取字段
        func_body_orig = tmp_dict.get("Function_orig") or "/* 无函数体 */"
        func_body_prep = tmp_dict.get("Function") or "/* 暂未提供 */"
        declares = tmp_dict.get("Declare", [])
        call_funcs = tmp_dict.get("Call_Function", [])

        # 合并声明和调用函数（去除重复空行，保留结构）
        declare_str = "\n".join(declares).strip() if declares else "/* 暂未提供 */"
        call_func_str = (
            "\n".join(call_funcs).strip() if call_funcs else "/* 暂未提供 */"
        )

        # 拼接结果
        result = (
            (f"**函数体**:\n```\n{func_body_orig}\n```\n\n")
            + (f"**预处理后函数体**:\n```\n{func_body_prep}\n```\n\n")
            + (f"**相关声明和定义**:\n```\n{declare_str}\n```\n\n")
            + (f"**被调用函数**:\n```\n{call_func_str}\n```\n")
        )
        return result

    def _process_response(
        self, raw_text: str | None, ai_input: AIInput
    ) -> list[AIOutput]:
        filename = ai_input.file or "未知文件路径"
        func_name = ai_input.FunctionName or "未知函数名"

        if not isinstance(raw_text, str) or not raw_text:
            logger.error("AI结果无效")
            return [
                AIOutput(
                    file=filename,
                    FunctionName=func_name,
                    description="AI接口出错, 无法返回结果",
                    ruleCode=self.rule,
                    title=self.rule,
                    chatAIOriginalRes="None",
                    error_flag=True,
                )
            ]
        logger.debug(f"AI回复的原始信息为:\n{raw_text}")

        # 无问题, AI有结果
        if "未发现高置信度问题" in raw_text:
            return [
                AIOutput(
                    file=filename,
                    FunctionName=func_name,
                    description="未发现高置信度问题",
                    ruleCode=self.rule,
                    title=self.rule,
                    chatAIOriginalRes=raw_text,
                )
            ]
        # 无效结果, AI结果异常
        elif not all(["## 代码走查报告" in raw_text, "问题详情和越界分析" in raw_text]):
            return [
                AIOutput(
                    file=filename,
                    FunctionName=func_name,
                    description="AI接口出错, 结果格式异常",
                    ruleCode=self.rule,
                    title=self.rule,
                    chatAIOriginalRes=raw_text,
                    error_flag=True,
                )
            ]

        # 有效结果
        promble_info = raw_text.split("## 代码走查报告")[-1]
        problems_info = promble_info.split("### 问题")
        res: list[AIOutput] = []
        for one_problem_str in problems_info:
            if not "问题行处的原始代码" in one_problem_str:
                continue

            # 提取问题行号, 问题行号从1开始 (提示词约定)
            try:
                line_num = (
                    one_problem_str.split("问题行处的原始代码")[0]
                    .split("问题行号")[1]
                    .strip(":：\n\t ")
                )
            except:
                line_num = ""  # 无效行号

            # 提取问题原始代码
            try:
                raw_code = (
                    one_problem_str.split("问题详情和越界分析")[0]
                    .split("问题行处的原始代码")[1]
                    .strip(":：\n\t ")
                )
                raw_code = raw_code.strip(" `\n\t\r")
            except:
                raw_code = ""

            # 提取问题描述
            try:
                problem_des = (
                    one_problem_str.split("风险等级")[0]
                    .split("问题详情和越界分析")[1]
                    .strip(":：\n\t ")
                )
            except:
                problem_des = ""

            # 提取问题等级
            try:
                level = (
                    one_problem_str.split("修复建议")[0]
                    .split("风险等级")[1]
                    .strip(":：\n\t ")
                )
            except:
                level = ""

            # 提取问题建议
            try:
                suggestion = (
                    one_problem_str.split("修复建议")[1]
                    .strip(":：\n\t ")
                    .splitlines()[0]
                    .strip(":：\n\t ")
                )
            except:
                suggestion = ""

            # 计算问题行号 (绝对行号)
            try:
                line_num = max(int(line_num) - 1, 0) + ai_input.begin_line
            except:
                line_num = 0  # 表示无效行号，在最后表格处理中，禁止为0的行号

            one_item = AIOutput(
                file=filename,
                FunctionName=func_name,
                location=AIOutput_location(startLine=line_num),
                illegalCode=raw_code,
                description=problem_des,
                severity=level,
                ruleCode=self.rule,
                title=self.rule,
                chatAIOriginalRes=one_problem_str,
                suggestion=suggestion,
                valid_flag=True,
            )

            res.append(one_item)

        # 若没有生成合法结果
        if not res:
            one_item = AIOutput(
                file=filename,
                FunctionName=func_name,
                description=f"AI输出内容非法, 原始输出为:\n{raw_text}",
                ruleCode=self.rule,
                title=self.rule,
                chatAIOriginalRes=raw_text,
                error_flag=True,
            )
            res.append(one_item)
        return res

    def _find_similar_neighbor(
        self, func_body_orig: str, raw_line: str, line_number: int
    ) -> int:
        """
        附近是否存在高相似度行号, 如果存在则直接使用, 不用全局搜索

        :param func_body_orig: 函数体
        :param raw_line: 问题代码文本
        :line_number: 相对行号, 从0开始
        """
        if not isinstance(func_body_orig, str) or not isinstance(line_number, int):
            logger.error("类型错误")
            return -2

        theta = 0.9  # 置信度阈值
        line_num_max_offset = 5
        orig_code_list = func_body_orig.splitlines()

        # def max_min(cur, lower, upper):
        #     return min(max(lower, cur), upper)

        # 最大最小行号
        # max_num = len(orig_code_list)
        # min_num = 0
        #
        # candidate_nums = range(
        #     max_min(line_number - line_num_max_offset, min_num, max_num - 1),
        #     max_min(line_number + line_num_max_offset, min_num, max_num),
        # )

        l = max(0, line_number - line_num_max_offset)
        r = line_number + line_num_max_offset
        candidate_code = orig_code_list[l : r + 1]

        if not candidate_code:
            return -1

        for idx, ca_code in enumerate(candidate_code, 0):
            if str_similarity(raw_line, ca_code) >= theta:
                return idx + l

        return -1

    def _correctLineNum(self, ai_res: AIOutput, ai_input: AIInput):
        """
        修正行号, 修正代码行, 原地修改
        """
        if not ai_res.illegalCode:
            return

        neighbor_line_num = self._find_similar_neighbor(
            ai_input.Function_orig,
            ai_res.illegalCode,
            ai_res.startLine - ai_input.begin_line,  # 从0开始的相对行号
        )

        if neighbor_line_num >= 0:
            logger.debug(f"{str(ai_res)} 使用附近行号 {neighbor_line_num}")
            ai_res.startLine = neighbor_line_num + ai_input.begin_line
            return

        if neighbor_line_num == -2:
            logger.debug(f"{str(ai_res)} 类型错误无法修正行号")
            return

        logger.debug(f"{str(ai_res)} 需要修正行号")

        orig_code = ai_input.Function_orig

        # 精准匹配
        for line_num, line in enumerate(orig_code.splitlines()):
            if line.strip(" \t\r") == ai_res.illegalCode:
                ai_res.startLine = line_num + ai_input.begin_line
                logger.debug(f"{str(ai_res)} 已修正行号")
                return

        # 若无法精准匹配，则次优匹配
        similarities: list[float] = [
            str_similarity(line.strip(" \t\r"), ai_res.illegalCode)
            for line in orig_code.splitlines()
        ]

        if max(similarities) >= 0.8:
            ai_res.startLine = (
                similarities.index(max(similarities)) + ai_input.begin_line
            )
            logger.debug(f"{str(ai_res)} 已修正行号")
        else:
            logger.warning(f"{str(ai_res)} 无法修正行号")

    async def do(self, ai_input: AIInput) -> list[AIOutput]:
        """
        data: 格式形如 input.json
        begin_line: 行号, 用来处理相对位置
        """
        # 错误的输入直接标记
        if ai_input.error_flag:
            return [AIOutput.from_bad_input(ai_input, self.rule, self.rule)]

        for_request = self._convert_input_to_string(ai_input)
        logger.debug(f"即将请求AI-{ai_input}")
        tmp_res: str | None = await self.requestAI(self.base_prompt + for_request)
        if tmp_res is None:
            logger.debug(f"请求AI失败-{ai_input}-{self.rule}")
        else:
            logger.debug(f"请求AI成功-{ai_input}-{self.rule}")

        # 解析结果
        res: list[AIOutput] = self._process_response(tmp_res, ai_input)

        # 修正行号
        [self._correctLineNum(r, ai_input) for r in res]
        return res


class acheckChatYC(_BaseChatYC):
    """
    误报检查实现
    """

    def __init__(
        self,
        rule: str = "AI走查-数组越界",
        model: str = "GLM-5.3",
        session: ClientSession | None = None,
    ):
        super().__init__(session)
        self.model = model  # 默认模型
        self.rule = rule  # 默认规则为代码走查
        self.base_prompt = PromptGenerator.get_prompt_for_checker(rule)  # 基础提示词

    def _convert_input_to_string(self, ai_input: AIInput, ai_output: AIOutput) -> str:
        """
        将 self.base_prompt 和 ai_input, ai_output融合

        ## Part 1: 代码上下文
        函数体:
        {func_body}

        预处理后函数体：
        {func_body_preprocessed}

        相关声明:
        {declare_str}

        被调用函数:
        {call_func_str}

        ## Part 2: 走查报告详情
        ### 问题
        问题行处的原始代码：{raw_code}
        问题详情和越界分析: {problem_info}
        修复建议: {suggestion}
        """

        # 提取字段
        func_body_prep = ai_input.Function or "/* 无函数体 */"
        declares = ai_input.Declare or []
        call_funcs = ai_input.Call_Function or []

        # 合并声明和调用函数（去除重复空行，保留结构）
        declare_str = "\n".join(declares).strip() if declares else "/* 暂未提供 */"
        call_func_str = (
            "\n".join(call_funcs).strip() if call_funcs else "/* 暂未提供 */"
        )

        # 拼接结果
        part1 = (
            "## Part 1: 代码上下文\n"
            + (f"**函数体**:\n```\n{func_body_prep}\n```\n\n")
            + (f"**相关声明和定义**:\n```\n{declare_str}\n```\n\n")
            + (f"**被调用函数**:\n```\n{call_func_str}\n```\n")
        )

        raw_code = ai_output.illegalCode
        problem_info = ai_output.description
        suggestion = ai_output.suggestion
        part2 = (
            "## Part 2: 走查报告详情\n"
            + f"问题行处的原始代码：{raw_code}\n"
            + f"问题详情和越界分析: {problem_info}\n"
            + f"修复建议: {suggestion}"
        )
        return self.base_prompt + part1 + "\n" + part2

    def _process_response(
        self,
        raw_text,
        ai_output: AIOutput,
    ) -> AIOutput:
        """
        处理AI答复raw_text, 然年信息填入到 ai_output的false_alarm中
        """
        try:
            alarm_str = (
                raw_text.split("是否误报")[1].split("误报分析")[0].strip("#:：\n\t ")
            )
            if "是" in alarm_str:
                ai_output.false_alarm = True
                logger.info("已将一处标记为误报")
            elif "否" in alarm_str:
                ai_output.false_alarm = False
                logger.info("已将一处标记为非误报")
        except Exception as e:
            ai_output.false_alarm = False
            ai_output.false_alarm_des = "发生错误: 在提取误报标志时"
            logger.error(f"发生错误: {ai_output} 在提取误报标志时, {e}")

        if not ai_output.false_alarm and not ai_output.false_alarm_des:
            return ai_output

        try:
            alarm_des: str = raw_text.split("误报分析")[1].strip("#:：\n\t ")
            while "如果是误报则填写" in alarm_des:
                alarm_des = "\n".join(alarm_des.splitlines()[1:])

            if ai_output.false_alarm_des:
                ai_output.false_alarm_des += "\n" + alarm_des

        except Exception as e:
            if ai_output.false_alarm_des:
                ai_output.false_alarm_des += "\n" + "发生错误: 在提取误报描述时"
            else:
                ai_output.false_alarm_des = "发生错误: 在提取误报标志时"
            logger.error(
                f"发生错误: {ai_output} 在提取误报标志时 {e}, 程序忽略该错误继续执行, 并保留被检查对象"
            )
        return ai_output

    async def do(self, ai_input: AIInput, ai_output: AIOutput) -> AIOutput:
        if ai_output.valid_flag or ai_output.error_flag:
            ai_output.false_alarm = True
            ai_output.false_alarm_des = f"走查报告无效, 无需检查"
            logger.debug(f"无效的走查报告, 无需检查误报, {ai_output}")
            return ai_output

        logger.debug(f"误报检查开始 {ai_output}")
        raw_text = await self.requestAI(
            self._convert_input_to_string(ai_input, ai_output)
        )
        return self._process_response(raw_text, ai_output)


# 将get_func_info 封装为异步函数, TODO可以用装饰器实现
async def get_func_info_async(
    preprocessor: Preprocessor,
    source_file: Path,
    func_name: str,
    executor: ThreadPoolExecutor,
) -> AIInput:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, get_func_info, preprocessor, source_file, func_name, True
    )


async def generate_inputs(
    entity: dict[Path, Iterable],
    por: Preprocessor,
    executor: ThreadPoolExecutor,
    # progress_bar: ProgerssBar = None
    max_inflight: int = 8,  # 在途解析任务上限
) -> AsyncGenerator[AIInput, None]:
    """
    异步生成器：流式生成AIInput

    用有界滑动窗口限制并发解析: 在途解析任务恒 ≤ max_inflight。
    下游AI走查已经被信号量限成并发N, 但"解析函数->AIInput"这一步若不限,
    大工程会一次性把所有函数的解析任务全打进内存(随函数总数线性涨)。
    """
    logger.info("<异步生成器>启动")

    # 惰性展平为 (file, func) 迭代器; funcs 为空才调用 FuncAnalyser
    def gen_pairs():
        for file, funcs in entity.items():
            if not file:
                continue
            if not funcs:
                funcs = FuncAnalyser(file, por=por).func_names
                # 不在这里增加任务数量
            for func in funcs:
                yield file, func

    pairs = iter(gen_pairs())
    pending: list[asyncio.Future] = []

    def refill():
        # 把窗口填满(至多 max_inflight 个在途), 没有更多对就停
        while len(pending) < max_inflight:
            try:
                file, func = next(pairs)
            except StopIteration:
                return
            # ensure_future 包装成 Task(asyncio.wait 要求显式 Task/Future)
            pending.append(
                asyncio.ensure_future(get_func_info_async(por, file, func, executor))
            )

    refill()
    while pending:
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for fut in done:
            try:
                ai_input = await fut
                logger.debug("异步生成器产出一个AIInput")
                yield ai_input
            except Exception as e:
                logger.error(f"生成AIInput失败: {e}")
        # 移除已完成, 再补足窗口, 保证在途量恒定
        pending = [f for f in pending if f not in done]
        refill()

    logger.info("<异步生成器>完成")


@ProgressBar.progress_wrapper
async def process_input(
    ai_input: AIInput,
    rule: str,
    reviewers: dict[str, AIReviewer_async],
    checkers: dict[str, AIChecker_async],
    semaphore: asyncio.Semaphore,
    progress_bar: ProgressBar = None,  # !这个参数不能删除, 留给wrapper使用
) -> list[AIOutput]:
    """
    传入AIInput, 走查规则rule, 并进行检查
    """
    logger.debug(f"对 {ai_input} 进行规则 {rule} 走查")
    res: list[AIOutput] = []

    # AI审查
    async with semaphore:
        review_outputs: list[AIOutput] = await reviewers[rule].do(ai_input)

    # AI检查
    async def check_task(
        ai_input: AIInput,
        ai_output: AIOutput,
        checker: AIChecker_async,
        semaphore: asyncio.Semaphore,
    ) -> AIOutput:
        async with semaphore:
            return await checker.do(ai_input, ai_output)

    if checkers and checkers.get(rule):  # 需要检查
        check_coro_tasks = []
        for ai_output in review_outputs:
            if ai_output.error_flag or ai_output.valid_flag:
                res.append(ai_output)  # 错误条目无需检查
            else:
                check_coro_tasks.append(
                    check_task(ai_input, ai_output, checkers[rule], semaphore)
                )

        tmp_res = await asyncio.gather(*check_coro_tasks, return_exceptions=True)
        for tr in tmp_res:
            if isinstance(tr, Exception):
                logger.exception(f"协程发生异常: {tr}")
            else:
                res.append(tr)
    else:  # 跳过检查
        logger.debug(f"对{ai_input}规则{rule}, 跳过检查走查结果")
        res = review_outputs

    return res


class arequestChatTest:
    def __init__(
        self,
        rule: str = "AI走查-数组越界",
        model: str = "Qwen3.5-397B-A17B",
        session: ClientSession | None = None,
    ):
        self.model = model  # 默认模型
        self.rule = rule  # 默认规则为代码走查
        pass

    async def do(self, ai_input: AIInput) -> list[AIOutput]:
        """
        将随机返回 error_flag 有效/无效的结果
        """
        if ai_input.error_flag:
            return [AIOutput.from_bad_input(ai_input, self.rule, self.rule)]
        filename = ai_input.file or "未知文件路径"
        func_name = ai_input.FunctionName or "未知函数名"

        err_output = AIOutput(
            file=filename,
            FunctionName=func_name,
            description="AI接口出错, 无法返回结果",
            ruleCode=self.rule,
            title=self.rule,
            chatAIOriginalRes="None",
            error_flag=True,
        )

        null_output = AIOutput(
            file=filename,
            FunctionName=func_name,
            description="未发现高置信度问题",
            ruleCode=self.rule,
            title=self.rule,
            chatAIOriginalRes="# 未发现高置信度问题",
        )

        valid_output = AIOutput(
            file=filename,
            FunctionName=func_name,
            location=AIOutput_location(startLine=1),
            illegalCode="int i = 1.1 //错误代码",
            description="问题描述",
            severity="高",
            ruleCode=self.rule,
            title=self.rule,
            chatAIOriginalRes="原始描述",
            suggestion="修复建议",
            valid_flag=True,
        )

        import random

        num = random.randint(0, 2)
        match num:
            case 0:
                res = [err_output]
            case 1:
                res = [null_output]
            case 2:
                res = [valid_output]
            case _:
                res = [valid_output]
        return res


async def TotalTask(
    entity: dict[Path, Iterable],  # 待走查的条目
    review_rules: Iterable[str],  # 要走查的规则集合
    por: Preprocessor,  # 预处理器
    context: ReviewContext,
    git_data: dict,
    _sql: SQLiteServe,
    max_concurrent_ai: int = 4,  # 最大并发AI连接数
    need_check: bool = False,  # 是否需要检查
    progress_bar: ProgressBar = None,
    just_test: bool = False,
) -> tuple[list[AIInput], list[AIOutput]]:
    semaphore = asyncio.Semaphore(max_concurrent_ai)
    logger.info(f"网络并发数量限制为: {max_concurrent_ai}")
    res_input: list[AIInput] = []
    res_output: list[AIOutput] = []

    checker_type = arequestChatYC if not just_test else arequestChatTest
    # 为每个走查规则生成一个走查器对象
    reviewers: dict[str, AIReviewer_async] = {
        rule: checker_type(rule, context.reviewer_model, context.session)
        for rule in review_rules
    }

    # 为每个走查规则生成一个检查器对象
    if just_test or not need_check:
        checkers: dict[str, AIChecker_async] = {}
    else:
        checkers: dict[str, AIChecker_async] = {
            rule: acheckChatYC(rule, context.checker_model, context.session)
            for rule in review_rules
        }

    tasks: list[asyncio.Task] = []
    # 任务 -> (该条目对应的 sqlite_data, 规则名), 用于走查完成后入库
    task_map: dict[asyncio.Task, tuple[sqlite_data, str]] = {}
    async for ai_input in generate_inputs(entity, por, context.executor):
        _sqlite_data = sqlite_data.from_AIInput(ai_input, git_data)
        res_input.append(ai_input)
        for rule in review_rules:
            # 检查下在数据库里是否存在, 如果存在则直接跳过
            if _sqlite_data.in_database(rule, _sql):
                logger.debug("条目已在数据库中")
                # 已走查过(同一仓库+同一身份+同一规则), 直接跳过
                continue

            t = asyncio.create_task(
                process_input(
                    ai_input, rule, reviewers, checkers, semaphore, progress_bar
                ),
                # name=str(ai_input) + "-" + str(rule),
            )
            t.name = str(ai_input) + "-" + str(rule)
            tasks.append(t)
            task_map[t] = (_sqlite_data, rule)

    for t in asyncio.as_completed(tasks):
        try:
            cur_output = await t
            if isinstance(cur_output, list):
                res_output.extend(cur_output)
                # 走查结束后, 只有未发生错误的(该条目+该规则)才能入库,
                # 并把该规则的走查结果随行一并存档
                _sd, rule = task_map.get(t, (None, None))
                if _sd is not None and all(not o.error_flag for o in cur_output):
                    _sd.save_into_database(rule, _sql, list(cur_output))
            else:
                logger.error(
                    f"走查任务发生异常, 返回了一个不可迭代对象类型为{type(cur_output)}"
                )
        except Exception as e:
            if hasattr(t, "name"):
                logger.error(f"走查任务{t.name}发生异常: {e}")
            else:
                logger.error(f"未知走查任务发生异常: {e}")
    return res_input, res_output
