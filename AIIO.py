
from dataclasses import asdict, dataclass, field
from MyPyLib.LogSet import logSetup
from copy import deepcopy

logger = logSetup(__file__)


@dataclass
class AIInput_prompt:
    """提示词模板"""

    role: str = ""
    analysis: str = ""


@dataclass
class AIInput:
    """代码分析请求数据结构"""

    begin_line: int = 1  # 函数体在原始文件中的起始位置
    file: str = ""  # 文件相对路径
    FunctionName: str = ""  # 函数名
    Function: str = ""  # 函数体 (预处理后)
    Function_orig: str = ""  # 函数体 (预处理前)
    Call_Function: list[str] = field(default_factory=list)  # 调用函数
    Declare: list[str] = field(default_factory=list)  # 相关声明(结构体定义/变量声明等)
    prompt_template: AIInput_prompt | None = None  # 额外提示词(用于杜的接口使用)
    error_flag: bool = False  # 生成AIInput过程中是否发生错误

    @classmethod
    def from_dict(cls, orig_dict: dict):
        if "Call Function" in orig_dict and "Call_Function" not in orig_dict:
            new_dict = orig_dict.copy()
            new_dict["Call_Function"] = new_dict["Call Function"]
            del new_dict["Call Function"]
            return cls(**new_dict)
        else:
            return cls(**orig_dict)

    def to_dict(self) -> dict:
        tmp_res = asdict(self)
        if not self.prompt_template:
            del tmp_res["prompt_template"]
        return tmp_res

    @property
    def role(self):
        if self.prompt_template:
            return self.prompt_template.role
        else:
            # 这里不能自动填充属性
            logger.error("缺少属性")
            return ""

    def __str__(self):
        return self.file + self.FunctionName

    def to_string_for_request(self) -> str:
        tmp_dict = self.to_dict()

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


@dataclass
class AIOutput_location:
    startLine: int = 0


@dataclass
class AIOutput:
    """走查结果的标准形式"""

    # 错误代码绝对行号，加上了AIInput.begin_line
    location: AIOutput_location = field(default_factory=lambda: AIOutput_location(0))

    file: str = ""  # 文件路径
    FunctionName: str = ""  # 函数名
    illegalCode: str = ""  # 有问题的代码
    description: str = ""  # 问题描述
    severity: str = ""  # 风险等级
    ruleCode: str = ""  # 规则代码
    title: str = ""  # 标题
    chatAIOriginalRes: str = ""  # AI原始分析结果
    suggestion: str = ""  # 修复建议
    error_flag: bool = False  # 生成AI结果中是否发生错误
    valid_flag: bool = False  # 无效数据
    false_alarm: bool = False  # 误报
    false_alarm_des: str = ""  # 误报描述

    @property
    def startLine(self):
        """
        绝对行号
        """
        if self.location:
            return self.location.startLine
        else:
            logger.debug("自动填充成员")
            self.location = AIOutput_location(startLine=0)
            return 0

    @startLine.setter
    def startLine(self, value):
        """
        绝对行号
        """
        if self.location:
            self.location.startLine = value
        else:
            logger.debug("自动填充成员")
            self.location = AIOutput_location(startLine=value)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_bad_input(cls, ai_input: AIInput, rule="", title=""):
        if not ai_input.error_flag:
            raise ValueError("非法调用该函数")

        return cls(
            file=ai_input.file,
            FunctionName=ai_input.FunctionName,
            error_flag=True,
            ruleCode=rule,
            title=title,
            description="AI走查输入错误, 请检查输入",
        )

    def copy(self) -> "AIOutput":
        return deepcopy(self)

    def __str__(self):
        return f"{self.file}:{self.FunctionName}:{self.startLine}--{self.ruleCode}"

