"""
test_sql.py — 用"罐装 AIInput + 假走查器(不连 AI)"验证 for_sqllite.py。

覆盖:
  1. 去重三元组 (git_addr + identify + rule_code): 第二轮 re-review 应被跳过
  2. 跨仓库隔离: 相同代码换了仓库地址 -> 视为未走查
  3. 走查结果随行存档, 并可经 SQLiteServe 查询方法读回
  4. 出错(error_flag)的走查不入库

跑法:  python test_sql.py
不依赖 aiohttp / 真实 AI / 真实工程根。
"""
import tempfile
from pathlib import Path

from AIIO import AIInput, AIOutput, AIOutput_location
from for_sqllite import sqlite_data, SQLiteServe
from info_for_llm_review import aiinput_for_test

RULE_SET = ("rule-1x", "rule-2x", "rule-3d")


def _fake_review(ai_input: AIInput, rule: str) -> list[AIOutput]:
    """确定性假走查器(对齐 AIChat.arequestChatTest 的三类输出, 但按规则而非随机)。

    rule-1x -> 有效问题(真实发现)
    rule-2x -> 未发现问题(null, 无 error, 会入库)
    rule-3d -> 走查出错(error_flag, 不入库)
    """
    file = ai_input.file or "未知文件"
    fn = ai_input.FunctionName or "未知函数"
    if ai_input.error_flag:
        return [AIOutput.from_bad_input(ai_input, rule, rule)]
    if rule == "rule-1x":
        return [
            AIOutput(
                file=file, FunctionName=fn, ruleCode=rule, title=rule,
                description="可能存在数组越界", severity="高",
                illegalCode="return arr[idx];",
                location=AIOutput_location(startLine=ai_input.begin_line + 2),
                chatAIOriginalRes="原始结果-rule1",
                suggestion="访问前检查 idx 上限", valid_flag=True,
            )
        ]
    if rule == "rule-2x":
        return [
            AIOutput(file=file, FunctionName=fn, ruleCode=rule, title=rule,
                     description="未发现高置信度问题",
                     chatAIOriginalRes="# 未发现高置信度问题")
        ]
    return [
        AIOutput(file=file, FunctionName=fn, ruleCode=rule, title=rule,
                 description="AI接口出错, 无法返回结果",
                 chatAIOriginalRes="None", error_flag=True)
    ]


def main_test_sql():
    print("=" * 62)
    print("验证 SQLiteServe: 去重(git_addr+identify+rule_code) 与 结果存档")
    print("=" * 62)

    db = SQLiteServe(Path(tempfile.mkdtemp(prefix="llmreview_")) / "review.db")
    git_a = {"git_addr": "git@example:A.git", "git_branch": "main", "git_hash": "aaaa"}
    git_b = dict(git_a)
    git_b["git_addr"] = "git@example:B.git"

    # ---- 第一轮走查: 与 TotalTask 相同的入库判定(全部 error_flag 才不存) ----
    print("\n-- 第一轮走查入库 --")
    for ai in aiinput_for_test:
        sd = sqlite_data.from_AIInput(ai, git_a)
        for rule in RULE_SET:
            if sd.in_database(rule, db):  # 去重
                print(f"    跳过(已存在) {ai.file}:{ai.FunctionName}/{rule}")
                continue
            outs = _fake_review(ai, rule)
            if all(not o.error_flag for o in outs):
                saved = sd.save_into_database(rule, db, outs)
                print(f"    入库 {ai.file}:{ai.FunctionName} / {rule} -> saved={saved}")
            else:
                print(f"    失败未入库 {ai.file}:{ai.FunctionName} / {rule} (error_flag)")

    # ---- 第二轮: 去重 ----
    print("\n-- 第二轮走查(去重验证) --")
    for ai in aiinput_for_test:
        sd = sqlite_data.from_AIInput(ai, git_a)
        for rule in RULE_SET:
            hit = sd.in_database(rule, db)
            print(f"    {ai.file}:{ai.FunctionName}/{rule}: in_database={hit}")

    # ---- 跨仓库隔离 ----
    print("\n-- 跨仓库隔离 --")
    sd_b = sqlite_data.from_AIInput(aiinput_for_test[0], git_b)
    print(f"    相同代码+不同仓库 rule-1x 是否命中: {sd_b.in_database('rule-1x', db)} (期望 False)")

    # ---- 查询 ----
    print("\n-- SQLiteServe 查询 --")
    print(f"    count              = {db.count()}")
    print(f"    distinct_rules     = {db.distinct_rules()}")
    st = db.statistics()
    print(f"    statistics         = {st}")
    r1 = db.results_by_rule("rule-1x", only_real=True)
    print(f"    results_by_rule(rule-1x, only_real=True) 共 {len(r1)} 条:")
    for x in r1:
        print(f"      - {x['path']}:{x.get('func_name')} [{x.get('severity')}] {x.get('description')}")

    # 读取单条目结果
    first_sd = sqlite_data.from_AIInput(aiinput_for_test[0], git_a)
    loaded = db.load_results(first_sd, "rule-1x")
    print(f"    load_results(first_input, rule-1x) = {loaded}")

    # ---- 断言 ----
    print("\n-- 断言 --")
    n = len(aiinput_for_test)
    # rule-1x 与 rule-2x 对每个输入都成功入库一次 => n 行各
    assert db.count(git_addr=git_a["git_addr"], rule_code="rule-1x") == n
    assert db.count(git_addr=git_a["git_addr"], rule_code="rule-2x") == n
    # rule-3d 全部 error_flag, 不应有任何行
    assert db.count(git_addr=git_a["git_addr"], rule_code="rule-3d") == 0
    # 第二轮去重: 同一仓库+同一身份+同一规则应命中
    for ai in aiinput_for_test:
        sd = sqlite_data.from_AIInput(ai, git_a)
        assert sd.in_database("rule-1x", db) is True
        assert sd.in_database("rule-2x", db) is True
        assert sd.in_database("rule-3d", db) is False  # 从未入库
    # 跨仓库隔离
    assert sd_b.in_database("rule-1x", db) is False
    # 结果已存档且可读回
    assert loaded and "越界" in loaded[0]["description"]
    # 身份哈希: 同代码 = 同 identity, 不同仓库不影响 identity
    assert first_sd._identity == sd_b._identity
    assert len(first_sd._identity) == 64  # sha256 hex

    db.close()
    print("全部断言通过  (临时 DB: 通过 tempfile 自动清理)")


if __name__ == "__main__":
    main_test_sql()