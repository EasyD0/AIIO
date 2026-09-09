import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

from AIIO import AIInput
from MyPyLib.LogSet import logSetup

logger = logSetup(__file__)


def _default_format_exe() -> Path | None:
    """返回默认的 clang-format / clangd-format 可执行文件路径"""
    exe = os.environ.get("CLANG_FORMAT_EXE")
    if exe:
        return Path(exe)
    found = shutil.which("clang-format") or shutil.which("clangd-format")
    return Path(found) if found else None


def rm_comments(code: str) -> str:
    """
    删除代码注释
    支持 // 行注释与 /* */ 块注释, 并正确处理字符串/字符字面量, 避免误删字面量中的 // 或 /*。
    """
    if not code:
        return ""
    out = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""

        # 行注释
        if c == "/" and nxt == "/":
            while i < n and code[i] != "\n":
                i += 1
            out.append("\n")  # 保留换行, 维持行号/结构
            continue
        # 块注释
        if c == "/" and nxt == "*":
            i += 2
            while i < n and not (code[i] == "*" and i + 1 < n and code[i + 1] == "/"):
                if code[i] == "\n":
                    out.append("\n")  # 保留块注释内的换行
                i += 1
            i += 2  # 跳过 "*/"
            continue
        # 字符串字面量
        if c in ('"', "'"):
            quote = c
            out.append(c)
            i += 1
            while i < n:
                if code[i] == "\\":
                    out.append(code[i])
                    if i + 1 < n:
                        out.append(code[i + 1])
                    i += 2
                    continue
                out.append(code[i])
                i += 1
                if code[i - 1] == quote:
                    break
            continue
        out.append(c)
        i += 1
    return "".join(out)


def format_c_code(code: str, clangd_format_exe: Path = None) -> str:
    """
    格式化代码，并且删除空行
    优先用 clang-format 格式化; exe 不可用时降级为仅去除空行。
    """
    if not code:
        return ""

    exe = clangd_format_exe or _default_format_exe()
    if exe is not None:
        try:
            res = subprocess.run(
                [str(exe), "-style=file"],
                input=code,
                text=True,
                capture_output=True,
                timeout=30,
            )
            if res.returncode != 0:
                logger.warning(
                    f"clang-format(-style=file)失败({res.stderr.strip()}), 回退 -style=LLVM"
                )
                res = subprocess.run(
                    [str(exe), "-style=LLVM"],
                    input=code,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
            if res.returncode == 0:
                code = res.stdout
        except Exception as e:
            logger.warning(f"clang-format调用异常: {e}")

    # 删除空行
    return "\n".join(line.rstrip() for line in code.splitlines() if line.strip())


class SQLiteServe:
    """SQLite 存取服务。

    去重依据只有三个字段: git_addr(git仓库地址) + identify(代码身份) + rule_code(规则名)。
    git_branch/git_hash 仅随行存储, 不参与去重。
    """

    def __init__(self, data_path: Path):
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(data_path))
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS code_review(
                identify   TEXT NOT NULL,
                rule_code  TEXT NOT NULL,
                git_addr   TEXT, git_branch TEXT, git_hash TEXT,
                path TEXT, func_name TEXT, func_body TEXT,
                result TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (git_addr, identify, rule_code))"""
        )
        self._ensure_result_column()
        self.conn.commit()

    def _ensure_result_column(self):
        """兼容旧库: 若 code_review 缺少 result 列则补上"""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(code_review)")}
        if "result" not in cols:
            self.conn.execute("ALTER TABLE code_review ADD COLUMN result TEXT NOT NULL DEFAULT ''")

    def search(self, _sqlite_data: "sqlite_data", rule_code: str) -> bool:
        """同一 git 仓库地址 + 同一代码身份 + 同一规则 => 已走查过, 返回 True"""
        row = self.conn.execute(
            "SELECT 1 FROM code_review WHERE git_addr=? AND identify=? AND rule_code=?",
            (_sqlite_data.git_addr, _sqlite_data._identity, rule_code),
        ).fetchone()
        return row is not None

    def save(
        self,
        _sqlite_data: "sqlite_data",
        rule_code: str,
        results=None,
    ) -> bool:
        """入库一行(去重组), 并把该规则的走查结果 JSON 一并存进同一行。

        results: 可序列化的 AIOutput 对象列表(需含 to_dict)。
        入库与结果存档为同一行的原子动作; 已存在(IGNORE)则不覆盖, 保持首次存档。
        """
        result_text = ""
        if results:
            result_text = json.dumps([r.to_dict() if hasattr(r, "to_dict") else r for r in results])
        cur = self.conn.execute(
            """INSERT OR IGNORE INTO code_review
               (git_addr, identify, rule_code, git_branch, git_hash, path,
                func_name, func_body, result)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                _sqlite_data.git_addr,
                _sqlite_data._identity,
                rule_code,
                _sqlite_data.git_branch,
                _sqlite_data.git_hash,
                _sqlite_data.path,
                _sqlite_data.func_name,
                _sqlite_data.func_body,
                result_text,
            ),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def close(self):
        self.conn.close()

    # ---------- 查询工具 ----------

    @staticmethod
    def _rows_to_dicts(cursor) -> list[dict]:
        """游标 -> dict 列表(带列名), 并把 result 字段反序列化"""
        cols = [d[0] for d in cursor.description]
        out = []
        for row in cursor.fetchall():
            item = dict(zip(cols, row))
            item["result"] = SQLiteServe._parse_results(item.get("result", ""))
            out.append(item)
        return out

    @staticmethod
    def _parse_results(result_text: str) -> list[dict]:
        """result JSON -> list[dict]; 空/损坏 -> []"""
        if not result_text:
            return []
        try:
            data = json.loads(result_text)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    @staticmethod
    def _is_real(r: dict) -> bool:
        """真实问题(非误报/非无效/非错误): 用于结果过滤"""
        return not (
            r.get("false_alarm")
            or r.get("valid_flag")
            or r.get("error_flag")
        )

    @staticmethod
    def _build_where(filters: dict) -> tuple[str, list]:
        """根据过滤字段(键=列名, 值=None则忽略)构造 WHERE 子句"""
        conds, params = [], []
        allowed = {"git_addr", "rule_code", "identify", "path", "func_name", "git_branch", "git_hash"}
        for key, val in filters.items():
            if key in allowed and val is not None:
                conds.append(f"{key}=?")
                params.append(val)
        where = " WHERE " + " AND ".join(conds) if conds else ""
        return where, params

    # ---------- 通用查询 ----------

    def find(self, _sqlite_data: "sqlite_data", rule_code: str):
        """按去重组精确查单行; 未走查返回 None. 返回行含解析后的 result"""
        rows = self.rows(git_addr=_sqlite_data.git_addr, identify=_sqlite_data._identity, rule_code=rule_code)
        return rows[0] if rows else None

    def rows(self, *, git_addr=None, rule_code=None, identify=None, path=None, func_name=None) -> list[dict]:
        """灵活多条件查询(所有条件可选, 一起 AND), 返回所有匹配行"""
        where, params = self._build_where({
            "git_addr": git_addr, "rule_code": rule_code, "identify": identify,
            "path": path, "func_name": func_name,
        })
        cur = self.conn.execute(f"SELECT * FROM code_review{where}", params)
        return self._rows_to_dicts(cur)

    def count(self, *, git_addr=None, rule_code=None, identify=None, path=None, func_name=None) -> int:
        where, params = self._build_where({
            "git_addr": git_addr, "rule_code": rule_code, "identify": identify,
            "path": path, "func_name": func_name,
        })
        return self.conn.execute(f"SELECT COUNT(*) FROM code_review{where}", params).fetchone()[0]

    def distinct_rules(self, *, git_addr=None) -> list[str]:
        where, params = self._build_where({"git_addr": git_addr})
        rows = self.conn.execute(
            f"SELECT DISTINCT rule_code FROM code_review{where} ORDER BY rule_code", params
        ).fetchall()
        return [r[0] for r in rows]

    # ---------- 结果读取 ----------

    def load_results(self, _sqlite_data: "sqlite_data", rule_code: str) -> list[dict]:
        """该 (entry, rule) 的走查结果 dict 列表; 未走查返回 []"""
        row = self.find(_sqlite_data, rule_code)
        return row["result"] if row else []

    def results_by_rule(self, rule_code: str, *, git_addr=None, only_real=False) -> list[dict]:
        """跨所有行展平该规则的走查结果, 每条补上 path/func_name/git_addr"""
        rows = self.rows(rule_code=rule_code, git_addr=git_addr)
        out = []
        for row in rows:
            for r in row["result"]:
                if only_real and not self._is_real(r):
                    continue
                item = dict(r)
                item["path"] = row["path"]
                item["func_name"] = row["func_name"]
                item["git_addr"] = row["git_addr"]
                out.append(item)
        return out

    def results_by_path(self, path: str, *, git_addr=None, only_real=False) -> list[dict]:
        """某个文件的所有走查结果(合并各行 result, 按相同 result 去重)"""
        rows = self.rows(path=path, git_addr=git_addr)
        out, seen = [], set()
        for row in rows:
            for r in row["result"]:
                if only_real and not self._is_real(r):
                    continue
                # 以描述+代码(若有)作为去重指纹, 避免同文件多规则重复展示同一条
                fingerprint = (r.get("description", ""), r.get("illegalCode", ""))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                item = dict(r)
                item["path"] = row["path"]
                item["func_name"] = row["func_name"]
                item["git_addr"] = row["git_addr"]
                out.append(item)
        return out

    # ---------- 统计 ----------

    def statistics(self, *, git_addr=None) -> dict:
        """概览统计: 行数/身份数/文件数/函数数/规则数 + 按风险等级计数"""
        keys = {"identify": "distinct_identifies", "path": "distinct_paths",
                "func_name": "distinct_functions", "rule_code": "distinct_rules"}
        res = {"rows": self.count(git_addr=git_addr)}
        for col, key in keys.items():
            res[key] = self.conn.execute(
                f"SELECT COUNT(DISTINCT {col}) FROM code_review" + (
                    " WHERE git_addr=?" if git_addr else ""
                ), ([git_addr] if git_addr else [])
            ).fetchone()[0]

        # 从每行的 result 统计风险等级
        severity_counter: dict[str, int] = {}
        for row in self.rows(git_addr=git_addr):
            for r in row["result"]:
                if self._is_real(r):
                    sev = r.get("severity") or "未知"
                    severity_counter[sev] = severity_counter.get(sev, 0) + 1
        res["severity_counts"] = severity_counter
        return res


@dataclass
class sqlite_data:
    git_addr: str
    git_branch: str
    git_hash: str

    path: str  # 相对工程根的路径
    func_name: str
    func_body: str

    _identity: str

    # 下面这三个成员用于形成 _identity
    func_body_prep: str = ""  # 预处理后函数体, 无注释且格式化后
    # 预处理后关联代码, 如结构体声明, 全局变量定义, 类型定义等，无注释且格式化后，经过排序后
    rela_code_prep: list[str] = field(default_factory=list)
    # 该函数调用函数的代码(预处理后)，无注释且格式化后，经过排序后
    call_func_prep: list[str] = field(default_factory=list)

    def get_identify(self) -> str:
        """
        返回一串关于代码内容的hash码, 需要考虑如下字段
        func_body_prep 需要格式化
        rela_code_prep 需要格式化, 并排序
        call_func_prep 需要格式化, 并排序
        """
        parts = [format_c_code(rm_comments(self.func_body_prep))]
        parts += sorted(
            format_c_code(rm_comments(x)) for x in self.rela_code_prep
        )
        parts += sorted(
            format_c_code(rm_comments(x)) for x in self.call_func_prep
        )

        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def in_database(self, rule_code: str, database) -> bool:
        """
        检查这条目+规则 是否已存在于sqlite数据库中(同一仓库+同一身份+同一规则)
        """
        return database.search(self, rule_code)

    def save_into_database(self, rule_code: str, database, results=None) -> bool:
        """
        将数据存入sqlite数据库, 返回是否成功
        results: 该规则的走查结果(AIOutput)列表, 随行一并存档
        """
        return database.save(self, rule_code, results)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_AIInput(cls, input: AIInput, git_data: dict) -> "sqlite_data":
        """
        从一个AIInput来生成数据库的数据
        func_body_prep 需要格式化
        rela_code_prep 需要格式化, 并排序
        call_func_prep 需要格式化, 并排序

        git_data 含有所有git信息
        """
        obj = cls(
            git_addr=git_data.get("git_addr", ""),
            git_branch=git_data.get("git_branch", ""),
            git_hash=git_data.get("git_hash", ""),
            path=input.file,
            func_name=input.FunctionName,
            func_body=input.Function or input.Function_orig,
            _identity="",
        )
        obj.func_body_prep = format_c_code(rm_comments(input.Function))
        obj.rela_code_prep = sorted(
            format_c_code(rm_comments(d)) for d in input.Declare
        )
        obj.call_func_prep = sorted(
            format_c_code(rm_comments(c)) for c in input.Call_Function
        )
        obj._identity = obj.get_identify()
        return obj