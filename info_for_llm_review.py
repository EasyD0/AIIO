"""
测试用: 提供一组"罐装"AIInput 假输入, 供 test_sql.py 验证数据库。
不连接 AI、不需要真实工程与真实预处理。Preprocessor 等正式实现接入后,
把这里的 aiinput_for_test 换成真实提取结果即可。
"""
from dataclasses import replace

from AIIO import AIInput


def _mk(
    file: str,
    fn: str,
    orig: str,
    prepped: str,
    declares: list[str],
    calls: list[str],
    begin: int = 1,
) -> AIInput:
    return AIInput(
        begin_line=begin,
        file=file,
        FunctionName=fn,
        Function=prepped,      # 预处理后(这里直接给无注释体)
        Function_orig=orig,    # 预处理前
        Call_Function=calls,
        Declare=declares,
        prompt_template=None,
        error_flag=False,
    )


# 罐装输入池: 内容刻意互相不同, 以便数据库"身份哈希(identify)"能把它们区分开。
aiinput_for_test: list[AIInput] = [
    _mk(
        file="src/foo.c", fn="check_index", begin=5,
        orig="""int check_index(int idx, int size){
    int arr[10];
    return arr[idx];
}""",
        prepped="""int check_index(int idx, int size){
    int arr[10];
    return arr[idx];
}""",
        declares=["typedef struct { int len; int *data; } Vec;", "#include <stdio.h>"],
        calls=["memcpy();"],
    ),
    _mk(
        file="src/bar.c", fn="parse_line", begin=12,
        orig="""int parse_line(char *line){
    int n = strlen(line);
    return n;
}""",
        prepped="""int parse_line(char *line){
    int n = strlen(line);
    return n;
}""",
        declares=["static int g_count = 0;"],
        calls=["strlen();"],
    ),
    _mk(
        file="src/baz.c", fn="unused_helper", begin=40,
        orig="""void unused_helper(int x){ return; }""",
        prepped="""void unused_helper(int x){ return; }""",
        declares=[],
        calls=[],
    ),
]


_index = 0


def get_func_info(*any) -> AIInput:
    """
    按调用顺序循环返回罐装输入, 保证测试可重复。

    参数对齐 get_func_info_async 的调用方式: (preprocessor, source_file, func_name, True)。
    这里忽略入参, 直接轮换返回池中对象。返回的是浅拷贝, 避免调用方改动污染池。
    """
    global _index
    result = aiinput_for_test[_index % len(aiinput_for_test)]
    _index += 1
    return replace(result)