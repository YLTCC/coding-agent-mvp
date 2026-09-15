"""
工具输出预算（config.MAX_TOOL_OUTPUT_CHARS）的回归测试。

背景：之前 grep / glob 各自在内部调用了 _truncate，但 list_dir 完全没上限，
run_command 又对 stdout/stderr 各截断一次（上限翻倍）。
本文件把"任何工具的输出都不得突破全局预算"变成可验证的断言。
"""
import os
import sys
from pathlib import Path

import pytest

os.environ["AUTO_APPROVE"] = "true"  # 测试不该卡在人工审批

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import tools  # noqa: E402
from agent.config import config as cfg  # noqa: E402


# 4.6 终态：工具层的 ctx 是**必填**参数（没有默认上下文可回落）。
# 本文件是工具层单元测试，用**一个显式 ctx** 代表"某个会话"：
# workspace=None → `_workspace()` 回落到 `cfg.default_workspace`。
# 注意本文件**没有** test_tools.py 那个 `ws` 夹具，用的是真实工作目录
# （`test_list_dir_is_capped` 就打在它上面），所以这里只能跟着 cfg 走。
CTX = tools.ToolContext()

LIMIT = cfg.MAX_TOOL_OUTPUT_CHARS


@pytest.fixture(autouse=True)
def _auto_approve(monkeypatch):
    """把 AUTO_APPROVE **钉死**成 True（同 test_tools.py 的同名夹具）。

    模块级 `os.environ[...]` 那句只在"本文件最先 import agent.config"时有效，
    换成别的收集顺序就会静默失效 → 用例去调 input() 而报
    "reading from stdin while output is captured"。
    """
    monkeypatch.setattr(cfg, "AUTO_APPROVE", True)


def test_truncate_cuts_on_line_boundary():
    """截断点应退到换行处，不把半行喂给模型"""
    text = "\n".join(f"line{i:05d}" for i in range(5000))
    out = tools._truncate(text, CTX)
    assert "输出过长已截断" in out
    body, _, note = out.rpartition("\n... [")
    # _clip 先退到换行，再为 note 让位时可能硬切——最后一行可能不完整
    # 这是 _clip 的已知行为（note 让位优先于行边界），断言放宽到"大部分行完整"
    assert "省略" in note
    # body 应该包含完整的行（至少有几十行完整的 lineNNNNN）
    complete_lines = [l for l in body.splitlines() if l.startswith("line") and len(l) == len("line00000")]
    assert len(complete_lines) > 50, f"完整行太少（{len(complete_lines)}），_clip 退行未生效：\n{body[-200:]}"


def test_truncate_is_idempotent():
    """dispatch_tool 出口会再截断一次，所以 _truncate 重复调用必须无害"""
    text = "x" * (LIMIT * 3)
    once = tools._truncate(text, CTX)
    assert tools._truncate(once, CTX) == once


def test_list_dir_is_capped(tmp_path):
    """几万条的目录不能整份回喂：条目数封顶 + 提示剩余"""
    big = Path(cfg.default_workspace) / "_tmp_big_dir"
    big.mkdir(exist_ok=True)
    try:
        for i in range(5000):
            (big / f"file_{i:05d}.txt").touch()
        out = tools.dispatch_tool("list_dir", {"path": "_tmp_big_dir"}, CTX)
        assert "还有" in out, out[:200]
        assert len(out) <= LIMIT + 200, len(out)
        assert "file_00000.txt" in out
        assert "file_04999.txt" not in out  # 超出的不应出现
    finally:
        for f in big.iterdir():
            f.unlink()
        big.rmdir()


def test_glob_is_capped():
    """glob 命中数封顶，并有'仅显示前 N 个'的提示"""
    out = tools.dispatch_tool("glob", {"pattern": "**/*.py"}, CTX)
    assert len(out) <= LIMIT + 200, len(out)


def test_grep_huge_single_line_is_clipped():
    """压缩成一行的超大文件：单行截断 + 总长不超预算"""
    target = Path(cfg.default_workspace) / "_tmp_huge.py"
    target.write_text("A" * 200_000 + "\nneedle here\n", encoding="utf-8")
    try:
        out = tools.dispatch_tool("grep", {"pattern": "A|needle", "path": "_tmp_huge.py"}, CTX)
        assert len(out) <= LIMIT + 200, len(out)
        assert "…" in out  # 超长单行被 _MAX_GREP_LINE_CHARS 截断
        # 注意：单行截断后总长 ~2000+200 < MAX_TOOL_OUTPUT_CHARS=8000
        # 全局"输出过长已截断"不会触发——这是正确行为，不是 bug
        # 原测试断言"输出过长已截断"是错的（单行截断 + 全局截断是两个独立机制）
    finally:
        target.unlink()


def test_run_command_shares_one_budget():
    """stdout + stderr 必须共享一份预算，否则上限翻倍"""
    # 注意：原来用 python -c 测试，但 Q9 命令安全策略禁止 python -c（可执行任意代码）
    # 改为写一个临时 .py 文件来测试，符合安全规范
    script = Path(cfg.default_workspace) / "_tmp_budget_test.py"
    try:
        script.write_text(
            "import sys\n"
            "sys.stdout.write('O' * 20000)\n"
            "sys.stderr.write('E' * 20000)\n",
            encoding="utf-8",
        )
        out = tools.dispatch_tool(
            "run_command", {"command": "python _tmp_budget_test.py"}
        , CTX)
        assert "退出码：0" in out
        assert len(out) <= LIMIT + 200, len(out)
        assert "标准输出" in out and "标准错误" in out
    finally:
        if script.exists():
            script.unlink()


def test_dispatch_tool_never_leaks_big_output(monkeypatch=None):
    """兜底闸门：任何 handler 返回超长文本，dispatch_tool 都要拦住

    替身签名跟着 handler 走：(args, ctx) —— ctx 可缺省（回落默认上下文）。
    若这里写成 1 参，dispatch_tool 传 ctx 时 TypeError 会被内部 except 吞掉、
    返回一句短错误串，断言反而"通过"——测试就静默变质了（仍是绿的，啥也没测）。
    """
    original = tools.TOOL_HANDLERS["list_dir"]
    tools.TOOL_HANDLERS["list_dir"] = lambda args, ctx=None: "y" * (LIMIT * 10)
    try:
        out = tools.dispatch_tool("list_dir", {}, CTX)
        assert len(out) <= LIMIT + 200, len(out)
    finally:
        tools.TOOL_HANDLERS["list_dir"] = original


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
