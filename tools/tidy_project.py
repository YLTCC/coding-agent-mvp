"""tools/tidy_project.py —— 一次性整理脚本：把根目录散落的一次性脚本 / 输出 / 临时目录归档。

原则：**只移动，不删除**。默认预演（dry-run），加 --apply 才真正动文件。
用法：
    python tools/tidy_project.py              # 预演：只打印计划
    python tools/tidy_project.py --apply      # 执行：移动到 docs/ 与 archive/
    python tools/tidy_project.py --apply --purge-cache   # 顺带清掉缓存目录

整理完可以把这个脚本本身也删掉（它是一次性工具）。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 保留在根目录（项目本体）──────────────────────────────────
KEEP_FILES = {
    "main.py", "run_web.py", "calculator.py", "conftest.py",
    "pytest.ini", "requirements.txt", "README.md",
    ".env.example", ".gitignore",
    ".env",  # 真实 API Key，绝不能移动
}
KEEP_DIRS = {"agent", "web", "dist_tests", "sessions", "tools", "docs", "archive"}
IGNORE_DIRS = {".pytest_cache", "__pycache__"}

# ── 要去的地方 ───────────────────────────────────────────────
DOCS = {
    "4.6todolist.md", "webtodolist.md", "agentfankui.md",
    "interview_qa.md", "note1.md", "note2.md", "note3.md",
}
# 无下划线前缀、但同样是一次性诊断的脚本
LEGACY_SCRIPTS = {"check_eol.py", "dbg_lines.py", "probe_turns.py", "demo_stream.py"}
# demo_stream.py 原列 TOOL_MOVES，但其 docstring 自述为"阶段二动工前的探针脚本"，
# 与 _verify*.py / _probe*.py 同性质，应归入一次性脚本。
TOOL_MOVES: set[str] = set()
TEMP_DIRS = {
    "_cli_capture_ws", "_safety_lab", "_smoke_sessions", "_smoke_ws",
    "_gate_demo", "_gate_demo2",
    # _tmp_big_dir 是 test_output_budget.py:65 的产物（5000 个 0 字节空文件）。
    # 该测试本应用 tmp_path fixture，却用了 cfg.default_workspace（项目根），
    # 中断时 finally 清理没跑完，残留到根目录。归档到 archive/tmp/ 后可随时删。
    "_tmp_big_dir",
}
BACKUPS = ["agent/agent.py.blankline.bak"]


def classify(p: Path) -> str:
    name = p.name
    if p.is_dir():
        if name in TEMP_DIRS:
            return "tmp"
        if name in KEEP_DIRS:
            return "keep"
        if name in IGNORE_DIRS:
            return "cache"
        return "unknown"
    if name in KEEP_FILES:
        return "keep"
    if name.startswith("test_") and name.endswith(".py"):
        return "keep"
    if name in DOCS:
        return "docs"
    if name in TOOL_MOVES:
        return "tools"
    if name.endswith(".txt"):
        return "outputs"
    if name.endswith(".py") and (name.startswith("_") or name in LEGACY_SCRIPTS):
        return "scripts"
    return "unknown"


def main() -> int:
    apply = "--apply" in sys.argv
    purge = "--purge-cache" in sys.argv

    plan: dict[str, list[Path]] = {
        "scripts": [], "outputs": [], "tmp": [], "docs": [], "tools": [],
        "keep": [], "cache": [], "unknown": [],
    }
    for p in sorted(ROOT.iterdir(), key=lambda x: x.name):
        plan[classify(p)].append(p)
    for rel in BACKUPS:
        bp = ROOT / rel
        if bp.exists():
            plan["backup"] = plan.get("backup", []) + [bp]

    mode = "执行（会移动文件）" if apply else "预演（不改动任何文件）"
    print("=" * 68)
    print(f"整理计划 —— {mode}")
    print("=" * 68)

    dest_names = {
        "scripts": "archive/scripts/  <- 一次性探针 / 修复 / 验证脚本",
        "outputs": "archive/outputs/  <- 运行输出留存（.txt）",
        "tmp": "archive/tmp/      <- 临时工作区 / 冒烟目录",
        "docs": "docs/             <- 项目文档",
        "tools": "tools/            <- 继续留用的工具脚本",
        "backup": "archive/backups/  <- 被污染文件的备份",
    }
    for key, label in dest_names.items():
        items = plan.get(key, [])
        if not items:
            continue
        print(f"\n[{label}]  {len(items)} 个")
        for p in items:
            print(f"    {p.name}")

    print(f"\n[保留在根目录]  {len(plan['keep'])} 个")
    kept = ", ".join(p.name for p in plan["keep"])
    print(f"    {kept}")

    if plan["cache"]:
        print(f"\n[缓存目录（不在本次移动范围）]  {len(plan['cache'])} 个")
        for p in plan["cache"]:
            print(f"    {p.name}")
        print("    -> 交付前手动删除；或用 --apply --purge-cache")

    if plan["unknown"]:
        print(f"\n  [WARN] 未分类条目 {len(plan['unknown'])} 个，请人工确认：")
        for p in plan["unknown"]:
            print(f"    {p.name}")

    moved = sum(len(plan.get(k, [])) for k in ("scripts", "outputs", "tmp", "docs", "tools", "backup"))
    print("\n" + "-" * 68)
    print(f"  合计将移动 {moved} 个条目")

    if not apply:
        print("  这只是一次预演。确认无误后执行：")
        print("      python tools/tidy_project.py --apply")
        print("=" * 68)
        return 0

    # ── 真正执行 ─────────────────────────────────────────────
    print("-" * 68)
    dirs = {
        "scripts": ROOT / "archive" / "scripts",
        "outputs": ROOT / "archive" / "outputs",
        "tmp": ROOT / "archive" / "tmp",
        "backup": ROOT / "archive" / "backups",
        "docs": ROOT / "docs",
        "tools": ROOT / "tools",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    failed = 0
    for key, d in dirs.items():
        for p in plan.get(key, []):
            target = d / p.name
            if target.exists():
                print(f"  [跳过] {p.name}（目标已存在）")
                failed += 1
                continue
            try:
                shutil.move(str(p), str(target))
                print(f"  [已移动] {p.name}  ->  {d.relative_to(ROOT).as_posix()}/")
            except OSError as e:
                print(f"  [失败] {p.name}: {e}")
                failed += 1

    if purge:
        for p in plan["cache"]:
            try:
                shutil.rmtree(p)
                print(f"  [已删除缓存] {p.name}")
            except OSError as e:
                print(f"  [删除失败] {p.name}: {e}")
                failed += 1

    print("-" * 68)
    print(f"  完成；失败 {failed} 个" if failed else "  完成；全部成功")
    print("  接着验证：python tools/list_tests.py  再  python -m pytest -q")
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
