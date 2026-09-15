"""tools/make_dist_tests.py —— 把测试相关的 15 个文件复制到 dist_tests/。
只复制、不修改源文件；带字节数校验。跑法：python tools/make_dist_tests.py

（原位置在仓库根目录 _make_dist_tests.py；搬到 tools/ 后 ROOT 需上溯一级。）
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "dist_tests"

# 11 个测试文件（与 pytest.ini 的 testpaths 一一对应）
TESTS = [
    "test_tools.py",
    "test_output_budget.py",
    "test_llm_retry.py",
    "test_context_compression.py",
    "test_session_log.py",
    "test_instance_isolation.py",
    "test_calculator.py",
    "test_web_events.py",
    "test_web_approvals.py",
    "test_web_sessions.py",
    "test_web_api.py",
]
# 4 个支撑文件
SUPPORT = [
    "conftest.py",       # 公共 fixture（pytest 自动加载，testpaths 里不会列）
    "pytest.ini",        # 闸门本身：testpaths
    "requirements.txt",  # 依赖声明
    "calculator.py",     # test_calculator.py 的直接依赖
]

FILES = TESTS + SUPPORT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if DST.exists():
        print(f"[提示] {DST.name}/ 已存在，将只更新其中的同名文件")
    DST.mkdir(exist_ok=True)

    ok = True
    print(f"复制目标：{DST}")
    print("-" * 66)
    for name in FILES:
        src = ROOT / name
        if not src.exists():
            print(f"  [缺失] {name}  <- 源文件不存在，跳过")
            ok = False
            continue
        dst = DST / name
        shutil.copy2(src, dst)  # copy2 保留 mtime
        same = sha(src) == sha(dst) and src.stat().st_size == dst.stat().st_size
        mark = "OK " if same else "!! "
        print(f"  [{mark}] {name:<32} {src.stat().st_size:>6} 字节")
        if not same:
            ok = False

    print("-" * 66)
    print(f"  共 {len(FILES)} 个文件；{'全部字节一致' if ok else '存在失败/不一致'}")

    # 反向检查：dist_tests 里不该出现别的东西
    extra = sorted(p.name for p in DST.iterdir() if p.name not in FILES)
    if extra:
        print(f"  [注意] dist_tests/ 中另有非本次复制的条目：{extra}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
