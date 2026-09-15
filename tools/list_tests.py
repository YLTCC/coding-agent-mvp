"""tools/list_tests.py —— 列出所有测试文件：用例数、行数、所属分组。
只读脚本，不改动任何东西。跑法：python tools/list_tests.py

（原位置在仓库根目录 _list_tests.py；搬到 tools/ 后 ROOT 需上溯一级。）
"""
from __future__ import annotations

import configparser
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1) 从 pytest.ini 读 testpaths（闸门本身）
ini = configparser.ConfigParser()
ini.read(ROOT / "pytest.ini", encoding="utf-8")
gated = [line.strip() for line in ini["pytest"]["testpaths"].splitlines() if line.strip()]

# 2) 磁盘上实际存在的 test_*.py
on_disk = sorted(p.name for p in ROOT.glob("test_*.py"))

# 3) 跑收集器，统计每个文件的用例数
out = subprocess.run(
    [sys.executable, "-m", "pytest", "--collect-only", "-q"],
    cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
)
counts = Counter()
for line in out.stdout.splitlines():
    if "::" in line:
        counts[line.split("::", 1)[0]] += 1

total = sum(counts.values())

print("=" * 68)
print("测试文件清单（磁盘实际存在的 test_*.py）")
print("=" * 68)
for name in on_disk:
    lines = len((ROOT / name).read_text(encoding="utf-8").splitlines())
    gate = "已挂" if name in gated else "!! 未挂 !!"
    print(f"  {name:<34} {counts.get(name, 0):>3} 用例  {lines:>5} 行  [{gate}]")

print("-" * 68)
print(f"  小计：{len(on_disk)} 个文件，{total} 个用例")

# 4) 交叉校验：漏挂 / 幽灵条目
missing = [n for n in on_disk if n not in gated]
ghost = [n for n in gated if n not in on_disk]
print("-" * 68)
print(f"  pytest.ini 里挂着 {len(gated)} 条 testpaths")
if missing:
    print(f"  [WARN] 磁盘上有但 pytest.ini 漏挂（= 永远绿）：{missing}")
elif ghost:
    print(f"  [WARN] pytest.ini 挂了但磁盘上不存在：{ghost}")
else:
    print("  [OK] testpaths 与磁盘文件完全一致，无漏挂、无幽灵条目")

# 5) 公共支撑文件
print("-" * 68)
extra = ["conftest.py", "pytest.ini", "requirements.txt"]
for name in extra:
    p = ROOT / name
    if p.exists():
        print(f"  [支撑] {name}  ({len(p.read_text(encoding='utf-8').splitlines())} 行)")
print("=" * 68)
