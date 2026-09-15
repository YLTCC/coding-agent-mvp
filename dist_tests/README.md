# dist_tests —— 测试文件打包副本

本目录是测试相关文件的**独立副本**，用于交付 / 归档。共 15 个文件，
包含 **235 个用例**（11 个测试文件），已实测全部通过。

---

## 一、怎么跑（重要）

**必须在项目根目录下运行**，把 `dist_tests` 当参数传进去：

```cmd
python -m pytest dist_tests -q
```

预期输出：

```
235 passed in ~39s
```

### ❌ 不要进入本目录再跑

```cmd
cd dist_tests
python -m pytest -q          ← 失败！
```

会得到 10 个收集错误：

```
ModuleNotFoundError: No module named 'agent'
ModuleNotFoundError: No module named 'web'
ModuleNotFoundError: No module named 'main'
!!!!!!!!!!!!!!!!!! Interrupted: 10 errors during collection !!!!!!!!!!!!!!!!!!!
```

**原因**：测试文件里有跨目录导入，例如

| 测试文件 | 需要的生产模块 |
|---|---|
| `test_tools.py` | `agent.tools` |
| `test_context_compression.py` | `agent.agent`、`agent.config` |
| `test_web_api.py` | `web.app`、`web.events` |
| `test_instance_isolation.py` | `main` |
| `test_calculator.py` | `calculator`（本目录内已带） |

`agent/`、`web/`、`main.py` 都在它们的**上一级目录**，所以本目录必须待在项目里，
通过工作目录 + `sys.path` 才能让这些 import 生效。

---

## 二、本目录必须放在哪里

本目录**不能**脱离项目单独解压使用。两种可行姿势：

**✅ 姿势 A（推荐）**：留在项目里

```
coding-agent-MVP/
├── agent/                 ← 生产代码
├── web/                   ← 生产代码
├── main.py                ← 生产代码
├── run_web.py
└── dist_tests/            ← 本目录
    ├── README.md
    ├── pytest.ini
    ├── conftest.py
    └── test_*.py ...
```

然后：`python -m pytest dist_tests -q`

**✅ 姿势 B**：整份打包给别人

必须把 `agent/`、`web/`、`main.py`、`run_web.py`、`requirements.txt`
和 `dist_tests/` **一起**给对方，对方在根目录跑同样的命令。

**❌ 只发 `dist_tests/` 一个文件夹给对方 → 对方跑不起来。**

---

## 三、文件清单

### 测试文件（11 个，235 用例）

| 文件 | 用例 | 被测模块 |
|---|---:|---|
| `test_tools.py` | 85 | `agent/tools.py` |
| `test_context_compression.py` | 33 | `agent/agent.py`、`agent/config.py` |
| `test_session_log.py` | 28 | `agent/session_log.py`、`agent.py`、`tools.py` |
| `test_web_sessions.py` | 17 | `web/app.py`、`web/session_registry.py` |
| `test_llm_retry.py` | 15 | `agent/llm.py` |
| `test_web_api.py` | 14 | `web/app.py`、`web/events.py` |
| `test_web_approvals.py` | 14 | `web/session_registry.py` |
| `test_web_events.py` | 9 | `web/events.py` |
| `test_output_budget.py` | 7 | `agent/tools.py`、`agent/config.py` |
| `test_calculator.py` | 7 | `calculator.py` |
| `test_instance_isolation.py` | 6 | `main.py`、`agent/session_log.py` |
| **合计** | **235** | |

### 支撑文件（4 个）

| 文件 | 作用 | 漏了会怎样 |
|---|---|---|
| `pytest.ini` | **闸门本身**：`testpaths` 列出全部 11 个测试文件 | 用例可能全部不跑，或"永远绿" |
| `conftest.py` | 公共夹具与替身（`web_cfg`、`ScriptedLLM`、`parse_frame`） | Web 相关测试大面积报错 |
| `requirements.txt` | 依赖声明 | 环境装不起来 |
| `calculator.py` | `test_calculator.py` 的直接依赖 | `ImportError`，该文件 7 个用例收集失败 |

---

## 四、`pytest.ini` 为什么是纯 ASCII 的

**不要**在 `pytest.ini` 里写中文注释。

`iniconfig` 用系统默认编码（GBK Windows 上是 GBK）读这个文件，
中文注释会让**整个收集阶段崩溃**。原始文件里也专门留了这条 NOTE：

```ini
; NOTE (ASCII only: iniconfig reads this file with the locale default codec,
;       Chinese comments here crash collection on a GBK Windows box).
```

中文说明一律写在 `README.md`（本文件）里，不要搬进 `pytest.ini`。

---

## 五、`testpaths` 是闸门，不是备注

`pytest.ini` 里的 `testpaths` 是**硬闸门**：没被列进去的测试文件，
它的用例**永远不会执行，但看起来像通过了**。

`test_instance_isolation.py`（6 个隔离用例）和 `test_calculator.py`（7 个用例）
就曾经因此被静默跳过很久。**每新增一个测试文件，必须同步加进 `testpaths`。**

当前 `testpaths` 与本目录磁盘上的 11 个 `test_*.py` 完全一致，无遗漏、无幽灵条目。

---

## 六、临时脚本不要打包

项目里的一次性诊断脚本（已统一归档到 `archive/`，见仓库根 `README.md` §6）
**不属于交付内容**，不要放进本目录。

> 注：`tools/make_dist_tests.py` 的自检里有一条"反向检查"，会提示本目录中
> 出现了它清单之外的文件。预期会看到：
>
> ```
> [注意] dist_tests/ 中另有非本次复制的条目：['.pytest_cache', 'README.md', '__pycache__']
> ```
>
> - `README.md` —— 本文件，是有意放进去的，忽略即可。
> - `.pytest_cache`、`__pycache__` —— 跑过 pytest 之后的**缓存产物**，
>   **压缩交付前请删掉这两个目录**（它们是解释器/pytest 生成的，不属于源码）。
