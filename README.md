# 手写 Coding Agent（MVP）—— CLI + Web 两条入口

一个自己从零实现的 Coding Agent：模型 + 工具 + 循环。
支持流式输出、工具调用（读写文件 / 搜索 / 执行命令）、人工审批、上下文压缩、
以及 **JSONL 会话日志（可重放、可审计）**。

阶段三·b 给它接了第二根线：**Web 模式（FastAPI + SSE）**，
于是同一个 agent 既能跑在终端里，也能跑在浏览器里。

---

## 1. 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 配 Key
copy .env.example .env          # 然后填入 LLM_API_KEY

# 3a. CLI 模式（原有体验，行为未变）
python main.py

# 3b. Web 模式
python run_web.py               # 打开 http://127.0.0.1:8000/
```

> Web 模式**必须 `workers=1`**（`run_web.py` 已经写死）。
> 活跃会话在**进程内**的注册表里，多 worker 会让"建会话落在 A 进程、
> 发消息打到 B 进程" → 404，**而且不会报任何错**。

跑测试：

```bash
python -m pytest -q             # 全部（含 web）
python -m pytest -q test_instance_isolation.py test_web_api.py
```

> ⚠️ `pytest.ini` 的 `testpaths` 就是**闸门本身**：漏挂一个文件 =
> 那个文件的用例"永远绿"（从不执行却看起来通过）。新增测试文件必须加进去。

---

## 2. Web 模式长什么样

```
浏览器（web/static）  ──POST /api/sessions────────────────→  建会话，返回 sid
                     ──POST /api/sessions/{sid}/messages───→  派任务（202，不阻塞）
                     ←─GET  /api/sessions/{sid}/stream─────  SSE：文本 / 工具 / 审批 / 统计
                     ──POST /api/sessions/{sid}/approvals/{aid}→  点按钮（允许 / 拒绝）
                     ──GET  /api/sessions/{sid}/replay─────→  断线重连的"已提交态"
```

设计上的三条硬约束（与内部的 4.6 改造同构）：

1. **CLI 行为逐字节不变** —— 输出通道默认实现就是原来的 `print`，
   改造前后跑同一个任务，终端输出 `diff` 为空。
2. **两个浏览器会话并发不串台** —— 会话身份（sid / 审批 / 审计 / 日志）各自独立；
   若两个会话指向**同一个** workspace 路径，文件层共享是用户显式选择，UI 会标记。
3. **落盘仍是同一份 JSONL 契约** —— 一个会话一个文件，`replay()` 仍能重建状态。

### 2.1 前端**故意**做得很小

不做：文件树、代码高亮编辑器、diff 视图。
理由：这些会**绕过审批与沙箱语义**——"点一下就能看到文件"等于给了一条
不经过工具层（也就没有凭据封锁、没有审批、没有预算）的旁路。
要浏览文件，就让模型调 `list_dir` / `read_file`。

### 2.2 服务端不持久化活跃会话

活跃会话在内存里。**重启服务端 = 活跃会话丢失，但历史仍可 `replay`**
（日志一直在磁盘上）。不假装持久。

---

## 3. 安全边界（请务必读完）

### 3.1 绑定与鉴权

| 配置 | 默认 | 说明 |
|---|---|---|
| `WEB_HOST` | `127.0.0.1` | 只监听本机 |
| `WEB_TOKEN` | 空（不鉴权） | 设了就要求 `Authorization: Bearer <token>`；SSE 用 `?token=` |

🔴 **`WEB_HOST=0.0.0.0` 且 `WEB_TOKEN` 为空 = 把你的终端开放给全网。**
`run_command` 的黑白名单挡的是"危险命令**形态**"（`del` / `format` / `cmd /c` /
`python -c` …），它**挡不住"用 python 跑一个脚本"**——那是设计上的已知缺口
（完备解需要 OS 级沙箱）。所以这条路要么绑本机，要么加 token。

### 3.2 workspace 由服务端派生

前端只能提交**工作区名字**，服务端 `resolve()` 后校验它必须在
`WEB_WORKSPACE_ROOT` 之下。提交任意路径（`C:\Windows`、`../..`）会被 400 拒掉。
web 层**不提供**"列目录 / 读文件 / 下载"的旁路 API。

### 3.3 会话审计日志的位置

`AGENT_SESSION_DIR` 是会话日志（JSONL）目录。它必须待在 workspace **之外**，
否则会话 A 的 agent 一句 `read_file sessions/<B>.jsonl` 就能拿到 B 的
system prompt、全部对话与工具输出；还能用 `write_file` / `edit_file`
**篡改自己或别人的审计日志**——而审计的价值恰恰是"事后不可抵赖"。

- **Web 模式**：默认落在 `%LOCALAPPDATA%\coding-agent\sessions`。
  启动时**断言**它不在 `WEB_WORKSPACE_ROOT` 之内，不满足就**拒绝启动并给出修法**
  （只打 warning 等于没做：警告会被忽略、日志会滚走）。
- **CLI 模式（诚实承认）**：默认 `./sessions` **仍然在工作目录里**。
  单用户 MVP 可接受，拦截靠工具层那道**按路径**的守卫
  （读/写 `SESSION_DIR` 下的任何文件一律拒绝，并记 `source=policy` 的审计）。
  **不是"已经移出"**。
- 那道守卫是**按路径**判的，不是"文件名带 `*.jsonl` 就拦"。
  后者会误伤你自己的数据（`train_data.jsonl` 不是审计日志），
  而且拒绝文案会**说谎**（回喂"属于凭据/密钥类文件"，可它根本不是）。

### 3.4 同名 workspace 的三条风险（**用户已确认接受**）

两个会话可以指向同一个工作区目录（UI 会标「共享工作区」）。
这是**显式共享**，不是隔离——文件层**根本不隔离**，且这一点不可能靠代码解决
（给每个会话建专属子目录？建空目录则 agent 看不到你的项目；指向已有目录则"独占"是假的）。

所以把它**写下来**（"能接受"的前提是写下来，不是藏起来）：

1. **文件层不隔离**：`sessions/` 分开了，**业务文件没有**。
   会话 A 写的文件，会话 B 立刻能读到；
2. **并发写无检测、静默覆盖**：实测两个会话对同一路径写入时
   **都说"已成功"**，磁盘上只剩后写的那一版——没有报错、没有版本冲突、没有审计标记。
   是"成功地在别人的文件上成功"；
3. **审批是按会话弹的，detail 前缀完全相同**：A 批准的一次写入可能改到 B
   正在读/正在写的文件；而 A、B 审批窗口里的 `detail` 长得一模一样，
   **你无法从弹窗分辨是哪个会话的操作**，只能靠自己记得哪个 tab 是哪个会话。

缓解只有两条，都不完美：UI 上的**常驻**「共享工作区」标记（不是一闪而过的提示），
以及日志里的 `session_start.shared=true` + `shared_with=[...]`（事后可对账）。

---

## 4. 配置速查

`.env` 里所有 Web 项都是可选的（见 `.env.example` 的注释）：

```
WEB_HOST=127.0.0.1          # 别改成 0.0.0.0，除非配了 WEB_TOKEN
WEB_PORT=8000
WEB_TOKEN=                  # 留空 = 不鉴权
WEB_WORKSPACE_ROOT=         # 留空 = AGENT_WORKSPACE / 当前目录
WEB_APPROVAL_TIMEOUT=300    # 秒；等不到答案 → 默认拒绝（fail-closed）
WEB_QUEUE_MAXSIZE=256       # SSE 队列上限（消费者会消失，所以必须有界）
```

---

## 5. **明确不做**的事（不是"还没做"，是"想清楚了不做"）

| 不做 | 为什么 |
|---|---|
| 取消（cancel）按钮 | 工具层**没有取消点**，硬中断会写出半截文件。v1 决策：客户端关页面 ≠ 任务停止，**任务继续跑完、结果只落盘**。 |
| 多 worker / 横向扩展 | 活跃会话是进程内状态。要扩展得先把注册表挪到 Redis 之类的地方。 |
| 前端文件浏览器 / 编辑器 | 会绕过审批与沙箱（见 §2.1）。 |
| 重放"未定型的 delta" | 需要把流式 delta 落盘，正是 note3 反对的写放大。重连只能从**最后一个落盘记录**续。 |
| 数据库 | 单机内存态 + JSONL 落盘够用；JSONL 还能直接 `tail` 和 `replay`。 |
| 完整 OS 级沙箱 | `run_command` 走 `shell=True`，文件层沙箱管不到 shell。当前是 90 分解（命令黑白名单 + 审批 + 超时）。 |

---

## 6. 项目结构

```
main.py                      CLI 入口 + 组合根 build_session()
run_web.py                   Web 入口（uvicorn，workers=1）
agent/
  agent.py                   主循环（async）+ 输出通道 _out()
  tools.py                   7 个工具 + 沙箱/审批/审计/凭据封锁/命令策略
  config.py                  配置（含 WEB_*）
  session_log.py             JSONL 会话日志（单点 seq + writer 线程 + 重放）
  llm.py                     OpenAI 兼容客户端（流式 + 重试）
web/
  app.py                     FastAPI 路由 + SSE + lifespan
  session_registry.py        sid → WebSession；同名 workspace 的共享标记
  events.py                  UI 事件模型 + SSE 序列化 + 有界队列
  approvals.py               审批的 sync→async 桥
  schemas.py                 请求/响应模型
  static/                    极简前端
test_*.py                    共 11 个文件、235 个用例（其中 4 个文件、54 个用例是 web 验收 W-T1~W-T12）
dist_tests/                  测试打包副本（15 个文件，可整体交付；见其 README.md）
tools/                       留用的工具脚本（make_dist_tests / list_tests）
docs/                        设计与过程文档（todolist / note / 问答稿）
archive/                     一次性脚本与运行输出的归档（确认后可整个删掉）
sessions/                    CLI 会话日志（运行时产物，已在 .gitignore 中）
```

> 测试文件共 11 个、235 个用例（`python tools/list_tests.py` 可核对闸门一致性）。
> 重新生成交付副本：`python tools/make_dist_tests.py`。

---

## 7. 排障

| 现象 | 多半是 |
|---|---|
| 启动直接失败，提示"拒绝启动：会话日志目录…在工作区根之内" | 按提示删掉 `.env` 里的 `AGENT_SESSION_DIR`，或把它指到工作区外 |
| 浏览器一直"连接中断" | 用了多 worker，或中间层缓冲了 SSE（响应头已带 `X-Accel-Buffering: no`） |
| 点了按钮没反应 | 会话已被判定为"没人看"（比如你切到了别的会话），审批会被 fail-closed 拒绝；重新选中会话再派任务 |
| 派任务返回 409 | 该会话已有一个任务在跑（每会话同时只允许 1 个，避免两条主线往同一个上下文里插消息） |
| 点"允许"返回 409 | 这个审批已经超时或被处理过。**不会**改变结论（防止静默改判） |
