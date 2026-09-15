"""
agent.py —— Agent 主循环（整个项目的心脏，async 版）

核心思想：Agent = LLM + 工具 + 循环
  1. 把「系统提示 + 对话历史 + 工具列表」发给模型
  2. 模型要么直接回复文本（任务结束），要么请求调用工具
  3. 本地执行工具，把结果以 role="tool" 的消息塞回历史
  4. 重复，直到模型不再调用工具（或达到最大轮次）

阶段三 async 改造（Q12）：
  - run_task 改 async def，主循环用 async for chunk in chat_stream
  - 工具执行仍 sync（文件 IO/subprocess），用 asyncio.to_thread 搬进
    threadpool 跑——不阻塞事件循环，未来 Web 多会话并发也成立
  - 审批：构造时可选注入 approval_callback（sync 函数），Web 路径
    通过它把"在 threadpool 同步等" 转成"在主循环里 async 等用户响应"

消息流转图：
  user ──> assistant(文本?) ──是──> 结束
              │否(tool_calls)
              ▼
           本地执行工具 ──> tool(结果) ──> 回到模型
"""
import asyncio
import json
from typing import Callable

from . import llm
from .config import config
from .session_log import SessionLogger, make_sink, normalize_message, payload_of
from .tools import (
    TOOL_SCHEMAS,
    ToolContext,
    dispatch_tool,
)

# 模块级 config 的别名：CodingAgent.__init__ 的形参也叫 config，会遮蔽全局名。
_default_config = config


def build_system_prompt(workspace, model: str) -> str:
    """按**本会话**的沙箱根与模型名构造系统提示词（§4.6 §5.2 第 3 类失败形态）。

    为什么必须是函数而不是模块级常量：提示词里写着"你能操作哪个目录"，
    它与沙箱根是**同一份事实**。做成 import 期 f-string，工作目录就被冻死，
    per-session workspace 不可能；更糟的是"提示词说 A、沙箱其实是 B"——
    模型会去试 A 的路径、被拦、白烧几轮，而代码里没有任何地方报错。
    """
    return f"""若没有发生结构和设计上的改变不要修改注释；你是一个运行在Windows CMD用户终端里的 Coding Agent，使用没有Unix命令的能力；当用户给你任务时，你要专注于用户给的任务，不要考虑不相关的问题；你可以读写工作目录中的文件并执行命令来完成编程任务。

工作目录（沙箱根目录）：{workspace}
你的底层模型是 ：{model}（通过 OpenAI 兼容接口调用）。
被问及 "你是谁 / 什么模型" 时，以这条信息为准回答，不要凭训练记忆猜身份

你可用的工具：
- list_dir：查看目录结构
- read_file：读取文件内容
- grep：按正则搜索文件内容（只读，免审批），用于快速定位代码
- glob：按文件名模式查找文件（只读，免审批），例如 **/*.py
- write_file：创建或覆盖文件
- run_command：执行 shell 命令（运行脚本、跑测试等）
- edit_file：精确替换已有文件中的一段内容（修改已有文件时优先使用）

工作规范（必须遵守,you must follow）：
1. 接到任务后，先用 list_dir / read_file 了解相关代码，不要凭空猜测文件内容。
不清楚某段代码/某个文件在哪时，优先用 grep（搜内容）/ glob（搜文件名）快速定位，
定位到 文件:行号 后再 read_file 精读，比逐个文件盲读省上下文。
2. 修改文件前必须先 read_file 读取现状；write_file 会整体覆盖文件，务必写入完整内容。修改已有文件优先用 edit_file：只传改动片段，old_string 必须与文件逐字一致（含缩进）；
报错"未找到"就重新 read_file，报错"匹配多处"就加长上下文，不要换 write_file 整体覆盖。
3. 每次修改后，用 run_command 实际运行/测试来验证结果（例如 python 脚本、pytest）。
4. 如果命令报错，仔细阅读错误信息，定位原因后修改，再重新验证——这是正常的自我纠错流程。
5. 所有文件路径都相对于工作目录；不要尝试访问工作目录之外的路径（会被拦截）。
6. 任务完成后，用简洁的中文总结：你改了哪些文件、验证结果如何。
"""


# 默认沙箱的提示词常量：main.py、测试、以及**重放**都 import 它。
# 它是 build_system_prompt(默认值) 的求值结果，二者必须逐字一致（否则
# "日志里的 system_prompt" 与 "运行态的 messages[0]" 会对不上，重放直接错）
SYSTEM_PROMPT = build_system_prompt(config.default_workspace, config.LLM_MODEL)


# ------------------------------------------------------------
# W1（web 化前置）：输出通道
#
# 改造前 agent 的输出直接 `print` 到 stdout。stdout 是**进程级共享**的：
#   - 浏览器拿不到任何东西；
#   - 两个会话并发时两块输出会交错成乱码。
# 于是把所有输出收敛成**一个**出口 `CodingAgent._out(kind, **fields)`，
# 默认实现 `cli_out` 逐字节复刻改造前的 print（CLI 行为一字节都不变）。
#
# 为什么必须是"单一出口"而不是"web 里另接一根线"：print 与队列投递的
# **时序**必须由同一处保证，否则前端会看到"工具卡片排在文本之前/之后"
# 随机翻转——这种 bug 靠人工 review 和复现都抓不住。
#
# ⚠️ `_out` 的调用方**跨线程**：run_task 的流水在主循环线程，而 tools 的
#    审批提示在 threadpool 线程（asyncio.to_thread）。web 侧的实现必须自己
#    处理这个（见 web/session_registry.py 的 call_soon_threadsafe）。
# ------------------------------------------------------------

def cli_out(kind: str, **fields) -> None:
    """`_out` 的默认实现：与改造前的 print **逐字节一致**。

    `kind` 取值（与 web 侧 UI 事件同名，两边共用一个词汇表）：
      delta / stream_end / empty_reply / tool_call / tool_parse_error /
      max_iterations / compress_failed / task_cost / log_path
    """
    if kind == "delta":
        # 分片必须原样透传（一 delta 一事件）：在服务端做行缓冲会把"边到边"
        # 的流式体验重新变成"憋一段吐一段"
        if fields.get("first"):
            print("\n🤖 Agent：", end="", flush=True)
        print(fields["text"], end="", flush=True)
    elif kind == "stream_end":
        print()
    elif kind == "empty_reply":
        print("\n🤖 Agent：（空回复）")
    elif kind == "tool_call":
        print(f"  🔧 调用工具：{fields['name']}({fields['brief']})")
    elif kind == "tool_parse_error":
        print(f"  ❌ 工具 {fields['name']} 参数解析失败：{fields['error']}")
    elif kind == "max_iterations":
        print(f"\n⚠️  已达到最大循环轮次（{fields['limit']}），任务被迫中止。"
              f"可以拆小任务后重试，或调大 .env 中的 MAX_ITERATIONS。")
    elif kind == "compress_failed":
        print(f"  ⚠️  上下文压缩失败（任务继续）：{fields['error']}")
    elif kind == "task_cost":
        print(f"\n📊 本次任务：{fields['iterations']} 轮循环 | 消耗 {fields['used']} tokens"
              f"（prompt {fields['prompt_tokens']} / completion {fields['completion_tokens']}）"
              f"\n   cache 命中率：{fields['hit_rate']:.1f}%"
              f"（hit {fields['cache_hit']} / miss {fields['cache_miss']}）")
    elif kind == "log_path":
        print(f"会话日志：{fields['path']}")
    else:
        # 不认识的 kind = 代码 bug。这里**故意**不静默跳过：
        # "说不知道"比"装作没事"便宜（本项目最忌讳的失败形态是不报错的错结论）
        raise ValueError(f"未知的输出事件类型：{kind!r}")


# ------------------------------------------------------------
# 上下文压缩：滑窗（只保留最近 N 轮）+ 滚动摘要（把滑出窗口的轮次折成一段文字）
#
# 为什么两层都要：
#   - 只有滑窗：10 轮前用户说过的约束（"别动 utils.py"）会被静默忘掉
#   - 只有摘要：历史原文照样发出去，token 一点没省
# 于是：滑出窗口的**完整轮次**交给模型压成摘要，摘要作为一条 user 消息
# 挂在 system 后面；再过若干轮，新滑出的一批和上一版摘要一起重写成一份。
# ------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT = (
    "你是上下文压缩器。把给定的对话历史压缩成一份信息密度极高的中文摘要，"
    "供后续对话继续使用。\n"
    "必须保留：用户提出过的任务与约束、已做过的关键决定、改动过的文件与关键代码位置、"
    "执行过的命令及其结果、尚未解决的问题和下一步计划。\n"
    "尽量原样保留文件名、函数名、路径、报错信息等专有名词。\n"
    "不要寒暄、不要解释你在做什么，直接输出摘要正文。\n"
    "如果输入里还给了【上一版摘要】，请把它和新增历史合并重写成一份完整摘要，"
    "不要简单拼接、不要重复罗列。"
)

# 摘要注入时带的说明（让模型知道这段是"压缩过的旧对话"，不是用户刚说的话）
SUMMARY_HEADER = "【以下是本次会话较早内容的摘要，供你参考；近期对话原文在下面】"


def _count_turns(messages: list[dict]) -> int:
    """数“轮”：一条 role="user" 消息起一轮，其后的 assistant / tool 都算这一轮。"""
    return sum(1 for m in messages if m.get("role") == "user")


def _window_start(messages: list[dict], window: int) -> int:
    """窗口起点下标：保留最后 window 轮，且保证切出来的首条是 user 消息。"""
    if window < 1:
        window = 1
    starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not starts:
        return len(messages)          # 没有 user 就切不出合法窗口：宁可空
    if len(starts) <= window:
        return starts[0]              # 轮数不够：从最早一轮开始（顺手丢掉开头零碎消息）
    return starts[-window]


def _drop_leading_orphan_tools(history: list[dict]) -> list[dict]:
    """丢掉开头的孤儿 tool 消息（它对应的 assistant(tool_calls) 已被裁掉，接口会 400）"""
    while history and history[0].get("role") == "tool":
        history = history[1:]
    return history


def _summary_message(summary: str) -> dict:
    """把摘要包装成一条 user 消息（OpenAI 里没有“摘要”这种 role，用 user 最稳）。

    摘要正文放最前面（保持"滑窗后的历史紧跟 system 的摘要"这段一眼可辨），
    说明性 header 作为尾巴附在后面，提示模型下面才是近期对话原文。
    """
    return {"role": "user", "content": f"{summary}\n{SUMMARY_HEADER}"}


def _cap_summary(text: str, max_chars: int | None = None) -> str:
    """摘要自身也要有预算：超了截断，防止摘要膨胀成新的肿瘤。

    保留**尾部**而不是头部：摘要里越靠后越接近当前状态（最新约束 / 下一步），
    砍头部丢的是早已过时的开场信息，砍尾部才是丢西瓜。
    max_chars 缺省时用 config.SUMMARY_MAX_CHARS（测试可显式传小值）。
    """
    text = (text or "").strip()
    limit = config.SUMMARY_MAX_CHARS if max_chars is None else max_chars
    if limit > 0 and len(text) > limit:
        text = text[-limit:]
    return text


def _hard_cap(history: list[dict], limit: int) -> list[dict]:
    """
    最后一道硬砍（安全网）：窗口切完还是太长时，只留最后 limit 条消息。

    这里**严格**按条数上限切（宁可不在 user 边界起切，也不突破 limit），
    是从"语义保真"向"不炸 token 上限"的妥协——语义层已经交给滑窗和摘要去管。
    切完若首条是孤儿 tool，由调用方统一丢弃（_build_context 里做）。
    """
    if limit <= 0 or len(history) <= limit:
        return history
    return history[len(history) - limit:]


def _messages_for_summary(messages: list[dict]) -> list[dict]:
    """
    裁剪送进摘要器的历史消息。

    为什么不原样把 self.messages 丢过去：tool 消息里常有上万字的文件内容，
    原样送过去 token 花在噪声上，真正重要的"用户约束 / 改了什么文件"反而被淹。
    所以工具结果按 SUMMARY_TOOL_RESULT_CHARS 截断；其余消息保持**原始 role**
    （user / assistant / tool）不变——摘要模型需要看到对话结构，才能分清
    "这句是用户提的要求"还是"这句是助手的汇报"，压出来的摘要才不会张冠李戴。
    """
    limit = config.SUMMARY_TOOL_RESULT_CHARS
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            content = m.get("content") or ""
            if limit > 0 and len(content) > limit:
                content = content[:limit] + f"…（原 {len(content)} 字，已截断）"
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": content,
            })
        else:
            out.append(dict(m))
    return out


async def _collect_text(messages: list[dict], stream_fn=None) -> tuple[str, object | None]:
    """
    非流式用途（摘要）：复用 chat_stream 把整段回复收成一个字符串。

    复用的好处是重试、超时、usage 处理全都在 llm 层，这里不重复实现；
    注意这里不传 tools —— 摘要调用绝不能反过来触发工具调用。

    stream_fn 缺省（None）→ **调用点**取模块级 llm.chat_stream：这样
    `monkeypatch.setattr(agent_mod.llm, "chat_stream", ...)` 依旧生效；
    per-session 时由 CodingAgent 传自己那份（LLMClient 或假模型）。
    """
    stream_fn = llm.chat_stream if stream_fn is None else stream_fn
    parts: list[str] = []
    usage = None
    async for chunk in stream_fn(messages):
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        piece = getattr(chunk.choices[0].delta, "content", None)
        if piece:
            parts.append(piece)
    return "".join(parts), usage


async def _summarize(old_messages: list[dict], prev_summary: str | None, stream_fn=None) -> str:
    """
    把一段历史压成摘要（async）。

    prev_summary 不为空时作为"上一版摘要"一起喂回去，让模型把它和新增历史
    **重写合并**成一份新摘要——而不是把两段摘要拼起来（拼起来长度会随压缩
    次数线性膨胀，摘要最终会变成新的肿瘤）。
    失败时向外抛异常，交由 compress_history 兜底：摘要挂了不能带崩主任务。

    stream_fn 同 _collect_text：缺省时回落到模块级 llm.chat_stream。
    """
    request: list[dict] = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]
    if prev_summary:
        request.append({"role": "user", "content": f"【上一版摘要】\n{prev_summary}"})
    request.extend(_messages_for_summary(old_messages))
    text, _ = await _collect_text(request, stream_fn)
    return text


class CodingAgent:
    """保存对话历史、驱动"模型↔工具"循环的 Agent 实例（async 版）

    §4.6 前置改造（step 5）：构造里**不再写任何全局**——只存自己的
    config / llm / ctx / logger。于是同一进程里两个 Agent 并发跑不串台：
    各自的沙箱、审批、审计、日志、模型客户端都是自己的。
    """

    def __init__(
        self,
        approval_callback: Callable[[str, str], bool] | None = None,
        session_log: SessionLogger | None = None,
        *,
        config=None,
        llm=None,
        ctx: ToolContext | None = None,
        out: Callable[..., None] | None = None,
    ) -> None:
        """
        :param approval_callback: 审批回调（sync），可选。
            签名 cb(action: str, detail: str) -> bool。
            **只在没有 ctx 时生效**——那时它被放进本实例现造的私有 ctx（见下）。
        :param session_log: 会话日志（note3 的 JSONL）。None = 不落盘，
            行为与加日志之前**完全一致**（现有测试全靠这个默认值不受影响）。
        :param config: 该实例使用的配置对象。None = 模块级默认 config（同一对象，
            所以 monkeypatch.setattr(cfg, "X", ...) 依旧生效）。
            ⚠️ 存的是**引用**，标量一律在调用点现读——构造期抄一份会让
            "会话中途改配置不生效"，而日志里是旧快照 → 记的和跑的对不上。
        :param llm: 该实例的模型客户端（LLMClient 或等价对象，需有
            `chat_stream(messages, tools=None)`）。None = 模块级 llm.chat_stream（默认实例）。
        :param ctx: 该会话的 ToolContext（workspace / approval / event_sink / logger）。
            None = **给本实例现造一个私有的 ctx**（4.6 终态）。注意这**不是**回落：
            改造前它写的是模块级默认上下文（进程内共享 → 两个实例会串台），
            现在构造里**一个全局都不碰**，每个实例的 ctx 都是自己的。
        """
        # messages 是完整对话历史，OpenAI 消息格式
        #
        # 先定沙箱根：本会话的 ctx.workspace 优先，否则默认 config.default_workspace
        self._config = _default_config if config is None else config
        self._llm = llm
        # 4.6 终态：没有 ctx 时**不再往进程级默认上下文上写**，而是给本实例现造一个。
        # 这个差别是本质性的——改造前 `CodingAgent()` 拿到的是**全进程共享**的那个
        # ctx，两个这样的实例会互相看见对方的审批回调与审计 sink（静默串台，
        # 而且不报错）；现在每个实例的 ctx 都是自己的，构造里一个全局都不碰。
        if ctx is None:
            ctx = ToolContext(
                workspace=None,       # None → _workspace() 回落到 config.default_workspace
                approval_callback=approval_callback,
                event_sink=make_sink(session_log) if session_log is not None else None,
                logger=session_log,
                config=self._config,
                out=out,
            )
        self.ctx = ctx
        # W1 输出通道：显式传入优先，其次取本会话 ctx 上的那个（web 走后者）。
        # 都是 None → cli_out（与改造前逐字节一致）。
        self._out_impl = out if out is not None else ctx.out
        workspace = (
            ctx.workspace if ctx.workspace is not None else self._config.default_workspace
        )
        # 提示词与沙箱**同源**：同一个 workspace 值同时喂给这里和 tools（§5.2）
        self.system_prompt = build_system_prompt(workspace, self._config.LLM_MODEL)
        self.messages: list[dict] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.reset_stats()

        # 会话日志（可为 None）。seq 由 logger 单点分配，这里只负责在
        # **commit 边界**调用它——流式 delta 绝不落盘（note3 §4.3）
        self.session_log = session_log
        # _message_seqs 与 self.messages 逐位对齐，记录每条消息落盘时的 seq。
        # 有了它才能在裁剪时算出 dropped_upto_seq（水位）。
        # 首元素对应 system 消息：它不是源事件，故为 None
        self._message_seqs: list[int | None] = [None]

        # 4.6 终态：这里原本有一段 `if ctx is None:` —— 把 approval_callback /
        # session_log 写进 tools 的**模块级默认上下文**。默认上下文已随
        # `_default_ctx` 一起删除，本构造函数不再触碰任何进程级状态
        # （没有 ctx 时上面已经给本实例现造了一个私有的）。

        # 上下文压缩状态：
        #   _summary          已滑出窗口的历史滚动摘要（None=还没压过）
        #   _compressed_turns self.messages 中「已被摘要覆盖」的**轮数**（user 消息数）
        #                     用"轮数"而不是"下标"记边界：新增消息不会让边界失准，
        #                     也不必真的把消息从 self.messages 里删掉，
        #                     原始历史仍然完整保留（既方便调试，也方便以后落盘）
        #   _dropped_upto_seq 水位：内存态只保留 seq > 它的源事件（初始 0 = 什么都没丢）
        self._summary: str | None = None
        self._compressed_turns: int = 0
        self._dropped_upto_seq: int = 0

    # --------------------------------------------------------
    # W1 输出通道：**唯一**出口
    # --------------------------------------------------------
    def _out(self, kind: str, **fields) -> None:
        """产出一次"给人看"的输出/UI 事件。

        - 实例没注入 out → `cli_out`（终端 print，与改造前逐字节一致）；
        - web 注入的实现 → 转成 SSE 事件推给浏览器。

        ⚠️ **不要**在别处直接 `print`：单一出口是事件时序的保证（见模块头注释）。
        ⚠️ 本方法**可能在任何线程被调用**（tools 的审批提示在 threadpool 里），
           注入的实现必须自己保证线程安全。
        """
        if self._out_impl is None:
            cli_out(kind, **fields)
        else:
            self._out_impl(kind, **fields)

    # --------------------------------------------------------
    # 会话日志：唯一的追加入口 + 派生状态快照
    # --------------------------------------------------------
    def _emit(self, rtype: str, **fields) -> int | None:
        """投一条记录到会话日志。没有 logger 时是 no-op（返回 None）。"""
        if self.session_log is None:
            return None
        return self.session_log.emit(rtype, **fields)

    def _append_message(self, msg: dict) -> None:
        """**唯一的**消息追加入口：写内存 + 落盘（一个 commit 边界）。

        三处源事件（user / assistant / tool）都必须走这里，否则
        `_message_seqs` 会与 `self.messages` 错位、水位随之算错。
        """
        msg = normalize_message(msg)      # content=None → ""（落盘前必做）
        self.messages.append(msg)
        seq = self._emit(msg["role"], **payload_of(msg))
        self._message_seqs.append(seq)

    def _write_state(self) -> None:
        """把三个派生状态整体快照成一行 `state`（低频：只在派生态真变时写）。

        `summary` 存全文，这正是"不全量 dump"的取舍点：摘要若不落盘，
        重放就得重跑 `_summarize`（非确定 + 烧 token）。
        """
        self._emit(
            "state",
            summary=self._summary,
            compressed_turns=self._compressed_turns,
            dropped_upto_seq=self._dropped_upto_seq,
        )

    def reset_conversation(self) -> None:
        """开新会话：清空对话历史和压缩状态（/clear 用）

        日志侧：仍 append 一行 `clear`，**不新建文件、不截断文件**。
        于是"一个文件 = 一个会话的全部生命史（含几次 clear）"，重放只有一处分支。
        """
        self.messages = self.messages[:1]
        self._message_seqs = self._message_seqs[:1]
        self._summary = None
        self._compressed_turns = 0
        self._dropped_upto_seq = 0
        self.reset_stats()
        self._emit("clear")

    def reset_stats(self) -> None:
        """重置运行统计（__init__ 时初始化；/clear 开新会话时也调用）"""
        self.stats = {
            "tasks": 0,              # 执行过的任务数
            "requests": 0,           # 模型请求次数（每轮循环 1 次）
            "tool_calls": 0,         # 实际执行的工具调用次数
            "prompt_tokens": 0,      # 输入 token 累计（大头：每轮重发历史）
            "completion_tokens": 0,  # 输出 token 累计
            # DeepSeek prefix cache 命中统计（Q13 缓存优化）
            # 命中按 0.1x 计费，未命中按 1x——命中率直接决定成本
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }

    # --------------------------------------------------------
    # 上下文管理：滑窗（只留最近 N 轮）+ 滚动摘要（把滑出窗口的轮次折成一段文字）
    # --------------------------------------------------------
    def _uncompressed_history(self, history: list[dict]) -> list[dict]:
        """切掉已被摘要覆盖的前 _compressed_turns 轮，只留未压缩的历史。"""
        if self._compressed_turns <= 0:
            return history
        starts = [i for i, m in enumerate(history) if m.get("role") == "user"]
        if self._compressed_turns >= len(starts):
            return []  # 全部轮次都已折进摘要
        return history[starts[self._compressed_turns]:]

    def _build_context(self) -> list[dict]:
        """
        构造发给模型的消息列表（同步、无副作用，绝不改 self.messages）。

        Q13 P0：单任务内 messages 只增不减，保证前缀稳定 → DeepSeek prefix cache 可命中。
        压缩状态（_summary / _compressed_turns）由 compress_history 在任务开头维护，
        _build_context 只读不改——不在每轮 shift 窗口，否则前缀漂移、cache 全废。
        语义层的"滑窗"已由 _compressed_turns 表达，物理裁剪移到 _maybe_trim_after_task。
        """
        context = [self.messages[0]]
        if self._summary:
            context.append(_summary_message(self._summary))
        context.extend(self._uncompressed_history(self.messages[1:]))
        return context

    async def compress_history(self) -> bool:
        """
        把"已滑出窗口的完整轮次"折进滚动摘要。

        触发时机：滑出窗口的轮数 - 已压缩轮数 >= HISTORY_COMPRESS_LAG 才压，
        避免"滑出一轮就压一轮"（每轮多一次 API 调用，比不压还贵）。
        返回 True 表示本次真的压缩了；False 表示没到阈值 / 无新内容 / 摘要失败。

        安全性：
          - 只压**完整轮次**（压缩边界落在 user 消息上），尾部残缺轮次
            （assistant 刚发出 tool_calls、工具结果还没回来）永远留在窗口里，
            否则下一轮请求会因 tool_call_id 失配报 400
          - 摘要模型失败时保留原摘要、不推进游标，返回 False，主任务照常继续
          - 绝不就地修改 self.messages
        """
        history = self.messages[1:]
        starts = [i for i, m in enumerate(history) if m.get("role") == "user"]
        if not starts:
            return False

        window_start_idx = _window_start(history, self._config.HISTORY_WINDOW_TURNS)
        # 窗口起点之前都是完整轮次，其数量就是"已滑出窗口的轮数"
        turns_before_window = starts.index(window_start_idx)
        if turns_before_window - self._compressed_turns < self._config.HISTORY_COMPRESS_LAG:
            return False
        if self._compressed_turns >= len(starts):
            return False

        old_messages = history[starts[self._compressed_turns]:window_start_idx]
        if not old_messages:
            return False

        try:
            summary = await _summarize(old_messages, self._summary, self._stream)
        except Exception as e:  # noqa: BLE001 —— 摘要失败绝不能带崩主任务
            self._out("compress_failed", error=f"{type(e).__name__}: {e}")
            return False

        # 预算在调用点读本实例的 config（显式传参，避免模块级 helper 读全局）
        summary = _cap_summary(summary, self._config.SUMMARY_MAX_CHARS)
        if not summary:
            return False

        self._summary = summary
        self._compressed_turns = turns_before_window
        # 派生态变了（摘要 + 已压缩轮数）→ 落一行 state 快照。
        # 低频：压缩本来就要攒够 HISTORY_COMPRESS_LAG 轮才触发，不是每轮都写
        self._write_state()
        return True

    def _maybe_trim_after_task(self) -> None:
        """任务结束后裁剪一次，避免跨任务累积爆 token。

        单任务内不裁剪是为了 prefix cache 命中；任务结束后才裁一次，
        保证下一任务开始时前缀稳定（下一任务内再次只增不减）。

        注意：裁剪会同步回退 _compressed_turns——被物理删掉的轮次不该再算进
        "已折进摘要的轮数"，否则 _build_context 的切片下标会整体错位。
        """
        history = self.messages[1:]
        if len(history) <= self._config.MAX_HISTORY_MESSAGES:
            return
        cut = len(history) - self._config.MAX_HISTORY_MESSAGES
        removed_turns = sum(1 for m in history[:cut] if m.get("role") == "user")
        # 水位：被物理丢弃的源事件里 seq 最大的那个。
        # 这里是两个坐标系的**交换点**：裁掉 k 条消息 → 水位前移，
        # 同时 _compressed_turns 回退，两者一起表达"丢了什么"。
        dropped_seqs = [s for s in self._message_seqs[1:1 + cut] if s is not None]
        # 丢弃开头孤立的 tool 消息（它对应的 assistant 调用记录已被裁掉）
        retained = _drop_leading_orphan_tools(history[cut:])
        self.messages = [self.messages[0]] + retained
        # _message_seqs 同步切到"保留区"的尾部，与 self.messages 保持逐位对齐。
        # 这里"取末尾 len(retained) 个"成立，靠的是 **retained 必是 history 的后缀**：
        # _drop_leading_orphan_tools 只从头部剥，绝不动中段/尾部。若将来它改成
        # 会丢中间元素，本切片会静默错位——改动那个纯函数时务必回来核对这里。
        # （等价写法：起点 = len(seqs) - len(retained) = 1 + cut + 被剥掉的孤儿数）
        # 注意：孤儿 tool 被 _drop_leading_orphan_tools 丢掉时**不推进水位**——
        # 它的 seq > 新水位，水位表达不了它，重放靠复用同一个纯函数丢弃（note3 §3.2）
        if retained:
            kept_from = len(self._message_seqs) - len(retained)
            self._message_seqs = self._message_seqs[:1] + self._message_seqs[kept_from:]
        else:
            # 保留区被孤儿 tool 吃光（末 MAX_HISTORY_MESSAGES 条全是 tool，
            # 即 MAX_ITERATIONS 用尽中止、一轮多工具的形态）→ 只剩 system 一条
            self._message_seqs = self._message_seqs[:1]
        self._compressed_turns = max(0, self._compressed_turns - removed_turns)

        advanced = False
        if dropped_seqs:
            new_watermark = max(dropped_seqs)
            if new_watermark > self._dropped_upto_seq:
                self._dropped_upto_seq = new_watermark
                advanced = True
        if advanced or removed_turns:
            self._write_state()

    # --------------------------------------------------------
    # 依赖的取用点（**调用点**读，绝不构造期缓存）
    # --------------------------------------------------------
    def _stream(self, messages: list[dict], tools: list[dict] | None = None):
        """取"这次该用哪个 chat_stream"。

        实例没注入 llm 时回落到**模块级** `llm.chat_stream` —— 回落发生在
        调用点，所以 `monkeypatch.setattr(agent_mod.llm, "chat_stream", ...)`
        （压缩 / 会话日志测试全靠它）依旧生效；构造期把它绑死就会静默失效，
        测试还是绿的，但已经不测任何东西了。
        """
        if self._llm is None:
            return llm.chat_stream(messages, tools)
        return self._llm.chat_stream(messages, tools)

    async def _run_tool(self, name: str, args: dict) -> str:
        """在 threadpool 里执行工具。

        4.6 终态：ctx 必填，所以**恒定传三参**——`self.ctx` 现在保证非 None
        （构造时就地现造）。注意这条同时约束了测试替身：monkeypatch 进来的
        `dispatch_tool` 必须是 3 参（`(name, args, ctx)`），2 参替身会 TypeError。
        """
        return await asyncio.to_thread(dispatch_tool, name, args, self.ctx)

    # --------------------------------------------------------
    # 主循环：处理一个用户任务（async）
    # --------------------------------------------------------
    async def run_task(self, user_input: str) -> None:
        # 1. 用户消息入历史（commit 边界 ①：落盘）
        self._append_message({"role": "user", "content": user_input})
        self.stats["tasks"] += 1
        # 记录任务开始时的累计值，用于算"本次任务"的增量（不是累计）
        start_prompt = self.stats["prompt_tokens"]
        start_completion = self.stats["completion_tokens"]
        start_hit = self.stats["prompt_cache_hit_tokens"]
        start_miss = self.stats["prompt_cache_miss_tokens"]

        # 1.5 请求模型之前先压缩历史：把已滑出窗口的旧轮次折进摘要，
        #     保证上下文不会随任务变长无限膨胀。攒不够滞后量时是 no-op，不额外花 API。
        await self.compress_history()

        # 2. 循环：模型调用 <-> 工具执行
        for step in range(1, self._config.MAX_ITERATIONS + 1):
            # 2.1 async 流式调模型，边到边消费 chunk
            #     与 sync 版完全同构，仅 for → async for
            content_parts: list[str] = []
            tool_call_acc: dict[int, dict] = {}
            usage = None
            print_prefix_shown = False

            async for chunk in self._stream(self._build_context(), tools=TOOL_SCHEMAS):
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    self._out("delta", text=delta.content, first=not print_prefix_shown)
                    print_prefix_shown = True
                    content_parts.append(delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_acc:
                            tool_call_acc[idx] = {
                                "id": tc.id or "",
                                "name": (tc.function.name if tc.function and tc.function.name else ""),
                                "arguments": "",
                            }
                        if tc.id and not tool_call_acc[idx]["id"]:
                            tool_call_acc[idx]["id"] = tc.id
                        if tc.function and tc.function.name and not tool_call_acc[idx]["name"]:
                            tool_call_acc[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_call_acc[idx]["arguments"] += tc.function.arguments

            if print_prefix_shown:
                self._out("stream_end")

            # 2.2 统计请求数和 token
            self.stats["requests"] += 1
            if usage:
                self.stats["prompt_tokens"] += usage.prompt_tokens
                self.stats["completion_tokens"] += usage.completion_tokens
                # DeepSeek 返回 prompt_cache_hit_tokens / prompt_cache_miss_tokens
                # 其他兼容接口可能不返回，用 getattr 兜底防 AttributeError
                self.stats["prompt_cache_hit_tokens"] += getattr(usage, "prompt_cache_hit_tokens", 0) or 0
                self.stats["prompt_cache_miss_tokens"] += getattr(usage, "prompt_cache_miss_tokens", 0) or 0

            # 2.3 把聚合后的模型回复存入历史
            #     commit 边界 ②：**定稿后**才落盘，流式过程中的 delta 一律不落
            content = "".join(content_parts)
            assistant_msg: dict = {"role": "assistant", "content": content}
            if tool_call_acc:
                tool_calls_list = []
                for idx in sorted(tool_call_acc):
                    tc = tool_call_acc[idx]
                    tool_calls_list.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    })
                assistant_msg["tool_calls"] = tool_calls_list
            self._append_message(assistant_msg)

            # 2.4 模型没有请求工具 -> 任务结束
            if not tool_call_acc:
                if not print_prefix_shown and not content:
                    self._out("empty_reply")
                self._out("stream_end")
                self._report_task_cost(step, start_prompt, start_completion, start_hit, start_miss)
                self._maybe_trim_after_task()  # Q13：任务结束后才裁剪，保护单任务内的 prefix cache
                return

            # 2.5 模型请求了工具 -> 逐个执行
            #     dispatch_tool 是 sync（含 _ask_approval），
            #     用 asyncio.to_thread 搬进 threadpool 跑不阻塞事件循环
            for idx in sorted(tool_call_acc):
                tc = tool_call_acc[idx]
                tool_name = tc["name"]
                try:
                    tool_args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError as e:
                    tool_result = f"参数解析失败（不是合法 JSON）：{e}"
                    tool_args = None
                    self._out("tool_parse_error", name=tool_name, error=str(e))

                if tool_args is not None:
                    self._out("tool_call", name=tool_name, brief=_brief(tool_args))
                    # sync 工具执行搬进 threadpool；如果里面调了
                    # input() 或注入的 sync callback，也只阻塞这个线程
                    tool_result = await self._run_tool(tool_name, tool_args)
                    self.stats["tool_calls"] += 1

                self._append_message(
                    {"role": "tool", "tool_call_id": tc["id"], "content": tool_result}
                )

        # 3. 达到最大轮次还没结束
        self._out("max_iterations", limit=self._config.MAX_ITERATIONS)
        self._report_task_cost(self._config.MAX_ITERATIONS, start_prompt, start_completion, start_hit, start_miss)
        self._maybe_trim_after_task()  # Q13：异常中止也要裁剪，避免下任务带残留

    def _report_task_cost(
        self, step: int,
        start_prompt: int, start_completion: int,
        start_hit: int, start_miss: int,
    ) -> None:
        """任务结束时打印本次消耗（数据来自 API usage，不是模型自述）。

        所有指标都是"本次任务"的增量，不是累计——第二任务起才不会把上次的
        hit 算进当前任务的命中率。原版用累计 hit / 累计 prompt，首任务碰巧
        对，第二任务起就错。
        """
        cur_prompt = self.stats["prompt_tokens"]
        cur_completion = self.stats["completion_tokens"]
        task_prompt = cur_prompt - start_prompt
        task_completion = cur_completion - start_completion
        used = task_prompt + task_completion
        task_hit = self.stats["prompt_cache_hit_tokens"] - start_hit
        task_miss = self.stats["prompt_cache_miss_tokens"] - start_miss
        hit_rate = (task_hit / task_prompt * 100) if task_prompt > 0 else 0
        # 只落**本次任务增量**，不落累计：累计 = 增量的纯函数（免费且确定的重算），
        # 落冗余字段只会制造"两个数打架"的风险（note3 §2.5）
        self._emit(
            "task_end",
            iterations=step,
            prompt_tokens=task_prompt,
            completion_tokens=task_completion,
            cache_hit=task_hit,
            cache_miss=task_miss,
        )
        self._out(
            "task_cost",
            iterations=step, used=used,
            prompt_tokens=task_prompt, completion_tokens=task_completion,
            hit_rate=hit_rate, cache_hit=task_hit, cache_miss=task_miss,
        )


def _brief(args: dict, max_len: int = 120) -> str:
    """把工具参数压缩成一行用于终端展示（长内容省略）"""
    parts = []
    for k, v in args.items():
        s = str(v).replace("\n", "\\n")
        if len(s) > max_len:
            s = s[:max_len] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
