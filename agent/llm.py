"""
llm.py —— 大模型客户端（很薄的一层封装，async 流式版本）

DeepSeek、通义千问、Kimi 等国产模型都提供 OpenAI 兼容接口，
所以直接用 openai SDK，只需改 base_url 和 model 即可切换厂商。

阶段三改造：sync generator → async generator（AsyncOpenAI + async for）
  - async LLM 调用是阶段三 async 改造的核心动机
  - 工具执行仍 sync（文件 IO/subprocess），通过 asyncio.to_thread 调度
  - 重试逻辑保持原样（Q11），next(stream) → anext(stream)，for → async for

兼容性已验证（demo_stream.py 2026-09-14）：
  - 文本 delta 分片 ✅
  - tool_calls 增量聚合 ✅
  - usage 在最后 chunk ✅
  - AsyncOpenAI 与 OpenAI 同协议，仅 IO 模型不同 ✅

§4.6 前置改造（step 4）：新增 `LLMClient`（每会话一份，持有自己的 client+config）。
模块级 `client` / `chat_stream` **原样保留**，作为默认实例与回落路径——
test_llm_retry.py 全靠 patch 这些模块级名字工作，所以它们必须仍在**调用点**被查。
"""
import asyncio
import random
import time
from typing import AsyncIterator
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError

from .config import config

# 全局复用一个 async 客户端（内部自带连接池，不要每次调用都新建）
# max_retries=0：关掉 SDK 内置重试，重试统一由 chat_stream 应用层做，
# 避免"SDK 层 × 应用层"双层重试乘法（2×3=最多 9 次请求，等待复利）
#
# §4.6 改造后：这个模块级 client 是**默认实例**用的，per-session 请用 LLMClient。
# 老路径（模块级 chat_stream + 这个 client）保持原样：test_llm_retry.py 全程
# 靠 patch `llm.client` / `llm.config` / `llm.asyncio` / `llm.random` 工作，
# 所以模块级 chat_stream 里的这些名字必须始终是**调用点**查全局。
client = AsyncOpenAI(
    api_key=config.LLM_API_KEY,
    base_url=config.LLM_BASE_URL,
    timeout=config.LLM_TIMEOUT_SECONDS,
    max_retries=0,
)

# 模块级 config 的别名：LLMClient.__init__ 的形参也叫 config，会遮蔽全局名。
_default_config = config


# ============================================================
# 重试策略（Q11）—— 与 sync 版完全一致，重试分类是纯函数无需改 async
# ============================================================

_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})
_RETRYABLE_TYPES = (APITimeoutError, APIConnectionError)
_RETRY_MAX_WAIT = 30.0


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否值得重试。

    用显式谓词而非 except 链分类——openai 异常是继承体系
    （RateLimitError 是 APIStatusError 的子类），except 分支顺序写错时
    子类会被父类截胡；谓词函数没有这个陷阱，也方便用 SimpleNamespace 测试。
    """
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    status = getattr(exc, "status_code", None)
    return status in _RETRYABLE_STATUS


def _retry_delay(exc: Exception, attempt: int, cfg=None) -> float:
    """计算第 attempt 次重试（从 0 开始）前的等待秒数。

    优先级：服务端 Retry-After 响应头 > 指数退避 + 抖动。

    cfg 缺省（None）→ 调用点回落到模块级 config，所以
    `monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_DELAY", ...)` 仍然生效。
    """
    cfg = config if cfg is None else cfg
    resp = getattr(exc, "response", None)
    retry_after = None
    if resp is not None:
        headers = getattr(resp, "headers", None)
        if headers is not None:
            retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(float(retry_after), _RETRY_MAX_WAIT)
        except (TypeError, ValueError):
            pass
    delay = cfg.LLM_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
    return min(delay, _RETRY_MAX_WAIT)


def _err_desc(exc: Exception) -> str:
    """给人看的错误简述：优先 status_code，否则用异常类名。"""
    status = getattr(exc, "status_code", None)
    if status is not None:
        return f"HTTP {status}"
    return type(exc).__name__


async def _chat_stream_impl(client, cfg, messages: list[dict], tools: list[dict] | None = None):
    """
    async 流式调用的**共享实现**：客户端与配置由调用方注入（§4.6 决策 A/B）。

    为什么要抽这一层：模块级 `chat_stream`（默认实例，老路径）与 `LLMClient`
    （per-session）必须**只有一份**重试/流式逻辑——两份实现必然会漂移，
    而漂移的那一份还照常能跑（最危险）。

    :param client: AsyncOpenAI 实例（哪个会话的客户端由调用方决定）
    :param cfg:    配置对象（哪个会话的模型 / 温度 / 重试参数）
    """
    kwargs = {
        "model": cfg.LLM_MODEL,
        "messages": messages,
        "temperature": cfg.LLM_TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    for attempt in range(cfg.LLM_MAX_RETRIES + 1):
        try:
            stream = await client.chat.completions.create(**kwargs)
            # anext 而非 next：async generator 必须用 await 取首片
            # 建连或等首字节失败 → 可安全重试
            first_chunk = await anext(stream)
        except StopAsyncIteration:
            return  # 空流（接口异常），无 chunk 可给
        except Exception as exc:
            if not _is_retryable(exc) or attempt >= cfg.LLM_MAX_RETRIES:
                raise
            wait = _retry_delay(exc, attempt, cfg)
            print(
                f"  ⚠️  API 请求失败（{_err_desc(exc)}），{wait:.1f}s 后重试"
                f"（第 {attempt + 1}/{cfg.LLM_MAX_RETRIES} 次）…",
                flush=True,
            )
            # asyncio.sleep 而非 time.sleep：不阻塞事件循环，其他协程可继续跑
            await asyncio.sleep(wait)
            continue

        # —— 以下在重试 try 外：首 chunk 已拿到，中途断流不再重试 ——
        # 模型身份以接口返回的 model 字段为准，不信模型自我介绍（note1 Q2）
        if hasattr(first_chunk, "model") and first_chunk.model:
            print(f"  [model] 本次请求实际模型：{first_chunk.model}")
        yield first_chunk
        async for chunk in stream:
            yield chunk
        return


class LLMClient:
    """**每会话一份**的模型客户端（§4.6 决策 A：env 驱动的配置要能实例化）。

    改造前 `client` 与 `chat_stream` 都是模块级的：key/model 在 import 期绑定，
    无法注入假 client，两个 Agent 也没法走不同厂商 / 不同模型。

    这里持有自己的 AsyncOpenAI 与自己的 config 引用——**不读任何全局**。
    """

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else _default_config
        # 每个实例一个客户端：AsyncOpenAI 自带连接池，跨会话共享连接池
        # 会把"两个会话的模型配置"搅在一起（base_url/key 是实例级的）
        self.client = AsyncOpenAI(
            api_key=self.config.LLM_API_KEY,
            base_url=self.config.LLM_BASE_URL,
            timeout=self.config.LLM_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def chat_stream(self, messages: list[dict], tools: list[dict] | None = None):
        """与模块级同名函数行为完全一致，只是用自己的 client / config。"""
        return _chat_stream_impl(self.client, self.config, messages, tools)


async def chat_stream(messages: list[dict], tools: list[dict] | None = None) -> AsyncIterator:
    """
    async 流式调用聊天补全接口。返回 async chunk 迭代器，由调用方消费并聚合。

    这是**默认实例**的入口（老路径 / CLI 不注入 client 时用）；per-session
    请用 LLMClient(...).chat_stream(...)。本函数只做转发：client / config
    都在**调用点**查模块全局，所以 test_llm_retry.py 那套
    `monkeypatch.setattr(llm.client ...)` 的 patch 通路原样有效。

    每个 chunk 的结构（OpenAI 流式协议，与 sync 版完全一致）：
      - chunk.choices[0].delta.content     文本增量（可能为 None）
      - chunk.choices[0].delta.tool_calls 工具调用增量（可能为 None）
      - chunk.usage                       只在最后一个 chunk（choices 为空时）

    重试边界（Q11 的关键约束）：
      - 作用域只覆盖"建连 + 首 chunk"，不含任何 yield
      - 首 chunk 一旦交给上层，文本可能已打印到屏幕；中途 SSE 断流再重发
        会导致重复输出、tool_calls 聚合错乱——所以 yield 在重试 try 块外
      - async 版同等约束：anext(stream) 取首 chunk 失败可安全重试；
        首 chunk 之后断流，异常直接向上抛，不重发

    :param messages: 对话历史，OpenAI 消息格式
    :param tools:    工具 schema 列表；传入后模型可以返回 tool_calls
    :return:         async chunk 迭代器（async for 消费）
    """
    async for chunk in _chat_stream_impl(client, config, messages, tools):
        yield chunk
