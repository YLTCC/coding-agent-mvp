"""
test_llm_retry.py —— 应用层重试机制回归测试（note1 Q11 + Q12 async 版）

不打真实 API：用 monkeypatch 替换 AsyncOpenAI 的 create 方法，
用带 status_code 属性的假异常模拟 429/5xx/401。

核心不变量（async 版保持与 sync 版一致）：
  1. 可重试错误（429/5xx/网络抖动）失败 N 次后成功 → 最终拿到 chunk
  2. 不可重试错误（401/400/403）→ 立即抛，create 只调一次，不浪费退避
  3. 重试次数用尽 → 抛出最后一次异常
  4. 首 chunk 已 yield 之后断流 → 不重试（重发会导致重复输出/聚合错乱）
  5. 退避：Retry-After 优先 > 指数退避 + 抖动，30s 封顶

async 版的差异点（Q12 引入，测试要覆盖）：
  - create 返回 async generator 而非 sync iterator
  - 首 chunk 用 anext 而非 next
  - 后续用 async for 而非 for
  - time.sleep 改 asyncio.sleep（不阻塞事件循环）
"""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from openai import APITimeoutError

from agent import llm


# ---------- 测试替身 ----------

class FakeAPIError(Exception):
    """模拟 openai APIStatusError：只需要 status_code/response 两个属性。"""

    def __init__(self, status_code: int, response=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = response


def _fake_chunk(model: str = "fake-model"):
    return SimpleNamespace(model=model, choices=[], usage=None)


def _fake_async_stream(chunks):
    """把 list 包装成 async generator，模拟 AsyncOpenAI 流式响应。"""
    async def gen():
        for c in chunks:
            yield c
    return gen()


@pytest.fixture
def no_sleep(monkeypatch):
    """替换 asyncio.sleep / 随机抖动，测试跑得快且退避可精确断言。"""
    slept: list[float] = []
    async def fake_sleep(s):
        slept.append(s)
    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0.0)
    return slept


def _patch_create(monkeypatch, side_effects: list):
    """把 AsyncOpenAI 的 create 换成按列表依次返回/抛出的 async 替身。

    side_effects 元素：
      - Exception 实例：create() 直接 raise（模拟建连/等首字节失败）
      - list[chunk]：create() 返回 async generator（正常路径）
      - callable：调用之得到结果（用于"首 chunk 成功后断流"场景）
    """
    calls = {"n": 0}

    async def fake_create(**kwargs):
        idx = calls["n"]
        calls["n"] += 1
        item = side_effects[min(idx, len(side_effects) - 1)]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item()  # 期望返回 async generator
        return _fake_async_stream(item)

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
    return calls


# ---------- 1. 重试分类谓词（纯函数，不需要 async）----------

class TestIsRetryable:
    def test_retryable_status(self):
        for status in (408, 409, 429, 500, 502, 503, 504):
            assert llm._is_retryable(FakeAPIError(status)) is True

    def test_non_retryable_status(self):
        for status in (400, 401, 403, 404, 422):
            assert llm._is_retryable(FakeAPIError(status)) is False

    def test_network_error_types(self):
        exc = APITimeoutError.__new__(APITimeoutError)
        assert llm._is_retryable(exc) is True

    def test_plain_exception_not_retryable(self):
        assert llm._is_retryable(ValueError("bad params")) is False


# ---------- 2. 退避策略（纯函数）----------

class TestRetryDelay:
    def test_exponential_backoff(self, no_sleep, monkeypatch):
        monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_DELAY", 1.0)
        assert llm._retry_delay(FakeAPIError(429), 0) == pytest.approx(1.0)
        assert llm._retry_delay(FakeAPIError(429), 1) == pytest.approx(2.0)
        assert llm._retry_delay(FakeAPIError(429), 2) == pytest.approx(4.0)

    def test_retry_after_header_priority(self):
        resp = SimpleNamespace(headers={"retry-after": "5"})
        assert llm._retry_delay(FakeAPIError(429, response=resp), 2) == pytest.approx(5.0)

    def test_retry_after_capped(self):
        resp = SimpleNamespace(headers={"retry-after": "999"})
        assert llm._retry_delay(FakeAPIError(429, response=resp), 0) == 30.0

    def test_exponential_capped(self, monkeypatch):
        monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0.0)
        assert llm._retry_delay(FakeAPIError(503), 10) == 30.0

    def test_invalid_retry_after_falls_back(self, monkeypatch):
        monkeypatch.setattr(llm.random, "uniform", lambda a, b: 0.0)
        resp = SimpleNamespace(headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        assert llm._retry_delay(FakeAPIError(429, response=resp), 0) == pytest.approx(1.0)


# ---------- 3. chat_stream 端到端重试行为（async）----------

class TestChatStreamRetry:
    @pytest.mark.asyncio
    async def test_fail_twice_then_success(self, monkeypatch, no_sleep):
        """429 失败两次，第三次成功：最终拿到 chunk，create 调 3 次，sleep 2 次"""
        good_chunks = [_fake_chunk(), _fake_chunk()]
        calls = _patch_create(monkeypatch, [
            FakeAPIError(429),
            FakeAPIError(503),
            good_chunks,
        ])

        chunks = []
        async for c in llm.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(c)

        assert calls["n"] == 3
        assert len(chunks) == 2
        assert len(no_sleep) == 2

    @pytest.mark.asyncio
    async def test_401_fails_immediately(self, monkeypatch, no_sleep):
        """密钥错误不可重试：create 只调 1 次，异常直接抛，不 sleep"""
        calls = _patch_create(monkeypatch, [FakeAPIError(401)])

        with pytest.raises(FakeAPIError) as exc_info:
            async for _ in llm.chat_stream([{"role": "user", "content": "hi"}]):
                pass

        assert exc_info.value.status_code == 401
        assert calls["n"] == 1
        assert no_sleep == []

    @pytest.mark.asyncio
    async def test_400_fails_immediately(self, monkeypatch, no_sleep):
        """请求非法不可重试"""
        calls = _patch_create(monkeypatch, [FakeAPIError(400)])

        with pytest.raises(FakeAPIError):
            async for _ in llm.chat_stream([{"role": "user", "content": "hi"}]):
                pass

        assert calls["n"] == 1
        assert no_sleep == []

    @pytest.mark.asyncio
    async def test_retries_exhausted(self, monkeypatch, no_sleep):
        """持续 429：重试次数用尽后抛出最后一次异常"""
        calls = _patch_create(monkeypatch, [FakeAPIError(429)])

        with pytest.raises(FakeAPIError) as exc_info:
            async for _ in llm.chat_stream([{"role": "user", "content": "hi"}]):
                pass

        assert exc_info.value.status_code == 429
        assert calls["n"] == llm.config.LLM_MAX_RETRIES + 1
        assert len(no_sleep) == llm.config.LLM_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_no_retry_after_first_chunk(self, monkeypatch, no_sleep):
        """首 chunk 已 yield 之后流中断：绝不重试（否则用户看到重复输出）"""

        def broken_stream():
            async def gen():
                yield _fake_chunk()
                raise FakeAPIError(500)
            return gen()

        calls = _patch_create(monkeypatch, [broken_stream])
        gen = llm.chat_stream([{"role": "user", "content": "hi"}])

        first = await gen.__anext__()
        assert first.model == "fake-model"

        with pytest.raises(FakeAPIError):
            async for _ in gen:
                pass

        assert calls["n"] == 1
        assert no_sleep == []

    @pytest.mark.asyncio
    async def test_empty_stream_returns_nothing(self, monkeypatch, no_sleep):
        """空流（anext 立即 StopAsyncIteration）：静默结束，不重试不报错"""
        calls = _patch_create(monkeypatch, [[]])

        chunks = []
        async for c in llm.chat_stream([{"role": "user", "content": "hi"}]):
            chunks.append(c)

        assert chunks == []
        assert calls["n"] == 1
        assert no_sleep == []
