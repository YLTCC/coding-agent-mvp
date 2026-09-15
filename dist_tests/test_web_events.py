"""test_web_events.py —— UI 事件 / SSE 序列化 / 有界队列（决策 I + J）

对应 webtodolist.md §6 的：
  W-T4（部分）SSE 契约：单行 JSON、`id:` 只给落盘事件
  W-T10（部分）断连背压：队列有界、分级丢、记账可读

跑法：python -m pytest test_web_events.py -q
"""
import json

from web.events import (
    DISK_EVENT_TYPES,
    UI_ONLY_TYPES,
    EventQueue,
    WebEvent,
    parse_last_event_id,
    record_to_event,
    sse_frame,
)


# ============================================================
# W-T4：SSE 契约
# ============================================================

def test_sse_frame_is_single_line_json():
    """data: 必须是**单行** JSON —— 审批 detail 里有换行，裸露会破坏分帧。"""
    event = WebEvent(kind="approval", fields={"detail": "第一行\n第二行"}, seq=3)
    frame = sse_frame(event)

    assert frame.endswith("\n\n")
    # 结构上是 3 行（event / id / data）+ 结尾空行 → 恰好 4 个换行。
    # 再多一个就说明 data 里有裸换行（那会直接切断 SSE 分帧）
    assert frame.count("\n") == 4, repr(frame)
    lines = frame.split("\n")
    assert lines[0] == "event: approval"
    assert lines[1] == "id: 3"
    assert lines[2].startswith("data: ")
    payload = json.loads(lines[2][len("data: "):])
    assert payload["detail"] == "第一行\n第二行"      # 换行用字面反斜杠 n 表达
    assert payload["type"] == "approval"


def test_sse_frame_omits_id_for_non_durable_events():
    """决策 I：**只有**落盘事件带 `id:`。

    负向条**必须**在断言里：只断言"`id:` == `seq`"的测试，对一个
    "给 delta 另起一个 web 计数器、同时塞进 `id:`"的**错误实现照样是绿的**
    —— 那会让两套序号静默互相顶替（`id: 7` 到底是落盘第 7 条还是第 7 个 delta？），
    而且 `Last-Event-ID` 会拿到没有意义的数。
    """
    delta = sse_frame(WebEvent(kind="delta", fields={"text": "x"}, seq=None))
    assert "id:" not in delta, "delta 不该带 id（它本来就不可续传）"
    assert delta.startswith("event: delta\n")
    assert json.loads(delta.split("data: ")[1].strip())["text"] == "x"

    for kind in ("tool_call", "replay", "lagged", "approval_request", "task_cost"):
        assert "id:" not in sse_frame(WebEvent(kind=kind, fields={}))


def test_event_type_vocabulary_is_disjoint():
    """两个事件类别的词汇表不能重叠 —— 否则"要不要写 id:"就成了看情况的猜测。"""
    assert DISK_EVENT_TYPES & UI_ONLY_TYPES == set()
    # delta 一定在"不可续传"那一边（决策 I 的核心事实）
    assert "delta" in UI_ONLY_TYPES
    assert "delta" not in DISK_EVENT_TYPES
    # 落盘记录类型必须与 session_log 的 KNOWN_TYPES 对齐
    from agent.session_log import KNOWN_TYPES

    assert DISK_EVENT_TYPES == set(KNOWN_TYPES)


def test_record_to_event_strips_envelope_and_keeps_seq():
    record = {"v": 1, "seq": 9, "ts": 123, "type": "tool", "session_id": "s",
              "content": "结果", "tool_call_id": "c1"}
    event = record_to_event(record)
    assert event.kind == "tool"
    assert event.seq == 9                      # ← SSE 的 id
    assert "seq" not in event.fields and "v" not in event.fields
    assert event.fields == {"content": "结果", "tool_call_id": "c1"}


def test_parse_last_event_id():
    assert parse_last_event_id(None) is None
    assert parse_last_event_id("") is None
    assert parse_last_event_id("7") == 7
    assert parse_last_event_id(" 12 ") == 12
    assert parse_last_event_id("abc") is None


# ============================================================
# W-T10：有界队列 + 分级丢 + 记账
# ============================================================

def test_queue_never_grows_without_bound():
    """无界队列 + 会消失的消费者 = OOM（决策 J 的失败形态）。"""
    q = EventQueue(maxsize=4)
    for i in range(100):
        q.push(WebEvent(kind="delta", fields={"text": str(i)}))
    assert q.qsize() == 4
    assert q.dropped_deltas == 96          # 丢了多少必须**可读**，绝不静默
    assert q.dropped_events == 0
    # 留下的是最新的 4 个（丢最旧，不是丢最新）
    assert [e.fields["text"] for e in q.snapshot()] == ["96", "97", "98", "99"]


def test_queue_drops_delta_before_durable_events():
    """落盘事件**不可丢**：队列满时先丢 delta。"""
    q = EventQueue(maxsize=4)
    q.push(WebEvent(kind="delta", fields={"text": "d0"}))
    q.push(WebEvent(kind="delta", fields={"text": "d1"}))
    q.push(WebEvent(kind="assistant", fields={"content": "定稿"}, seq=1))
    q.push(WebEvent(kind="task_end", fields={"iterations": 1}, seq=2))

    q.push(WebEvent(kind="delta", fields={"text": "新"}))
    kinds = [e.kind for e in q.snapshot()]
    assert q.dropped_deltas == 1
    assert q.dropped_events == 0
    assert "assistant" in kinds and "task_end" in kinds, "落盘事件被丢了"


def test_queue_counts_when_it_must_drop_a_durable_event():
    """实在没有 delta 可丢时也必须**记账**，而不是静默挤掉。"""
    q = EventQueue(maxsize=2)
    for i in range(5):
        q.push(WebEvent(kind="tool", fields={"content": str(i)}, seq=i))
    assert q.qsize() == 2
    assert q.dropped_events == 3
    assert [e.seq for e in q.snapshot()] == [3, 4]


async def test_queue_get_returns_in_order():
    q = EventQueue(maxsize=4)
    for i in range(3):
        q.push(WebEvent(kind="user", fields={"content": str(i)}, seq=i))
    got = [await q.get() for _ in range(3)]
    assert [e.seq for e in got] == [0, 1, 2]
    assert q.try_get() is None
