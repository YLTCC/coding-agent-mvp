"""test_web_approvals.py —— 审批桥（决策 C.1 / C.2 / C.3）[W-T3 / W-T3b]

这个文件对应的失败形态**有三个是"不报错的"**（决策 C.1 的表）：
  1. threadpool 线程里 `loop.create_future()` → 不抛异常，连 set_debug 也不报；
  2. 审批 POST 写成 `def` → `set_result` 走 `loop.call_soon()`（不写 self-pipe）
     → loop 空闲时**唤醒丢失**，await 侧一直不醒；
  3. `future` + `threading.Event` 双通道 → 超时后晚到的"允许"仍能 set_result
     → **静默改判**（UI 显示"已批准"，工具早已按拒绝返回）。
所以这里的断言必须**同时**钉住"对的写法"和"错的写法会红"。

跑法：python -m pytest test_web_approvals.py -q
"""
import asyncio
import threading
import time
import warnings

from web.session_registry import WebSession


def _make_session(cfg, *, attached=True) -> WebSession:
    """造一个只有"会话外壳"的 WebSession：审批桥只需要 loop / cfg / 队列。"""
    session = WebSession(
        sid="testsid",
        workspace=cfg.WEB_WORKSPACE_ROOT,
        cfg=cfg,
        loop=asyncio.get_running_loop(),
    )
    if attached:
        session.consumer_attached()
    return session


async def _wait_event(session: WebSession, kind: str, timeout: float = 3.0) -> dict:
    """等一条事件。

    ⚠️ 等 `approval_request` 时要**多等一步**：这条事件是 worker 线程先投的，
    而"填 `_pending`"的协程是随后才被排进 loop 的——两者之间隔着一个 loop 迭代。
    不等的话，测试会在"按钮已渲染、桥还没开始等"的窗口里点按钮，于是
    `resolve()` 找不到 pending。生产里人点不了这么快，测试里必须补这一步。
    """
    while True:
        event = await asyncio.wait_for(session.events.get(), timeout)
        if event.kind == kind:
            break
    if kind == "approval_request":
        deadline = time.perf_counter() + timeout
        while session.approvals.pending_count == 0:
            assert time.perf_counter() < deadline, "协程一直没进 _await_decision"
            await asyncio.sleep(0.002)
    return event.fields


# ============================================================
# W-T3：三态 + 边界
# ============================================================

async def test_W_T3_allow(web_cfg):
    session = _make_session(web_cfg)
    bridge = session.approvals

    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", "a.txt"))
    fields = await _wait_event(session, "approval_request")
    assert fields["action"] == "写入文件" and fields["detail"] == "a.txt"
    assert bridge.pending_count == 1

    # 路由层（async def）在 loop 线程里调 resolve
    assert bridge.resolve(fields["aid"], True) is True
    assert await asyncio.wait_for(task, 2.0) is True
    assert bridge.pending_count == 0, "结果落地后不该留下 pending"


async def test_W_T3_deny(web_cfg):
    session = _make_session(web_cfg)
    bridge = session.approvals

    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "执行命令", "echo hi"))
    fields = await _wait_event(session, "approval_request")
    assert bridge.resolve(fields["aid"], False) is True
    assert await asyncio.wait_for(task, 2.0) is False


async def test_W_T3_timeout_is_fail_closed(web_cfg):
    """等不到答案 → **默认拒绝**。放行 = 把沙箱交给网络抖动。"""
    web_cfg.WEB_APPROVAL_TIMEOUT = 0.2
    session = _make_session(web_cfg)
    bridge = session.approvals
    assert bridge.timeout == 0.2

    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", "a.txt"))
    await _wait_event(session, "approval_request")
    t0 = time.perf_counter()
    assert await asyncio.wait_for(task, 3.0) is False
    assert time.perf_counter() - t0 < 2.0


async def test_W_T3_late_click_does_not_change_conclusion(web_cfg):
    """超时已拒之后又点"允许" → **不能静默改判**（决策 C.1 第 3 条 / C.3）。

    路由层把 `resolve() is False` 转成 HTTP 409。这里断言桥的返回值，
    也就是 409 的依据。
    """
    web_cfg.WEB_APPROVAL_TIMEOUT = 0.2
    session = _make_session(web_cfg)
    bridge = session.approvals

    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", "a.txt"))
    fields = await _wait_event(session, "approval_request")
    assert await asyncio.wait_for(task, 3.0) is False        # 超时 → 拒绝

    # 晚到的"允许"：找不到 pending → False → 路由回 409
    assert bridge.resolve(fields["aid"], True) is False


async def test_W_T3_timeout_leaves_no_zombie_future(web_cfg):
    """`finally: pending.pop(...)` 不能省：否则待决表会一直涨，
    而且晚到的结果还能 set_result 成功。
    """
    web_cfg.WEB_APPROVAL_TIMEOUT = 0.2
    session = _make_session(web_cfg)
    bridge = session.approvals

    for i in range(3):
        task = asyncio.create_task(
            asyncio.to_thread(bridge.callback, "写入文件", f"f{i}.txt")
        )
        await _wait_event(session, "approval_request")
        assert await asyncio.wait_for(task, 3.0) is False

    assert bridge.pending_count == 0, "留下了僵尸 future"


async def test_W_T3_repeated_click_is_refused(web_cfg):
    session = _make_session(web_cfg)
    bridge = session.approvals
    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", "a.txt"))
    fields = await _wait_event(session, "approval_request")
    assert bridge.resolve(fields["aid"], True) is True
    assert bridge.resolve(fields["aid"], True) is False, "重复点击必须 409，不能改判"
    assert await asyncio.wait_for(task, 2.0) is True


def test_W_T3_closed_loop_returns_false_without_warning(web_cfg):
    """关服竞态：loop 已关 → 返回 False、**不向上抛**、也不留"协程未 await"的告警。

    不 `coro.close()` 的话日志会刷
    `RuntimeWarning: coroutine ... was never awaited`。
    """
    closed = asyncio.new_event_loop()
    closed.close()
    session = WebSession(sid="x", workspace=web_cfg.WEB_WORKSPACE_ROOT,
                         cfg=web_cfg, loop=closed)
    session.consumer_attached()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert session.approvals.callback("写入文件", "a.txt") is False
    leaked = [w for w in caught if "never awaited" in str(w.message)]
    assert not leaked, [str(w.message) for w in leaked]


async def test_W_T3_bridge_errors_fail_closed(web_cfg):
    """桥自身出任何问题都必须 fail-closed（返回 False），绝不能把 run_task 带崩。"""
    session = _make_session(web_cfg)
    bridge = session.approvals

    def boom(*_a, **_k):
        raise RuntimeError("模拟 run_coroutine_threadsafe 意外失败")

    original = asyncio.run_coroutine_threadsafe
    asyncio.run_coroutine_threadsafe = boom
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert bridge.callback("写入文件", "a.txt") is False
        leaked = [w for w in caught if "never awaited" in str(w.message)]
        assert not leaked, [str(w.message) for w in leaked]
    finally:
        asyncio.run_coroutine_threadsafe = original


# ============================================================
# W-T3b：线程约束
# ============================================================

async def test_W_T3b_worker_thread_is_not_loop_thread(web_cfg):
    """① 回调**确实**跑在非 loop 线程（这是全部麻烦的根源），
    且 future **不是**在那个线程里建的（决策 C.1 第 1 条）。
    """
    session = _make_session(web_cfg)
    bridge = session.approvals
    loop_tid = threading.get_ident()

    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", "a.txt"))
    fields = await _wait_event(session, "approval_request")
    bridge.resolve(fields["aid"], True)
    assert await asyncio.wait_for(task, 2.0) is True

    assert len(bridge.thread_ids["worker"]) == 1
    worker_tid = next(iter(bridge.thread_ids["worker"]))
    assert worker_tid != loop_tid, (
        "回调居然和 loop 同线程 —— 工具没在 threadpool 里跑，"
        "这个测试的前提变了，整座桥是否还需要得重新评估"
    )
    # future 建在 loop 线程里：换成"线程里 create_future"这条会变红
    assert bridge.thread_ids["loop"] == {loop_tid}
    assert session.loop_thread_id == loop_tid


async def test_W_T3b_wakeup_latency_under_100ms(web_cfg):
    """③ 单会话 + 零其它流量下点"允许"，唤醒延迟 < 100ms。

    这条钉的是决策 C.1 第 2 条：`set_result` 若走了跨线程的 `call_soon()`
    （比如审批 POST 被写成 `def`），**loop 空闲时唤醒会丢**——在真服务器里
    常被响应回程流量顺带捎醒（高负载自愈、低负载才卡），所以只能靠这条
    "无其它流量"的用例确定性复现。
    """
    session = _make_session(web_cfg)
    bridge = session.approvals

    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", "a.txt"))
    fields = await _wait_event(session, "approval_request")

    t0 = time.perf_counter()
    assert bridge.resolve(fields["aid"], True) is True
    assert await asyncio.wait_for(task, 2.0) is True
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"唤醒延迟 {elapsed * 1000:.1f}ms —— 疑似唤醒丢失"


async def test_W_T3b_resolve_runs_on_loop_thread(web_cfg):
    """② 的单元版：`resolve()` 在哪个线程被调，就要求哪个线程是 loop 线程。

    集成版（真服务器 + 真 POST）在 test_web_api.py，那里把审批 POST 改成 `def`
    就会让 `resolve_threads` 多出一个非 loop 线程 id → 变红。
    """
    session = _make_session(web_cfg)
    bridge = session.approvals
    loop_tid = threading.get_ident()

    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", "a.txt"))
    fields = await _wait_event(session, "approval_request")
    bridge.resolve(fields["aid"], True)
    assert await asyncio.wait_for(task, 2.0) is True

    assert bridge.resolve_threads == {loop_tid}


# ============================================================
# 决策 J：断连 = 立即拒绝，不进入等待
# ============================================================

async def test_detached_approval_is_refused_immediately(web_cfg):
    """没有消费者时**不进入等待**：否则 threadpool 线程白占满超时，
    用户看到的是"卡住 5 分钟然后失败"（决策 J / H）。
    """
    session = _make_session(web_cfg, attached=False)
    bridge = session.approvals

    t0 = time.perf_counter()
    assert bridge.callback("写入文件", "a.txt") is False     # 同步直调：不等
    assert time.perf_counter() - t0 < 0.1
    assert bridge.pending_count == 0

    # 拒绝的原因必须让用户**事后看得见**（磁盘上的 approval 行看不出"因为断连"）
    await asyncio.sleep(0.01)
    event = session.events.try_get()
    assert event is not None and event.kind == "approval_missed"
    assert event.fields["reason"] == "client_disconnected"


async def test_disconnect_wakes_pending_approval(web_cfg):
    """消费者消失 → 唤醒并**拒绝**所有待决审批（不留僵尸线程）。"""
    web_cfg.WEB_APPROVAL_TIMEOUT = 30.0
    session = _make_session(web_cfg)
    bridge = session.approvals

    task = asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", "a.txt"))
    fields = await _wait_event(session, "approval_request")
    assert bridge.pending_count == 1

    session.consumer_detached()                        # 浏览器关页

    assert await asyncio.wait_for(task, 2.0) is False   # 不等满 30s
    assert bridge.pending_count == 0
    # 断连之后晚到的点击同样不生效（不会改判）
    assert bridge.resolve(fields["aid"], True) is False


async def test_reject_all_returns_count(web_cfg):
    web_cfg.WEB_APPROVAL_TIMEOUT = 30.0
    session = _make_session(web_cfg)
    bridge = session.approvals

    tasks = [asyncio.create_task(asyncio.to_thread(bridge.callback, "写入文件", f"f{i}"))
             for i in range(3)]
    for _ in range(3):
        await _wait_event(session, "approval_request")
    assert bridge.pending_count == 3
    assert bridge.reject_all() == 3
    results = [await asyncio.wait_for(t, 2.0) for t in tasks]
    assert results == [False, False, False]
    assert bridge.pending_count == 0
