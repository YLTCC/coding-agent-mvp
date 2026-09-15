"""approvals.py —— 审批的 sync→async 桥（决策 C，本项目架构含金量最高的一步）

**为什么需要一座桥**：工具在 threadpool 里跑
（`agent.py` 的 `asyncio.to_thread(dispatch_tool, ...)`），`_ask_approval` 是 sync，
它的回调在 `tools.py` 里被**同步直调** → **回调一定跑在 threadpool 线程里**。
而"等用户点按钮"必须发生在**主事件循环**里。两边要接上。

**桥为什么长这样**（决策 C.1 的三处跨线程误用，其中两处**不报错**）：

  | 错误写法                                   | 症状                                                 |
  |--------------------------------------------|------------------------------------------------------|
  | 在 threadpool 线程里 `loop.create_future()`| **不抛异常**，连 `set_debug(True)` 也不报 → 隐形       |
  | 审批 POST 写成 `def`（→ `set_result` 跨线程）| `set_result` 走 `loop.call_soon()`（**不写 self-pipe**）→ loop 空闲时**唤醒丢失**，await 侧一直不醒 |
  | `future` + `threading.Event` 双通道并存     | 两个真相源；超时后晚到的"允许"仍能 `set_result` 成功 → **静默改判** |

正确写法只有**一个原语**，就是文档保证线程安全的那个：
`asyncio.run_coroutine_threadsafe(coro, loop)` 返回 `concurrent.futures.Future`，
线程侧 `cf.result(timeout)` 就是"自带超时的阻塞等待"——**不需要自己造 threading.Event**。

  ```
  threadpool 线程（回调跑在这里）            主事件循环
    cf = run_coroutine_threadsafe(           async def _await_decision(aid, timeout):
          _await_decision(aid, T), loop)       fut = loop.create_future()  ← 在 loop 线程内建
    # ↑ 线程安全（文档保证）                    pending[aid] = fut
    granted = cf.result(T + 1.0)               try:
    # ↑ 自带超时的阻塞等待，不阻塞 loop            return await wait_for(fut, timeout)
                                               finally:
                                                 pending.pop(aid, None)   ← 不留僵尸 future

                         POST /approvals/{aid}（**必须 async def**）
                           → resolve(aid, allow)：fut.set_result(...)
                              找不到 / 已完成 → 返回 False → HTTP 409
  ```

**超时只能有一处真相**：loop 侧 `wait_for` 管超时，线程侧 `cf.result(T + 1.0)`
只是兜底（差 1 秒，保证线程不会比 loop 先放弃）。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .session_registry import WebSession

# 断连/关会话时给"被拒绝的审批"用的原因串
REASON_DISCONNECTED = "client_disconnected"
REASON_SESSION_CLOSED = "session_closed"


class ApprovalBridge:
    """一个会话一座桥。线程侧 `callback` 是注入给 ToolContext 的那个 sync 回调。"""

    def __init__(self, session: "WebSession") -> None:
        self.session = session
        self._loop = session.loop
        self.timeout: float = float(session.cfg.WEB_APPROVAL_TIMEOUT)
        # aid → 在 **loop 线程内**创建的 future。线程侧永远拿不到它。
        self._pending: dict[str, asyncio.Future] = {}
        # 观测用：两组线程 id（W-T3b 断言的就是这两个）
        #   "worker" = 回调实际跑的线程；"loop" = 建 future 的线程
        # 把 `run_coroutine_threadsafe` 换成"线程里 create_future"，后者会≠loop → 用例变红
        self.thread_ids: dict[str, set[int]] = {"worker": set(), "loop": set()}
        # 观测用：resolve() 被调用的线程（审批 POST 若写成 `def` 会≠loop → 用例变红）
        self.resolve_threads: set[int] = set()

    # ============================================================
    # 线程侧（**跑在 threadpool**）—— 注入给 ToolContext.approval_callback
    # ============================================================
    def callback(self, action: str, detail: str) -> bool:
        """sync 回调：把"等用户点按钮"转成"在 loop 里 async 等"。

        返回值语义与 CLI 的 `input()` 完全一致：True=允许，False=拒绝。
        """
        self.thread_ids["worker"].add(threading.get_ident())

        # 断连即拒（决策 J）：没有消费者时**不进入等待**，直接拒绝。
        # 若先等再发现没人看，用户会看到"卡住 5 分钟然后失败"，
        # 而且 threadpool 线程被白占——十几个会话就能把池子吃光。
        if self.session.detached:
            self.session.push_approval_missed(action, detail, REASON_DISCONNECTED)
            return False

        aid = uuid.uuid4().hex[:12]
        self.session.push_approval_request(aid, action, detail)

        coro = self._await_decision(aid, self.timeout)
        try:
            cf = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # loop 已关（关服竞态）。**必须显式 close 掉协程**，
            # 否则日志会刷 "RuntimeWarning: coroutine ... was never awaited"。
            # 绝不允许向上抛：那会把 run_task 带崩（决策 C.3）
            coro.close()
            return False

        try:
            # 兜底超时比 loop 侧多 1s：保证"先由 loop 侧判超时"，不会两边各写一个真相
            return bool(cf.result(self.timeout + 1.0))
        except concurrent.futures.TimeoutError:
            cf.cancel()
            return False
        except Exception:  # noqa: BLE001 —— 桥出任何问题都 fail-closed
            return False

    # ============================================================
    # loop 侧
    # ============================================================
    async def _await_decision(self, aid: str, timeout: float) -> bool:
        """在 **loop 线程**里等审批结果。"""
        self.thread_ids["loop"].add(threading.get_ident())
        fut = self._loop.create_future()      # ← 只在 loop 线程内建（决策 C.1 第 1 条）
        self._pending[aid] = fut
        try:
            return bool(await asyncio.wait_for(fut, timeout))
        except asyncio.TimeoutError:
            # fail-closed：等不到答案时放行 = 把沙箱交给网络抖动
            return False
        finally:
            # 不留僵尸 future：超时后若不清，`_pending` 会一直涨，
            # 而且晚到的"允许"还能 set_result 成功（静默改判）
            self._pending.pop(aid, None)

    def resolve(self, aid: str, allow: bool) -> bool:
        """落地一条审批结果。**必须由 `async def` 的路由调用**（决策 C.2）。

        写成 `def` 的话 FastAPI 会把它丢进 threadpool（`run_in_threadpool`），
        于是 `set_result` 又变成跨线程 → **复发"唤醒丢失"**。
        这个坑**从类型签名上看不出来**，所以 W-T3b 拿测试钉住它。

        返回 False = 找不到 / 已完成 → 路由回 **409**（晚到、重复点击）。
        """
        self.resolve_threads.add(threading.get_ident())
        fut = self._pending.get(aid)
        if fut is None or fut.done():
            return False
        fut.set_result(bool(allow))
        return True

    def reject_all(self, reason: str = REASON_DISCONNECTED) -> int:
        """唤醒并**拒绝**所有待决审批。返回处理了几条。

        用于：SSE 消费者消失 / 会话被关 / 关服。
        不能让 threadpool 线程永久占住（决策 C.3、J）。
        """
        n = 0
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_result(False)
                n += 1
        return n

    @property
    def pending_count(self) -> int:
        return len(self._pending)
