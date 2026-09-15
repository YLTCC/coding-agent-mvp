"""
main.py —— Coding Agent MVP 的命令行入口（async 版）

用法：
    1. 复制 .env.example 为 .env，填入 LLM_API_KEY
    2. pip install -r requirements.txt
    3. python main.py
    4. 在交互界面直接用中文给 Agent 派任务，输入 exit 退出

阶段三 async 改造（Q12）：
    - main() 改 async，用 asyncio.run 启动事件循环
    - agent.run_task 是 async，调用前加 await
    - 默认审批走 CLI input（不注入 callback），UX 与 sync 版完全一致
    - KeyboardInterrupt 在 asyncio 版有特殊处理：to_thread 的 input 不可中断，
      但流式 await anext 期间 Ctrl+C 能正常触发，长任务中断仍可用
"""
import asyncio
import sys
from pathlib import Path
from typing import Callable

from agent.agent import CodingAgent, build_system_prompt
from agent.config import config
from agent.llm import LLMClient
from agent.session_log import SessionLogger, config_snapshot, make_sink
from agent.tools import ToolContext


BANNER = r"""
====================================================
   手写 Coding Agent (MVP)  ——  LLM + 工具 + 循环
====================================================
"""

# 模块级 config 的别名：下面几个工厂函数的形参也叫 config，会遮蔽全局名
_config = config


def check_env() -> bool:
    """启动前检查必要配置，缺失时给出明确提示而不是让程序在调用时报错"""
    if not config.LLM_API_KEY or config.LLM_API_KEY.startswith("sk-在这里"):
        print("❌ 未配置 LLM_API_KEY！")
        print("   请复制 .env.example 为 .env，并填入你的 API Key 后重试。")
        return False
    print(f"模型：{config.LLM_MODEL}  接口：{config.LLM_BASE_URL}")
    print(f"工作目录（沙箱）：{config.default_workspace}")
    print(f"自动审批：{'开启（高危操作不再询问）' if config.AUTO_APPROVE else '关闭（写文件/执行命令前会询问）'}")
    return True


def open_session_log(workspace, *, config=None, out=None,
                     session_start_extra=None) -> SessionLogger | None:
    """按配置开一个会话日志。关掉时返回 None，Agent 行为与加日志前完全一致。

    workspace 必须显式传入**本会话**的沙箱根：日志里那句 session_start.workspace
    是"审计对账"用的，写全局默认值就等于日志说谎（§4.6 §5.2 第 3 类失败形态）。
    system_prompt 同理——它与沙箱根必须同源，否则重放出的 messages[0] 和运行态对不上。

    out：W1 的输出通道。None → print（CLI 逐字节不变）；web 传自己的实现。
    session_start_extra：追加进 session_start 的字段（web 记 shared/shared_with）。
    """
    cfg = config if config is not None else _config
    if not cfg.SESSION_LOG_ENABLED:
        return None
    logger = SessionLogger.open_session(
        cfg.SESSION_DIR,
        system_prompt=build_system_prompt(workspace, cfg.LLM_MODEL),
        workspace=str(workspace),
        model=cfg.LLM_MODEL,
        config=config_snapshot(cfg),
        extra=session_start_extra,
    )
    if out is None:
        print(f"会话日志：{logger.path}")
    else:
        out("log_path", path=str(logger.path))
    return logger


def build_session(
    workspace=None,
    approval_callback: "Callable[[str, str], bool] | None" = None,
    *,
    config=None,
    llm_client=None,
    out=None,
    event_sink=None,
    session_start_extra=None,
) -> "tuple[CodingAgent, SessionLogger | None]":
    """**组合根**：唯一 new 这些对象的地方（§4.6 §3）。

    会话身份（workspace / 审批 / 审计 / 日志）在这里一次装配好，按引用传给
    ToolContext；工具层不再从任何全局去"猜"自己属于哪个会话。

        Config ─┐
        LLMClient ─┤→ CodingAgent(config, llm, ctx, session_log)
        ToolContext ─┘
          └─ workspace（沙箱根，同时进提示词）/ approval_callback
             / event_sink（由 logger 包成）/ logger / out（输出通道）

    返回 (agent, logger)；logger 为 None 表示本次没开日志。

    Web 化新增的两个可选参数（缺省 None → CLI 行为**一字节不变**）：
      - `out`：W1 的输出通道，同时塞进 agent 与 ToolContext（单一出口）。
      - `event_sink`：审计事件外出通道。web 传自己的 multicast sink
        （落盘 + 推 SSE），缺省仍是 `make_sink(logger)`。
    """
    cfg = config if config is not None else _config
    ws = Path(workspace).resolve() if workspace is not None else cfg.default_workspace

    logger = open_session_log(ws, config=cfg, out=out,
                              session_start_extra=session_start_extra)
    if event_sink is None:
        # 审计事件外出通道：tools 只认一个 (type, fields) 回调，这里把 logger 包进去。
        # 包的是**本会话**的 logger —— 并发时事件才不会写进别人的文件。
        event_sink = make_sink(logger) if logger is not None else None
    ctx = ToolContext(
        workspace=ws,
        approval_callback=approval_callback,
        event_sink=event_sink,
        logger=logger,
        config=cfg,
        out=out,
    )
    agent = CodingAgent(
        config=cfg,
        llm=LLMClient(cfg) if llm_client is None else llm_client,
        ctx=ctx,
        session_log=logger,
        out=out,
    )
    return agent, logger


async def main_async() -> None:
    """async 主入口。input() 仍在主线程同步调用——asyncio 不要求所有 IO 都 async，
    只要求不阻塞事件循环；但 input() 默认是阻塞的。
    解决：input() 在没有运行中协程时阻塞主线程是 OK的，事件循环只在 run_task 期间活跃。
    这里用 input() 拉取用户输入、asyncio.run 内 await run_task 的方式分阶段使用线程。
    """
    # 组合根：CLI 这一个会话的身份在这里装配（不传 approval_callback → 走 CLI input 审批）
    agent, session_log = build_session()
    print("\n准备就绪！直接输入任务即可（输入 exit 退出，/clear 清空对话历史）\n")

    try:
        await _repl(agent)
    finally:
        # 退出前把队列排干再关：不丢已提交的行（崩溃才允许丢尾部，正常退出不该丢）
        if session_log is not None:
            session_log.close()


async def _repl(agent: CodingAgent) -> None:
    while True:
        # input() 在主线程同步调用，此刻没有运行中的 async 任务
        try:
            user_input = input("👤 你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出"):
            print("再见！")
            break
        if user_input == "/clear":
            agent.reset_conversation()   # 清空对话历史 + 压缩状态 + 统计
            print("（对话历史和统计已清空）\n")
            continue
        if user_input == "/stats":
            s = agent.stats
            total = s["prompt_tokens"] + s["completion_tokens"]
            hit = s["prompt_cache_hit_tokens"]
            miss = s["prompt_cache_miss_tokens"]
            hit_rate = (hit / s["prompt_tokens"] * 100) if s["prompt_tokens"] > 0 else 0
            print(f"\n📊 累计统计：任务 {s['tasks']} | 模型请求 {s['requests']} 次 | "
                  f"工具调用 {s['tool_calls']} 次")
            print(f"   tokens：prompt {s['prompt_tokens']} + completion "
                  f"{s['completion_tokens']} = {total}")
            print(f"   cache 命中率：{hit_rate:.1f}%（hit {hit} / miss {miss}）\n")
            continue

        try:
            # run_task 是 async，await 后事件循环跑流式 + 工具执行
            await agent.run_task(user_input)
        except KeyboardInterrupt:
            # 流式 await 期间 Ctrl+C 能触发；to_thread 的 input 不可中断但当前已退出 run_task
            print("\n⏹️  任务已中断，可以继续下达新任务。\n")
        except Exception as e:
            print(f"\n❌ 运行出错：{type(e).__name__}: {e}")
            print("（通常是网络/API 问题，检查网络和 Key 后可重试）\n")


def main() -> None:
    print(BANNER)
    if not check_env():
        sys.exit(1)
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
