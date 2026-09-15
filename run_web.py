"""run_web.py —— Web 模式启动入口（阶段三·b）

用法：
    python run_web.py

⚠️ **必须 `workers=1`**：活跃会话在**进程内**的 `registry: dict[sid, WebSession]`
里（决策 E 的既定取舍：单机内存态 + JSONL 落盘，不上数据库）。
多 worker 的后果是"建会话落在 A 进程、发消息打到 B 进程" → 404，
**而且不会报任何错**——正是本项目最忌讳的"不报错的错误结论"。
真要横向扩展，得先把 registry 挪到 Redis 之类的地方，那是另一张票。

⚠️ 默认只绑 127.0.0.1（决策 G）。改成 0.0.0.0 而**又没配 WEB_TOKEN**，
等于把你的 CMD 开放给全网：`run_command` 的白名单挡的是"危险命令形态"，
**挡不住"用 python 跑一个脚本"**。
"""
from __future__ import annotations

import os

import uvicorn

from agent.config import Config
from web.app import create_app


def main() -> None:
    cfg = Config()

    # ⚠️ 先把 app 建出来：`create_app` 会做决策 K 的启动断言，
    # 并**把 SESSION_DIR 挪到工作区之外**（位置那道锁）。所以下面打印的
    # 必须是**挪完之后**的值——先打印再建 app 会打印出一个已经不成立的路径。
    app = create_app(cfg)

    # 启动前把话说清楚（和 CLI 的 check_env 对称）
    if not cfg.LLM_API_KEY or cfg.LLM_API_KEY.startswith("sk-在这里"):
        print("❌ 未配置 LLM_API_KEY！请先填好 .env（Web 模式也要调模型）")
        raise SystemExit(1)
    print(f"模型：{cfg.LLM_MODEL}  接口：{cfg.LLM_BASE_URL}")
    print(f"工作区根（服务端派生 workspace 的根）：{cfg.WEB_WORKSPACE_ROOT}")
    print(f"会话日志目录：{cfg.SESSION_DIR}")
    if cfg.WEB_HOST not in ("127.0.0.1", "localhost"):
        if not cfg.WEB_TOKEN:
            print("🔴 危险：正在监听非本机地址，且没有配置 WEB_TOKEN。")
            print("   `run_command` 的白名单挡不住'用 python 跑一个脚本'，")
            print("   这等于把你的终端交给网络上的任何人。建议立刻在 .env 里设 WEB_TOKEN。")
        else:
            print(f"⚠️  监听 {cfg.WEB_HOST}（已启用 WEB_TOKEN 鉴权）")
    print(f"打开 http://{cfg.WEB_HOST}:{cfg.WEB_PORT}/ 使用\n")

    uvicorn.run(app, host=cfg.WEB_HOST, port=cfg.WEB_PORT, workers=1, log_level="info")


if __name__ == "__main__":
    main()
