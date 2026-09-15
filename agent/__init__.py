# agent 包：一个手写的极简 Coding Agent
# 模块划分：
#   config.py —— 配置加载（API Key、模型、工作目录等）
#   tools.py  —— 工具层（读文件 / 写文件 / 列目录 / 执行命令），含沙箱与权限控制
#   llm.py    —— 大模型客户端（OpenAI 兼容接口）
#   agent.py  —— Agent 主循环（对话历史、工具调用、上下文管理）
