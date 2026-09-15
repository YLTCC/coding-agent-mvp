"""
config.py —— 配置加载（实例化版）

所有可变参数都从环境变量 / .env 文件读取，代码里不写死任何密钥。

§4.6 前置改造（step 1）：值不再在**类体**（import 期）求值，而是挪进 `__init__`。
  为什么值得改：类体求值把 env 冻死在"谁先 import agent.config 谁说了算"那一刻，
  于是测试结果会依赖 pytest 的收集顺序（note3 §7.3 那条隔离隐患的根因）；
  而且进程级全局没法 per-session。
  过渡态保留模块级 `config = Config()`：老代码 `from .config import config` 照旧可用。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（不存在也不报错）
load_dotenv()


class Config:
    """运行期配置（只读），集中管理，避免各处散落 os.getenv"""

    def __init__(self) -> None:
        # ---------- 模型相关 ----------
        # API Key：必填，没有就没法调模型
        self.LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
        # 接口地址：DeepSeek 官方地址；换成通义/Kimi 等时改这里
        self.LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip()
        # 模型名：DeepSeek 用 deepseek-chat
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat").strip()
        # 采样温度：0.2 偏低，让 Agent 输出更稳定、少自由发挥（写代码场景合适）
        self.LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

        # 单次请求超时（秒）。openai SDK 默认 600s 太离谱，挂起 10 分钟用户会以为程序死了
        self.LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
        # 可重试错误（429/5xx/网络抖动）的最大重试次数（不含首次请求）
        self.LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
        # 指数退避基数（秒）：实际等待 = base * 2^attempt + 抖动，封顶 30s
        self.LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))

        # ---------- Agent 行为相关 ----------
        # **默认**工作目录：不配置 AGENT_WORKSPACE 时用它。
        #
        # 名字里那个 "default" 是刻意的（§4.6 决策 G）：沙箱根是**会话身份**，
        # 由 ToolContext.workspace 承载；这里只是"没有会话级指定时"的种子值。
        # 若这里仍叫 WORKSPACE，任何一处漏改都会静默地拿全局值当会话沙箱，
        # 而"工具跑在 A 的目录、日志记着 B 的目录"这类错误不会报错——最危险。
        self.default_workspace: Path = Path(os.getenv("AGENT_WORKSPACE", os.getcwd())).resolve()

        # 单个任务最多循环多少轮（每轮 = 一次模型调用 + 可能的工具执行）
        # 防止模型陷入死循环无限烧 token
        self.MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "25"))

        # 工具返回结果保留的最大字符数，超出截断（控制上下文长度）
        self.MAX_TOOL_OUTPUT_CHARS: int = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "8000"))

        # 对话历史最多保留多少条消息（超出后丢弃最旧的，系统提示词永远保留）
        # 注意：这是"最后一道硬砍"的安全网，语义压缩由下面的 HISTORY_WINDOW_TURNS 负责。
        # role="user" 的原话在这道硬砍里也不丢弃（见 agent._build_context）
        self.MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

        # ---------- 上下文压缩（滑窗 + 滚动摘要）----------
        # 窗口大小，单位是"轮" = 一条 user 消息算一轮（注意不是"条消息"：
        # 一轮任务通常产生 4~10 条消息，所以 8 轮远比 20 条宽松，别把两个数字混着比）
        self.HISTORY_WINDOW_TURNS: int = int(os.getenv("HISTORY_WINDOW_TURNS", "8"))
        # 压缩滞后：滑出窗口的轮数攒够这么多才压缩一次。
        # 不设滞后会变成"滑出一轮压一轮"，每轮多一次 API 调用，比不压还贵
        self.HISTORY_COMPRESS_LAG: int = int(os.getenv("HISTORY_COMPRESS_LAG", "4"))
        # 累积摘要的字符预算上限，超出后保留**尾部**并截断（防止摘要自己膨胀成新的肿瘤）
        # 保留尾部不是头部：摘要里越靠后越接近当前状态（最新约束 / 下一步计划），
        # 砍头部丢的是早已过时的开场信息，砍尾部才是丢西瓜。见 _cap_summary
        self.SUMMARY_MAX_CHARS: int = int(os.getenv("SUMMARY_MAX_CHARS", "2000"))
        # 送进摘要器时，单条工具结果最多保留多少字符（工具输出常有上万字的文件内容，
        # 原样喂给摘要器既贵又会把真正重要的约束挤掉）
        self.SUMMARY_TOOL_RESULT_CHARS: int = int(os.getenv("SUMMARY_TOOL_RESULT_CHARS", "400"))

        # 执行 shell 命令的超时时间（秒），超时自动杀掉，防止命令挂死
        self.COMMAND_TIMEOUT: int = int(os.getenv("COMMAND_TIMEOUT", "120"))

        # 是否自动批准高危操作（写文件 / 执行命令）
        # false = 每次都要用户在终端输入 y 确认（安全，推荐学习阶段使用）
        self.AUTO_APPROVE: bool = os.getenv("AUTO_APPROVE", "false").lower() == "true"

        # ---------- 会话日志（JSONL，note3 的那套格式）----------
        # 是否把会话落盘成 append-only 的 JSONL（sessions/<session_id>.jsonl）
        self.SESSION_LOG_ENABLED: bool = os.getenv("SESSION_LOG_ENABLED", "true").lower() == "true"
        # 会话文件目录。相对路径按当前工作目录解析
        self.SESSION_DIR: Path = Path(os.getenv("AGENT_SESSION_DIR", "sessions")).resolve()

        # ---------- Web 层（阶段三·b）----------
        # ⚠️ 这里的**默认值**是 CLI 的语义。web 模式会在启动时调
        # `web.session_registry.prepare_web_config()` 把 SESSION_DIR 移出沙箱
        # （决策 K 第 1 条：审计目录待在 workspace 里 = 跨会话读别人对话 + 审计可篡改）。
        # 刻意**不在 Config 里**做这件事：那会把 CLI 的 sessions/ 也一起搬走，
        # 而 CLI 是单用户、且 W7 要求**诚实承认**它仍在 workspace 内。
        self.WEB_HOST: str = os.getenv("WEB_HOST", "127.0.0.1").strip()
        self.WEB_PORT: int = int(os.getenv("WEB_PORT", "8000"))
        # 门：Authorization: Bearer <token>（SSE 用 ?token=，EventSource 不能设 header）。
        # 留空 = 不鉴权（默认只绑 127.0.0.1，别把它改成 0.0.0.0）
        self.WEB_TOKEN: str = os.getenv("WEB_TOKEN", "").strip()
        # 工作区根：前端只能提交**工作区名字**，服务端 resolve 后校验它在此根之下（决策 D）。
        # 缺省与 CLI 的默认沙箱同一个值
        self.WEB_WORKSPACE_ROOT: Path = Path(
            os.getenv("WEB_WORKSPACE_ROOT") or os.getenv("AGENT_WORKSPACE") or os.getcwd()
        ).resolve()
        # 审批超时（秒）：等不到答案时 **fail-closed**（默认拒绝）。
        # 放行 = 把沙箱交给网络抖动
        self.WEB_APPROVAL_TIMEOUT: float = float(os.getenv("WEB_APPROVAL_TIMEOUT", "300"))
        # SSE 队列上限（决策 J）：消费者会消失，所以队列**必须**有界。
        # 256 条落盘事件 ≈ 一条磁盘记录一个事件，天然有界
        self.WEB_QUEUE_MAXSIZE: int = int(os.getenv("WEB_QUEUE_MAXSIZE", "256"))
        # 静态文件目录（前端）。相对路径按项目根解析
        self.WEB_STATIC_DIR: Path = Path(
            os.getenv("WEB_STATIC_DIR")
            or str(Path(__file__).resolve().parent.parent / "web" / "static")
        ).resolve()

    @classmethod
    def from_env(cls) -> "Config":
        """**每会话一份**配置的正式入口（组合根用）。

        与 `Config()` 等价，只是把"这是从环境现读的一份"写成代码里看得见的事实；
        per-session 场景请走这里，不要用下面的模块级 `config`。
        """
        return cls()


# 过渡态兼容：老代码 `from .config import config` 照旧可用。
# 它是**默认值**而非"唯一真值"——会话级身份（尤其是 workspace）请走 ToolContext。
config = Config()
