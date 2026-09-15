"""
tools.py —— 工具层（Agent 的"手"）

Agent 本身只会输出文字，真正能读文件、改文件、跑命令的是这里的工具函数。
每个工具包含两部分：
  1. schema：给大模型看的"工具说明书"（名字、用途、参数格式，遵循 OpenAI function calling JSON格式 规范）
  2. handler：本地真正执行逻辑的 Python 函数

安全设计（学习重点）：
  - 路径沙箱：所有文件操作都被限制在**本会话的沙箱根**内，禁止 ../ 越界
    （沙箱根来自 ToolContext.workspace，工具层只在 `_workspace(ctx)` 一处读它）
  - 人工审批：写文件 / 执行命令属于高危操作，默认要用户输入 y 确认
  - 只读免审：list_dir / read_file / grep / glob 为纯只读工具，不触发审批
  - 命令超时：shell 命令有超时限制，防止挂死
"""
import fnmatch
import locale
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import config

# ReDoS 防护二选一：装了第三方 regex 就能给【单次匹配】设超时；
# 没装则退化为 stdlib re + 行级 deadline（只能防累积超时，挡不住单条灾难性回溯）
try:
    import regex as _regex
    _HAS_REGEX_TIMEOUT = True
except ImportError:  # 保持零依赖可跑：降级路径不是错误，只是防护变弱
    _regex = re
    _HAS_REGEX_TIMEOUT = False


# ============================================================
# 共享常量
# ============================================================

# 遍历目录时统一跳过的噪音目录（缓存、版本控制、依赖、虚拟环境、IDE 配置、构建产物）。
# list_dir / grep / glob 复用同一份，避免各写一套、改漏一处。
# 统一存小写，匹配时也转小写：Windows 文件系统大小写不敏感，.VENV 和 .venv 是同一个目录
_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".cache",
    ".idea", ".vscode", "dist", "build", "output",
})

# 凭据/密钥类文件名模式（fnmatch 语法，按纯文件名匹配，模式已为小写）。
# 这些文件的内容绝不能进入模型上下文：grep/read_file 读到的文本会随对话发给第三方模型 API。
_SENSITIVE_FILE_PATTERNS: frozenset[str] = frozenset({
    ".env", ".env.*",                       # 环境变量文件（API Key 等）
    "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore", "*.jks",  # 证书/私钥/密钥库
    "id_rsa", "id_rsa.*", "id_dsa", "id_ecdsa", "id_ed25519",   # SSH 私钥
    ".netrc", ".npmrc", ".pypirc", ".htpasswd",                 # 各类含令牌的配置
})

# .env.* 中的模板文件只有占位符、不含真密钥，显式放行（.env.example 等）
_SENSITIVE_EXEMPT: frozenset[str] = frozenset({
    ".env.example", ".env.sample", ".env.template", ".env.dist",
})


# ============================================================
# 命令安全策略常量（run_command 的多层防御）
# ============================================================

# 危险动词：绝对拒绝，不询问用户（一旦执行不可逆 / 可绕过所有黑名单）
# - 破坏性：del/format/rmdir 一旦执行不可逆
# - 系统操作：shutdown/reg/sc 改系统状态
# - 子壳绕过：cmd/powershell/bash 让所有黑名单失效（cmd /c 任意命令）
# - 网络外传：curl/wget 可把沙箱内文件外送
_DANGEROUS_CMDS: frozenset[str] = frozenset({
    # 破坏性
    "del", "erase", "rmdir", "rd", "format", "mklink",
    # 系统操作
    "shutdown", "logoff", "taskkill", "sc", "wmic",
    "reg", "regedit", "net", "netsh",
    # 子壳绕过（让所有黑名单失效）
    "cmd", "powershell", "pwsh", "bash", "sh", "wsl",
    # 网络外传
    "curl", "wget", "ftp", "scp", "sftp",
    # 启动任意程序
    "start",
})

# 白名单：首 token 在此名单内时，仍要过层 3（flag + 路径校验）
# 不在两名单内的命令 → 走人工审批（让用户判断）
_ALLOWLIST: frozenset[str] = frozenset({
    "python", "py", "pytest", "py.test", "pip", "pip3", "poetry",
    "git", "node", "npm", "npx", "yarn", "pnpm",
    "dir", "type", "echo", "cd", "where", "findstr",
    "cat", "ls", "grep",  # Git Bash / WSL 里可能可用
})

# 危险 flag：即使首 token 在白名单，带这些 flag 也直接拒
# python -c / node -e / cmd /c 都可执行任意代码字符串，绕过文件层沙箱
_DANGEROUS_FLAGS: dict[str, frozenset[str]] = {
    "python":  frozenset({"-c", "-W", "-x", "-i"}),
    "py":      frozenset({"-c", "-W", "-x", "-i"}),
    "cmd":     frozenset({"/c", "/k", "/r"}),
    "node":    frozenset({"-e", "--eval", "-p", "--print"}),
}


def _is_sensitive_file(name: str) -> bool:
    """判断文件名是否属于凭据/密钥类（大小写不敏感，跨平台行为一致）"""
    low = name.lower()
    if low in _SENSITIVE_EXEMPT:
        return False
    return any(fnmatch.fnmatch(low, pat) for pat in _SENSITIVE_FILE_PATTERNS)


def _sensitive_denied(name: str, ctx: "ToolContext") -> str:
    """读/搜凭据文件被策略拒绝时，统一回喂给模型的明确提示（不能伪装成"文件不存在"）

    顺带记一条审计事件（source="policy"）：这类拒绝**根本没打扰用户**，
    与"用户点了拒绝"是不同的东西，审计上看不出区别就白记了（note3 §2.4）。
    这里是三个调用点的唯一漏斗，所以事件记在最里面。
    """
    _emit_event("approval", ctx=ctx, action="读取凭据文件", detail=name,
                granted=False, source="policy")
    return (f"错误：{name} 属于凭据/密钥类文件（如 .env、*.pem、id_rsa），"
            f"安全策略禁止读入对话上下文——其内容会被发送给模型 API，存在密钥外泄风险。"
            f"如需修改请让用户手动编辑该文件。")


def _is_protected_audit_path(path: "Path", ctx: "ToolContext") -> bool:
    """判断一个（已 resolve 的）路径是否落在**会话审计目录**之下。

    §决策 K 第 2 条。为什么不复用 `_is_sensitive_file`：
      - 那个函数是**纯文件名**匹配（6 个调用点全传 basename），写不出"只拦
        sessions 目录"的模式；落地只能变成 `*.jsonl`，于是**误伤用户自己的数据**
        （`train_data.jsonl` 不是审计日志），而且拒绝文案会**说谎**——回喂的是
        "属于凭据/密钥类文件（如 .env、*.pem、id_rsa）"，实际它根本不是。
      - 实测：3 行样例里"按名字判"误伤 2 行、理由全假；"按路径判"是
        True/False/False（真审计日志 / 用户数据 / 别项目的日志），恰好只拦该拦的；
        且**误配 SESSION_DIR 也照样拦得住**（判别式跟着配置走）——这正是
        名字型模式做不到的：它不知道日志在哪。

    判别式读 `_cfg(ctx).SESSION_DIR` —— **不是** `_workspace(ctx)`：
    审计目录按设计就在 workspace **之外**（web 下），用 workspace 判必错。

    必须在 `_safe_resolve` **之后**调用（此时已是 resolve 过的绝对路径，
    `is_relative_to` 才成立）。

    ⚠️ 纵深防御，两层都要：web 下"位置"（SESSION_DIR 出沙箱 + 启动断言）挡的是
    **配置错**；本守卫挡的是"配置对、但 agent 自己走到了"。更关键的是 **CLI 模式
    没有那层位置约束**（CLI 的 sessions/ 就在 workspace 里），本守卫是 CLI 下
    **唯一那道锁**。
    """
    session_dir = Path(_cfg(ctx).SESSION_DIR).resolve()
    try:
        return Path(path).resolve().is_relative_to(session_dir)
    except OSError:
        return False


def _audit_denied(path: "Path | str", ctx: "ToolContext") -> str:
    """访问审计日志被策略拒绝时的**说真话**回喂 + policy 审计。

    与 `_sensitive_denied` 同构（都是"拒绝要说清楚为什么"+ `source="policy"`），
    但文案据实写成"会话审计日志"。**绝不能套用凭据类那句**——那是撒谎，
    模型会据此推断"这个文件里有密钥"，从而做出完全错误的下一步。
    """
    session_dir = Path(_cfg(ctx).SESSION_DIR).resolve()
    _emit_event("approval", ctx=ctx, action="访问会话审计日志", detail=str(path),
                granted=False, source="policy")
    return (f"错误：{path} 属于会话审计日志（会话日志目录：{session_dir}），"
            f"安全策略禁止 Agent 读入或修改。它记录的是会话本身的审计轨迹："
            f"读进上下文会跨会话泄漏对话内容，改动会破坏“事后不可抵赖”。"
            f"如需查看历史，请让用户直接用编辑器打开该文件。")

# grep 单次最多返回多少条匹配，超出只提示剩余条数，避免撑爆上下文
_MAX_GREP_MATCHES: int = 100

# glob 单次最多返回多少个文件，超出只提示剩余个数（之前完全没有上限，大仓库会撑爆上下文）
_MAX_GLOB_HITS: int = 200

# list_dir 单次最多列出多少个条目，超出只提示剩余个数
# （它曾是唯一没过预算的工具：一个几万条的目录会被整份拼进上下文）
_MAX_DIR_ENTRIES: int = 200

# grep 单文件体积上限（字节）。超过则不读入内存，避免一个大文件把内存/上下文打满
_MAX_GREP_FILE_BYTES: int = 2_000_000

# grep 单行参与匹配/回显的最大字符数。压缩过的 js/json 常有几十万字符的单行，
# 截断后既保护上下文，也大幅降低正则灾难性回溯（ReDoS）的风险
_MAX_GREP_LINE_CHARS: int = 2000

# grep 整体时间预算（秒）。正则本身可能是回溯炸弹，用总时长兜底，
# 超预算就提前收工并告知模型"结果可能不完整"，而不是让整个 Agent 卡死
_GREP_TIME_BUDGET: float = 10.0

#grep给[统计头+不完整结果警告]预留的字符数
#必须永远可见：不能杯匹配行顶掉！
_GREP_RESERVED_CHARS: int = 256


# ============================================================
# 会话上下文（§4.6 前置改造：把"靠 import 全局隐式连起来"的依赖改成显式传参）
# ============================================================

# 模块级 config 的别名。ToolContext.__init__ 的形参也叫 config，会遮蔽全局名，
# 所以这里留一个不被遮蔽的引用供回落使用。
_default_config = config


class ToolContext:
    """一次会话的**身份**：workspace + 审批回调 + 审计 sink + logger + 配置引用。

    判断标准（§4.6 决策 A）：**同时跑两个 Agent 实例会串台的东西，就必须实例化。**
    于是这里只有两类东西：

      - 四样会话身份：workspace / approval_callback / event_sink / logger
        （A 的审批绝不能弹给 B，A 的审计绝不能写进 B 的日志）
      - config **引用**：只服务于"调参"（超时 / 预算 / 上限），
        调用点每次 `ctx.config.X` 现读——**绝不在构造期把标量抄进实例**。
        抄一份的后果是"会话中途改 env 不生效"，而 session_start 落盘的还是旧快照
        → 日志记的和实际跑的对不上（正好违背 note3 §2.1 立日志的初衷）。

    ⚠️ workspace 是**会话身份**不是调参旋钮（§4.6 §5.1）：它有三个必须同源的
    消费者（沙箱校验 / 结果相对化+cwd / 提示词+日志），所以**物理上不从 config 取**，
    只走 `ctx.workspace`；缺省时由 `_workspace()` 回落到模块级配置的
    `default_workspace`（§4.6 决策 G 改名的那个字段）。

    构造后不再改：要换目录 = 换会话 = 换 ctx（不提供 set_workspace —— 所有逃逸
    检查都隐含"沙箱根是个常量"这个假设，给个 setter 等于把假设打穿）。
    """

    def __init__(
        self,
        *,
        workspace: "str | Path | None" = None,
        approval_callback: "Callable[[str, str], bool] | None" = None,
        event_sink: "Callable[[str, dict], None] | None" = None,
        logger: Any = None,
        config: Any = None,
        out: "Callable[..., None] | None" = None,
    ) -> None:
        # 构造时归一化：逃逸检查拿解析后的绝对路径去比 is_relative_to，
        # 若 workspace 是相对路径 / 未 resolve，比较会**恒为假**——工具层整体哑掉，
        # 而且报错信息会误导（fail-closed，不致命但极难排查）
        self.workspace: Path | None = (
            Path(workspace).resolve() if workspace is not None else None
        )
        self.approval_callback = approval_callback
        self.event_sink = event_sink
        self.logger = logger
        self.config = config if config is not None else _default_config
        # W1 输出通道：agent 与 tools 共用**同一个**出口（决策 B 的"单一出口"）。
        # None → 直接 print（CLI）；web 注入实现后，审批提示也进 SSE。
        self.out = out


# **4.6 终态已落地**：`_default_ctx` 与 `set_approval_callback` / `set_event_sink`
# 已删除。工具层**不存在**任何进程级的会话状态——`ctx` 是必填参数，漏传即
# `TypeError`（"没有 ctx"从"静默串台"变成了**语法错误**，这正是决策 D 的终态）。
# 唯一剩下的全局是 `config.default_workspace`（决策 G 改名的那个**种子值**），
# 它只作为 `ToolContext.workspace is None` 时的回落目标，见 `_workspace()`。


def _cfg(ctx: "ToolContext"):
    """取"这次调用该用的配置对象"（只给调参用）。

    必须在**调用点**读（§4.6 决策 C）：构造期缓存会让 monkeypatch 静默失效，
    测试还是绿的，但已经不测任何东西了——这是本次改造最坏的失败模式。
    """
    return ctx.config if ctx.config is not None else _default_config


def _workspace(ctx: "ToolContext") -> Path:
    """工具层**唯一**读取沙箱根的地方（§4.6 §5.3：12 处读取收敛成 1 个 accessor）。

    ctx.workspace 为 None 时回落到模块级配置的 `default_workspace`（决策 C 的回落
    协议）：`ToolContext()` 这种"不带 workspace"的 ctx 拿到的是**默认工作目录**。
    回落同样发生在**调用点**，所以 `monkeypatch.setattr(cfg, "default_workspace", ...)`
    继续有效。
    """
    return ctx.workspace if ctx.workspace is not None else config.default_workspace


# ============================================================
# 安全辅助函数
# ============================================================

def _safe_resolve(rel_path: str, ctx: "ToolContext") -> Path:
    """
    把模型传入的（可能是相对的、可能带 ../ 的）路径解析成绝对路径，
    并校验它必须位于沙箱工作目录内。越界直接抛异常。

    ctx 必填：沙箱根是**会话身份**，没有"默认会话"这种东西（4.6 终态）。
    """
    root = _workspace(ctx)
    # 相对路径以沙箱根为基准解析；绝对路径直接取
    p = Path(rel_path) 
    #字符串包装为Path对象
    if not p.is_absolute():
        p = root / p
        #相对路径以沙箱根为基准解析；绝对路径直接取
    # resolve() 会展开 ../、符号链接等，得到真实绝对路径
    p = p.resolve()

    # 关键校验：解析后的路径必须在沙箱根之下
    # is_relative_to 是 Python 3.9+ 的方法
    if not p.is_relative_to(root):
        raise PermissionError(
            f"越界操作被拦截：{p} 不在工作目录 {root} 内"
        )
    return p


def _clip(text: str, limit: int) -> str:
    """
    按指定上限截断文本（内部实现）。

    截断点尽量退到换行处：按字符硬切会让模型看到"半行"，
    可能误以为文件内容本来就是那样，进而做错判断。
    """
    if len(text) <= limit:
        return text
    # 提示语本身也要占字符，所以真正能保留的正文要先把提示语的长度扣掉。
    # 否则"正文 + 提示语"会略超 limit，被 dispatch_tool 出口再截一次、
    # 提示语被改写成两份，长度和内容都会漂移（截断必须幂等）。
    total = len(text)
    cut = text[:limit]
    newline = cut.rfind("\n")
    if newline > limit // 2:  # 离行尾不远才退行，否则（如压缩成一行的 js）宁可硬切
        cut = cut[:newline]
    # 上面的退行让正文变短，提示语要报的"省略字符数"随之变化，
    # 所以再按"正文 + 提示语 ≤ limit"回缩一次，保证返回值本身不超预算。
    for _ in range(3):
        note = _cut_note(total, total - len(cut))
        if len(cut) + len(note) <= limit:
            break
        cut = cut[: max(0, limit - len(note))]
    # 告诉模型被砍掉了多少：它才知道这是不完整结果，可缩小范围重查
    return cut + _cut_note(total, total - len(cut))


def _cut_note(total_chars: int, omitted_chars: int) -> str:
    """截断提示语。抽成函数是为了能先算长度、再决定正文保留多少"""
    return (
        f"\n... [输出过长已截断：省略 {omitted_chars} 字符"
        f"（原文 {total_chars} 字符）]"
    )


def _truncate(text: str, ctx: "ToolContext") -> str:
    """按全局预算截断（工具返回值的统一出口，所有 handler 都过这里）。

    这是所有工具输出的统一预算闸门，任何工具都不应绕过
    config.MAX_TOOL_OUTPUT_CHARS；dispatch_tool 出口还会再兜一次底。

    ctx 必填。预算仍在**调用点**读（决策 C），
    所以 `monkeypatch.setattr(cfg, "MAX_TOOL_OUTPUT_CHARS", ...)` 仍然生效。
    """
    return _clip(text, _cfg(ctx).MAX_TOOL_OUTPUT_CHARS)


def _read_text_auto(path: Path) -> tuple[str, str]:
    """
    读取文本并自动识别编码，返回 (文本, 编码标签)。

    背景：固定 UTF-8 解码时，GBK 文件（中文 Windows 老文件、旧版记事本默认编码）
    会变成乱码，grep 搜中文还会静默返回"未找到"——不报错的错误结论最危险。

    尝试顺序（顺序不能换）：
      1. UTF-8（含 BOM）严格解码：现代文件主流，strict 能可靠识别"不是 UTF-8"
      2. GBK 回退：GBK 几乎能解码任意字节串，所以必须放在 UTF-8 之后
      3. UTF-8 + replace 兜底：破损文件也不崩，行为与旧实现一致
    前 8KB 含 NUL 字节的视为二进制（文本编码几乎不含 NUL），直接走兜底，
    避免 GBK 把随机二进制解成乱码"伪中文"。
    """
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        return raw.decode("utf-8", errors="replace"), "binary"
    try:
        return raw.decode("utf-8-sig"), "utf-8"  # utf-8-sig 同时兼容无 BOM 并剥掉 BOM
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("gbk"), "gbk"
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), "unknown"


def _decode_subprocess_output(raw: bytes) -> str:
    """
    把 run_command 捕获到的子进程输出字节解成文本（尝试顺序不能换）。

    为什么不能固定 `encoding="utf-8"`：同一条管道里**两种编码并存**，且没有任何
    字段声明是哪一种——

      - cmd.exe 的**内建**命令（`echo %CD%` / `dir` / `type`…）按**控制台代码页**
        输出，中文 Windows 上是 GBK(936)；
      - **python 子进程**受 env 里的 PYTHONIOENCODING=utf-8 影响，按 UTF-8 输出。

    固定 UTF-8 再 errors="replace" 会把 GBK 路径里的中文**静默**换成 `?`：模型
    看到的沙箱路径与真实路径不一致，却依然拿到"退出码 0（成功）"。路径里一旦有
    中文（中文用户名、中文目录名），`cd`/`dir`/测试脚本回显的路径全部失真，
    而模型会以为那是真的——不报错的错误结论最危险。

    顺序与 `_read_text_auto` 一致：UTF-8 严格 → 系统代码页 → UTF-8 兜底。
    UTF-8 必须放最前：它是最严格的（非法字节序列会抛异常），误判概率最低。
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        # 系统首选编码（中文 Windows = cp936/GBK，Linux 通常 = UTF-8）
        return raw.decode(locale.getpreferredencoding(False))
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")

# ============================================================
# 行尾（EOL）处理：读回一律 LF，写盘保留文件原有风格
#
# 为什么必须有这一层：Windows 上文本模式读写会做 newline 翻译。
#   - read_file 用文本模式回喂 CRLF 文件时，模型看到的是「行 CR LF」；
#     它按所见内容构造的 old_string 往往是 LF，于是
#     content.count(old_string) 恒为 0，edit_file 永远报"未找到匹配内容"。
#   - write_file 用文本模式写 LF 内容，磁盘上会变成 CRLF；而内容里本来
#     就有的 CRLF 会被再翻译一次，磁盘上出现「CR CR LF」，每读一遍写一遍多一层 CR。
# 解法：读进来先归一成 LF（并明确告知模型），写出去一律按二进制落盘，
# 编辑时再把内容还原成文件原有的行尾风格——既不破坏文件，也不让模型猜行尾。
# ============================================================

def _detect_eol(text: str) -> str:
    """
    判断文本的主要行尾风格，返回 "crlf" / "cr" / "lf"。

    混合行尾的文件按多数派判定：编辑保存时统一成该风格
    （并在回喂里说明），行为可预期，比逐行保留简单、也不会放大混乱。
    """
    crlf = text.count("\r\n")
    cr = text.count("\r") - crlf        # 减掉 CRLF 里的 CR，剩下的才是孤立 CR
    lone_lf = text.count("\n") - crlf   # 减掉 CRLF 里的 LF，剩下的才是裸 LF
    if crlf == 0 and cr == 0:
        return "lf"  # 没有 CR 系行尾（或只有裸 LF）：不引入 CR
    # 混合行尾按多数派判定，"统一化"时改动的行数最少
    if lone_lf > crlf + cr:
        return "lf"
    return "crlf" if crlf >= cr else "cr"


def _to_lf(text: str) -> str:
    """把 CRLF / 孤立 CR 统一成 LF（read_file 回喂、edit_file 匹配前都要走）"""
    if "\r" not in text:
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _from_lf(text: str, eol: str) -> str:
    """把 LF 文本还原成指定行尾风格（写盘前调用，保证编辑不改变文件风格）"""
    if eol == "crlf":
        return text.replace("\n", "\r\n")
    if eol == "cr":
        return text.replace("\n", "\r")
    return text


def _write_text_preserve(path: Path, text: str) -> None:
    """
    以二进制方式写文本，不做任何 newline 翻译（工具落盘的统一出口）。

    绝不能改用 path.write_text()：Windows 下文本模式会把 LF 翻成 CRLF，
    内容里已有的 CRLF 会被再翻一次变成「CR CR LF」，文件每转一圈就多一层 CR。
    行尾要不要 CRLF，由调用方通过 _from_lf 显式决定。
    """
    path.write_bytes(text.encode("utf-8"))

# 已决定：做真 SSE 流式输出（不做"假SSE"）。截断策略（按语义还是按上限）在实现时再定。

def _ask_approval(action: str, detail: str, ctx: "ToolContext") -> bool:
    """
    高危操作前的人工审批（Claude Code 也是这个思路）。
    AUTO_APPROVE=true 时跳过询问，直接放行。

    阶段三审批事件化（Q12）：审批策略由外部注入。
      - 优先级：AUTO_APPROVE → ctx.approval_callback → 默认 CLI input()
      - callback 放进 ToolContext，多个 Agent 并发时各自的审批只弹给自己
        （§4.6 决策 B）；4.6 终态后**只有**这一条路，没有进程级回调了

    保持 sync 签名是为了让 dispatch_tool / 工具执行层整体不动；
    agent.run_task 在 async 主循环里用 asyncio.to_thread 调用 dispatch_tool，
    自然把这个 sync 调用搬进 threadpool，不阻塞事件循环。
    """
    if _cfg(ctx).AUTO_APPROVE:
        _emit_event("approval", ctx=ctx, action=action, detail=detail,
                    granted=True, source="auto")
        return True
    # 优先用本会话注入的 callback（Web 路径 / per-session 路径走这条）
    if ctx.approval_callback is not None:
        granted = bool(ctx.approval_callback(action, detail))
        _emit_event("approval", ctx=ctx, action=action, detail=detail,
                    granted=granted, source="user")
        return granted
    # 默认 CLI 路径：input() 同步询问
    #
    # W1：这两行提示也走 ctx 提供的输出通道（web 下进 SSE）。
    # web 会话**一定**有 approval_callback（registry 装配时必给），所以下面
    # 的 input() 分支在 web 下走不到；万一走到了（ctx.out 有、callback 没有），
    # 绝不能退回 input()——那会在 uvicorn 的 threadpool 里**永久阻塞**一个线程，
    # 表现是"点了按钮没反应"且不报错。宁可 fail-closed 直接拒绝。
    if ctx.out is not None:
        ctx.out("approval_prompt", action=action, detail=detail)
    else:
        print(f"\n⚠️  Agent 请求{action}：")
        print(f"    {detail}")
    if ctx.out is not None:
        granted = False
    else:
        answer = input("    是否允许？[y/N] ").strip().lower()
        granted = answer in ("y", "yes")
    _emit_event("approval", ctx=ctx, action=action, detail=detail,
                granted=granted, source="user")
    return granted


# ============================================================
# 审批回调 / 审计 sink 的注入方式（4.6 终态后只剩一条路）
#
# 回调与 sink 都放在 **ToolContext** 上，随实例走：
#   - 组合根（main.build_session / web/session_registry）装配 ctx 时注入；
#   - 原来的 set_approval_callback / set_event_sink（写模块级默认 ctx）已删除。
# 这样"注入给谁"在签名上就读得出来，A 的审批绝不可能弹给 B。
# ============================================================


def _emit_event(rtype: str, ctx: "ToolContext", **fields) -> None:
    """投一条审计事件。

    审计失败**绝不能**影响工具执行：审批已经做出决定了，日志写不进去
    只是少一条记录，不该让用户的命令失败。所以这里吞掉异常。
    （与 I6 不冲突：I6 要求的是"读到损坏日志时必须报错"，不是"写不进去要报错"。）

    事件去哪个会话，完全由传入的 ctx 决定（4.6 终态：没有默认 ctx 了）。

    🔴 **sink 是用位置参数调的**：`sink(rtype, fields)` —— 不是 `sink(rtype, **fields)`。
       写错签名**不会在这里报错**（异常被下面那句 `except: pass` 吞掉），
       现象是"审计静默消失"：磁盘上没有 approval 行、web 的 SSE 上也收不到。
       web 层的 `WebSession.sink` 是同一形态，所以 W-T11 专门抓这一条。
    """
    sink = ctx.event_sink
    if sink is None:
        return
    try:
        sink(rtype, fields)
    except Exception:  # noqa: BLE001
        pass


# ============================================================
# 命令安全策略（run_command 的多层防御）
# note1 Q5/Q8 销账：run_command 用 shell=True 直通 cmd.exe，
# 文件层沙箱管不到 shell。这是 90 分解（黑白名单），
# 完备解需要 OS 级沙箱（AppContainer / bubblewrap）。
# ============================================================

# cmd 元字符：用于拆分组合命令（&& || | &）
# 注意：必须先处理引号，引号内的元字符不算分隔符
_CMD_SEPARATORS_DOUBLE: tuple[str, ...] = ("&&", "||")


def _split_command_chain(command: str) -> list[str]:
    """把组合命令按 && || | & 拆成段，引号内的元字符不分割。

    简化方案：用状态机扫描，跟踪引号（" 和 '）开关。
    完整 cmd tokenizer 过重，MVP 不需要。

    例如：'type a.txt && python b.py | grep x' → ['type a.txt', 'python b.py', 'grep x']
    """
    segments: list[str] = []
    buf: list[str] = []
    in_quote: str | None = None  # None / '"' / "'"
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        # 不在引号内
        if ch in ('"', "'"):
            in_quote = ch
            buf.append(ch)
            i += 1
            continue
        # 检查双字符分隔符（优先于单字符）
        pair = command[i:i + 2]
        if pair in _CMD_SEPARATORS_DOUBLE:
            if buf:
                segments.append("".join(buf))
                buf = []
            i += 2
            continue
        if ch in ("|", "&"):
            if buf:
                segments.append("".join(buf))
                buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


def _extract_first_token(segment: str) -> str:
    """取一段命令的首 token（命令名），小写化。

    用 shlex 切分；Windows 路径里反斜杠可能让 shlex 出错，做个兜底。
    去掉可能的引号和路径前缀，只留命令名（python.exe → python）。
    """
    try:
        parts = shlex.split(segment, posix=False)
    except ValueError:
        parts = segment.split()
    if not parts:
        return ""
    name = parts[0].strip('"').strip("'")
    # 取 basename（python.exe → python），统一用 / 切分再取最后一段
    name = name.replace("\\", "/").split("/")[-1]
    # 去扩展名
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return name.lower()


def _looks_like_path(arg: str) -> bool:
    """判断参数是否像文件路径（含 \\ / 或有扩展名），避免误伤普通参数。"""
    return "\\" in arg or "/" in arg or "." in arg


def _check_command_safety(command: str, ctx: "ToolContext") -> str | None:
    """
    校验命令安全性。返回 None 表示通过，返回字符串表示拒绝原因。

    三层防御：
      1. 元字符拆分：按 & | && || 切成段（引号内不切），每段独立校验，任一段拒 → 整条拒
      2. 首 token 判定：
         - 危险动词黑名单（del/format/cmd/powershell...）→ 直接拒
         - 白名单内（python/pytest/pip/git...）→ 进层 3
         - 不在两名单 → 走人工审批（让用户判断）
      3. 危险 flag + 路径参数校验：
         - python -c / cmd /c / node -e → 直接拒（可执行任意代码字符串）
         - 命令里出现的路径参数过 _safe_resolve + _is_sensitive_file

    已知局限（挂 note1 新 Q9）：
      - 引号内 -c payload 不解析（被 python -c 规则整条兜住，不是 payload 解析拦）
      - 重定向 > >> 未单独处理（默认当普通字符进 subprocess，可加 _safe_resolve 校验）
      - ^ 转义未处理（cmd 里 de^l 等于 del，建议含 ^ 的命令当可疑拒绝）
      - %VAR% 环境变量展开未处理（建议含 %...% 时拒绝或先展开）
      - OS 级沙箱才是 100 分解
    """
    # 层 1：拆分组合命令
    segments = _split_command_chain(command)
    if not segments:
        return "空命令"

    for seg in segments:
        first = _extract_first_token(seg)
        if not first:
            continue

        # 层 2a：危险动词 → 绝对拒，不询问用户
        if first in _DANGEROUS_CMDS:
            return (f"命令 {first!r} 属于危险命令（破坏性/子壳绕过/网络外传），"
                    f"安全策略禁止执行。如需该操作请用户手动执行。")

        # 层 2b：白名单内或带危险 flag 的命令 → 进层 3
        if first in _ALLOWLIST or first in _DANGEROUS_FLAGS:
            # 层 3a：危险 flag 检测
            dangerous = _DANGEROUS_FLAGS.get(first, frozenset())
            if dangerous:
                try:
                    tokens = shlex.split(seg, posix=False)
                except ValueError:
                    tokens = seg.split()
                for t in tokens[1:]:
                    flag = t.strip('"').strip("'").lower()
                    if flag in dangerous:
                        return (f"命令 {first!r} 带危险 flag {flag!r}"
                                f"（可执行任意代码/绕过沙箱），已拒绝。"
                                f"如需运行脚本请写成 .py 文件后用 python xxx.py 执行。")

            # 层 3b：路径参数校验
            try:
                tokens = shlex.split(seg, posix=False)
            except ValueError:
                tokens = seg.split()
            for t in tokens[1:]:
                arg = t.strip('"').strip("'")
                # 跳过 flag（-开头 / /开头在 Windows 也是 flag）
                if arg.startswith("-") or arg.startswith("/"):
                    continue
                if not _looks_like_path(arg):
                    continue
                # 凭据文件检测
                basename = arg.replace("\\", "/").split("/")[-1]
                if _is_sensitive_file(basename):
                    return _sensitive_denied(basename, ctx)
                # 沙箱边界检测：能 resolve 成路径的才查
                try:
                    resolved = _safe_resolve(arg, ctx)
                except PermissionError:
                    return (f"命令参数 {arg!r} 指向工作目录之外，已拒绝。"
                            f"Agent 只能操作工作目录内的文件。")
                # 审计日志：命令里也不许碰（如 `type sessions\xxx.jsonl`）
                if _is_protected_audit_path(resolved, ctx):
                    return _audit_denied(resolved, ctx)

    return None  # 全部通过


# ============================================================
# 工具的具体实现（handler）
# 约定：每个 handler 接收 (参数 dict, ctx)，返回字符串（模型只认文本）
#   - ctx **必填**（4.6 终态）：漏传即 TypeError，不存在"默认会话"
#   - handler 内部一切"环境相关"的东西（沙箱根 / 审批 / 事件 / 预算）
#     都必须从 ctx 取，**不许**直接读全局 config 上的沙箱字段
# ============================================================

def t_list_dir(args: dict[str, Any], ctx: "ToolContext") -> str:
    """列出目录内容（只读，安全）"""
    target = _safe_resolve(args.get("path", "."), ctx)
    # 审计目录连"存在性"都不暴露（决策 K 第 2 条）
    if _is_protected_audit_path(target, ctx):
        return _audit_denied(target, ctx)
    if not target.exists():
        return f"错误：目录不存在：{target}"
    if not target.is_dir():
        return f"错误：{target} 不是目录"

    # 算一次就够：每个条目再 resolve 一遍会在大目录上明显变慢
    session_dir = Path(_cfg(ctx).SESSION_DIR).resolve()
    lines = []
    extra = 0
    # iterdir 遍历一层；用 / 后缀标记目录，方便模型理解结构
    for item in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        # 噪音目录 / 凭据文件统一跳过（小写比较，兼容 Windows 大小写不敏感）
        if item.is_dir() and item.name.lower() in _SKIP_DIRS:
            continue
        if item.is_file() and _is_sensitive_file(item.name):
            continue
        # 审计目录本身也不列出来：列出来等于告诉模型"这里有个可以读的 jsonl"
        try:
            if item.resolve().is_relative_to(session_dir):
                continue
        except OSError:
            pass
        # 条目数封顶：大目录全量回喂同样会撑爆上下文
        if len(lines) >= _MAX_DIR_ENTRIES:
            extra += 1
            continue
        prefix = "📁 " if item.is_dir() else "📄 "
        lines.append(prefix + item.name + ("/" if item.is_dir() else ""))
    if not lines:
        return f"目录 {target} 为空"
    total = len(lines) + extra
    head = f"目录 {target} 的内容："
    if extra:
        head += f"（共 {total} 个条目，仅显示前 {len(lines)} 个，还有 {extra} 个未显示）"
    return _truncate(head + "\n" + "\n".join(lines), ctx)


def t_read_file(args: dict[str, Any], ctx: "ToolContext") -> str:
    """读取文件内容（只读，安全）。支持 start_line/end_line 分页读取大文件"""
    target = _safe_resolve(args["path"], ctx)
    # 审计日志：读/写都拒（决策 K 第 2 条）。放在 exists() 之前——fail-closed，
    # 连"这个文件存不存在"都不泄漏
    if _is_protected_audit_path(target, ctx):
        return _audit_denied(target, ctx)
    if not target.exists():
        return f"错误：文件不存在：{target}"
    if target.is_dir():
        return f"错误：{target} 是目录，请用 list_dir 查看"
    # 凭据文件显式拦截：必须回喂"被策略拒绝"，而不是读到内容或模糊地报"不存在"
    if _is_sensitive_file(target.name):
        return _sensitive_denied(target.name, ctx)
    try:
        # 自动识别 UTF-8/GBK：旧记事本/国内工具默认 GBK，固定 UTF-8 会乱码
        content, enc = _read_text_auto(target)
    except Exception as e:
        return f"读取失败：{e}"
    if not content.strip():
        return f"文件 {target} 为空"

    # 行尾统一成 LF 再回喂：模型按"所见内容"构造 old_string，看到 LF 才会写 LF。
    # 若回喂的是 CRLF，edit_file 里 content.count(old_string) 会恒为 0（永远"未找到"）
    eol = _detect_eol(content)
    content = _to_lf(content)

    lines = content.splitlines()
    total = len(lines)
    head = f"文件 {target}（共 {total} 行）"
    if enc == "gbk":
        # 明确告知编码：edit_file 保存会统一写成 UTF-8，模型需要预知这个转换
        head += "（检测到 GBK 编码，编辑保存后将转为 UTF-8）"
    if eol != "lf":
        # 明确告知行尾：回喂已归一为 LF，保存时保留原风格，模型不必手写 CR
        head += f"（检测到 {eol.upper()} 行尾，回喂内容已统一为 LF；保存时保留原行尾）"

    # 分页：两个参数都省略时读全文（向后兼容）
    start, end = args.get("start_line"), args.get("end_line")
    if start is None and end is None:
        return head + " 的内容：\n" + _truncate(content, ctx)

    # 只传一个参数时补齐另一个：start 缺省为 1，end 缺省为总行数（读到末尾）
    if start is not None and end is None:
        end = total
    if end is not None and start is None:
        start = 1

    # 边界校验：错误以字符串回喂，让模型自己修正参数
    if not isinstance(start, int) or not isinstance(end, int):
        return "错误：start_line / end_line 必须是整数（行号从 1 开始，含两端）"
    if start < 1 or end < 1:
        return "错误：start_line / end_line 必须 ≥ 1（行号从 1 开始）"
    if start > end:
        return f"错误：start_line({start}) 不能大于 end_line({end})"
    if start > total:
        return f"错误：start_line({start}) 超出总行数 {total}"

    # end 超总行数不报错：读「到末尾」是合理意图，收敛到最后一行即可
    end = min(end, total)
    body = "\n".join(lines[start - 1 : end])
    if end < total:
        body += f"\n（提示：第 {end + 1}-{total} 行未显示，可用 start_line={end + 1} 继续读取）"
    return head + f" 当前显示第 {start}-{end} 行：\n" + _truncate(body, ctx)


# ============================================================
# 遍历 / 模式匹配辅助（grep 与 glob 共用，保证两者语义一致）
# ============================================================

def _iter_files(root: Path, ctx: "ToolContext") -> Iterator[Path]:
    """
    遍历 root 下的所有文件，跳过 _SKIP_DIRS 里的噪音目录。

    关键点：跳过判断只看「相对 root 的路径片段」，而不是绝对路径片段。
    若直接拿绝对路径的 parts 去判断，一旦工作目录自身位于 .git / venv /
    node_modules 之下（例如 .../venv/project），所有文件都会被误判成噪音，
    grep / glob 会静默返回"未找到"，且不报任何错，极难排查。

    逃逸防线的沙箱根来自 `_workspace(ctx)`：per-session 下必须是**本会话**的根。

    审计目录（`_cfg(ctx).SESSION_DIR`）整棵子树跳过：grep/glob 连"存在性"都不该
    暴露（决策 K 第 2 条）。web 下它本就在沙箱外、沙箱检查已过滤掉，
    但 **CLI 下它就在沙箱里**，这里才是那道锁。
    """
    sandbox = _workspace(ctx)
    session_dir = Path(_cfg(ctx).SESSION_DIR).resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        kept = []
        for d in dirnames:
            if d.lower() in _SKIP_DIRS:
                continue
            # junction 防线：Windows 目录联接（mklink /J）的 islink() 返回 False，
            # followlinks=False 挡不住它，必须 resolve 展开真实路径后判边界
            real = (Path(dirpath) / d).resolve()
            if not real.is_relative_to(sandbox):
                continue
            if real.is_relative_to(session_dir):
                continue      # 审计目录整棵子树跳过（决策 K 第 2 条）
            kept.append(d)
        # 原地裁剪待遍历目录：噪音目录整棵子树直接跳过，比 rglob 后再逐文件过滤更高效
        dirnames[:] = sorted(kept)
        for name in filenames:
            # 凭据/密钥文件整类跳过：grep/glob 连"存在性"都不暴露，避免密钥进入模型上下文
            if _is_sensitive_file(name):
                continue
            f = Path(dirpath) / name
            # symlink 文件防线：链接在工作区内、目标在外，is_file() 仍为 True，
            # 读文本时会跟随链接——resolve 后判边界才能防住借道读沙箱外文件
            real = f.resolve()
            if not real.is_relative_to(sandbox):
                continue
            if real.is_relative_to(session_dir):
                continue      # 审计日志整类跳过（同上）
            yield f


def _rel_posix(path: Path, base: Path) -> str:
    """把 path 相对 base 的路径转成「正斜杠」形式，便于跨平台做 glob 匹配"""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:  # 不在 base 之下（如符号链接跳出去），返回空串表示"无法比较"
        return ""


def _glob_to_regex(pattern: str) -> re.Pattern:
    """
    把 glob 模式翻译成锚定正则，语义与 shell 一致（这正是 fnmatch 做不到的）。

    为什么不能用 fnmatch：fnmatch 的 `*` 会吃掉 `/`，导致 `agent/*.py`
    连 `agent/sub/deep.py` 也匹配，glob 语义被破坏。这里自己翻译：

      - `*`        只匹配单层内的任意字符，不跨 `/`
      - `**/`       匹配零级或多级目录（`**/*.py` 因此能命中根目录下的 .py）
      - `**`        匹配任意字符（含 `/`）
      - `?`         匹配除 `/` 外的单个字符
      - `[abc]` / `[!abc]`  字符集

    顺带把 `\\` 归一化成 `/`：Windows 上模型常会照抄返回值里的 `agent\tools.py`，
    不归一化则必然失配、反复试错。
    """
    pat = pattern.replace("\\", "/")
    out: list[str] = []
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            j = i
            while j < n and pat[j] == "*":
                j += 1
            if j - i >= 2:
                # `**/` 允许"零级目录"，这样 **/*.py 能同时匹配 a.py 和 x/a.py
                if j < n and pat[j] == "/":
                    out.append("(?:.*/)?")
                    j += 1
                else:
                    out.append(".*")
            else:
                out.append("[^/]*")
            i = j
            continue
        if c == "?":
            out.append("[^/]")
            i += 1
            continue
        if c == "[":
            j = i + 1
            if j < n and pat[j] in "!^":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j >= n:  # 没有闭合的 ]，退化成普通字符
                out.append(re.escape(c))
                i += 1
                continue
            inner = pat[i + 1 : j]
            if inner.startswith("!"):
                inner = "^" + inner[1:]
            out.append("[" + inner + "]")
            i = j + 1
            continue
        out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def _count_re_matches(rx, probe: str, deadline: float) -> int | None:
    """
    数一行文本里的正则命中次数。

    返回 None 表示"搜索时间预算已耗尽"（调用方应据此收工）：
      - 有 regex 库：每次调用都传【剩余预算】，单次灾难性回溯也会被引擎中断，
        超时抛 TimeoutError（它是内置 TimeoutError 的子类）
      - 无 regex 库：调用前用 monotonic 检查（第一层），只能保证行与行之间有界，
        挡不住正在执行的单次 C 层匹配
    """
    if _HAS_REGEX_TIMEOUT:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            return sum(1 for _ in rx.finditer(probe, timeout=remaining))
        except TimeoutError:
            return None
    # 降级路径：stdlib re，行级粒度检查
    if time.monotonic() > deadline:
        return None
    return sum(1 for _ in rx.finditer(probe))


def t_grep(args: dict[str, Any], ctx: "ToolContext") -> str:
    """
    按正则表达式搜索文件内容（只读，安全，无需审批）。
    返回 相对路径:行号: 该行内容，方便模型定位代码后精准 read_file / edit_file。

    path 既可以是目录（递归搜索），也可以是单个文件（只搜这一个文件）。
    命中行保留原始缩进（不被 strip 破坏），单行过长时截断后再回显。
    """
    pattern = args.get("pattern", "")
    if not pattern:
        return "错误：pattern 不能为空"
    # 正则先编译：坏正则应作为"可回喂的错误"返回，而不是让整个 Agent 循环崩掉
    # 走 _regex：装了 regex 库时编译出的 Pattern 才支持 finditer(timeout=)
    try:
        regex = _regex.compile(pattern)
    except _regex.error as e:
        return f"错误：正则表达式无效：{pattern!r}（{e}）"

    root = _safe_resolve(args.get("path", "."), ctx)
    # 审计日志：单文件直连也堵死（目录递归路径已由 _iter_files 过滤）
    if _is_protected_audit_path(root, ctx):
        return _audit_denied(root, ctx)
    if not root.exists():
        return f"错误：路径不存在：{root}"
    # 显式指定凭据文件时直接拒绝（目录递归路径已由 _iter_files 过滤，这里堵单文件直连）
    if root.is_file() and _is_sensitive_file(root.name):
        return _sensitive_denied(root.name, ctx)

    # 单文件时只搜这一个文件（模型很自然会这么用，之前会直接报"不是目录"）
    files = [root] if root.is_file() else _iter_files(root, ctx)

    include = args.get("include")  # 可选：按文件名过滤，如 *.py
    matches: list[str] = []
    total = 0        # 命中行数（用于"共 N 行"）
    total_occ = 0    # 命中次数（同一行出现两次算两次，用于"共 N 处"）
    scanned = 0
    skipped_large = 0
    hit_budget = False
    used = 0  # 已收集匹配行占用的字符数
    # 匹配行只能花「总预算 - 头部/警告预留」，保证返回不超过 MAX_TOOL_OUTPUT_CHARS
    line_budget = _cfg(ctx).MAX_TOOL_OUTPUT_CHARS - _GREP_RESERVED_CHARS
    deadline = time.monotonic() + _GREP_TIME_BUDGET
    sandbox = _workspace(ctx)
    for file in files:
        if not file.is_file():
            continue
        if include and not fnmatch.fnmatch(file.name, include):
            continue
        try:
            if file.stat().st_size > _MAX_GREP_FILE_BYTES:
                skipped_large += 1
                continue
            # 自动识别编码：否则 GBK 文件搜中文会静默零结果（最危险的错误形式）
            text, _enc = _read_text_auto(file)
            if _enc == "binary":
                continue  # 含 NUL 字节的二进制文件：解码出来全是乱码，搜了只产出垃圾命中
        except Exception:
            continue  # 无权限/读不到的文件跳过即可，不打断搜索
        scanned += 1
        rel = file.relative_to(sandbox)
        for lineno, line in enumerate(text.splitlines(), 1):
            # 超长单行只取前 N 字符做匹配：既保护上下文，也降低 ReDoS 风险
            probe = line if len(line) <= _MAX_GREP_LINE_CHARS else line[:_MAX_GREP_LINE_CHARS]
            occ = _count_re_matches(regex, probe, deadline)
            if occ is None:  # 时间预算耗尽（含单次匹配被 regex 库中断）：停止本行及后续
                hit_budget = True
                break
            if occ:
                total += 1
                total_occ += occ
                # 只收集前 N 行，但 total / total_occ 继续累加，便于告知还有多少没显示
                if len(matches) < _MAX_GREP_MATCHES:
                    # 用 rstrip 而不是 strip：行首缩进是代码语义的一部分，不能吃掉
                    shown = line.rstrip()
                    if len(shown) > _MAX_GREP_LINE_CHARS:
                        shown = shown[:_MAX_GREP_LINE_CHARS] + "…"
                    entry =f"{rel}:{lineno}: {shown}"
                #条数闸+字符预算闸：任一满就只计数不收集
                    if used+len(entry)+1<=line_budget:
                       matches.append(entry)
                       used+=len(entry)+1 #+1是因为每个匹配行之间有换行符
        # 收工条件：内层单次匹配超时已置位，或文件粒度的 deadline 到期
        if hit_budget or time.monotonic() > deadline:
            break

    if total == 0:
        # 超时且零命中时不能只说"未找到"——搜索没跑完，这是不完整结果，必须告知模型
        if hit_budget:
            return (f"未找到匹配 {pattern!r} 的内容"
                    f"（注意：搜索已用满 {_GREP_TIME_BUDGET:g} 秒时间预算，结果可能不完整，"
                    f"请收窄 path/include 或简化正则后重试）")
        if skipped_large:
            return f"未找到匹配 {pattern!r} 的内容（另有 {skipped_large} 个体积过大的文件被跳过）"
        return f"未找到匹配 {pattern!r} 的内容"

    if total > len(matches):
        head = (
            f"共 {total} 行匹配（{total_occ} 处，已扫描 {scanned} 个文件），"
            f"仅显示前 {len(matches)} 行："
        )
    else:
        head = f"共 {total} 行匹配（{total_occ} 处，已扫描 {scanned} 个文件）："
    lines_out = [head]
    if skipped_large:
        lines_out.append(
            f"... [另有 {skipped_large} 个体积超过 {_MAX_GREP_FILE_BYTES} 字节的文件被跳过]"
        )
    if hit_budget:
        lines_out.append(f"... [搜索已用满 {_GREP_TIME_BUDGET:g} 秒时间预算，结果可能不完整]")
    lines_out.extend(matches)
    return _truncate("\n".join(lines_out), ctx)


def t_glob(args: dict[str, Any], ctx: "ToolContext") -> str:
    """
    按文件名模式查找文件路径（只读，安全，无需审批）。

    pattern 语义与 shell 一致（这正是 fnmatch 做不到的，它的 `*` 会跨目录）：
      - `*.py`        只匹配搜索起点【顶层】的 .py
      - `**/*.py`     递归匹配所有层级（含顶层）
      - `agent/*.py`  只匹配 agent 下一层的 .py
    pattern 默认相对搜索起点 path 匹配，同时兜底支持相对沙箱根的完整写法
    （这样模型把上一次 glob 返回的 agent/tools.py 直接当 pattern 传进来也能命中）。
    """
    pattern = args.get("pattern", "")
    if not pattern:
        return "错误：pattern 不能为空"

    root = _safe_resolve(args.get("path", "."), ctx)
    # 审计日志：单文件直连堵死（目录递归由 _iter_files 过滤）
    if _is_protected_audit_path(root, ctx):
        return _audit_denied(root, ctx)
    if not root.exists():
        return f"错误：路径不存在：{root}"

    regex = _glob_to_regex(pattern)
    sandbox = _workspace(ctx)

    def hit(file: Path) -> bool:
        # 主语义：相对搜索根 path；兜底：相对沙箱根的完整相对路径
        return bool(regex.match(_rel_posix(file, root))) or bool(
            regex.match(_rel_posix(file, sandbox))
        )

    if root.is_file():
        # 凭据文件连存在性都不暴露（目录递归已由 _iter_files 过滤，这里堵单文件直连）
        if _is_sensitive_file(root.name):
            return f"未找到匹配 {pattern!r} 的文件（凭据/密钥类文件按安全策略不列出）"
        # 单文件：候选只有一个，额外允许按文件名匹配，不会造成误命中
        files = [root] if (regex.match(root.name) or hit(root)) else []
    else:
        files = [f for f in _iter_files(root, ctx) if hit(f)]

    hits = sorted(str(f.relative_to(sandbox)) for f in files)
    if not hits:
        return f"未找到匹配 {pattern!r} 的文件"

    # 结果必须封顶：大仓库里 `**/*` 可能列出几万个路径，全量回喂会直接撑爆上下文
    total = len(hits)
    if total > _MAX_GLOB_HITS:
        head = f"共 {total} 个文件，仅显示前 {_MAX_GLOB_HITS} 个（请缩小 pattern 或 path 范围）："
    else:
        head = f"共 {total} 个文件："
    return _truncate(head + "\n" + "\n".join(hits[:_MAX_GLOB_HITS]), ctx)


def t_write_file(args: dict[str, Any], ctx: "ToolContext") -> str:
    """写入/覆盖文件（高危，需审批）"""
    target = _safe_resolve(args["path"], ctx)
    # 审计日志：**在 _ask_approval 之前**就拒（决策 K 第 2 条）。
    # 否则用户要为一次必然失败的写点一次"允许"——既浪费注意力，也让审批
    # 窗口里出现"点了允许但什么都没发生"这种会让用户怀疑自己操作的现象。
    if _is_protected_audit_path(target, ctx):
        return _audit_denied(target, ctx)
    content = args.get("content", "")
    if not isinstance(content, str):
        content = str(content)  # 模型偶尔传数字/None，统一成字符串再落盘

    # 审批：告诉用户要写哪个文件、多少字节
    if not _ask_approval("写入文件", f"{target}（{len(content)} 字符）", ctx):
        return "用户拒绝了该写入操作，文件未修改。请调整方案或询问用户。"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)  # 父目录不存在则自动创建
        # 二进制写：文本模式会把 LF 翻译成 CRLF，内容里已有的 CRLF 会被再翻一次，
        # 磁盘上出现「CR CR LF」——文件每轮读写多一层 CR（曾把 agent.py 写坏）
        _write_text_preserve(target, content)
        return f"已成功写入 {target}（{len(content)} 字符）"
    except Exception as e:
        return f"写入失败：{e}"

def t_edit_file(args:dict[str,Any], ctx: "ToolContext") -> str:
    """精确替换文件中的一段内容（高危，需要用户审批）。
    与write_file不同，write_file是直接覆盖文件内容，而edit_file是精确替换文件中的一段内容。
    且old_string必须匹配一致，从机制上避免“改错位置”和“漏抄代码”
    """
    target =_safe_resolve(args["path"], ctx)
    # 审计日志：审批之前就拒（与 t_write_file 同理，决策 K 第 2 条）
    if _is_protected_audit_path(target, ctx):
        return _audit_denied(target, ctx)
    old_string = args.get("old_string","")
    new_string = args.get("new_string","")
    #前置校验：错误全部以字符串形式返回，完成回喂模型
    if not target.exists():
        return f"错误：文件不存在：{target}，这是一个新文件，用write_file创建"
    if target.is_dir():
        return f"错误：{target} 是目录,不可编辑，请用list_dir查看目录内容"
    if old_string == "":
        return f"错误：old_string不能为空"
    # 自动识别编码：GBK 文件按 GBK 读出才能逐字匹配（保存统一写 UTF-8）
    content, src_enc = _read_text_auto(target)
    # 行尾归一化：content 与 old_string/new_string 都折成 LF 再匹配。
    # read_file 回喂的就是 LF，模型按所见内容构造的 old_string 必然对得上；
    # 否则 CRLF 文件里 count(old_string) 恒为 0，只能反复报"未找到匹配内容"。
    eol = _detect_eol(content)
    content = _to_lf(content)
    old_string = _to_lf(old_string)
    new_string = _to_lf(new_string)
    match_count=content.count(old_string)
    if match_count == 0:
        #    未找到匹配的字符串：{old_string}"
        hint=" "
        if old_string.strip() and old_string.strip() in content:
            hint="（提示：去掉首尾空白后能匹配上，疑似缩进/空行不一致，请逐字复制 read_file 中的内容）"
        return f"错误：未找到匹配内容，文件可能已被修改，请重新read_file查看后再试{hint}"
    if match_count > 1:
        return f"错误：文件中存在{match_count}个匹配内容，无法精确替换，请检查后重试"
    #唯一性校验，进入审批，仅仅预览改动片段，不回显全文
    preview_old=old_string[:80].replace("\n","\\n")
    preview_new=new_string[:80].replace("\n","\\n")
    if not _ask_approval(
        "精确编辑",
        f"{target}\n 替换：{preview_old}\n 修改为了： {preview_new}",
        ctx,
    ):
        return "用户拒绝了本次编辑操作，文件未修改"
    #count=1,replace最安全——只动了唯一匹配处
    new_content=content.replace(old_string,new_string,1)
    try:
        # 二进制写 + 显式还原原行尾：文本模式在 Windows 上会把 LF 再翻成 CRLF
        _write_text_preserve(target, _from_lf(new_content, eol))
        note = "（文件已由 GBK 转为 UTF-8 编码）" if src_enc == "gbk" else ""
        if eol != "lf":
            note += f"（保留原 {eol.upper()} 行尾）"
        return f"已成功编辑 {target}:1处替换成功{note}"
    except Exception as e:
        return f"编辑失败：{e}"

def t_run_command(args: dict[str, Any], ctx: "ToolContext") -> str:
    """执行 shell 命令（高危，需审批 + 命令安全策略 + 超时限制）"""
    command = args["command"]
    # 审批/审计里的 detail 必须**自证会话身份**：命令本身（如 `echo %CD%`）看不出
    # 它在哪个沙箱跑，而"这条命令是在 A 的目录还是 B 的目录执行"恰恰是审批人唯一
    # 需要判断的事（`del *.tmp` 在哪个目录后果完全不同）。带上本会话的 cwd 之后，
    # 日志里每条 approval 都能对号入座——I1 就是靠这个断言"审批事件没串台"。
    detail = f"{command}\n工作目录（沙箱根）：{_workspace(ctx)}"

    # 命令安全策略：在人工审批之前——策略拒了就不必打扰用户
    # （把策略决策外包给用户是不负责任：用户没能力判断每条命令的安全性）
    reject_reason = _check_command_safety(command, ctx)
    if reject_reason:
        # 策略拒绝 ≠ 用户拒绝：用户压根没被问过。审计上必须区分（note3 §2.4）
        _emit_event("approval", ctx=ctx, action="执行命令", detail=detail,
                    granted=False, source="policy")
        return reject_reason

    if not _ask_approval("执行命令", detail, ctx):
        return "用户拒绝了该命令，未执行。"

    try:
        # shell=True：允许管道、重定向等 shell 语法；cwd 锁定在**本会话**的沙箱根
        # Windows 下 shell=True 走 cmd.exe，跨平台够用
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(_workspace(ctx)),
            env={**os.environ, "PYTHONIOENCODING" : "utf-8" },# 确保 Python 输出也用 utf-8 编码
            capture_output=True,       # 捕获 stdout/stderr，而不是直接打到终端
            # 收**字节**而不是 text=True 固定编码：cmd 内建命令按控制台代码页(GBK)输出，
            # python 子进程按 UTF-8 输出，两种编码混在同一条管道里。固定 UTF-8 会把
            # GBK 路径里的中文静默换成 `?`（详见 _decode_subprocess_output）
            timeout=_cfg(ctx).COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"命令超时（>{_cfg(ctx).COMMAND_TIMEOUT}秒）已被终止：{command}"
    except Exception as e:
        return f"命令执行异常：{e}"

    stdout = _decode_subprocess_output(result.stdout or b"")
    stderr = _decode_subprocess_output(result.stderr or b"")

    # 把退出码、标准输出、标准错误一起回喂给模型——
    # 报错信息是 Agent 自我纠错的关键来源，必须完整返回
    # 注意：stdout 和 stderr 必须共享同一份预算。各自 _truncate 一次的话，
    # 单次返回的上限会翻倍到 2×MAX_TOOL_OUTPUT_CHARS，等于绕过了全局预算。
    share = _cfg(ctx).MAX_TOOL_OUTPUT_CHARS // 2
    parts = [f"退出码：{result.returncode}（0=成功，非0=失败）"]
    if stdout.strip():
        parts.append("标准输出：\n" + _clip(stdout.strip(), share))
    if stderr.strip():
        parts.append("标准错误：\n" + _clip(stderr.strip(), share))
    return "\n".join(parts)


# ============================================================
# 工具注册表：schema（给模型看）+ handler（本地执行）
# ============================================================

# 每个 schema 就是一份 JSON Schema，模型靠它决定"要不要调、怎么调"
# 所以 description 要写清楚：工具干什么用、参数什么含义
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出指定目录下的文件和子目录。用于了解项目结构。path 省略时列出工作目录根目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径，相对路径相对于工作目录，例如 src 或 ./src/utils"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取某个文本文件的内容。修改文件前必须先调用本工具。大文件务必用 start_line/end_line 分页读取，避免撑爆上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要读取的文件路径，相对路径相对于工作目录"},
                    "start_line": {"type": "integer", "description": "起始行号（从 1 开始，含）。与 end_line 搭配使用"},
                    "end_line": {"type": "integer", "description": "结束行号（含）。两个参数都省略时读取全文"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入文件（不存在则创建，存在则整体覆盖）。修改已有文件时优先使用edit_file工具，用于创建新文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要写入的文件路径，相对路径相对于工作目录"},
                    "content": {"type": "string", "description": "要写入的完整文件内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "精确替换文件中的一段内容。修改已有文件时优先使用本工具（只传改动片段，"
                "省 token 且不会误删其他内容）。要求 old_string 在文件中唯一匹配："
                "匹配 0 处说明内容已变化需重新 read_file；匹配多处说明上下文不足，"
                "请连同前后行一起提供。old_string 必须与文件逐字一致，包括行首缩进。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要修改的文件路径，相对路径相对于工作目录"},
                    "old_string": {"type": "string", "description": "要被替换的原文片段，必须与文件内容逐字一致（含缩进），且在文件中唯一"},
                    "new_string": {"type": "string", "description": "替换后的新内容"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "在工作目录中执行 shell 命令并返回输出。用于运行脚本、执行测试、安装依赖等验证工作。"
                "安全策略：禁止 del/format/rmdir/cmd/powershell/curl 等破坏性/子壳/外传命令；"
                "禁止 python -c / node -e 等执行任意代码字符串的 flag；"
                "禁止读取凭据文件（.env/*.pem/id_rsa）或访问工作目录外的路径。"
                "如需运行代码请写成 .py 文件后用 python xxx.py 执行。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整的 shell 命令，例如 python hello.py 或 pip list"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "在工作目录内按正则表达式搜索文件内容，返回 文件:行号: 该行内容（保留行首缩进）。用于快速定位代码，比逐个 read_file 省上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "正则表达式，例如 def t_grep"},
                    "path": {"type": "string", "description": "搜索起点：可以是目录（递归搜索），也可以是单个文件"},
                    "include": {"type": "string", "description": "文件名过滤，glob 语法，例如 *.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "按文件名模式（如 **/*.py）查找文件路径，返回匹配的文件列表。语义同 shell：`*` 不跨目录，`*.py` 只匹配顶层，`**/*.py` 才递归。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob 模式，例如 **/*.py 或 agent/*.py"},
                    "path": {"type": "string", "description": "搜索起点：可以是目录，也可以是单个文件"},
                },
                "required": ["pattern"],
            },
        },
    },

]

# 工具名 -> 处理函数 的映射，Agent 循环里按名字查表执行
# handler 签名统一为 (args, ctx) —— ctx 必填，1 参替身会在调用点直接 TypeError
TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "list_dir": t_list_dir,
    "read_file": t_read_file,
    "grep": t_grep,
    "glob": t_glob,
    "write_file": t_write_file,
    "edit_file": t_edit_file,
    "run_command": t_run_command,
}


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    ctx: "ToolContext",
) -> str:
    """
    统一的工具分发入口：Agent 循环拿到模型的 tool_call 后调用这里。
    未知工具 / 参数错误都转成字符串返回给模型（让模型自己看到错误并调整），
    而不是抛异常打断整个循环。

    ctx：本次调用所属会话的上下文（per-session 沙箱 / 审批 / 审计）。
    **必填**（4.6 终态）：工具层没有任何默认上下文可回落，漏传就是 TypeError，
    "忘了传 ctx"不可能再静默串台。
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"错误：未知工具 {name}，可用工具：{list(TOOL_HANDLERS.keys())}"
    try:
        # 出口统一过预算：不管哪个工具返回什么，都不可能把超过
        # MAX_TOOL_OUTPUT_CHARS 的内容灌进上下文（未来新增工具也自动受约束；
        # 工具内部若已截断，这一层就是幂等的空操作）
        return _truncate(handler(arguments, ctx), ctx)
    except Exception as e:
        # 任何工具内部异常都捕获并回喂，保证 Agent 循环不中断
        return f"工具 {name} 执行出错：{type(e).__name__}: {e}"
