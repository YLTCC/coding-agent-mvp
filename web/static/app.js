/* app.js —— 极简前端（决策 F / I / J / W5）
 *
 * 三个刻意的设计：
 *  1. **只给"当前选中的会话"开一条 SSE**（`EventSource` 在 HTTP/1.1 下每域名约 6 连接）。
 *     代价是切走之后那个会话变成 detached，它的审批会立即被拒（fail-closed，决策 J）。
 *     真要同时盯多个会话，得改成"一条连接 + 事件里带 sid"，那是下一票的事。
 *  2. **事件去重靠 `id:` + 客户端单调计数**（决策 I）：`id:` 只出现在落盘事件上，
 *     delta 一定没有 `id:`。客户端的计数只活在前端，**绝不进入 `id:`**。
 *  3. 审批必须**看得见"共享工作区"**（决策 D 选 C）：同 workspace 的会话在头部
 *     挂一条**常驻**提示条，不是一闪而过的 toast——共享是持续状态，不是事件。
 */
"use strict";

const API = "";
const state = {
  sid: null,
  es: null,
  sessions: [],
  localSeq: 0,       // 纯前端的去重/动画计数，不进 SSE 的 id:
  lastDiskSeq: 0,    // 已渲染的最大落盘 seq（服务端会用 Last-Event-ID 续传）
  assistantBuf: null,
  openToolCard: null,   // 最近一张工具卡片（工具结果落盘后挂回它）
};

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* 忽略 */ }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.status === 204 ? null : res.json();
}

/* ---------------- 会话列表 ---------------- */

async function refreshSessions() {
  const data = await api("/api/sessions");
  state.sessions = data.sessions;
  renderSessions();
}

function renderSessions() {
  const ul = $("session-list");
  ul.innerHTML = "";
  for (const s of state.sessions) {
    const li = document.createElement("li");
    if (s.sid === state.sid) li.className = "active";
    const ws = s.workspace.split(/[\\/]/).pop() || s.workspace;
    li.innerHTML =
      `<div>${escapeHtml(ws)}` +
      (s.shared ? `<span class="badge">共享工作区（${s.shared_with.length + 1} 个活跃会话）</span>` : "") +
      `</div><div class="sid">${s.sid}${s.inflight ? " · 运行中" : ""}</div>`;
    if (s.shared) {
      li.title = "同工作区的活跃会话：" + s.shared_with.join(", ") +
        "\n注意：文件层不隔离，并发写会静默覆盖";
    }
    li.onclick = () => selectSession(s.sid);
    ul.appendChild(li);
  }
}

async function createSession() {
  const name = $("ws-name").value.trim();
  try {
    const s = await api("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workspace: name || null }),
    });
    await refreshSessions();
    selectSession(s.sid);
  } catch (e) {
    alert("建会话失败：" + e.message);
  }
}

/* ---------------- 选中 + SSE ---------------- */

function selectSession(sid) {
  if (state.sid === sid) return;
  if (state.es) { state.es.close(); state.es = null; }
  state.sid = sid;
  state.localSeq = 0;
  state.lastDiskSeq = 0;
  state.assistantBuf = null;
  state.openToolCard = null;
  $("stream").innerHTML = "";
  renderSessions();

  const info = state.sessions.find((s) => s.sid === sid);
  $("title").textContent = info ? info.workspace : sid;
  const banner = $("shared-banner");
  if (info && info.shared) {
    // 常驻提示条（决策 D）：共享是**持续状态**，一闪而过的提示等于没提示
    banner.textContent =
      `共享工作区：还有 ${info.shared_with.length} 个活跃会话指向同一目录` +
      `（${info.shared_with.join(", ")}）——文件层不隔离，并发写会静默覆盖`;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }

  openStream(sid, 0);
}

function openStream(sid, lastEventId) {
  // 用 query 传 Last-Event-ID：EventSource 自己重连时会带 header，
  // 我们手动重连（比如切回来）时只能走 query
  let url = `/api/sessions/${sid}/stream`;
  if (lastEventId) url += `?last_event_id=${lastEventId}`;
  const es = new EventSource(url);
  state.es = es;

  es.onopen = () => { state.localSeq = 0; };
  es.onerror = () => {
    // EventSource 自带重连；服务端用 Last-Event-ID 从"最后一个落盘记录"续
    pushNotice("连接中断，EventSource 会自动重连（未定型的 delta 不补发）", "notice");
  };

  const kinds = [
    "replay", "delta", "stream_end", "empty_reply", "tool_call", "tool_parse_error",
    "max_iterations", "compress_failed", "task_cost", "log_path",
    "approval_request", "approval_missed", "lagged", "error",
    "session_start", "user", "assistant", "tool", "state", "task_end", "approval", "clear",
  ];
  for (const kind of kinds) {
    es.addEventListener(kind, (ev) => {
      let payload = {};
      try { payload = JSON.parse(ev.data); } catch (e) { payload = { raw: ev.data }; }
      if (ev.lastEventId) state.lastDiskSeq = Number(ev.lastEventId) || state.lastDiskSeq;
      handle(kind, payload);
    });
  }
}

/* ---------------- 渲染 ---------------- */

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function atBottom() {
  const el = $("stream");
  return el.scrollHeight - el.scrollTop - el.clientHeight < 60;
}
function scroll(wasBottom) { if (wasBottom) $("stream").scrollTop = $("stream").scrollHeight; }

function appendNode(node) {
  const wasBottom = atBottom();
  $("stream").appendChild(node);
  scroll(wasBottom);
}

function pushNotice(text, cls) {
  const d = document.createElement("div");
  d.className = "card " + (cls || "notice");
  d.textContent = text;
  appendNode(d);
}

function pushMsg(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.innerHTML = `<div class="role">${escapeHtml(role)}</div>` +
                `<div class="body">${escapeHtml(text)}</div>`;
  appendNode(d);
  return d.querySelector(".body");
}

function ensureAssistantBuf() {
  if (!state.assistantBuf) state.assistantBuf = pushMsg("assistant", "");
  return state.assistantBuf;
}

function handle(kind, p) {
  switch (kind) {
    case "replay": {
      // 已提交态（服务端 replay_file 的结果）：重连/切换时重建历史
      $("stream").innerHTML = "";
      state.assistantBuf = null;
      state.openToolCard = null;
      for (const m of (p.messages || [])) {
        if (m.role === "system") continue;
        pushMsg(m.role, m.content || "");
      }
      if (p.summary) pushNotice("（更早的对话已被压缩成摘要）", "notice");
      break;
    }
    case "delta":
      ensureAssistantBuf().textContent += p.text;
      break;
    case "stream_end":
      state.assistantBuf = null;
      break;
    case "empty_reply":
      pushNotice("Agent：（空回复）", "notice");
      break;
    case "tool_call": {
      // 折叠卡片（W5）：summary 显示工具名，展开看参数
      const d = document.createElement("details");
      d.className = "card tool";
      d.innerHTML =
        `<summary>🔧 调用工具：${escapeHtml(p.name)}</summary>` +
        `<div class="body">${escapeHtml(p.brief)}</div>` +
        `<div class="body result">（等待结果…）</div>`;
      appendNode(d);
      state.openToolCard = d;      // 结果来了（event: tool）挂回这张卡上
      state.assistantBuf = null;
      break;
    }
    case "tool_parse_error":
      pushNotice(`❌ 工具 ${p.name} 参数解析失败：${p.error}`, "error");
      break;
    case "max_iterations":
      pushNotice(`⚠️ 已达最大循环轮次（${p.limit}），任务被迫中止`, "error");
      break;
    case "compress_failed":
      pushNotice(`⚠️ 上下文压缩失败（任务继续）：${p.error}`, "notice");
      break;
    case "task_cost":
      $("stats").textContent =
        `${p.iterations} 轮 · ${p.used} tokens（prompt ${p.prompt_tokens} / completion ${p.completion_tokens}）` +
        ` · cache 命中 ${Number(p.hit_rate).toFixed(1)}%`;
      break;
    case "log_path":
      pushNotice(`会话日志：${p.path}`, "notice");
      break;
    case "approval_request": {
      state.assistantBuf = null;
      const d = document.createElement("div");
      d.className = "card approval";
      d.dataset.aid = p.aid;
      d.innerHTML = `<div class="card-head">⚠️ Agent 请求${escapeHtml(p.action)}（${p.timeout}s 内不点 = 拒绝）</div>` +
                    `<div class="detail">${escapeHtml(p.detail)}</div>`;
      const bar = document.createElement("div");
      const allow = document.createElement("button");
      allow.className = "allow"; allow.textContent = "允许";
      const deny = document.createElement("button");
      deny.className = "deny"; deny.textContent = "拒绝";
      allow.onclick = () => decide(p.aid, true, d);
      deny.onclick = () => decide(p.aid, false, d);
      bar.appendChild(allow); bar.appendChild(deny);
      d.appendChild(bar);
      appendNode(d);
      break;
    }
    case "approval_missed":
      pushNotice(`审批未送达用户，已按拒绝处理（reason=${p.reason}）：${p.action}`, "error");
      break;
    case "approval":
      pushNotice(`${p.granted ? "✅ 已允许" : "⛔ 已拒绝"}${p.action}（source=${p.source}）`, "notice");
      break;
    case "lagged":
      pushNotice(p.detail || `断线期间跳过了 ${p.dropped_deltas} 个片段`, "notice");
      break;
    case "error":
      pushNotice("❌ " + p.error, "error");
      break;
    case "tool":
      // 工具结果（落盘事件）：挂回刚才那张折叠卡片，而不是另起一块
      if (state.openToolCard) {
        const slot = state.openToolCard.querySelector(".result");
        if (slot) slot.textContent = "→ " + (p.content || "");
        state.openToolCard = null;
      }
      break;
    case "user":
    case "assistant":
      state.assistantBuf = null;
      break;
    default:
      break;
  }
}

async function decide(aid, allow, card) {
  try {
    await api(`/api/sessions/${state.sid}/approvals/${aid}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ allow }),
    });
    card.querySelectorAll("button").forEach((b) => (b.disabled = true));
  } catch (e) {
    // 409 = 晚到/重复点击。**必须说出来**：静默处理会让用户以为点击生效了
    pushNotice(`本次点击不生效：${e.message}`, "error");
    card.querySelectorAll("button").forEach((b) => (b.disabled = true));
  }
}

/* ---------------- 发任务 ---------------- */

async function send() {
  const text = $("input").value.trim();
  if (!text || !state.sid) return;
  state.assistantBuf = null;
  pushMsg("user", text);
  $("input").value = "";
  try {
    await api(`/api/sessions/${state.sid}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    refreshSessions();
  } catch (e) {
    pushNotice("派任务失败：" + e.message, "error");
  }
}

/* ---------------- 启动 ---------------- */

$("create-btn").onclick = createSession;
$("send-btn").onclick = send;
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});
refreshSessions();
setInterval(refreshSessions, 5000);
