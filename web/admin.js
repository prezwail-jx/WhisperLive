const els = {
  api: document.getElementById("apiInput"),
  interval: document.getElementById("intervalInput"),
  start: document.getElementById("startButton"),
  stop: document.getElementById("stopButton"),
  badge: document.getElementById("statusBadge"),
  updatedAt: document.getElementById("updatedAt"),
  rows: document.getElementById("clientRows"),
  activeCount: document.getElementById("activeCount"),
  translationCount: document.getElementById("translationCount"),
  recentCount: document.getElementById("recentCount"),
  totalCount: document.getElementById("totalCount"),
};

let timer = null;

function setStatus(text, state) {
  els.badge.textContent = text;
  els.badge.className = `badge ${state}`;
}

function apiBaseUrl() {
  return els.api.value.replace(/\/$/, "");
}

function apiUrl() {
  return `${apiBaseUrl()}/admin/clients`;
}

function deleteClientUrl(uid) {
  return `${apiBaseUrl()}/admin/clients/${encodeURIComponent(uid)}`;
}

function fmtSeconds(value) {
  const seconds = Number(value || 0);
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.floor(seconds % 60)}s`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function clientState(client) {
  if (!client.connected) return ["disconnected", "断开"];
  if (client.translation_enabled && client.translation_msgs === 0 && client.connected_seconds > 8) {
    return ["no_translation", "无翻译"];
  }
  if (client.last_activity_seconds_ago <= 5) return ["active", "活跃"];
  return ["idle", "空闲"];
}

function render(data) {
  const clients = data.clients || [];
  const active = clients.filter((client) => client.connected).length;
  const translating = clients.filter((client) => client.connected && client.translation_enabled).length;
  const recent = clients.filter((client) => client.connected && client.last_activity_seconds_ago <= 5).length;

  els.activeCount.textContent = active;
  els.translationCount.textContent = translating;
  els.recentCount.textContent = recent;
  els.totalCount.textContent = clients.length;
  els.updatedAt.textContent = `刷新 ${new Date().toLocaleTimeString()}`;

  if (!clients.length) {
    els.rows.innerHTML = '<tr><td colspan="11" class="empty">暂无 Client</td></tr>';
    return;
  }

  els.rows.innerHTML = clients.map((client) => {
    const [stateClass, stateText] = clientState(client);
    return `<tr>
      <td><span class="state ${stateClass}">${stateText}</span></td>
      <td class="uid" title="${escapeHtml(client.uid)}">${escapeHtml(client.uid)}</td>
      <td>${fmtSeconds(client.connected_seconds)}</td>
      <td>${escapeHtml(client.language || "auto")}</td>
      <td class="model" title="${escapeHtml(client.model)}">${escapeHtml(client.model || "-")}</td>
      <td>${client.segment_msgs} / ${client.segment_items}</td>
      <td>${client.translation_msgs} / ${client.translation_items}</td>
      <td>${fmtSeconds(client.last_activity_seconds_ago)}</td>
      <td class="text" title="${escapeHtml(client.last_source_text)}">${escapeHtml(client.last_source_text || "-")}</td>
      <td class="text" title="${escapeHtml(client.last_translation_text)}">${escapeHtml(client.last_translation_text || "-")}</td>
      <td class="actions-cell">${client.connected ? "-" : `<button class="delete-client" type="button" data-uid="${escapeHtml(client.uid)}" title="删除断开记录" aria-label="删除断开记录 ${escapeHtml(client.uid)}">×</button>`}</td>
    </tr>`;
  }).join("");
}

async function refresh() {
  try {
    setStatus("刷新中", "busy");
    const response = await fetch(apiUrl(), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    render(data);
    setStatus("已连接", "ready");
  } catch (error) {
    setStatus("连接错误", "error");
    els.updatedAt.textContent = String(error);
  }
}

async function deleteClient(uid) {
  try {
    setStatus("删除中", "busy");
    const response = await fetch(deleteClientUrl(uid), { method: "DELETE" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await refresh();
  } catch (error) {
    setStatus("删除失败", "error");
    els.updatedAt.textContent = String(error);
  }
}

function startPolling() {
  stopPolling();
  refresh();
  const interval = Math.max(500, Number(els.interval.value || 1000));
  timer = window.setInterval(refresh, interval);
}

function stopPolling() {
  if (timer) {
    window.clearInterval(timer);
    timer = null;
  }
}

els.rows.addEventListener("click", (event) => {
  const button = event.target.closest(".delete-client");
  if (!button) return;
  deleteClient(button.dataset.uid);
});

els.start.addEventListener("click", startPolling);
els.stop.addEventListener("click", () => {
  stopPolling();
  setStatus("已停止", "idle");
});

startPolling();
