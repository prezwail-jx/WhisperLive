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
  hotwordSelect: document.getElementById("hotwordSelectInput"),
  hotwordStatus: document.getElementById("hotwordStatus"),
  hotwordFileList: document.getElementById("hotwordFileList"),
  hotwordPreview: document.getElementById("hotwordPreview"),
  refreshHotwords: document.getElementById("refreshHotwordsButton"),
};

let timer = null;

function setStatus(text, state) {
  els.badge.textContent = text;
  els.badge.className = `badge ${state}`;
}

function initializeDefaultApi() {
  const legacyDefault = "http://localhost:8000";
  const defaultOrigin = window.location.origin === "null" ? legacyDefault : window.location.origin;
  if (!els.api.value || els.api.value === legacyDefault) {
    els.api.value = defaultOrigin;
  }
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

function hotwordListUrl() {
  return `${apiBaseUrl()}/admin/hotwords`;
}

function hotwordUrl(meetingName) {
  const meeting = String(meetingName || "").trim();
  if (!meeting) throw new Error("请先选择会议热词");
  return `${apiBaseUrl()}/admin/hotwords/${encodeURIComponent(meeting)}`;
}

function fmtSeconds(value) {
  const seconds = Number(value || 0);
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.floor(seconds % 60)}s`;
}

function fmtTime(value) {
  if (!value) return "-";
  return new Date(Number(value) * 1000).toLocaleString();
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
    els.rows.innerHTML = '<tr><td colspan="14" class="empty">暂无 Client</td></tr>';
    return;
  }

  els.rows.innerHTML = clients.map((client) => {
    const [stateClass, stateText] = clientState(client);
    const hotwordFile = client.hotwords_file || "-";
    return `<tr>
      <td><span class="state ${stateClass}">${stateText}</span></td>
      <td class="name" title="${escapeHtml(client.client_name)}">${escapeHtml(client.client_name || "-")}</td>
      <td class="uid" title="${escapeHtml(client.uid)}">${escapeHtml(client.uid)}</td>
      <td class="name" title="${escapeHtml(client.meeting_name)}">${escapeHtml(client.meeting_name || "-")}</td>
      <td class="name" title="${escapeHtml(hotwordFile)}">${client.hotwords_locked ? "锁定" : "-"} / ${Number(client.hotwords_count || 0)} / ${escapeHtml(hotwordFile)}</td>
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

function renderHotwordList(meetings) {
  els.hotwordSelect.innerHTML = '<option value="">请选择会议热词</option>';
  if (!meetings.length) {
    els.hotwordFileList.innerHTML = '<div class="empty small">暂无会议热词文件</div>';
    els.hotwordStatus.textContent = "暂无热词文件";
    els.hotwordPreview.textContent = "把 txt 文件放到服务器 config/hotwords.d 后点击刷新。";
    return;
  }

  els.hotwordStatus.textContent = `已扫描 ${meetings.length} 个会议热词文件`;
  meetings.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.meeting_name;
    option.textContent = `${item.meeting_name} (${item.count || 0})`;
    option.dataset.filename = item.filename || "";
    els.hotwordSelect.appendChild(option);
  });

  els.hotwordFileList.innerHTML = meetings.map((item) => `
    <button class="hotword-file-item" type="button" data-meeting="${escapeHtml(item.meeting_name)}">
      <strong>${escapeHtml(item.meeting_name)}</strong>
      <span>${escapeHtml(item.filename || "-")}</span>
      <span>${Number(item.count || 0)} 个热词</span>
      <span>${escapeHtml(fmtTime(item.updated_at))}</span>
    </button>
  `).join("");
}

async function refreshHotwordList() {
  try {
    const selected = els.hotwordSelect.value;
    const response = await fetch(hotwordListUrl(), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const meetings = data.meetings || [];
    renderHotwordList(meetings);
    if (selected && meetings.some((item) => item.meeting_name === selected)) {
      els.hotwordSelect.value = selected;
    }
  } catch (error) {
    els.hotwordStatus.textContent = `热词列表加载失败：${error}`;
  }
}

async function showMeetingHotwords(meetingName) {
  const response = await fetch(hotwordUrl(meetingName), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  els.hotwordSelect.value = data.meeting_name || meetingName;
  els.hotwordPreview.textContent = data.text || "该会议暂无热词文件。";
  els.hotwordStatus.textContent = data.filename ? `当前：${data.meeting_name} / ${data.filename} / ${data.count} 个热词` : `当前：${data.meeting_name} / 无文件`;
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

els.hotwordFileList.addEventListener("click", (event) => {
  const button = event.target.closest(".hotword-file-item");
  if (!button) return;
  showMeetingHotwords(button.dataset.meeting).catch((error) => {
    els.hotwordStatus.textContent = String(error);
  });
});

els.hotwordSelect.addEventListener("change", () => {
  if (!els.hotwordSelect.value) {
    els.hotwordPreview.textContent = "选择列表中的会议查看内容。";
    return;
  }
  showMeetingHotwords(els.hotwordSelect.value).catch((error) => {
    els.hotwordStatus.textContent = String(error);
  });
});

els.refreshHotwords.addEventListener("click", refreshHotwordList);

initializeDefaultApi();
els.start.addEventListener("click", startPolling);
els.stop.addEventListener("click", () => {
  stopPolling();
  setStatus("已停止", "idle");
});

refreshHotwordList();
startPolling();
