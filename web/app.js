const TARGET_SAMPLE_RATE = 16000;

const elements = {
  form: document.getElementById("settingsForm"),
  server: document.getElementById("serverInput"),
  meetingName: document.getElementById("meetingNameInput"),
  meetingSelect: document.getElementById("meetingSelectInput"),
  refreshMeetings: document.getElementById("refreshMeetingsButton"),
  hotwordStatus: document.getElementById("hotwordStatus"),
  language: document.getElementById("languageInput"),
  start: document.getElementById("startButton"),
  stop: document.getElementById("stopButton"),
  exportLog: document.getElementById("exportLogButton"),
  generateSummary: document.getElementById("generateSummaryButton"),
  downloadSummary: document.getElementById("downloadSummaryButton"),
  summarySession: document.getElementById("summarySessionInput"),
  summaryTemplate: document.getElementById("summaryTemplateInput"),
  summaryVersion: document.getElementById("summaryVersionInput"),
  refreshSummarySessions: document.getElementById("refreshSummarySessionsButton"),
  clearLog: document.getElementById("clearLogButton"),
  status: document.getElementById("connectionStatus"),
  languageStatus: document.getElementById("languageStatus"),
  sourceText: document.getElementById("sourceText"),
  translationText: document.getElementById("translationText"),
  clearSource: document.getElementById("clearSourceButton"),
  clearTranslation: document.getElementById("clearTranslationButton"),
};

let socket = null;
let mediaStream = null;
let audioContext = null;
let processor = null;
let sourceNode = null;
let uid = null;
let isServerReady = false;
let sourceSegments = [];
let translatedSegments = [];
let currentSessionId = null;
let currentSessionStartedAt = null;
let currentConfig = null;
let hasStoppedCurrentSession = false;
let summaryGenerated = false;
let summaryGenerating = false;
let selectedSummarySessionId = null;
let selectedSummarySessionStatus = null;
let summaryVersions = [];
let lockedHotwords = { hotwords: "", filename: "", count: 0 };
let clientInstanceId = null;

const DEFAULT_BACKEND = "faster_whisper";
const DEFAULT_MODEL = "model/asr/small";
const DEFAULT_DISPLAY_SEGMENTS = 8;

function getDisplayLimit() {
  return DEFAULT_DISPLAY_SEGMENTS;
}

function createUid() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getClientInstanceId() {
  const key = "whisperlive_client_instance_id";
  let saved = window.localStorage.getItem(key);
  if (!saved) {
    saved = createUid();
    window.localStorage.setItem(key, saved);
  }
  return saved;
}

function defaultWebSocketUrl() {
  if (!window.location.host) {
    return "ws://localhost:9090";
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

function initializeDefaults() {
  const legacyDefault = "ws://localhost:9090";
  if (!elements.server.value || elements.server.value === legacyDefault) {
    elements.server.value = defaultWebSocketUrl();
  }
  clientInstanceId = getClientInstanceId();
  const savedMeeting = window.localStorage.getItem("whisperlive_meeting_name");
  if (savedMeeting && !elements.meetingName.value) {
    elements.meetingName.value = savedMeeting;
  }
  updateHotwordStatus("等待开始时加载会议热词文件");
  loadMeetingOptions().catch(() => {
    updateHotwordStatus("热词文件列表暂不可用，可手动填写会议号");
  });
  loadSummarySessions().catch(() => {});
}

function countHotwordText(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#")).length;
}

function hotwordPromptFromText(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
    .join(" ");
}

function updateHotwordStatus(text = "") {
  elements.hotwordStatus.textContent = text || "服务端按会议号匹配 txt；开始时自动加载并锁定。";
}

function hotwordApiBaseUrl() {
  const serverUrl = elements.server.value.trim();
  if (serverUrl.startsWith("wss://")) return serverUrl.replace(/^wss:/, "https:").replace(/\/ws\/?$/, "");
  if (serverUrl.startsWith("ws://")) return serverUrl.replace(/^ws:/, "http:").replace(/\/ws\/?$/, "");
  return window.location.origin === "null" ? "http://localhost:9093" : window.location.origin;
}

function hotwordListUrl() {
  return `${hotwordApiBaseUrl().replace(/\/$/, "")}/admin/hotwords`;
}

function hotwordApiUrl(meetingName) {
  return `${hotwordApiBaseUrl().replace(/\/$/, "")}/admin/hotwords/${encodeURIComponent(meetingName)}`;
}

function meetingLogApiUrl() {
  return `${hotwordApiBaseUrl().replace(/\/$/, "")}/admin/meeting-logs`;
}

function meetingLogDownloadUrl(sessionId, format = "md") {
  return `${meetingLogApiUrl()}/${encodeURIComponent(sessionId)}?format=${encodeURIComponent(format)}`;
}

function summaryApiUrl(sessionId) {
  return `${meetingLogApiUrl()}/${encodeURIComponent(sessionId)}/summary`;
}

function summaryInfoUrl(sessionId) {
  return `${summaryApiUrl(sessionId)}/info`;
}

function summaryDownloadUrl(sessionId, format = "md", version = "") {
  const params = new URLSearchParams({ format });
  if (version) params.set("version", version);
  return `${summaryApiUrl(sessionId)}?${params.toString()}`;
}


function selectedSummaryTemplate() {
  return (elements.summaryTemplate && elements.summaryTemplate.value) || "auto";
}

function renderSummaryVersions() {
  if (!elements.summaryVersion) return;
  const selected = elements.summaryVersion.value;
  elements.summaryVersion.innerHTML = '<option value="">最新版</option>';
  summaryVersions.slice().reverse().forEach((item) => {
    const option = document.createElement("option");
    option.value = String(item.version);
    option.textContent = `v${item.version} · ${item.template || "legacy"} · ${item.generated_at || ""}`;
    elements.summaryVersion.appendChild(option);
  });
  elements.summaryVersion.value = Array.from(elements.summaryVersion.options).some((option) => option.value === selected) ? selected : "";
  elements.summaryVersion.disabled = !summaryVersions.length;
}

async function loadSummaryInfo(sessionId = selectedSummarySessionId) {
  if (!sessionId) {
    summaryGenerated = false;
    summaryVersions = [];
    renderSummaryVersions();
    updateSummaryButtons();
    return;
  }
  const response = await fetch(summaryInfoUrl(sessionId), { cache: "no-store" });
  if (response.status === 404) {
    summaryGenerated = false;
    summaryVersions = [];
  } else {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    summaryGenerated = Boolean(data.has_summary);
    summaryVersions = data.versions || [];
  }
  renderSummaryVersions();
  updateSummaryButtons();
}

async function loadSummarySessions(preferredSessionId = selectedSummarySessionId || currentSessionId) {
  if (!elements.summarySession) return;
  const response = await fetch(meetingLogApiUrl(), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  const sessions = (data.sessions || []).filter((item) => item.status === "finished");
  elements.summarySession.innerHTML = '<option value="">暂无已结束会议</option>';
  sessions.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.session_id;
    option.textContent = `${item.meeting_name || "未命名会议"} · ${item.created_at || item.session_id}`;
    option.dataset.status = item.status || "";
    elements.summarySession.appendChild(option);
  });
  const hasPreferred = sessions.some((item) => item.session_id === preferredSessionId);
  if (!hasPreferred && preferredSessionId && preferredSessionId === currentSessionId && hasStoppedCurrentSession) {
    const option = document.createElement("option");
    option.value = preferredSessionId;
    option.textContent = "当前会议 · 等待后端完成记录";
    option.dataset.status = "finished";
    elements.summarySession.appendChild(option);
  }
  const nextId = (hasPreferred || (preferredSessionId === currentSessionId && hasStoppedCurrentSession))
    ? preferredSessionId
    : (sessions[0] && sessions[0].session_id) || null;
  selectedSummarySessionId = nextId;
  selectedSummarySessionStatus = nextId ? "finished" : null;
  elements.summarySession.value = nextId || "";
  await loadSummaryInfo(nextId);
}


async function loadMeetingOptions() {
  if (!elements.meetingSelect) return;
  const response = await fetch(hotwordListUrl(), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  const meetings = data.meetings || [];
  const current = elements.meetingName.value.trim();
  elements.meetingSelect.innerHTML = '<option value="">手动填写会议号</option>';
  meetings.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.meeting_name;
    option.textContent = `${item.meeting_name} (${item.count || 0})`;
    option.dataset.filename = item.filename || "";
    elements.meetingSelect.appendChild(option);
  });
  if (current && meetings.some((item) => item.meeting_name === current)) {
    elements.meetingSelect.value = current;
  }
  updateHotwordStatus(meetings.length ? `已发现 ${meetings.length} 个服务器热词文件` : "服务器暂无会议热词文件，可手动填写会议号");
}

async function fetchMeetingHotwordSnapshot(meetingName) {
  if (!meetingName) {
    return { hotwords: "", filename: "", count: 0 };
  }
  const response = await fetch(hotwordApiUrl(meetingName), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  const text = data.text || "";
  return {
    hotwords: hotwordPromptFromText(text),
    filename: data.filename || "",
    count: Number(data.count || countHotwordText(text)),
  };
}

function setStatus(text, state = "idle") {
  elements.status.textContent = text;
  elements.status.className = `status ${state}`;
}

function safeExportFilenamePrefix(value) {
  const cleaned = String(value || "")
    .trim()
    .replace(/[\/\\:*?"<>|\x00-\x1F]/g, "_")
    .replace(/^[ ._]+|[ ._]+$/g, "");
  return cleaned || "meeting-log";
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function filenameFromContentDisposition(value, fallback) {
  const match = String(value || "").match(/filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i);
  const encoded = match && (match[1] || match[2]);
  if (!encoded) return fallback;
  try { return decodeURIComponent(encoded); } catch (_error) { return encoded; }
}

async function exportMeetingLog() {
  if (!currentSessionId) throw new Error("当前没有可导出的后端日志 session");
  const response = await fetch(meetingLogDownloadUrl(currentSessionId, "md"), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const blob = await response.blob();
  const filenamePrefix = safeExportFilenamePrefix((currentConfig && (currentConfig.meeting_name || currentConfig.client_name)) || elements.meetingName.value.trim());
  const fallback = `${filenamePrefix || "meeting-log"}-${currentSessionStartedAt || new Date().toISOString()}.md`.replace(/[:.]/g, "-");
  downloadBlob(filenameFromContentDisposition(response.headers.get("Content-Disposition"), fallback), blob);
  setStatus("后端日志已下载", "ready");
}

async function generateSummary() {
  if (!selectedSummarySessionId) throw new Error("请选择已结束的会议 session");
  if (selectedSummarySessionStatus !== "finished") throw new Error("请先停止会议后再生成总结");
  summaryGenerating = true; updateSummaryButtons(); setStatus("总结生成中", "busy");
  try {
    const response = await fetch(summaryApiUrl(selectedSummarySessionId), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ template: selectedSummaryTemplate() }),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok || !result.generated) throw new Error(result.error || `HTTP ${response.status}`);
    summaryGenerated = true;
    await loadSummaryInfo(selectedSummarySessionId);
    setStatus(result.summary && result.summary.latest_version > 1 ? "总结已重新生成" : "总结已生成", "ready");
    return result;
  } finally {
    summaryGenerating = false; updateSummaryButtons();
  }
}

async function downloadSummary() {
  if (!selectedSummarySessionId) throw new Error("请选择可下载总结的会议 session");
  const version = elements.summaryVersion ? elements.summaryVersion.value : "";
  const response = await fetch(summaryDownloadUrl(selectedSummarySessionId, "md", version), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const blob = await response.blob();
  const fallback = `meeting-summary-${selectedSummarySessionId}${version ? `-v${version}` : ""}.md`;
  downloadBlob(filenameFromContentDisposition(response.headers.get("Content-Disposition"), fallback), blob);
  setStatus("总结已下载", "ready");
}

function updateSummaryButtons() {
  const canGenerate = Boolean(selectedSummarySessionId) && selectedSummarySessionStatus === "finished";
  if (elements.generateSummary) {
    elements.generateSummary.disabled = !canGenerate || summaryGenerating;
    elements.generateSummary.textContent = summaryGenerating ? "生成中" : (summaryGenerated ? "重新生成" : "生成总结");
  }
  if (elements.downloadSummary) {
    elements.downloadSummary.disabled = !selectedSummarySessionId || !summaryGenerated || summaryGenerating;
  }
}

function clearMeetingLog() {
  currentSessionId = null;
  currentSessionStartedAt = null;
  hasStoppedCurrentSession = false;
  summaryGenerating = false;
  updateSummaryButtons();
}


function renderSegments(target, segments, emptyText) {
  const displaySegments = segments.slice(-getDisplayLimit());

  if (!displaySegments.length) {
    target.textContent = emptyText;
    target.classList.add("muted");
    return;
  }

  target.classList.remove("muted");
  target.innerHTML = "";
  const fragment = document.createDocumentFragment();
  displaySegments.forEach((segment) => {
    const paragraph = document.createElement("p");
    paragraph.className = `segment${segment.completed ? "" : " incomplete"}`;
    paragraph.textContent = segment.text.trim();
    fragment.appendChild(paragraph);
  });
  target.appendChild(fragment);
  target.scrollTop = target.scrollHeight;
}

function handleMessage(event) {
  const message = JSON.parse(event.data);
  if (message.uid !== uid) {
    return;
  }

  if (message.status === "WAIT") {
    setStatus("等待", "busy");
    return;
  }

  if (message.status === "ERROR" || message.status === "WARNING") {
    setStatus(message.status, "error");
    console.warn(message.message);
    return;
  }

  if (message.message === "SERVER_READY") {
    isServerReady = true;
    setStatus("已连接", "ready");
    return;
  }

  if (message.language) {
    elements.languageStatus.textContent = `${message.language} (${Number(message.language_prob || 0).toFixed(2)})`;
  }

  if (message.segments) {
    sourceSegments = message.segments.slice();
    renderSegments(elements.sourceText, sourceSegments, "等待语音输入...");
  }

  if (message.translated_segments) {
    translatedSegments = message.translated_segments.slice();
    renderSegments(elements.translationText, translatedSegments, "等待翻译结果...");
  }
}

function downsampleTo16k(input, inputSampleRate) {
  if (inputSampleRate === TARGET_SAMPLE_RATE) {
    return input;
  }

  const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j += 1) {
      sum += input[j];
      count += 1;
    }
    output[i] = count ? sum / count : 0;
  }

  return output;
}

function sendConfig(event) {
  const targetSocket = event?.target || socket;
  if (!targetSocket || targetSocket.readyState !== WebSocket.OPEN) {
    return;
  }

  uid = createUid();
  const selectedLanguage = elements.language.value || null;
  const meetingName = elements.meetingName.value.trim();
  const payload = {
    uid,
    session_id: currentSessionId,
    session_started_at: currentSessionStartedAt,
    server: elements.server.value.trim(),
    client_instance_id: clientInstanceId || getClientInstanceId(),
    client_name: meetingName || `Client-${uid.slice(0, 8)}`,
    meeting_name: meetingName,
    hotwords: lockedHotwords.hotwords || null,
    hotwords_count: lockedHotwords.count || 0,
    hotwords_file: lockedHotwords.filename || "",
    hotwords_locked: true,
    backend: DEFAULT_BACKEND,
    language: selectedLanguage,
    task: "transcribe",
    model: DEFAULT_MODEL,
    use_vad: true,
    send_last_n_segments: DEFAULT_DISPLAY_SEGMENTS,
    no_speech_thresh: 0.45,
    clip_audio: false,
    same_output_threshold: 2,
    min_segment_rms: 0.002,
    max_incomplete_segment_seconds: 0.0,
    enable_translation: true,
    target_language: "auto",
    translation_provider: "helsinki_zh_en",
    zh_en_model_path: "model/opus-mt-zh-en",
    en_zh_model_path: "model/opus-mt-en-zh",
  };
  currentConfig = payload;
  targetSocket.send(JSON.stringify(payload));
}

async function requestMicrophoneStream() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("当前页面无法访问麦克风。请使用 HTTPS，或在 client 本机通过 http://localhost 打开页面。");
  }

  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      channelCount: 1,
    },
  });
}

async function startCapture() {
  setStatus("连接中", "busy");
  currentSessionId = createUid();
  currentSessionStartedAt = new Date().toISOString();
  hasStoppedCurrentSession = false;
  summaryGenerated = false;
  summaryGenerating = false;
  selectedSummarySessionId = currentSessionId;
  selectedSummarySessionStatus = "active";
  summaryVersions = [];
  renderSummaryVersions();
  if (elements.summarySession) elements.summarySession.value = "";
  updateSummaryButtons();
  sourceSegments = [];
  translatedSegments = [];
  renderSegments(elements.sourceText, sourceSegments, "等待语音输入...");
  renderSegments(elements.translationText, translatedSegments, "等待翻译结果...");
  const meetingName = elements.meetingName.value.trim();
  if (meetingName) {
    window.localStorage.setItem("whisperlive_meeting_name", meetingName);
  }
  try {
    lockedHotwords = await fetchMeetingHotwordSnapshot(meetingName);
    updateHotwordStatus(
      lockedHotwords.filename
        ? `已锁定热词文件 ${lockedHotwords.filename}，${lockedHotwords.count} 个热词`
        : "未找到会议热词文件，本次使用默认热词"
    );
  } catch (error) {
    lockedHotwords = { hotwords: "", filename: "", count: 0 };
    updateHotwordStatus("热词预取不可用，将由服务端按会议号匹配");
  }

  mediaStream = await requestMicrophoneStream();

  socket = new WebSocket(elements.server.value.trim());
  socket.binaryType = "arraybuffer";
  socket.addEventListener("open", sendConfig);
  socket.addEventListener("message", handleMessage);
  socket.addEventListener("error", () => setStatus("连接错误", "error"));
  socket.addEventListener("close", () => {
    if (elements.start.disabled) {
      setStatus("已断开", "idle");
    }
    stopCapture(false);
  });

  audioContext = new AudioContext();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (event) => {
    if (!socket || socket.readyState !== WebSocket.OPEN || !isServerReady) {
      return;
    }
    const input = event.inputBuffer.getChannelData(0);
    const output = downsampleTo16k(input, audioContext.sampleRate);
    socket.send(output.buffer);
  };

  sourceNode.connect(processor);
  processor.connect(audioContext.destination);

  elements.start.disabled = true;
  elements.stop.disabled = false;
  elements.form.querySelectorAll("input, select").forEach((item) => {
    item.disabled = true;
  });
  elements.meetingName.disabled = true;
  if (elements.meetingSelect) elements.meetingSelect.disabled = true;
  if (elements.refreshMeetings) elements.refreshMeetings.disabled = true;
}

function stopCapture(sendEnd = true) {
  if (sendEnd && currentSessionId) {
    hasStoppedCurrentSession = true;
    selectedSummarySessionId = currentSessionId;
    selectedSummarySessionStatus = "finished";
    updateSummaryButtons();
    window.setTimeout(() => {
      loadSummarySessions(currentSessionId).catch(() => {});
    }, 500);
  }
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (socket && socket.readyState === WebSocket.OPEN) {
    if (sendEnd) {
      socket.send(new TextEncoder().encode("END_OF_AUDIO"));
    }
    socket.close();
  }

  socket = null;
  isServerReady = false;
  elements.start.disabled = false;
  elements.stop.disabled = true;
  elements.form.querySelectorAll("input, select").forEach((item) => {
    item.disabled = false;
  });
  elements.meetingName.disabled = false;
  if (elements.meetingSelect) elements.meetingSelect.disabled = false;
  if (elements.refreshMeetings) elements.refreshMeetings.disabled = false;
}

elements.start.addEventListener("click", () => {
  startCapture().catch((error) => {
    console.error(error);
    setStatus("启动失败", "error");
    stopCapture(false);
  });
});

elements.stop.addEventListener("click", () => {
  setStatus("停止中", "busy");
  stopCapture(true);
});

elements.exportLog.addEventListener("click", () => {
  exportMeetingLog().catch((error) => {
    console.error(error);
    setStatus("日志导出失败", "error");
  });
});
if (elements.generateSummary) {
  elements.generateSummary.addEventListener("click", () => {
    generateSummary().catch((error) => {
      console.error(error);
      setStatus("总结生成失败", "error");
    });
  });
}
if (elements.downloadSummary) {
  elements.downloadSummary.addEventListener("click", () => {
    downloadSummary().catch((error) => {
      console.error(error);
      setStatus("总结下载失败", "error");
    });
  });
}
if (elements.summarySession) {
  elements.summarySession.addEventListener("change", () => {
    selectedSummarySessionId = elements.summarySession.value || null;
    const selected = elements.summarySession.selectedOptions[0];
    selectedSummarySessionStatus = selectedSummarySessionId ? (selected.dataset.status || "finished") : null;
    loadSummaryInfo(selectedSummarySessionId).catch((error) => {
      console.error(error);
      setStatus("总结信息加载失败", "error");
    });
  });
}
if (elements.refreshSummarySessions) {
  elements.refreshSummarySessions.addEventListener("click", () => {
    loadSummarySessions().catch((error) => {
      console.error(error);
      setStatus("会议列表刷新失败", "error");
    });
  });
}
elements.clearLog.addEventListener("click", clearMeetingLog);

elements.meetingName.addEventListener("change", () => {
  const meetingName = elements.meetingName.value.trim();
  window.localStorage.setItem("whisperlive_meeting_name", meetingName);
  if (elements.meetingSelect) {
    const exists = Array.from(elements.meetingSelect.options).some((option) => option.value === meetingName);
    elements.meetingSelect.value = meetingName && exists ? meetingName : "";
  }
  updateHotwordStatus("等待开始时加载会议热词文件");
});

if (elements.meetingSelect) {
  elements.meetingSelect.addEventListener("change", () => {
    if (!elements.meetingSelect.value) return;
    elements.meetingName.value = elements.meetingSelect.value;
    window.localStorage.setItem("whisperlive_meeting_name", elements.meetingName.value.trim());
    updateHotwordStatus("等待开始时加载会议热词文件");
  });
}

if (elements.refreshMeetings) {
  elements.refreshMeetings.addEventListener("click", () => {
    loadMeetingOptions().catch((error) => {
      updateHotwordStatus(`热词文件列表加载失败：${error}`);
    });
  });
}


elements.clearSource.addEventListener("click", () => {
  sourceSegments = [];
  renderSegments(elements.sourceText, sourceSegments, "等待语音输入...");
});

elements.clearTranslation.addEventListener("click", () => {
  translatedSegments = [];
  renderSegments(elements.translationText, translatedSegments, "等待翻译结果...");
});

updateSummaryButtons();
initializeDefaults();
