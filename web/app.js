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
  summaryTemplateFile: document.getElementById("summaryTemplateFileInput"),
  analyzeSummaryTemplate: document.getElementById("analyzeSummaryTemplateButton"),
  summaryTemplateEditor: document.getElementById("summaryTemplateEditor"),
  summaryTemplateName: document.getElementById("summaryTemplateNameInput"),
  summaryTemplateFields: document.getElementById("summaryTemplateFields"),
  addSummaryTemplateField: document.getElementById("addSummaryTemplateFieldButton"),
  saveSummaryTemplate: document.getElementById("saveSummaryTemplateButton"),
  refreshSummarySessions: document.getElementById("refreshSummarySessionsButton"),
  clearLog: document.getElementById("clearLogButton"),
  status: document.getElementById("connectionStatus"),
  languageStatus: document.getElementById("languageStatus"),
  sourceText: document.getElementById("sourceText"),
  translationText: document.getElementById("translationText"),
  clearSource: document.getElementById("clearSourceButton"),
  clearTranslation: document.getElementById("clearTranslationButton"),
  settingsButton: document.getElementById("settingsButton"),
  closeSettings: document.getElementById("closeSettingsButton"),
  settingsDrawer: document.getElementById("settingsDrawer"),
  settingsBackdrop: document.getElementById("settingsBackdrop"),
  meetingTitle: document.getElementById("meetingTitle"),
  transcriptWorkspace: document.getElementById("transcriptWorkspace"),
  interleavedText: document.getElementById("interleavedText"),
  sourcePaneTitle: document.getElementById("sourcePaneTitle"),
  translationPaneTitle: document.getElementById("translationPaneTitle"),
  translationEnabled: document.getElementById("translationEnabledInput"),
  translationDirection: document.getElementById("translationDirectionInput"),
  translationDirectionField: document.getElementById("translationDirectionField"),
  displayMode: document.getElementById("displayModeInput"),
  singleLanguage: document.getElementById("singleLanguageInput"),
  singleLanguageField: document.getElementById("singleLanguageField"),
  sourceFontSize: document.getElementById("sourceFontSizeInput"),
  sourceFontSizeValue: document.getElementById("sourceFontSizeValue"),
  sourceFontColor: document.getElementById("sourceFontColorInput"),
  translationFontSize: document.getElementById("translationFontSizeInput"),
  translationFontSizeValue: document.getElementById("translationFontSizeValue"),
  translationFontColor: document.getElementById("translationFontColorInput"),
  resetCaptionStyle: document.getElementById("resetCaptionStyleButton"),
  toolStatus: document.getElementById("toolStatus"),
  viewModeButtons: Array.from(document.querySelectorAll("[data-view-mode]")),
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
const sourceSegmentStore = new Map();
const translationSegmentStore = new Map();
const translatedSourceIds = new Set();
let currentSessionId = null;
let currentSessionStartedAt = null;
let currentConfig = null;
let hasStoppedCurrentSession = false;
let summaryGenerated = false;
let summaryGenerating = false;
let selectedSummarySessionId = null;
let selectedSummarySessionStatus = null;
let summaryVersions = [];
let summaryTemplateDraft = null;
let customSummaryTemplates = [];
let lockedHotwords = { hotwords: "", filename: "", count: 0, translationCount: 0 };
let clientInstanceId = null;
let displayMode = "split";
let singleLanguageMode = "source";
let detectedSourceLanguage = null;

const DEFAULT_BACKEND = "faster_whisper";
const DEFAULT_MODEL = "model/asr/small";
const DEFAULT_DISPLAY_SEGMENTS = 16;
const MAX_SESSION_SEGMENTS = 500;
const DEFAULT_CAPTION_STYLE = {
  sourceFontSize: 20,
  sourceFontColor: "#f4f7f5",
  translationFontSize: 20,
  translationFontColor: "#36d98b",
};
const CAPTION_STYLE_STORAGE_KEYS = {
  sourceFontSize: "whisperlive_source_font_size",
  sourceFontColor: "whisperlive_source_font_color",
  translationFontSize: "whisperlive_translation_font_size",
  translationFontColor: "whisperlive_translation_font_color",
};

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

function clampCaptionFontSize(value, fallback) {
  if (value === null || value === undefined || String(value).trim() === "") return fallback;
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(40, Math.max(14, Math.round(number)));
}

function normalizeCaptionColor(value, fallback) {
  const color = String(value || "").trim();
  return /^#[0-9a-f]{6}$/i.test(color) ? color.toLowerCase() : fallback;
}

function readCaptionStyle() {
  return {
    sourceFontSize: clampCaptionFontSize(
      window.localStorage.getItem(CAPTION_STYLE_STORAGE_KEYS.sourceFontSize),
      DEFAULT_CAPTION_STYLE.sourceFontSize,
    ),
    sourceFontColor: normalizeCaptionColor(
      window.localStorage.getItem(CAPTION_STYLE_STORAGE_KEYS.sourceFontColor),
      DEFAULT_CAPTION_STYLE.sourceFontColor,
    ),
    translationFontSize: clampCaptionFontSize(
      window.localStorage.getItem(CAPTION_STYLE_STORAGE_KEYS.translationFontSize),
      DEFAULT_CAPTION_STYLE.translationFontSize,
    ),
    translationFontColor: normalizeCaptionColor(
      window.localStorage.getItem(CAPTION_STYLE_STORAGE_KEYS.translationFontColor),
      DEFAULT_CAPTION_STYLE.translationFontColor,
    ),
  };
}

function applyCaptionStyle(style, persist = false) {
  const normalized = {
    sourceFontSize: clampCaptionFontSize(style.sourceFontSize, DEFAULT_CAPTION_STYLE.sourceFontSize),
    sourceFontColor: normalizeCaptionColor(style.sourceFontColor, DEFAULT_CAPTION_STYLE.sourceFontColor),
    translationFontSize: clampCaptionFontSize(style.translationFontSize, DEFAULT_CAPTION_STYLE.translationFontSize),
    translationFontColor: normalizeCaptionColor(style.translationFontColor, DEFAULT_CAPTION_STYLE.translationFontColor),
  };
  const rootStyle = document.documentElement.style;
  rootStyle.setProperty("--source-font-size", `${normalized.sourceFontSize}px`);
  rootStyle.setProperty("--source-font-color", normalized.sourceFontColor);
  rootStyle.setProperty("--translation-font-size", `${normalized.translationFontSize}px`);
  rootStyle.setProperty("--translation-font-color", normalized.translationFontColor);
  elements.sourceFontSize.value = String(normalized.sourceFontSize);
  elements.sourceFontSizeValue.textContent = `${normalized.sourceFontSize}px`;
  elements.sourceFontColor.value = normalized.sourceFontColor;
  elements.translationFontSize.value = String(normalized.translationFontSize);
  elements.translationFontSizeValue.textContent = `${normalized.translationFontSize}px`;
  elements.translationFontColor.value = normalized.translationFontColor;
  if (persist) {
    Object.entries(CAPTION_STYLE_STORAGE_KEYS).forEach(([name, key]) => {
      window.localStorage.setItem(key, String(normalized[name]));
    });
  }
}

function currentCaptionStyleFromControls() {
  return {
    sourceFontSize: elements.sourceFontSize.value,
    sourceFontColor: elements.sourceFontColor.value,
    translationFontSize: elements.translationFontSize.value,
    translationFontColor: elements.translationFontColor.value,
  };
}

function resetCaptionStyle() {
  Object.values(CAPTION_STYLE_STORAGE_KEYS).forEach((key) => window.localStorage.removeItem(key));
  applyCaptionStyle(DEFAULT_CAPTION_STYLE);
}

function initializeDefaults() {
  const legacyDefault = "ws://localhost:9090";
  if (!elements.server.value || elements.server.value === legacyDefault) {
    elements.server.value = defaultWebSocketUrl();
  }
  clientInstanceId = getClientInstanceId();
  const savedMeeting = window.localStorage.getItem("whisperlive_meeting_name");
  const savedServer = window.localStorage.getItem("whisperlive_server_url");
  const savedLanguage = window.localStorage.getItem("whisperlive_source_language");
  const savedTranslationEnabled = window.localStorage.getItem("whisperlive_translation_enabled");
  const savedTranslationDirection = window.localStorage.getItem("whisperlive_translation_direction");
  displayMode = window.localStorage.getItem("whisperlive_display_mode") || "split";
  singleLanguageMode = window.localStorage.getItem("whisperlive_single_language") || "source";
  if (savedMeeting && !elements.meetingName.value) elements.meetingName.value = savedMeeting;
  if (savedServer) elements.server.value = savedServer;
  if (savedLanguage !== null) elements.language.value = savedLanguage;
  if (savedTranslationEnabled !== null) elements.translationEnabled.checked = savedTranslationEnabled === "true";
  if (savedTranslationDirection) elements.translationDirection.value = savedTranslationDirection;
  elements.displayMode.value = displayMode;
  elements.singleLanguage.value = singleLanguageMode;
  applyCaptionStyle(readCaptionStyle());
  updateTranslationControls();
  setDisplayMode(displayMode);
  updateMeetingTitle();
  updateHotwordStatus("等待开始时加载会议热词文件");
  loadMeetingOptions().catch(() => {
    updateHotwordStatus("热词文件列表暂不可用，可手动填写会议号");
  });
  loadSummarySessions().catch(() => {});
  loadSummaryTemplates().catch(() => {});
}

function parseHotwordText(text) {
  const hotwords = [];
  let translationCount = 0;
  String(text || "").split(/\r?\n/).forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) return;
    if (!line.includes("=>")) {
      hotwords.push(line);
      return;
    }
    const separator = line.indexOf("=>");
    const source = line.slice(0, separator).trim();
    const target = line.slice(separator + 2).trim();
    if (!source || !target) return;
    hotwords.push(source);
    translationCount += 1;
  });
  return { hotwords, translationCount };
}

function countHotwordText(text) {
  return parseHotwordText(text).hotwords.length;
}

function hotwordPromptFromText(text) {
  return parseHotwordText(text).hotwords.join(" ");
}

function updateHotwordStatus(text = "") {
  elements.hotwordStatus.textContent = text || "等待加载";
}

function updateMeetingTitle() {
  elements.meetingTitle.textContent = elements.meetingName.value.trim() || "实时同传";
}

function setToolStatus(text, state = "") {
  if (!elements.toolStatus) return;
  elements.toolStatus.textContent = text;
  elements.toolStatus.className = `tool-status ${state}`.trim();
}

function openSettings() {
  elements.settingsBackdrop.hidden = false;
  requestAnimationFrame(() => elements.settingsBackdrop.classList.add("visible"));
  elements.settingsDrawer.classList.add("open");
  elements.settingsDrawer.setAttribute("aria-hidden", "false");
  elements.settingsButton.setAttribute("aria-expanded", "true");
  document.body.classList.add("drawer-open");
}

function closeSettings() {
  elements.settingsBackdrop.classList.remove("visible");
  elements.settingsDrawer.classList.remove("open");
  elements.settingsDrawer.setAttribute("aria-hidden", "true");
  elements.settingsButton.setAttribute("aria-expanded", "false");
  document.body.classList.remove("drawer-open");
  window.setTimeout(() => {
    if (!elements.settingsDrawer.classList.contains("open")) elements.settingsBackdrop.hidden = true;
  }, 240);
}

function updateTranslationControls() {
  const enabled = elements.translationEnabled.checked;
  elements.translationDirection.disabled = !enabled;
  elements.translationDirectionField.classList.toggle("disabled", !enabled);
}

function normalizeLanguage(value) {
  const language = String(value || "").toLowerCase();
  if (language === "zh" || language.startsWith("zh-")) return "zh";
  if (language === "en" || language.startsWith("en-")) return "en";
  return language || null;
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


function summaryTemplateApiUrl(path = "") {
  return `${hotwordApiBaseUrl().replace(/\/$/, "")}/admin/summary-templates${path}`;
}

function selectedSummaryTemplate() {
  const value = (elements.summaryTemplate && elements.summaryTemplate.value) || "auto";
  if (value.startsWith("custom:")) {
    return { template: "custom", custom_template_id: value.slice(7) };
  }
  return { template: value };
}

async function loadSummaryTemplates(preferredId = "") {
  if (!elements.summaryTemplate) return;
  const response = await fetch(summaryTemplateApiUrl(), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  customSummaryTemplates = data.templates || [];
  const current = preferredId ? `custom:${preferredId}` : elements.summaryTemplate.value;
  elements.summaryTemplate.querySelectorAll('optgroup[data-custom-templates]').forEach((node) => node.remove());
  if (customSummaryTemplates.length) {
    const group = document.createElement("optgroup");
    group.label = "自定义模板";
    group.dataset.customTemplates = "true";
    customSummaryTemplates.forEach((item) => {
      const option = document.createElement("option");
      option.value = `custom:${item.id}`;
      option.textContent = item.name || item.id;
      group.appendChild(option);
    });
    elements.summaryTemplate.appendChild(group);
  }
  if (Array.from(elements.summaryTemplate.options).some((option) => option.value === current)) {
    elements.summaryTemplate.value = current;
  }
}

function renderSummaryTemplateFields() {
  if (!elements.summaryTemplateFields || !summaryTemplateDraft) return;
  elements.summaryTemplateFields.innerHTML = "";
  summaryTemplateDraft.fields.forEach((field, index) => {
    const card = document.createElement("div");
    card.className = "summary-template-field";
    const heading = document.createElement("strong");
    heading.textContent = field.heading;
    card.appendChild(heading);
    const grid = document.createElement("div");
    grid.className = "summary-template-field-grid";
    const label = document.createElement("input");
    label.value = field.label || field.heading;
    label.placeholder = "字段名称";
    label.addEventListener("input", () => { field.label = label.value; });
    const key = document.createElement("input");
    key.value = field.key || `field_${index + 1}`;
    key.placeholder = "field_key";
    key.addEventListener("input", () => { field.key = key.value; });
    const type = document.createElement("select");
    [["text", "文本"], ["list", "列表"], ["evidence_list", "带证据列表"], ["table", "表格"]].forEach(([value, name]) => {
      const option = document.createElement("option"); option.value = value; option.textContent = name; type.appendChild(option);
    });
    type.value = field.type || "text";
    type.addEventListener("change", () => { field.type = type.value; renderSummaryTemplateFields(); });
    const description = document.createElement("input");
    description.value = field.description || "";
    description.placeholder = "从会议原文提取的内容说明";
    description.addEventListener("input", () => { field.description = description.value; });
    grid.append(label, key, type, description);
    card.appendChild(grid);
    if (field.type === "table") {
      const columns = document.createElement("input");
      columns.value = (field.columns || []).join(",");
      columns.placeholder = "表格列名，使用逗号分隔";
      columns.addEventListener("input", () => { field.columns = columns.value.split(/[,，]/).map((item) => item.trim()).filter(Boolean); });
      card.appendChild(columns);
    }
    const actions = document.createElement("div");
    actions.className = "summary-template-field-actions";
    [["上移", -1], ["下移", 1]].forEach(([name, offset]) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "ghost-button"; button.textContent = name;
      button.disabled = index + offset < 0 || index + offset >= summaryTemplateDraft.fields.length;
      button.addEventListener("click", () => {
        const target = index + offset;
        [summaryTemplateDraft.fields[index], summaryTemplateDraft.fields[target]] = [summaryTemplateDraft.fields[target], summaryTemplateDraft.fields[index]];
        renderSummaryTemplateFields();
      });
      actions.appendChild(button);
    });
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "ghost-button"; remove.textContent = "删除";
    remove.addEventListener("click", () => { summaryTemplateDraft.fields.splice(index, 1); renderSummaryTemplateFields(); });
    actions.appendChild(remove);
    card.appendChild(actions);
    elements.summaryTemplateFields.appendChild(card);
  });
}

async function analyzeSummaryTemplate() {
  const file = elements.summaryTemplateFile && elements.summaryTemplateFile.files[0];
  if (!file) throw new Error("请先选择 Markdown 模板文件");
  if (!file.name.toLowerCase().endsWith(".md")) throw new Error("只支持上传 .md 文件");
  const form = new FormData(); form.append("file", file);
  const response = await fetch(summaryTemplateApiUrl("/analyze"), { method: "POST", body: form });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  summaryTemplateDraft = result;
  elements.summaryTemplateName.value = file.name.replace(/\.md$/i, "");
  elements.summaryTemplateEditor.hidden = false;
  renderSummaryTemplateFields();
  setToolStatus("模板分析完成，请确认字段后保存。", "success");
}

async function saveSummaryTemplate() {
  if (!summaryTemplateDraft) throw new Error("请先分析模板");
  const response = await fetch(summaryTemplateApiUrl(`/${encodeURIComponent(summaryTemplateDraft.draft_id)}/confirm`), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: elements.summaryTemplateName.value.trim(), fields: summaryTemplateDraft.fields }),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  const templateId = result.template && result.template.id;
  summaryTemplateDraft = null;
  elements.summaryTemplateEditor.hidden = true;
  elements.summaryTemplateFile.value = "";
  await loadSummaryTemplates(templateId);
  setToolStatus("自定义总结模板已保存。", "success");
}

function renderSummaryVersions() {
  if (!elements.summaryVersion) return;
  const selected = elements.summaryVersion.value;
  elements.summaryVersion.innerHTML = '<option value="">最新版</option>';
  summaryVersions.slice().reverse().forEach((item) => {
    const option = document.createElement("option");
    option.value = String(item.version);
    option.textContent = `v${item.version} · ${item.template_name || item.template || "legacy"} · ${item.generated_at || ""}`;
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
    return { hotwords: "", filename: "", count: 0, translationCount: 0 };
  }
  const response = await fetch(hotwordApiUrl(meetingName), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  const text = data.text || "";
  const parsed = parseHotwordText(text);
  return {
    hotwords: parsed.hotwords.join(" "),
    filename: data.filename || "",
    count: Number(data.count || parsed.hotwords.length),
    translationCount: Number(data.translation_count || parsed.translationCount),
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
  setToolStatus("当前会议日志已下载。", "success");
}

async function generateSummary() {
  if (!selectedSummarySessionId) throw new Error("请选择已结束的会议 session");
  if (selectedSummarySessionStatus !== "finished") throw new Error("请先停止会议后再生成总结");
  summaryGenerating = true; updateSummaryButtons(); setStatus("总结生成中", "busy");
  try {
    const response = await fetch(summaryApiUrl(selectedSummarySessionId), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(selectedSummaryTemplate()),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok || !result.generated) throw new Error(result.error || `HTTP ${response.status}`);
    summaryGenerated = true;
    await loadSummaryInfo(selectedSummarySessionId);
    setStatus(result.summary && result.summary.latest_version > 1 ? "总结已重新生成" : "总结已生成", "ready");
    setToolStatus(result.summary && result.summary.latest_version > 1 ? "总结新版本已生成。" : "总结已生成。", "success");
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
  setToolStatus("总结文件已下载。", "success");
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
  clearTranscriptState();
  renderTranscriptViews();
  updateSummaryButtons();
}


function segmentTimeValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function sourceSegmentStoreKey(segment) {
  const utteranceId = String(segment?.utterance_id || "").trim();
  if (utteranceId) {
    if (segment.completed === false) return `source-draft:${utteranceId}`;
    return `source-final:${utteranceId}:${segmentTimeValue(segment.start).toFixed(3)}:${segmentTimeValue(segment.end).toFixed(3)}`;
  }
  if (segment.completed === false) return "source-draft:fallback";
  return `source-final:${segmentTimeValue(segment.start).toFixed(3)}:${segmentTimeValue(segment.end).toFixed(3)}`;
}

function translationSegmentStoreKey(segment) {
  const sourceIds = (segment?.source_utterance_ids || [segment?.utterance_id])
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .join(",");
  return `translation:${sourceIds}:${segmentTimeValue(segment?.start).toFixed(3)}:${segmentTimeValue(segment?.end).toFixed(3)}`;
}

function sortSegmentsByTime(segments) {
  return segments.sort((left, right) => {
    const startDelta = segmentTimeValue(left.start) - segmentTimeValue(right.start);
    if (startDelta) return startDelta;
    const endDelta = segmentTimeValue(left.end) - segmentTimeValue(right.end);
    if (endDelta) return endDelta;
    return Number(left.completed !== false) - Number(right.completed !== false);
  });
}

function pruneSegmentStore(store) {
  if (store.size <= MAX_SESSION_SEGMENTS) return;
  const ordered = Array.from(store.entries()).sort(([, left], [, right]) => {
    const startDelta = segmentTimeValue(left.start) - segmentTimeValue(right.start);
    if (startDelta) return startDelta;
    return segmentTimeValue(left.end) - segmentTimeValue(right.end);
  });
  ordered.slice(0, ordered.length - MAX_SESSION_SEGMENTS).forEach(([key]) => store.delete(key));
}

function rebuildTranslatedSourceIds() {
  translatedSourceIds.clear();
  translationSegmentStore.forEach((segment) => {
    (segment.source_utterance_ids || [segment.utterance_id])
      .map((value) => String(value || "").trim())
      .filter(Boolean)
      .forEach((utteranceId) => translatedSourceIds.add(utteranceId));
  });
}

function syncSegmentArrays() {
  sourceSegments = sortSegmentsByTime(Array.from(sourceSegmentStore.values()));
  translatedSegments = sortSegmentsByTime(Array.from(translationSegmentStore.values()));
  rebuildTranslatedSourceIds();
}

function mergeSourceSnapshot(segments) {
  const incoming = Array.isArray(segments) ? segments : [];

  // A server snapshot contains the current draft. Remove the previous draft,
  // then add the new one while retaining all completed history.
  sourceSegmentStore.forEach((segment, key) => {
    if (segment.completed === false) sourceSegmentStore.delete(key);
  });

  incoming.forEach((segment) => {
    const copy = { ...segment };
    sourceSegmentStore.set(sourceSegmentStoreKey(copy), copy);
  });
  pruneSegmentStore(sourceSegmentStore);
  syncSegmentArrays();
}

function mergeTranslationSnapshot(segments) {
  const incoming = Array.isArray(segments) ? segments : [];
  incoming.forEach((segment) => {
    const copy = {
      ...segment,
      source_utterance_ids: Array.isArray(segment.source_utterance_ids)
        ? segment.source_utterance_ids.slice()
        : segment.source_utterance_ids,
    };
    translationSegmentStore.set(translationSegmentStoreKey(copy), copy);
  });
  pruneSegmentStore(translationSegmentStore);
  syncSegmentArrays();
}

function clearSourceSegmentState() {
  sourceSegmentStore.clear();
  sourceSegments = [];
}

function clearTranslationSegmentState() {
  translationSegmentStore.clear();
  translatedSourceIds.clear();
  translatedSegments = [];
}

function clearTranscriptState() {
  clearSourceSegmentState();
  clearTranslationSegmentState();
}

function segmentGroupKey(segment, index) {
  const utteranceId = String(segment?.utterance_id || "").trim();
  if (utteranceId) return `utterance:${utteranceId}`;
  return `segment:${index}:${segment?.start ?? ""}:${segment?.end ?? ""}`;
}

function joinDisplayText(previous, current) {
  const left = String(previous || "").trim();
  const right = String(current || "").trim();
  if (!left) return right;
  if (!right) return left;
  const needsSpace = /[A-Za-z0-9]$/.test(left) && /^[A-Za-z0-9]/.test(right);
  return `${left}${needsSpace ? " " : ""}${right}`;
}

function groupSegmentsForDisplay(segments) {
  const groups = [];
  segments.forEach((segment, index) => {
    const key = segmentGroupKey(segment, index);
    const previous = groups.at(-1);
    if (previous && previous.group_key === key && segment.utterance_id) {
      previous.text = joinDisplayText(previous.text, segment.text);
      previous.end = segment.end;
      previous.completed = previous.completed !== false && segment.completed !== false;
      return;
    }
    groups.push({ ...segment, group_key: key, text: String(segment.text || "").trim() });
  });
  return groups;
}

function renderSegments(target, segments, emptyText) {
  const displaySegments = groupSegmentsForDisplay(segments).slice(-getDisplayLimit());
  if (!displaySegments.length) {
    target.textContent = emptyText;
    target.classList.add("muted");
    target.dataset.lastGroupKey = "";
    return;
  }
  const previousLastGroupKey = target.dataset.lastGroupKey || "";
  const nextLastGroupKey = displaySegments.at(-1)?.group_key || "";
  target.classList.remove("muted");
  target.innerHTML = "";
  const fragment = document.createDocumentFragment();
  displaySegments.forEach((segment) => {
    const paragraph = document.createElement("p");
    paragraph.className = `segment${segment.completed ? "" : " incomplete"}`;
    paragraph.textContent = String(segment.text || "").trim();
    fragment.appendChild(paragraph);
  });
  target.appendChild(fragment);
  if (previousLastGroupKey !== nextLastGroupKey || target.scrollTop + target.clientHeight >= target.scrollHeight - 24) {
    target.scrollTop = target.scrollHeight;
  }
  target.dataset.lastGroupKey = nextLastGroupKey;
}

function overlapDuration(left, right) {
  const leftStart = Number(left.start);
  const leftEnd = Number(left.end);
  const rightStart = Number(right.start);
  const rightEnd = Number(right.end);
  if (![leftStart, leftEnd, rightStart, rightEnd].every(Number.isFinite)) return 0;
  return Math.max(0, Math.min(leftEnd, rightEnd) - Math.max(leftStart, rightStart));
}

function interleavedSourceKey(source, index) {
  const utteranceId = String(source?.utterance_id || "").trim();
  if (utteranceId) return `source-utterance:${utteranceId}`;
  return `source-start:${Number(source?.start || 0).toFixed(3)}:${Number(source?.end || 0).toFixed(3)}`;
}

function translationSourceIds(translation) {
  return (translation.source_utterance_ids || [translation.utterance_id])
    .map((value) => String(value || "").trim())
    .filter(Boolean);
}

function matchedSourceIndexes(sources, translation) {
  const sourceIds = new Set(translationSourceIds(translation));
  if (sourceIds.size) {
    const indexes = sources
      .map((source, index) => sourceIds.has(String(source.utterance_id || "").trim()) ? index : -1)
      .filter((index) => index >= 0);
    if (indexes.length === sourceIds.size) return indexes;
    return [];
  }
  return sources
    .map((source, index) => overlapDuration(source, translation) > 0 ? index : -1)
    .filter((index) => index >= 0);
}

function areIndexesContiguous(indexes) {
  return indexes.every((value, index) => index === 0 || value === indexes[index - 1] + 1);
}

function interleavedTranslationKey(translation, sourceIndexes) {
  const sourceIds = translationSourceIds(translation);
  if (sourceIds.length) return `translation-group:${sourceIds.join(",")}`;
  return `translation-range:${Number(translation.start || 0).toFixed(3)}:${Number(translation.end || 0).toFixed(3)}:${sourceIndexes.join(",")}`;
}

function buildInterleavedCompletedRows(sources, translations) {
  const rows = [];
  const consumedSourceIndexes = new Set();
  const unmatchedTranslations = [];

  translations.forEach((translation) => {
    const indexes = matchedSourceIndexes(sources, translation)
      .filter((index) => !consumedSourceIndexes.has(index));
    if (!indexes.length || !areIndexesContiguous(indexes)) {
      unmatchedTranslations.push(translation);
      return;
    }

    const matchedSources = indexes.map((index) => sources[index]);
    const sourceText = matchedSources
      .map((source) => String(source.text || "").trim())
      .filter(Boolean)
      .reduce(joinDisplayText, "");
    indexes.forEach((index) => consumedSourceIndexes.add(index));
    rows.push({
      key: interleavedTranslationKey(translation, indexes),
      start: Number(matchedSources[0]?.start ?? translation.start) || 0,
      source: sourceText,
      translation: String(translation.text || "").trim(),
      pending: false,
    });
  });

  sources.forEach((source, index) => {
    if (consumedSourceIndexes.has(index)) return;
    rows.push({
      key: interleavedSourceKey(source, index),
      start: Number(source.start) || 0,
      source: String(source.text || "").trim(),
      translation: "翻译中...",
      pending: true,
    });
  });

  unmatchedTranslations.forEach((translation, index) => {
    if (translationSourceIds(translation).length) return;
    rows.push({
      key: `translation:${translation.group_key || index}`,
      start: Number(translation.start) || 0,
      source: "（原文片段处理中）",
      translation: String(translation.text || "").trim(),
      pending: false,
    });
  });

  return rows;
}

function updateInterleavedRow(container, row) {
  let source = container.querySelector(".interleaved-source");
  let translation = container.querySelector(".interleaved-translation");
  if (!source || !translation) {
    source = document.createElement("p");
    source.className = "interleaved-source";
    translation = document.createElement("p");
    translation.className = "interleaved-translation";
    container.replaceChildren(source, translation);
  }
  source.textContent = row.source || "（原文片段处理中）";
  translation.className = `interleaved-translation${row.pending ? " pending" : ""}`;
  translation.textContent = row.translation;
}

function renderInterleavedRows(rows) {
  const target = elements.interleavedText;
  const wasNearBottom = target.scrollTop + target.clientHeight >= target.scrollHeight - 24;
  if (!target.querySelector(".interleaved-row")) {
    target.replaceChildren();
  }
  const existing = new Map(
    Array.from(target.querySelectorAll(".interleaved-row[data-row-key]"))
      .map((node) => [node.dataset.rowKey, node])
  );
  const visibleKeys = new Set(rows.map((row) => row.key));

  existing.forEach((node, key) => {
    if (!visibleKeys.has(key)) node.remove();
  });

  rows.forEach((row) => {
    let container = existing.get(row.key);
    if (!container) {
      container = document.createElement("section");
      container.className = "interleaved-row";
      container.dataset.rowKey = row.key;
    }
    updateInterleavedRow(container, row);
    target.appendChild(container);
  });

  if (wasNearBottom) {
    target.scrollTop = target.scrollHeight;
  }
}

function renderInterleaved() {
  const completedSources = groupSegmentsForDisplay(sourceSegments.filter((item) => item.completed !== false));
  const displayTranslations = groupSegmentsForDisplay(translatedSegments);
  const rows = buildInterleavedCompletedRows(completedSources, displayTranslations);
  const latestIncomplete = groupSegmentsForDisplay(sourceSegments.filter((item) => item.completed === false)).slice(-1)[0];
  if (latestIncomplete && latestIncomplete.text) {
    rows.push({
      key: String(latestIncomplete.utterance_id || "").trim()
        ? interleavedSourceKey(latestIncomplete, completedSources.length)
        : "source-draft",
      start: Number(latestIncomplete.start) || Number.MAX_SAFE_INTEGER,
      source: latestIncomplete.text.trim(),
      translation: "识别中...",
      pending: true,
    });
  }
  rows.sort((left, right) => left.start - right.start);
  const visibleRows = rows.slice(-getDisplayLimit());
  if (!visibleRows.length) {
    elements.interleavedText.textContent = "等待语音输入...";
    elements.interleavedText.classList.add("muted");
    return;
  }
  elements.interleavedText.classList.remove("muted");
  renderInterleavedRows(visibleRows);
}

function resolveSingleLanguageStream() {
  if (singleLanguageMode === "source") return { segments: sourceSegments, title: "原文", pane: "source" };
  if (singleLanguageMode === "translation") return { segments: translatedSegments, title: "翻译", pane: "translation" };
  const sourceLanguage = normalizeLanguage(detectedSourceLanguage || elements.language.value);
  const translatedLanguage = normalizeLanguage(translatedSegments.at(-1)?.target_language);
  if (singleLanguageMode === sourceLanguage) return { segments: sourceSegments, title: singleLanguageMode === "zh" ? "中文" : "English", pane: "source" };
  if (singleLanguageMode === translatedLanguage) return { segments: translatedSegments, title: singleLanguageMode === "zh" ? "中文" : "English", pane: "translation" };
  return { segments: [], title: singleLanguageMode === "zh" ? "中文" : "English", pane: "source" };
}

function renderTranscriptViews() {
  elements.sourcePaneTitle.textContent = "原文";
  elements.translationPaneTitle.textContent = "翻译";
  renderSegments(elements.sourceText, sourceSegments, "等待语音输入...");
  renderSegments(elements.translationText, translatedSegments, elements.translationEnabled.checked ? "等待翻译结果..." : "翻译已关闭");
  renderInterleaved();
  elements.transcriptWorkspace.classList.remove("show-translation");
  if (displayMode === "single") {
    const selected = resolveSingleLanguageStream();
    if (selected.pane === "translation") {
      elements.translationPaneTitle.textContent = selected.title;
      elements.transcriptWorkspace.classList.add("show-translation");
      renderSegments(elements.translationText, selected.segments, `等待${selected.title}内容...`);
    } else {
      elements.sourcePaneTitle.textContent = selected.title;
      renderSegments(elements.sourceText, selected.segments, `等待${selected.title}内容...`);
    }
  }
}

function setDisplayMode(mode) {
  displayMode = ["split", "interleaved", "single"].includes(mode) ? mode : "split";
  window.localStorage.setItem("whisperlive_display_mode", displayMode);
  elements.displayMode.value = displayMode;
  elements.singleLanguageField.hidden = displayMode !== "single";
  elements.transcriptWorkspace.className = `transcript-workspace mode-${displayMode}`;
  elements.viewModeButtons.forEach((button) => button.classList.toggle("active", button.dataset.viewMode === displayMode));
  renderTranscriptViews();
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
    detectedSourceLanguage = message.language;
    elements.languageStatus.textContent = `${message.language} (${Number(message.language_prob || 0).toFixed(2)})`;
  }

  if (message.segments) {
    mergeSourceSnapshot(message.segments);
    renderTranscriptViews();
  }

  if (message.translated_segments) {
    mergeTranslationSnapshot(message.translated_segments);
    renderTranscriptViews();
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
    vad_parameters: {
      threshold: 0.5,
      min_silence_duration_ms: 600,
      speech_pad_ms: 300,
    },
    send_last_n_segments: DEFAULT_DISPLAY_SEGMENTS,
    no_speech_thresh: 0.45,
    clip_audio: false,
    same_output_threshold: 5,
    min_segment_rms: 0.002,
    max_incomplete_segment_seconds: 0.0,
    enable_translation: elements.translationEnabled.checked,
    target_language: elements.translationDirection.value || "auto",
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
  clearTranscriptState();
  detectedSourceLanguage = null;
  renderTranscriptViews();
  const meetingName = elements.meetingName.value.trim();
  if (meetingName) {
    window.localStorage.setItem("whisperlive_meeting_name", meetingName);
  updateMeetingTitle();
  }
  try {
    lockedHotwords = await fetchMeetingHotwordSnapshot(meetingName);
    updateHotwordStatus(
      lockedHotwords.filename
        ? `${lockedHotwords.filename} · ${lockedHotwords.count} 个热词 · ${lockedHotwords.translationCount} 条固定翻译`
        : "使用默认热词"
    );
  } catch (error) {
    lockedHotwords = { hotwords: "", filename: "", count: 0, translationCount: 0 };
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
  [
    elements.server,
    elements.language,
    elements.meetingName,
    elements.translationEnabled,
    elements.translationDirection,
    elements.meetingSelect,
  ].filter(Boolean).forEach((item) => { item.disabled = true; });
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
  [
    elements.server,
    elements.language,
    elements.meetingName,
    elements.translationEnabled,
    elements.meetingSelect,
  ].filter(Boolean).forEach((item) => { item.disabled = false; });
  updateTranslationControls();
  if (elements.refreshMeetings) elements.refreshMeetings.disabled = false;
}

elements.settingsButton.addEventListener("click", openSettings);
elements.closeSettings.addEventListener("click", closeSettings);
elements.settingsBackdrop.addEventListener("click", closeSettings);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeSettings();
});
elements.viewModeButtons.forEach((button) => {
  button.addEventListener("click", () => setDisplayMode(button.dataset.viewMode));
});
elements.displayMode.addEventListener("change", () => setDisplayMode(elements.displayMode.value));
elements.singleLanguage.addEventListener("change", () => {
  singleLanguageMode = elements.singleLanguage.value;
  window.localStorage.setItem("whisperlive_single_language", singleLanguageMode);
  renderTranscriptViews();
});
[elements.sourceFontSize, elements.sourceFontColor, elements.translationFontSize, elements.translationFontColor]
  .forEach((input) => {
    input.addEventListener("input", () => applyCaptionStyle(currentCaptionStyleFromControls(), true));
  });
elements.resetCaptionStyle.addEventListener("click", resetCaptionStyle);
elements.translationEnabled.addEventListener("change", () => {
  window.localStorage.setItem("whisperlive_translation_enabled", String(elements.translationEnabled.checked));
  updateTranslationControls();
  renderTranscriptViews();
});
elements.translationDirection.addEventListener("change", () => {
  window.localStorage.setItem("whisperlive_translation_direction", elements.translationDirection.value);
});
elements.server.addEventListener("change", () => window.localStorage.setItem("whisperlive_server_url", elements.server.value.trim()));
elements.language.addEventListener("change", () => window.localStorage.setItem("whisperlive_source_language", elements.language.value));

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
    setToolStatus(`日志导出失败：${error.message}`, "error");
  });
});
if (elements.analyzeSummaryTemplate) {
  elements.analyzeSummaryTemplate.addEventListener("click", () => {
    setToolStatus("正在分析 Markdown 模板…");
    analyzeSummaryTemplate().catch((error) => {
      console.error(error);
      setToolStatus(`模板分析失败：${error.message}`, "error");
    });
  });
}
if (elements.saveSummaryTemplate) {
  elements.saveSummaryTemplate.addEventListener("click", () => {
    saveSummaryTemplate().catch((error) => {
      console.error(error);
      setToolStatus(`模板保存失败：${error.message}`, "error");
    });
  });
}
if (elements.addSummaryTemplateField) {
  elements.addSummaryTemplateField.addEventListener("click", () => {
    if (!summaryTemplateDraft) return;
    const used = new Set(summaryTemplateDraft.fields.map((field) => field.heading));
    const section = (summaryTemplateDraft.sections || []).find((item) => !used.has(item.heading));
    if (!section) {
      setToolStatus("模板中的所有标题都已配置字段。", "error");
      return;
    }
    summaryTemplateDraft.fields.push({
      key: `field_${summaryTemplateDraft.fields.length + 1}`,
      label: section.heading,
      heading: section.heading,
      type: "text",
      description: `根据会议原文填写“${section.heading}”`,
      columns: [],
    });
    renderSummaryTemplateFields();
  });
}
if (elements.generateSummary) {
  elements.generateSummary.addEventListener("click", () => {
    generateSummary().catch((error) => {
      console.error(error);
      setStatus("总结生成失败", "error");
      setToolStatus(`总结生成失败：${error.message}`, "error");
    });
  });
}
if (elements.downloadSummary) {
  elements.downloadSummary.addEventListener("click", () => {
    downloadSummary().catch((error) => {
      console.error(error);
      setStatus("总结下载失败", "error");
      setToolStatus(`总结下载失败：${error.message}`, "error");
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
  updateMeetingTitle();
  updateHotwordStatus("等待开始时加载会议热词文件");
});

if (elements.meetingSelect) {
  elements.meetingSelect.addEventListener("change", () => {
    if (!elements.meetingSelect.value) return;
    elements.meetingName.value = elements.meetingSelect.value;
    window.localStorage.setItem("whisperlive_meeting_name", elements.meetingName.value.trim());
    updateMeetingTitle();
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
  clearSourceSegmentState();
  renderTranscriptViews();
});

elements.clearTranslation.addEventListener("click", () => {
  clearTranslationSegmentState();
  renderTranscriptViews();
});

updateSummaryButtons();
initializeDefaults();
