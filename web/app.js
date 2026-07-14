const TARGET_SAMPLE_RATE = 16000;

const elements = {
  form: document.getElementById("settingsForm"),
  server: document.getElementById("serverInput"),
  meetingName: document.getElementById("meetingNameInput"),
  hotwordFile: document.getElementById("hotwordFileInput"),
  clearHotwordFile: document.getElementById("clearHotwordFileButton"),
  hotwordStatus: document.getElementById("hotwordStatus"),
  language: document.getElementById("languageInput"),
  translationModeHint: document.getElementById("translationModeHint"),
  translationProvider: document.getElementById("translationProviderInput"),
  faceToFaceEnabled: document.getElementById("faceToFaceEnabledInput"),
  faceToFaceMode: document.getElementById("faceToFaceModeInput"),
  translationTarget: document.getElementById("translationTargetInput"),
  start: document.getElementById("startButton"),
  stop: document.getElementById("stopButton"),
  continueMeeting: document.getElementById("continueButton"),
  finishInterrupted: document.getElementById("finishInterruptedButton"),
  exportLog: document.getElementById("exportLogButton"),
  exportLogDocx: document.getElementById("exportLogDocxButton"),
  exportInterleavedLogDocx: document.getElementById("exportInterleavedLogDocxButton"),
  generateSummary: document.getElementById("generateSummaryButton"),
  downloadSummary: document.getElementById("downloadSummaryButton"),
  downloadSummaryDocx: document.getElementById("downloadSummaryDocxButton"),
  summarySession: document.getElementById("summarySessionInput"),
  summaryTemplate: document.getElementById("summaryTemplateInput"),
  deleteSummaryTemplate: document.getElementById("deleteSummaryTemplateButton"),
  summaryVersion: document.getElementById("summaryVersionInput"),
  summaryTemplateFile: document.getElementById("summaryTemplateFileInput"),
  analyzeSummaryTemplate: document.getElementById("analyzeSummaryTemplateButton"),
  summaryTemplateEditor: document.getElementById("summaryTemplateEditor"),
  summaryTemplateName: document.getElementById("summaryTemplateNameInput"),
  summaryTemplateFields: document.getElementById("summaryTemplateFields"),
  addSummaryTemplateField: document.getElementById("addSummaryTemplateFieldButton"),
  saveSummaryTemplate: document.getElementById("saveSummaryTemplateButton"),
  refreshSummarySessions: document.getElementById("refreshSummarySessionsButton"),
  loadTranscriptEditor: document.getElementById("loadTranscriptEditorButton"),
  transcriptEditorStatus: document.getElementById("transcriptEditorStatus"),
  transcriptEditorRows: document.getElementById("transcriptEditorRows"),
  speakerManagerList: document.getElementById("speakerManagerList"),
  speakerStats: document.getElementById("speakerStats"),
  speakerRenameGuide: document.getElementById("speakerRenameGuide"),
  newSpeakerName: document.getElementById("newSpeakerNameInput"),
  addSpeaker: document.getElementById("addSpeakerButton"),
  clearLog: document.getElementById("clearLogButton"),
  status: document.getElementById("connectionStatus"),
  drawerConnectionStatus: document.getElementById("drawerConnectionStatus"),
  drawerMeetingName: document.getElementById("drawerMeetingName"),
  drawerTranslationStatus: document.getElementById("drawerTranslationStatus"),
  languageStatus: document.getElementById("languageStatus"),
  sourceText: document.getElementById("sourceText"),
  translationText: document.getElementById("translationText"),
  clearSource: document.getElementById("clearSourceButton"),
  clearTranslation: document.getElementById("clearTranslationButton"),
  summaryQuick: document.getElementById("summaryQuickButton"),
  summaryPanel: document.getElementById("summaryPanel"),
  summaryDrawer: document.getElementById("summaryDrawer"),
  closeSummary: document.getElementById("closeSummaryButton"),
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
  diarizationEnabled: document.getElementById("diarizationEnabledInput"),
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
  translationProviderField: document.getElementById("translationProviderField"),
  faceToFaceModeField: document.getElementById("faceToFaceModeField"),
  languageField: document.getElementById("languageField"),
  translationTargetField: document.getElementById("translationTargetField"),
  sourceLanguageButton: document.getElementById("sourceLanguageButton"),
  targetLanguageButton: document.getElementById("targetLanguageButton"),
  directionToggleButton: document.getElementById("directionToggleButton"),
  fullscreenButton: document.getElementById("fullscreenButton"),
  serviceModeButtons: Array.from(document.querySelectorAll("[data-service-mode]")),
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
let sourceClearedBefore = null;
let translationClearedBefore = null;
let currentSessionId = null;
let currentSessionStartedAt = null;
let currentConfig = null;
let currentServerBackend = null;
let hasStoppedCurrentSession = false;
let summaryGenerated = false;
let summaryGenerating = false;
let selectedSummarySessionId = null;
let selectedSummarySessionStatus = null;
let summaryVersions = [];
let summaryTemplateDraft = null;
let customSummaryTemplates = [];
let transcriptEditorData = null;
let translationModels = [];
const translationModelByValue = new Map();
let lockedHotwords = { hotwords: "", filename: "", count: 0, translationCount: 0, translationGlossary: {} };
let clientInstanceId = null;
let displayMode = "split";
let singleLanguageMode = "translation";
let serviceMode = "standard";
let directionMode = "auto";
let selectedSourceLanguage = "zh";
let selectedTargetLanguageCode = "en";
let immersiveFullscreenFallback = false;
let detectedSourceLanguage = null;
let resumeNextConnection = false;
let intentionallyClosingSocket = false;
let reconnectController = null;

const DEFAULT_BACKEND = "faster_whisper";
const DEFAULT_MODEL = "model/asr/small";
const DEFAULT_DISPLAY_SEGMENTS = 16;
const DEFAULT_DISPLAY_MODE = "stacked";
const DEFAULT_TRANSLATION_MODE = "interpretation";
const DEFAULT_SERVICE_MODE = "standard";
const SERVICE_MODES = ["standard", "accurate", "conversation", "transcription"];
const TRANSLATION_ERROR_SUFFIX = "（翻译出错）";
const MAX_SESSION_SEGMENTS = 500;
const DEFAULT_CAPTION_STYLE = {
  sourceFontSize: 20,
  sourceFontColor: "#f4f7f5",
  translationFontSize: 20,
  translationFontColor: "#36d98b",
};
const FULLSCREEN_CAPTION_SCALE = 1.5;
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

function webSocketPathForMode(mode = serviceMode) {
  return mode === "accurate" ? "/ws-accurate" : "/ws-standard";
}

function webSocketUrlForMode(url, mode = serviceMode) {
  const fallback = defaultWebSocketUrl();
  const rawUrl = String(url || fallback).trim() || fallback;
  try {
    const parsed = new URL(rawUrl);
    parsed.pathname = webSocketPathForMode(mode);
    parsed.search = "";
    parsed.hash = "";
    return parsed.toString();
  } catch (_error) {
    const parsed = new URL(fallback);
    parsed.pathname = webSocketPathForMode(mode);
    return parsed.toString();
  }
}

function syncWebSocketUrlForMode(mode = serviceMode) {
  if (!elements.server) return;
  elements.server.value = webSocketUrlForMode(elements.server.value, mode);
}

function translationDeviceForMode(mode = serviceMode) {
  return mode === "accurate" ? "cuda:1" : "cpu";
}

function clampCaptionFontSize(value, fallback) {
  if (value === null || value === undefined || String(value).trim() === "") return fallback;
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(40, Math.max(14, Math.round(number)));
}

function fullscreenCaptionFontSize(value) {
  return Math.round(value * FULLSCREEN_CAPTION_SCALE);
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
  rootStyle.setProperty("--fullscreen-source-font-size", `${fullscreenCaptionFontSize(normalized.sourceFontSize)}px`);
  rootStyle.setProperty("--source-font-color", normalized.sourceFontColor);
  rootStyle.setProperty("--translation-font-size", `${normalized.translationFontSize}px`);
  rootStyle.setProperty("--fullscreen-translation-font-size", `${fullscreenCaptionFontSize(normalized.translationFontSize)}px`);
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

function normalizeServiceMode(value) {
  return SERVICE_MODES.includes(value) ? value : DEFAULT_SERVICE_MODE;
}

function isTranslationServiceMode(mode = serviceMode) {
  return mode !== "transcription";
}

function oppositeLanguage(language) {
  return normalizeLanguage(language) === "zh" ? "en" : "zh";
}

function normalizeDirectionMode(value) {
  return value === "specified" ? "specified" : "auto";
}

function syncDirectionLanguages() {
  selectedSourceLanguage = normalizeLanguage(selectedSourceLanguage) || "zh";
  selectedTargetLanguageCode = normalizeLanguage(selectedTargetLanguageCode) || oppositeLanguage(selectedSourceLanguage);
  if (selectedSourceLanguage === selectedTargetLanguageCode) {
    selectedTargetLanguageCode = oppositeLanguage(selectedSourceLanguage);
  }
  if (elements.language) elements.language.value = selectedSourceLanguage;
  if (elements.translationTarget) elements.translationTarget.value = directionMode === "auto" ? "auto" : selectedTargetLanguageCode;
}

function persistServiceModeState() {
  window.localStorage.setItem("whisperlive_service_mode", serviceMode);
  window.localStorage.setItem("whisperlive_direction_mode", directionMode);
  window.localStorage.setItem("whisperlive_direction_source_language", selectedSourceLanguage);
  window.localStorage.setItem("whisperlive_direction_target_language", selectedTargetLanguageCode);
}

function serviceModeLabel(mode = serviceMode) {
  if (mode === "accurate") return "高精同传";
  if (mode === "conversation") return "对话翻译";
  if (mode === "transcription") return "语音识别";
  return "普通同传";
}

function fallbackTranslationModels() {
  return [
    { value: "helsinki_zh_en", label: "Helsinki 轻量实时", provider: "helsinki_zh_en", zh_en_model_path: "model/opus-mt-zh-en", en_zh_model_path: "model/opus-mt-en-zh", available: true },
    { value: "nllb_200_600m", label: "NLLB-200 600M 高质量", provider: "nllb_200_600m", nllb_model_path: "model/NLLB-200-600M", available: true },
  ];
}

function defaultTranslationProviderForMode(mode = serviceMode) {
  if (mode === "accurate") return "nllb_200_3_3b";
  if (mode === "standard") return "nllb_200_600m";
  return "helsinki_zh_en";
}

function translationModelMetadata(value = effectiveTranslationProvider()) {
  return translationModelByValue.get(value) || fallbackTranslationModels().find((model) => model.value === value) || null;
}

function availableTranslationModelValue(value) {
  if (translationModelByValue.has(value)) return value;
  return "";
}

function preferredTranslationProviderForMode(mode = serviceMode) {
  const preferred = defaultTranslationProviderForMode(mode);
  if (mode === "accurate") {
    return availableTranslationModelValue(preferred)
      || availableTranslationModelValue("nllb_200_distilled_1_3b")
      || availableTranslationModelValue("nllb_200_600m")
      || availableTranslationModelValue("helsinki_zh_en")
      || translationModels[translationModels.length - 1]?.value
      || preferred;
  }
  return availableTranslationModelValue(preferred)
    || availableTranslationModelValue("nllb_200_600m")
    || availableTranslationModelValue("helsinki_zh_en")
    || translationModels[0]?.value
    || preferred;
}

function effectiveTranslationProvider() {
  return elements.translationProvider?.value || preferredTranslationProviderForMode();
}

function renderTranslationModelOptions(models, preferredValue = "") {
  translationModels = (models || []).filter((model) => model && model.available !== false && model.value);
  translationModelByValue.clear();
  translationModels.forEach((model) => translationModelByValue.set(model.value, model));
  if (!elements.translationProvider) return;
  const selected = preferredValue || elements.translationProvider.value || preferredTranslationProviderForMode();
  elements.translationProvider.replaceChildren();
  if (!translationModels.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "未发现可用翻译模型";
    elements.translationProvider.appendChild(option);
    elements.translationProvider.disabled = true;
    return;
  }
  translationModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.value;
    option.textContent = model.label || model.value;
    elements.translationProvider.appendChild(option);
  });
  elements.translationProvider.value = translationModelByValue.has(selected) ? selected : preferredTranslationProviderForMode();
}

async function loadTranslationModels(preferredValue = "") {
  try {
    const response = await fetch(translationModelsUrl(), { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderTranslationModelOptions(data.models || [], preferredValue);
  } catch (error) {
    console.warn("Failed to load translation models, using fallback options", error);
    renderTranslationModelOptions(fallbackTranslationModels(), preferredValue);
  }
  if (elements.translationProvider && translationModels.length && !translationModelByValue.has(elements.translationProvider.value)) {
    elements.translationProvider.value = preferredTranslationProviderForMode();
  }
  updateTranslationControls();
}

function setTranslationEnabledForMode() {
  const enabled = isTranslationServiceMode();
  if (!elements.translationEnabled) return;
  elements.translationEnabled.checked = enabled;
  elements.translationEnabled.value = enabled ? "true" : "false";
}

function applyServiceModeDisplayDefaults() {
  if (serviceMode !== "transcription") return;
  singleLanguageMode = "source";
  if (elements.singleLanguage) elements.singleLanguage.value = singleLanguageMode;
  window.localStorage.setItem("whisperlive_single_language", singleLanguageMode);
  if (displayMode !== "single") setDisplayMode("single");
}

function setServiceMode(nextMode, persist = true) {
  serviceMode = normalizeServiceMode(nextMode);
  if (serviceMode === "conversation") directionMode = "auto";
  if (serviceMode === "transcription") directionMode = "specified";
  if (elements.translationProvider && translationModels.length) {
    elements.translationProvider.value = preferredTranslationProviderForMode(serviceMode);
  }
  syncWebSocketUrlForMode(serviceMode);
  syncDirectionLanguages();
  setTranslationEnabledForMode();
  applyServiceModeDisplayDefaults();
  updateTranslationControls();
  renderTranscriptViews();
  if (persist) persistServiceModeState();
}

function setDirectionMode(nextMode, persist = true) {
  if (serviceMode === "conversation") {
    directionMode = "auto";
  } else if (serviceMode === "transcription") {
    directionMode = "specified";
  } else {
    directionMode = normalizeDirectionMode(nextMode);
  }
  syncDirectionLanguages();
  updateTranslationControls();
  renderTranscriptViews();
  if (persist) persistServiceModeState();
}

function toggleSourceLanguage() {
  selectedSourceLanguage = oppositeLanguage(selectedSourceLanguage);
  selectedTargetLanguageCode = oppositeLanguage(selectedSourceLanguage);
  setDirectionMode(directionMode);
}

function toggleTargetLanguage() {
  selectedTargetLanguageCode = oppositeLanguage(selectedTargetLanguageCode);
  selectedSourceLanguage = oppositeLanguage(selectedTargetLanguageCode);
  setDirectionMode("specified");
}

function serviceModeHint() {
  if (serviceMode === "transcription") return `当前模式：语音识别 · ${languageLabel(selectedSourceLanguage)}原文`;
  if (serviceMode === "conversation") return "当前模式：对话翻译 · 自适应中文/English";
  const provider = serviceMode === "accurate" ? "高精翻译" : "标准翻译";
  return `当前模式：${serviceModeLabel()} · ${provider} · ${translationModeLabel()}`;
}

function updateModeToolbar(forceConnectionLocked = false) {
  elements.serviceModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.serviceMode === serviceMode);
    button.disabled = Boolean(forceConnectionLocked);
  });
  document.body.dataset.serviceMode = serviceMode;
  document.body.classList.toggle("translation-disabled", !isTranslationServiceMode());

  if (elements.sourceLanguageButton) {
    elements.sourceLanguageButton.textContent = languageLabel(selectedSourceLanguage);
    elements.sourceLanguageButton.disabled = Boolean(forceConnectionLocked || serviceMode === "conversation");
  }
  if (elements.targetLanguageButton) {
    elements.targetLanguageButton.textContent = serviceMode === "transcription" ? "无翻译" : languageLabel(selectedTargetLanguageCode);
    elements.targetLanguageButton.disabled = Boolean(forceConnectionLocked || serviceMode === "conversation" || serviceMode === "transcription");
  }
  if (elements.directionToggleButton) {
    elements.directionToggleButton.textContent = serviceMode === "conversation" ? "自适应" : (directionMode === "auto" && serviceMode !== "transcription" ? "↔" : "→");
    elements.directionToggleButton.disabled = Boolean(forceConnectionLocked || serviceMode === "conversation" || serviceMode === "transcription");
    elements.directionToggleButton.setAttribute("aria-label", serviceMode === "conversation" ? "当前对话翻译自适应语言" : (directionMode === "auto" ? "当前自动互译，点击改为指定方向" : "当前指定方向，点击改为自动互译"));
  }
}

async function enterCaptionFullscreen() {
  closeAllDrawers();
  try {
    if (document.documentElement.requestFullscreen) {
      await document.documentElement.requestFullscreen();
      immersiveFullscreenFallback = false;
    } else {
      immersiveFullscreenFallback = true;
    }
  } catch (_error) {
    immersiveFullscreenFallback = true;
  }
  document.body.classList.add("is-caption-fullscreen");
  updateFullscreenButton();
}

async function exitCaptionFullscreen() {
  immersiveFullscreenFallback = false;
  if (document.fullscreenElement && document.exitFullscreen) {
    await document.exitFullscreen().catch(() => {});
  }
  document.body.classList.remove("is-caption-fullscreen");
  updateFullscreenButton();
}

function updateFullscreenButton() {
  if (!elements.fullscreenButton) return;
  const active = document.fullscreenElement || immersiveFullscreenFallback || document.body.classList.contains("is-caption-fullscreen");
  elements.fullscreenButton.textContent = active ? "退出全屏" : "全屏";
  elements.fullscreenButton.setAttribute("aria-pressed", active ? "true" : "false");
}

function initializeDefaults() {
  const legacyDefault = "ws://localhost:9090";
  if (!elements.server.value || elements.server.value === legacyDefault) {
    elements.server.value = defaultWebSocketUrl();
  }
  clientInstanceId = getClientInstanceId();
  const savedMeeting = window.localStorage.getItem("whisperlive_meeting_name");
  const savedServer = window.localStorage.getItem("whisperlive_server_url");
  const savedTranslationProvider = window.localStorage.getItem("whisperlive_translation_provider");
  const savedFaceToFaceEnabled = window.localStorage.getItem("whisperlive_face_to_face_enabled");
  const savedFaceToFaceMode = window.localStorage.getItem("whisperlive_face_to_face_mode");
  const savedTranslationTarget = window.localStorage.getItem("whisperlive_translation_target");
  const savedServiceMode = window.localStorage.getItem("whisperlive_service_mode");
  const savedDirectionMode = window.localStorage.getItem("whisperlive_direction_mode");
  const savedDirectionSource = window.localStorage.getItem("whisperlive_direction_source_language");
  const savedDirectionTarget = window.localStorage.getItem("whisperlive_direction_target_language");
  const savedDiarizationEnabled = window.localStorage.getItem("whisperlive_diarization_enabled");
  displayMode = window.localStorage.getItem("whisperlive_display_mode") || DEFAULT_DISPLAY_MODE;
  singleLanguageMode = window.localStorage.getItem("whisperlive_single_language") || "translation";
  if (savedMeeting && !elements.meetingName.value) elements.meetingName.value = savedMeeting;
  if (savedServer) elements.server.value = savedServer;
  if (elements.translationEnabled) elements.translationEnabled.checked = true;
  window.localStorage.removeItem("whisperlive_translation_enabled");
  renderTranslationModelOptions(fallbackTranslationModels(), savedTranslationProvider || defaultTranslationProviderForMode(serviceMode));
  if (savedTranslationProvider && elements.translationProvider) elements.translationProvider.value = savedTranslationProvider;
  if (elements.faceToFaceMode) {
    elements.faceToFaceMode.value = normalizeTranslationMode(savedFaceToFaceMode, savedFaceToFaceEnabled);
  }
  window.localStorage.removeItem("whisperlive_face_to_face_enabled");
  if (savedTranslationTarget && elements.translationTarget) elements.translationTarget.value = savedTranslationTarget;
  serviceMode = normalizeServiceMode(savedServiceMode);
  directionMode = normalizeDirectionMode(savedDirectionMode || (faceToFaceMode() === "specified" ? "specified" : "auto"));
  selectedSourceLanguage = normalizeLanguage(savedDirectionSource || elements.language?.value) || "zh";
  selectedTargetLanguageCode = normalizeLanguage(savedDirectionTarget || selectedTargetLanguage()) || oppositeLanguage(selectedSourceLanguage);
  if (savedDiarizationEnabled !== null) elements.diarizationEnabled.checked = savedDiarizationEnabled === "true";
  elements.displayMode.value = displayMode;
  elements.singleLanguage.value = singleLanguageMode;
  applyCaptionStyle(readCaptionStyle());
  setServiceMode(serviceMode, false);
  setDisplayMode(displayMode);
  updateMeetingTitle();
  updateHotwordStatus("未上传热词");
  applyBackendLanguageDefault().catch(() => applyBackendLanguageFallback());
  loadTranslationModels(defaultTranslationProviderForMode(serviceMode)).catch(() => {});
  loadSummarySessions().catch(() => {});
  loadSummaryTemplates().catch(() => {});
}

function normalizeHotwordLine(rawLine) {
  let line = String(rawLine || "").trim();
  if (!line || line.startsWith("#") || line.startsWith("```")) return "";
  line = line.replace(/^[-*+]\s+\[[ xX]\]\s+/, "");
  line = line.replace(/^[-*+]\s+/, "");
  line = line.replace(/^\d+[.)、．）]\s+/, "");
  return line.trim();
}

function parseHotwordText(text) {
  const hotwords = [];
  const translationGlossary = {};
  let translationCount = 0;
  String(text || "").split(/\r?\n/).forEach((rawLine) => {
    const line = normalizeHotwordLine(rawLine);
    if (!line) return;
    if (!line.includes("=>")) {
      hotwords.push(line);
      return;
    }
    const separator = line.indexOf("=>");
    const source = line.slice(0, separator).trim();
    const target = line.slice(separator + 2).trim();
    if (!source || !target) return;
    translationGlossary[source] = target;
    translationCount += 1;
  });
  return { hotwords, translationCount, translationGlossary };
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
  const title = elements.meetingName.value.trim() || "实时同传";
  elements.meetingTitle.textContent = title;
  if (elements.drawerMeetingName) elements.drawerMeetingName.textContent = title;
}

function updateDrawerTranslationStatus() {
  if (!elements.drawerTranslationStatus || !elements.translationEnabled) return;
  if (!elements.translationEnabled.checked) {
    elements.drawerTranslationStatus.textContent = "已关闭";
    return;
  }
  elements.drawerTranslationStatus.textContent = serviceMode === "transcription" ? "已关闭" : serviceModeLabel();
}

function setToolStatus(text, state = "") {
  if (!elements.toolStatus) return;
  elements.toolStatus.textContent = text;
  elements.toolStatus.className = `tool-status ${state}`.trim();
}

function showDrawerBackdrop() {
  elements.settingsBackdrop.hidden = false;
  requestAnimationFrame(() => elements.settingsBackdrop.classList.add("visible"));
  document.body.classList.add("drawer-open");
}

function hideDrawerBackdropIfIdle() {
  const settingsOpen = elements.settingsDrawer && elements.settingsDrawer.classList.contains("open");
  const summaryOpen = elements.summaryDrawer && elements.summaryDrawer.classList.contains("open");
  if (settingsOpen || summaryOpen) return;
  elements.settingsBackdrop.classList.remove("visible");
  document.body.classList.remove("drawer-open");
  window.setTimeout(() => {
    const stillIdle = !elements.settingsDrawer.classList.contains("open") && !(elements.summaryDrawer && elements.summaryDrawer.classList.contains("open"));
    if (stillIdle) elements.settingsBackdrop.hidden = true;
  }, 240);
}

function closeSettings() {
  elements.settingsDrawer.classList.remove("open");
  elements.settingsDrawer.setAttribute("aria-hidden", "true");
  elements.settingsButton.setAttribute("aria-expanded", "false");
  hideDrawerBackdropIfIdle();
}

function closeSummaryDrawer() {
  if (!elements.summaryDrawer) return;
  elements.summaryDrawer.classList.remove("open");
  elements.summaryDrawer.setAttribute("aria-hidden", "true");
  if (elements.summaryQuick) elements.summaryQuick.setAttribute("aria-expanded", "false");
  hideDrawerBackdropIfIdle();
}

function closeAllDrawers() {
  closeSettings();
  closeSummaryDrawer();
}

function openSettings() {
  closeSummaryDrawer();
  showDrawerBackdrop();
  elements.settingsDrawer.classList.add("open");
  elements.settingsDrawer.setAttribute("aria-hidden", "false");
  elements.settingsButton.setAttribute("aria-expanded", "true");
}

function openSummaryDrawer() {
  closeSettings();
  if (!elements.summaryDrawer) return;
  showDrawerBackdrop();
  elements.summaryDrawer.classList.add("open");
  elements.summaryDrawer.setAttribute("aria-hidden", "false");
  if (elements.summaryQuick) elements.summaryQuick.setAttribute("aria-expanded", "true");
}

function updateTranslationControls(forceConnectionLocked = false) {
  if (serviceMode === "conversation") directionMode = "auto";
  if (serviceMode === "transcription") directionMode = "specified";
  syncDirectionLanguages();
  setTranslationEnabledForMode();

  const autoDetectMode = faceToFaceEnabled();
  if (elements.faceToFaceMode) {
    elements.faceToFaceMode.value = autoDetectMode ? "interpretation" : "specified";
    elements.faceToFaceMode.disabled = Boolean(forceConnectionLocked || serviceMode === "conversation" || serviceMode === "transcription");
  }
  if (elements.translationProvider) {
    elements.translationProvider.disabled = Boolean(forceConnectionLocked || serviceMode === "transcription" || !translationModels.length);
  }
  if (elements.language) {
    elements.language.disabled = Boolean(forceConnectionLocked || serviceMode === "conversation");
  }
  if (elements.translationTarget) {
    elements.translationTarget.disabled = Boolean(forceConnectionLocked || serviceMode === "conversation" || serviceMode === "transcription" || directionMode === "auto");
  }
  if (elements.translationProviderField) elements.translationProviderField.hidden = serviceMode === "transcription";
  if (elements.faceToFaceModeField) elements.faceToFaceModeField.hidden = serviceMode === "transcription";
  if (elements.translationTargetField) elements.translationTargetField.hidden = serviceMode === "transcription";
  if (elements.languageField) elements.languageField.hidden = serviceMode === "conversation";
  if (elements.translationModeHint) {
    elements.translationModeHint.textContent = serviceModeHint();
  }
  updateModeToolbar(forceConnectionLocked);
  updateDrawerTranslationStatus();
}

function normalizeLanguage(value) {
  const language = String(value || "").toLowerCase();
  if (language === "zh" || language.startsWith("zh-")) return "zh";
  if (language === "en" || language.startsWith("en-")) return "en";
  return language || null;
}

function hotwordApiBaseUrl() {
  const serverUrl = elements.server.value.trim();
  const toAdminBase = (url) => url
    .replace(/\/ws-gpu0\/?$/, "/admin-gpu0")
    .replace(/\/ws-gpu1\/?$/, "/admin-gpu1")
    .replace(/\/ws-standard\/?$/, "")
    .replace(/\/ws-accurate\/?$/, "")
    .replace(/\/ws\/?$/, "");
  if (serverUrl.startsWith("wss://")) return toAdminBase(serverUrl.replace(/^wss:/, "https:"));
  if (serverUrl.startsWith("ws://")) return toAdminBase(serverUrl.replace(/^ws:/, "http:"));
  return window.location.origin === "null" ? "http://localhost:9093" : window.location.origin;
}

function adminClientsUrl() {
  return `${hotwordApiBaseUrl().replace(/\/$/, "")}/admin/clients`;
}

function translationModelsUrl() {
  return `${hotwordApiBaseUrl().replace(/\/$/, "")}/admin/translation-models`;
}

function normalizeBackend(value) {
  const backend = String(value || "").toLowerCase();
  if (backend.includes("funasr")) return "funasr";
  if (backend.includes("whisper")) return "faster_whisper";
  return DEFAULT_BACKEND;
}

function isWhisperBackend(backend) {
  return normalizeBackend(backend) === "faster_whisper";
}

function defaultLanguageForBackend(backend) {
  return normalizeBackend(backend) === "funasr" ? "zh" : "en";
}

function lockedTranslationTargetForBackend(backend, language) {
  return normalizeLanguage(language) === "en" ? "zh" : "en";
}

function normalizeTranslationMode(value, legacyFaceToFaceEnabled = null) {
  const mode = String(value || "");
  if (legacyFaceToFaceEnabled === "false") return "specified";
  if (mode === "interpretation" || mode === "auto") return "interpretation";
  if (mode === "specified" || mode === "standard" || mode === "meeting_record") return "specified";
  return DEFAULT_TRANSLATION_MODE;
}

function faceToFaceMode() {
  return normalizeTranslationMode(elements.faceToFaceMode?.value);
}

function faceToFaceEnabled() {
  return isTranslationServiceMode() && directionMode === "auto";
}

function selectedTargetLanguage() {
  if (serviceMode === "transcription") return "auto";
  if (directionMode === "auto") return "auto";
  return normalizeLanguage(selectedTargetLanguageCode) || oppositeLanguage(selectedSourceLanguage);
}

function selectedTranslationProviderConfig() {
  const value = effectiveTranslationProvider();
  const model = translationModelMetadata(value);
  if (!model) return { provider: value || "helsinki_zh_en", nllbModelPath: "model/NLLB-200-600M" };
  return {
    provider: model.provider || value,
    nllbModelPath: model.nllb_model_path || "model/NLLB-200-600M",
    zhEnModelPath: model.zh_en_model_path || "model/opus-mt-zh-en",
    enZhModelPath: model.en_zh_model_path || "model/opus-mt-en-zh",
  };
}

function syncSpecifiedTranslationTarget() {
  if (!elements.translationTarget || faceToFaceEnabled()) return;
  const source = normalizeLanguage(elements.language?.value) || defaultLanguageForBackend(currentServerBackend || DEFAULT_BACKEND);
  let target = normalizeLanguage(elements.translationTarget.value);
  if (!target || target === source) {
    target = lockedTranslationTargetForBackend(currentServerBackend || DEFAULT_BACKEND, source);
    elements.translationTarget.value = target;
  }
}

function selectedFaceToFaceTarget() {
  if (faceToFaceEnabled()) return "auto";
  if (serviceMode === "transcription") return "auto";
  syncDirectionLanguages();
  return normalizeLanguage(selectedTargetLanguageCode) || lockedTranslationTargetForBackend(currentServerBackend || DEFAULT_BACKEND, selectedSourceLanguage);
}

function languageLabel(language) {
  if (language === "auto") return "自动互译";
  return normalizeLanguage(language) === "zh" ? "中文" : "English";
}

function translationModeLabel() {
  if (serviceMode === "transcription") return `识别语言：${languageLabel(selectedSourceLanguage)}`;
  if (faceToFaceEnabled()) return "自动识别语言：中文 ↔ English";
  const source = normalizeLanguage(selectedSourceLanguage) || defaultLanguageForBackend(currentServerBackend || DEFAULT_BACKEND);
  const target = selectedFaceToFaceTarget();
  return `指定翻译：${languageLabel(source)} → ${languageLabel(target)}`;
}

function applyBackendLanguage(backend) {
  currentServerBackend = normalizeBackend(backend);
  if (!selectedSourceLanguage) selectedSourceLanguage = defaultLanguageForBackend(currentServerBackend);
  syncDirectionLanguages();
  updateTranslationControls();
}

function applyBackendLanguageFallback() {
  applyBackendLanguage(DEFAULT_BACKEND);
}

async function applyBackendLanguageDefault() {
  const response = await fetch(adminClientsUrl(), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  const backend = data.server_backend || (data.clients && data.clients[0] && data.clients[0].backend) || DEFAULT_BACKEND;
  applyBackendLanguage(backend);
}

function meetingLogApiUrl() {
  return `${hotwordApiBaseUrl().replace(/\/$/, "")}/admin/meeting-logs`;
}

function meetingLogDownloadUrl(sessionId, format = "md", layout = "sections") {
  const params = new URLSearchParams({ format });
  if (layout && layout !== "sections") params.set("layout", layout);
  return `${meetingLogApiUrl()}/${encodeURIComponent(sessionId)}?${params.toString()}`;
}

function meetingLogFinishUrl(sessionId) {
  return `${meetingLogApiUrl()}/${encodeURIComponent(sessionId)}/finish`;
}

function summaryApiUrl(sessionId) {
  return `${meetingLogApiUrl()}/${encodeURIComponent(sessionId)}/summary`;
}

function summaryInfoUrl(sessionId) {
  return `${summaryApiUrl(sessionId)}/info`;
}

function transcriptApiUrl(sessionId, segmentId = "") {
  const base = `${meetingLogApiUrl()}/${encodeURIComponent(sessionId)}/transcript`;
  return segmentId ? `${base}/${encodeURIComponent(segmentId)}` : base;
}

function speakersApiUrl(sessionId, speakerId = "") {
  const base = `${meetingLogApiUrl()}/${encodeURIComponent(sessionId)}/speakers`;
  return speakerId ? `${base}/${encodeURIComponent(speakerId)}` : base;
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

function selectedCustomSummaryTemplateId() {
  const value = (elements.summaryTemplate && elements.summaryTemplate.value) || "";
  return value.startsWith("custom:") ? value.slice(7) : "";
}

function updateSummaryTemplateDeleteButton() {
  if (!elements.deleteSummaryTemplate) return;
  const templateId = selectedCustomSummaryTemplateId();
  elements.deleteSummaryTemplate.hidden = false;
  elements.deleteSummaryTemplate.disabled = !templateId;
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
  updateSummaryTemplateDeleteButton();
}

function createSummaryTemplateField(section) {
  const usedKeys = new Set((summaryTemplateDraft.fields || []).map((field) => field.key));
  let suffix = (section.line_index || 0) + 1;
  let key = `field_${suffix}`;
  while (usedKeys.has(key)) {
    suffix += 1;
    key = `field_${suffix}`;
  }
  return {
    key,
    label: section.heading,
    heading: section.heading,
    type: /事项|要点|问题|风险|结论|议题/.test(section.heading) ? "list" : "text",
    description: `根据会议原文填写“${section.heading}”`,
    columns: [],
    required: false,
    metadata_enrichment: false,
  };
}

function sortSummaryTemplateFields() {
  const order = new Map((summaryTemplateDraft.sections || []).map((section, index) => [section.heading, index]));
  summaryTemplateDraft.fields.sort((left, right) =>
    (order.get(left.heading) ?? Number.MAX_SAFE_INTEGER) - (order.get(right.heading) ?? Number.MAX_SAFE_INTEGER));
}

function renderSummaryTemplateFields() {
  if (!elements.summaryTemplateFields || !summaryTemplateDraft) return;
  elements.summaryTemplateFields.innerHTML = "";
  const sections = (summaryTemplateDraft.sections || []).length
    ? summaryTemplateDraft.sections
    : (summaryTemplateDraft.fields || []).map((field, index) => ({
      heading: field.heading,
      level: 2,
      line_index: index,
      role: "field",
    }));
  const fieldsByHeading = new Map((summaryTemplateDraft.fields || []).map((field) => [field.heading, field]));

  sections.forEach((section) => {
    const field = fieldsByHeading.get(section.heading);
    const card = document.createElement("div");
    card.className = `summary-template-field${field ? "" : " is-container"}`;

    const headingRow = document.createElement("div");
    headingRow.className = "summary-template-section-heading";
    const heading = document.createElement("strong");
    heading.textContent = section.heading;
    const level = document.createElement("span");
    level.className = "summary-template-level";
    level.textContent = `${section.level || 2} 级${field ? "内容字段" : "结构标题"}`;
    headingRow.append(heading, level);
    card.appendChild(headingRow);

    const generationOptions = document.createElement("div");
    generationOptions.className = "summary-template-field-options";
    const generationLabel = document.createElement("label");
    const generatesContent = document.createElement("input");
    generatesContent.type = "checkbox";
    generatesContent.checked = Boolean(field);
    generatesContent.addEventListener("change", () => {
      const fieldIndex = summaryTemplateDraft.fields.findIndex((item) => item.heading === section.heading);
      if (generatesContent.checked && fieldIndex < 0) {
        summaryTemplateDraft.fields.push(createSummaryTemplateField(section));
        section.role = "field";
        sortSummaryTemplateFields();
      } else if (!generatesContent.checked && fieldIndex >= 0) {
        summaryTemplateDraft.fields.splice(fieldIndex, 1);
        section.role = "container";
      }
      renderSummaryTemplateFields();
    });
    generationLabel.append(generatesContent, document.createTextNode("生成该标题内容"));
    generationOptions.appendChild(generationLabel);
    card.appendChild(generationOptions);

    if (!field) {
      const hint = document.createElement("p");
      hint.className = "summary-template-container-hint";
      hint.textContent = "仅作为分组标题保留，不发送给总结模型。";
      card.appendChild(hint);
      elements.summaryTemplateFields.appendChild(card);
      return;
    }

    const grid = document.createElement("div");
    grid.className = "summary-template-field-grid";
    const label = document.createElement("input");
    label.value = field.label || field.heading;
    label.placeholder = "字段名称";
    label.addEventListener("input", () => { field.label = label.value; });
    const key = document.createElement("input");
    key.value = field.key || createSummaryTemplateField(section).key;
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
    const options = document.createElement("div");
    options.className = "summary-template-field-options";
    const requiredLabel = document.createElement("label");
    const required = document.createElement("input");
    required.type = "checkbox";
    required.checked = Boolean(field.required);
    required.addEventListener("change", () => { field.required = required.checked; });
    requiredLabel.append(required, document.createTextNode("必填字段"));
    options.appendChild(requiredLabel);
    if (field.type === "text") {
      const proseLabel = document.createElement("label");
      const prose = document.createElement("input");
      prose.type = "checkbox";
      prose.checked = field.output_style === "prose";
      prose.addEventListener("change", () => { field.output_style = prose.checked ? "prose" : null; });
      proseLabel.append(prose, document.createTextNode("自然段正文（无编号）"));
      options.appendChild(proseLabel);
      const residualLabel = document.createElement("label");
      const residual = document.createElement("input");
      residual.type = "checkbox";
      residual.checked = Boolean(field.residual);
      residual.addEventListener("change", () => { field.residual = residual.checked; });
      residualLabel.append(residual, document.createTextNode("仅保留未覆盖事项"));
      options.appendChild(residualLabel);
    } else {
      field.output_style = null;
      field.residual = false;
    }
    if (field.type === "table") {
      const metadataLabel = document.createElement("label");
      const metadata = document.createElement("input");
      metadata.type = "checkbox";
      metadata.checked = Boolean(field.metadata_enrichment);
      metadata.addEventListener("change", () => { field.metadata_enrichment = metadata.checked; });
      metadataLabel.append(metadata, document.createTextNode("补充会议信息"));
      options.appendChild(metadataLabel);
    } else {
      field.metadata_enrichment = false;
    }
    card.appendChild(options);
    if ((field.derive_from_fields || []).length) {
      const derivedHint = document.createElement("p");
      derivedHint.className = "summary-template-container-hint";
      derivedHint.textContent = "该字段将根据有内容的固定专题标题自动生成，不调用 LLM。";
      card.appendChild(derivedHint);
    }
    elements.summaryTemplateFields.appendChild(card);
  });
}

async function analyzeSummaryTemplate() {
  const file = elements.summaryTemplateFile && elements.summaryTemplateFile.files[0];
  if (!file) throw new Error("请先选择模板文件");
  if (!/\.(md|docx)$/i.test(file.name)) throw new Error("只支持上传 .md 或 .docx 模板文件");
  const form = new FormData(); form.append("file", file);
  const response = await fetch(summaryTemplateApiUrl("/analyze"), { method: "POST", body: form });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  summaryTemplateDraft = result;
  elements.summaryTemplateName.value = file.name.replace(/\.(md|docx)$/i, "");
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

async function deleteSelectedSummaryTemplate() {
  const templateId = selectedCustomSummaryTemplateId();
  if (!templateId) throw new Error("请选择自定义总结模板");
  const template = customSummaryTemplates.find((item) => item.id === templateId) || {};
  const templateName = template.name || templateId;
  if (!window.confirm(`确认删除自定义模板“${templateName}”？\n历史总结文件不受影响，但之后不能再选择这个模板生成新总结。`)) {
    return;
  }
  const response = await fetch(summaryTemplateApiUrl(`/${encodeURIComponent(templateId)}`), { method: "DELETE" });
  const result = await response.json().catch(() => ({}));
  if (!response.ok || !result.deleted) throw new Error(result.error || `HTTP ${response.status}`);
  elements.summaryTemplate.value = "auto";
  await loadSummaryTemplates();
  setToolStatus("自定义总结模板已删除。", "success");
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
  if (!transcriptEditorData) renderTranscriptEditor();
  await loadSummaryInfo(nextId);
}


function setTranscriptEditorStatus(text, state = "") {
  if (!elements.transcriptEditorStatus) return;
  elements.transcriptEditorStatus.textContent = text;
  elements.transcriptEditorStatus.className = `transcript-editor-status${state ? ` ${state}` : ""}`;
}

function transcriptEditorIdleMessage() {
  return selectedSummarySessionId
    ? "已选择会议，点击“加载校对内容”。"
    : "请先在上方选择会议日志。";
}

async function transcriptRequest(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 409 && selectedSummarySessionId) await loadTranscriptEditor(selectedSummarySessionId).catch(() => {});
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  transcriptEditorData = data;
  renderTranscriptEditor();
  return data;
}

function speakerSegmentCounts() {
  const counts = new Map();
  let unassigned = 0;
  (transcriptEditorData?.segments || []).forEach((segment) => {
    const speakerId = String(segment.speaker_id || "").trim();
    if (!speakerId) {
      unassigned += 1;
      return;
    }
    counts.set(speakerId, (counts.get(speakerId) || 0) + 1);
  });
  return { counts, unassigned };
}

function renderSpeakerStats() {
  if (!elements.speakerStats) return;
  elements.speakerStats.replaceChildren();
  if (!transcriptEditorData) return;
  const speakers = transcriptEditorData.speakers || [];
  const { counts, unassigned } = speakerSegmentCounts();
  const items = [`未分配 ${unassigned} 段`];
  speakers.forEach((speaker) => {
    items.push(`${speaker.name || speaker.speaker_id} ${counts.get(speaker.speaker_id) || 0} 段`);
  });
  items.forEach((item) => {
    const badge = document.createElement("span");
    badge.textContent = item;
    elements.speakerStats.appendChild(badge);
  });
}

function renderSpeakerRenameGuide() {
  if (!elements.speakerRenameGuide) return;
  elements.speakerRenameGuide.replaceChildren();
  if (!transcriptEditorData) return;
  const speakers = transcriptEditorData.speakers || [];
  const { counts } = speakerSegmentCounts();
  const heading = document.createElement("div");
  heading.className = "speaker-guide-heading";
  const title = document.createElement("strong");
  title.textContent = "快速改名";
  const hint = document.createElement("span");
  hint.textContent = speakers.length ? "把 Speaker 1 / Speaker 2 改成真实姓名。" : "当前日志没有自动说话人，可在高级管理中新增。";
  heading.append(title, hint);
  elements.speakerRenameGuide.appendChild(heading);
  if (!speakers.length) return;

  speakers.forEach((speaker) => {
    const row = document.createElement("div");
    row.className = "speaker-guide-row";
    row.dataset.speakerId = speaker.speaker_id;
    const current = document.createElement("div");
    current.className = "speaker-guide-current";
    const currentName = document.createElement("strong");
    currentName.textContent = speaker.name || speaker.speaker_id;
    const currentCount = document.createElement("small");
    currentCount.textContent = `${counts.get(speaker.speaker_id) || 0} 段`;
    current.append(currentName, currentCount);
    const input = document.createElement("input");
    input.value = speaker.name || "";
    input.maxLength = 80;
    input.placeholder = "填写真实姓名";
    const save = document.createElement("button");
    save.type = "button";
    save.className = "secondary-button";
    save.dataset.action = "rename-speaker-guide";
    save.textContent = "保存";
    row.append(current, input, save);
    elements.speakerRenameGuide.appendChild(row);
  });
}

function renderSpeakerManager() {
  if (!elements.speakerManagerList) return;
  elements.speakerManagerList.replaceChildren();
  const speakers = transcriptEditorData?.speakers || [];
  speakers.forEach((speaker) => {
    const row = document.createElement("div");
    row.className = "speaker-manager-row";
    row.dataset.speakerId = speaker.speaker_id;
    const input = document.createElement("input");
    input.value = speaker.name || speaker.speaker_id;
    input.maxLength = 80;
    const rename = document.createElement("button");
    rename.type = "button"; rename.className = "ghost-button"; rename.dataset.action = "rename-speaker"; rename.textContent = "改名";
    const target = document.createElement("select");
    target.innerHTML = '<option value="">合并到…</option>';
    speakers.filter((item) => item.speaker_id !== speaker.speaker_id).forEach((item) => {
      const option = document.createElement("option");
      option.value = item.speaker_id; option.textContent = item.name || item.speaker_id; target.appendChild(option);
    });
    const merge = document.createElement("button");
    merge.type = "button"; merge.className = "ghost-button"; merge.dataset.action = "merge-speaker"; merge.textContent = "合并";
    merge.disabled = speakers.length < 2;
    row.append(input, rename, target, merge);
    elements.speakerManagerList.appendChild(row);
  });
}

function renderTranscriptEditor() {
  if (!elements.transcriptEditorRows) return;
  elements.transcriptEditorRows.replaceChildren();
  if (!transcriptEditorData) {
    renderSpeakerStats();
    renderSpeakerRenameGuide();
    renderSpeakerManager();
    setTranscriptEditorStatus(transcriptEditorIdleMessage());
    return;
  }
  renderSpeakerStats();
  renderSpeakerRenameGuide();
  renderSpeakerManager();
  const notices = [`修订版本 ${Number(transcriptEditorData.transcript_revision || 0)}`];
  if (transcriptEditorData.translation_stale) notices.push("译文已过期");
  if (transcriptEditorData.summary_stale) notices.push("总结已过期，请重新生成");
  setTranscriptEditorStatus(notices.join(" · "), transcriptEditorData.summary_stale ? "warning" : "ready");
  const speakers = transcriptEditorData.speakers || [];
  (transcriptEditorData.segments || []).forEach((segment) => {
    const row = document.createElement("article");
    row.className = "transcript-editor-row"; row.dataset.segmentId = segment.segment_id;
    const meta = document.createElement("div");
    meta.className = "transcript-editor-meta"; meta.textContent = `${segment.start || "0"} - ${segment.end || "0"}`;
    const speaker = document.createElement("select");
    speaker.innerHTML = '<option value="">未分配说话人</option>';
    speakers.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.speaker_id; option.textContent = item.name || item.speaker_id; speaker.appendChild(option);
    });
    speaker.value = segment.speaker_id || "";
    const text = document.createElement("textarea");
    text.rows = 3; text.maxLength = 10000; text.value = segment.text || "";
    const actions = document.createElement("div");
    actions.className = "transcript-editor-actions";
    const restore = document.createElement("button");
    restore.type = "button"; restore.className = "ghost-button"; restore.dataset.action = "restore-segment"; restore.textContent = "恢复原文";
    restore.disabled = (segment.text || "") === (segment.original_text || "");
    const save = document.createElement("button");
    save.type = "button"; save.className = "secondary-button"; save.dataset.action = "save-segment"; save.textContent = "保存";
    actions.append(restore, save); row.append(meta, speaker, text, actions); elements.transcriptEditorRows.appendChild(row);
  });
}

async function loadTranscriptEditor(sessionId = selectedSummarySessionId) {
  if (!sessionId) throw new Error("请先在上方选择会议日志");
  setTranscriptEditorStatus("正在加载校对内容…");
  const response = await fetch(transcriptApiUrl(sessionId), { cache: "no-store" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  transcriptEditorData = data;
  renderTranscriptEditor();
}

async function saveTranscriptEditorRow(row, restoreOriginal = false) {
  const segmentId = row.dataset.segmentId;
  const segment = (transcriptEditorData?.segments || []).find((item) => item.segment_id === segmentId);
  const textarea = row.querySelector("textarea");
  if (restoreOriginal && segment) textarea.value = segment.original_text || "";
  setTranscriptEditorStatus("正在保存校对…");
  await transcriptRequest(transcriptApiUrl(selectedSummarySessionId, segmentId), {
    method: "PATCH",
    body: JSON.stringify({
      text: textarea.value,
      speaker_id: row.querySelector("select").value || null,
      expected_revision: transcriptEditorData.transcript_revision,
    }),
  });
  await loadSummaryInfo(selectedSummarySessionId);
}

async function addTranscriptSpeaker() {
  const name = elements.newSpeakerName.value.trim();
  if (!name) throw new Error("请输入说话人姓名");
  await transcriptRequest(speakersApiUrl(selectedSummarySessionId), {
    method: "POST",
    body: JSON.stringify({ name, expected_revision: transcriptEditorData.transcript_revision }),
  });
  elements.newSpeakerName.value = "";
}

async function renameTranscriptSpeaker(row) {
  await transcriptRequest(speakersApiUrl(selectedSummarySessionId, row.dataset.speakerId), {
    method: "PATCH",
    body: JSON.stringify({ name: row.querySelector("input").value, expected_revision: transcriptEditorData.transcript_revision }),
  });
  await loadSummaryInfo(selectedSummarySessionId);
}

async function renameTranscriptSpeakerFromGuide(row) {
  const name = row.querySelector("input").value.trim();
  if (!name) throw new Error("请输入真实姓名");
  setTranscriptEditorStatus("正在保存说话人名称…");
  await transcriptRequest(speakersApiUrl(selectedSummarySessionId, row.dataset.speakerId), {
    method: "PATCH",
    body: JSON.stringify({ name, expected_revision: transcriptEditorData.transcript_revision }),
  });
  await loadSummaryInfo(selectedSummarySessionId);
}

async function mergeTranscriptSpeaker(row) {
  const targetId = row.querySelector("select").value;
  if (!targetId) throw new Error("请选择要合并到的说话人");
  await transcriptRequest(`${speakersApiUrl(selectedSummarySessionId)}/merge`, {
    method: "POST",
    body: JSON.stringify({
      source_speaker_id: row.dataset.speakerId,
      target_speaker_id: targetId,
      expected_revision: transcriptEditorData.transcript_revision,
    }),
  });
  await loadSummaryInfo(selectedSummarySessionId);
}


async function readHotwordFileText(file) {
  if (!file) return { text: "", filename: "" };
  if (/\.(txt|md)$/i.test(file.name)) {
    return { text: await file.text(), filename: file.name };
  }
  throw new Error("只支持上传 .txt 或 .md 热词文件");
}

async function loadUploadedHotwords(file) {
  const { text, filename } = await readHotwordFileText(file);
  const parsed = parseHotwordText(text);
  lockedHotwords = {
    hotwords: parsed.hotwords.join(" "),
    filename,
    count: parsed.hotwords.length,
    translationCount: parsed.translationCount,
    translationGlossary: parsed.translationGlossary,
  };
  updateHotwordStatus(
    filename
      ? `${filename} · ${lockedHotwords.count} 个识别热词 · ${lockedHotwords.translationCount} 条固定翻译`
      : "未上传热词"
  );
}

function clearUploadedHotwords() {
  lockedHotwords = { hotwords: "", filename: "", count: 0, translationCount: 0, translationGlossary: {} };
  if (elements.hotwordFile) elements.hotwordFile.value = "";
  updateHotwordStatus("未上传热词");
}

function setStatus(text, state = "idle") {
  elements.status.textContent = text;
  elements.status.className = `status ${state}`;
  if (elements.drawerConnectionStatus) elements.drawerConnectionStatus.textContent = text;
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

async function exportMeetingLog(format = "md", layout = "sections") {
  if (!currentSessionId) throw new Error("当前没有可导出的后端日志 session");
  const normalizedFormat = format === "docx" ? "docx" : "md";
  const normalizedLayout = layout === "interleaved" ? "interleaved" : "sections";
  const response = await fetch(meetingLogDownloadUrl(currentSessionId, normalizedFormat, normalizedLayout), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const blob = await response.blob();
  const filenamePrefix = safeExportFilenamePrefix((currentConfig && (currentConfig.meeting_name || currentConfig.client_name)) || elements.meetingName.value.trim());
  const suffix = normalizedLayout === "interleaved" ? `-interleaved.${normalizedFormat}` : `.${normalizedFormat}`;
  const fallback = `${filenamePrefix || "meeting-log"}-${currentSessionStartedAt || new Date().toISOString()}${suffix}`.replace(/[:.]/g, "-").replace(/-(md|docx)$/i, ".$1");
  downloadBlob(filenameFromContentDisposition(response.headers.get("Content-Disposition"), fallback), blob);
  if (normalizedLayout === "interleaved") {
    setStatus("中英日志 DOCX 已下载", "ready");
    setToolStatus("中英穿插会议日志 DOCX 已下载。", "success");
  } else {
    setStatus(normalizedFormat === "docx" ? "日志 DOCX 已下载" : "后端日志已下载", "ready");
    setToolStatus(normalizedFormat === "docx" ? "当前会议日志 DOCX 已下载。" : "当前会议日志已下载。", "success");
  }
}

function summaryErrorDetailText(error) {
  const details = error && error.details;
  const missing = details && Array.isArray(details.missing_fields) ? details.missing_fields : [];
  const labels = missing.map((field) => field && (field.label || field.key)).filter(Boolean);
  if (!labels.length) return "";
  return "；缺失字段：" + labels.join("、");
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
    if (!response.ok || !result.generated) {
      const error = new Error(result.error || `HTTP ${response.status}`);
      error.code = result.error_code || "";
      error.details = result.details || {};
      throw error;
    }
    summaryGenerated = true;
    await loadSummaryInfo(selectedSummarySessionId);
    setStatus(result.summary && result.summary.latest_version > 1 ? "总结已重新生成" : "总结已生成", "ready");
    setToolStatus(result.summary && result.summary.latest_version > 1 ? "总结新版本已生成。" : "总结已生成。", "success");
    return result;
  } finally {
    summaryGenerating = false; updateSummaryButtons();
  }
}

async function downloadSummary(format = "md") {
  if (!selectedSummarySessionId) throw new Error("请选择可下载总结的会议 session");
  const version = elements.summaryVersion ? elements.summaryVersion.value : "";
  const normalizedFormat = format === "docx" ? "docx" : "md";
  const response = await fetch(summaryDownloadUrl(selectedSummarySessionId, normalizedFormat, version), { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const blob = await response.blob();
  const fallback = `meeting-summary-${selectedSummarySessionId}${version ? `-v${version}` : ""}.${normalizedFormat}`;
  downloadBlob(filenameFromContentDisposition(response.headers.get("Content-Disposition"), fallback), blob);
  setStatus(normalizedFormat === "docx" ? "总结 DOCX 已下载" : "总结已下载", "ready");
  setToolStatus(normalizedFormat === "docx" ? "总结 DOCX 文件已下载。" : "总结文件已下载。", "success");
}

async function downloadSummaryDocx() {
  return downloadSummary("docx");
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
  if (elements.downloadSummaryDocx) {
    elements.downloadSummaryDocx.disabled = !selectedSummarySessionId || !summaryGenerated || summaryGenerating;
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

function segmentBoundaryValue(segment) {
  const end = Number(segment?.end);
  if (Number.isFinite(end)) return end;
  const start = Number(segment?.start);
  return Number.isFinite(start) ? start : null;
}

function maxSegmentBoundary(segments) {
  return segments.reduce((maxValue, segment) => {
    const value = segmentBoundaryValue(segment);
    if (value === null) return maxValue;
    return maxValue === null ? value : Math.max(maxValue, value);
  }, null);
}

function shouldIgnoreClearedSegment(segment, clearBoundary) {
  if (clearBoundary === null) return false;
  const value = segmentBoundaryValue(segment);
  return value !== null && value <= clearBoundary;
}

function markSourceClearedBefore() {
  sourceClearedBefore = maxSegmentBoundary(sourceSegments);
}

function markTranslationClearedBefore() {
  translationClearedBefore = maxSegmentBoundary(translatedSegments);
}

function resetTranscriptClearBoundaries() {
  sourceClearedBefore = null;
  translationClearedBefore = null;
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
    if (shouldIgnoreClearedSegment(segment, sourceClearedBefore)) return;
    const copy = { ...segment };
    sourceSegmentStore.set(sourceSegmentStoreKey(copy), copy);
  });
  pruneSegmentStore(sourceSegmentStore);
  syncSegmentArrays();
}

function mergeTranslationSnapshot(segments) {
  const incoming = Array.isArray(segments) ? segments : [];
  incoming.forEach((segment) => {
    if (shouldIgnoreClearedSegment(segment, translationClearedBefore)) return;
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
  resetTranscriptClearBoundaries();
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

function hasTranslationWarning(segment) {
  return Boolean(segment?.translation_warning) || String(segment?.text || "").trim().endsWith(TRANSLATION_ERROR_SUFFIX);
}

function translationDisplayText(text) {
  const value = String(text || "").trim();
  return value.endsWith(TRANSLATION_ERROR_SUFFIX)
    ? value.slice(0, -TRANSLATION_ERROR_SUFFIX.length).trim()
    : value;
}

function appendTranslationWarningMark(target) {
  const mark = document.createElement("span");
  mark.className = "translation-warning-mark";
  mark.title = "翻译异常";
  mark.setAttribute("aria-label", "翻译异常");
  mark.textContent = "!";
  target.appendChild(document.createTextNode(" "));
  target.appendChild(mark);
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
      previous.translation_warning = previous.translation_warning || segment.translation_warning;
      previous.has_translation_warning = previous.has_translation_warning || hasTranslationWarning(segment);
      return;
    }
    groups.push({
      ...segment,
      group_key: key,
      text: String(segment.text || "").trim(),
      has_translation_warning: hasTranslationWarning(segment),
    });
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
    const isTranslationPane = target === elements.translationText;
    paragraph.textContent = isTranslationPane ? translationDisplayText(segment.text) : String(segment.text || "").trim();
    if (isTranslationPane && segment.has_translation_warning) appendTranslationWarningMark(paragraph);
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
      translation: translationDisplayText(translation.text),
      translationWarning: hasTranslationWarning(translation),
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
      translation: translationDisplayText(translation.text),
      translationWarning: hasTranslationWarning(translation),
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
  if (row.translationWarning) appendTranslationWarningMark(translation);
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
  renderSegments(elements.sourceText, sourceSegments, "等待原文内容...");
  renderSegments(elements.translationText, translatedSegments, elements.translationEnabled.checked ? "等待翻译内容..." : "翻译已关闭");
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
  displayMode = ["split", "stacked", "interleaved", "single"].includes(mode) ? mode : "split";
  window.localStorage.setItem("whisperlive_display_mode", displayMode);
  elements.displayMode.value = displayMode;
  singleLanguageMode = displayMode === "single" ? "translation" : singleLanguageMode;
  if (elements.singleLanguage) elements.singleLanguage.value = singleLanguageMode;
  window.localStorage.setItem("whisperlive_single_language", singleLanguageMode);
  elements.singleLanguageField.hidden = true;
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
    resumeNextConnection = false;
    if (reconnectController) reconnectController.markConnected();
    if (elements.continueMeeting) elements.continueMeeting.hidden = true;
    if (elements.finishInterrupted) elements.finishInterrupted.hidden = true;
    setStatus(message.resumed ? "已重连" : "已连接", "ready");
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
  const backend = currentServerBackend || DEFAULT_BACKEND;
  const translationEnabled = isTranslationServiceMode();
  const autoDetectMode = translationEnabled && faceToFaceEnabled();
  syncDirectionLanguages();
  let selectedLanguage = normalizeLanguage(selectedSourceLanguage) || defaultLanguageForBackend(backend);
  if (autoDetectMode) {
    selectedLanguage = null;
  } else if (!["zh", "en"].includes(selectedLanguage)) {
    selectedLanguage = defaultLanguageForBackend(backend);
  }
  const selectedTranslationTarget = translationEnabled ? selectedFaceToFaceTarget() : "auto";
  const translationMode = autoDetectMode ? "mixed_interpretation" : "standard";
  const translationProvider = selectedTranslationProviderConfig();
  const meetingName = elements.meetingName.value.trim();
  const payload = {
    uid,
    session_id: currentSessionId,
    session_started_at: currentSessionStartedAt,
    resume_session: Boolean(resumeNextConnection),
    server: elements.server.value.trim(),
    client_instance_id: clientInstanceId || getClientInstanceId(),
    client_name: meetingName || `Client-${uid.slice(0, 8)}`,
    meeting_name: meetingName,
    hotwords: lockedHotwords.hotwords || null,
    hotwords_count: lockedHotwords.count || 0,
    hotwords_file: lockedHotwords.filename || "",
    hotwords_locked: true,
    translation_glossary: lockedHotwords.translationGlossary || {},
    translation_glossary_count: lockedHotwords.translationCount || 0,
    backend: currentServerBackend || DEFAULT_BACKEND,
    language: selectedLanguage,
    task: "transcribe",
    model: DEFAULT_MODEL,
    use_vad: true,
    vad_parameters: {
      threshold: 0.5,
      min_silence_duration_ms: 900,
      speech_pad_ms: 300,
    },
    send_last_n_segments: DEFAULT_DISPLAY_SEGMENTS,
    no_speech_thresh: 0.45,
    clip_audio: false,
    same_output_threshold: 7,
    min_segment_rms: 0.022,
    min_transcription_chunk_seconds: 2.5,
    max_incomplete_segment_seconds: 10.0,
    service_mode: serviceMode,
    enable_diarization: elements.diarizationEnabled.checked,
    enable_translation: translationEnabled,
    target_language: selectedTranslationTarget,
    translation_mode: translationMode,
    translation_provider: translationProvider.provider,
    translation_device: translationDeviceForMode(serviceMode),
    translation_merge_enabled: true,
    translation_merge_max_chars: 240,
    translation_merge_max_delay: 1.8,
    translation_merge_gap_seconds: 1.0,
    zh_en_model_path: translationProvider.zhEnModelPath || "model/opus-mt-zh-en",
    en_zh_model_path: translationProvider.enZhModelPath || "model/opus-mt-en-zh",
    nllb_model_path: translationProvider.nllbModelPath,
  };
  currentConfig = payload;
  targetSocket.send(JSON.stringify(payload));
}

async function requestMicrophoneStream() {
  return window.WhisperLiveAudioCapture.requestMicrophoneStream();
}

function setConnectionInputsDisabled(disabled) {
  [
    elements.server,
    elements.language,
    elements.meetingName,
    elements.translationProvider,
    elements.faceToFaceEnabled,
    elements.faceToFaceMode,
    elements.translationTarget,
    elements.sourceLanguageButton,
    elements.targetLanguageButton,
    elements.directionToggleButton,
    ...elements.serviceModeButtons,
    elements.diarizationEnabled,
    elements.hotwordFile,
    elements.clearHotwordFile,
  ].filter(Boolean).forEach((item) => { item.disabled = disabled; });
  updateTranslationControls(disabled);
}

function openMeetingSocket(resume = false) {
  resumeNextConnection = Boolean(resume);
  intentionallyClosingSocket = false;
  isServerReady = false;
  const wsUrl = webSocketUrlForMode(elements.server.value, serviceMode);
  elements.server.value = wsUrl;
  socket = window.WhisperLiveWsClient.open(wsUrl, {
    open: sendConfig,
    message: handleMessage,
    error: () => setStatus("连接错误", "error"),
    close: handleSocketClose,
  });
}

function handleSocketClose() {
  socket = null;
  isServerReady = false;
  if (intentionallyClosingSocket || hasStoppedCurrentSession) return;
  selectedSummarySessionStatus = "interrupted";
  setStatus("连接中断，准备重连", "busy");
  if (!reconnectController) {
    reconnectController = new window.WhisperLiveReconnectController({
      onStatus: (attempt, total) => setStatus(`正在重连 ${attempt}/${total}，断线期间音频未记录`, "busy"),
      onReconnect: () => openMeetingSocket(true),
      onFailed: () => markSessionInterrupted(),
    });
  }
  reconnectController.schedule();
}

function markSessionInterrupted() {
  setStatus("会议已中断", "error");
  if (mediaStream) {
    window.WhisperLiveAudioCapture.stopTracks(mediaStream);
    mediaStream = null;
  }
  if (processor) { processor.disconnect(); processor = null; }
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  elements.start.disabled = true;
  elements.stop.disabled = true;
  if (elements.continueMeeting) elements.continueMeeting.hidden = false;
  if (elements.finishInterrupted) elements.finishInterrupted.hidden = false;
  elements.server.disabled = false;
}

async function startCapture() {
  setStatus("连接中", "busy");
  const newSession = window.WhisperLiveSessionState.createSession();
  currentSessionId = newSession.sessionId;
  currentSessionStartedAt = newSession.startedAt;
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
  }
  updateMeetingTitle();
  updateHotwordStatus(
    lockedHotwords.filename
      ? `${lockedHotwords.filename} · ${lockedHotwords.count} 个识别热词 · ${lockedHotwords.translationCount} 条固定翻译`
      : "未上传热词"
  );

  mediaStream = await requestMicrophoneStream();

  openMeetingSocket(false);

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
  if (elements.continueMeeting) elements.continueMeeting.hidden = true;
  if (elements.finishInterrupted) elements.finishInterrupted.hidden = true;
  setConnectionInputsDisabled(true);
}

function stopCapture(sendEnd = true) {
  if (reconnectController) reconnectController.reset();
  intentionallyClosingSocket = Boolean(sendEnd);
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
    window.WhisperLiveAudioCapture.stopTracks(mediaStream);
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
  if (elements.continueMeeting) elements.continueMeeting.hidden = true;
  if (elements.finishInterrupted) elements.finishInterrupted.hidden = true;
  setConnectionInputsDisabled(false);
}


if (elements.loadTranscriptEditor) {
  elements.loadTranscriptEditor.addEventListener("click", () => {
    loadTranscriptEditor().catch((error) => setTranscriptEditorStatus(`加载失败：${error.message}`, "error"));
  });
}
if (elements.addSpeaker) {
  elements.addSpeaker.addEventListener("click", () => {
    addTranscriptSpeaker().catch((error) => setTranscriptEditorStatus(`新增失败：${error.message}`, "error"));
  });
}
if (elements.speakerRenameGuide) {
  elements.speakerRenameGuide.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action='rename-speaker-guide']");
    const row = button && button.closest(".speaker-guide-row");
    if (!button || !row) return;
    renameTranscriptSpeakerFromGuide(row).catch((error) => setTranscriptEditorStatus(`保存失败：${error.message}`, "error"));
  });
}
if (elements.speakerManagerList) {
  elements.speakerManagerList.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    const row = button && button.closest(".speaker-manager-row");
    if (!button || !row) return;
    const action = button.dataset.action === "rename-speaker" ? renameTranscriptSpeaker : mergeTranscriptSpeaker;
    action(row).catch((error) => setTranscriptEditorStatus(`操作失败：${error.message}`, "error"));
  });
}
if (elements.transcriptEditorRows) {
  elements.transcriptEditorRows.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    const row = button && button.closest(".transcript-editor-row");
    if (!button || !row) return;
    saveTranscriptEditorRow(row, button.dataset.action === "restore-segment")
      .catch((error) => setTranscriptEditorStatus(`保存失败：${error.message}`, "error"));
  });
}

if (elements.summaryQuick) elements.summaryQuick.addEventListener("click", openSummaryDrawer);
elements.settingsButton.addEventListener("click", openSettings);
elements.closeSettings.addEventListener("click", closeSettings);
if (elements.closeSummary) elements.closeSummary.addEventListener("click", closeSummaryDrawer);
elements.settingsBackdrop.addEventListener("click", closeAllDrawers);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (immersiveFullscreenFallback || document.body.classList.contains("is-caption-fullscreen")) {
    exitCaptionFullscreen().catch((error) => console.error(error));
    return;
  }
  closeAllDrawers();
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
if (elements.translationProvider) {
  elements.translationProvider.addEventListener("change", () => {
    window.localStorage.setItem("whisperlive_translation_provider", elements.translationProvider.value || defaultTranslationProviderForMode());
    updateTranslationControls();
  });
}
if (elements.faceToFaceMode) {
  elements.faceToFaceMode.addEventListener("change", () => {
    setDirectionMode(faceToFaceMode() === "specified" ? "specified" : "auto");
    window.localStorage.setItem("whisperlive_face_to_face_mode", faceToFaceMode());
  });
}
if (elements.translationTarget) {
  elements.translationTarget.addEventListener("change", () => {
    const rawTarget = String(elements.translationTarget.value || "auto");
    if (rawTarget === "auto") {
      setDirectionMode("auto");
    } else {
      const target = normalizeLanguage(rawTarget) || oppositeLanguage(selectedSourceLanguage);
      selectedTargetLanguageCode = target;
      selectedSourceLanguage = oppositeLanguage(target);
      setDirectionMode("specified");
    }
    window.localStorage.setItem("whisperlive_translation_target", selectedTargetLanguage());
  });
}
elements.language.addEventListener("change", () => {
  selectedSourceLanguage = normalizeLanguage(elements.language.value) || selectedSourceLanguage;
  selectedTargetLanguageCode = oppositeLanguage(selectedSourceLanguage);
  setDirectionMode(directionMode);
});
elements.serviceModeButtons.forEach((button) => {
  button.addEventListener("click", () => setServiceMode(button.dataset.serviceMode));
});
if (elements.sourceLanguageButton) {
  elements.sourceLanguageButton.addEventListener("click", toggleSourceLanguage);
}
if (elements.targetLanguageButton) {
  elements.targetLanguageButton.addEventListener("click", toggleTargetLanguage);
}
if (elements.directionToggleButton) {
  elements.directionToggleButton.addEventListener("click", () => {
    setDirectionMode(directionMode === "auto" ? "specified" : "auto");
  });
}
if (elements.fullscreenButton) {
  elements.fullscreenButton.addEventListener("click", () => {
    const active = document.fullscreenElement || immersiveFullscreenFallback || document.body.classList.contains("is-caption-fullscreen");
    (active ? exitCaptionFullscreen() : enterCaptionFullscreen()).catch((error) => console.error(error));
  });
}
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && !immersiveFullscreenFallback) {
    document.body.classList.remove("is-caption-fullscreen");
  }
  updateFullscreenButton();
});
elements.diarizationEnabled.addEventListener("change", () => {
  window.localStorage.setItem("whisperlive_diarization_enabled", String(elements.diarizationEnabled.checked));
});
elements.server.addEventListener("change", () => {
  window.localStorage.setItem("whisperlive_server_url", elements.server.value.trim());
  applyBackendLanguageDefault().catch(() => applyBackendLanguageFallback());
  loadTranslationModels(defaultTranslationProviderForMode(serviceMode)).catch(() => {});
});

async function continueInterruptedMeeting() {
  if (!currentSessionId) return;
  setStatus("继续会议中", "busy");
  mediaStream = await requestMicrophoneStream();
  audioContext = new AudioContext();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    if (!socket || socket.readyState !== WebSocket.OPEN || !isServerReady) return;
    const input = event.inputBuffer.getChannelData(0);
    const output = downsampleTo16k(input, audioContext.sampleRate);
    socket.send(output.buffer);
  };
  sourceNode.connect(processor);
  processor.connect(audioContext.destination);
  elements.stop.disabled = false;
  if (elements.continueMeeting) elements.continueMeeting.hidden = true;
  if (elements.finishInterrupted) elements.finishInterrupted.hidden = true;
  setConnectionInputsDisabled(true);
  openMeetingSocket(true);
}

async function finishInterruptedMeeting() {
  if (!currentSessionId) return;
  setStatus("结束会议中", "busy");
  const response = await fetch(meetingLogFinishUrl(currentSessionId), { method: "POST" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  hasStoppedCurrentSession = true;
  selectedSummarySessionId = currentSessionId;
  selectedSummarySessionStatus = "finished";
  updateSummaryButtons();
  await loadSummarySessions(currentSessionId).catch(() => {});
  stopCapture(false);
  setStatus("已停止", "idle");
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

if (elements.continueMeeting) {
  elements.continueMeeting.addEventListener("click", () => {
    continueInterruptedMeeting().catch((error) => {
      console.error(error);
      setStatus("继续失败", "error");
    });
  });
}

if (elements.finishInterrupted) {
  elements.finishInterrupted.addEventListener("click", () => {
    finishInterruptedMeeting().catch((error) => {
      console.error(error);
      setStatus("结束失败", "error");
    });
  });
}

function handleMeetingLogExport(format = "md", layout = "sections") {
  exportMeetingLog(format, layout).catch((error) => {
    console.error(error);
    setStatus("日志导出失败", "error");
    setToolStatus(`日志导出失败：${error.message}`, "error");
  });
}

elements.exportLog.addEventListener("click", () => {
  handleMeetingLogExport("md", "sections");
});
if (elements.exportLogDocx) {
  elements.exportLogDocx.addEventListener("click", () => {
    handleMeetingLogExport("docx", "sections");
  });
}
if (elements.exportInterleavedLogDocx) {
  elements.exportInterleavedLogDocx.addEventListener("click", () => {
    handleMeetingLogExport("docx", "interleaved");
  });
}
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
if (elements.summaryTemplate) {
  elements.summaryTemplate.addEventListener("change", updateSummaryTemplateDeleteButton);
}
if (elements.deleteSummaryTemplate) {
  elements.deleteSummaryTemplate.addEventListener("click", () => {
    deleteSelectedSummaryTemplate().catch((error) => {
      console.error(error);
      setToolStatus(`模板删除失败：${error.message}`, "error");
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
    summaryTemplateDraft.fields.push(createSummaryTemplateField(section));
    section.role = "field";
    sortSummaryTemplateFields();
    renderSummaryTemplateFields();
  });
}
if (elements.generateSummary) {
  elements.generateSummary.addEventListener("click", () => {
    generateSummary().catch((error) => {
      console.error(error);
      setStatus("总结生成失败", "error");
      const prefix = error.code === "summary_quality_insufficient"
        ? "总结不完整，未保存新版本"
        : "总结生成失败";
      setToolStatus(`${prefix}：${error.message}${summaryErrorDetailText(error)}`, "error");
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
if (elements.downloadSummaryDocx) {
  elements.downloadSummaryDocx.addEventListener("click", () => {
    downloadSummaryDocx().catch((error) => {
      console.error(error);
      setStatus("总结 DOCX 下载失败", "error");
      setToolStatus(`总结 DOCX 下载失败：${error.message}`, "error");
    });
  });
}
if (elements.summarySession) {
  elements.summarySession.addEventListener("change", () => {
    selectedSummarySessionId = elements.summarySession.value || null;
    const selected = elements.summarySession.selectedOptions[0];
    selectedSummarySessionStatus = selectedSummarySessionId ? (selected.dataset.status || "finished") : null;
    transcriptEditorData = null;
    renderTranscriptEditor();
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
  updateMeetingTitle();
});

if (elements.hotwordFile) {
  elements.hotwordFile.addEventListener("change", () => {
    const file = elements.hotwordFile.files && elements.hotwordFile.files[0];
    if (!file) {
      clearUploadedHotwords();
      return;
    }
    updateHotwordStatus("正在解析热词文件…");
    loadUploadedHotwords(file).catch((error) => {
      console.error(error);
      clearUploadedHotwords();
      updateHotwordStatus(`热词文件解析失败：${error.message}`);
    });
  });
}

if (elements.clearHotwordFile) {
  elements.clearHotwordFile.addEventListener("click", clearUploadedHotwords);
}


elements.clearSource.addEventListener("click", () => {
  markSourceClearedBefore();
  clearSourceSegmentState();
  renderTranscriptViews();
});

elements.clearTranslation.addEventListener("click", () => {
  markTranslationClearedBefore();
  clearTranslationSegmentState();
  renderTranscriptViews();
});

updateSummaryButtons();
initializeDefaults();
