const TARGET_SAMPLE_RATE = 16000;

const elements = {
  form: document.getElementById("settingsForm"),
  server: document.getElementById("serverInput"),
  model: document.getElementById("modelInput"),
  language: document.getElementById("languageInput"),
  segments: document.getElementById("segmentsInput"),
  start: document.getElementById("startButton"),
  stop: document.getElementById("stopButton"),
  status: document.getElementById("connectionStatus"),
  languageStatus: document.getElementById("languageStatus"),
  sampleRateStatus: document.getElementById("sampleRateStatus"),
  directionStatus: document.getElementById("directionStatus"),
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

function createUid() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function setStatus(text, state = "idle") {
  elements.status.textContent = text;
  elements.status.className = `status ${state}`;
}

function appendOrReplaceSegment(list, segment) {
  const index = list.findIndex((item) => item.start === segment.start && item.end === segment.end);
  if (index >= 0) {
    list[index] = segment;
    return;
  }
  list.push(segment);
  list.sort((a, b) => Number(a.start) - Number(b.start));
}

function renderSegments(target, segments, emptyText) {
  if (!segments.length) {
    target.textContent = emptyText;
    target.classList.add("muted");
    return;
  }

  target.classList.remove("muted");
  target.innerHTML = "";
  const fragment = document.createDocumentFragment();
  segments.forEach((segment) => {
    const paragraph = document.createElement("p");
    paragraph.className = `segment${segment.completed ? "" : " incomplete"}`;
    paragraph.textContent = segment.text.trim();
    fragment.appendChild(paragraph);
  });
  target.appendChild(fragment);
  target.scrollTop = target.scrollHeight;
}

function updateDirection(segment) {
  if (!segment) {
    elements.directionStatus.textContent = "自动";
    return;
  }
  const source = segment.source_language || segment.language || "?";
  const target = segment.target_language || "?";
  elements.directionStatus.textContent = `${source} -> ${target}`;
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
    message.segments.forEach((segment) => appendOrReplaceSegment(sourceSegments, segment));
    renderSegments(elements.sourceText, sourceSegments, "等待语音输入...");
  }

  if (message.translated_segments) {
    message.translated_segments.forEach((segment) => appendOrReplaceSegment(translatedSegments, segment));
    updateDirection(translatedSegments[translatedSegments.length - 1]);
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

function sendConfig() {
  uid = createUid();
  const selectedLanguage = elements.language.value || null;
  const payload = {
    uid,
    language: selectedLanguage,
    task: "transcribe",
    model: elements.model.value,
    use_vad: true,
    send_last_n_segments: Number(elements.segments.value || 8),
    no_speech_thresh: 0.45,
    clip_audio: false,
    same_output_threshold: 2,
    enable_translation: true,
    target_language: "auto",
    translation_provider: "helsinki_zh_en",
    zh_en_model_path: "model/opus-mt-zh-en",
    en_zh_model_path: "model/opus-mt-en-zh",
  };
  socket.send(JSON.stringify(payload));
}

async function startCapture() {
  setStatus("连接中", "busy");
  sourceSegments = [];
  translatedSegments = [];
  renderSegments(elements.sourceText, sourceSegments, "等待语音输入...");
  renderSegments(elements.translationText, translatedSegments, "等待翻译结果...");
  updateDirection(null);

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

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      channelCount: 1,
    },
  });

  audioContext = new AudioContext();
  elements.sampleRateStatus.textContent = `${audioContext.sampleRate} -> 16 kHz`;
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
}

function stopCapture(sendEnd = true) {
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

elements.clearSource.addEventListener("click", () => {
  sourceSegments = [];
  renderSegments(elements.sourceText, sourceSegments, "等待语音输入...");
});

elements.clearTranslation.addEventListener("click", () => {
  translatedSegments = [];
  renderSegments(elements.translationText, translatedSegments, "等待翻译结果...");
  updateDirection(null);
});
