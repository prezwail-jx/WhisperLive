import json
import re
import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch

from whisper_live import metrics as wl_metrics
from whisper_live.backend.base import ServeClientBase
from whisper_live.vad import VoiceActivityDetector


@dataclass
class FunASRSegment:
    text: str
    start: float
    end: float
    no_speech_prob: float = 0.0


class ServeClientFunASR(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_KEY = None
    SINGLE_MODEL_LOCK = threading.Lock()
    SINGLE_MODEL_INIT_LOCK = threading.Lock()
    FINAL_MODEL = None
    FINAL_MODEL_KEY = None
    FINAL_MODEL_LOCK = threading.Lock()
    FINAL_MODEL_INIT_LOCK = threading.Lock()

    MODE_SENSEVOICE = "sensevoice"
    MODE_PARAFORMER_STREAMING = "paraformer_streaming"

    VAD_WINDOW_SECONDS = 0.5
    STREAMING_WINDOW_SECONDS = 0.8
    MIN_SPEECH_SECONDS = 1.0
    END_SILENCE_SECONDS = 0.8
    PARTIAL_INTERVAL_SECONDS = 1.5
    STREAMING_SENTENCE_ENDPOINT_MIN_SECONDS = 4.0
    STREAMING_SOFT_MAX_SPEECH_SECONDS = 12.0
    STREAMING_SOFT_MAX_MIN_CHARS = 12
    FINAL_SPLIT_TARGET_CHARS = 70
    FINAL_SPLIT_MAX_CHARS = 110
    FINAL_SPLIT_MIN_CHARS = 18
    FINAL_SPLIT_WEAK_BOUNDARIES = (
        "但是",
        "所以",
        "因此",
        "因为",
        "不过",
        "另外",
        "也就是说",
        "换句话说",
        "举个例子",
        "具体来说",
        "在这种情况下",
    )
    MAX_SPEECH_SECONDS = 16.0
    SPEECH_PAD_SECONDS = 0.3
    SENTENCE_END_RE = re.compile(r'[。！？!?；;]+[\s\)\]\}）】》’”"]*$')

    def __init__(
        self,
        websocket,
        task="transcribe",
        device="auto",
        language=None,
        client_uid=None,
        model="iic/SenseVoiceSmall",
        mode="sensevoice",
        punc_model=None,
        vad_model=None,
        final_model="model/funasr/SenseVoiceSmall",
        final_device=None,
        final_refine=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=3,
        translation_queue=None,
        hotwords=None,
        min_segment_rms=0.0015,
        max_incomplete_segment_seconds=6.0,
        use_vad=True,
        vad_threshold=0.5,
    ):
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            translation_queue,
            min_segment_rms=min_segment_rms,
            max_incomplete_segment_seconds=max_incomplete_segment_seconds,
        )
        self.mode = mode or self.MODE_SENSEVOICE
        if self.mode not in {self.MODE_SENSEVOICE, self.MODE_PARAFORMER_STREAMING}:
            raise ValueError(f"Unsupported FunASR mode: {self.mode}")
        self.model_size_or_path = model or "iic/SenseVoiceSmall"
        self.punc_model_size_or_path = punc_model
        self.vad_model_size_or_path = vad_model
        self.final_model_size_or_path = final_model
        self.final_device = self._resolve_device(final_device or device)
        self.final_refine = bool(final_refine and final_model)
        self.final_transcriber = None
        self.punctuator = None
        self.language = language or "auto"
        self.task = task
        self.hotwords = hotwords
        self.device = self._resolve_device(device)
        self.use_vad = use_vad
        self.vad_detector = VoiceActivityDetector(threshold=vad_threshold, frame_rate=self.RATE) if use_vad else None

        window_seconds = self.STREAMING_WINDOW_SECONDS if self._is_streaming_mode() else self.VAD_WINDOW_SECONDS
        self.vad_window_samples = int(window_seconds * self.RATE)
        self.speech_pad_samples = int(self.SPEECH_PAD_SECONDS * self.RATE)
        self.pre_speech_audio = np.empty(0, dtype=np.float32)
        self.speech_buffer = np.empty(0, dtype=np.float32)
        self.speech_start_time = None
        self.utterance_sequence = 0
        self.current_utterance_id = None
        self.silence_seconds = 0.0
        self.last_partial_seconds = 0.0

        self.streaming_cache = {}
        self.streaming_chunk_size = [0, 10, 5]
        self.streaming_encoder_chunk_look_back = 4
        self.streaming_decoder_chunk_look_back = 1
        self.streaming_partial_text = ""

        try:
            model_key = (self.mode, self.model_size_or_path, self.device)
            if single_model:
                with ServeClientFunASR.SINGLE_MODEL_INIT_LOCK:
                    if ServeClientFunASR.SINGLE_MODEL is None or ServeClientFunASR.SINGLE_MODEL_KEY != model_key:
                        logging.info("Loading shared FunASR model")
                        self.create_model()
                        ServeClientFunASR.SINGLE_MODEL = self.transcriber
                        ServeClientFunASR.SINGLE_MODEL_KEY = model_key
                    else:
                        logging.info("Reusing shared FunASR model")
                        self.transcriber = ServeClientFunASR.SINGLE_MODEL
            else:
                self.create_model()
        except Exception as e:
            logging.error(f"Failed to load FunASR model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load FunASR model: {str(self.model_size_or_path)}"
            }))
            self.websocket.close()
            return

        if self._is_streaming_mode() and self.final_refine:
            self.create_final_model(single_model=single_model)

        if self._is_streaming_mode() and self.punc_model_size_or_path:
            self.create_punc_model()

        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "funasr"
        }))

    def _is_streaming_mode(self):
        return self.mode == self.MODE_PARAFORMER_STREAMING

    def _resolve_device(self, device):
        if device and device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def create_model(self):
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ImportError("funasr is not installed. Install it with: pip install funasr") from exc

        logging.info("Loading FunASR model: %s on %s", self.model_size_or_path, self.device)
        self.transcriber = AutoModel(
            model=self.model_size_or_path,
            device=self.device,
            disable_update=True,
        )

    def create_final_model(self, single_model=False):
        try:
            from funasr import AutoModel
            model_key = (self.final_model_size_or_path, self.final_device)
            if single_model:
                with ServeClientFunASR.FINAL_MODEL_INIT_LOCK:
                    if ServeClientFunASR.FINAL_MODEL is None or ServeClientFunASR.FINAL_MODEL_KEY != model_key:
                        logging.info("Loading shared FunASR final model: %s on %s", self.final_model_size_or_path, self.final_device)
                        ServeClientFunASR.FINAL_MODEL = AutoModel(
                            model=self.final_model_size_or_path,
                            device=self.final_device,
                            disable_update=True,
                        )
                        ServeClientFunASR.FINAL_MODEL_KEY = model_key
                    else:
                        logging.info("Reusing shared FunASR final model")
                    self.final_transcriber = ServeClientFunASR.FINAL_MODEL
            else:
                logging.info("Loading FunASR final model: %s on %s", self.final_model_size_or_path, self.final_device)
                self.final_transcriber = AutoModel(
                    model=self.final_model_size_or_path,
                    device=self.final_device,
                    disable_update=True,
                )
        except Exception as exc:
            self.final_refine = False
            self.final_transcriber = None
            logging.warning("FunASR final refinement disabled: %s", exc)

    def create_punc_model(self):
        try:
            from funasr import AutoModel
            logging.info("Loading FunASR punctuation model: %s on %s", self.punc_model_size_or_path, self.device)
            self.punctuator = AutoModel(
                model=self.punc_model_size_or_path,
                device=self.device,
                disable_update=True,
            )
        except Exception as exc:
            self.punctuator = None
            logging.warning("FunASR punctuation model disabled: %s", exc)

    def speech_to_text(self):
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break
            if self.frames_np is None:
                time.sleep(0.05)
                continue

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if input_bytes.shape[0] < self.vad_window_samples:
                time.sleep(0.05)
                continue

            consumed = 0
            total = input_bytes.shape[0]
            while consumed + self.vad_window_samples <= total:
                window = input_bytes[consumed:consumed + self.vad_window_samples].astype(np.float32, copy=False)
                with self.lock:
                    window_start = self.timestamp_offset + (consumed / self.RATE)
                voice_active = self._is_voice_active(window)
                self._process_vad_window(window, window_start, voice_active)
                consumed += self.vad_window_samples

            if consumed:
                self._advance_timestamp(consumed / self.RATE)
                wl_metrics.track_audio_processed(consumed / self.RATE)
            else:
                time.sleep(0.05)

    def _advance_timestamp(self, seconds):
        with self.lock:
            self.timestamp_offset += seconds

    def _is_voice_active(self, audio_window):
        if not self.use_vad or self.vad_detector is None:
            return True
        try:
            return bool(self.vad_detector(audio_window))
        except Exception as e:
            logging.error("FunASR VAD failed, treating window as speech: %s", e)
            return True

    def _process_vad_window(self, window, window_start, voice_active):
        if self._is_streaming_mode():
            self._process_streaming_window(window, window_start, voice_active)
            return
        self._process_sensevoice_window(window, window_start, voice_active)

    def _process_sensevoice_window(self, window, window_start, voice_active):
        window_seconds = window.shape[0] / self.RATE
        if voice_active:
            if self.speech_buffer.size == 0:
                self.speech_start_time = max(0.0, window_start - (self.pre_speech_audio.shape[0] / self.RATE))
                self._begin_utterance()
                self.speech_buffer = self._concat_audio(self.pre_speech_audio, window)
            else:
                self.speech_buffer = self._concat_audio(self.speech_buffer, window)
            self.silence_seconds = 0.0
            self._maybe_emit_partial()
            if self._speech_duration() >= self.MAX_SPEECH_SECONDS:
                self._emit_current_speech(completed=True, reason="max_speech")
            return

        if self.speech_buffer.size == 0:
            self._remember_pre_speech(window)
            return

        self.speech_buffer = self._concat_audio(self.speech_buffer, window)
        self.silence_seconds += window_seconds
        if self.silence_seconds >= self.END_SILENCE_SECONDS:
            self._emit_current_speech(completed=True, reason="silence")

    def _process_streaming_window(self, window, window_start, voice_active):
        window_seconds = window.shape[0] / self.RATE
        if voice_active:
            if self.speech_buffer.size == 0:
                self.streaming_cache = {}
                self.streaming_partial_text = ""
                self.speech_start_time = max(0.0, window_start - (self.pre_speech_audio.shape[0] / self.RATE))
                self._begin_utterance()
                self.speech_buffer = self._concat_audio(self.pre_speech_audio, window)
            else:
                self.speech_buffer = self._concat_audio(self.speech_buffer, window)
            self.silence_seconds = 0.0
            self._emit_streaming_partial(window, window_start)
            if self._should_endpoint_on_sentence():
                self._emit_streaming_final(np.empty(0, dtype=np.float32), reason="sentence_punctuation")
            elif self._should_endpoint_on_soft_max():
                self._emit_streaming_final(np.empty(0, dtype=np.float32), reason="soft_max_speech")
            elif self._speech_duration() >= self.MAX_SPEECH_SECONDS:
                self._emit_streaming_final(np.empty(0, dtype=np.float32), reason="max_speech")
            return

        if self.speech_buffer.size == 0:
            self._remember_pre_speech(window)
            return

        self.speech_buffer = self._concat_audio(self.speech_buffer, window)
        self.silence_seconds += window_seconds
        if self.silence_seconds >= self.END_SILENCE_SECONDS:
            self._emit_streaming_final(window, reason="silence")

    def _concat_audio(self, *arrays):
        valid = [array for array in arrays if array is not None and array.size > 0]
        if not valid:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(valid).astype(np.float32, copy=False)

    def _remember_pre_speech(self, window):
        self.pre_speech_audio = self._concat_audio(self.pre_speech_audio, window)
        if self.pre_speech_audio.shape[0] > self.speech_pad_samples:
            self.pre_speech_audio = self.pre_speech_audio[-self.speech_pad_samples:]

    def _speech_duration(self):
        return self.speech_buffer.shape[0] / self.RATE

    def _begin_utterance(self):
        self.utterance_sequence += 1
        start = float(self.speech_start_time or 0.0)
        self.current_utterance_id = f"{self.client_uid}:{self.utterance_sequence}:{start:.3f}"

    def _format_utterance_segment(self, start, end, text, completed):
        segment = self.format_segment(start, end, text, completed=completed)
        if self.current_utterance_id:
            segment["utterance_id"] = self.current_utterance_id
        return segment

    def _maybe_emit_partial(self):
        duration = self._speech_duration()
        if duration < self.MIN_SPEECH_SECONDS:
            return
        if duration - self.last_partial_seconds < self.PARTIAL_INTERVAL_SECONDS:
            return
        self.last_partial_seconds = duration
        self._emit_current_speech(completed=False, reason="partial")

    def _emit_current_speech(self, completed, reason):
        if self.speech_buffer.size == 0 or self.speech_start_time is None:
            return
        audio = self.speech_buffer.copy()
        duration = audio.shape[0] / self.RATE
        if duration < self.MIN_SPEECH_SECONDS:
            if completed:
                self._reset_speech_state()
            return

        t0 = time.time()
        result = self.transcribe_audio(audio)
        wl_metrics.track_transcription_latency(time.time() - t0)
        text = self._extract_text(result).strip()
        if not self._is_meaningful_text(text):
            if completed:
                self._reset_speech_state()
            return
        if self._is_low_energy_audio(audio, text):
            if completed:
                self._reset_speech_state()
            return

        start = self.speech_start_time
        end = start + duration
        segment = self._format_utterance_segment(start, end, text, completed=completed)
        if completed:
            self._commit_completed_segment(segment, text, reason, duration)
            self._reset_speech_state()
            return

        self.send_transcription_to_client(self.prepare_segments(segment))

    def _emit_streaming_partial(self, audio, window_start):
        duration = self._speech_duration()
        if duration < self.MIN_SPEECH_SECONDS:
            return
        t0 = time.time()
        result = self.transcribe_streaming_audio(audio, is_final=False)
        wl_metrics.track_transcription_latency(time.time() - t0)
        text = self._extract_text(result).strip()
        if not self._is_meaningful_text(text):
            return
        merged_text = self._merge_streaming_text(self.streaming_partial_text, text)
        if merged_text == self.streaming_partial_text:
            return
        self.streaming_partial_text = merged_text
        start = self.speech_start_time or 0.0
        end = max(start, window_start + (audio.shape[0] / self.RATE))
        segment = self._format_utterance_segment(start, end, merged_text, completed=False)
        self.send_transcription_to_client(self.prepare_segments(segment))

    def _should_endpoint_on_sentence(self):
        if self._speech_duration() < self.STREAMING_SENTENCE_ENDPOINT_MIN_SECONDS:
            return False
        return self._has_sentence_endpoint(self.streaming_partial_text)

    def _has_sentence_endpoint(self, text):
        return bool(self.SENTENCE_END_RE.search(str(text or "").strip()))

    def _should_endpoint_on_soft_max(self):
        if self._speech_duration() < self.STREAMING_SOFT_MAX_SPEECH_SECONDS:
            return False
        text = re.sub(r"[\s。！？!?,，、.；;：:\-—…]+", "", str(self.streaming_partial_text or ""))
        return len(text) >= self.STREAMING_SOFT_MAX_MIN_CHARS

    def _emit_streaming_final(self, final_audio, reason):
        if self.speech_buffer.size == 0 or self.speech_start_time is None:
            return
        audio = self.speech_buffer.copy()
        duration = audio.shape[0] / self.RATE
        if duration < self.MIN_SPEECH_SECONDS:
            self._reset_speech_state()
            return

        fallback_text = self._streaming_final_text(final_audio)
        refined_text = self._refine_final_text(audio)
        text = (refined_text or fallback_text).strip()
        text = self._punctuate_text(text).strip()
        if not self._is_meaningful_text(text) or self._is_low_energy_audio(audio, text):
            self._reset_speech_state()
            return

        start = self.speech_start_time
        end = start + duration
        text_parts = self._split_final_text(text)
        segments = self._segments_from_text_parts(start, end, text_parts)
        self._commit_completed_segments(segments, reason, duration)
        self._reset_speech_state()

    def _streaming_final_text(self, final_audio):
        final_audio = self._concat_audio(final_audio)
        if final_audio.size == 0:
            return self.streaming_partial_text.strip()
        t0 = time.time()
        result = self.transcribe_streaming_audio(final_audio, is_final=True)
        wl_metrics.track_transcription_latency(time.time() - t0)
        text = self._extract_text(result).strip()
        return self._merge_streaming_text(self.streaming_partial_text, text).strip()

    def _refine_final_text(self, audio):
        if not self.final_refine or self.final_transcriber is None:
            return ""
        try:
            t0 = time.time()
            result = self.transcribe_final_audio(audio)
            wl_metrics.track_transcription_latency(time.time() - t0)
            text = self._extract_text(result).strip()
            if text:
                logging.info("[FUNASR_FINAL_REFINE] uid=%s text=%r", self.client_uid, text[:80])
            return text
        except Exception as exc:
            logging.warning("FunASR final refinement failed; using streaming text: %s", exc)
            return ""

    def _normalize_final_text(self, text):
        text = str(text or "").strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[，,、]+([。！？!?；;])", r"\1", text)
        text = re.sub(r"([。！？!?；;])[，,、]+", r"\1", text)
        text = re.sub(r"[，,]{2,}", "，", text)
        text = re.sub(r"、{2,}", "、", text)
        text = re.sub(r"。{2,}", "。", text)
        text = re.sub(r"！{2,}", "！", text)
        text = re.sub(r"？{2,}", "？", text)
        text = re.sub(r"；{2,}", "；", text)
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"(?<!\.)\.{2,}(?!\.)", ".", text)
        return text.strip()

    def _repair_fragmented_punctuation(self, text):
        text = str(text or "").strip()
        if not text:
            return ""

        cjk = r"\u4e00-\u9fff"
        text = re.sub(rf"([{cjk}]{{1,3}})。\s*\1(?=[{cjk}])", r"\1", text)

        standalone_fragments = {
            "是",
            "对",
            "好",
            "嗯",
            "啊",
            "哦",
            "行",
            "可以",
            "不是",
            "没有",
            "谢谢",
        }
        followers = r"(?:这|那)(?:个|种|些|样)?|而|就|再|继续"

        def repair_match(match):
            prefix = match.group("prefix")
            follower = match.group("follower")
            if prefix in standalone_fragments:
                return match.group(0)
            return prefix + follower

        return re.sub(
            rf"(?P<prefix>[{cjk}]{{1,3}})。\s*(?P<follower>{followers})(?=[{cjk}])",
            repair_match,
            text,
        ).strip()

    def _effective_text_length(self, text):
        return len(re.sub(r"[\s。！？!?,，、.；;：:\-—…]+", "", str(text or "")))

    def _split_final_text(self, text):
        text = self._normalize_final_text(text)
        text = self._repair_fragmented_punctuation(text)
        if not text:
            return []

        sentences = self._split_final_sentences(text)
        merged = self._merge_final_sentences(sentences)
        parts = []
        for segment_text in merged:
            parts.extend(self._split_long_clause(segment_text))
        return [part for part in parts if self._is_meaningful_text(part)] or [text]

    def _split_final_sentences(self, text):
        sentences = []
        current = []
        strong_endings = set("。！？!?；;")
        for char in text:
            current.append(char)
            if char in strong_endings:
                part = "".join(current).strip()
                if part:
                    sentences.append(part)
                current = []
        tail = "".join(current).strip()
        if tail:
            sentences.append(tail)
        return sentences or [text]

    def _merge_final_sentences(self, sentences):
        merged = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not current:
                current = sentence
                continue
            candidate = current + sentence
            if self._effective_text_length(candidate) <= self.FINAL_SPLIT_MAX_CHARS:
                current = candidate
                continue
            if self._effective_text_length(current) < self.FINAL_SPLIT_MIN_CHARS:
                current = candidate
                continue
            merged.append(current)
            current = sentence
        if current:
            merged.append(current)
        return merged

    def _split_long_clause(self, text):
        text = text.strip()
        if self._effective_text_length(text) <= self.FINAL_SPLIT_MAX_CHARS:
            return [text]

        parts = []
        remaining = text
        while self._effective_text_length(remaining) > self.FINAL_SPLIT_MAX_CHARS:
            split_at = self._find_weak_split_index(remaining)
            if split_at is None:
                break
            head = remaining[:split_at].strip()
            tail = remaining[split_at:].strip()
            if not head or not tail:
                break
            parts.append(head)
            remaining = tail
        if remaining:
            parts.append(remaining.strip())
        return parts or [text]

    def _find_weak_split_index(self, text):
        target_chars = self.FINAL_SPLIT_TARGET_CHARS
        max_chars = self.FINAL_SPLIT_MAX_CHARS
        min_chars = self.FINAL_SPLIT_MIN_CHARS
        candidates = []
        for phrase in self.FINAL_SPLIT_WEAK_BOUNDARIES:
            start = 0
            while True:
                index = text.find(phrase, start)
                if index < 0:
                    break
                if index > 0:
                    prefix_len = self._effective_text_length(text[:index])
                    suffix_len = self._effective_text_length(text[index:])
                    if min_chars <= prefix_len <= max_chars and suffix_len >= min_chars:
                        candidates.append((abs(target_chars - prefix_len), index))
                start = index + len(phrase)
        if candidates:
            return sorted(candidates)[0][1]

        punctuation_candidates = []
        for match in re.finditer(r"[，,、：:]", text):
            index = match.end()
            prefix_len = self._effective_text_length(text[:index])
            suffix_len = self._effective_text_length(text[index:])
            if min_chars <= prefix_len <= max_chars and suffix_len >= min_chars:
                punctuation_candidates.append((abs(target_chars - prefix_len), index))
        if punctuation_candidates:
            return sorted(punctuation_candidates)[0][1]
        return None

    def _segments_from_text_parts(self, start, end, parts):
        parts = [part.strip() for part in parts if self._is_meaningful_text(part)]
        if not parts:
            return []
        duration = max(0.001, end - start)
        lengths = [max(1, self._effective_text_length(part)) for part in parts]
        total = sum(lengths) or len(parts)
        segments = []
        cursor = start
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                part_end = end
            else:
                part_end = start + duration * (sum(lengths[:index + 1]) / total)
                part_end = max(cursor + 0.001, min(part_end, end))
            segments.append(self._format_utterance_segment(cursor, part_end, part, completed=True))
            cursor = part_end
        return segments

    def _commit_completed_segments(self, segments, reason, duration):
        if not segments:
            return
        for segment in segments:
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            self.text.append(text)
            self.transcript.append(segment)
            if self.translation_queue:
                try:
                    self.translation_queue.put(segment.copy(), timeout=0.1)
                except queue.Full:
                    logging.warning("Translation queue is full, skipping segment")
            logging.info(
                "[FUNASR_SEGMENT_COMPLETE] uid=%s mode=%s reason=%s duration=%.2fs text=%r",
                self.client_uid,
                self.mode,
                reason,
                duration,
                text[:80],
            )
        self._trim_transcript()
        self.send_transcription_to_client(self.prepare_segments())

    def _commit_completed_segment(self, segment, text, reason, duration):
        self.text.append(text)
        self.transcript.append(segment)
        if self.translation_queue:
            try:
                self.translation_queue.put(segment.copy(), timeout=0.1)
            except queue.Full:
                logging.warning("Translation queue is full, skipping segment")
        self._trim_transcript()
        self.send_transcription_to_client(self.prepare_segments())
        logging.info(
            "[FUNASR_SEGMENT_COMPLETE] uid=%s mode=%s reason=%s duration=%.2fs text=%r",
            self.client_uid,
            self.mode,
            reason,
            duration,
            text[:80],
        )

    def _merge_streaming_text(self, previous, current):
        previous = str(previous or "").strip()
        current = str(current or "").strip()
        if not previous:
            return current
        if not current:
            return previous
        if current.startswith(previous):
            return current
        if previous.endswith(current) or current in previous:
            return previous
        return previous + current

    def _punctuate_text(self, text):
        if not text or self.punctuator is None:
            return text
        try:
            result = self.punctuator.generate(input=text)
            punctuated = self._extract_text(result).strip()
            return punctuated or text
        except Exception as exc:
            logging.warning("FunASR punctuation failed; using raw streaming text: %s", exc)
            return text

    def _is_meaningful_text(self, text):
        stripped = str(text or "").strip()
        if not stripped:
            return False
        return bool(re.sub(r"[\s。！？!?,，、.；;：:\-—…]+", "", stripped))

    def _is_low_energy_audio(self, audio, text):
        if self.min_segment_rms <= 0 or audio.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if rms >= self.min_segment_rms:
            return False
        logging.info(
            "[LOW_ENERGY_SEGMENT_DROP] uid=%s rms=%.6f threshold=%.6f text=%r",
            self.client_uid,
            rms,
            self.min_segment_rms,
            str(text or "").strip()[:80],
        )
        return True

    def _reset_speech_state(self):
        tail = self.speech_buffer[-self.speech_pad_samples:].copy() if self.speech_buffer.size else np.empty(0, dtype=np.float32)
        self.pre_speech_audio = tail
        self.speech_buffer = np.empty(0, dtype=np.float32)
        self.speech_start_time = None
        self.current_utterance_id = None
        self.silence_seconds = 0.0
        self.last_partial_seconds = 0.0
        self.streaming_cache = {}
        self.streaming_partial_text = ""

    def transcribe_audio(self, input_sample):
        kwargs = {
            "input": input_sample,
            "language": self.language or "auto",
            "use_itn": True,
            "batch_size_s": 60,
        }
        if self.hotwords:
            kwargs["hotword"] = self.hotwords

        if ServeClientFunASR.SINGLE_MODEL:
            with ServeClientFunASR.SINGLE_MODEL_LOCK:
                return self._generate(**kwargs)
        return self._generate(**kwargs)

    def transcribe_streaming_audio(self, input_sample, is_final=False):
        kwargs = {
            "input": input_sample,
            "cache": self.streaming_cache,
            "is_final": is_final,
            "chunk_size": self.streaming_chunk_size,
            "encoder_chunk_look_back": self.streaming_encoder_chunk_look_back,
            "decoder_chunk_look_back": self.streaming_decoder_chunk_look_back,
        }
        if self.hotwords:
            kwargs["hotword"] = self.hotwords
        if ServeClientFunASR.SINGLE_MODEL:
            with ServeClientFunASR.SINGLE_MODEL_LOCK:
                return self._generate_streaming(**kwargs)
        return self._generate_streaming(**kwargs)

    def transcribe_final_audio(self, input_sample):
        kwargs = {
            "input": input_sample,
            "language": self.language or "auto",
            "use_itn": True,
            "batch_size_s": 60,
        }
        if self.hotwords:
            kwargs["hotword"] = self.hotwords
        if ServeClientFunASR.FINAL_MODEL:
            with ServeClientFunASR.FINAL_MODEL_LOCK:
                return self._generate_final(**kwargs)
        return self._generate_final(**kwargs)

    def _generate(self, **kwargs):
        try:
            return self.transcriber.generate(**kwargs)
        except TypeError:
            if "hotword" in kwargs:
                logging.warning("FunASR model does not accept hotword; retrying without hotwords")
                kwargs.pop("hotword", None)
                return self.transcriber.generate(**kwargs)
            raise

    def _generate_streaming(self, **kwargs):
        try:
            return self.transcriber.generate(**kwargs)
        except TypeError:
            if "hotword" in kwargs:
                logging.warning("FunASR streaming model does not accept hotword; retrying without hotwords")
                kwargs.pop("hotword", None)
                return self.transcriber.generate(**kwargs)
            raise

    def _generate_final(self, **kwargs):
        try:
            return self.final_transcriber.generate(**kwargs)
        except TypeError:
            if "hotword" in kwargs:
                logging.warning("FunASR final model does not accept hotword; retrying without hotwords")
                kwargs.pop("hotword", None)
                return self.final_transcriber.generate(**kwargs)
            raise

    def _clean_text(self, text):
        return re.sub(r"<\|[^|]+\|>", "", str(text or ""))

    def _extract_text(self, result):
        if result is None:
            return ""
        if isinstance(result, str):
            return self._clean_text(result)
        if isinstance(result, dict):
            return self._clean_text(result.get("text") or result.get("sentence") or "")
        if isinstance(result, list):
            parts = [self._extract_text(item).strip() for item in result]
            return " ".join(part for part in parts if part)
        return self._clean_text(result)

    def _extract_segments(self, result, duration):
        if isinstance(result, list) and len(result) > 1:
            step = max(0.0, duration) / len(result)
            segments = []
            for index, item in enumerate(result):
                text = self._extract_text(item).strip()
                if not text:
                    continue
                segments.append(FunASRSegment(
                    text=text,
                    start=index * step,
                    end=(index + 1) * step,
                ))
            return segments

        text = self._extract_text(result).strip()
        if not text:
            return []
        return [FunASRSegment(text=text, start=0.0, end=max(0.0, duration))]

    def handle_transcription_output(self, result, duration):
        segments = self._extract_segments(result, duration)
        if not segments:
            return

        last_segment = self.update_segments(segments, duration)
        output_segments = self.prepare_segments(last_segment)
        if output_segments:
            self.send_transcription_to_client(output_segments)
