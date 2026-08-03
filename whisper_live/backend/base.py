import json
import logging
import re
import threading
import time
import queue
import unicodedata
import numpy as np

from whisper_live import metrics as wl_metrics

try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    client_uid: str
    """A unique identifier for the client."""
    websocket: object
    """The WebSocket connection for the client."""
    send_last_n_segments: int
    """Number of most recent segments to send to the client."""
    no_speech_thresh: float
    """Segments with no speech probability above this threshold will be discarded."""
    clip_audio: bool
    """Whether to clip audio with no valid segments."""
    same_output_threshold: int
    """Number of repeated outputs before considering it as a valid segment."""
    min_segment_rms: float
    """Minimum RMS energy required before accepting an ASR segment."""
    max_incomplete_segment_seconds: float
    """Maximum audio duration to keep reprocessing a single incomplete segment."""
    sentence_completion_min_seconds: float
    """Minimum incomplete segment duration before sentence-ending punctuation can complete it."""
    max_pending_audio_seconds: float
    """Maximum unprocessed audio duration to keep before advancing the realtime offset."""

    MAX_TRANSCRIPT_LENGTH = 500
    MAX_TRANSLATION_QUEUE_SIZE = 100
    MAX_PENDING_AUDIO_SECONDS = 8.0
    MAX_CONFIGURABLE_PENDING_AUDIO_SECONDS = 30.0
    OPENCC_CONFIG = "t2s"
    OPENCC_UNAVAILABLE_LOGGED = False
    SILENCE_HALLUCINATION_PHRASES = {
        "thank you",
        "thank you very much",
        "thanks",
        "thanks very much",
        "thanks for watching",
        "thank you for watching",
        "bye",
        "I don't know.",
        "I'm going to put it in a little bit.",
        "So, I'm going to show you what I'm going to do here.",
        "bye bye",
        "you",
        "You",
        "oh",
        ".",
        "看我了",
        "我看你了",
        "yeah",
        "wow",
        "all right",
        "exactly",
        "right",
        "ok",
        "okay",
        "alright",
        "um",
        "uh",
        "hmm",
        "so",
        "well",
        "no",
        "mm-hmm",
        "aha",
        "eh",
        "优优独播剧场——YoYo Television Series Exclusive优优独播剧场",
        "优优独播剧场——YoYo Television Series Exclusive",
        "erm",
        "hello",
        "hi",
        "hey",
        "goodbye",
        "see you",
        "good night",
        "nice",
        "great",
        "good",
        "wonderful",
        "beautiful",
        "amazing",
        "subscribe",
        "I'll see you next time.",
        "我",
        "你",
        "嗯",
        "哦",
        "啊",
        "好",
        "Obrigado",
        "按订阅 继续合発舞",
    }
    GRATITUDE_HALLUCINATION_PHRASES = {
        "thank you",
        "thank you very much",
        "thanks",
        "thanks very much",
        "thanks for watching",
        "thank you for watching",
        "goodbye",
        "see you",
        "bye bye",
    }
    MIXED_INTERPRETATION_NOISE_PHRASES = {
        ".",
        "thank you",
        "You",
        "you",
        "let's go",
        "stop it",
        "it's okay",
        "hasta la próxima",
        "c'est un simple qui est fan",
        "com os nossos filhos de grandeza",
        "продолжение следует",
        "yeah",
        "exactly",
        "right",
        "sure",
        "yes",
        "ok",
        "okay",
        "no",
        "well",
        "so",
        "hello",
        "hi",
        "hey",
        "goodbye",
        "good morning",
        "good afternoon",
        "nice",
        "great",
        "mhm",        # 嗯哼（常见清嗓幻觉）
        "mm-hmm",
        "hi ho zang",
    }
    HARD_DROP_HALLUCINATION_PHRASES = (
        "优优独播剧场",
        "YoYo Television Series Exclusive",
        "今天年纪归宽市原围会提供",
        "市场—— wears-mêmes request typlaş用比较 Nordic掉",
    )
    MAX_BOUNDARY_DEDUPE_WORDS = 6
    MAX_SHORT_GRATITUDE_SECONDS = 0.5
    HOTWORD_DOMINANCE_THRESHOLD = 0.80
    HOTWORD_NO_SPEECH_THRESHOLD = 0.35
    HOTWORD_REPEAT_THRESHOLD = 3
    MIN_HOTWORD_MATCH_CHARS = 2
    MIXED_NOISE_MIN_EXTENDED_LATIN_WORDS = 2
    MIXED_NOISE_MIN_SCRIPT_SWITCHES = 3
    MIXED_NOISE_MIN_LATIN_WORDS = 4
    SEGMENTATION_TEXT_PREVIEW_LIMIT = 80
    SHORT_COMPLETED_FRAGMENT_SECONDS = 1.0
    SHORT_FRAGMENT_HOLD_SECONDS = 0.7
    SHORT_FRAGMENT_MAX_GAP_SECONDS = 1.0
    MIN_NEW_AUDIO_SECONDS = 0.0
    SENTENCE_COMPLETION_STABLE_OBSERVATIONS = 2
    SENTENCE_COMPLETION_TRAILING_SILENCE_SECONDS = 0.6

    def __init__(
        self,
        client_uid,
        websocket,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
        diarization=None,
        word_timestamps=False,
        min_segment_rms=0.0015,
        max_incomplete_segment_seconds=0.0,
        sentence_completion_min_seconds=0.0,
        min_transcription_chunk_seconds=1.0,
        stable_utterance_ids=False,
        hotword_terms=None,
        max_pending_audio_seconds=None,
        segmentation_profile_v2=False,
        short_fragment_hold_seconds=SHORT_FRAGMENT_HOLD_SECONDS,
        min_new_audio_seconds=MIN_NEW_AUDIO_SECONDS,
    ):
        self.client_uid = client_uid
        self.websocket = websocket
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        if min_segment_rms is None:
            min_segment_rms = 0.0015
        self.min_segment_rms = max(0.0, float(min_segment_rms))
        if max_incomplete_segment_seconds is None:
            max_incomplete_segment_seconds = 0.0
        self.max_incomplete_segment_seconds = max(0.0, float(max_incomplete_segment_seconds))
        if sentence_completion_min_seconds is None:
            sentence_completion_min_seconds = 0.0
        self.sentence_completion_min_seconds = max(0.0, float(sentence_completion_min_seconds))
        if max_pending_audio_seconds is None:
            max_pending_audio_seconds = self.MAX_PENDING_AUDIO_SECONDS
        self.max_pending_audio_seconds = min(
            self.MAX_CONFIGURABLE_PENDING_AUDIO_SECONDS,
            max(1.0, float(max_pending_audio_seconds)),
        )
        if min_transcription_chunk_seconds is None:
            min_transcription_chunk_seconds = 1.0
        self.min_transcription_chunk_seconds = max(0.1, float(min_transcription_chunk_seconds))
        self.diarization = diarization
        self.word_timestamps = word_timestamps

        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ""
        self.prev_out = ""
        self.exit = False
        self.same_output_count = 0
        self.transcript = []
        self.end_time_for_same_output = None
        self.translation_queue = translation_queue
        self.translation_draft_callback = None
        self.asr_finalization_requested = False
        self.asr_finalization_completed = threading.Event()
        self.asr_finalization_status = None
        self.admin_status_callback = None
        self.opencc_converter = self._create_opencc_converter()
        self.stable_utterance_ids = bool(stable_utterance_ids)
        self.utterance_sequence = 0
        self.current_utterance_id = None
        self.hotword_terms = tuple(str(term or "") for term in (hotword_terms or []) if str(term or "").strip())
        self.hotword_match_terms = self._prepare_hotword_match_terms(self.hotword_terms)
        self.segmentation_profile_v2 = bool(segmentation_profile_v2)
        self.pending_completed_segment = None
        self.short_fragment_hold_seconds = max(0.0, float(short_fragment_hold_seconds or 0.0))
        self.min_new_audio_seconds = max(0.0, float(min_new_audio_seconds or 0.0))
        self.last_transcription_audio_end = None

        # Optional post-processing callable for segments.
        # If set, called with a segment dict and must return a segment dict.
        # Allows external projects to plug in custom post-processing
        # (e.g. PII redaction, formatting, diarization) without modifying
        # WhisperLive's core code.
        self.segment_post_processor = None
        self.completed_text_post_processor = None

        # threading
        self.lock = threading.Lock()

    def _segmentation_diagnostic(self, event, **metadata):
        metadata["uid"] = self.client_uid
        if "text" in metadata:
            metadata["text"] = repr(str(metadata["text"] or "").strip()[:self.SEGMENTATION_TEXT_PREVIEW_LIMIT])
        logging.info(
            "[SEGMENTATION_DIAGNOSTIC] event=%s %s",
            event,
            " ".join(f"{key}={value}" for key, value in sorted(metadata.items())),
        )

    @staticmethod
    def _text_has_sentence_boundary(text):
        return bool(re.search(r"[。！？.!?][\s\"'“”‘’）)\]}]*$", str(text or "").strip()))

    @classmethod
    def _log_opencc_unavailable_once(cls, message):
        if cls.OPENCC_UNAVAILABLE_LOGGED:
            return
        logging.warning(message)
        cls.OPENCC_UNAVAILABLE_LOGGED = True

    def _create_opencc_converter(self):
        if OpenCC is None:
            self._log_opencc_unavailable_once(
                "OpenCC is not installed; ASR text will not be converted to Simplified Chinese."
            )
            return None
        try:
            return OpenCC(self.OPENCC_CONFIG)
        except Exception as e:
            logging.warning("Failed to initialize OpenCC config %s: %s", self.OPENCC_CONFIG, e)
            return None

    def normalize_asr_text(self, text):
        if text is None or self.opencc_converter is None:
            return text
        try:
            return self.opencc_converter.convert(str(text))
        except Exception as e:
            logging.warning("OpenCC conversion failed: %s", e)
            return text

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            released_segments = self.flush_pending_completed_segments()
            if released_segments:
                self.send_transcription_to_client(released_segments)

            if self.frames_np is None:
                if self.asr_finalization_requested:
                    self._finish_asr_finalization("completed")
                time.sleep(0.05)
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            finalizing = self.asr_finalization_requested
            if duration < self.min_transcription_chunk_seconds and not finalizing:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            if duration <= 0:
                if finalizing:
                    self._finish_asr_finalization("completed")
                time.sleep(0.05)
                continue
            audio_end = self.timestamp_offset + duration
            if (
                not finalizing
                and self.last_transcription_audio_end is not None
                and audio_end - self.last_transcription_audio_end < self.min_new_audio_seconds
            ):
                time.sleep(0.05)
                continue
            try:
                input_sample = input_bytes.copy()
                self.last_transcription_audio_end = audio_end
                t0 = time.time()
                result = self.transcribe_audio(input_sample)

                if result is None or (
                    self.language is None
                    and not getattr(self, "allow_language_auto_per_chunk", False)
                ):
                    self.timestamp_offset += duration
                    if finalizing:
                        self._finish_asr_finalization("completed")
                    time.sleep(0.25)    # wait for voice activity, result is None when no voice activity
                    continue
                wl_metrics.track_transcription_latency(time.time() - t0)
                wl_metrics.track_audio_processed(duration)
                self.handle_transcription_output(result, duration, force_complete_last=finalizing)
                if finalizing:
                    self._finish_asr_finalization("completed")

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                wl_metrics.track_error("transcription")
                if finalizing:
                    self._finish_asr_finalization("failed")
                time.sleep(0.01)

    def transcribe_audio(self):
        raise NotImplementedError

    def handle_transcription_output(self, result, duration, force_complete_last=False):
        raise NotImplementedError

    def request_asr_finalization(self):
        self.asr_finalization_status = None
        self.asr_finalization_completed.clear()
        self.asr_finalization_requested = True
        logging.info("[ASR_FINALIZE_REQUESTED] uid=%s", self.client_uid)

    def _finish_asr_finalization(self, status="completed"):
        if not self.asr_finalization_requested and self.asr_finalization_completed.is_set():
            return
        self.asr_finalization_status = status
        self.asr_finalization_requested = False
        self.asr_finalization_completed.set()
        logging.info("[ASR_FINALIZE_%s] uid=%s", str(status or "completed").upper(), self.client_uid)

    def wait_for_asr_finalization(self, timeout=0):
        if not self.asr_finalization_requested and self.asr_finalization_completed.is_set():
            return self.asr_finalization_status or "completed"
        if self.asr_finalization_completed.wait(max(0.0, float(timeout or 0.0))):
            return self.asr_finalization_status or "completed"
        logging.warning("[ASR_FINALIZE_TIMEOUT] uid=%s timeout=%.2f", self.client_uid, float(timeout or 0.0))
        return "timed_out"

    def format_segment(self, start, end, text, completed=False, speaker=None, words=None):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.
            speaker (str, optional): Speaker label from diarization.
            words (list, optional): Word-level timestamps and probabilities.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places and the 'text'
                of the transcription.
        """
        seg = {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': self.normalize_asr_text(text),
            'completed': completed,
            'language': getattr(self, "current_language", None) or getattr(self, "language", None)
        }
        if speaker is not None:
            seg['speaker'] = speaker
        if words is not None:
            seg['words'] = words
        return seg

    def _ensure_utterance_id(self, start):
        if not self.stable_utterance_ids:
            return None
        if self.current_utterance_id is None:
            self.utterance_sequence += 1
            self.current_utterance_id = (
                f"{self.client_uid}:{self.utterance_sequence}:{float(start):.3f}"
            )
        return self.current_utterance_id

    def _attach_utterance_id(self, segment, start, utterance_id=None):
        if not self.stable_utterance_ids:
            return segment
        segment["utterance_id"] = utterance_id or self._ensure_utterance_id(start)
        return segment

    def _finish_utterance(self):
        if self.stable_utterance_ids:
            self.current_utterance_id = None

    def _notify_translation_draft_segment(self, segment):
        callback = self.translation_draft_callback
        if callback is None or not segment:
            return
        try:
            callback(segment.copy())
        except Exception as error:
            logging.error(
                "[TRANSLATION_DRAFT_CALLBACK_ERROR] uid=%s utterance_id=%s error=%s",
                self.client_uid,
                segment.get("utterance_id"),
                str(error)[:160],
            )

    def _post_process_completed_text(self, text, reason="completed"):
        processor = self.completed_text_post_processor
        if processor is None or not str(text or "").strip():
            return text
        try:
            result = processor(text, reason=reason)
            return text if result is None else str(result)
        except Exception as error:
            logging.error(
                "[ASR_TEXT_POST_PROCESSOR_ERROR] uid=%s reason=%s error=%s",
                self.client_uid,
                reason,
                str(error)[:160],
            )
            return text

    @staticmethod
    def _segment_time(segment, key):
        try:
            return float(segment.get(key, 0.0))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def _completed_segment_has_terminal_boundary(self, segment):
        return bool(re.search(r"[！？!?][\s\"'“”‘’）)\]}]*$", str(segment.get("text") or "").strip()))

    def _completed_segments_compatible(self, previous, current):
        if previous.get("language") != current.get("language"):
            return False
        if previous.get("speaker") != current.get("speaker"):
            return False
        gap = self._segment_time(current, "start") - self._segment_time(previous, "end")
        return 0.0 <= gap <= self.SHORT_FRAGMENT_MAX_GAP_SECONDS

    def _merge_completed_segments(self, previous, current):
        merged = previous.copy()
        merged["end"] = current["end"]
        previous_text = str(previous.get("text") or "").rstrip()
        current_text = self._dedupe_leading_word_overlap(previous_text, current.get("text"))
        separator = "" if not previous_text or not current_text or re.search(r"[\u3400-\u9fff]$", previous_text) else " "
        merged["text"] = f"{previous_text.rstrip('，,;；:：')}{separator}{str(current_text).lstrip()}".strip()
        if previous.get("words") is not None or current.get("words") is not None:
            merged["words"] = list(previous.get("words") or []) + list(current.get("words") or [])
        return merged

    def _emit_completed_segment(self, segment, reason):
        with self.lock:
            self.text.append(segment["text"])
            self.transcript.append(segment)
        self._notify_translation_draft_segment(segment)
        if self.translation_queue:
            try:
                self.translation_queue.put(segment.copy(), timeout=0.1)
            except queue.Full:
                logging.warning("Translation queue is full, skipping segment")
        self._segmentation_diagnostic(
            "completion",
            reason=reason,
            start=segment.get("start"),
            end=segment.get("end"),
            text=segment.get("text"),
        )
        return segment

    def flush_pending_completed_segments(self, force=False):
        with self.lock:
            pending = self.pending_completed_segment
            if pending is None:
                return []
            if not force and time.monotonic() - pending["held_at"] < self.short_fragment_hold_seconds:
                return []
            self.pending_completed_segment = None
        self._segmentation_diagnostic("short_fragment_release", reason="flush" if force else "hold_expired", text=pending["segment"].get("text"))
        return [self._emit_completed_segment(pending["segment"], pending["reason"])]

    def _stage_completed_segment(self, segment, reason):
        released = self.flush_pending_completed_segments()
        if not self.segmentation_profile_v2:
            return released + [self._emit_completed_segment(segment, reason)]
        with self.lock:
            pending = self.pending_completed_segment
            if pending is not None:
                self.pending_completed_segment = None
        if pending is not None:
            if (
                not self._completed_segment_has_terminal_boundary(pending["segment"])
                and not self._completed_segment_has_terminal_boundary(segment)
                and self._completed_segments_compatible(pending["segment"], segment)
            ):
                merged = self._merge_completed_segments(pending["segment"], segment)
                self._segmentation_diagnostic("short_fragment_merge", text=merged.get("text"))
                return released + [self._emit_completed_segment(merged, "short_fragment_merge")]
            self._segmentation_diagnostic("short_fragment_release", reason="incompatible", text=pending["segment"].get("text"))
            released.append(self._emit_completed_segment(pending["segment"], pending["reason"]))
        duration = self._segment_time(segment, "end") - self._segment_time(segment, "start")
        if duration <= self.SHORT_COMPLETED_FRAGMENT_SECONDS and not self._completed_segment_has_terminal_boundary(segment):
            with self.lock:
                self.pending_completed_segment = {"segment": segment, "reason": reason, "held_at": time.monotonic()}
            self._segmentation_diagnostic("short_fragment_hold", duration=f"{duration:.3f}", text=segment.get("text"))
            return released
        return released + [self._emit_completed_segment(segment, reason)]

    def _has_sentence_completion_trailing_silence(self, rel_end, duration):
        if duration - rel_end < self.SENTENCE_COMPLETION_TRAILING_SILENCE_SECONDS:
            return False
        rms = self._audio_rms_for_relative_range(rel_end, duration, duration)
        return rms is not None and rms < self.min_segment_rms

    def add_frames(self, frame_np):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """
        self.lock.acquire()
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
            # check timestamp offset(should be >= self.frame_offset)
            # this basically means that there is no speech as timestamp offset hasnt updated
            # and is less than frame_offset
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)
        self.trim_pending_audio_if_needed()
        self.lock.release()

    def trim_pending_audio_if_needed(self):
        if self.frames_np is None or self.max_pending_audio_seconds <= 0:
            return

        frames_duration = self.frames_np.shape[0] / self.RATE
        processed_seconds = max(0.0, self.timestamp_offset - self.frames_offset)
        pending_seconds = max(0.0, frames_duration - processed_seconds)
        if pending_seconds <= self.max_pending_audio_seconds:
            return

        new_timestamp_offset = self.frames_offset + frames_duration - self.max_pending_audio_seconds
        dropped_seconds = max(0.0, new_timestamp_offset - self.timestamp_offset)
        if dropped_seconds <= 0:
            return

        self.timestamp_offset = new_timestamp_offset
        logging.warning(
            "[REALTIME_DROP] uid=%s pending=%.2fs keep=%.2fs dropped=%.2fs",
            self.client_uid,
            pending_seconds,
            self.max_pending_audio_seconds,
            dropped_seconds,
        )
        self._segmentation_diagnostic(
            "realtime_audio_drop",
            pending=f"{pending_seconds:.2f}",
            kept=f"{self.max_pending_audio_seconds:.2f}",
            dropped=f"{dropped_seconds:.2f}",
        )

    def clip_audio_if_no_valid_segment(self):
        """
        Update the timestamp offset based on audio buffer status.
        Clip audio if the current chunk exceeds 30 seconds, this basically implies that
        no valid segment for the last 30 seconds from whisper
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing based on the current offsets.

        Calculates which part of the audio data should be processed next, based on
        the difference between the current timestamp offset and the frame's offset, scaled by
        the audio sample rate (RATE). It then returns this chunk of audio data along with its
        duration in seconds.

        Returns:
            tuple: A tuple containing:
                - input_bytes (np.ndarray): The next chunk of audio data to be processed.
                - duration (float): The duration of the audio chunk in seconds.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles the recent segments of transcribed text, ensuring that only the
        specified number of the most recent segments are included. It also appends the most
        recent segment of text if provided (which is considered incomplete because of the possibility
        of the last word being truncated in the audio chunk).

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def get_audio_chunk_duration(self, input_bytes):
        """
        Calculates the duration of the provided audio chunk.

        Args:
            input_bytes (numpy.ndarray): The audio chunk for which to calculate the duration.

        Returns:
            float: The duration of the audio chunk in seconds.
        """
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.

        This method formats the transcription segments into a JSON object and attempts to send
        this object to the client. If an error occurs during the send operation, it logs the error.

        If a ``segment_post_processor`` callable is set, each segment is passed through it
        before sending. The callable receives a segment dict and must return a segment dict.

        Returns:
            segments (list): A list of transcription segments to be sent to the client.
        """
        segments = [
            seg for seg in segments
            if not self._is_hard_drop_hallucination_text((seg or {}).get("text"))
        ]

        if self.segment_post_processor is not None:
            processed = []
            for seg in segments:
                try:
                    result = self.segment_post_processor(seg)
                    processed.append(result if result is not None else seg)
                except Exception as e:
                    logging.error(f"[ERROR]: segment_post_processor failed: {e}")
                    processed.append(seg)
            segments = processed

        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "segments": segments,
                })
            )
            if self.admin_status_callback:
                try:
                    self.admin_status_callback(segments)
                except Exception as e:
                    logging.error(f"[ERROR]: admin status update failed: {e}")
            for seg in segments:
                wl_metrics.track_segment_emitted(completed=seg.get("completed", False))
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def disconnect(self):
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.DISCONNECT
        }))

    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logging.info("Cleaning up.")
        self.exit = True

    def get_segment_no_speech_prob(self, segment):
        return getattr(segment, "no_speech_prob", 0)

    def get_segment_start(self, segment):
        return getattr(segment, "start", getattr(segment, "start_ts", 0))

    def get_segment_end(self, segment):
        return getattr(segment, "end", getattr(segment, "end_ts", 0))

    def _audio_rms_for_relative_range(self, start, end, duration):
        if self.min_segment_rms <= 0 or self.frames_np is None:
            return None

        start = max(0.0, min(float(start or 0.0), duration))
        end = max(start, min(float(end or 0.0), duration))
        if end <= start:
            return 0.0

        with self.lock:
            samples_offset = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            start_sample = samples_offset + int(start * self.RATE)
            end_sample = samples_offset + int(end * self.RATE)
            audio_slice = self.frames_np[start_sample:end_sample].copy()

        if audio_slice.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_slice, dtype=np.float64))))

    @classmethod
    def _normalized_phrase(cls, text):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s']+", " ", str(text or "").lower())).strip()

    @classmethod
    def _compact_hotword_text(cls, text):
        normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
        return re.sub(r"[^\w]+", "", normalized, flags=re.UNICODE)

    @classmethod
    def _prepare_hotword_match_terms(cls, terms):
        prepared = []
        seen = set()
        for term in terms or []:
            compact = cls._compact_hotword_text(term)
            if len(compact) < cls.MIN_HOTWORD_MATCH_CHARS or compact in seen:
                continue
            seen.add(compact)
            prepared.append(compact)
        prepared.sort(key=len, reverse=True)
        return tuple(prepared)

    @staticmethod
    def _merged_coverage_length(spans):
        if not spans:
            return 0
        spans = sorted(spans)
        total = 0
        cur_start, cur_end = spans[0]
        for start, end in spans[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
                continue
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        total += cur_end - cur_start
        return total

    def _hotword_dominance(self, text):
        if not self.hotword_match_terms:
            return 0.0
        compact_text = self._compact_hotword_text(text)
        if not compact_text:
            return 0.0
        spans = []
        for term in self.hotword_match_terms:
            start = 0
            while True:
                index = compact_text.find(term, start)
                if index < 0:
                    break
                end = index + len(term)
                spans.append((index, end))
                start = index + 1
        covered = self._merged_coverage_length(spans)
        return covered / len(compact_text) if covered else 0.0

    def _max_repeated_hotword_count(self, text):
        if not self.hotword_match_terms:
            return 0
        compact_text = self._compact_hotword_text(text)
        if not compact_text:
            return 0
        max_count = 0
        for term in self.hotword_match_terms:
            if not term:
                continue
            count = 0
            start = 0
            while True:
                index = compact_text.find(term, start)
                if index < 0:
                    break
                count += 1
                start = index + len(term)
            max_count = max(max_count, count)
        return max_count

    def _max_consecutive_hotword_count(self, text):
        if not self.hotword_match_terms:
            return 0
        compact_text = self._compact_hotword_text(text)
        if not compact_text:
            return 0
        max_consecutive = 0
        for term in self.hotword_match_terms:
            if not term:
                continue
            term_len = len(term)
            positions = []
            start = 0
            while True:
                index = compact_text.find(term, start)
                if index < 0:
                    break
                positions.append(index)
                start = index + 1
            if not positions:
                continue
            consecutive = 1
            for i in range(1, len(positions)):
                if positions[i] == positions[i - 1] + term_len:
                    consecutive += 1
                else:
                    max_consecutive = max(max_consecutive, consecutive)
                    consecutive = 1
            max_consecutive = max(max_consecutive, consecutive)
        return max_consecutive

    def _log_hotword_hallucination_drop(self, stage, reason, text, dominance, no_speech_prob, rms, rms_threshold, repeat_count=0):
        logging.info(
            "[HOTWORD_HALLUCINATION_DROP] uid=%s stage=%s reason=%s dominance=%.3f repeat_count=%d no_speech=%.3f rms=%s rms_threshold=%.6f text=%r",
            self.client_uid,
            stage,
            reason,
            dominance,
            repeat_count,
            no_speech_prob,
            "none" if rms is None else f"{rms:.6f}",
            rms_threshold,
            str(text or "").strip()[:80],
        )
        self._segmentation_diagnostic("hallucination_drop", stage=stage, reason=reason, text=text)

    def _hotword_hallucination_drop_reason(self, segment, start, end, duration, text, stage):
        consecutive_count = self._max_consecutive_hotword_count(text)
        if consecutive_count >= self.HOTWORD_REPEAT_THRESHOLD:
            dominance = self._hotword_dominance(text)
            no_speech_prob = float(self.get_segment_no_speech_prob(segment) or 0.0)
            self._log_hotword_hallucination_drop(
                stage,
                "consecutive_repeated_hotword",
                text,
                dominance,
                no_speech_prob,
                None,
                self.min_segment_rms,
                repeat_count=consecutive_count,
            )
            return "consecutive_repeated_hotword"

        dominance = self._hotword_dominance(text)
        if dominance < self.HOTWORD_DOMINANCE_THRESHOLD:
            return None

        no_speech_prob = float(self.get_segment_no_speech_prob(segment) or 0.0)
        repeat_count = self._max_repeated_hotword_count(text)
        if repeat_count >= self.HOTWORD_REPEAT_THRESHOLD:
            self._log_hotword_hallucination_drop(
                stage,
                "repeated_hotword",
                text,
                dominance,
                no_speech_prob,
                None,
                self.min_segment_rms,
                repeat_count=repeat_count,
            )
            return "repeated_hotword"

        no_speech_threshold = min(float(self.no_speech_thresh), self.HOTWORD_NO_SPEECH_THRESHOLD)
        if no_speech_prob >= no_speech_threshold:
            self._log_hotword_hallucination_drop(
                stage,
                "no_speech",
                text,
                dominance,
                no_speech_prob,
                None,
                self.min_segment_rms,
                repeat_count=repeat_count,
            )
            return "no_speech"

        rms = self._audio_rms_for_relative_range(start, end, duration)
        if rms is not None and rms < self.min_segment_rms:
            self._log_hotword_hallucination_drop(
                stage,
                "low_energy",
                text,
                dominance,
                no_speech_prob,
                rms,
                self.min_segment_rms,
                repeat_count=repeat_count,
            )
            return "low_energy"
        return None

    @classmethod
    def _is_silence_hallucination_phrase(cls, text):
        normalized = cls._normalized_phrase(text)
        return normalized in cls.SILENCE_HALLUCINATION_PHRASES

    @classmethod
    def _is_gratitude_hallucination_phrase(cls, text):
        normalized = cls._normalized_phrase(text)
        return normalized in cls.GRATITUDE_HALLUCINATION_PHRASES

    @classmethod
    def _is_hard_drop_hallucination_text(cls, text):
        value = str(text or "")
        if not value.strip():
            return False
        normalized = cls._normalized_phrase(value)
        compact = cls._compact_hotword_text(value)
        for phrase in cls.HARD_DROP_HALLUCINATION_PHRASES:
            phrase_normalized = cls._normalized_phrase(phrase)
            phrase_compact = cls._compact_hotword_text(phrase)
            if phrase_normalized and phrase_normalized in normalized:
                return True
            if phrase_compact and phrase_compact in compact:
                return True
        return False

    def _should_hard_drop_hallucination_text(self, text, stage):
        if not self._is_hard_drop_hallucination_text(text):
            return False
        logging.info(
            "[HARD_HALLUCINATION_DROP] uid=%s stage=%s text=%r",
            self.client_uid,
            stage,
            str(text or "").strip()[:80],
        )
        self._segmentation_diagnostic("hallucination_drop", stage=stage, reason="hard", text=text)
        return True

    @staticmethod
    def _has_zh_script(text):
        return bool(re.search(r"[一-鿿]", str(text or "")))

    @staticmethod
    def _has_ascii_word(text):
        return bool(re.search(r"[A-Za-z0-9]", str(text or "")))

    @staticmethod
    def _is_latin_letter(char):
        return bool(char and char.isalpha() and "LATIN" in unicodedata.name(char, ""))

    @classmethod
    def _is_latin_extended_mixed_noise_text(cls, text):
        value = unicodedata.normalize("NFKC", str(text or ""))
        if not cls._has_zh_script(value) or not cls._has_ascii_word(value):
            return False

        scripts = []
        latin_words = []
        current_latin_word = []

        def flush_latin_word():
            if current_latin_word:
                latin_words.append("".join(current_latin_word))
                current_latin_word.clear()

        for char in value:
            if cls._has_zh_script(char):
                flush_latin_word()
                script = "zh"
            elif cls._is_latin_letter(char):
                current_latin_word.append(char)
                script = "latin"
            else:
                flush_latin_word()
                continue
            if not scripts or scripts[-1] != script:
                scripts.append(script)
        flush_latin_word()

        extended_latin_words = [
            word for word in latin_words
            if any(ord(char) > 127 and cls._is_latin_letter(char) for char in word)
            and any(char.islower() for char in word)
        ]
        return (
            len(extended_latin_words) >= cls.MIXED_NOISE_MIN_EXTENDED_LATIN_WORDS
            and max(0, len(scripts) - 1) >= cls.MIXED_NOISE_MIN_SCRIPT_SWITCHES
            and len(latin_words) >= cls.MIXED_NOISE_MIN_LATIN_WORDS
        )

    @staticmethod
    def _has_korean_script(text):
        return bool(re.search(r"[가-힯]", str(text or "")))

    @staticmethod
    def _has_japanese_kana(text):
        return bool(re.search(r"[぀-ヿ]", str(text or "")))

    @staticmethod
    def _has_cyrillic_script(text):
        return bool(re.search(r"[Ѐ-ӿ]", str(text or "")))

    @staticmethod
    def _is_zh_en_language(language):
        language = str(language or "").strip().lower()
        return language == "zh" or language.startswith("zh-") or language == "en" or language.startswith("en-")

    def _log_mixed_interpretation_noise_drop(self, reason, text):
        logging.info(
            "[MIXED_INTERPRETATION_NOISE_DROP] uid=%s reason=%s raw_language=%s zh=%.3f en=%.3f text=%r",
            self.client_uid,
            reason,
            getattr(self, "current_raw_language", None),
            (getattr(self, "current_zh_en_candidates", {}) or {}).get("zh", 0.0),
            (getattr(self, "current_zh_en_candidates", {}) or {}).get("en", 0.0),
            str(text or "").strip()[:80],
        )

    def _is_mixed_interpretation_noise_text(self, text):
        if not getattr(self, "mixed_interpretation", False):
            return False
        value = str(text or "").strip()
        normalized = self._normalized_phrase(value)
        if not normalized:
            return True
        if normalized in self.MIXED_INTERPRETATION_NOISE_PHRASES:
            self._log_mixed_interpretation_noise_drop("phrase", value)
            return True

        has_zh = self._has_zh_script(value)
        has_ascii = self._has_ascii_word(value)
        has_korean = self._has_korean_script(value)
        has_japanese = self._has_japanese_kana(value)
        has_cyrillic = self._has_cyrillic_script(value)
        has_foreign_script = has_korean or has_japanese or has_cyrillic

        if has_foreign_script:
            self._log_mixed_interpretation_noise_drop("foreign_script", value)
            return True

        if self._is_latin_extended_mixed_noise_text(value):
            self._log_mixed_interpretation_noise_drop("latin_extended_mixed_structure", value)
            return True

        has_zh_or_ascii = has_zh or has_ascii
        has_other_letters = bool(re.search(r"[^\W\d_]", value, re.UNICODE))
        if has_other_letters and not has_zh_or_ascii:
            self._log_mixed_interpretation_noise_drop("non_zh_en", value)
            return True

        raw_language = getattr(self, "current_raw_language", None)
        if raw_language and not self._is_zh_en_language(raw_language):
            zh_en_candidates = getattr(self, "current_zh_en_candidates", {}) or {}
            best_zh_en = max(zh_en_candidates.values(), default=0.0)
            if best_zh_en < 0.20 and has_other_letters and not has_zh:
                self._log_mixed_interpretation_noise_drop("raw_language", value)
                return True

        return False

    def _is_probable_gratitude_hallucination(self, text, rel_start, rel_end, has_following_segment):
        if not self._is_gratitude_hallucination_phrase(text):
            return False

        duration = max(0.0, float(rel_end or 0.0) - float(rel_start or 0.0))
        if duration < self.MAX_SHORT_GRATITUDE_SECONDS:
            reason = "short"
        elif has_following_segment:
            reason = "middle"
        else:
            return False

        logging.info(
            "[GRATITUDE_HALLUCINATION_DROP] uid=%s reason=%s duration=%.3fs text=%r",
            self.client_uid,
            reason,
            duration,
            str(text or "").strip()[:80],
        )
        self._segmentation_diagnostic("hallucination_drop", stage="gratitude", reason=reason, text=text)
        return True

    @staticmethod
    def _word_spans(text):
        return list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text or "")))

    @classmethod
    def _dedupe_leading_word_overlap(cls, previous_text, current_text):
        current = str(current_text or "")
        previous_words = [m.group(0).lower() for m in cls._word_spans(previous_text)]
        current_matches = cls._word_spans(current)
        current_words = [m.group(0).lower() for m in current_matches]
        max_overlap = min(cls.MAX_BOUNDARY_DEDUPE_WORDS, len(previous_words), len(current_words))
        for size in range(max_overlap, 0, -1):
            if previous_words[-size:] == current_words[:size]:
                cut_at = current_matches[size - 1].end()
                deduped = current[:current_matches[0].start()] + current[cut_at:].lstrip()
                logging.info(
                    "[BOUNDARY_DEDUPE] overlap_words=%d previous=%r current=%r deduped=%r",
                    size,
                    str(previous_text or "").strip()[-80:],
                    current.strip()[:80],
                    deduped.strip()[:80],
                )
                return deduped
        return current

    def _dedupe_completed_text(self, text):
        if not self.text:
            return text
        return self._dedupe_leading_word_overlap(self.text[-1], text)

    def _is_low_energy_range(self, start, end, duration, text):
        rms = self._audio_rms_for_relative_range(start, end, duration)
        if rms is None:
            return False

        threshold = self.min_segment_rms
        if self._is_silence_hallucination_phrase(text):
            threshold = max(threshold, self.min_segment_rms * 2.5)

        if rms >= threshold:
            return False
        logging.info(
            "[LOW_ENERGY_SEGMENT_DROP] uid=%s rms=%.6f threshold=%.6f text=%r",
            self.client_uid,
            rms,
            threshold,
            str(text or "").strip()[:80],
        )
        self._segmentation_diagnostic(
            "hallucination_low_energy_drop" if self._is_silence_hallucination_phrase(text) else "ordinary_low_energy_drop",
            rms=f"{rms:.6f}", threshold=f"{threshold:.6f}", text=text,
        )
        return True

    def _identify_speaker(self, segment):
        """Run diarization on a segment's audio slice if diarization is enabled.

        Returns:
            str or None: Speaker label, or None if diarization is disabled or audio unavailable.
        """
        if self.diarization is None or self.frames_np is None:
            return None
        try:
            seg_start = self.get_segment_start(segment)
            seg_end = self.get_segment_end(segment)
            start_sample = int(seg_start * self.RATE)
            end_sample = int(seg_end * self.RATE)
            samples_offset = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            audio_slice = self.frames_np[samples_offset + start_sample:samples_offset + end_sample]
            if len(audio_slice) < self.RATE * 0.3:
                return None
            return self.diarization.identify_speaker(audio_slice, self.RATE)
        except Exception as e:
            logging.error(f"Diarization error: {e}")
            return None

    def _extract_words(self, segment, time_offset):
        """Extracts word-level timestamps from a segment if word_timestamps is enabled."""
        if not self.word_timestamps:
            return None
        words = getattr(segment, "words", None)
        if not words:
            return None
        return [
            {
                "word": self.normalize_asr_text(w.word),
                "start": "{:.3f}".format(time_offset + w.start),
                "end": "{:.3f}".format(time_offset + w.end),
                "probability": round(w.probability, 4),
            }
            for w in words
        ]

    def update_segments(self, segments, duration, force_complete_last=False):
        """
        Processes the segments from Whisper and updates the transcript.
        Uses helper methods to account for differences between backends.

        Args:
            segments (list): List of segments returned by the transcriber.
            duration (float): Duration of the current audio chunk.

        Returns:
            dict or None: The last processed segment (if any).
        """
        offset = None
        self.current_out = ''
        last_segment = None

        # Process complete segments only if there are more than one
        # and if the last segment's no_speech_prob is below the threshold.
        if len(segments) > 1 and self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            completed_candidates = segments[:-1]
            completed_utterance_id = None
            if self.stable_utterance_ids and completed_candidates:
                with self.lock:
                    completed_start = self.timestamp_offset + self.get_segment_start(completed_candidates[0])
                completed_utterance_id = self._ensure_utterance_id(completed_start)
            for index, s in enumerate(completed_candidates):
                text_ = s.text
                rel_start = self.get_segment_start(s)
                rel_end = min(duration, self.get_segment_end(s))
                has_following_segment = index < len(completed_candidates) - 1 or len(segments) > 1
                with self.lock:
                    start = self.timestamp_offset + rel_start
                    end = self.timestamp_offset + rel_end
                if start >= end:
                    continue
                if self._should_hard_drop_hallucination_text(text_, "completed"):
                    offset = rel_end
                    continue
                if self._hotword_hallucination_drop_reason(s, rel_start, rel_end, duration, text_, "completed"):
                    offset = rel_end
                    continue
                if self.get_segment_no_speech_prob(s) > self.no_speech_thresh:
                    continue
                if self._is_mixed_interpretation_noise_text(text_):
                    offset = rel_end
                    continue
                if self._is_probable_gratitude_hallucination(text_, rel_start, rel_end, has_following_segment):
                    offset = rel_end
                    continue
                if self._is_low_energy_range(rel_start, rel_end, duration, text_):
                    offset = rel_end
                    continue
                text_ = self._dedupe_completed_text(text_)
                text_ = self._post_process_completed_text(text_, reason="completed")
                if not text_.strip():
                    offset = rel_end
                    continue
                speaker = self._identify_speaker(s)
                words = self._extract_words(s, self.timestamp_offset)
                completed_segment = self.format_segment(start, end, text_, completed=True, speaker=speaker, words=words)
                self._attach_utterance_id(
                    completed_segment,
                    start,
                    utterance_id=completed_utterance_id,
                )
                self._stage_completed_segment(completed_segment, "whisper_segment")
                offset = rel_end
            self._finish_utterance()

        # Process the last segment if its no_speech_prob is acceptable.
        if self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            rel_start = self.get_segment_start(segments[-1])
            rel_end = min(duration, self.get_segment_end(segments[-1]))
            if self._should_hard_drop_hallucination_text(segments[-1].text, "partial"):
                offset = rel_end
            elif self._is_mixed_interpretation_noise_text(segments[-1].text):
                offset = rel_end
            elif self._is_probable_gratitude_hallucination(segments[-1].text, rel_start, rel_end, False):
                offset = rel_end
            elif self._hotword_hallucination_drop_reason(segments[-1], rel_start, rel_end, duration, segments[-1].text, "partial"):
                pass
            elif self._is_low_energy_range(rel_start, rel_end, duration, segments[-1].text):
                offset = rel_end
            elif force_complete_last:
                completed_text = self._dedupe_completed_text(segments[-1].text)
                completed_text = self._post_process_completed_text(completed_text, reason="finalize_complete")
                if self._should_hard_drop_hallucination_text(completed_text, "finalize_complete"):
                    completed_text = ""
                if self._is_mixed_interpretation_noise_text(completed_text):
                    completed_text = ""
                if completed_text.strip() and rel_end > rel_start:
                    with self.lock:
                        start = self.timestamp_offset + rel_start
                        end = self.timestamp_offset + rel_end
                    if not self.text or self.text[-1].strip().lower() != completed_text.strip().lower():
                        speaker = self._identify_speaker(segments[-1])
                        words = self._extract_words(segments[-1], self.timestamp_offset)
                        completed_segment = self.format_segment(start, end, completed_text, completed=True, speaker=speaker, words=words)
                        self._attach_utterance_id(completed_segment, start)
                        self._stage_completed_segment(completed_segment, "finalize_complete")
                        logging.info(
                            "[ASR_FINALIZE_COMPLETE_SEGMENT] uid=%s start=%.3f end=%.3f text=%r",
                            self.client_uid,
                            start,
                            end,
                            completed_text[:80],
                        )
                offset = rel_end
                last_segment = None
                self.current_out = ""
                self._finish_utterance()
            else:
                self.current_out += segments[-1].text
                words = self._extract_words(segments[-1], self.timestamp_offset)
                with self.lock:
                    last_segment = self.format_segment(
                        self.timestamp_offset + rel_start,
                        self.timestamp_offset + rel_end,
                        self.current_out,
                        completed=False,
                        words=words
                    )
                    self._attach_utterance_id(
                        last_segment,
                        self.timestamp_offset + rel_start,
                    )

        if last_segment is not None:
            self._notify_translation_draft_segment(last_segment)

        # Handle repeated output logic.
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_count += 1

            # if we remove the audio because of same output on the nth reptition we might remove the
            # audio thats not yet transcribed so, capturing the time when it was repeated for the first time
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = self.get_segment_end(segments[-1])
            time.sleep(0.1)  # wait briefly for any new voice activity
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # If the same incomplete segment is repeated too many times,
        # append it to the transcript and update the offset.
        if self.same_output_count > self.same_output_threshold:
            repeated_end = min(duration, self.end_time_for_same_output)
            low_energy_repeated = self._is_low_energy_range(0.0, repeated_end, duration, self.current_out)
            if not low_energy_repeated and (not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower()):
                completed_text = self._dedupe_completed_text(self.current_out)
                completed_text = self._post_process_completed_text(completed_text, reason="repeated_complete")
                if not completed_text.strip():
                    completed_text = ""
                if self._should_hard_drop_hallucination_text(completed_text, "repeated_complete"):
                    completed_text = ""
                if self._is_mixed_interpretation_noise_text(completed_text):
                    completed_text = ""
                if completed_text:
                    with self.lock:
                        completed_segment = self.format_segment(
                            self.timestamp_offset,
                            self.timestamp_offset + repeated_end,
                            completed_text,
                            completed=True
                        )
                        self._attach_utterance_id(
                            completed_segment,
                            self.timestamp_offset,
                        )
                    self._stage_completed_segment(completed_segment, "repeated_output")

            self.current_out = ''
            offset = repeated_end
            self.same_output_count = 0
            last_segment = None
            self._finish_utterance()
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        last_rel_start = self.get_segment_start(segments[-1])
        last_rel_end = min(duration, self.get_segment_end(segments[-1]))
        sentence_boundary_complete = (
            self.sentence_completion_min_seconds > 0
            and (
                duration >= self.sentence_completion_min_seconds
                if not self.segmentation_profile_v2
                else last_rel_end - last_rel_start >= self.sentence_completion_min_seconds
            )
            and self._text_has_sentence_boundary(self.current_out)
            and (
                not self.segmentation_profile_v2
                or (
                    self.same_output_count >= self.SENTENCE_COMPLETION_STABLE_OBSERVATIONS
                    and self._has_sentence_completion_trailing_silence(
                        last_rel_end, duration,
                    )
                )
            )
        )
        duration_limit_complete = (
            self.max_incomplete_segment_seconds > 0
            and duration >= self.max_incomplete_segment_seconds
        )
        if (
            offset is None
            and last_segment is not None
            and (sentence_boundary_complete or duration_limit_complete)
            and self.current_out.strip()
        ):
            completion_reason = "sentence_boundary" if sentence_boundary_complete else "duration_limit"
            repeated_end = min(duration, self.get_segment_end(segments[-1]))
            if repeated_end > 0:
                logging.info(
                    "[FORCE_COMPLETE_INCOMPLETE] uid=%s reason=%s duration=%.2fs threshold=%.2fs sentence_min=%.2fs text=%r",
                    self.client_uid,
                    completion_reason,
                    duration,
                    self.max_incomplete_segment_seconds,
                    self.sentence_completion_min_seconds,
                    self.current_out.strip()[:80],
                )
                completed_text = self._dedupe_completed_text(self.current_out)
                completed_text = self._post_process_completed_text(completed_text, reason="force_complete")
                if self._should_hard_drop_hallucination_text(completed_text, "force_complete"):
                    completed_text = ""
                if self._is_mixed_interpretation_noise_text(completed_text):
                    completed_text = ""
                if completed_text.strip():
                    with self.lock:
                        completed_segment = self.format_segment(
                            self.timestamp_offset,
                            self.timestamp_offset + repeated_end,
                            completed_text,
                            completed=True
                        )
                        self._attach_utterance_id(
                            completed_segment,
                            self.timestamp_offset,
                        )
                    self._stage_completed_segment(completed_segment, completion_reason)
                self.current_out = ''
                offset = repeated_end
                self.same_output_count = 0
                last_segment = None
                self._finish_utterance()
                self.end_time_for_same_output = None

        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset
            if last_segment is None:
                self._finish_utterance()

        self._trim_transcript()
        return last_segment

    def _trim_transcript(self):
        """Trims transcript and text lists to prevent unbounded memory growth."""
        if len(self.transcript) > self.MAX_TRANSCRIPT_LENGTH:
            self.transcript = self.transcript[-self.MAX_TRANSCRIPT_LENGTH:]
        if len(self.text) > self.MAX_TRANSCRIPT_LENGTH:
            self.text = self.text[-self.MAX_TRANSCRIPT_LENGTH:]
