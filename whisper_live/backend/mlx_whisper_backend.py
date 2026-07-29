import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass

import soundfile as sf

from whisper_live.backend.base import ServeClientBase


@dataclass
class MLXSegment:
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0


class ServeClientMLXWhisper(ServeClientBase):
    MODEL_MAP = {
        "tiny": "mlx-community/whisper-tiny",
        "base": "mlx-community/whisper-base",
        "small": "mlx-community/whisper-small",
        "medium": "mlx-community/whisper-medium",
        "large-v3": "mlx-community/whisper-large-v3",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        "turbo": "mlx-community/whisper-large-v3-turbo",
    }

    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        language=None,
        client_uid=None,
        model="small",
        initial_prompt=None,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=7,
        translation_queue=None,
        min_segment_rms=0.0015,
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
        )

        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "mlx_whisper is not installed. Install it with: pip install mlx mlx-whisper"
            ) from exc

        self.mlx_whisper = mlx_whisper
        self.language = language
        self.task = task or "transcribe"
        self.initial_prompt = initial_prompt
        self.model_size_or_path = model or "small"
        self.model_ref = self._resolve_model_ref(self.model_size_or_path)
        self._mlx_last_partial = ""
        self._mlx_repeat_count = 0

        logging.info(f"Loading MLX Whisper backend with model: {self.model_ref}")

        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "mlx_whisper",
                }
            )
        )

    def _resolve_model_ref(self, model):
        expanded_model = os.path.abspath(os.path.expanduser(model))
        if os.path.exists(expanded_model):
            return expanded_model
        return self.MODEL_MAP.get(model, model)

    def transcribe_audio(self, input_sample):
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(tmp_path, input_sample, self.RATE, subtype="PCM_16")

            kwargs = {
                "path_or_hf_repo": self.model_ref,
                "word_timestamps": False,
            }
            if self.language:
                kwargs["language"] = self.language
            if self.task:
                kwargs["task"] = self.task
            if self.initial_prompt:
                kwargs["initial_prompt"] = self.initial_prompt

            with self.SINGLE_MODEL_LOCK:
                result = self.mlx_whisper.transcribe(tmp_path, **kwargs)

            if self.language is None and result.get("language"):
                self.language = result["language"]
                self.websocket.send(json.dumps(
                    {"uid": self.client_uid, "language": self.language, "language_prob": 1.0}
                ))

            return self._convert_segments(result, self.get_audio_chunk_duration(input_sample))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _convert_segments(self, result, duration):
        segments = result.get("segments") or []
        converted = []
        for segment in segments:
            text = segment.get("text", "")
            if not text:
                continue
            converted.append(
                MLXSegment(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=text,
                    no_speech_prob=float(segment.get("no_speech_prob", 0.0)),
                )
            )

        if not converted and result.get("text"):
            converted.append(
                MLXSegment(
                    start=0.0,
                    end=duration,
                    text=result["text"],
                    no_speech_prob=0.0,
                )
            )
        return converted

    def handle_transcription_output(self, result, duration, force_complete_last=False):
        if not len(result):
            return

        if len(result) > 1:
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)
            if segments:
                self.send_transcription_to_client(segments)
            return

        last_segment = self._handle_last_segment(result[0], duration)
        segments = self.prepare_segments(last_segment)
        if segments:
            self.send_transcription_to_client(segments)

    def _append_completed_segments(self, segments):
        for segment in segments:
            if self.transcript and self.transcript[-1]["text"].strip().lower() == segment["text"].strip().lower():
                continue
            self.text.append(segment["text"])
            self.transcript.append(segment)
            if self.translation_queue:
                try:
                    self.translation_queue.put(segment.copy(), timeout=0.1)
                except Exception:
                    logging.warning("Translation queue is full, skipping segment")

    def _handle_last_segment(self, segment, duration):
        if self.get_segment_no_speech_prob(segment) > self.no_speech_thresh:
            return None

        text = segment.text.strip()
        if not text:
            return None

        with self.lock:
            start = self.timestamp_offset + self.get_segment_start(segment)
            end = self.timestamp_offset + min(duration, self.get_segment_end(segment))
        if start >= end:
            return None

        last_segment = self.format_segment(start, end, segment.text, completed=False)
        normalized_text = text.lower()
        if normalized_text == self._mlx_last_partial:
            self._mlx_repeat_count += 1
        else:
            self._mlx_last_partial = normalized_text
            self._mlx_repeat_count = 1

        if self._mlx_repeat_count >= self.same_output_threshold:
            completed_segment = self.format_segment(start, end, segment.text, completed=True)
            self._append_completed_segments([completed_segment])
            with self.lock:
                self.timestamp_offset += min(duration, self.get_segment_end(segment))
            self._mlx_last_partial = ""
            self._mlx_repeat_count = 0
            return None

        return last_segment
