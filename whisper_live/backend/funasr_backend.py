import json
import re
import logging
import threading
from dataclasses import dataclass

import torch

from whisper_live.backend.base import ServeClientBase


@dataclass
class FunASRSegment:
    text: str
    start: float
    end: float
    no_speech_prob: float = 0.0


class ServeClientFunASR(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()
    SINGLE_MODEL_INIT_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        device="auto",
        language=None,
        client_uid=None,
        model="iic/SenseVoiceSmall",
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=3,
        translation_queue=None,
        hotwords=None,
        min_segment_rms=0.0015,
        max_incomplete_segment_seconds=6.0,
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
        self.model_size_or_path = model or "iic/SenseVoiceSmall"
        self.language = language or "auto"
        self.task = task
        self.hotwords = hotwords
        self.device = self._resolve_device(device)

        try:
            if single_model:
                with ServeClientFunASR.SINGLE_MODEL_INIT_LOCK:
                    if ServeClientFunASR.SINGLE_MODEL is None:
                        logging.info("Loading shared FunASR model")
                        self.create_model()
                        ServeClientFunASR.SINGLE_MODEL = self.transcriber
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

        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.SERVER_READY,
            "backend": "funasr"
        }))

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

    def _generate(self, **kwargs):
        try:
            return self.transcriber.generate(**kwargs)
        except TypeError:
            if "hotword" in kwargs:
                logging.warning("FunASR model does not accept hotword; retrying without hotwords")
                kwargs.pop("hotword", None)
                return self.transcriber.generate(**kwargs)
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
