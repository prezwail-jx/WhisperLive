import os
import io
import asyncio
import time
import threading
import queue
import json
import functools
import logging
import math
import re
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse, JSONResponse, FileResponse
import uvicorn
from faster_whisper import WhisperModel
import torch

from enum import Enum

from whisper_live import metrics as wl_metrics
import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from whisper_live.vad import VoiceActivityDetector
from whisper_live.backend.base import ServeClientBase
from whisper_live.meeting import (
    AsrTextCorrector,
    MeetingDocConverter,
    MeetingAsrCorrectionStore,
    MeetingHotwordStore,
    MeetingLogStore,
    apply_timeline_offset_to_segments,
    MeetingSummaryService,
    SummaryGenerationError,
    SummaryTemplateStore,
    count_hotwords,
    hotword_text_to_prompt,
    normalize_asr_hotwords,
    normalize_hotword_text,
    parse_hotword_config,
)
from whisper_live.meeting.transcript import TranscriptRevisionConflict

logging.basicConfig(level=logging.INFO)


class ClientManager:
    RECENT_STATUS_TTL_SECONDS = 600
    RECENT_STATUS_MAX_RECORDS = 100

    def __init__(self, max_clients=4, max_connection_time=600):
        """
        Initializes the ClientManager with specified limits on client connections and connection durations.

        Args:
            max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
            max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                                 to 600 seconds (10 minutes).
        """
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.lock = threading.Lock()
        self.client_status = {}
        self.recent_client_status = {}

    def _purge_recent_statuses_locked(self, now=None):
        now = time.time() if now is None else now
        expired = [key for key, status in self.recent_client_status.items()
                   if now - status.get("disconnected_at", now) >= self.RECENT_STATUS_TTL_SECONDS]
        for key in expired:
            del self.recent_client_status[key]
        overflow = len(self.recent_client_status) - self.RECENT_STATUS_MAX_RECORDS
        if overflow > 0:
            oldest = sorted(self.recent_client_status, key=lambda key: self.recent_client_status[key].get("disconnected_at", 0))
            for key in oldest[:overflow]:
                del self.recent_client_status[key]

    @staticmethod
    def _recent_status_key(status):
        return status.get("client_instance_id") or status.get("uid")

    @staticmethod
    def _latest_segment_text(segments):
        if not segments:
            return ""
        for segment in reversed(segments):
            text = segment.get("text", "") if isinstance(segment, dict) else ""
            if text:
                return text
        return ""

    @staticmethod
    def _formal_translation_segments(segments):
        return [
            segment
            for segment in segments or []
            if isinstance(segment, dict) and segment.get("completed") is not False
        ]

    def register_client_status(self, websocket, client, options, backend):
        now = time.time()
        uid = getattr(client, "client_uid", options.get("uid"))
        status = {
            "uid": uid,
            "client_instance_id": options.get("client_instance_id") or "",
            "client_name": options.get("client_name") or options.get("meeting_name") or f"Client-{str(uid)[:8]}",
            "meeting_name": options.get("meeting_name") or "",
            "hotwords_file": options.get("hotwords_file") or "",
            "hotwords_source": options.get("hotwords_source") or "none",
            "hotwords_count": int(options.get("hotwords_count") or count_hotwords(options.get("hotwords"))),
            "hotwords_original_count": int(options.get("hotwords_original_count") or 0),
            "hotwords_rejected_count": int(options.get("hotwords_rejected_count") or 0),
            "hotwords_truncated": bool(options.get("hotwords_truncated", False)),
            "hotwords_truncation_reasons": list(options.get("hotwords_truncation_reasons") or []),
            "hotwords_enabled": bool(options.get("hotwords_enabled", False)),
            "hotwords_disabled_reason": options.get("hotwords_disabled_reason") or "",
            "hotwords_locked": True,
            "connected": True,
            "connected_at": now,
            "disconnected_at": None,
            "backend": backend.value if isinstance(backend, BackendType) else str(backend),
            "language": options.get("language"),
            "model": options.get("model"),
            "translation_enabled": bool(options.get("enable_translation", False)),
            "target_language": options.get("target_language", "auto"),
            "segment_msgs": 0,
            "segment_items": 0,
            "translation_msgs": 0,
            "translation_items": 0,
            "last_activity_at": now,
            "last_source_text": "",
            "last_translation_text": "",
        }
        with self.lock:
            self._purge_recent_statuses_locked(now)
            instance_id = status.get("client_instance_id")
            if instance_id:
                for old_websocket, old_status in list(self.client_status.items()):
                    if old_websocket is websocket:
                        continue
                    if old_status.get("client_instance_id") != instance_id:
                        continue
                    if old_status.get("connected"):
                        continue
                    del self.client_status[old_websocket]
                self.recent_client_status.pop(instance_id, None)
            self.client_status[websocket] = status

    def update_client_message(self, websocket, message_type, segments):
        if message_type == "translated_segments":
            segments = self._formal_translation_segments(segments)
            if not segments:
                return
        now = time.time()
        text = self._latest_segment_text(segments)
        with self.lock:
            status = self.client_status.get(websocket)
            if not status:
                return
            if message_type == "segments":
                status["segment_msgs"] += 1
                status["segment_items"] += len(segments or [])
                if text:
                    status["last_source_text"] = text
            elif message_type == "translated_segments":
                status["translation_msgs"] += 1
                status["translation_items"] += len(segments or [])
                if text:
                    status["last_translation_text"] = text
            status["last_activity_at"] = now

    def mark_client_disconnected(self, websocket):
        now = time.time()
        with self.lock:
            status = self.client_status.get(websocket)
            if status:
                status["connected"] = False
                status["disconnected_at"] = now
                status["last_activity_at"] = now
                snapshot = dict(status)
                self.recent_client_status[self._recent_status_key(snapshot)] = snapshot
                del self.client_status[websocket]
                self._purge_recent_statuses_locked(now)

    def get_client_status_snapshot(self):
        now = time.time()
        with self.lock:
            self._purge_recent_statuses_locked(now)
            statuses = [dict(status) for status in self.client_status.values()]
            statuses.extend(dict(status) for status in self.recent_client_status.values())
        for status in statuses:
            connected_at = status.get("connected_at") or now
            last_activity_at = status.get("last_activity_at") or connected_at
            status["connected_seconds"] = round((status.get("disconnected_at") or now) - connected_at, 3)
            status["last_activity_seconds_ago"] = round(now - last_activity_at, 3)
        statuses.sort(key=lambda item: item.get("connected_at", 0), reverse=True)
        return {"server_time": now, "clients": statuses}

    def get_client_status_entry(self, uid):
        with self.lock:
            for websocket, status in self.client_status.items():
                if status.get("uid") == uid:
                    return websocket, dict(status)
        return None, None

    def delete_disconnected_client_status(self, uid):
        with self.lock:
            self._purge_recent_statuses_locked()
            for websocket, status in list(self.client_status.items()):
                if status.get("uid") != uid:
                    continue
                if status.get("connected"):
                    return "connected"
                del self.client_status[websocket]
                return "deleted"
            for key, status in list(self.recent_client_status.items()):
                if status.get("uid") == uid:
                    del self.recent_client_status[key]
                    return "deleted"
        return "not_found"

    def add_client(self, websocket, client):
        """
        Adds a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        with self.lock:
            self.clients[websocket] = client
            self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        """
        Retrieves a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        with self.lock:
            if websocket in self.clients:
                return self.clients[websocket]
            return False

    def remove_client(self, websocket):
        """
        Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
        client if necessary.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        with self.lock:
            client = self.clients.pop(websocket, None)
            self.start_times.pop(websocket, None)
        if client:
            client.cleanup()

    def get_wait_time(self):
        """
        Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

        Returns:
            The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
        """
        with self.lock:
            wait_time = None
            for start_time in self.start_times.values():
                current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
                if wait_time is None or current_client_time_remaining < wait_time:
                    wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        """
        Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        with self.lock:
            if len(self.clients) >= self.max_clients:
                wait_time = None
                for start_time in self.start_times.values():
                    remaining = self.max_connection_time - (time.time() - start_time)
                    if wait_time is None or remaining < wait_time:
                        wait_time = remaining
                wait_time_minutes = wait_time / 60 if wait_time is not None else 0
                response = {"uid": options["uid"], "status": "WAIT", "message": wait_time_minutes}
                websocket.send(json.dumps(response))
                return True
            return False

    def is_client_timeout(self, websocket):
        """
        Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit, False otherwise.
        """
        with self.lock:
            elapsed_time = time.time() - self.start_times[websocket]
            client = self.clients.get(websocket)
        if elapsed_time >= self.max_connection_time and client:
            client.disconnect()
            logging.warning(f"Client with uid '{client.client_uid}' disconnected due to overtime.")
            return True
        return False


class BackendType(Enum):
    FASTER_WHISPER = "faster_whisper"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    MLX_WHISPER = "mlx_whisper"
    FUNASR = "funasr"

    @staticmethod
    def valid_types() -> List[str]:
        return [backend_type.value for backend_type in BackendType]

    @staticmethod
    def is_valid(backend: str) -> bool:
        return backend in BackendType.valid_types()

    def is_faster_whisper(self) -> bool:
        return self == BackendType.FASTER_WHISPER

    def is_tensorrt(self) -> bool:
        return self == BackendType.TENSORRT
    
    def is_openvino(self) -> bool:
        return self == BackendType.OPENVINO

    def is_mlx_whisper(self) -> bool:
        return self == BackendType.MLX_WHISPER

    def is_funasr(self) -> bool:
        return self == BackendType.FUNASR


class TranscriptionServer:
    RATE = 16000
    FINALIZATION_BUDGET_SECONDS = 15.0
    LOCAL_ASR_MODEL_ROOT = "model/asr"
    LOCAL_ASR_MODEL_NAMES = {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v3-turbo", "large-v3",
    }
    TRANSLATION_DRAFT_INTERVAL_DEFAULT = 1.2
    TRANSLATION_DRAFT_INTERVAL_MIN = 0.5
    TRANSLATION_DRAFT_INTERVAL_MAX = 10.0
    TRANSLATION_DRAFT_MIN_DELTA_DEFAULT = 8
    TRANSLATION_DRAFT_SOURCE_CHARS_DEFAULT = 220
    TRANSLATION_DRAFT_SOURCE_CHARS_MIN = 32
    TRANSLATION_DRAFT_SOURCE_CHARS_MAX = 220
    TRANSLATION_READABILITY_SENTENCES_DEFAULT = 2
    TRANSLATION_READABILITY_SENTENCES_MAX = 2
    TRANSLATION_READABILITY_CHARS_DEFAULT = 220
    TRANSLATION_READABILITY_CHARS_MIN = 32
    TRANSLATION_READABILITY_CHARS_MAX = 220

    def __init__(self):
        self.client_manager = None
        self.no_voice_activity_chunks = 0
        self.use_vad = True
        self.single_model = False
        self.batch_config = None
        self.raw_pcm_input = False
        self.segment_post_processor = None
        self.default_hotwords = None
        self.translation_device = "cpu"
        self.asr_device_index = 0
        self.meeting_hotwords = MeetingHotwordStore()
        self.meeting_logs = MeetingLogStore()
        self.summary_templates = SummaryTemplateStore()
        self.meeting_summary = MeetingSummaryService()
        self.warmup_lock = threading.Lock()
        self.warmup_status = {
            "state": "idle",
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "asr_status": "idle",
            "translation_status": "idle",
            "error": "",
        }

    HOTWORD_UPLOAD_MAX_BYTES = 2 * 1024 * 1024

    @staticmethod
    def _decode_text_upload(content):
        for encoding in ("utf-8-sig", "utf-8"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _extract_docx_text(content):
        Document = MeetingDocConverter._document_class()
        document = Document(io.BytesIO(content))
        lines = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                lines.append(text)
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    lines.append("\t".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _normalize_uploaded_hotword_line(raw_line):
        line = str(raw_line or "").strip()
        if not line or line.startswith("#") or line.startswith("```"):
            return ""
        line = re.sub(r"^[-*+]\s+\[[ xX]\]\s+", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)、．）]\s+", "", line)
        return line.strip()

    @classmethod
    def _normalize_uploaded_hotword_text(cls, text):
        return "\n".join(
            line for line in (cls._normalize_uploaded_hotword_line(raw_line) for raw_line in str(text or "").splitlines())
            if line
        )

    @classmethod
    def parse_hotword_upload(cls, filename, content):
        filename = os.path.basename(str(filename or "hotwords.txt"))
        extension = os.path.splitext(filename)[1].lower()
        if extension not in {".txt", ".md", ".docx"}:
            raise ValueError("only .txt, .md and .docx hotword files are supported")
        if len(content or b"") > cls.HOTWORD_UPLOAD_MAX_BYTES:
            raise ValueError("hotword file is too large")
        if extension == ".docx":
            text = cls._extract_docx_text(content or b"")
        else:
            text = cls._decode_text_upload(content or b"")
        normalized_upload_text = cls._normalize_uploaded_hotword_text(text)
        parsed = parse_hotword_config(normalized_upload_text)
        return {
            "filename": filename,
            "text": text,
            "normalized_text": parsed["text"],
            "count": parsed["count"],
            "translation_count": parsed["translation_count"],
        }

    @staticmethod
    def extract_translation_terms(value):
        terms = []
        if isinstance(value, dict):
            iterable = list(value.keys()) + list(value.values())
        elif isinstance(value, str):
            if "\n" in value or "\r" in value:
                try:
                    parsed = parse_hotword_config(value)
                    iterable = parsed.get("hotwords") or []
                except Exception:
                    iterable = re.split(r"[\s,;；]+", value)
            else:
                iterable = re.split(r"[\s,;；]+", value)
        else:
            iterable = value or []
        for item in iterable:
            term = str(item or "").strip()
            if term and "=>" not in term and re.search(r"[A-Za-z0-9]", term):
                terms.append(term)
        return list(dict.fromkeys(terms))

    @staticmethod
    def hotwords_preview(value, limit=10):
        if not value:
            return []
        if isinstance(value, dict):
            terms = value.get("terms") or value.get("hotwords") or []
        elif isinstance(value, (list, tuple, set)):
            terms = value
        else:
            terms = None
        if terms is not None:
            preview = []
            for term in terms:
                term = str(term or "").strip()
                if term and term not in preview:
                    preview.append(term)
                if limit is not None and len(preview) >= limit:
                    break
            return preview
        text = str(value or "")
        if "\n" in text or "\r" in text:
            try:
                terms = parse_hotword_config(text).get("hotwords") or []
            except Exception:
                terms = []
        else:
            terms = re.split(r"[\s,;；]+", text)

        preview = []
        for term in terms:
            term = str(term or "").strip()
            if term and term not in preview:
                preview.append(term)
            if limit is not None and len(preview) >= limit:
                break
        return preview

    @staticmethod
    def load_hotwords_file(path):
        if not path:
            return None
        if not os.path.isfile(path):
            logging.warning(f"Hotwords file not found: {path}")
            return None

        hotwords = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip()
                if not word or word.startswith("#"):
                    continue
                hotwords.append(word)

        if not hotwords:
            logging.info(f"Hotwords file is empty: {path}")
            return None

        return "\n".join(hotwords)

    @staticmethod
    def canonical_hotwords_from_options(options):
        terms = options.get("hotword_terms")
        if terms:
            return normalize_asr_hotwords(terms=terms)
        return normalize_asr_hotwords(options.get("hotwords"))

    @staticmethod
    def apply_canonical_hotwords(options, canonical, source="none", filename=None, locked=None):
        canonical = canonical or normalize_asr_hotwords(terms=[])
        prompt = canonical.get("prompt")
        if prompt:
            options["hotwords"] = prompt
        else:
            options.pop("hotwords", None)
        options["hotword_terms"] = list(canonical.get("terms") or [])
        options["hotwords_source"] = source
        options["hotwords_count"] = int(canonical.get("accepted_count") or 0)
        options["hotwords_original_count"] = int(canonical.get("original_count") or 0)
        options["hotwords_rejected_count"] = int(canonical.get("rejected_count") or 0)
        options["hotwords_truncated"] = bool(canonical.get("truncated", False))
        options["hotwords_truncation_reasons"] = list(canonical.get("truncation_reasons") or [])
        options["hotwords_enabled"] = bool(prompt)
        options["hotwords_disabled_reason"] = ""
        if filename is not None:
            options["hotwords_file"] = filename or ""
        if locked is not None:
            options["hotwords_locked"] = bool(locked)

    @staticmethod
    def asr_hotwords_enabled(options):
        return options.get("service_mode") == "accurate"

    @staticmethod
    def resolve_max_pending_audio_seconds(options):
        value = options.get("max_pending_audio_seconds")
        if value is None:
            value = 18.0 if options.get("service_mode") == "accurate" else ServeClientBase.MAX_PENDING_AUDIO_SECONDS
        return min(
            ServeClientBase.MAX_CONFIGURABLE_PENDING_AUDIO_SECONDS,
            max(1.0, float(value)),
        )

    def apply_standard_segmentation_defaults(self, options):
        if (
            not self.backend.is_faster_whisper()
            or options.get("service_mode") == "accurate"
        ):
            return
        options.update({
            "same_output_threshold": 9,
            "max_incomplete_segment_seconds": 12.0,
            "sentence_completion_min_seconds": 3.0,
            "conservative_segmentation": True,
            "short_fragment_hold_seconds": 2.5,
            "min_new_audio_seconds": 0.25,
        })

    @staticmethod
    def _config_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 1
        return str(value or "").strip().lower() == "true"

    @staticmethod
    def _bounded_config_number(value, default, minimum, maximum, integer=False):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if not math.isfinite(number):
            number = float(default)
        number = min(float(maximum), max(float(minimum), number))
        return int(number) if integer else number

    @staticmethod
    def _normalized_language(value):
        value = str(value or "").strip().lower().replace("_", "-")
        if value == "zh" or value.startswith("zh-"):
            return "zh"
        if value == "en" or value.startswith("en-"):
            return "en"
        return value or None

    @classmethod
    def normalize_translation_draft_options(cls, options, backend=None):
        source_language = cls._normalized_language(options.get("language"))
        target_language = cls._normalized_language(options.get("target_language")) or "auto"
        if target_language == "auto" and source_language == "en":
            resolved_target_language = "zh"
        elif target_language == "auto" and source_language == "zh":
            resolved_target_language = "en"
        else:
            resolved_target_language = target_language
        translation_mode = options.get("translation_mode", "standard")
        backend_name = getattr(backend, "value", backend)
        backend_supported = backend is None or backend_name == BackendType.FASTER_WHISPER.value
        draft_eligible = (
            backend_supported
            and cls._config_bool(options.get("enable_translation"))
            and options.get("service_mode") == "accurate"
            and (
                (translation_mode == "standard" and source_language == "en" and resolved_target_language == "zh")
                or (translation_mode == "mixed_interpretation" and target_language in ("auto", "zh"))
            )
            and cls._config_bool(options.get("translation_draft_enabled"))
        )
        context_requested = (
            cls._config_bool(options.get("translation_readability_context_enabled"))
            or draft_eligible
        )
        context_eligible = (
            backend_supported
            and cls._config_bool(options.get("enable_translation"))
            and options.get("service_mode") == "accurate"
            and context_requested
            and (
                (
                    translation_mode == "standard"
                    and (source_language, resolved_target_language) in (("en", "zh"), ("zh", "en"))
                )
                or (translation_mode == "mixed_interpretation" and target_language in ("auto", "zh", "en"))
            )
        )

        options["translation_draft_enabled"] = draft_eligible
        options["translation_readability_context_enabled"] = context_eligible
        options["translation_draft_interval_seconds"] = cls._bounded_config_number(
            options.get("translation_draft_interval_seconds"),
            cls.TRANSLATION_DRAFT_INTERVAL_DEFAULT,
            cls.TRANSLATION_DRAFT_INTERVAL_MIN,
            cls.TRANSLATION_DRAFT_INTERVAL_MAX,
        )
        options["translation_draft_min_delta_chars"] = cls._bounded_config_number(
            options.get("translation_draft_min_delta_chars"),
            cls.TRANSLATION_DRAFT_MIN_DELTA_DEFAULT,
            1,
            cls.TRANSLATION_DRAFT_SOURCE_CHARS_MAX,
            integer=True,
        )
        options["translation_draft_max_source_chars"] = cls._bounded_config_number(
            options.get("translation_draft_max_source_chars"),
            cls.TRANSLATION_DRAFT_SOURCE_CHARS_DEFAULT,
            cls.TRANSLATION_DRAFT_SOURCE_CHARS_MIN,
            cls.TRANSLATION_DRAFT_SOURCE_CHARS_MAX,
            integer=True,
        )

        if context_eligible:
            context_sentences = cls._bounded_config_number(
                options.get("translation_readability_context_sentences"),
                cls.TRANSLATION_READABILITY_SENTENCES_DEFAULT,
                0,
                cls.TRANSLATION_READABILITY_SENTENCES_MAX,
                integer=True,
            )
            context_chars = cls._bounded_config_number(
                options.get("translation_readability_context_max_chars"),
                cls.TRANSLATION_READABILITY_CHARS_DEFAULT,
                cls.TRANSLATION_READABILITY_CHARS_MIN,
                cls.TRANSLATION_READABILITY_CHARS_MAX,
                integer=True,
            )
            if context_sentences == 0:
                context_chars = 0
        else:
            context_sentences = 0
            context_chars = 0
        options["translation_readability_context_sentences"] = context_sentences
        options["translation_readability_context_max_chars"] = context_chars

        logging.info(
            "[TRANSLATION_DRAFT_CONFIG] uid=%s eligible=%s context_eligible=%s interval=%.2f min_delta=%d "
            "max_source_chars=%d context_sentences=%d context_max_chars=%d",
            options.get("uid"),
            str(draft_eligible).lower(),
            str(context_eligible).lower(),
            options["translation_draft_interval_seconds"],
            options["translation_draft_min_delta_chars"],
            options["translation_draft_max_source_chars"],
            context_sentences,
            context_chars,
        )
        return options

    @classmethod
    def disable_canonical_hotwords(cls, options, canonical, source="none", filename=None, locked=None, reason="service_mode"):
        canonical = canonical or normalize_asr_hotwords(terms=[])
        options.pop("hotwords", None)
        options["hotword_terms"] = []
        options["hotwords_source"] = source
        options["hotwords_count"] = 0
        options["hotwords_original_count"] = int(canonical.get("original_count") or 0)
        options["hotwords_rejected_count"] = int(canonical.get("rejected_count") or 0)
        options["hotwords_truncated"] = bool(canonical.get("truncated", False))
        options["hotwords_truncation_reasons"] = list(canonical.get("truncation_reasons") or [])
        options["hotwords_enabled"] = False
        options["hotwords_disabled_reason"] = reason if canonical.get("original_count") else ""
        if filename is not None:
            options["hotwords_file"] = filename or ""
        if locked is not None:
            options["hotwords_locked"] = bool(locked)

    def apply_asr_hotword_policy(self, options, canonical, source="none", filename=None, locked=None):
        if self.asr_hotwords_enabled(options):
            self.apply_canonical_hotwords(options, canonical, source=source, filename=filename, locked=locked)
            return
        self.disable_canonical_hotwords(options, canonical, source=source, filename=filename, locked=locked)

    @classmethod
    def disable_hotword_features(cls, options, canonical=None, source="none"):
        cls.disable_canonical_hotwords(
            options,
            canonical,
            source=source,
            filename="",
            locked=True,
            reason="service_mode",
        )
        options["hotwords_disabled_reason"] = "service_mode"
        options["translation_glossary"] = {}
        options["translation_glossary_count"] = 0
        options["translation_terms"] = []

    def apply_meeting_hotwords(self, options):
        has_client_hotwords = bool(options.get("hotwords") or options.get("hotword_terms"))
        has_client_glossary = bool(options.get("translation_glossary"))
        if not self.asr_hotwords_enabled(options):
            canonical = self.canonical_hotwords_from_options(options) if has_client_hotwords else None
            source = "client" if has_client_hotwords or has_client_glossary else "none"
            self.disable_hotword_features(options, canonical, source=source)
            return
        if options.get("hotwords_locked") and (has_client_hotwords or has_client_glossary):
            canonical = self.canonical_hotwords_from_options(options) if has_client_hotwords else normalize_asr_hotwords(terms=[])
            self.apply_asr_hotword_policy(options, canonical, source="client", locked=True)
            options["translation_terms"] = self.extract_translation_terms(options.get("translation_glossary") or canonical.get("terms"))
            return
        meeting_name = options.get("meeting_name")
        if not meeting_name or not self.meeting_hotwords:
            return
        stored = self.meeting_hotwords.get(meeting_name)
        canonical = normalize_asr_hotwords(stored.get("text"))
        if has_client_hotwords:
            self.apply_asr_hotword_policy(options, self.canonical_hotwords_from_options(options), source="client", locked=True)
        elif canonical.get("prompt"):
            self.apply_asr_hotword_policy(
                options,
                canonical,
                source="meeting",
                filename=stored.get("filename") or "",
                locked=True,
            )
        if canonical.get("prompt") or stored.get("translation_glossary"):
            options["hotwords_file"] = stored.get("filename") or ""
            options["hotwords_locked"] = True
            options["translation_glossary"] = dict(stored.get("translation_glossary") or {})
            options["translation_glossary_count"] = int(stored.get("translation_count") or 0)
            options["translation_terms"] = self.extract_translation_terms(stored.get("text"))
            if not canonical.get("prompt") and not has_client_hotwords:
                self.apply_asr_hotword_policy(
                    options,
                    canonical,
                    source="meeting",
                    filename=stored.get("filename") or "",
                    locked=True,
                )

    def asr_corrections_enabled(self, options):
        if not self.backend.is_faster_whisper():
            return False
        if not self._config_bool(options.get("enable_translation")):
            return False
        source_language = self._normalized_language(options.get("language"))
        target_language = self._normalized_language(options.get("target_language")) or "auto"
        translation_mode = options.get("translation_mode", "standard")
        if translation_mode == "mixed_interpretation":
            return target_language in ("auto", "en")
        return source_language == "zh" and target_language in ("auto", "en")

    def apply_meeting_asr_corrections(self, options):
        options["asr_corrections_enabled"] = False
        options["asr_corrections_count"] = 0
        options["asr_corrections_file"] = ""
        options["asr_correction_rules"] = []
        if not self.asr_corrections_enabled(options):
            return
        if not self.meeting_asr_corrections:
            return
        rules = []
        filenames = []
        if self.asr_corrections_file:
            try:
                record = self.meeting_asr_corrections.get_file(self.asr_corrections_file)
                rules.extend(record.get("rules") or [])
                if record.get("filename"):
                    filenames.append(record.get("filename"))
            except Exception as exc:
                logging.warning("Failed to load global ASR corrections from %r: %s", self.asr_corrections_file, exc)
        meeting_name = options.get("meeting_name")
        if meeting_name:
            try:
                record = self.meeting_asr_corrections.get(meeting_name)
                rules.extend(record.get("rules") or [])
                if record.get("filename"):
                    filenames.append(record.get("filename"))
            except Exception as exc:
                logging.warning("Failed to load ASR corrections for meeting %r: %s", meeting_name, exc)
                return
        rules = sorted(dict(rules).items(), key=lambda item: len(item[0]), reverse=True)
        if not rules:
            return
        options["asr_corrections_enabled"] = True
        options["asr_corrections_count"] = len(rules)
        options["asr_corrections_file"] = ",".join(filenames)
        options["asr_correction_rules"] = rules

    def build_asr_text_corrector(self, options):
        rules = options.get("asr_correction_rules") or []
        if not rules or not options.get("asr_corrections_enabled"):
            return None
        corrector = AsrTextCorrector(rules)
        uid = options.get("uid")
        filename = options.get("asr_corrections_file") or ""

        def correct_text(text, reason="completed"):
            if not re.search(r"[\u4e00-\u9fff]", str(text or "")):
                return text
            corrected, replacements = corrector.correct(text)
            if replacements and corrected != text:
                logging.info(
                    "[ASR_TEXT_CORRECTED] uid=%s reason=%s replacements=%d file=%s",
                    uid,
                    reason,
                    replacements,
                    filename,
                )
            return corrected

        return correct_text

    def apply_default_hotwords(self, options):
        if options.get("hotwords") or options.get("hotwords_locked"):
            return
        if self.default_hotwords and self.asr_hotwords_enabled(options):
            self.apply_asr_hotword_policy(
                options,
                normalize_asr_hotwords(self.default_hotwords),
                source="default",
                locked=False,
            )

    def get_admin_clients_payload(self):
        backend = self.backend.value if isinstance(getattr(self, "backend", None), BackendType) else str(getattr(self, "backend", "") or "")
        if not self.client_manager:
            return {"server_time": time.time(), "server_backend": backend, "clients": []}
        payload = self.client_manager.get_client_status_snapshot()
        payload["server_backend"] = backend
        return payload

    def get_admin_warmup_status_payload(self):
        with self.warmup_lock:
            status = dict(self.warmup_status)
        status["server_time"] = time.time()
        status["server_backend"] = (
            self.backend.value if isinstance(getattr(self, "backend", None), BackendType)
            else str(getattr(self, "backend", "") or "")
        )
        return status

    @staticmethod
    def _model_dir_has_files(path, required=(), required_any=(), required_any_patterns=()):
        if not path or not os.path.isdir(path):
            return False
        for filename in required:
            if not os.path.isfile(os.path.join(path, filename)):
                return False
        if required_any or required_any_patterns:
            filenames = [
                filename for filename in os.listdir(path)
                if os.path.isfile(os.path.join(path, filename))
            ]
            exact_match = any(filename in filenames for filename in required_any)
            pattern_match = any(
                re.fullmatch(pattern, filename)
                for pattern in required_any_patterns
                for filename in filenames
            )
            if not exact_match and not pattern_match:
                return False
        return True

    @staticmethod
    def _translation_model_label(name):
        normalized = str(name or "").lower()
        if "3.3" in normalized or "3_3" in normalized or "3-3" in normalized:
            return "NLLB-200 3.3B 高质量"
        if "1.3" in normalized or "1_3" in normalized or "1-3" in normalized:
            return "NLLB-200 1.3B 高质量"
        if "600" in normalized:
            return "NLLB-200 600M 高质量"
        return str(name or "NLLB 翻译模型")

    @staticmethod
    def _translation_model_value(name, path):
        normalized = str(name or "").lower()
        if "3.3" in normalized or "3_3" in normalized or "3-3" in normalized:
            return "nllb_200_3_3b"
        if "1.3" in normalized or "1_3" in normalized or "1-3" in normalized:
            return "nllb_200_distilled_1_3b"
        if "600" in normalized:
            return "nllb_200_600m"
        return f"nllb:{path}"

    @staticmethod
    def _translation_model_sort_key(model):
        value = str(model.get("value") or "").lower()
        label = str(model.get("label") or "").lower()
        path = str(model.get("nllb_model_path") or "").lower()
        text = " ".join((value, label, path))
        if value == "helsinki_zh_en":
            return (0, label)
        if "600" in text:
            return (1, label)
        if "1.3" in text or "1_3" in text or "1-3" in text:
            return (2, label)
        if "3.3" in text or "3_3" in text or "3-3" in text:
            return (3, label)
        return (9, label or value or path)

    def get_translation_models_payload(self):
        models = []
        zh_en_path = "model/opus-mt-zh-en"
        en_zh_path = "model/opus-mt-en-zh"
        helsinki_required = ("config.json", "source.spm", "target.spm", "vocab.json")
        if (
            self._model_dir_has_files(zh_en_path, required=helsinki_required)
            and self._model_dir_has_files(en_zh_path, required=helsinki_required)
        ):
            models.append({
                "value": "helsinki_zh_en",
                "label": "Helsinki 轻量实时",
                "provider": "helsinki_zh_en",
                "zh_en_model_path": zh_en_path,
                "en_zh_model_path": en_zh_path,
                "available": True,
            })

        model_root = "model"
        if os.path.isdir(model_root):
            for name in sorted(os.listdir(model_root)):
                path = os.path.join(model_root, name)
                if not os.path.isdir(path):
                    continue
                normalized = name.lower()
                if "nllb" not in normalized:
                    continue
                if not self._model_dir_has_files(
                    path,
                    required=("config.json", "tokenizer_config.json"),
                    required_any=("pytorch_model.bin", "model.safetensors"),
                    required_any_patterns=(
                        r"pytorch_model-\d{5}-of-\d{5}\.bin",
                        r"model-\d{5}-of-\d{5}\.safetensors",
                    ),
                ):
                    continue
                models.append({
                    "value": self._translation_model_value(name, path),
                    "label": self._translation_model_label(name),
                    "provider": "nllb",
                    "nllb_model_path": path,
                    "available": True,
                })

        seen = set()
        deduped = []
        for item in models:
            key = item.get("value")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        deduped.sort(key=self._translation_model_sort_key)
        return {"models": deduped}

    def has_active_clients(self):
        if not self.client_manager:
            return False
        snapshot = self.client_manager.get_client_status_snapshot()
        return any(client.get("connected") for client in snapshot.get("clients", []))

    def start_admin_warmup(self, config, force=False):
        with self.warmup_lock:
            if self.warmup_status.get("state") == "running":
                return JSONResponse(status_code=409, content={
                    "started": False,
                    "error": "warmup already running",
                    "status": dict(self.warmup_status),
                })

            if self.has_active_clients() and not force:
                return JSONResponse(status_code=409, content={
                    "started": False,
                    "error": "active clients exist; retry with force=true",
                    "status": dict(self.warmup_status),
                })

            now = time.time()
            self.warmup_status = {
                "state": "running",
                "started_at": now,
                "finished_at": None,
                "duration_seconds": None,
                "asr_status": "pending",
                "translation_status": "pending",
                "error": "",
            }

        threading.Thread(target=self.run_admin_warmup, args=(config,), daemon=True).start()
        return {"started": True, "status": self.get_admin_warmup_status_payload()}

    def run_admin_warmup(self, config):
        started_at = time.time()
        asr_status = "skipped"
        translation_status = "skipped"
        error = ""
        state = "success"
        try:
            asr_status = self.warmup_asr(config)
            translation_status = self.warmup_translation(config)
            failed_parts = [
                status for status in (asr_status, translation_status)
                if str(status).startswith("failed")
            ]
            if failed_parts:
                state = "failed"
                error = "; ".join(failed_parts)
        except Exception as exc:
            state = "failed"
            error = str(exc)
            logging.exception("[ADMIN_WARMUP_FAILED] %s", exc)
        finally:
            finished_at = time.time()
            with self.warmup_lock:
                self.warmup_status.update({
                    "state": state,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_seconds": round(finished_at - started_at, 3),
                    "asr_status": asr_status,
                    "translation_status": translation_status,
                    "error": error,
                })

    def warmup_asr(self, config):
        backend = config["backend"]
        if not self.single_model:
            logging.info("[ADMIN_WARMUP] ASR skipped because single_model is disabled")
            return "skipped:not_single_model"

        if backend.is_faster_whisper():
            return self.warmup_faster_whisper(config)
        if backend.is_funasr():
            return self.warmup_funasr(config)
        logging.info("[ADMIN_WARMUP] ASR warmup skipped for backend=%s", backend.value)
        return f"skipped:{backend.value}"

    @staticmethod
    def warmup_audio(duration_seconds=1.0):
        sample_count = max(1, int(ServeClientBase.RATE * duration_seconds))
        timeline = np.arange(sample_count, dtype=np.float32) / ServeClientBase.RATE
        return (0.02 * np.sin(2 * np.pi * 440.0 * timeline)).astype(np.float32)

    @staticmethod
    def _warmup_compute_type(device, device_index=0):
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(int(device_index or 0))
            return "float16" if major >= 7 else "float32"
        return "int8"

    def warmup_faster_whisper(self, config):
        from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper

        model_path = config.get("faster_whisper_custom_model_path") or self.resolve_asr_model_path("small")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        asr_device_index = int(config.get("asr_device_index") or 0)
        with ServeClientFasterWhisper.SINGLE_MODEL_INIT_LOCK:
            if ServeClientFasterWhisper.SINGLE_MODEL is None:
                warm_client = object.__new__(ServeClientFasterWhisper)
                warm_client.model_sizes = [
                    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
                    "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
                    "distil-medium.en", "distil-large-v2", "distil-large-v3",
                    "large-v3-turbo", "turbo"
                ]
                warm_client.model_size_or_path = model_path
                warm_client.cache_path = self.cache_path
                warm_client.asr_device_index = asr_device_index
                warm_client.compute_type = self._warmup_compute_type(device, asr_device_index)
                logging.info(
                    "[ADMIN_WARMUP] Loading shared faster-whisper model: %s on device=%s index=%s",
                    model_path,
                    device,
                    asr_device_index if device == "cuda" else 0,
                )
                warm_client.create_model(device)
                ServeClientFasterWhisper.SINGLE_MODEL = warm_client.transcriber
            transcriber = ServeClientFasterWhisper.SINGLE_MODEL

        audio = self.warmup_audio()
        segments, _info = transcriber.transcribe(
            audio,
            language="en",
            task="transcribe",
            vad_filter=False,
            initial_prompt=None,
            hotwords=None,
            word_timestamps=False,
        )
        list(segments or [])

        if (
            self.batch_config is not None
            and ServeClientFasterWhisper.BATCH_WORKER is None
            and ServeClientFasterWhisper.SINGLE_MODEL is not None
        ):
            from whisper_live.batch_inference import BatchInferenceWorker
            worker = BatchInferenceWorker(
                transcriber=ServeClientFasterWhisper.SINGLE_MODEL,
                **self.batch_config,
            )
            worker.start()
            ServeClientFasterWhisper.BATCH_WORKER = worker
            logging.info("[ADMIN_WARMUP] Batch inference worker started")

        logging.info("[ADMIN_WARMUP] faster-whisper warmup completed")
        return "success"

    def warmup_funasr(self, config):
        from whisper_live.backend.funasr_backend import ServeClientFunASR

        class WarmupWebSocket:
            def send(self, _message):
                return None

            def close(self):
                return None

        model = config.get("funasr_model")
        mode = config.get("funasr_mode") or "sensevoice"
        if mode == "paraformer_streaming":
            model = model or "model/funasr/paraformer-zh-streaming"
        else:
            model = self.resolve_funasr_model_path(model, model)

        client = ServeClientFunASR(
            WarmupWebSocket(),
            language="zh",
            task="transcribe",
            client_uid="admin-warmup",
            model=model,
            device=config.get("funasr_device") or "auto",
            mode=mode,
            punc_model=None,
            vad_model=config.get("funasr_vad_model"),
            final_model=config.get("funasr_final_model") or "model/funasr/SenseVoiceSmall",
            final_device=config.get("funasr_final_device"),
            final_refine=bool(config.get("funasr_final_refine", True)),
            single_model=True,
            use_vad=False,
        )
        try:
            audio = self.warmup_audio()
            if mode == "paraformer_streaming":
                client.transcribe_streaming_audio(audio, is_final=False)
            else:
                client.transcribe_audio(audio)
        finally:
            client.cleanup()
            if getattr(client, "trans_thread", None):
                client.trans_thread.join(timeout=1.0)
        logging.info("[ADMIN_WARMUP] FunASR warmup completed")
        return "success"

    def warmup_translation(self, config):
        from whisper_live.backend.translation_backend import ServeClientTranslation

        class WarmupWebSocket:
            def send(self, _message):
                return None

        translation_client = ServeClientTranslation(
            client_uid="admin-warmup",
            websocket=WarmupWebSocket(),
            translation_queue=queue.Queue(),
            target_language="auto",
            model_name=config.get("translation_provider") or "helsinki_zh_en",
            zh_en_model_path=config.get("zh_en_model_path") or "model/opus-mt-zh-en",
            en_zh_model_path=config.get("en_zh_model_path") or "model/opus-mt-en-zh",
            nllb_model_path=config.get("nllb_model_path") or "model/NLLB-200-600M",
            translation_device=config.get("translation_device") or self.translation_device,
        )
        if not translation_client.model_loaded:
            return "failed:not_loaded"
        translation_client.translate_text("hello", "en")
        translation_client.translate_text("你好", "zh")
        translation_client.cleanup()
        logging.info("[ADMIN_WARMUP] Translation warmup completed")
        return "success"

    def session_timeline_offset(self, websocket):
        client = self.client_manager.get_client(websocket) if self.client_manager else None
        return float(getattr(client, "meeting_log_timeline_offset_seconds", 0.0) or 0.0) if client else 0.0

    def offset_client_segment(self, websocket, segment):
        return apply_timeline_offset_to_segments([segment], self.session_timeline_offset(websocket))[0]

    def offset_client_segments(self, websocket, segments):
        return apply_timeline_offset_to_segments(segments, self.session_timeline_offset(websocket))

    @staticmethod
    def infer_segment_text_language(text):
        text = str(text or "")
        cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))
        if latin_count >= 4 and cjk_count == 0:
            return "en"
        if cjk_count > 0:
            return "zh"
        if latin_count >= 4:
            return "en"
        return None

    def process_client_segment(self, websocket, base_processor, translation_mode, segment):
        if base_processor is not None:
            segment = base_processor(segment) or segment
        if translation_mode == "mixed_interpretation":
            segment_language = str(segment.get("language") or "").strip().lower()
            if segment_language not in ("zh", "en"):
                inferred_language = self.infer_segment_text_language(segment.get("text"))
            else:
                inferred_language = None
            if inferred_language:
                segment = segment.copy()
                segment["language"] = inferred_language
        return self.offset_client_segment(websocket, segment)

    def handle_client_segments(self, websocket, message_type, segments):
        if message_type == "translated_segments":
            segments = ClientManager._formal_translation_segments(segments)
            if not segments:
                return
        if self.client_manager:
            self.client_manager.update_client_message(websocket, message_type, segments)
        client = self.client_manager.get_client(websocket) if self.client_manager else None
        session_id = getattr(client, "meeting_log_session_id", None)
        kind = "translation" if message_type == "translated_segments" else "source"
        try:
            self.meeting_logs.append_segments(session_id, kind, segments)
        except Exception as exc:
            logging.error("Failed to append meeting log segments: %s", exc)

    def finalize_client_meeting_log(self, websocket, interrupted=False):
        client = self.client_manager.get_client(websocket) if self.client_manager else None
        session_id = getattr(client, "meeting_log_session_id", None)
        connection_generation = getattr(client, "meeting_log_connection_generation", None)
        try:
            if interrupted:
                return self.meeting_logs.interrupt_session(
                    session_id, expected_generation=connection_generation,
                )
            return self.meeting_logs.finish_session(session_id)
        except Exception as exc:
            logging.error("Failed to finalize meeting log: %s", exc)
            return None

    def _cleanup_unregistered_connection(self, websocket, client=None, translation_client=None,
                                         translation_thread=None, reason="initialization_failure"):
        """Release resources that were allocated before manager ownership is established."""
        logging.warning("[RUNTIME_CLEANUP] reason=%s uid=%s", reason, getattr(client, "client_uid", ""))
        if translation_client:
            translation_client.cleanup()
        if translation_thread and translation_thread is not threading.current_thread():
            translation_thread.join(timeout=2.0)
        if client:
            joined = client.stop_and_join(timeout=2.0)
            if not joined:
                logging.warning("[RUNTIME_ASR_JOIN_TIMEOUT] uid=%s", getattr(client, "client_uid", ""))
        try:
            websocket.close()
        except Exception as exc:
            logging.debug("Failed to close runtime connection: %s", exc)

    def schedule_cleanup(self, websocket, reason):
        """Run cleanup once outside a worker thread after a delivery failure."""
        client = self.client_manager.get_client(websocket) if self.client_manager else None
        if not client or getattr(client, "runtime_cleanup_scheduled", False):
            return
        client.runtime_cleanup_scheduled = True
        logging.warning("[RUNTIME_DELIVERY_FAILURE] reason=%s uid=%s", reason, getattr(client, "client_uid", ""))
        threading.Thread(target=self.cleanup, args=(websocket,), daemon=True).start()

    def generate_meeting_summary(self, session_id, template="auto", custom_template_id=None):
        if hasattr(self.meeting_logs, "refresh_sessions"):
            self.meeting_logs.refresh_sessions(force=True)
        info = self.meeting_logs.session_info(session_id)
        if not info:
            raise KeyError("meeting log session not found")
        if info.get("status") != "finished":
            raise RuntimeError("请停止会议后再生成总结")
        payload = self.meeting_logs.get_session_payload(session_id)
        if template == "custom":
            definition = self.summary_templates.get(custom_template_id)
            if not definition:
                raise KeyError("custom summary template not found")
            summary = self.meeting_summary.generate_custom(payload, definition)
        else:
            template = self.meeting_summary.validate_template(template)
            summary = self.meeting_summary.generate(payload, template=template)
        result = self.meeting_logs.write_summary(session_id, summary)
        return {"generated": True, "summary": result, "data": summary}

    def _default_cors_origins(self, websocket_port):
        return [
            f"http://localhost:{websocket_port}",
            f"http://127.0.0.1:{websocket_port}",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

    def resolve_asr_model_path(self, model):
        if model in self.LOCAL_ASR_MODEL_NAMES:
            local_model = os.path.join(self.LOCAL_ASR_MODEL_ROOT, model)
            if os.path.isdir(local_model):
                return local_model
        return model

    def resolve_funasr_model_path(self, client_model, server_model):
        server_model = server_model or "iic/SenseVoiceSmall"
        client_model = str(client_model or "").strip()
        if not client_model:
            return server_model
        if client_model in self.LOCAL_ASR_MODEL_NAMES or client_model.startswith("model/asr/"):
            return server_model
        if client_model == "iic/SenseVoiceSmall" and server_model != "iic/SenseVoiceSmall":
            return server_model
        if (
            client_model.startswith("model/funasr/")
            or client_model.startswith("/")
            or client_model.startswith("~")
            or "/" in client_model
        ):
            return client_model
        return server_model

    def initialize_client(
        self, websocket, options, faster_whisper_custom_model_path,
        whisper_tensorrt_path, trt_multilingual, trt_py_session=False,
        funasr_model=None, funasr_device="auto", asr_device_index=0,
        funasr_mode="sensevoice", funasr_punc_model=None, funasr_vad_model=None,
        funasr_final_model="model/funasr/SenseVoiceSmall", funasr_final_device=None,
        funasr_final_refine=True, use_vad=None,
    ):
        client: Optional[ServeClientBase] = None
        translation_client = None
        translation_thread = None

        if options.get("translation_mode") == "mixed_interpretation" and self.backend.is_faster_whisper():
            options["language"] = None

        # Check if client wants translation
        enable_translation = options.get("enable_translation", False)
        
        # Create translation queue if translation is enabled
        translation_queue = None
        
        if enable_translation:
            target_language = options.get("target_language", "auto")
            translation_device = options.get("translation_device", self.translation_device)
            translation_queue = queue.Queue(maxsize=ServeClientBase.MAX_TRANSLATION_QUEUE_SIZE)
            from whisper_live.backend.translation_backend import ServeClientTranslation
            translation_client = ServeClientTranslation(
                client_uid=options["uid"],
                websocket=websocket,
                translation_queue=translation_queue,
                target_language=target_language,
                send_last_n_segments=options.get("send_last_n_segments", 10),
                model_name=options.get("translation_provider", "helsinki_zh_en"),
                zh_en_model_path=options.get("zh_en_model_path", "model/opus-mt-zh-en"),
                translation_device=translation_device,
                translation_max_chars=options.get("translation_max_chars", 220),
                translation_max_wait_seconds=options.get("translation_max_wait_seconds", 3.0),
                translation_incomplete_max_wait_seconds=options.get("translation_incomplete_max_wait_seconds"),
                translation_context_seconds=options.get("translation_context_seconds", 0.0),
                en_zh_model_path=options.get("en_zh_model_path", "model/opus-mt-en-zh"),
                nllb_model_path=options.get("nllb_model_path", "model/NLLB-200-600M"),
                translation_glossary=options.get("translation_glossary"),
                translation_terms=options.get("translation_terms") or self.extract_translation_terms(options.get("hotwords")),
                translation_mode=options.get("translation_mode", "standard"),
                translation_merge_enabled=options.get("translation_merge_enabled", True),
                translation_merge_max_chars=options.get("translation_merge_max_chars", 180),
                translation_merge_max_delay=options.get("translation_merge_max_delay", 1.2),
                translation_merge_gap_seconds=options.get("translation_merge_gap_seconds", 1.0),
                service_mode=options.get("service_mode", "standard"),
                source_language=options.get("language"),
                translation_draft_enabled=options.get("translation_draft_enabled", False),
                translation_readability_context_enabled=options.get("translation_readability_context_enabled", False),
                translation_draft_interval_seconds=options.get("translation_draft_interval_seconds", 1.2),
                translation_draft_min_delta_chars=options.get("translation_draft_min_delta_chars", 8),
                translation_draft_max_source_chars=options.get("translation_draft_max_source_chars", 220),
                translation_readability_context_sentences=options.get("translation_readability_context_sentences", 0),
                translation_readability_context_max_chars=options.get("translation_readability_context_max_chars", 0),
                translation_zh_en_sentence_buffer_enabled=options.get("translation_zh_en_sentence_buffer_enabled", True),
                translation_zh_en_idle_seconds=options.get("translation_zh_en_idle_seconds", 1.2),
                translation_zh_en_max_audio_seconds=options.get("translation_zh_en_max_audio_seconds", 8.0),
                translation_zh_en_max_gap_seconds=options.get("translation_zh_en_max_gap_seconds", 1.0),
            )
            
            # Start translation thread
            translation_thread = threading.Thread(
                target=translation_client.speech_to_text,
                daemon=True
            )
            translation_thread.start()
            
            logging.info(f"Translation enabled for client {options['uid']} with target language: {target_language}")

        if self.backend.is_tensorrt():
            try:
                from whisper_live.backend.trt_backend import ServeClientTensorRT
                client = ServeClientTensorRT(
                    websocket,
                    multilingual=trt_multilingual,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=whisper_tensorrt_path,
                    single_model=self.single_model,
                    use_py_session=trt_py_session,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    translation_queue=translation_queue,
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                )
                logging.info("Running TensorRT backend.")
            except Exception as e:
                logging.error(f"TensorRT-LLM not supported: {e}")
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "TensorRT-LLM not supported on Server yet. "
                               "Reverting to available backend: 'faster_whisper'"
                }))
                self._cleanup_unregistered_connection(
                    websocket, client, translation_client, translation_thread, "tensorrt_initialization_failure",
                )
                return False
        
        if self.backend.is_openvino():
            try:
                from whisper_live.backend.openvino_backend import ServeClientOpenVINO
                client = ServeClientOpenVINO(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                )
                logging.info("Running OpenVINO backend.")
            except Exception as e:
                logging.error(f"OpenVINO not supported: {e}")
                self.client_uid = options["uid"]
                self._cleanup_unregistered_connection(
                    websocket, client, translation_client, translation_thread, "openvino_initialization_failure",
                )
                return False

        if self.backend.is_mlx_whisper():
            try:
                from whisper_live.backend.mlx_whisper_backend import ServeClientMLXWhisper
                client = ServeClientMLXWhisper(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    initial_prompt=options.get("initial_prompt"),
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    translation_queue=translation_queue,
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                )
                logging.info("Running MLX Whisper backend.")
            except Exception as e:
                logging.error(f"MLX Whisper not supported: {e}")
                self._cleanup_unregistered_connection(
                    websocket, client, translation_client, translation_thread, "mlx_initialization_failure",
                )
                return False

        if self.backend.is_funasr():
            try:
                from whisper_live.backend.funasr_backend import ServeClientFunASR
                if funasr_mode == "paraformer_streaming":
                    options["model"] = funasr_model or "model/funasr/paraformer-zh-streaming"
                else:
                    options["model"] = self.resolve_funasr_model_path(options.get("model"), funasr_model)
                client = ServeClientFunASR(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    device=funasr_device,
                    mode=funasr_mode,
                    punc_model=funasr_punc_model,
                    vad_model=funasr_vad_model,
                    final_model=funasr_final_model,
                    final_device=funasr_final_device,
                    final_refine=funasr_final_refine,
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 3),
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                    max_incomplete_segment_seconds=options.get("max_incomplete_segment_seconds", 6.0),
                    use_vad=self.use_vad if use_vad is None else use_vad,
                    translation_queue=translation_queue,
                    hotwords=options.get("hotwords"),
                    diarization=self._create_diarizer(options),
                )
                logging.info("Running FunASR backend.")
            except Exception as e:
                logging.error(f"FunASR not supported: {e}")
                self._cleanup_unregistered_connection(
                    websocket, client, translation_client, translation_thread, "funasr_initialization_failure",
                )
                return False

        try:
            if self.backend.is_faster_whisper():
                from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
                # model is of the form namespace/repo_name and not a filesystem path
                if faster_whisper_custom_model_path is not None:
                    logging.info(f"Using custom model {faster_whisper_custom_model_path}")
                    options["model"] = faster_whisper_custom_model_path
                else:
                    options["model"] = self.resolve_asr_model_path(options["model"])
                max_pending_audio_seconds = self.resolve_max_pending_audio_seconds(options)
                logging.info(
                    "[ASR_BUFFER_CONFIG] uid=%s service_mode=%s max_pending=%.2f",
                    options["uid"],
                    options.get("service_mode"),
                    max_pending_audio_seconds,
                )
                client = ServeClientFasterWhisper(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    initial_prompt=options.get("initial_prompt"),
                    vad_parameters=options.get("vad_parameters"),
                    use_vad=self.use_vad if use_vad is None else use_vad,
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                    max_incomplete_segment_seconds=options.get("max_incomplete_segment_seconds", 0.0),
                    sentence_completion_min_seconds=options.get("sentence_completion_min_seconds", 0.0),
                    conservative_segmentation=options.get("conservative_segmentation", False),
                    short_fragment_hold_seconds=options.get(
                        "short_fragment_hold_seconds", ServeClientBase.SHORT_FRAGMENT_HOLD_SECONDS,
                    ),
                    min_new_audio_seconds=options.get(
                        "min_new_audio_seconds", ServeClientBase.MIN_NEW_AUDIO_SECONDS,
                    ),
                    max_pending_audio_seconds=max_pending_audio_seconds,
                    min_transcription_chunk_seconds=options.get("min_transcription_chunk_seconds", 1.0),
                    cache_path=self.cache_path,
                    translation_queue=translation_queue,
                    hotwords=options.get("hotwords"),
                    hotword_terms=options.get("hotword_terms"),
                    diarization=self._create_diarizer(options),
                    word_timestamps=options.get("word_timestamps", False),
                    mixed_interpretation=options.get("translation_mode") == "mixed_interpretation",
                    mixed_language_retry_enabled=(
                        options.get("service_mode") == "accurate"
                        and options.get("translation_mode") == "mixed_interpretation"
                    ),
                    asr_device_index=asr_device_index,
                    defer_start=True,
                )

                logging.info("Running faster_whisper backend.")

                # Start batch inference worker on first client (after model is loaded)
                if (self.batch_config is not None
                        and ServeClientFasterWhisper.BATCH_WORKER is None
                        and ServeClientFasterWhisper.SINGLE_MODEL is not None):
                    from whisper_live.batch_inference import BatchInferenceWorker
                    worker = BatchInferenceWorker(
                        transcriber=ServeClientFasterWhisper.SINGLE_MODEL,
                        **self.batch_config,
                    )
                    worker.start()
                    ServeClientFasterWhisper.BATCH_WORKER = worker
        except Exception as e:
            logging.error(e)
            self._cleanup_unregistered_connection(
                websocket, client, translation_client, translation_thread, "asr_initialization_failure",
            )
            return

        if client is None:
            raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")

        # Attach segment post-processor if configured
        if self.segment_post_processor is not None:
            client.segment_post_processor = self.segment_post_processor
        if self.backend.is_faster_whisper():
            client.completed_text_post_processor = self.build_asr_text_corrector(options)

        if translation_client:
            client.translation_client = translation_client
            client.translation_thread = translation_thread
            if self.backend.is_faster_whisper():
                client.translation_draft_callback = translation_client.observe_asr_segment
            translation_client.delivery_failure_callback = functools.partial(
                self.schedule_cleanup, websocket,
            )

        if translation_client:
            translation_client.admin_status_callback = functools.partial(
                self.handle_client_segments, websocket, "translated_segments"
            )
        client.admin_status_callback = functools.partial(
            self.handle_client_segments, websocket, "segments"
        )
        client.use_vad = self.use_vad if use_vad is None else use_vad
        client.delivery_failure_callback = functools.partial(self.schedule_cleanup, websocket)
        try:
            if options.get("resume_session"):
                log_info = self.meeting_logs.resume_session(options, backend=self.backend)
            else:
                log_info = self.meeting_logs.start_session(options, backend=self.backend)
            if not log_info:
                raise RuntimeError("meeting log session was not initialized")
        except Exception as exc:
            self._reject_meeting_log_initialization(
                websocket, options, client, translation_client, translation_thread, exc,
            )
            return False
        client.meeting_log_session_id = log_info["session_id"]
        client.meeting_log_timeline_offset_seconds = float(log_info.get("timeline_offset_seconds") or 0.0)
        client.meeting_log_connection_generation = int(log_info.get("connection_generation") or log_info.get("connection_count") or 1)
        base_segment_processor = getattr(client, "segment_post_processor", None)
        client.segment_post_processor = functools.partial(
            self.process_client_segment,
            websocket,
            base_segment_processor,
            options.get("translation_mode", "standard"),
        )
        if translation_client:
            translation_client.segment_post_processor = functools.partial(self.offset_client_segment, websocket)
        self.client_manager.add_client(websocket, client)
        self.client_manager.register_client_status(websocket, client, options, self.backend)
        if self.backend.is_faster_whisper():
            client.start(send_ready=False)
        try:
            websocket.send(json.dumps({
                "uid": getattr(client, "client_uid", options.get("uid")),
                "message": getattr(client, "SERVER_READY", "SERVER_READY"),
                "backend": self.backend.value if hasattr(self.backend, "value") else str(self.backend),
                "session_id": getattr(client, "meeting_log_session_id", None),
                "session_status": (log_info or {}).get("status"),
                "resumed": bool(options.get("resume_session")),
                "connection_count": (log_info or {}).get("connection_count") or 1,
                "timeline_offset_seconds": getattr(client, "meeting_log_timeline_offset_seconds", 0.0),
                "connection_generation": getattr(client, "meeting_log_connection_generation", 1),
            }))
        except Exception as exc:
            logging.warning("[RUNTIME_READY_DELIVERY_FAILED] uid=%s error=%s", options.get("uid"), exc)
            self.cleanup(websocket)
            return False
        return True

    @staticmethod
    def _meeting_log_error_type(exc):
        if isinstance(exc, KeyError):
            return "session_not_found"
        message = str(exc).lower()
        if "finished" in message:
            return "session_finished"
        if "client_instance_id mismatch" in message:
            return "client_instance_mismatch"
        return "session_unreadable"

    def _reject_meeting_log_initialization(
        self, websocket, options, client, translation_client, translation_thread, exc,
    ):
        resumed = bool(options.get("resume_session"))
        error_type = self._meeting_log_error_type(exc)
        logging.error("Failed to %s meeting log session: %s", "resume" if resumed else "start", exc)
        try:
            websocket.send(json.dumps({
                "uid": options.get("uid"),
                "status": "ERROR",
                "code": "SESSION_RESUME_FAILED" if resumed else "SESSION_START_FAILED",
                "error_type": error_type,
                "session_id": options.get("session_id") or options.get("uid"),
            }))
        except Exception as send_exc:
            logging.debug("Failed to send meeting session initialization error: %s", send_exc)
        if translation_client:
            translation_client.cleanup()
        if translation_thread:
            translation_thread.join(timeout=2.0)
        if client:
            client.cleanup()
        try:
            websocket.close()
        except Exception as close_exc:
            logging.debug("Failed to close rejected meeting session: %s", close_exc)

    def _create_diarizer(self, options):
        """Create a SpeakerDiarizer if the client requested diarization.

        Returns:
            SpeakerDiarizer or None
        """
        if not options.get("enable_diarization", False):
            return None
        try:
            from whisper_live.diarization import SpeakerDiarizer
            return SpeakerDiarizer(
                similarity_threshold=options.get("diarization_threshold", 0.55),
                max_speakers=options.get("max_speakers", 10),
                hf_token=options.get("hf_token"),
            )
        except ImportError:
            logging.warning("pyannote.audio not installed; diarization disabled")
            return None

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        if self.raw_pcm_input:
            audio_np = np.frombuffer(frame_data, dtype=np.int16)
            return audio_np.astype(np.float32) / 32768.0
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket, faster_whisper_custom_model_path,
                              whisper_tensorrt_path, trt_multilingual, trt_py_session=False,
                              funasr_model=None, funasr_device="auto", asr_device_index=0,
                              funasr_mode="sensevoice", funasr_punc_model=None, funasr_vad_model=None,
                              funasr_final_model="model/funasr/SenseVoiceSmall", funasr_final_device=None,
                              funasr_final_refine=True):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)
            self.apply_standard_segmentation_defaults(options)
            self.normalize_translation_draft_options(options, backend=self.backend)
            self.apply_meeting_hotwords(options)
            self.apply_default_hotwords(options)
            self.apply_meeting_asr_corrections(options)

            hotword_terms = options.get("hotword_terms") or []
            hotwords_all = self.hotwords_preview(hotword_terms or options.get("hotwords"), limit=None)
            hotwords_preview = hotwords_all[:10]
            hotwords_count = int(options.get("hotwords_count") or len(hotwords_all))
            logging.info(
                "Client hotwords: uid=%s service_mode=%s translation_mode=%s source=%s enabled=%s disabled_reason=%s count=%s original_count=%s rejected_count=%s truncated=%s reasons=%s file=%s preview=%s batch_inference=%s",
                options.get("uid"),
                options.get("service_mode") or "",
                options.get("translation_mode", "standard"),
                options.get("hotwords_source") or "none",
                bool(options.get("hotwords_enabled", False)),
                options.get("hotwords_disabled_reason") or "",
                hotwords_count,
                int(options.get("hotwords_original_count") or len(hotword_terms or [])),
                int(options.get("hotwords_rejected_count") or 0),
                bool(options.get("hotwords_truncated", False)),
                list(options.get("hotwords_truncation_reasons") or []),
                options.get("hotwords_file") or "",
                hotwords_preview,
                self.batch_config is not None,
            )
            logging.info(
                "Client ASR corrections: uid=%s enabled=%s count=%d file=%s",
                options.get("uid"),
                bool(options.get("asr_corrections_enabled", False)),
                int(options.get("asr_corrections_count") or 0),
                options.get("asr_corrections_file") or "",
            )

            use_vad = options.get('use_vad', self.use_vad)
            if self.client_manager.is_server_full(websocket, options):
                wl_metrics.track_connection_rejected(reason="full")
                websocket.close()
                return False  # Indicates that the connection should not continue

            if self.backend.is_tensorrt() and use_vad:
                self.vad_detector = VoiceActivityDetector(frame_rate=self.RATE)
            initialized = self.initialize_client(websocket, options, faster_whisper_custom_model_path,
                                                 whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session,
                                                 funasr_model=funasr_model, funasr_device=funasr_device,
                                                 asr_device_index=asr_device_index,
                                                 funasr_mode=funasr_mode, funasr_punc_model=funasr_punc_model,
                                                 funasr_vad_model=funasr_vad_model,
                                                  funasr_final_model=funasr_final_model,
                                                  funasr_final_device=funasr_final_device,
                                                  funasr_final_refine=funasr_final_refine,
                                                  use_vad=use_vad)
            if not initialized:
                return False
            wl_metrics.track_connection_opened()
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            setattr(websocket, "whisperlive_end_of_audio", True)
            if self.backend.is_faster_whisper() and hasattr(client, "request_asr_finalization"):
                client.request_asr_finalization()
            elif self.backend.is_tensorrt():
                client.set_eos(True)
            return False

        if self.backend.is_tensorrt():
            voice_active = self.voice_activity(websocket, frame_np)
            if voice_active:
                self.no_voice_activity_chunks = 0
                client.set_eos(False)
            if getattr(client, "use_vad", self.use_vad) and not voice_active:
                return True

        client.add_frames(frame_np)
        return True

    def recv_audio(self,
                   websocket,   
                   backend: BackendType = BackendType.FASTER_WHISPER,
                   faster_whisper_custom_model_path=None,
                   whisper_tensorrt_path=None,
                   trt_multilingual=False,
                   trt_py_session=False,
                   funasr_model=None,
                   funasr_device="auto",
                   funasr_mode="sensevoice",
                   funasr_punc_model=None,
                   funasr_vad_model=None,
                   funasr_final_model="model/funasr/SenseVoiceSmall",
                   funasr_final_device=None,
                   funasr_final_refine=True):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            backend (str): The backend to run the server with.
            faster_whisper_custom_model_path (str): path to custom faster whisper model.
            whisper_tensorrt_path (str): Required for tensorrt backend.
            trt_multilingual(bool): Only used for tensorrt, True if multilingual model.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        # Backend selection is process configuration set by run(), not connection state.
        if getattr(self, "backend", backend) != backend:
            raise RuntimeError("Connection backend does not match server configuration")
        if not self.handle_new_connection(websocket, faster_whisper_custom_model_path,
                                          whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session,
                                          funasr_model=funasr_model, funasr_device=funasr_device,
                                          asr_device_index=self.asr_device_index,
                                          funasr_mode=funasr_mode, funasr_punc_model=funasr_punc_model,
                                          funasr_vad_model=funasr_vad_model,
                                          funasr_final_model=funasr_final_model,
                                          funasr_final_device=funasr_final_device,
                                          funasr_final_refine=funasr_final_refine):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            wl_metrics.track_connection_closed()
            del websocket

    def run(self,
            host,
            port=9090,
            backend="tensorrt",
            faster_whisper_custom_model_path=None,
            whisper_tensorrt_path=None,
            funasr_model=None,
            funasr_mode="sensevoice",
            funasr_punc_model=None,
            funasr_vad_model=None,
            funasr_final_model="model/funasr/SenseVoiceSmall",
            funasr_final_device=None,
            funasr_final_refine=True,
            funasr_device="auto",
            trt_multilingual=False,
            trt_py_session=False,
            single_model=False,
            max_clients=4,
            max_connection_time=600,
            cache_path="~/.cache/whisper-live/",
            rest_port=8000,
            enable_rest=False,
            cors_origins: Optional[str] = None,
            batch_enabled=False,
            batch_max_size=8,
            batch_window_ms=50,
            raw_pcm_input=False,
            metrics_port: int = 0,
            hotwords_file=None,
            asr_device_index=0,
            translation_device="cpu",
            meeting_hotwords_dir="config/hotwords.d",
            asr_corrections_dir="config/asr_corrections.d",
            asr_corrections_file=None,
            meeting_logs_dir="logs",
            summary_base_url="http://127.0.0.1:8001/v1",
            summary_model="qwen3-8b-awq",
            summary_startup_command="bash scripts/start_summary_llm_service.sh",
            summary_timeout=600,
            summary_ready_timeout=300,
            summary_max_chars_per_chunk=4000,
            summary_idle_shutdown_seconds=60,
            summary_templates_dir="config/summary_templates",
            segment_post_processor=None):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
            batch_enabled (bool): Enable cross-client GPU batch inference for
                the faster_whisper backend. When enabled, ``single_model`` is
                forced to True and a ``BatchInferenceWorker`` is started after
                the first client connects. Defaults to False.
            batch_max_size (int): Maximum number of requests per GPU batch.
                Defaults to 8.
            batch_window_ms (int): Maximum time in milliseconds to wait for
                the batch to fill after the first request arrives. Defaults
                to 50.
            segment_post_processor (callable, optional): A callable that receives
                a transcription segment dict and returns a modified segment dict.
                Applied to every segment before sending to the client. Useful for
                plugging in custom post-processing (e.g. formatting, redaction).
                Defaults to None.
        """
        self.cache_path = cache_path
        self.raw_pcm_input = raw_pcm_input
        self.asr_device_index = int(asr_device_index or 0)
        self.translation_device = translation_device
        self.backend = BackendType(backend)
        self.meeting_hotwords = MeetingHotwordStore(meeting_hotwords_dir)
        self.meeting_asr_corrections = MeetingAsrCorrectionStore(asr_corrections_dir)
        self.asr_corrections_file = asr_corrections_file
        self.meeting_logs = MeetingLogStore(meeting_logs_dir)
        self.summary_templates = SummaryTemplateStore(summary_templates_dir)
        self.meeting_summary = MeetingSummaryService(
            base_url=summary_base_url, model=summary_model, startup_command=summary_startup_command,
            timeout=summary_timeout, ready_timeout=summary_ready_timeout,
            max_chars_per_chunk=summary_max_chars_per_chunk, idle_shutdown_seconds=summary_idle_shutdown_seconds,
        )
        self.default_hotwords = self.load_hotwords_file(hotwords_file)
        if self.default_hotwords:
            logging.info(
                "Loaded %d default hotword tokens from %s",
                len(self.default_hotwords.split()),
                hotwords_file,
            )

        if max_clients < 1:
            raise ValueError(f"max_clients must be >= 1, got {max_clients}")
        if max_connection_time <= 0:
            raise ValueError(f"max_connection_time must be > 0, got {max_connection_time}")
        if batch_enabled and batch_max_size < 1:
            raise ValueError(f"batch_max_size must be >= 1, got {batch_max_size}")
        if batch_enabled and batch_window_ms < 0:
            raise ValueError(f"batch_window_ms must be >= 0, got {batch_window_ms}")

        self.segment_post_processor = segment_post_processor
        self.client_manager = ClientManager(max_clients, max_connection_time)
        if faster_whisper_custom_model_path is not None and not os.path.exists(faster_whisper_custom_model_path):
            if "/" not in faster_whisper_custom_model_path:
                raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path or HuggingFace model.")
        if whisper_tensorrt_path is not None and not os.path.exists(whisper_tensorrt_path):
            raise ValueError(f"TensorRT model '{whisper_tensorrt_path}' is not a valid path.")

        # Batch inference config
        if batch_enabled:
            single_model = True  # Batch mode requires shared model
            self.batch_config = {
                'max_batch_size': batch_max_size,
                'batch_window_ms': batch_window_ms,
                'max_pending_requests': max(1, min(max_clients * 2, batch_max_size * 4)),
            }
            logging.info(f"Batch inference enabled (max_batch={batch_max_size}, window={batch_window_ms}ms)")
        else:
            self.batch_config = None

        if single_model:
            if faster_whisper_custom_model_path or whisper_tensorrt_path or backend == BackendType.FUNASR.value:
                logging.info("Custom model option was provided. Switching to single model mode.")
                self.single_model = True
                # TODO: load model initially
            else:
                logging.info("Single model mode currently only works with custom models.")
        if not BackendType.is_valid(backend):
            raise ValueError(f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")

        # Start Prometheus metrics endpoint if port is specified
        if metrics_port > 0:
            wl_metrics.start_metrics_server(metrics_port)

        # Admin status API is always available on rest_port. The OpenAI-compatible
        # REST API is added to the same app when enable_rest is true.
        app = FastAPI(title="WhisperLive Admin API")
        origins = [o.strip() for o in cors_origins.split(',')] if cors_origins else self._default_cors_origins(port)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        warmup_config = {
            "backend": BackendType(backend),
            "faster_whisper_custom_model_path": faster_whisper_custom_model_path,
            "asr_device_index": self.asr_device_index,
            "funasr_model": funasr_model,
            "funasr_device": funasr_device,
            "funasr_mode": funasr_mode,
            "funasr_vad_model": funasr_vad_model,
            "funasr_final_model": funasr_final_model,
            "funasr_final_device": funasr_final_device,
            "funasr_final_refine": funasr_final_refine,
            "translation_device": translation_device,
            "translation_provider": "helsinki_zh_en",
            "zh_en_model_path": "model/opus-mt-zh-en",
            "en_zh_model_path": "model/opus-mt-en-zh",
            "nllb_model_path": "model/NLLB-200-600M",
        }

        @app.get("/admin/clients")
        async def admin_clients():
            return self.get_admin_clients_payload()

        @app.get("/admin/warmup/status")
        async def admin_warmup_status():
            return self.get_admin_warmup_status_payload()

        @app.get("/admin/translation-models")
        async def admin_translation_models():
            return self.get_translation_models_payload()

        @app.post("/admin/warmup")
        async def admin_warmup(request: Request, force: bool = False):
            config = dict(warmup_config)
            try:
                body = await request.json()
            except Exception:
                body = {}
            if isinstance(body, dict):
                for key in (
                    "translation_provider",
                    "zh_en_model_path",
                    "en_zh_model_path",
                    "nllb_model_path",
                    "asr_device_index",
                    "translation_device",
                ):
                    if body.get(key):
                        config[key] = body[key]
            return self.start_admin_warmup(config, force=force)

        @app.delete("/admin/clients/{uid}")
        async def delete_admin_client(uid: str):
            websocket, status = self.client_manager.get_client_status_entry(uid)
            if not status:
                return JSONResponse(
                    status_code=404,
                    content={"deleted": False, "uid": uid, "error": "client not found"},
                )

            if status.get("connected"):
                client = self.client_manager.get_client(websocket)
                if client:
                    try:
                        client.disconnect()
                    except Exception as e:
                        logging.warning("Admin failed to notify client disconnect: uid=%s error=%s", uid, e)
                self.cleanup(websocket)
                self.client_manager.delete_disconnected_client_status(uid)
                return {"deleted": True, "uid": uid, "disconnected": True}

            result = self.client_manager.delete_disconnected_client_status(uid)
            if result == "deleted":
                return {"deleted": True, "uid": uid, "disconnected": False}
            return JSONResponse(
                status_code=404,
                content={"deleted": False, "uid": uid, "error": "client not found"},
            )

        @app.get("/admin/hotwords")
        async def list_admin_hotwords():
            return self.meeting_hotwords.list()

        @app.get("/admin/hotwords/{meeting_name}")
        async def get_admin_hotwords(meeting_name: str):
            try:
                return self.meeting_hotwords.get(meeting_name)
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @app.post("/admin/hotwords/parse-upload")
        async def parse_admin_hotword_upload(file: UploadFile):
            try:
                content = await file.read()
                return self.parse_hotword_upload(file.filename, content)
            except RuntimeError as exc:
                return JSONResponse(status_code=501, content={"error": str(exc)})
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})
            except Exception as exc:
                logging.error("Failed to parse uploaded hotword file: %s", exc)
                return JSONResponse(status_code=500, content={"error": str(exc)})

        @app.post("/admin/meeting-logs")
        async def save_admin_meeting_log(request: Request):
            try:
                payload = await request.json()
                return self.meeting_logs.save(payload)
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"saved": False, "error": "request body must be valid JSON"})
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"saved": False, "error": str(exc)})
            except Exception as exc:
                logging.error("Failed to save meeting log: %s", exc)
                return JSONResponse(status_code=500, content={"saved": False, "error": str(exc)})

        @app.get("/admin/meeting-logs")
        async def list_admin_meeting_logs():
            return self.meeting_logs.list_sessions()

        @app.get("/admin/meeting-logs/{session_id}")
        async def download_admin_meeting_log(session_id: str, format: str = "md", layout: str = "sections"):
            file_format = str(format or "md").lower()
            if file_format not in {"md", "json", "docx"}:
                return JSONResponse(status_code=404, content={"error": "unsupported meeting log format"})
            result = self.meeting_logs.get_session_file(session_id, file_format, layout=layout)
            if not result or not os.path.isfile(result[0]):
                return JSONResponse(status_code=404, content={"error": "meeting log not found"})
            return FileResponse(result[0], media_type=result[1], filename=result[2])

        @app.get("/admin/meeting-logs/{session_id}/info")
        async def get_admin_meeting_log_info(session_id: str):
            info = self.meeting_logs.session_info(session_id)
            return info if info else JSONResponse(status_code=404, content={"error": "meeting log session not found"})

        @app.get("/admin/meeting-logs/{session_id}/transcript")
        async def get_admin_meeting_transcript(session_id: str):
            result = self.meeting_logs.get_transcript(session_id)
            return result if result else JSONResponse(status_code=404, content={"error": "meeting log session not found"})

        @app.patch("/admin/meeting-logs/{session_id}/transcript/{segment_id}")
        async def update_admin_meeting_transcript_segment(session_id: str, segment_id: str, request: Request):
            try:
                body = await request.json()
                return self.meeting_logs.update_transcript_segment(
                    session_id,
                    segment_id,
                    body.get("text"),
                    body.get("speaker_id"),
                    body.get("expected_revision"),
                )
            except TranscriptRevisionConflict as exc:
                return JSONResponse(status_code=409, content={"error": str(exc), "error_code": "transcript_revision_conflict"})
            except KeyError as exc:
                return JSONResponse(status_code=404, content={"error": str(exc)})
            except RuntimeError as exc:
                return JSONResponse(status_code=409, content={"error": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @app.post("/admin/meeting-logs/{session_id}/speakers")
        async def create_admin_meeting_speaker(session_id: str, request: Request):
            try:
                body = await request.json()
                return self.meeting_logs.add_transcript_speaker(
                    session_id, body.get("name"), body.get("expected_revision")
                )
            except TranscriptRevisionConflict as exc:
                return JSONResponse(status_code=409, content={"error": str(exc), "error_code": "transcript_revision_conflict"})
            except KeyError as exc:
                return JSONResponse(status_code=404, content={"error": str(exc)})
            except RuntimeError as exc:
                return JSONResponse(status_code=409, content={"error": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @app.patch("/admin/meeting-logs/{session_id}/speakers/{speaker_id}")
        async def rename_admin_meeting_speaker(session_id: str, speaker_id: str, request: Request):
            try:
                body = await request.json()
                return self.meeting_logs.rename_transcript_speaker(
                    session_id, speaker_id, body.get("name"), body.get("expected_revision")
                )
            except TranscriptRevisionConflict as exc:
                return JSONResponse(status_code=409, content={"error": str(exc), "error_code": "transcript_revision_conflict"})
            except KeyError as exc:
                return JSONResponse(status_code=404, content={"error": str(exc)})
            except RuntimeError as exc:
                return JSONResponse(status_code=409, content={"error": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @app.post("/admin/meeting-logs/{session_id}/speakers/merge")
        async def merge_admin_meeting_speakers(session_id: str, request: Request):
            try:
                body = await request.json()
                return self.meeting_logs.merge_transcript_speakers(
                    session_id,
                    body.get("source_speaker_id"),
                    body.get("target_speaker_id"),
                    body.get("expected_revision"),
                )
            except TranscriptRevisionConflict as exc:
                return JSONResponse(status_code=409, content={"error": str(exc), "error_code": "transcript_revision_conflict"})
            except KeyError as exc:
                return JSONResponse(status_code=404, content={"error": str(exc)})
            except RuntimeError as exc:
                return JSONResponse(status_code=409, content={"error": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @app.post("/admin/meeting-logs/{session_id}/finish")
        async def finish_admin_meeting_log(session_id: str, request: Request):
            info = self.meeting_logs.finish_session(session_id)
            return info if info else JSONResponse(status_code=404, content={"error": "meeting log session not found"})

        @app.get("/admin/summary-templates")
        async def list_admin_summary_templates():
            return self.summary_templates.list()

        @app.get("/admin/summary-templates/{template_id}")
        async def get_admin_summary_template(template_id: str):
            definition = self.summary_templates.get(template_id)
            if not definition:
                return JSONResponse(status_code=404, content={"error": "summary template not found"})
            return {key: value for key, value in definition.items() if key != "markdown"}

        @app.delete("/admin/summary-templates/{template_id}")
        async def delete_admin_summary_template(template_id: str):
            result = self.summary_templates.delete(template_id)
            if not result:
                return JSONResponse(status_code=404, content={"error": "summary template not found"})
            return result

        @app.post("/admin/summary-templates/analyze")
        async def analyze_admin_summary_template(file: UploadFile):
            try:
                filename = os.path.basename(file.filename or "")
                lower_filename = filename.lower()
                if not lower_filename.endswith((".md", ".docx")):
                    raise ValueError("只支持上传 .md 或 .docx 模板文件")
                content = await file.read(SummaryTemplateStore.MAX_FILE_BYTES + 1)
                if len(content) > SummaryTemplateStore.MAX_FILE_BYTES:
                    raise ValueError("模板文件不能超过 2 MB")
                if lower_filename.endswith(".md"):
                    try:
                        markdown = content.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError("Markdown 模板必须使用 UTF-8 编码") from exc
                    empty_sections_message = "Markdown 模板至少需要一个二级或更低级标题"
                else:
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
                            temp_file.write(content)
                            temp_path = temp_file.name
                        markdown = MeetingDocConverter.docx_to_md_text(
                            temp_path,
                            promote_plain_headings=True,
                        )
                    except Exception as exc:
                        raise ValueError(f"DOCX 模板解析失败: {exc}") from exc
                    finally:
                        if temp_path:
                            try:
                                os.unlink(temp_path)
                            except OSError:
                                pass
                    empty_sections_message = "DOCX 模板未识别到字段栏目；请使用标题样式，或包含会议基本信息、会议议题、讨论事项综述等栏目名"
                sections = self.summary_templates._extract_sections(markdown)
                if not sections:
                    raise ValueError(empty_sections_message)
                content_sections = [section for section in sections if section.get("role") != "container"]
                fields = await asyncio.to_thread(
                    self.meeting_summary.analyze_custom_template,
                    markdown,
                    content_sections,
                )
                return self.summary_templates.create_draft(filename, markdown, fields)
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})
            except Exception as exc:
                logging.error("Failed to analyze summary template: %s", exc)
                return JSONResponse(status_code=500, content={"error": str(exc)})

        @app.post("/admin/summary-templates/{draft_id}/confirm")
        async def confirm_admin_summary_template(draft_id: str, request: Request):
            try:
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                definition = self.summary_templates.confirm(draft_id, body.get("name"), body.get("fields"))
                return {"saved": True, "template": definition}
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"error": "request body must be valid JSON"})
            except KeyError as exc:
                return JSONResponse(status_code=404, content={"error": str(exc)})
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})
            except Exception as exc:
                logging.error("Failed to save summary template: %s", exc)
                return JSONResponse(status_code=500, content={"error": str(exc)})

        @app.post("/admin/meeting-logs/{session_id}/summary")
        async def generate_admin_meeting_summary(session_id: str, request: Request):
            try:
                raw_body = await request.body()
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                return await asyncio.to_thread(
                    self.generate_meeting_summary,
                    session_id,
                    body.get("template") or "auto",
                    body.get("custom_template_id"),
                )
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"generated": False, "error": "request body must be valid JSON"})
            except KeyError as exc:
                return JSONResponse(status_code=404, content={"generated": False, "error": str(exc)})
            except SummaryGenerationError as exc:
                logging.error("Meeting summary generation failed: %s", exc.code)
                return JSONResponse(
                    status_code=502,
                    content={
                        "generated": False,
                        "error": str(exc),
                        "error_code": exc.code,
                        "details": exc.details,
                    },
                )
            except RuntimeError as exc:
                return JSONResponse(status_code=409, content={"generated": False, "error": str(exc)})
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"generated": False, "error": str(exc)})
            except Exception as exc:
                logging.error("Failed to generate meeting summary: %s", exc)
                return JSONResponse(status_code=500, content={"generated": False, "error": str(exc)})

        @app.get("/admin/meeting-logs/{session_id}/summary")
        async def download_admin_meeting_summary(session_id: str, format: str = "md", version: Optional[int] = None):
            file_format = str(format or "md").lower()
            if file_format not in {"md", "json", "docx"}:
                return JSONResponse(status_code=404, content={"error": "unsupported summary format"})
            try:
                result = self.meeting_logs.get_summary_file(session_id, file_format, version=version)
            except RuntimeError as exc:
                return JSONResponse(status_code=500, content={"error": str(exc)})
            if not result or not os.path.isfile(result[0]):
                return JSONResponse(status_code=404, content={"error": "meeting summary not found"})
            return FileResponse(result[0], media_type=result[1], filename=result[2])

        @app.get("/admin/meeting-logs/{session_id}/summary/info")
        async def get_admin_meeting_summary_info(session_id: str):
            info = self.meeting_logs.summary_info(session_id)
            return info if info else JSONResponse(status_code=404, content={"error": "meeting log session not found"})

        if enable_rest:
            @app.post("/v1/audio/transcriptions")
            async def transcribe(
                file: UploadFile,
                model: str = Form(default="whisper-1"),
                language: Optional[str] = Form(default=None),
                prompt: Optional[str] = Form(default=None),
                response_format: str = Form(default="json"),
                temperature: float = Form(default=0.0),
                timestamp_granularities: Optional[List[str]] = Form(default=None),
                # Stubs for unsupported OpenAI params
                chunking_strategy: Optional[str] = Form(default=None),
                include: Optional[List[str]] = Form(default=None),
                known_speaker_names: Optional[List[str]] = Form(default=None),
                known_speaker_references: Optional[List[str]] = Form(default=None),
                stream: bool = Form(default=False),
                hotwords: Optional[str] = Form(default=None),
            ):
                if stream:
                    wl_metrics.track_rest_request(endpoint="transcriptions", status=400)
                    return JSONResponse({"error": "Streaming not supported in this backend."}, status_code=400)
                if chunking_strategy or known_speaker_names or known_speaker_references:
                    logging.warning("Diarization/chunking params ignored; not supported.")

                supported_formats = ["json", "text", "srt", "verbose_json", "vtt"]
                if response_format not in supported_formats:
                    wl_metrics.track_rest_request(endpoint="transcriptions", status=400)
                    return JSONResponse({"error": f"Unsupported response_format. Supported: {supported_formats}"}, status_code=400)

                if model != "whisper-1":
                    logging.warning(f"Model '{model}' requested; using 'small' as fallback.")
                model_name = faster_whisper_custom_model_path or self.resolve_asr_model_path("small")

                try:
                    suffix = os.path.splitext(file.filename)[1] or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_path = tmp.name

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if device == "cuda":
                        major, _ = torch.cuda.get_device_capability(self.asr_device_index)
                        compute_type = "float16" if major >= 7 else "float32"
                    else:
                        compute_type = "int8"

                    transcriber = WhisperModel(
                        model_name,
                        device=device,
                        device_index=self.asr_device_index if device == "cuda" else 0,
                        compute_type=compute_type,
                    )
                    segments, info = transcriber.transcribe(
                        tmp_path,
                        language=language,
                        initial_prompt=prompt,
                        temperature=temperature,
                        vad_filter=False,
                        word_timestamps=(timestamp_granularities and "word" in timestamp_granularities),
                        hotwords=hotwords,
                    )

                    text = " ".join([s.text.strip() for s in segments])
                    os.unlink(tmp_path)

                    if response_format == "text":
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return PlainTextResponse(text)
                    elif response_format == "json":
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return {"text": text}
                    elif response_format == "verbose_json":
                        verbose = {
                            "task": "transcribe",
                            "language": info.language,
                            "duration": info.duration,
                            "text": text,
                            "segments": []
                        }
                        for seg in segments:
                            seg_dict = {
                                "id": seg.id,
                                "seek": seg.seek,
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text.strip(),
                                "tokens": seg.tokens,
                                "temperature": seg.temperature,
                                "avg_logprob": seg.avg_logprob,
                                "compression_ratio": seg.compression_ratio,
                                "no_speech_prob": seg.no_speech_prob
                            }
                            if timestamp_granularities and "word" in timestamp_granularities:
                                seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words]
                            verbose["segments"].append(seg_dict)
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return verbose
                    elif response_format in ["srt", "vtt"]:
                        output = []
                        for i, seg in enumerate(segments, 1):
                            start = f"{int(seg.start // 3600):02}:{int((seg.start % 3600) // 60):02}:{seg.start % 60:06.3f}"
                            end = f"{int(seg.end // 3600):02}:{int((seg.end % 3600) // 60):02}:{seg.end % 60:06.3f}"
                            if response_format == "srt":
                                output.append(f"{i}\n{start.replace('.', ',')} --> {end.replace('.', ',')}\n{seg.text.strip()}\n")
                            else:  # vtt
                                output.append(f"{start} --> {end}\n{seg.text.strip()}\n")
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return PlainTextResponse("\n".join(output))
                except Exception as e:
                    wl_metrics.track_rest_request(endpoint="transcriptions", status=500)
                    wl_metrics.track_error("rest_transcription")
                    return JSONResponse({"error": str(e)}, status_code=500)

        threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": rest_port, "log_level": "info"},
            daemon=True
        ).start()
        if enable_rest:
            logging.info(f"OpenAI-compatible REST API started on http://0.0.0.0:{rest_port}")
        logging.info(f"Admin API available at http://0.0.0.0:{rest_port}/admin/clients")

        # Original WebSocket server (always supported)
        try:
            with serve(
                functools.partial(
                    self.recv_audio,
                    backend=BackendType(backend),
                    faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                    whisper_tensorrt_path=whisper_tensorrt_path,
                    funasr_model=funasr_model,
                    funasr_mode=funasr_mode,
                    funasr_punc_model=funasr_punc_model,
                    funasr_vad_model=funasr_vad_model,
                    funasr_final_model=funasr_final_model,
                    funasr_final_device=funasr_final_device,
                    funasr_final_refine=funasr_final_refine,
                    funasr_device=funasr_device,
                    trt_multilingual=trt_multilingual,
                    trt_py_session=trt_py_session,
                ),
                host,
                port
            ) as server:
                server.serve_forever()
        finally:
            worker = getattr(ServeClientBase, "BATCH_WORKER", None)
            try:
                from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
                worker = ServeClientFasterWhisper.BATCH_WORKER
                if worker:
                    worker.stop(timeout=5.0)
                ServeClientFasterWhisper.BATCH_WORKER = None
            except Exception as exc:
                logging.warning("[BATCH_WORKER_SHUTDOWN_FAILED] %s", exc)
            self.meeting_summary.close()

    def voice_activity(self, websocket, frame_np):
        """
        Evaluates the voice activity in a given audio frame and manages the state of voice activity detection.

        This method uses the configured voice activity detection (VAD) model to assess whether the given audio frame
        contains speech. If the VAD model detects no voice activity for more than three consecutive frames,
        it sets an end-of-speech (EOS) flag for the associated client. This method aims to efficiently manage
        speech detection to improve subsequent processing steps.

        Args:
            websocket: The websocket associated with the current client. Used to retrieve the client object
                    from the client manager for state management.
            frame_np (numpy.ndarray): The audio frame to be analyzed. This should be a NumPy array containing
                                    the audio data for the current frame.

        Returns:
            bool: True if voice activity is detected in the current frame, False otherwise. When returning False
                after detecting no voice activity for more than three consecutive frames, it also triggers the
                end-of-speech (EOS) flag for the client.
        """
        if not self.vad_detector(frame_np):
            self.no_voice_activity_chunks += 1
            if self.no_voice_activity_chunks > 3:
                client = self.client_manager.get_client(websocket)
                if not client.eos:
                    client.set_eos(True)
                time.sleep(0.1)    # Sleep 100m; wait some voice activity.
            return False
        return True

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        client = self.client_manager.get_client(websocket)
        if client:
            ended_by_client = bool(getattr(websocket, "whisperlive_end_of_audio", False))
            asr_status = "not_applicable"
            translation_status = "not_applicable"
            translation_timeout_count = 0
            if ended_by_client:
                started_at = time.monotonic()
                backend = getattr(self, "backend", None)
                if backend is not None and backend.is_faster_whisper() and hasattr(client, "wait_for_asr_finalization"):
                    asr_status = client.wait_for_asr_finalization(self.FINALIZATION_BUDGET_SECONDS)
                if hasattr(client, "flush_pending_completed_segments"):
                    released_segments = client.flush_pending_completed_segments(force=True)
                    if released_segments:
                        client.send_transcription_to_client(released_segments)
                elapsed = time.monotonic() - started_at
                remaining = max(0.0, self.FINALIZATION_BUDGET_SECONDS - elapsed)
                translation_client = getattr(client, "translation_client", None)
                if translation_client and hasattr(translation_client, "finalize_translation_drain"):
                    translation_status = translation_client.finalize_translation_drain(remaining)
                    translation_timeout_count = getattr(translation_client, "translation_timeout_count", 0)
                logging.info(
                    "[SESSION_FINALIZATION] uid=%s asr=%s translation=%s timeouts=%s elapsed=%.2f",
                    getattr(client, "client_uid", ""),
                    asr_status,
                    translation_status,
                    translation_timeout_count,
                    time.monotonic() - started_at,
                )
            elif hasattr(client, "flush_pending_completed_segments"):
                released_segments = client.flush_pending_completed_segments(force=True)
                if released_segments:
                    client.send_transcription_to_client(released_segments)

            if getattr(client, "runtime_cleanup_started", False):
                return
            client.runtime_cleanup_started = True
            cleanup_started = time.monotonic()
            if hasattr(client, 'translation_client') and client.translation_client:
                client.translation_client.cleanup()

            # Wait for translation thread to finish
            if hasattr(client, 'translation_thread') and client.translation_thread:
                client.translation_thread.join(timeout=2.0)
            if not client.stop_and_join(timeout=2.0):
                logging.warning("[RUNTIME_ASR_JOIN_TIMEOUT] uid=%s", getattr(client, "client_uid", ""))
            self.finalize_client_meeting_log(websocket, interrupted=not ended_by_client)
            if ended_by_client:
                try:
                    websocket.send(json.dumps({
                        "uid": getattr(client, "client_uid", None),
                        "message": "SESSION_FINALIZED",
                        "session_id": getattr(client, "meeting_log_session_id", None),
                        "session_status": "finished",
                        "asr_finalization": asr_status,
                        "translation_drain": translation_status,
                        "translation_timeout_count": translation_timeout_count,
                    }))
                except Exception as exc:
                    logging.debug("Failed to send SESSION_FINALIZED: %s", exc)
            self.client_manager.mark_client_disconnected(websocket)
            self.client_manager.remove_client(websocket)
            logging.info(
                "[RUNTIME_CLEANUP] uid=%s duration=%.3f",
                getattr(client, "client_uid", ""), time.monotonic() - cleanup_started,
            )
