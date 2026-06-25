import os
import asyncio
import time
import threading
import queue
import json
import functools
import logging
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
    MeetingDocConverter,
    MeetingHotwordStore,
    MeetingLogStore,
    apply_timeline_offset_to_segments,
    MeetingSummaryService,
    SummaryGenerationError,
    SummaryTemplateStore,
    count_hotwords,
    hotword_text_to_prompt,
    normalize_hotword_text,
    parse_hotword_config,
)

logging.basicConfig(level=logging.INFO)


class ClientManager:
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

    @staticmethod
    def _latest_segment_text(segments):
        if not segments:
            return ""
        for segment in reversed(segments):
            text = segment.get("text", "") if isinstance(segment, dict) else ""
            if text:
                return text
        return ""

    def register_client_status(self, websocket, client, options, backend):
        now = time.time()
        uid = getattr(client, "client_uid", options.get("uid"))
        status = {
            "uid": uid,
            "client_instance_id": options.get("client_instance_id") or "",
            "client_name": options.get("client_name") or options.get("meeting_name") or f"Client-{str(uid)[:8]}",
            "meeting_name": options.get("meeting_name") or "",
            "hotwords_file": options.get("hotwords_file") or "",
            "hotwords_count": int(options.get("hotwords_count") or count_hotwords(options.get("hotwords"))),
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
            self.client_status[websocket] = status

    def update_client_message(self, websocket, message_type, segments):
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

    def get_client_status_snapshot(self):
        now = time.time()
        with self.lock:
            statuses = [dict(status) for status in self.client_status.values()]
        for status in statuses:
            connected_at = status.get("connected_at") or now
            last_activity_at = status.get("last_activity_at") or connected_at
            status["connected_seconds"] = round((status.get("disconnected_at") or now) - connected_at, 3)
            status["last_activity_seconds_ago"] = round(now - last_activity_at, 3)
        statuses.sort(key=lambda item: item.get("connected_at", 0), reverse=True)
        return {"server_time": now, "clients": statuses}

    def delete_disconnected_client_status(self, uid):
        with self.lock:
            for websocket, status in list(self.client_status.items()):
                if status.get("uid") != uid:
                    continue
                if status.get("connected"):
                    return "connected"
                del self.client_status[websocket]
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
    LOCAL_ASR_MODEL_ROOT = "model/asr"
    LOCAL_ASR_MODEL_NAMES = {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v3-turbo", "large-v3",
    }

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
        self.meeting_hotwords = MeetingHotwordStore()
        self.meeting_logs = MeetingLogStore()
        self.summary_templates = SummaryTemplateStore()
        self.meeting_summary = MeetingSummaryService()

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

        return " ".join(hotwords)

    def apply_meeting_hotwords(self, options):
        meeting_name = options.get("meeting_name")
        if not meeting_name or not self.meeting_hotwords:
            return
        stored = self.meeting_hotwords.get(meeting_name)
        prompt = hotword_text_to_prompt(stored.get("text"))
        if prompt and not options.get("hotwords"):
            options["hotwords"] = prompt
        if prompt or stored.get("translation_glossary"):
            options["hotwords_count"] = stored.get("count") or count_hotwords(stored.get("text"))
            options["hotwords_file"] = stored.get("filename") or ""
            options["hotwords_locked"] = True
            options["translation_glossary"] = dict(stored.get("translation_glossary") or {})
            options["translation_glossary_count"] = int(stored.get("translation_count") or 0)

    def apply_default_hotwords(self, options):
        if options.get("hotwords"):
            return
        if self.default_hotwords:
            options["hotwords"] = self.default_hotwords

    def get_admin_clients_payload(self):
        if not self.client_manager:
            return {"server_time": time.time(), "clients": []}
        return self.client_manager.get_client_status_snapshot()

    def session_timeline_offset(self, websocket):
        client = self.client_manager.get_client(websocket) if self.client_manager else None
        return float(getattr(client, "meeting_log_timeline_offset_seconds", 0.0) or 0.0) if client else 0.0

    def offset_client_segment(self, websocket, segment):
        return apply_timeline_offset_to_segments([segment], self.session_timeline_offset(websocket))[0]

    def offset_client_segments(self, websocket, segments):
        return apply_timeline_offset_to_segments(segments, self.session_timeline_offset(websocket))

    def process_client_segment(self, websocket, base_processor, segment):
        if base_processor is not None:
            segment = base_processor(segment) or segment
        return self.offset_client_segment(websocket, segment)

    def handle_client_segments(self, websocket, message_type, segments):
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
        try:
            if interrupted:
                return self.meeting_logs.interrupt_session(session_id)
            return self.meeting_logs.finish_session(session_id)
        except Exception as exc:
            logging.error("Failed to finalize meeting log: %s", exc)
            return None

    def generate_meeting_summary(self, session_id, template="auto", custom_template_id=None):
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
        funasr_model=None, funasr_device="auto",
        funasr_mode="sensevoice", funasr_punc_model=None, funasr_vad_model=None,
        funasr_final_model="model/funasr/SenseVoiceSmall", funasr_final_device=None,
        funasr_final_refine=True,
    ):
        client: Optional[ServeClientBase] = None

        # Check if client wants translation
        enable_translation = options.get("enable_translation", False)
        
        # Create translation queue if translation is enabled
        translation_queue = None
        translation_client = None
        translation_thread = None
        
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
                en_zh_model_path=options.get("en_zh_model_path", "model/opus-mt-en-zh"),
                translation_glossary=options.get("translation_glossary"),
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
                self.backend = BackendType.FASTER_WHISPER
        
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
                self.backend = BackendType.FASTER_WHISPER
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "OpenVINO not supported on Server yet. "
                                "Reverting to available backend: 'faster_whisper'"
                }))

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
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "ERROR",
                    "message": str(e)
                }))
                websocket.close()
                raise

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
                    use_vad=self.use_vad,
                    translation_queue=translation_queue,
                    hotwords=options.get("hotwords"),
                )
                logging.info("Running FunASR backend.")
            except Exception as e:
                logging.error(f"FunASR not supported: {e}")
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "ERROR",
                    "message": str(e)
                }))
                websocket.close()
                raise

        try:
            if self.backend.is_faster_whisper():
                from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
                # model is of the form namespace/repo_name and not a filesystem path
                if faster_whisper_custom_model_path is not None:
                    logging.info(f"Using custom model {faster_whisper_custom_model_path}")
                    options["model"] = faster_whisper_custom_model_path
                else:
                    options["model"] = self.resolve_asr_model_path(options["model"])
                client = ServeClientFasterWhisper(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    initial_prompt=options.get("initial_prompt"),
                    vad_parameters=options.get("vad_parameters"),
                    use_vad=self.use_vad,
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                    max_incomplete_segment_seconds=options.get("max_incomplete_segment_seconds", 0.0),
                    cache_path=self.cache_path,
                    translation_queue=translation_queue,
                    hotwords=options.get("hotwords"),
                    diarization=self._create_diarizer(options),
                    word_timestamps=options.get("word_timestamps", False),
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
            return

        if client is None:
            raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")

        # Attach segment post-processor if configured
        if self.segment_post_processor is not None:
            client.segment_post_processor = self.segment_post_processor

        if translation_client:
            client.translation_client = translation_client
            client.translation_thread = translation_thread

        if translation_client:
            translation_client.admin_status_callback = functools.partial(
                self.handle_client_segments, websocket, "translated_segments"
            )
        client.admin_status_callback = functools.partial(
            self.handle_client_segments, websocket, "segments"
        )
        try:
            if options.get("resume_session"):
                log_info = self.meeting_logs.resume_session(options, backend=self.backend)
            else:
                log_info = self.meeting_logs.start_session(options, backend=self.backend)
            client.meeting_log_session_id = log_info.get("session_id") if log_info else options.get("session_id") or options.get("uid")
            client.meeting_log_timeline_offset_seconds = float((log_info or {}).get("timeline_offset_seconds") or 0.0)
        except Exception as exc:
            logging.error("Failed to start meeting log session: %s", exc)
            client.meeting_log_session_id = options.get("session_id") or options.get("uid")
            client.meeting_log_timeline_offset_seconds = 0.0
        base_segment_processor = getattr(client, "segment_post_processor", None)
        client.segment_post_processor = functools.partial(
            self.process_client_segment,
            websocket,
            base_segment_processor,
        )
        if translation_client:
            translation_client.segment_post_processor = functools.partial(self.offset_client_segment, websocket)
        self.client_manager.add_client(websocket, client)
        self.client_manager.register_client_status(websocket, client, options, self.backend)
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
            }))
        except Exception as exc:
            logging.debug("Failed to send meeting session ready metadata: %s", exc)

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
                              funasr_model=None, funasr_device="auto",
                              funasr_mode="sensevoice", funasr_punc_model=None, funasr_vad_model=None,
                              funasr_final_model="model/funasr/SenseVoiceSmall", funasr_final_device=None,
                              funasr_final_refine=True):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)
            self.apply_meeting_hotwords(options)
            self.apply_default_hotwords(options)

            self.use_vad = options.get('use_vad')
            if self.client_manager.is_server_full(websocket, options):
                wl_metrics.track_connection_rejected(reason="full")
                websocket.close()
                return False  # Indicates that the connection should not continue

            if self.backend.is_tensorrt() and self.use_vad:
                self.vad_detector = VoiceActivityDetector(frame_rate=self.RATE)
            self.initialize_client(websocket, options, faster_whisper_custom_model_path,
                                   whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session,
                                   funasr_model=funasr_model, funasr_device=funasr_device,
                                   funasr_mode=funasr_mode, funasr_punc_model=funasr_punc_model,
                                   funasr_vad_model=funasr_vad_model,
                                   funasr_final_model=funasr_final_model,
                                   funasr_final_device=funasr_final_device,
                                   funasr_final_refine=funasr_final_refine)
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
            if self.backend.is_tensorrt():
                client.set_eos(True)
            return False

        if self.backend.is_tensorrt():
            voice_active = self.voice_activity(websocket, frame_np)
            if voice_active:
                self.no_voice_activity_chunks = 0
                client.set_eos(False)
            if self.use_vad and not voice_active:
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
        self.backend = backend
        if not self.handle_new_connection(websocket, faster_whisper_custom_model_path,
                                          whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session,
                                          funasr_model=funasr_model, funasr_device=funasr_device,
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
            translation_device="cpu",
            meeting_hotwords_dir="config/hotwords.d",
            meeting_logs_dir="logs",
            summary_base_url="http://127.0.0.1:8001/v1",
            summary_model="qwen3-8b-awq",
            summary_startup_command="bash scripts/start_summary_llm_service.sh",
            summary_timeout=600,
            summary_ready_timeout=300,
            summary_max_chars_per_chunk=4000,
            summary_idle_shutdown_seconds=600,
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
        self.translation_device = translation_device
        self.meeting_hotwords = MeetingHotwordStore(meeting_hotwords_dir)
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

        @app.get("/admin/clients")
        async def admin_clients():
            return self.get_admin_clients_payload()

        @app.delete("/admin/clients/{uid}")
        async def delete_admin_client(uid: str):
            result = self.client_manager.delete_disconnected_client_status(uid)
            if result == "deleted":
                return {"deleted": True, "uid": uid}
            if result == "connected":
                return JSONResponse(
                    status_code=409,
                    content={"deleted": False, "uid": uid, "error": "client is still connected"},
                )
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
                    compute_type = "float16" if device == "cuda" else "int8"

                    transcriber = WhisperModel(model_name, device=device, compute_type=compute_type)
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
            if hasattr(client, 'translation_client') and client.translation_client:
                client.translation_client.cleanup()
                
            # Wait for translation thread to finish
            if hasattr(client, 'translation_thread') and client.translation_thread:
                client.translation_thread.join(timeout=2.0)
            self.finalize_client_meeting_log(
                websocket,
                interrupted=not bool(getattr(websocket, "whisperlive_end_of_audio", False)),
            )
            self.client_manager.mark_client_disconnected(websocket)
            self.client_manager.remove_client(websocket)
