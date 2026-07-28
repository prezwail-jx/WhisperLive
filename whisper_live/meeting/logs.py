import ast
import json
import logging
import os
import re
import threading
import time

from .common import atomic_write, now_iso
from .docs import DOCX_MIME_TYPE, MeetingDocConverter
from .hotwords import count_hotwords
from .sessions import (
    SESSION_ACTIVE,
    SESSION_FINISHED,
    SESSION_INTERRUPTED,
    can_resume_payload,
    seconds_between,
)
from .transcript import (
    create_speaker,
    merge_speakers,
    normalize_transcript,
    rename_speaker,
    transcript_view,
    update_segment,
)


class MeetingLogStore:
    UNSAFE_FILENAME_CHARS = set('/\\:*?"<>|')

    def __init__(self, directory="logs", refresh_interval_seconds=0.5):
        self.directory = directory or "logs"
        self.lock = threading.RLock()
        self.sessions = {}
        self.local_session_ids = set()
        self.session_file_mtimes = {}
        self.refresh_interval_seconds = max(0.0, float(refresh_interval_seconds))
        self.last_refresh_at = 0.0
        os.makedirs(self.directory, exist_ok=True)
        self._load_existing_sessions(force=True)

    def _load_existing_sessions(self, force=False):
        loaded = 0
        for root, directories, filenames in os.walk(self.directory):
            directories[:] = [name for name in directories if not name.endswith("-summaries")]
            for filename in filenames:
                if not filename.endswith(".json"):
                    continue
                if filename.endswith("-summary.json"):
                    base_name = filename[: -len("-summary.json")] + ".json"
                    if os.path.exists(os.path.join(root, base_name)):
                        continue
                json_path = os.path.join(root, filename)
                try:
                    mtime = os.path.getmtime(json_path)
                except OSError:
                    continue
                if not force and self.session_file_mtimes.get(json_path) == mtime:
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as file:
                        payload = json.load(file)
                except (OSError, ValueError):
                    continue
                if not isinstance(payload, dict) or not payload.get("session_id"):
                    continue
                if not isinstance(payload.get("source_segments"), list):
                    continue
                normalize_transcript(payload)
                session_id = str(payload["session_id"])
                current = self.sessions.get(session_id)
                if current and session_id in self.local_session_ids and current["payload"].get("status") == SESSION_ACTIVE:
                    self.session_file_mtimes[json_path] = mtime
                    continue
                stem, _ext = os.path.splitext(json_path)
                md_path = f"{stem}.md"
                payload.setdefault("status", SESSION_INTERRUPTED if payload.get("status") == SESSION_ACTIVE else payload.get("status") or SESSION_INTERRUPTED)
                payload.setdefault("connection_count", 1)
                payload.setdefault("audio_gaps", [])
                payload.setdefault("timeline_offset_seconds", 0.0)
                self.sessions[session_id] = {
                    "payload": payload,
                    "source_keys": {self.segment_key(item) for item in payload.get("source_segments") or [] if isinstance(item, dict)},
                    "translation_keys": {self.segment_key(item) for item in payload.get("translation_segments") or [] if isinstance(item, dict)},
                    "json_path": json_path,
                    "md_path": md_path,
                    "json_filename": filename,
                    "md_filename": os.path.basename(md_path),
                }
                self.session_file_mtimes[json_path] = mtime
                loaded += 1
        if loaded:
            logging.debug("Refreshed %d meeting log sessions from %s", loaded, self.directory)

    def refresh_sessions(self, force=False):
        now = time.time()
        with self.lock:
            if not force and now - self.last_refresh_at < self.refresh_interval_seconds:
                return
            self.last_refresh_at = now
            self._load_existing_sessions(force=force)

    @classmethod
    def safe_name(cls, value):
        name = str(value or "").strip() or "meeting"
        cleaned = "".join("_" if char in cls.UNSAFE_FILENAME_CHARS or ord(char) < 32 else char for char in name)
        cleaned = cleaned.strip(" ._") or "meeting"
        return cleaned[:80]

    @classmethod
    def safe_timestamp(cls, value):
        raw = str(value or "").strip() or cls.timestamp_for_filename()
        cleaned = "".join("_" if char in cls.UNSAFE_FILENAME_CHARS or ord(char) < 32 else char for char in raw)
        cleaned = cleaned.replace(":", "-").replace(".", "-").strip(" ._")
        return cleaned[:80] or cls.timestamp_for_filename()

    @staticmethod
    def timestamp_for_filename():
        return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())

    now_iso = staticmethod(now_iso)

    @staticmethod
    def segment_key(segment):
        return f"{segment.get('session_id', '')}|{segment.get('start', '')}|{segment.get('end', '')}|{segment.get('text', '')}"

    def session_paths(self, payload):
        meeting_dir = self.safe_name(payload.get("meeting_name") or payload.get("client_name") or "meeting")
        session_id = self.safe_name(payload.get("session_id") or payload.get("uid") or "session")
        started_at = self.safe_timestamp(payload.get("session_started_at") or payload.get("created_at"))
        directory = os.path.join(self.directory, meeting_dir)
        stem = f"{started_at}-{session_id}"
        return directory, stem, os.path.join(directory, f"{stem}.json"), os.path.join(directory, f"{stem}.md")

    def build_payload_from_options(self, options, backend=None):
        session_id = options.get("session_id") or options.get("uid")
        created_at = options.get("session_started_at") or self.now_iso()
        return {
            "meeting_id": options.get("meeting_id") or session_id,
            "session_id": session_id,
            "uid": options.get("uid"),
            "client_instance_id": options.get("client_instance_id") or "",
            "client_name": options.get("client_name") or options.get("meeting_name") or f"Client-{str(options.get('uid', ''))[:8]}",
            "meeting_name": options.get("meeting_name") or "",
            "hotwords_count": int(options.get("hotwords_count") or count_hotwords(options.get("hotwords"))),
            "hotwords_file": options.get("hotwords_file") or "",
            "created_at": created_at,
            "updated_at": created_at,
            "exported_at": None,
            "server": options.get("server") or "",
            "backend": backend.value if hasattr(backend, "value") else str(backend or options.get("backend") or ""),
            "model": options.get("model"),
            "source_language": options.get("language"),
            "translation_mode": options.get("translation_mode") or options.get("target_language", "auto"),
            "translation_provider": options.get("translation_provider") or "",
            "status": SESSION_ACTIVE,
            "connection_count": 1,
            "timeline_offset_seconds": 0.0,
            "interrupted_at": None,
            "resumed_at": None,
            "audio_gaps": [],
            "source_segments": [],
            "translation_segments": [],
            "speakers": [],
            "transcript_revision": 0,
            "transcript_edits": [],
            "transcript_updated_at": None,
            "translation_stale": False,
            "summary_stale": False,
        }

    def save(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("meeting log payload must be a JSON object")
        meeting_name = payload.get("meeting_name") or payload.get("client_name") or "meeting"
        filename_stem = f"{self.safe_name(meeting_name)}-{self.timestamp_for_filename()}"
        with self.lock:
            filename = f"{filename_stem}.json"
            path = os.path.join(self.directory, filename)
            suffix = 1
            while os.path.exists(path):
                filename = f"{filename_stem}-{suffix}.json"; path = os.path.join(self.directory, filename); suffix += 1
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2); file.write("\n")
        return {"saved": True, "filename": filename, "path": path}

    def start_session(self, options, backend=None):
        payload = self.build_payload_from_options(options, backend=backend)
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")
        directory, _stem, json_path, md_path = self.session_paths(payload)
        with self.lock:
            self.refresh_sessions(force=True)
            os.makedirs(directory, exist_ok=True)
            if session_id in self.sessions:
                raise ValueError("session_id already exists; use resume_session")
            record = {"payload": payload, "source_keys": set(), "translation_keys": set(), "json_path": json_path, "md_path": md_path,
                      "json_filename": os.path.basename(json_path), "md_filename": os.path.basename(md_path)}
            self.sessions[session_id] = record
            self.local_session_ids.add(session_id)
            self._write_record(record)
        return self.session_info(session_id)

    def resume_session(self, options, backend=None):
        session_id = options.get("session_id") or options.get("uid")
        if not session_id:
            raise ValueError("session_id is required")
        with self.lock:
            self.refresh_sessions(force=True)
            record = self.sessions.get(session_id)
            if not record:
                raise KeyError("meeting log session not found")
            ok, reason = can_resume_payload(record["payload"], options)
            if not ok:
                raise ValueError(reason)
            payload = record["payload"]
            now = self.now_iso()
            last_time = payload.get("interrupted_at") or payload.get("updated_at") or payload.get("created_at")
            offset = seconds_between(payload.get("created_at"), now)
            payload["status"] = SESSION_ACTIVE
            payload["backend"] = backend.value if hasattr(backend, "value") else str(backend or payload.get("backend") or "")
            payload["updated_at"] = now
            payload["resumed_at"] = now
            payload["connection_count"] = int(payload.get("connection_count") or 1) + 1
            self.local_session_ids.add(session_id)
            payload["timeline_offset_seconds"] = round(offset, 3)
            if last_time:
                payload.setdefault("audio_gaps", []).append({
                    "start_at": last_time,
                    "end_at": now,
                    "reason": "websocket_disconnected",
                })
            self._write_record(record)
            return self.session_info(session_id)

    def interrupt_session(self, session_id):
        if not session_id:
            return None
        with self.lock:
            self.refresh_sessions(force=True)
            record = self.sessions.get(session_id)
            if not record:
                return None
            if record["payload"].get("status") == SESSION_FINISHED:
                return self.session_info(session_id)
            now = self.now_iso()
            record["payload"]["status"] = SESSION_INTERRUPTED
            record["payload"]["interrupted_at"] = now
            record["payload"]["updated_at"] = now
            self._write_record(record)
            return self.session_info(session_id)

    def append_segments(self, session_id, kind, segments):
        if not session_id or kind not in ("source", "translation"):
            return None
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            target = "source_segments" if kind == "source" else "translation_segments"
            key_name = "source_keys" if kind == "source" else "translation_keys"
            changed = False
            for segment in segments or []:
                if not isinstance(segment, dict) or not segment.get("completed"):
                    continue
                text = str(segment.get("text") or "").strip()
                if not text:
                    continue
                normalized = dict(segment); normalized["text"] = text; normalized.setdefault("session_id", session_id)
                key = self.segment_key(normalized)
                if key in record[key_name]:
                    continue
                record[key_name].add(key); record["payload"][target].append(normalized); changed = True
            if changed:
                normalize_transcript(record["payload"])
                record["payload"]["updated_at"] = self.now_iso()
                self._sort_segments(record["payload"][target])
                self._write_record(record)
            return self.session_info(session_id)

    def finish_session(self, session_id):
        if not session_id:
            return None
        with self.lock:
            self.refresh_sessions(force=True)
            record = self.sessions.get(session_id)
            if not record:
                return None
            record["payload"]["status"] = SESSION_FINISHED
            record["payload"]["exported_at"] = self.now_iso()
            record["payload"]["updated_at"] = record["payload"]["exported_at"]
            self._write_record(record)
            return self.session_info(session_id)

    @staticmethod
    def _clone_payload(payload):
        return json.loads(json.dumps(payload, ensure_ascii=False))

    def get_session_payload(self, session_id):
        self.refresh_sessions()
        with self.lock:
            record = self.sessions.get(session_id)
            return self._clone_payload(record["payload"]) if record else None

    def get_transcript(self, session_id):
        self.refresh_sessions()
        with self.lock:
            record = self.sessions.get(session_id)
            return self._clone_payload(transcript_view(record["payload"])) if record else None

    @staticmethod
    def _require_finished(record):
        if record["payload"].get("status") != SESSION_FINISHED:
            raise RuntimeError("请结束会议后再校对转写")

    def _save_transcript_change(self, record):
        payload = record["payload"]
        payload["updated_at"] = self.now_iso()
        record["source_keys"] = {
            self.segment_key(item)
            for item in payload.get("source_segments") or []
            if isinstance(item, dict)
        }
        self._write_record(record)
        return self._clone_payload(transcript_view(payload))

    def update_transcript_segment(self, session_id, segment_id, text, speaker_id, expected_revision):
        self.refresh_sessions()
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                raise KeyError("meeting log session not found")
            self._require_finished(record)
            changed = update_segment(
                record["payload"], segment_id, text, speaker_id,
                expected_revision, self.now_iso(),
            )
            if changed:
                return self._save_transcript_change(record)
            return self._clone_payload(transcript_view(record["payload"]))

    def add_transcript_speaker(self, session_id, name, expected_revision):
        self.refresh_sessions()
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                raise KeyError("meeting log session not found")
            self._require_finished(record)
            create_speaker(record["payload"], name, expected_revision, self.now_iso())
            return self._save_transcript_change(record)

    def rename_transcript_speaker(self, session_id, speaker_id, name, expected_revision):
        self.refresh_sessions()
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                raise KeyError("meeting log session not found")
            self._require_finished(record)
            changed = rename_speaker(
                record["payload"], speaker_id, name,
                expected_revision, self.now_iso(),
            )
            if changed:
                return self._save_transcript_change(record)
            return self._clone_payload(transcript_view(record["payload"]))

    def merge_transcript_speakers(self, session_id, source_id, target_id, expected_revision):
        self.refresh_sessions()
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                raise KeyError("meeting log session not found")
            self._require_finished(record)
            merge_speakers(
                record["payload"], source_id, target_id,
                expected_revision, self.now_iso(),
            )
            return self._save_transcript_change(record)

    @staticmethod
    def summary_paths_for_record(record):
        stem, _ext = os.path.splitext(record["json_path"])
        return f"{stem}-summary.json", f"{stem}-summary.md"

    @staticmethod
    def summary_version_directory(record):
        stem, _ext = os.path.splitext(record["json_path"])
        return f"{stem}-summaries"

    _atomic_write = staticmethod(atomic_write)

    def _version_entries(self, record):
        directory = self.summary_version_directory(record)
        entries = []
        if not os.path.isdir(directory):
            return entries
        for filename in sorted(os.listdir(directory)):
            match = re.fullmatch(r"v(\d{4})\.json", filename)
            if not match:
                continue
            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    summary = json.load(file)
            except (OSError, ValueError) as exc:
                logging.warning("Failed to read summary version %s: %s", path, exc)
                continue
            version = int(match.group(1))
            entries.append({
                "version": version,
                "template": summary.get("summary_template") or "legacy",
                "template_name": summary.get("custom_template_name"),
                "custom_template_id": summary.get("custom_template_id"),
                "meeting_type": summary.get("meeting_type") or "other",
                "generated_at": summary.get("generated_at"),
                "transcript_revision": int(summary.get("transcript_revision") or 0),
                "json_filename": filename,
                "md_filename": f"v{version:04d}.md",
            })
        return entries

    def _ensure_legacy_summary_version(self, record):
        entries = self._version_entries(record)
        if entries:
            return entries
        summary_json, summary_md = self.summary_paths_for_record(record)
        if not os.path.isfile(summary_json):
            return []
        try:
            with open(summary_json, "r", encoding="utf-8") as file:
                summary = json.load(file)
        except (OSError, ValueError) as exc:
            logging.warning("Failed to import legacy summary %s: %s", summary_json, exc)
            return []
        summary = dict(summary)
        summary["version"] = 1
        summary.setdefault("summary_template", "legacy")
        directory = self.summary_version_directory(record)
        version_json = os.path.join(directory, "v0001.json")
        version_md = os.path.join(directory, "v0001.md")
        self._atomic_write(version_json, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if os.path.isfile(summary_md):
            with open(summary_md, "r", encoding="utf-8") as file:
                markdown = file.read()
        else:
            markdown = self.render_summary_markdown(summary)
        self._atomic_write(version_md, markdown)
        return self._version_entries(record)

    def session_info(self, session_id):
        self.refresh_sessions()
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            summary = self.summary_info(session_id)
            return {
                "session_id": session_id,
                "meeting_name": record["payload"].get("meeting_name") or record["payload"].get("client_name") or "",
                "created_at": record["payload"].get("created_at"),
                "updated_at": record["payload"].get("updated_at"),
                "json_path": record["json_path"],
                "md_path": record["md_path"],
                "json_filename": record["json_filename"],
                "md_filename": record["md_filename"],
                "summary_json_path": summary["json_path"],
                "summary_md_path": summary["md_path"],
                "summary_json_filename": summary["json_filename"],
                "summary_md_filename": summary["md_filename"],
                "has_summary": summary["has_summary"],
                "latest_summary_version": summary["latest_version"],
                "source_count": len(record["payload"].get("source_segments", [])),
                "translation_count": len(record["payload"].get("translation_segments", [])),
                "status": record["payload"].get("status"),
                "connection_count": record["payload"].get("connection_count") or 1,
                "timeline_offset_seconds": record["payload"].get("timeline_offset_seconds") or 0.0,
                "interrupted_at": record["payload"].get("interrupted_at"),
                "resumed_at": record["payload"].get("resumed_at"),
                "audio_gaps": list(record["payload"].get("audio_gaps") or []),
                "transcript_revision": int(record["payload"].get("transcript_revision") or 0),
                "translation_stale": bool(record["payload"].get("translation_stale")),
                "summary_stale": bool(record["payload"].get("summary_stale")),
            }

    def list_sessions(self):
        self.refresh_sessions()
        with self.lock:
            sessions = [self.session_info(session_id) for session_id in list(self.sessions)]
        sessions = [item for item in sessions if item]
        sessions.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {"sessions": sessions}

    def get_session_file(self, session_id, file_format="md", layout="sections"):
        file_format = str(file_format or "md").lower()
        layout = str(layout or "sections").lower()
        self.refresh_sessions(force=True)
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            if file_format == "json":
                return record["json_path"], "application/json", record["json_filename"]
            if file_format not in {"md", "docx"}:
                return None
            md_path = record["md_path"]
            md_filename = record["md_filename"]
            if layout == "interleaved":
                stem, _ext = os.path.splitext(record["md_path"])
                md_path = f"{stem}-interleaved.md"
                md_filename = os.path.basename(md_path)
                self._atomic_write(md_path, self.render_markdown(record["payload"], layout="interleaved"))
            if file_format == "docx":
                docx_path = os.path.splitext(md_path)[0] + ".docx"
                MeetingDocConverter.md_file_to_docx(md_path, docx_path)
                return docx_path, DOCX_MIME_TYPE, os.path.basename(docx_path)
            return md_path, "text/markdown; charset=utf-8", md_filename

    def write_summary(self, session_id, summary):
        self.refresh_sessions(force=True)
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            entries = self._ensure_legacy_summary_version(record)
            version = max((item["version"] for item in entries), default=0) + 1
            summary = dict(summary)
            summary["version"] = version
            summary["transcript_revision"] = int(record["payload"].get("transcript_revision") or 0)
            summary_json, summary_md = self.summary_paths_for_record(record)
            directory = self.summary_version_directory(record)
            version_json = os.path.join(directory, f"v{version:04d}.json")
            version_md = os.path.join(directory, f"v{version:04d}.md")
            json_content = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            markdown_content = self.render_summary_markdown(summary)
            self._atomic_write(version_json, json_content)
            self._atomic_write(version_md, markdown_content)
            self._atomic_write(summary_json, json_content)
            self._atomic_write(summary_md, markdown_content)
            record["payload"]["summary_stale"] = False
            self._write_record(record)
        return self.summary_info(session_id)

    def summary_info(self, session_id):
        self.refresh_sessions()
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            summary_json, summary_md = self.summary_paths_for_record(record)
            versions = self._version_entries(record)
            latest = versions[-1] if versions else None
            return {
                "session_id": session_id,
                "json_path": summary_json,
                "md_path": summary_md,
                "json_filename": os.path.basename(summary_json),
                "md_filename": os.path.basename(summary_md),
                "has_summary": os.path.isfile(summary_json) and os.path.isfile(summary_md),
                "latest_version": latest["version"] if latest else None,
                "latest_template": latest["template"] if latest else None,
                "transcript_revision": int(record["payload"].get("transcript_revision") or 0),
                "summary_stale": bool(record["payload"].get("summary_stale")),
                "versions": versions,
            }

    def get_summary_file(self, session_id, file_format="md", version=None):
        file_format = str(file_format or "md").lower()
        self.refresh_sessions(force=True)
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            if version is None:
                summary_json, summary_md = self.summary_paths_for_record(record)
                if file_format == "json":
                    return summary_json, "application/json", os.path.basename(summary_json)
                if file_format == "docx":
                    docx_path = os.path.splitext(summary_md)[0] + ".docx"
                    MeetingDocConverter.md_file_to_docx(summary_md, docx_path)
                    return docx_path, DOCX_MIME_TYPE, os.path.basename(docx_path)
                return summary_md, "text/markdown; charset=utf-8", os.path.basename(summary_md)
            try:
                version = int(version)
            except (TypeError, ValueError):
                return None
            if version < 1:
                return None
            directory = self.summary_version_directory(record)
            if file_format == "json":
                path = os.path.join(directory, f"v{version:04d}.json")
                return path, "application/json", os.path.basename(path)
            md_path = os.path.join(directory, f"v{version:04d}.md")
            if file_format == "docx":
                docx_path = os.path.splitext(md_path)[0] + ".docx"
                MeetingDocConverter.md_file_to_docx(md_path, docx_path)
                return docx_path, DOCX_MIME_TYPE, os.path.basename(docx_path)
            return md_path, "text/markdown; charset=utf-8", os.path.basename(md_path)

    @staticmethod
    def _sort_segments(segments):
        segments.sort(key=lambda item: (str(item.get("session_started_at") or ""), float(item.get("start") or 0)))

    def _write_record(self, record):
        payload = record["payload"]
        self._atomic_write(record["json_path"], json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        self._atomic_write(record["md_path"], self.render_markdown(payload))
        try:
            self.session_file_mtimes[record["json_path"]] = os.path.getmtime(record["json_path"])
        except OSError:
            pass

    @staticmethod
    def render_markdown(payload, layout="sections"):
        lines = [f"# {payload.get('meeting_name') or payload.get('client_name') or 'Meeting Log'}", "", "## 会议信息",
                 f"- Session ID: {payload.get('session_id') or ''}", f"- Client: {payload.get('client_name') or ''}",
                 f"- Started: {payload.get('created_at') or ''}", f"- Updated: {payload.get('updated_at') or ''}",
                 f"- Backend: {payload.get('backend') or ''}", f"- Model: {payload.get('model') or ''}",
                 f"- Source language: {payload.get('source_language') or 'auto'}",
                 f"- Status: {payload.get('status') or ''}",
                 f"- Connections: {payload.get('connection_count') or 1}",
                 f"- Timeline offset seconds: {payload.get('timeline_offset_seconds') or 0}", "", "## 连接中断记录"]
        gaps = payload.get("audio_gaps") or []
        if gaps:
            for gap in gaps:
                lines.append(f"- [{gap.get('start_at', '')} - {gap.get('end_at', '')}] {gap.get('reason') or 'websocket_disconnected'}，该时间段音频未记录")
        else:
            lines.append("- 无")
        speaker_names = {
            item.get("speaker_id"): item.get("name")
            for item in payload.get("speakers") or []
            if isinstance(item, dict)
        }
        if str(layout or "sections").lower() == "interleaved":
            lines.extend(MeetingLogStore._render_interleaved_segments(payload, speaker_names))
            lines.extend(["", "## AI 总结", "", "_待生成_", ""])
            return "\n".join(lines)
        lines.extend(["", "## 原文记录"])
        for segment in payload.get("source_segments", []):
            speaker = speaker_names.get(segment.get("speaker_id")) or segment.get("speaker")
            prefix = f"{speaker}：" if speaker else ""
            lines.append(f"- [{segment.get('start', '')} - {segment.get('end', '')}] {prefix}{segment.get('text', '')}")
        lines.extend(["", "## 翻译记录"])
        for segment in payload.get("translation_segments", []):
            lines.append(f"- [{segment.get('start', '')} - {segment.get('end', '')}] {segment.get('text', '')}")
        lines.extend(["", "## AI 总结", "", "_待生成_", ""])
        return "\n".join(lines)

    @staticmethod
    def _segment_time(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _translation_source_ids(segment):
        ids = segment.get("source_utterance_ids")
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list):
            ids = []
        if segment.get("utterance_id"):
            ids.append(segment.get("utterance_id"))
        return {str(item).strip() for item in ids if str(item or "").strip()}

    @staticmethod
    def _time_overlap(left, right):
        left_start = MeetingLogStore._segment_time(left.get("start"))
        left_end = MeetingLogStore._segment_time(left.get("end"))
        right_start = MeetingLogStore._segment_time(right.get("start"))
        right_end = MeetingLogStore._segment_time(right.get("end"))
        if None in (left_start, left_end, right_start, right_end):
            return 0.0
        return max(0.0, min(left_end, right_end) - max(left_start, right_start))

    @staticmethod
    def _find_translation_for_source(source, translations, used_indexes):
        source_id = str(source.get("utterance_id") or "").strip()
        if source_id:
            for index, translation in enumerate(translations):
                if index in used_indexes:
                    continue
                if source_id in MeetingLogStore._translation_source_ids(translation):
                    return index
        best_index = None
        best_score = 0.0
        source_start = MeetingLogStore._segment_time(source.get("start"))
        for index, translation in enumerate(translations):
            if index in used_indexes:
                continue
            overlap = MeetingLogStore._time_overlap(source, translation)
            if overlap > best_score:
                best_index = index
                best_score = overlap
                continue
            if best_score > 0 or source_start is None:
                continue
            translation_start = MeetingLogStore._segment_time(translation.get("start"))
            if translation_start is None:
                continue
            distance = abs(source_start - translation_start)
            score = max(0.0, 3.0 - distance)
            if score > best_score:
                best_index = index
                best_score = score
        return best_index if best_score > 0 else None

    @staticmethod
    def _render_interleaved_segments(payload, speaker_names):
        lines = ["", "## 中英对照记录"]
        sources = list(payload.get("source_segments") or [])
        translations = list(payload.get("translation_segments") or [])
        used_translation_indexes = set()
        if not sources:
            lines.append("- 无原文记录")
            return lines
        for source in sources:
            speaker = speaker_names.get(source.get("speaker_id")) or source.get("speaker")
            prefix = f"{speaker}：" if speaker else ""
            lines.append(f"- [{source.get('start', '')} - {source.get('end', '')}] 原文：{prefix}{source.get('text', '')}")
            translation_index = MeetingLogStore._find_translation_for_source(source, translations, used_translation_indexes)
            if translation_index is None:
                continue
            used_translation_indexes.add(translation_index)
            translation = translations[translation_index]
            lines.append(f"  - 译文：{translation.get('text', '')}")
        remaining = [
            translation for index, translation in enumerate(translations)
            if index not in used_translation_indexes
        ]
        if remaining:
            lines.extend(["", "## 未匹配翻译记录"])
            for translation in remaining:
                lines.append(f"- [{translation.get('start', '')} - {translation.get('end', '')}] {translation.get('text', '')}")
        return lines

    @staticmethod
    def _custom_literal_text(value):
        items = []
        for raw_line in str(value or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^[-*•·]\s*(.+)$", line)
            literal = match.group(1).strip() if match else line
            if literal[:1] not in ("{", "["):
                return ""
            try:
                parsed = ast.literal_eval(literal)
            except (SyntaxError, ValueError):
                return ""
            text = MeetingLogStore._custom_value_to_text(parsed)
            if text and text not in items:
                items.append(text)
        return "\n".join(items)

    @staticmethod
    def _custom_value_to_text(value):
        if value in (None, ""):
            return ""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            literal_text = MeetingLogStore._custom_literal_text(text)
            if literal_text:
                return literal_text
            if text[:1] in ("{", "["):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError):
                    return text
                parsed_text = MeetingLogStore._custom_value_to_text(parsed)
                return parsed_text or text
            return text
        if isinstance(value, (int, float, bool)):
            return str(value).strip()
        if isinstance(value, list):
            items = []
            for item in value:
                text = MeetingLogStore._custom_value_to_text(item)
                if text and text not in items:
                    items.append(text)
            return "\n".join(items)
        if isinstance(value, dict):
            for title_key in ("title", "topic", "name"):
                title = MeetingLogStore._custom_value_to_text(value.get(title_key)) if title_key in value else ""
                if title:
                    details = []
                    for detail_key in ("point", "content", "summary", "text", "value", "内容"):
                        if detail_key in value:
                            detail = MeetingLogStore._custom_value_to_text(value.get(detail_key))
                            if detail and detail != title:
                                details.append(detail)
                    return f"{title}：{'；'.join(details)}" if details else title
            for key in ("point", "text", "content", "summary", "value", "内容"):
                if key in value:
                    text = MeetingLogStore._custom_value_to_text(value.get(key))
                    if text:
                        return text
            parts = []
            for item_key, item_value in value.items():
                text = MeetingLogStore._custom_value_to_text(item_value)
                if text:
                    parts.append(f"{item_key}：{text}")
            return "；".join(parts)
        return str(value).strip()

    @staticmethod
    def _render_custom_field(field, value):
        field_type = field.get("type")
        if field_type == "text":
            return MeetingLogStore._custom_value_to_text(value)
        if field_type == "table":
            columns = field.get("columns") or ["内容"]
            rows = value if isinstance(value, list) else []
            if not rows:
                return ""
            lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
            for row in rows:
                row = row if isinstance(row, dict) else {}
                lines.append("| " + " | ".join(MeetingLogStore._custom_value_to_text(row.get(column)).replace("|", "\\|").replace("\n", " ") for column in columns) + " |")
            return "\n".join(lines)
        lines = []
        for item in value if isinstance(value, list) else []:
            if isinstance(item, dict):
                body = MeetingLogStore._custom_value_to_text(item)
                if not body:
                    continue
                lines.append(f"- {body}")
                if item.get("evidence_quote"):
                    quote = MeetingLogStore._custom_value_to_text(item.get("evidence_quote"))
                    lines.append(f"  - 依据 [{item.get('evidence_start', '')} - {item.get('evidence_end', '')}]：{quote}")
            else:
                body = MeetingLogStore._custom_value_to_text(item)
                if body:
                    lines.append(f"- {body}")
        return "\n".join(lines)

    @staticmethod
    def render_custom_summary_markdown(summary):
        markdown = str(summary.get("custom_template_markdown") or "").rstrip()
        fields = summary.get("custom_template_fields") or []
        data = summary.get("template_data") or {}
        residual_errors = {
            str(item.get("key") or ""): item
            for item in (summary.get("summary_quality") or {}).get("residual_generation_errors") or []
            if isinstance(item, dict) and item.get("key")
        }

        def render_field(field):
            key = str(field.get("key") or "")
            rendered = MeetingLogStore._render_custom_field(field, data.get(key))
            if not rendered and key in residual_errors:
                return "（该专题生成失败，请重新生成总结）"
            return rendered

        markdown = markdown.replace("{{meeting_name}}", str(summary.get("meeting_name") or "会议总结"))
        for field in fields:
            markdown = markdown.replace("{{" + str(field.get("key") or "") + "}}", render_field(field))

        lines = markdown.splitlines()
        replacements = {str(field.get("heading") or ""): render_field(field) for field in fields}
        output, index = [], 0
        while index < len(lines):
            match = re.match(r"^(#{2,6})\s+(.+?)\s*$", lines[index])
            if not match or match.group(2).strip() not in replacements:
                output.append(lines[index])
                index += 1
                continue
            heading = match.group(2).strip()
            output.append(lines[index])
            rendered = replacements[heading]
            if rendered:
                output.extend(["", rendered])
            index += 1
            while index < len(lines):
                next_heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", lines[index])
                if next_heading:
                    break
                index += 1
        output.extend([
            "", "## 生成信息",
            f"- Session ID: {summary.get('session_id') or ''}",
            f"- 模板: {summary.get('custom_template_name') or summary.get('custom_template_id') or ''}",
            f"- 模板版本: {summary.get('custom_template_revision') or 1}",
            f"- 生成时间: {summary.get('generated_at') or ''}",
            f"- 模型: {summary.get('model') or ''}",
            "",
        ])
        return "\n".join(output)

    @staticmethod
    def render_summary_markdown(summary):
        if summary.get("summary_template") == "custom":
            return MeetingLogStore.render_custom_summary_markdown(summary)

        def evidence_suffix(item):
            if not isinstance(item, dict) or not item.get("evidence_quote"):
                return ""
            return f"\n  - 依据 [{item.get('evidence_start', '')} - {item.get('evidence_end', '')}]：{item.get('evidence_quote')}"

        def append_text(lines, title, value):
            value = str(value or "").strip()
            if value:
                lines.extend(["", f"## {title}", value])

        def append_list(lines, title, values):
            values = values or []
            if not values:
                return
            lines.extend(["", f"## {title}"])
            for item in values:
                body = MeetingLogStore._custom_value_to_text(item)
                if body:
                    lines.append(f"- {body}")

        def append_evidence(lines, title, values, text_key="text", formatter=None):
            values = values or []
            if not values:
                return
            lines.extend(["", f"## {title}"])
            for item in values:
                if not isinstance(item, dict):
                    lines.append(f"- {item}")
                    continue
                body = formatter(item) if formatter else item.get(text_key) or ""
                lines.append(f"- {body}{evidence_suffix(item)}")

        template = summary.get("summary_template") or "auto"
        data = summary.get("template_data") or {}
        if not data:
            data = {
                "topics": summary.get("topics") or [],
                "decisions": summary.get("decisions") or [],
                "action_items": summary.get("action_items") or [],
                "risks": summary.get("risks") or [],
                "follow_ups": summary.get("follow_ups") or [],
                "open_questions": summary.get("open_questions") or [],
            }

        lines = [
            f"# {summary.get('meeting_name') or '会议总结'}", "", "## 会议信息",
            f"- Session ID: {summary.get('session_id') or ''}",
            f"- Version: {summary.get('version') or ''}",
            f"- Template: {template}",
            f"- Meeting type: {summary.get('meeting_type') or ''}",
            f"- Generated: {summary.get('generated_at') or ''}",
            f"- Model: {summary.get('model') or ''}",
        ]
        append_text(lines, "内容概述", summary.get("overview") or "未明确")

        action_formatter = lambda item: (
            f"{item.get('task') or '未明确'}（负责人：{item.get('owner') or '未明确'}；"
            f"时间：{item.get('deadline') or '未明确'}；状态：{item.get('status') or '未明确'}）"
        )

        if template == "project_meeting":
            append_text(lines, "项目状态", data.get("project_status"))
            append_evidence(lines, "项目进展", data.get("progress"))
            append_evidence(lines, "关键决策", data.get("decisions"))
            append_evidence(lines, "待办事项", data.get("action_items"), text_key="task", formatter=action_formatter)
            append_evidence(lines, "阻塞事项", data.get("blockers"))
            append_list(lines, "风险", data.get("risks"))
            append_list(lines, "未决问题", data.get("open_questions"))
            append_list(lines, "下一步", data.get("next_steps"))
        elif template == "customer_interview":
            append_list(lines, "客户画像", data.get("customer_profile"))
            append_evidence(lines, "客户需求", data.get("needs"))
            append_evidence(lines, "客户痛点", data.get("pain_points"))
            append_evidence(lines, "使用场景", data.get("use_cases"))
            append_evidence(lines, "产品反馈", data.get("feedback"))
            append_evidence(lines, "顾虑与异议", data.get("objections"))
            append_evidence(lines, "约定事项", data.get("action_items"), text_key="task", formatter=action_formatter)
            append_list(lines, "待确认问题", data.get("open_questions"))
            append_list(lines, "后续跟进", data.get("follow_ups"))
        elif template == "training_speech":
            append_text(lines, "核心主旨", data.get("thesis"))
            append_list(lines, "内容结构", data.get("outline"))
            append_evidence(lines, "核心观点", data.get("key_points"))
            append_list(lines, "论证逻辑", data.get("arguments"))
            append_evidence(lines, "案例与研究", data.get("cases"))
            append_evidence(lines, "关键数据", data.get("data_points"))
            append_evidence(lines, "重要原话", data.get("notable_quotes"))
            append_list(lines, "核心启示", data.get("takeaways"))
            append_evidence(lines, "内容时间线", data.get("timeline"))
            append_list(lines, "疑似识别问题", data.get("asr_uncertainties"))
        elif template == "discussion":
            append_list(lines, "讨论议题", data.get("discussion_topics"))
            append_evidence(
                lines,
                "主要观点",
                data.get("viewpoints"),
                formatter=lambda item: f"{item.get('speaker') or '发言人未明确'}：{item.get('text') or ''}",
            )
            append_evidence(lines, "已达成共识", data.get("consensus"))
            append_evidence(lines, "主要分歧", data.get("disagreements"))
            append_evidence(lines, "关键决策", data.get("decisions"))
            append_evidence(lines, "待办事项", data.get("action_items"), text_key="task", formatter=action_formatter)
            append_list(lines, "未决问题", data.get("open_questions"))
            append_list(lines, "后续跟进", data.get("follow_ups"))
        else:
            append_list(lines, "核心议题", data.get("topics"))
            append_evidence(lines, "关键观点", data.get("key_points"))
            append_evidence(lines, "关键决策", data.get("decisions"))
            append_evidence(lines, "待办事项", data.get("action_items"), text_key="task", formatter=action_formatter)
            append_list(lines, "风险与问题", data.get("risks"))
            append_list(lines, "未决问题", data.get("open_questions"))
            append_list(lines, "后续跟进", data.get("follow_ups"))

        quality = summary.get("summary_quality") or {}
        lines.extend([
            "", "## 证据校验",
            f"- 原文片段数：{quality.get('source_segment_count', 0)}",
            f"- 已验证带证据内容数：{quality.get('evidence_count', 0)}",
            f"- 已过滤无依据内容数：{quality.get('filtered_unverified_count', 0)}",
            "",
        ])
        return "\n".join(lines)
