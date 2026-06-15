import json
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.meeting import MeetingLogStore


class TestMeetingLogStore(unittest.TestCase):
    def test_save_meeting_log_writes_json_to_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            result = store.save({"meeting_name": "产品周会", "source_segments": [{"text": "你好"}]})

            self.assertTrue(result["saved"])
            self.assertTrue(result["filename"].startswith("产品周会-"))
            self.assertTrue(result["filename"].endswith(".json"))
            self.assertTrue(os.path.isfile(result["path"]))
            with open(result["path"], "r", encoding="utf-8") as file:
                data = json.load(file)
            self.assertEqual(data["source_segments"][0]["text"], "你好")

    def test_save_meeting_log_sanitizes_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            result = store.save({"meeting_name": "a/b:c*?", "source_segments": []})

            self.assertNotIn("/", result["filename"])
            self.assertEqual(os.path.basename(result["path"]), result["filename"])

    def test_save_meeting_log_does_not_overwrite_same_second_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            with mock.patch.object(MeetingLogStore, "timestamp_for_filename", return_value="2026-05-28T10-30-15"):
                first = store.save({"meeting_name": "会议", "source_segments": [{"text": "一"}]})
                second = store.save({"meeting_name": "会议", "source_segments": [{"text": "二"}]})

            self.assertNotEqual(first["filename"], second["filename"])
            self.assertTrue(os.path.isfile(first["path"]))
            self.assertTrue(os.path.isfile(second["path"]))

    def test_save_meeting_log_requires_object(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            with self.assertRaises(ValueError):
                store.save([{"meeting_name": "会议"}])

    def test_session_log_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-1", "meeting_name": "会议"})
            store.append_segments("session-1", "source", [{"start": "0", "end": "1", "text": "hello", "completed": True}])
            info = store.finish_session("session-1")
            self.assertEqual(info["source_count"], 1)
            self.assertTrue(os.path.isfile(info["json_path"]))
            self.assertTrue(os.path.isfile(info["md_path"]))

    def test_write_summary_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-1", "meeting_name": "会议"})
            store.finish_session("session-1")
            summary = {
                "session_id": "session-1", "meeting_name": "会议", "generated_at": "2026-06-08T12:00:00", "model": "qwen3-8b-awq",
                "overview": "讨论了项目进度。", "topics": ["项目进度"], "decisions": ["继续推进"],
                "action_items": [{"task": "整理纪要", "owner": "未明确", "deadline": "未明确", "status": "未明确"}],
                "risks": ["时间紧"], "follow_ups": ["下次复盘"],
            }
            info = store.write_summary("session-1", summary)
            self.assertTrue(info["has_summary"])
            self.assertEqual(info["latest_version"], 1)
            self.assertEqual(len(info["versions"]), 1)
            self.assertTrue(os.path.isfile(info["json_path"]))
            self.assertTrue(os.path.isfile(info["md_path"]))

            second = dict(summary, generated_at="2026-06-08T12:05:00", summary_template="discussion")
            info = store.write_summary("session-1", second)
            self.assertEqual(info["latest_version"], 2)
            self.assertEqual(len(info["versions"]), 2)
            version_file = store.get_summary_file("session-1", "json", version=1)
            self.assertTrue(os.path.isfile(version_file[0]))

            restored = MeetingLogStore(directory)
            restored_info = restored.summary_info("session-1")
            self.assertEqual(restored_info["latest_version"], 2)
            self.assertEqual(len(restored.list_sessions()["sessions"]), 1)

