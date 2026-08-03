import json
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.meeting import MeetingLogStore
from whisper_live.meeting.sessions import apply_timeline_offset_to_segments
from whisper_live.meeting.docs import DOCX_MIME_TYPE


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

    def test_session_log_preserves_translation_warning_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-warning", "meeting_name": "会议"})
            store.append_segments("session-warning", "source", [
                {
                    "start": "0.000",
                    "end": "1.000",
                    "text": "你好世界",
                    "completed": True,
                    "utterance_id": "uid-1:1:0.000",
                },
            ])
            store.append_segments("session-warning", "translation", [
                {
                    "start": "0.000",
                    "end": "1.000",
                    "text": "翻译暂不可用",
                    "completed": True,
                    "source_text": "你好世界",
                    "translation_warning": "source_echo",
                    "source_utterance_ids": ["uid-1:1:0.000"],
                    "utterance_id": "uid-1:1:0.000",
                },
            ])

            info = store.finish_session("session-warning")

            with open(info["json_path"], "r", encoding="utf-8") as file:
                data = json.load(file)
            translation = data["translation_segments"][0]
            self.assertEqual(translation["text"], "翻译暂不可用")
            self.assertEqual(translation["source_text"], "你好世界")
            self.assertEqual(translation["translation_warning"], "source_echo")
            self.assertEqual(translation["source_utterance_ids"], ["uid-1:1:0.000"])

            with open(info["md_path"], "r", encoding="utf-8") as file:
                markdown = file.read()
            self.assertIn("翻译暂不可用", markdown)
            self.assertNotIn("source_echo", markdown)

    def test_session_log_preserves_translation_confidence_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-low", "meeting_name": "会议"})
            store.append_segments("session-low", "source", [
                {
                    "start": "0.000",
                    "end": "1.000",
                    "text": "BHP shipped 12 tons of ore.",
                    "completed": True,
                    "utterance_id": "uid-1:1:0.000",
                },
            ])
            store.append_segments("session-low", "translation", [
                {
                    "start": "0.000",
                    "end": "1.000",
                    "text": "必和必拓运送了矿石。",
                    "completed": True,
                    "source_text": "BHP shipped 12 tons of ore.",
                    "translation_confidence": "low",
                    "utterance_id": "uid-1:1:0.000",
                },
            ])

            info = store.finish_session("session-low")

            with open(info["json_path"], "r", encoding="utf-8") as file:
                data = json.load(file)
            translation = data["translation_segments"][0]
            self.assertEqual(translation["translation_confidence"], "low")
            self.assertNotIn("translation_warning", translation)

    def test_finished_transcript_can_be_corrected_with_revision_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-edit", "meeting_name": "会议"})
            store.append_segments("session-edit", "source", [
                {"start": "0.000", "end": "1.000", "text": "原始文本", "completed": True},
            ])
            store.finish_session("session-edit")

            transcript = store.get_transcript("session-edit")
            segment_id = transcript["segments"][0]["segment_id"]
            updated = store.update_transcript_segment(
                "session-edit", segment_id, "校对文本", None, transcript["transcript_revision"]
            )

            self.assertEqual(updated["transcript_revision"], 1)
            self.assertEqual(updated["segments"][0]["original_text"], "原始文本")
            self.assertEqual(updated["segments"][0]["text"], "校对文本")
            self.assertTrue(updated["translation_stale"])
            self.assertTrue(updated["summary_stale"])
            with self.assertRaises(ValueError):
                store.update_transcript_segment("session-edit", segment_id, "冲突文本", None, 0)

            restored = MeetingLogStore(directory)
            restored_transcript = restored.get_transcript("session-edit")
            self.assertEqual(restored_transcript["segments"][0]["segment_id"], segment_id)
            self.assertEqual(restored_transcript["segments"][0]["text"], "校对文本")

    def test_manual_speaker_management_updates_segments_and_summary_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-speakers", "meeting_name": "会议"})
            store.append_segments("session-speakers", "source", [
                {"start": "0.000", "end": "1.000", "text": "第一段", "completed": True},
            ])
            store.finish_session("session-speakers")
            transcript = store.get_transcript("session-speakers")
            alice = store.add_transcript_speaker("session-speakers", "张三", transcript["transcript_revision"])
            alice_id = alice["speakers"][0]["speaker_id"]
            bob = store.add_transcript_speaker("session-speakers", "李四", alice["transcript_revision"])
            bob_id = next(item["speaker_id"] for item in bob["speakers"] if item["name"] == "李四")
            assigned = store.update_transcript_segment(
                "session-speakers",
                bob["segments"][0]["segment_id"],
                bob["segments"][0]["text"],
                alice_id,
                bob["transcript_revision"],
            )
            renamed = store.rename_transcript_speaker(
                "session-speakers", alice_id, "张经理", assigned["transcript_revision"]
            )
            merged = store.merge_transcript_speakers(
                "session-speakers", alice_id, bob_id, renamed["transcript_revision"]
            )

            self.assertEqual(len(merged["speakers"]), 1)
            self.assertEqual(merged["segments"][0]["speaker_id"], bob_id)
            self.assertFalse(merged["translation_stale"])
            self.assertTrue(merged["summary_stale"])

            info = store.write_summary("session-speakers", {
                "session_id": "session-speakers",
                "meeting_name": "会议",
                "generated_at": "2026-06-24T10:00:00",
                "overview": "总结",
            })
            self.assertFalse(info["summary_stale"])
            self.assertEqual(info["versions"][0]["transcript_revision"], merged["transcript_revision"])

    def test_session_log_docx_export_uses_markdown_converter(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-docx", "meeting_name": "会议"})
            store.append_segments("session-docx", "source", [
                {"start": "0.000", "end": "1.000", "text": "你好", "completed": True},
            ])
            store.finish_session("session-docx")

            def fake_docx(md_path, docx_path):
                with open(docx_path, "wb") as file:
                    file.write(b"docx")

            with mock.patch("whisper_live.meeting.logs.MeetingDocConverter.md_file_to_docx", side_effect=fake_docx) as converter:
                result = store.get_session_file("session-docx", "docx")

            self.assertEqual(result[1], DOCX_MIME_TYPE)
            self.assertTrue(result[0].endswith(".docx"))
            self.assertTrue(os.path.isfile(result[0]))
            converter.assert_called_once()

    def test_interleaved_session_log_pairs_source_and_translation(self):
        payload = {
            "meeting_name": "会议",
            "session_id": "session-interleaved",
            "source_segments": [
                {"start": "0.000", "end": "2.000", "text": "你好", "utterance_id": "u1"},
                {"start": "2.000", "end": "4.000", "text": "再见", "utterance_id": "u2"},
            ],
            "translation_segments": [
                {"start": "0.000", "end": "2.000", "text": "Hello", "source_utterance_ids": ["u1"]},
                {"start": "2.000", "end": "4.000", "text": "Goodbye", "source_utterance_ids": ["u2"]},
            ],
        }

        markdown = MeetingLogStore.render_markdown(payload, layout="interleaved")

        self.assertIn("## 中英对照记录", markdown)
        self.assertIn("原文：你好", markdown)
        self.assertIn("译文：Hello", markdown)
        self.assertIn("原文：再见", markdown)
        self.assertIn("译文：Goodbye", markdown)
        self.assertNotIn("## 原文记录", markdown)

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

            def fake_docx(md_path, docx_path):
                with open(docx_path, "wb") as file:
                    file.write(b"docx")

            with mock.patch("whisper_live.meeting.logs.MeetingDocConverter.md_file_to_docx", side_effect=fake_docx) as converter:
                latest_docx = store.get_summary_file("session-1", "docx")
                self.assertEqual(latest_docx[1], DOCX_MIME_TYPE)
                self.assertTrue(latest_docx[0].endswith("-summary.docx"))
                self.assertTrue(os.path.isfile(latest_docx[0]))
                version_docx = store.get_summary_file("session-1", "docx", version=1)
                self.assertEqual(os.path.basename(version_docx[0]), "v0001.docx")
                self.assertTrue(os.path.isfile(version_docx[0]))
                self.assertEqual(converter.call_count, 2)

            restored = MeetingLogStore(directory)
            restored_info = restored.summary_info("session-1")
            self.assertEqual(restored_info["latest_version"], 2)
            self.assertEqual(len(restored.list_sessions()["sessions"]), 1)

    def test_shared_directory_discovers_external_session_without_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            admin_store = MeetingLogStore(directory, refresh_interval_seconds=0)
            worker_store = MeetingLogStore(directory, refresh_interval_seconds=0)

            worker_store.start_session({"uid": "uid-1", "session_id": "gpu1-session", "meeting_name": "GPU1会议"})
            worker_store.append_segments("gpu1-session", "source", [
                {"start": "0.000", "end": "1.000", "text": "来自GPU1的日志", "completed": True},
            ])
            worker_store.finish_session("gpu1-session")

            sessions = admin_store.list_sessions()["sessions"]
            self.assertEqual([item["session_id"] for item in sessions], ["gpu1-session"])
            self.assertEqual(sessions[0]["status"], "finished")
            self.assertEqual(sessions[0]["source_count"], 1)

            log_file = admin_store.get_session_file("gpu1-session", "md")
            self.assertIsNotNone(log_file)
            with open(log_file[0], "r", encoding="utf-8") as file:
                self.assertIn("来自GPU1的日志", file.read())

    def test_shared_directory_external_session_supports_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            admin_store = MeetingLogStore(directory, refresh_interval_seconds=0)
            worker_store = MeetingLogStore(directory, refresh_interval_seconds=0)

            worker_store.start_session({"uid": "uid-1", "session_id": "gpu1-summary", "meeting_name": "GPU1会议"})
            worker_store.append_segments("gpu1-summary", "source", [
                {"start": "0.000", "end": "1.000", "text": "需要总结的内容", "completed": True},
            ])
            worker_store.finish_session("gpu1-summary")

            info = admin_store.write_summary("gpu1-summary", {
                "session_id": "gpu1-summary",
                "meeting_name": "GPU1会议",
                "generated_at": "2026-07-28T10:00:00",
                "overview": "总结内容",
                "topics": ["跨卡日志"],
                "decisions": [],
                "action_items": [],
                "risks": [],
                "follow_ups": [],
            })

            self.assertTrue(info["has_summary"])
            self.assertEqual(info["latest_version"], 1)
            summary_file = admin_store.get_summary_file("gpu1-summary", "md")
            self.assertIsNotNone(summary_file)
            with open(summary_file[0], "r", encoding="utf-8") as file:
                self.assertIn("总结内容", file.read())

    def test_session_can_be_interrupted_and_resumed_without_overwriting_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({
                "uid": "uid-1",
                "session_id": "session-1",
                "client_instance_id": "browser-1",
                "meeting_name": "会议",
                "session_started_at": "2026-06-18T10:00:00+00:00",
            })
            store.append_segments("session-1", "source", [{"start": "0.000", "end": "1.000", "text": "第一段", "completed": True}])

            with mock.patch.object(MeetingLogStore, "now_iso", return_value="2026-06-18T10:01:00+00:00"):
                interrupted = store.interrupt_session("session-1")
            self.assertEqual(interrupted["status"], "interrupted")

            with mock.patch.object(MeetingLogStore, "now_iso", return_value="2026-06-18T10:01:10+00:00"):
                resumed = store.resume_session({
                    "session_id": "session-1",
                    "client_instance_id": "browser-1",
                    "meeting_name": "会议",
                })
            self.assertEqual(resumed["status"], "active")
            self.assertEqual(resumed["source_count"], 1)
            self.assertEqual(resumed["connection_count"], 2)
            self.assertEqual(resumed["timeline_offset_seconds"], 70.0)
            self.assertEqual(len(resumed["audio_gaps"]), 1)

            store.append_segments("session-1", "source", [{"start": "70.000", "end": "71.000", "text": "第二段", "completed": True}])
            payload = store.get_session_payload("session-1")
            self.assertEqual([item["text"] for item in payload["source_segments"]], ["第一段", "第二段"])

    def test_finished_session_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-1", "client_instance_id": "browser-1"})
            store.finish_session("session-1")
            with self.assertRaises(ValueError):
                store.resume_session({"session_id": "session-1", "client_instance_id": "browser-1"})

    def test_stale_generation_cleanup_does_not_interrupt_resumed_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            started = store.start_session({"uid": "uid-1", "session_id": "session-1", "client_instance_id": "browser-1"})
            self.assertEqual(started["connection_generation"], 1)
            store.interrupt_session("session-1", expected_generation=1)
            resumed = store.resume_session({"session_id": "session-1", "client_instance_id": "browser-1"})

            stale_result = store.interrupt_session("session-1", expected_generation=1)

            self.assertEqual(resumed["connection_generation"], 2)
            self.assertEqual(stale_result["status"], "active")
            self.assertEqual(store.session_info("session-1")["status"], "active")

    def test_matching_generation_cleanup_interrupts_active_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            started = store.start_session({"uid": "uid-1", "session_id": "session-1"})

            interrupted = store.interrupt_session(
                "session-1", expected_generation=started["connection_generation"],
            )

            self.assertEqual(interrupted["status"], "interrupted")

    def test_resume_rejects_different_client_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MeetingLogStore(directory)
            store.start_session({"uid": "uid-1", "session_id": "session-1", "client_instance_id": "browser-1"})
            store.interrupt_session("session-1")
            with self.assertRaises(ValueError):
                store.resume_session({"session_id": "session-1", "client_instance_id": "browser-2"})

    def test_timeline_offset_applies_to_segment_and_words(self):
        segments = [{
            "start": "1.000",
            "end": "2.000",
            "text": "hello",
            "completed": True,
            "words": [{"word": "hello", "start": 1.1, "end": 1.5}],
        }]
        adjusted = apply_timeline_offset_to_segments(segments, 10.0)
        self.assertEqual(adjusted[0]["start"], "11.000")
        self.assertEqual(adjusted[0]["end"], "12.000")
        self.assertEqual(adjusted[0]["words"][0]["start"], 11.1)
        self.assertEqual(segments[0]["start"], "1.000")
