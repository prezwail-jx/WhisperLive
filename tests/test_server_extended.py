import json
import os
import time
import threading
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.server import TranscriptionServer, BackendType, ClientManager, MeetingHotwordStore, MeetingLogStore, MeetingSummaryService, SummaryGenerationError, count_hotwords


class TestClientManagerAddRemove(unittest.TestCase):
    def setUp(self):
        self.cm = ClientManager(max_clients=2, max_connection_time=60)

    def test_add_and_get_client(self):
        ws = MagicMock()
        client = MagicMock()
        self.cm.add_client(ws, client)
        self.assertIs(self.cm.get_client(ws), client)

    def test_get_nonexistent_client(self):
        ws = MagicMock()
        self.assertFalse(self.cm.get_client(ws))

    def test_remove_client_calls_cleanup(self):
        ws = MagicMock()
        client = MagicMock()
        self.cm.add_client(ws, client)
        self.cm.remove_client(ws)
        client.cleanup.assert_called_once()
        self.assertNotIn(ws, self.cm.clients)
        self.assertNotIn(ws, self.cm.start_times)

    def test_remove_nonexistent_client_no_error(self):
        ws = MagicMock()
        self.cm.remove_client(ws)  # should not raise


class TestClientManagerThreadSafety(unittest.TestCase):
    def test_concurrent_add_remove(self):
        cm = ClientManager(max_clients=100, max_connection_time=600)
        errors = []

        def add_clients(start_idx):
            try:
                for i in range(50):
                    ws = MagicMock(name=f"ws-{start_idx}-{i}")
                    client = MagicMock(name=f"client-{start_idx}-{i}")
                    cm.add_client(ws, client)
            except Exception as e:
                errors.append(e)

        def remove_clients():
            try:
                for _ in range(25):
                    with cm.lock:
                        if cm.clients:
                            ws = next(iter(cm.clients))
                        else:
                            continue
                    cm.remove_client(ws)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_clients, args=(0,)),
            threading.Thread(target=add_clients, args=(1,)),
            threading.Thread(target=remove_clients),
            threading.Thread(target=remove_clients),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

    def test_concurrent_get_client(self):
        cm = ClientManager(max_clients=100, max_connection_time=600)
        ws = MagicMock()
        client = MagicMock()
        cm.add_client(ws, client)
        errors = []
        results = []

        def get_many():
            try:
                for _ in range(100):
                    results.append(cm.get_client(ws))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertTrue(all(r is client for r in results))


class TestClientManagerServerFull(unittest.TestCase):
    def setUp(self):
        self.cm = ClientManager(max_clients=1, max_connection_time=60)

    def test_not_full_returns_false(self):
        ws = MagicMock()
        options = {"uid": "test"}
        self.assertFalse(self.cm.is_server_full(ws, options))

    def test_full_sends_wait_and_returns_true(self):
        ws1 = MagicMock()
        self.cm.add_client(ws1, MagicMock())

        ws2 = MagicMock()
        options = {"uid": "new-client"}
        self.assertTrue(self.cm.is_server_full(ws2, options))
        ws2.send.assert_called_once()
        sent = json.loads(ws2.send.call_args[0][0])
        self.assertEqual(sent["status"], "WAIT")
        self.assertEqual(sent["uid"], "new-client")


class TestClientManagerTimeout(unittest.TestCase):
    def setUp(self):
        self.cm = ClientManager(max_clients=4, max_connection_time=10)

    def test_not_timed_out(self):
        ws = MagicMock()
        client = MagicMock()
        self.cm.add_client(ws, client)
        self.assertFalse(self.cm.is_client_timeout(ws))

    def test_timed_out(self):
        ws = MagicMock()
        client = MagicMock()
        self.cm.add_client(ws, client)
        self.cm.start_times[ws] = time.time() - 20
        self.assertTrue(self.cm.is_client_timeout(ws))
        client.disconnect.assert_called_once()


class TestClientManagerGetWaitTime(unittest.TestCase):
    def test_no_clients_returns_zero(self):
        cm = ClientManager(max_clients=4, max_connection_time=600)
        self.assertEqual(cm.get_wait_time(), 0)

    def test_single_client_wait_time(self):
        cm = ClientManager(max_clients=4, max_connection_time=600)
        ws = MagicMock()
        cm.add_client(ws, MagicMock())
        cm.start_times[ws] = time.time() - 300
        wait = cm.get_wait_time()
        self.assertAlmostEqual(wait, 5.0, places=0)

    def test_multiple_clients_returns_minimum(self):
        cm = ClientManager(max_clients=4, max_connection_time=600)
        ws1, ws2 = MagicMock(), MagicMock()
        cm.add_client(ws1, MagicMock())
        cm.add_client(ws2, MagicMock())
        cm.start_times[ws1] = time.time() - 100
        cm.start_times[ws2] = time.time() - 500
        wait = cm.get_wait_time()
        # ws2 has 100s remaining = ~1.67 minutes
        self.assertAlmostEqual(wait, 100 / 60, places=0)


class TestBackendType(unittest.TestCase):
    def test_valid_types(self):
        valid = BackendType.valid_types()
        self.assertIn("faster_whisper", valid)
        self.assertIn("tensorrt", valid)
        self.assertIn("openvino", valid)
        self.assertIn("mlx_whisper", valid)

    def test_is_valid(self):
        self.assertTrue(BackendType.is_valid("faster_whisper"))
        self.assertFalse(BackendType.is_valid("nonexistent"))

    def test_type_checks(self):
        self.assertTrue(BackendType.FASTER_WHISPER.is_faster_whisper())
        self.assertFalse(BackendType.FASTER_WHISPER.is_tensorrt())
        self.assertTrue(BackendType.TENSORRT.is_tensorrt())
        self.assertTrue(BackendType.OPENVINO.is_openvino())
        self.assertTrue(BackendType.MLX_WHISPER.is_mlx_whisper())

    def test_enum_from_string(self):
        bt = BackendType("faster_whisper")
        self.assertEqual(bt, BackendType.FASTER_WHISPER)

    def test_invalid_enum_raises(self):
        with self.assertRaises(ValueError):
            BackendType("invalid_backend")


class TestTranscriptionServerInit(unittest.TestCase):
    def test_defaults(self):
        server = TranscriptionServer()
        self.assertIsNone(server.client_manager)
        self.assertTrue(server.use_vad)
        self.assertFalse(server.single_model)
        self.assertIsNone(server.batch_config)

    def test_run_invalid_backend_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, backend="nonexistent")

    def test_run_invalid_trt_path_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(
                host="localhost",
                port=9090,
                backend="tensorrt",
                whisper_tensorrt_path="/nonexistent/path",
            )

    def test_run_max_clients_zero_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, max_clients=0)

    def test_run_max_clients_negative_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, max_clients=-1)

    def test_run_max_connection_time_zero_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, max_connection_time=0)

    def test_run_batch_max_size_zero_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, batch_enabled=True, batch_max_size=0)

    def test_run_batch_window_ms_negative_raises(self):
        server = TranscriptionServer()
        with self.assertRaises(ValueError):
            server.run(host="localhost", port=9090, batch_enabled=True, batch_window_ms=-1)


class TestTranscriptionServerGetAudio(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()

    def test_end_of_audio_returns_false(self):
        ws = MagicMock()
        ws.recv.return_value = b"END_OF_AUDIO"
        result = self.server.get_audio_from_websocket(ws)
        self.assertFalse(result)

    def test_valid_audio_returns_numpy(self):
        import numpy as np
        ws = MagicMock()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        ws.recv.return_value = audio.tobytes()
        result = self.server.get_audio_from_websocket(ws)
        np.testing.assert_array_almost_equal(result, audio)

    def test_raw_pcm_input_normalizes_int16(self):
        import numpy as np
        self.server.raw_pcm_input = True
        ws = MagicMock()
        pcm = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        ws.recv.return_value = pcm.tobytes()
        result = self.server.get_audio_from_websocket(ws)
        expected = pcm.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(result, expected)
        self.assertTrue(result.dtype == np.float32)
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_raw_pcm_input_off_reads_float32(self):
        import numpy as np
        self.server.raw_pcm_input = False
        ws = MagicMock()
        audio = np.array([0.5, -0.5], dtype=np.float32)
        ws.recv.return_value = audio.tobytes()
        result = self.server.get_audio_from_websocket(ws)
        np.testing.assert_array_almost_equal(result, audio)


class TestTranscriptionServerHandleNewConnection(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()
        self.server.client_manager = ClientManager(max_clients=4, max_connection_time=600)
        self.server.cache_path = "~/.cache/whisper-live/"
        self.server.backend = BackendType.FASTER_WHISPER

    @mock.patch("websockets.WebSocketCommonProtocol")
    def test_invalid_json_returns_false(self, mock_ws):
        mock_ws.recv.return_value = "not valid json {{"
        result = self.server.handle_new_connection(mock_ws, None, None, False)
        self.assertFalse(result)

    @mock.patch("websockets.WebSocketCommonProtocol")
    def test_server_full_returns_false(self, mock_ws):
        # Fill server
        for i in range(4):
            self.server.client_manager.add_client(MagicMock(), MagicMock())

        mock_ws.recv.return_value = json.dumps({
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "tiny.en",
        })
        result = self.server.handle_new_connection(mock_ws, None, None, False)
        self.assertFalse(result)

    @mock.patch("whisper_live.server.threading.Thread")
    @mock.patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    @mock.patch("whisper_live.backend.translation_backend.ServeClientTranslation")
    def test_initialize_client_passes_server_translation_device(
        self, mock_translation_client, mock_faster_client, mock_thread
    ):
        mock_faster_client.return_value = MagicMock()
        mock_translation_client.return_value = MagicMock()
        self.server.translation_device = "cpu"
        options = {
            "uid": "test",
            "language": "zh",
            "task": "transcribe",
            "model": "small",
            "enable_translation": True,
        }

        self.server.initialize_client(MagicMock(), options, None, None, False)

        self.assertEqual(mock_translation_client.call_args.kwargs["translation_device"], "cpu")

    @mock.patch("whisper_live.server.threading.Thread")
    @mock.patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    @mock.patch("whisper_live.backend.translation_backend.ServeClientTranslation")
    def test_initialize_client_client_translation_device_overrides_server_default(
        self, mock_translation_client, mock_faster_client, mock_thread
    ):
        mock_faster_client.return_value = MagicMock()
        mock_translation_client.return_value = MagicMock()
        self.server.translation_device = "cpu"
        options = {
            "uid": "test",
            "language": "zh",
            "task": "transcribe",
            "model": "small",
            "enable_translation": True,
            "translation_device": "cuda",
        }

        self.server.initialize_client(MagicMock(), options, None, None, False)

        self.assertEqual(mock_translation_client.call_args.kwargs["translation_device"], "cuda")


class TestTranscriptionServerCleanup(unittest.TestCase):
    def setUp(self):
        self.server = TranscriptionServer()
        self.server.client_manager = ClientManager(max_clients=4, max_connection_time=600)

    def test_cleanup_removes_client(self):
        ws = MagicMock()
        client = MagicMock()
        self.server.client_manager.add_client(ws, client)
        self.server.cleanup(ws)
        self.assertNotIn(ws, self.server.client_manager.clients)
        client.cleanup.assert_called_once()


class TestMeetingHotwordStore(unittest.TestCase):
    def test_list_and_get_scan_txt_files_from_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "会议A.txt"), "w", encoding="utf-8") as file:
                file.write("# comment\n图灵科技\n\nfaster-whisper\n")
            with open(os.path.join(directory, "ignore.md"), "w", encoding="utf-8") as file:
                file.write("ignored")

            store = MeetingHotwordStore(directory)
            meetings = store.list()["meetings"]
            self.assertEqual(len(meetings), 1)
            self.assertEqual(meetings[0]["meeting_name"], "会议A")
            self.assertEqual(meetings[0]["filename"], "会议A.txt")
            self.assertEqual(meetings[0]["count"], 2)

            loaded = store.get("会议A")
            self.assertEqual(loaded["text"], "图灵科技\nfaster-whisper")
            self.assertEqual(loaded["count"], 2)

            missing = store.get("会议B")
            self.assertEqual(missing["count"], 0)
            self.assertEqual(missing["filename"], "")

    def test_count_hotwords_ignores_blank_lines_and_comments(self):
        self.assertEqual(count_hotwords("# c\nACE\n\nDocker"), 2)

    def test_apply_meeting_hotwords_before_default_hotwords(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "会议A.txt"), "w", encoding="utf-8") as file:
                file.write("图灵科技\nfaster-whisper")
            server = TranscriptionServer()
            server.meeting_hotwords = MeetingHotwordStore(directory)
            server.default_hotwords = "Default"

            options = {"uid": "client", "meeting_name": "会议A"}
            server.apply_meeting_hotwords(options)
            server.apply_default_hotwords(options)

            self.assertEqual(options["hotwords"], "图灵科技 faster-whisper")
            self.assertEqual(options["hotwords_count"], 2)
            self.assertEqual(options["hotwords_file"], "会议A.txt")
            self.assertTrue(options["hotwords_locked"])

            options = {"uid": "client", "meeting_name": "会议A", "hotwords": "Custom"}
            server.apply_meeting_hotwords(options)
            server.apply_default_hotwords(options)
            self.assertEqual(options["hotwords"], "Custom")


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


class TestMeetingSummaryService(unittest.TestCase):
    def test_extract_meeting_text_prefers_source_segments(self):
        service = MeetingSummaryService(startup_command="")
        payload = {"source_segments": [{"start": "0", "end": "1", "text": "hello"}], "translation_segments": [{"text": "你好"}]}
        text = service.extract_meeting_text(payload)
        self.assertIn("hello", text)
        self.assertNotIn("你好", text)

    def test_extract_meeting_text_does_not_fall_back_to_translation(self):
        service = MeetingSummaryService(startup_command="")
        payload = {"source_segments": [], "translation_segments": [{"text": "translation only"}]}
        self.assertEqual(service.extract_meeting_text(payload), "")

    def test_call_chat_disables_thinking_and_limits_output(self):
        service = MeetingSummaryService(startup_command="")
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "<think>hidden</think>{\"overview\":\"ok\"}"}, "finish_reason": "stop"}]
        }).encode("utf-8")
        response.__enter__.return_value = response
        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            content = service.call_chat([{"role": "user", "content": "test"}])
        request_payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(request_payload["max_tokens"], 1536)
        self.assertEqual(request_payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(content, {"content": '{"overview":"ok"}', "finish_reason": "stop"})

    def test_request_json_retries_truncated_response_with_compact_prompt(self):
        service = MeetingSummaryService(startup_command="")
        responses = [
            {"content": '{"overview":"cut', "finish_reason": "length"},
            {"content": '{"overview":"ok"}', "finish_reason": "stop"},
        ]
        with mock.patch.object(service, "call_chat", side_effect=responses) as call_chat:
            data = service.request_json([{"role": "user", "content": "test"}])
        self.assertEqual(data["overview"], "ok")
        self.assertEqual(call_chat.call_count, 2)
        retry_messages = call_chat.call_args_list[1].args[0]
        self.assertIn("更精简的完整 JSON", retry_messages[-1]["content"])

    def test_request_json_raises_after_two_invalid_responses(self):
        service = MeetingSummaryService(startup_command="")
        with mock.patch.object(service, "call_chat", return_value={"content": '{"overview":', "finish_reason": "stop"}):
            with self.assertRaises(SummaryGenerationError) as caught:
                service.request_json([{"role": "user", "content": "test"}])
        self.assertEqual(caught.exception.code, "summary_response_invalid_json")

    def test_split_text_uses_configured_character_budget(self):
        service = MeetingSummaryService(startup_command="", max_chars_per_chunk=2000)
        text = "\n".join(["x" * 900, "y" * 900, "z" * 900])
        chunks = service.split_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertLessEqual(max(map(len, chunks)), 2000)

    def test_normalize_summary_filters_unverified_decisions_and_actions(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-1",
            "meeting_name": "项目会",
            "source_segments": [{"start": 10.0, "end": 20.0, "text": "确认周五前由张三完成接口联调。"}],
        }
        data = {
            "meeting_type": "project_meeting",
            "overview": "确认接口联调安排。",
            "decisions": [
                {"text": "周五前完成联调", "evidence_start": 10, "evidence_end": 20, "evidence_quote": "确认周五前由张三完成接口联调。"},
                {"text": "上线延期", "evidence_start": 10, "evidence_end": 20, "evidence_quote": "原文不存在"},
            ],
            "action_items": [
                {"task": "完成接口联调", "owner": "张三", "deadline": "周五前", "status": None,
                 "evidence_start": 10, "evidence_end": 20, "evidence_quote": "周五前由张三完成接口联调"},
            ],
        }
        summary = service.normalize_summary(data, payload, template="project_meeting")
        self.assertEqual(len(summary["decisions"]), 1)
        self.assertEqual(len(summary["action_items"]), 1)
        self.assertEqual(summary["summary_quality"]["filtered_unverified_count"], 1)
        self.assertEqual(summary["summary_template"], "project_meeting")

    def test_training_template_has_independent_sections_and_evidence(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-2",
            "meeting_name": "培训",
            "source_segments": [{"start": 0, "end": 10, "text": "人类容易高估过去的变化，低估未来的变化。"}],
        }
        data = {
            "meeting_type": "training_speech",
            "overview": "讨论时间认知偏差。",
            "thesis": "心理学需要关注未来。",
            "outline": ["时间认知偏差", "未来导向"],
            "key_points": [{"text": "人类低估未来变化", "evidence_start": 0, "evidence_end": 10,
                            "evidence_quote": "人类容易高估过去的变化，低估未来的变化。"}],
            "arguments": ["通过过去与未来变化感知的对比展开论证"],
            "cases": [],
            "data_points": [],
            "notable_quotes": [],
            "takeaways": ["应主动培养未来导向"],
            "asr_uncertainties": [],
            "timeline": [{"text": "时间认知偏差", "evidence_start": 0, "evidence_end": 10,
                          "evidence_quote": "人类容易高估过去的变化，低估未来的变化。"}],
        }
        summary = service.normalize_summary(data, payload, template="training_speech")
        markdown = MeetingLogStore.render_summary_markdown(summary)
        self.assertEqual(summary["summary_quality"]["evidence_count"], 2)
        self.assertIn("## 核心主旨", markdown)
        self.assertIn("## 核心观点", markdown)
        self.assertIn("## 内容时间线", markdown)
        self.assertNotIn("## 关键决策", markdown)
        self.assertNotIn("## 待办事项", markdown)

    def test_each_template_uses_its_own_prompt_schema(self):
        service = MeetingSummaryService(startup_command="")
        self.assertIn("project_status", service.prompt_for_template("project_meeting"))
        self.assertIn("pain_points", service.prompt_for_template("customer_interview"))
        self.assertIn("notable_quotes", service.prompt_for_template("training_speech"))
        self.assertIn("disagreements", service.prompt_for_template("discussion"))

    def test_training_template_enforces_balanced_item_limits(self):
        service = MeetingSummaryService(startup_command="")
        segments = [{"start": 0, "end": 10, "text": "有效原文证据。"}]
        evidence_items = [
            {"text": f"观点{index}", "evidence_start": 0, "evidence_end": 10, "evidence_quote": "有效原文证据。"}
            for index in range(12)
        ]
        summary = service.normalize_summary({
            "meeting_type": "training_speech",
            "outline": [f"章节{index}" for index in range(12)],
            "key_points": evidence_items,
            "timeline": evidence_items,
        }, {"source_segments": segments}, template="training_speech")
        self.assertEqual(len(summary["template_data"]["outline"]), 8)
        self.assertEqual(len(summary["template_data"]["key_points"]), 6)
        self.assertEqual(len(summary["template_data"]["timeline"]), 8)

    def test_stage_prompt_only_requests_selected_fields(self):
        prompt = MeetingSummaryService.prompt_for_stage(
            "training_speech", "timeline", ("timeline",)
        )
        self.assertIn('"timeline"', prompt)
        self.assertNotIn('"key_points"', prompt)
        self.assertIn("禁止输出其他字段", prompt)

    def test_stage_truncation_falls_back_to_single_field_requests(self):
        service = MeetingSummaryService(startup_command="")
        truncated = SummaryGenerationError(
            "summary_response_truncated", "summary model response was truncated"
        )
        with mock.patch.object(
            service,
            "request_json",
            side_effect=[truncated, {"overview": "概述"}, {"thesis": "主旨"}],
        ) as request_json:
            result = service._request_stage_json(
                [
                    {"role": "system", "content": "stage"},
                    {"role": "user", "content": "source"},
                ],
                "training_speech",
                "foundation",
                ("overview", "thesis"),
                1024,
            )
        self.assertEqual(result, {"overview": "概述", "thesis": "主旨"})
        self.assertEqual(request_json.call_count, 3)
        self.assertIn("foundation.overview", request_json.call_args_list[1].args[0][0]["content"])

    def test_generate_uses_staged_pipeline_before_normalization(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-stage",
            "meeting_name": "培训",
            "source_segments": [
                {"start": 0, "end": 10, "text": "人类容易低估未来的变化。"}
            ],
        }
        staged_data = {
            "meeting_type": "training_speech",
            "overview": "讨论未来变化。",
            "thesis": "应重视未来变化。",
            "key_points": [
                {
                    "text": "人类容易低估未来变化",
                    "evidence_start": 0,
                    "evidence_end": 10,
                    "evidence_quote": "人类容易低估未来的变化。",
                }
            ],
        }
        with mock.patch.object(service, "ensure_ready"), mock.patch.object(
            service, "generate_staged", return_value=staged_data
        ) as generate_staged, mock.patch.object(service, "schedule_idle_shutdown"):
            summary = service.generate(payload, template="training_speech")
        generate_staged.assert_called_once()
        self.assertEqual(summary["overview"], "讨论未来变化。")
        self.assertEqual(summary["summary_quality"]["evidence_count"], 1)

    def test_summary_failure_does_not_write_new_version(self):
        server = object.__new__(TranscriptionServer)
        server.meeting_logs = MagicMock()
        server.meeting_logs.session_info.return_value = {"status": "finished"}
        server.meeting_logs.get_session_payload.return_value = {"source_segments": [{"text": "test"}]}
        server.meeting_summary = MagicMock()
        server.meeting_summary.validate_template.return_value = "training_speech"
        server.meeting_summary.generate.side_effect = SummaryGenerationError(
            "summary_response_truncated", "summary model response was truncated"
        )
        with self.assertRaises(SummaryGenerationError):
            server.generate_meeting_summary("session-1", "training_speech")
        server.meeting_logs.write_summary.assert_not_called()

    def test_validate_template_rejects_unknown_template(self):
        service = MeetingSummaryService(startup_command="")
        with self.assertRaises(ValueError):
            service.validate_template("unknown")

    def test_merge_summary_tree_merges_in_groups(self):
        service = MeetingSummaryService(startup_command="")
        summaries = [{"overview": str(index)} for index in range(9)]
        with mock.patch.object(
            service,
            "merge_summaries",
            side_effect=lambda group, _payload, _template="auto": {"overview": ",".join(item["overview"] for item in group)},
        ) as merge:
            result = service.merge_summary_tree(summaries, {})
        self.assertEqual(result["overview"], "0,1,2,3,4,5,6,7,8")
        self.assertEqual(merge.call_count, 3)


class TestClientManagerAdminStatus(unittest.TestCase):
    def setUp(self):
        self.cm = ClientManager(max_clients=4, max_connection_time=600)
        self.ws = MagicMock()
        self.client = MagicMock()
        self.client.client_uid = "uid-1"
        self.options = {
            "uid": "uid-1",
            "client_name": "会议室A",
            "meeting_name": "会议A",
            "hotwords": "图灵科技 faster-whisper",
            "hotwords_count": 2,
            "hotwords_file": "terms.txt",
            "language": "zh",
            "model": "model/asr/small",
            "enable_translation": True,
            "target_language": "en",
        }

    def test_register_client_status_snapshot(self):
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)

        payload = self.cm.get_client_status_snapshot()
        status = payload["clients"][0]

        self.assertEqual(status["uid"], "uid-1")
        self.assertEqual(status["client_name"], "会议室A")
        self.assertEqual(status["meeting_name"], "会议A")
        self.assertEqual(status["hotwords_count"], 2)
        self.assertEqual(status["hotwords_file"], "terms.txt")
        self.assertTrue(status["hotwords_locked"])
        self.assertTrue(status["connected"])
        self.assertEqual(status["backend"], "faster_whisper")
        self.assertEqual(status["language"], "zh")
        self.assertEqual(status["model"], "model/asr/small")
        self.assertTrue(status["translation_enabled"])
        self.assertEqual(status["target_language"], "en")

    def test_register_client_status_uses_default_name(self):
        self.options.pop("client_name")
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)

        status = self.cm.get_client_status_snapshot()["clients"][0]

        self.assertEqual(status["client_name"], "会议A")

    def test_update_client_message_tracks_asr_and_translation(self):
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)

        self.cm.update_client_message(self.ws, "segments", [{"text": "你好"}, {"text": "世界"}])
        self.cm.update_client_message(self.ws, "translated_segments", [{"text": "hello world"}])

        status = self.cm.get_client_status_snapshot()["clients"][0]
        self.assertEqual(status["segment_msgs"], 1)
        self.assertEqual(status["segment_items"], 2)
        self.assertEqual(status["last_source_text"], "世界")
        self.assertEqual(status["translation_msgs"], 1)
        self.assertEqual(status["translation_items"], 1)
        self.assertEqual(status["last_translation_text"], "hello world")

    def test_mark_client_disconnected_keeps_snapshot(self):
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)
        self.cm.mark_client_disconnected(self.ws)

        status = self.cm.get_client_status_snapshot()["clients"][0]
        self.assertFalse(status["connected"])
        self.assertIsNotNone(status["disconnected_at"])

    def test_same_client_instance_replaces_disconnected_snapshot(self):
        self.options["client_instance_id"] = "browser-1"
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)
        self.cm.mark_client_disconnected(self.ws)

        new_ws = MagicMock()
        new_client = MagicMock()
        new_client.client_uid = "uid-2"
        new_options = dict(self.options, uid="uid-2")
        self.cm.register_client_status(new_ws, new_client, new_options, BackendType.FASTER_WHISPER)

        clients = self.cm.get_client_status_snapshot()["clients"]
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0]["uid"], "uid-2")
        self.assertEqual(clients[0]["client_instance_id"], "browser-1")
        self.assertTrue(clients[0]["connected"])

    def test_same_client_instance_keeps_concurrent_connected_snapshot(self):
        self.options["client_instance_id"] = "browser-1"
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)

        new_ws = MagicMock()
        new_client = MagicMock()
        new_client.client_uid = "uid-2"
        new_options = dict(self.options, uid="uid-2")
        self.cm.register_client_status(new_ws, new_client, new_options, BackendType.FASTER_WHISPER)

        clients = self.cm.get_client_status_snapshot()["clients"]
        self.assertEqual(len(clients), 2)
        self.assertEqual({client["uid"] for client in clients}, {"uid-1", "uid-2"})

    def test_delete_disconnected_client_status_removes_snapshot(self):
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)
        self.cm.mark_client_disconnected(self.ws)

        result = self.cm.delete_disconnected_client_status("uid-1")

        self.assertEqual(result, "deleted")
        self.assertEqual(self.cm.get_client_status_snapshot()["clients"], [])

    def test_delete_connected_client_status_is_rejected(self):
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)

        result = self.cm.delete_disconnected_client_status("uid-1")

        self.assertEqual(result, "connected")
        self.assertEqual(len(self.cm.get_client_status_snapshot()["clients"]), 1)

    def test_delete_missing_client_status_returns_not_found(self):
        result = self.cm.delete_disconnected_client_status("missing")

        self.assertEqual(result, "not_found")


if __name__ == "__main__":
    unittest.main()
