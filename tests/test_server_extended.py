import json
import os
import time
import threading
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.server import TranscriptionServer, BackendType, ClientManager, MeetingHotwordStore, MeetingLogStore, MeetingSummaryService, SummaryTemplateStore, SummaryGenerationError, count_hotwords, hotword_text_to_prompt, parse_hotword_config


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


class TestHotwordUploadParsing(unittest.TestCase):
    def test_parse_markdown_hotword_upload_strips_list_markers(self):
        payload = """# 热词
- 彭凯平
* 历史终极幻觉
1. 时间韧性
- [x] Qwen=>Qwen
""".encode("utf-8")
        result = TranscriptionServer.parse_hotword_upload("hotwords.md", payload)
        self.assertEqual(result["filename"], "hotwords.md")
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["translation_count"], 1)
        self.assertIn("彭凯平", result["normalized_text"])
        self.assertIn("Qwen => Qwen", result["normalized_text"])
        self.assertNotIn("- 彭凯平", result["normalized_text"])

    def test_parse_hotword_upload_rejects_unknown_extension(self):
        with self.assertRaises(ValueError):
            TranscriptionServer.parse_hotword_upload("hotwords.csv", b"word")


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
    def test_initialize_client_passes_translation_glossary(
        self, mock_translation_client, mock_faster_client, mock_thread
    ):
        mock_faster_client.return_value = MagicMock()
        mock_translation_client.return_value = MagicMock()
        glossary = {"OpenAI": "开放人工智能"}
        options = {
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "small",
            "enable_translation": True,
            "translation_glossary": glossary,
        }

        self.server.initialize_client(MagicMock(), options, None, None, False)

        self.assertEqual(
            mock_translation_client.call_args.kwargs["translation_glossary"],
            glossary,
        )

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

    @mock.patch("whisper_live.server.threading.Thread")
    @mock.patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_initialize_client_passes_hotword_terms_to_faster_whisper(self, mock_faster_client, mock_thread):
        mock_faster_client.return_value = MagicMock()
        options = {
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "small",
            "hotwords": "Whisper small OpenAI",
            "hotword_terms": ["Whisper small", "OpenAI"],
            "service_mode": "accurate",
        }

        self.server.initialize_client(MagicMock(), options, None, None, False)

        self.assertEqual(
            mock_faster_client.call_args.kwargs["hotword_terms"],
            ["Whisper small", "OpenAI"],
        )


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


class TestMeetingHotwordIntegration(unittest.TestCase):
    def test_apply_meeting_hotwords_before_default_hotwords(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "会议A.txt"), "w", encoding="utf-8") as file:
                file.write("图灵科技\nfaster-whisper")
            server = TranscriptionServer()
            server.meeting_hotwords = MeetingHotwordStore(directory)
            server.default_hotwords = "Default"

            options = {"uid": "client", "meeting_name": "会议A", "service_mode": "accurate"}
            server.apply_meeting_hotwords(options)
            server.apply_default_hotwords(options)

            self.assertEqual(options["hotwords"], "图灵科技 faster-whisper")
            self.assertEqual(options["hotword_terms"], ["图灵科技", "faster-whisper"])
            self.assertEqual(options["hotwords_source"], "meeting")
            self.assertEqual(options["hotwords_count"], 2)
            self.assertEqual(options["hotwords_original_count"], 2)
            self.assertEqual(options["hotwords_rejected_count"], 0)
            self.assertFalse(options["hotwords_truncated"])
            self.assertEqual(options["hotwords_file"], "会议A.txt")
            self.assertTrue(options["hotwords_locked"])

            options = {"uid": "client", "meeting_name": "会议A", "hotwords": "Custom", "service_mode": "accurate"}
            server.apply_meeting_hotwords(options)
            server.apply_default_hotwords(options)
            self.assertEqual(options["hotwords"], "Custom")

    def test_translation_only_meeting_hotwords_do_not_apply_default_asr_hotwords(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "会议A.txt"), "w", encoding="utf-8") as file:
                file.write("NICE => 长三角国家技术创新中心")
            server = TranscriptionServer()
            server.meeting_hotwords = MeetingHotwordStore(directory)
            server.default_hotwords = "Default"
            options = {"uid": "client", "meeting_name": "会议A"}

            server.apply_meeting_hotwords(options)
            server.apply_default_hotwords(options)

            self.assertNotIn("hotwords", options)
            self.assertEqual(options["hotwords_count"], 0)
            self.assertEqual(options["hotwords_source"], "meeting")
            self.assertEqual(options["translation_glossary"], {"NICE": "长三角国家技术创新中心"})
            self.assertEqual(options["translation_glossary_count"], 1)
            self.assertTrue(options["hotwords_locked"])

    def test_meeting_translation_glossary_is_loaded_with_custom_hotwords(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "会议A.txt"), "w", encoding="utf-8") as file:
                file.write("OpenAI => 开放人工智能\n普通热词")
            server = TranscriptionServer()
            server.meeting_hotwords = MeetingHotwordStore(directory)
            options = {"uid": "client", "meeting_name": "会议A", "hotwords": "Custom", "service_mode": "accurate"}

            server.apply_meeting_hotwords(options)

            self.assertEqual(options["hotwords"], "Custom")
            self.assertEqual(options["hotwords_source"], "client")
            self.assertEqual(options["translation_glossary"], {"OpenAI": "开放人工智能"})
            self.assertEqual(options["translation_glossary_count"], 1)
            self.assertEqual(options["hotwords_count"], 1)

    def test_client_hotword_terms_preserve_phrase_boundary(self):
        server = TranscriptionServer()
        server.default_hotwords = "Default"
        options = {
            "uid": "client",
            "hotwords": "Whisper small OpenAI",
            "hotword_terms": ["Whisper small", "OpenAI"],
            "hotwords_locked": True,
            "service_mode": "accurate",
        }

        server.apply_meeting_hotwords(options)
        server.apply_default_hotwords(options)

        self.assertEqual(options["hotwords"], "Whisper small OpenAI")
        self.assertEqual(options["hotword_terms"], ["Whisper small", "OpenAI"])
        self.assertEqual(options["hotwords_source"], "client")
        self.assertEqual(options["hotwords_count"], 2)

    def test_standard_mode_disables_asr_hotwords_but_preserves_translation_glossary(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "会议A.txt"), "w", encoding="utf-8") as file:
                file.write("WhisperLive\nOpenAI => 开放人工智能")
            server = TranscriptionServer()
            server.meeting_hotwords = MeetingHotwordStore(directory)
            server.default_hotwords = "Default"
            options = {"uid": "client", "meeting_name": "会议A", "service_mode": "standard"}

            server.apply_meeting_hotwords(options)
            server.apply_default_hotwords(options)

            self.assertNotIn("hotwords", options)
            self.assertEqual(options["hotword_terms"], [])
            self.assertEqual(options["hotwords_source"], "meeting")
            self.assertFalse(options["hotwords_enabled"])
            self.assertEqual(options["hotwords_disabled_reason"], "service_mode")
            self.assertEqual(options["hotwords_original_count"], 1)
            self.assertEqual(options["hotwords_count"], 0)
            self.assertEqual(options["translation_glossary"], {"OpenAI": "开放人工智能"})

    def test_missing_service_mode_disables_client_asr_hotwords(self):
        server = TranscriptionServer()
        options = {
            "uid": "client",
            "hotwords": "WhisperLive",
            "hotword_terms": ["WhisperLive"],
            "hotwords_locked": True,
            "translation_glossary": {"OpenAI": "开放人工智能"},
        }

        server.apply_meeting_hotwords(options)

        self.assertNotIn("hotwords", options)
        self.assertEqual(options["hotwords_source"], "client")
        self.assertFalse(options["hotwords_enabled"])
        self.assertEqual(options["hotwords_disabled_reason"], "service_mode")
        self.assertEqual(options["translation_glossary"], {"OpenAI": "开放人工智能"})

    def test_client_translation_only_upload_blocks_default_hotwords(self):
        server = TranscriptionServer()
        server.default_hotwords = "Default"
        options = {
            "uid": "client",
            "hotwords_locked": True,
            "translation_glossary": {"OpenAI": "开放人工智能"},
        }

        server.apply_meeting_hotwords(options)
        server.apply_default_hotwords(options)

        self.assertNotIn("hotwords", options)
        self.assertEqual(options["hotwords_source"], "client")
        self.assertEqual(options["hotwords_count"], 0)
        self.assertEqual(options["translation_glossary"], {"OpenAI": "开放人工智能"})


class TestMeetingSummaryIntegration(unittest.TestCase):
    def test_custom_quality_failure_does_not_write_new_version(self):
        server = object.__new__(TranscriptionServer)
        server.meeting_logs = MagicMock()
        server.meeting_logs.session_info.return_value = {"status": "finished"}
        server.meeting_logs.get_session_payload.return_value = {"source_segments": [{"text": "test"}]}
        server.summary_templates = MagicMock()
        server.summary_templates.get.return_value = {"id": "custom", "fields": [{"key": "overview"}]}
        server.meeting_summary = MagicMock()
        server.meeting_summary.generate_custom.side_effect = SummaryGenerationError(
            "summary_quality_insufficient", "总结质量不足，未保存新版本",
            {"missing_fields": ["overview"]},
        )
        with self.assertRaises(SummaryGenerationError):
            server.generate_meeting_summary("session-1", "custom", "custom")
        server.meeting_logs.write_summary.assert_not_called()

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


class TestMeetingCompatibilityExports(unittest.TestCase):
    def test_server_reexports_meeting_types(self):
        from whisper_live import meeting

        self.assertIs(MeetingHotwordStore, meeting.MeetingHotwordStore)
        self.assertIs(MeetingLogStore, meeting.MeetingLogStore)
        self.assertIs(MeetingSummaryService, meeting.MeetingSummaryService)
        self.assertIs(SummaryTemplateStore, meeting.SummaryTemplateStore)


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
            "hotwords_source": "meeting",
            "hotwords_original_count": 3,
            "hotwords_rejected_count": 1,
            "hotwords_truncated": True,
            "hotwords_truncation_reasons": ["term_count"],
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
        self.assertEqual(status["hotwords_source"], "meeting")
        self.assertEqual(status["hotwords_count"], 2)
        self.assertEqual(status["hotwords_original_count"], 3)
        self.assertEqual(status["hotwords_rejected_count"], 1)
        self.assertTrue(status["hotwords_truncated"])
        self.assertEqual(status["hotwords_truncation_reasons"], ["term_count"])
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

    def test_get_client_status_entry_returns_websocket_and_snapshot(self):
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)

        websocket, status = self.cm.get_client_status_entry("uid-1")

        self.assertIs(websocket, self.ws)
        self.assertEqual(status["uid"], "uid-1")
        status["uid"] = "mutated"
        self.assertEqual(
            self.cm.get_client_status_snapshot()["clients"][0]["uid"],
            "uid-1",
        )

    def test_delete_missing_client_status_returns_not_found(self):
        result = self.cm.delete_disconnected_client_status("missing")

        self.assertEqual(result, "not_found")


if __name__ == "__main__":
    unittest.main()
