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

    def test_standard_mode_enables_conservative_segmentation(self):
        server = TranscriptionServer()
        server.backend = BackendType.FASTER_WHISPER
        options = {"service_mode": "standard"}

        server.apply_standard_segmentation_defaults(options)

        self.assertTrue(options["conservative_segmentation"])
        self.assertEqual(options["same_output_threshold"], 9)
        self.assertEqual(options["max_incomplete_segment_seconds"], 12.0)
        self.assertEqual(options["sentence_completion_min_seconds"], 3.0)
        self.assertEqual(options["short_fragment_hold_seconds"], 2.5)
        self.assertEqual(options["min_new_audio_seconds"], 0.25)

    def test_standard_segmentation_does_not_change_accurate_mode(self):
        server = TranscriptionServer()
        server.backend = BackendType.FASTER_WHISPER
        options = {"service_mode": "accurate"}

        server.apply_standard_segmentation_defaults(options)

        self.assertNotIn("conservative_segmentation", options)
        self.assertNotIn("short_fragment_hold_seconds", options)
        self.assertNotIn("min_new_audio_seconds", options)


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


class TestTranslationDraftConfig(unittest.TestCase):
    def eligible_options(self, **overrides):
        options = {
            "uid": "draft-client",
            "language": "en",
            "enable_translation": True,
            "service_mode": "accurate",
            "translation_mode": "standard",
            "target_language": "zh",
            "translation_draft_enabled": True,
        }
        options.update(overrides)
        return options

    def test_eligible_accurate_en_zh_uses_defaults(self):
        options = TranscriptionServer.normalize_translation_draft_options(self.eligible_options())

        self.assertTrue(options["translation_draft_enabled"])
        self.assertTrue(options["translation_readability_context_enabled"])
        self.assertEqual(options["translation_draft_interval_seconds"], 1.2)
        self.assertEqual(options["translation_draft_min_delta_chars"], 8)
        self.assertEqual(options["translation_draft_max_source_chars"], 220)
        self.assertEqual(options["translation_readability_context_sentences"], 2)
        self.assertEqual(options["translation_readability_context_max_chars"], 220)

    def test_auto_target_resolves_to_zh_for_english_source(self):
        options = TranscriptionServer.normalize_translation_draft_options(
            self.eligible_options(target_language="auto")
        )

        self.assertTrue(options["translation_draft_enabled"])

    def test_mixed_interpretation_enables_draft_and_context_for_runtime_direction(self):
        options = TranscriptionServer.normalize_translation_draft_options(
            self.eligible_options(translation_mode="mixed_interpretation", language=None, target_language="auto")
        )

        self.assertTrue(options["translation_draft_enabled"])
        self.assertTrue(options["translation_readability_context_enabled"])
        self.assertEqual(options["translation_readability_context_sentences"], 2)
        self.assertEqual(options["translation_readability_context_max_chars"], 220)

    def test_zh_en_can_enable_context_without_draft(self):
        options = TranscriptionServer.normalize_translation_draft_options(
            self.eligible_options(
                language="zh",
                target_language="en",
                translation_draft_enabled=False,
                translation_readability_context_enabled=True,
            )
        )

        self.assertFalse(options["translation_draft_enabled"])
        self.assertTrue(options["translation_readability_context_enabled"])
        self.assertEqual(options["translation_readability_context_sentences"], 2)
        self.assertEqual(options["translation_readability_context_max_chars"], 220)

    def test_non_faster_whisper_backend_is_disabled(self):
        options = TranscriptionServer.normalize_translation_draft_options(
            self.eligible_options(),
            backend=BackendType.FUNASR,
        )

        self.assertFalse(options["translation_draft_enabled"])
        self.assertEqual(options["translation_readability_context_sentences"], 0)
        self.assertEqual(options["translation_readability_context_max_chars"], 0)

    def test_old_or_ineligible_clients_are_disabled(self):
        cases = [
            self.eligible_options(translation_draft_enabled=None),
            self.eligible_options(translation_draft_enabled="false"),
            self.eligible_options(service_mode="standard"),
            self.eligible_options(service_mode="conversation"),
            self.eligible_options(service_mode="transcription", enable_translation=False),
        ]

        for options in cases:
            with self.subTest(options=options):
                normalized = TranscriptionServer.normalize_translation_draft_options(options)
                self.assertFalse(normalized["translation_draft_enabled"])
                self.assertFalse(normalized["translation_readability_context_enabled"])
                self.assertEqual(normalized["translation_readability_context_sentences"], 0)
                self.assertEqual(normalized["translation_readability_context_max_chars"], 0)

    def test_invalid_and_out_of_range_values_are_normalized(self):
        options = TranscriptionServer.normalize_translation_draft_options(self.eligible_options(
            translation_draft_interval_seconds="not-a-number",
            translation_draft_min_delta_chars=-5,
            translation_draft_max_source_chars=9999,
            translation_readability_context_sentences=99,
            translation_readability_context_max_chars=1,
        ))

        self.assertEqual(options["translation_draft_interval_seconds"], 1.2)
        self.assertEqual(options["translation_draft_min_delta_chars"], 1)
        self.assertEqual(options["translation_draft_max_source_chars"], 220)
        self.assertEqual(options["translation_readability_context_sentences"], 2)
        self.assertEqual(options["translation_readability_context_max_chars"], 32)

    def test_non_finite_and_upper_interval_are_bounded(self):
        non_finite = TranscriptionServer.normalize_translation_draft_options(
            self.eligible_options(translation_draft_interval_seconds=float("inf"))
        )
        upper = TranscriptionServer.normalize_translation_draft_options(
            self.eligible_options(translation_draft_interval_seconds=30)
        )

        self.assertEqual(non_finite["translation_draft_interval_seconds"], 1.2)
        self.assertEqual(upper["translation_draft_interval_seconds"], 10.0)

    def test_zero_context_sentences_disables_context_chars(self):
        options = TranscriptionServer.normalize_translation_draft_options(self.eligible_options(
            translation_readability_context_sentences=0,
            translation_readability_context_max_chars=220,
        ))

        self.assertTrue(options["translation_draft_enabled"])
        self.assertEqual(options["translation_readability_context_sentences"], 0)
        self.assertEqual(options["translation_readability_context_max_chars"], 0)


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
    @mock.patch("whisper_live.backend.translation_backend.ServeClientTranslation")
    def test_initialize_client_passes_translation_buffer_config(
        self, mock_translation_client, mock_faster_client, mock_thread
    ):
        mock_faster_client.return_value = MagicMock()
        mock_translation_client.return_value = MagicMock()
        options = {
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "small",
            "enable_translation": True,
            "translation_max_chars": 240,
            "translation_max_wait_seconds": 2.0,
            "translation_incomplete_max_wait_seconds": 4.0,
            "translation_context_seconds": 5.0,
            "translation_merge_enabled": False,
        }

        self.server.initialize_client(MagicMock(), options, None, None, False)

        kwargs = mock_translation_client.call_args.kwargs
        self.assertEqual(kwargs["translation_max_chars"], 240)
        self.assertEqual(kwargs["translation_max_wait_seconds"], 2.0)
        self.assertEqual(kwargs["translation_incomplete_max_wait_seconds"], 4.0)
        self.assertEqual(kwargs["translation_context_seconds"], 5.0)
        self.assertFalse(kwargs["translation_merge_enabled"])

    @mock.patch("whisper_live.server.threading.Thread")
    @mock.patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    @mock.patch("whisper_live.backend.translation_backend.ServeClientTranslation")
    def test_initialize_client_passes_translation_draft_config(
        self, mock_translation_client, mock_faster_client, mock_thread
    ):
        mock_faster_client.return_value = MagicMock()
        mock_translation_client.return_value = MagicMock()
        options = TranscriptionServer.normalize_translation_draft_options({
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "small",
            "enable_translation": True,
            "service_mode": "accurate",
            "translation_mode": "standard",
            "target_language": "zh",
            "translation_draft_enabled": True,
            "translation_draft_interval_seconds": 1.2,
            "translation_draft_min_delta_chars": 8,
            "translation_draft_max_source_chars": 220,
            "translation_readability_context_sentences": 2,
            "translation_readability_context_max_chars": 220,
        })

        self.server.initialize_client(MagicMock(), options, None, None, False)

        kwargs = mock_translation_client.call_args.kwargs
        self.assertEqual(kwargs["service_mode"], "accurate")
        self.assertEqual(kwargs["source_language"], "en")
        self.assertTrue(kwargs["translation_draft_enabled"])
        self.assertTrue(kwargs["translation_readability_context_enabled"])
        self.assertEqual(kwargs["translation_draft_interval_seconds"], 1.2)
        self.assertEqual(kwargs["translation_draft_min_delta_chars"], 8)
        self.assertEqual(kwargs["translation_draft_max_source_chars"], 220)
        self.assertEqual(kwargs["translation_readability_context_sentences"], 2)
        self.assertEqual(kwargs["translation_readability_context_max_chars"], 220)

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

    @mock.patch("whisper_live.server.threading.Thread")
    @mock.patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_initialize_client_accurate_mode_defaults_max_pending_audio_to_18(self, mock_faster_client, mock_thread):
        mock_faster_client.return_value = MagicMock()
        options = {
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "small",
            "service_mode": "accurate",
        }

        self.server.initialize_client(MagicMock(), options, None, None, False)

        self.assertAlmostEqual(
            mock_faster_client.call_args.kwargs["max_pending_audio_seconds"],
            18.0,
        )

    @mock.patch("whisper_live.server.threading.Thread")
    @mock.patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_initialize_client_non_accurate_mode_defaults_max_pending_audio_to_8(self, mock_faster_client, mock_thread):
        mock_faster_client.return_value = MagicMock()
        options = {
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "small",
            "service_mode": "standard",
        }

        self.server.initialize_client(MagicMock(), options, None, None, False)

        self.assertAlmostEqual(
            mock_faster_client.call_args.kwargs["max_pending_audio_seconds"],
            8.0,
        )

    @mock.patch("whisper_live.server.threading.Thread")
    @mock.patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_initialize_client_explicit_max_pending_audio_overrides_mode_default(self, mock_faster_client, mock_thread):
        mock_faster_client.return_value = MagicMock()
        options = {
            "uid": "test",
            "language": "en",
            "task": "transcribe",
            "model": "small",
            "service_mode": "accurate",
            "max_pending_audio_seconds": 12.0,
        }

        self.server.initialize_client(MagicMock(), options, None, None, False)

        self.assertAlmostEqual(
            mock_faster_client.call_args.kwargs["max_pending_audio_seconds"],
            12.0,
        )

    @patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_failed_resume_sends_error_without_ready_and_cleans_resources(self, mock_faster_client):
        websocket = MagicMock()
        client = MagicMock()
        client.SERVER_READY = "SERVER_READY"
        client.client_uid = "uid-1"
        mock_faster_client.return_value = client
        self.server.meeting_logs = MagicMock()
        self.server.meeting_logs.resume_session.side_effect = KeyError("meeting log session not found")
        options = {
            "uid": "uid-1", "session_id": "session-1", "language": "en", "task": "transcribe",
            "model": "small", "resume_session": True,
        }

        result = self.server.initialize_client(websocket, options, None, None, False)

        self.assertFalse(result)
        payload = json.loads(websocket.send.call_args.args[0])
        self.assertEqual(payload["status"], "ERROR")
        self.assertEqual(payload["code"], "SESSION_RESUME_FAILED")
        self.assertEqual(payload["error_type"], "session_not_found")
        self.assertNotEqual(payload.get("message"), "SERVER_READY")
        client.cleanup.assert_called_once()
        websocket.close.assert_called_once()
        self.assertFalse(self.server.client_manager.get_client(websocket))

    @patch("whisper_live.server.threading.Thread")
    @patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    @patch("whisper_live.backend.translation_backend.ServeClientTranslation")
    def test_asr_setup_failure_cleans_translation_resources(self, mock_translation, mock_faster_client, mock_thread):
        websocket = MagicMock()
        translation_client = MagicMock()
        translation_thread = MagicMock()
        mock_translation.return_value = translation_client
        mock_thread.return_value = translation_thread
        mock_faster_client.side_effect = RuntimeError("ASR setup failed")

        result = self.server.initialize_client(websocket, {
            "uid": "uid-1", "language": "en", "task": "transcribe", "model": "small",
            "enable_translation": True,
        }, None, None, False)

        self.assertIsNone(result)
        translation_client.cleanup.assert_called_once()
        translation_thread.join.assert_called_once_with(timeout=2.0)
        websocket.close.assert_called_once()
        self.assertFalse(self.server.client_manager.get_client(websocket))

    @patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_meeting_log_start_failure_does_not_register_or_send_ready(self, mock_faster_client):
        websocket = MagicMock()
        client = MagicMock(client_uid="uid-1", SERVER_READY="SERVER_READY")
        mock_faster_client.return_value = client
        self.server.meeting_logs = MagicMock()
        self.server.meeting_logs.start_session.side_effect = RuntimeError("log storage unavailable")

        result = self.server.initialize_client(websocket, {
            "uid": "uid-1", "language": "en", "task": "transcribe", "model": "small",
        }, None, None, False)

        self.assertFalse(result)
        client.cleanup.assert_called_once()
        websocket.close.assert_called_once()
        self.assertFalse(self.server.client_manager.get_client(websocket))
        sent_messages = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
        self.assertTrue(all(message.get("message") != "SERVER_READY" for message in sent_messages))

    @patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_ready_delivery_failure_runs_registered_cleanup(self, mock_faster_client):
        websocket = MagicMock()
        websocket.send.side_effect = ConnectionError("closed")
        client = MagicMock(client_uid="uid-1")
        client.stop_and_join.return_value = True
        mock_faster_client.return_value = client
        self.server.meeting_logs = MagicMock()
        self.server.meeting_logs.start_session.return_value = {"session_id": "session-1", "status": "active"}

        result = self.server.initialize_client(websocket, {
            "uid": "uid-1", "language": "en", "task": "transcribe", "model": "small",
        }, None, None, False)

        self.assertFalse(result)
        client.stop_and_join.assert_called_once_with(timeout=2.0)
        self.assertFalse(self.server.client_manager.get_client(websocket))

    @patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_session_vad_does_not_mutate_default_for_next_client(self, mock_faster_client):
        first_client = MagicMock(client_uid="first", SERVER_READY="SERVER_READY")
        second_client = MagicMock(client_uid="second", SERVER_READY="SERVER_READY")
        mock_faster_client.side_effect = [first_client, second_client]
        self.server.meeting_logs = MagicMock()
        self.server.meeting_logs.start_session.side_effect = [
            {"session_id": "first", "status": "active"},
            {"session_id": "second", "status": "active"},
        ]
        first = {"uid": "first", "language": "en", "task": "transcribe", "model": "small"}
        second = {"uid": "second", "language": "en", "task": "transcribe", "model": "small"}

        self.assertTrue(self.server.initialize_client(MagicMock(), first, None, None, False, use_vad=False))
        self.assertTrue(self.server.initialize_client(MagicMock(), second, None, None, False))

        self.assertTrue(self.server.use_vad)
        self.assertFalse(mock_faster_client.call_args_list[0].kwargs["use_vad"])
        self.assertTrue(mock_faster_client.call_args_list[1].kwargs["use_vad"])

    @patch("whisper_live.backend.faster_whisper_backend.ServeClientFasterWhisper")
    def test_successful_resume_sends_ready_with_session_metadata(self, mock_faster_client):
        websocket = MagicMock()
        client = MagicMock()
        client.SERVER_READY = "SERVER_READY"
        client.client_uid = "uid-1"
        mock_faster_client.return_value = client
        self.server.meeting_logs = MagicMock()
        self.server.meeting_logs.resume_session.return_value = {
            "session_id": "session-1", "status": "active", "connection_count": 2,
            "connection_generation": 2, "timeline_offset_seconds": 70.0,
            "audio_gaps": [{"reason": "websocket_disconnected"}],
        }
        options = {
            "uid": "uid-1", "session_id": "session-1", "language": "en", "task": "transcribe",
            "model": "small", "resume_session": True,
        }

        result = self.server.initialize_client(websocket, options, None, None, False)

        self.assertTrue(result)
        client.start.assert_called_once_with(send_ready=False)
        payload = json.loads(websocket.send.call_args.args[0])
        self.assertEqual(payload["message"], "SERVER_READY")
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["connection_count"], 2)
        self.assertEqual(payload["connection_generation"], 2)
        self.assertEqual(payload["timeline_offset_seconds"], 70.0)
        self.assertEqual(client.meeting_log_connection_generation, 2)


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

    def test_unexpected_cleanup_passes_client_connection_generation(self):
        ws = MagicMock()
        client = MagicMock()
        client.meeting_log_session_id = "session-1"
        client.meeting_log_connection_generation = 2
        self.server.meeting_logs = MagicMock()
        self.server.client_manager.add_client(ws, client)

        self.server.cleanup(ws)

        self.server.meeting_logs.interrupt_session.assert_called_once_with(
            "session-1", expected_generation=2,
        )

    def test_join_timeout_releases_active_client_slot(self):
        ws = MagicMock()
        client = MagicMock()
        client.client_uid = "uid-1"
        client.stop_and_join.return_value = False
        self.server.client_manager.add_client(ws, client)

        self.server.cleanup(ws)

        client.stop_and_join.assert_called_once_with(timeout=2.0)
        self.assertFalse(self.server.client_manager.get_client(ws))

    def test_shutdown_batch_worker_stops_and_clears_shared_reference(self):
        from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper

        worker = MagicMock()
        ServeClientFasterWhisper.BATCH_WORKER = worker

        self.server._shutdown_batch_worker()

        worker.stop.assert_called_once_with(timeout=5.0)
        self.assertIsNone(ServeClientFasterWhisper.BATCH_WORKER)

    def test_cleanup_end_of_audio_sends_session_finalized_after_drain(self):
        events = []
        ws = MagicMock()
        ws.whisperlive_end_of_audio = True
        ws.send.side_effect = lambda _payload: events.append("session_finalized")
        translation_client = MagicMock()
        translation_client.translation_timeout_count = 2
        translation_client.finalize_translation_drain.side_effect = lambda _timeout: events.append("translation_drain") or "timed_out"
        client = MagicMock()
        client.client_uid = "uid-1"
        client.meeting_log_session_id = "session-1"
        client.wait_for_asr_finalization.side_effect = lambda _timeout: events.append("asr_finalization") or "completed"
        client.translation_client = translation_client
        client.translation_thread = None
        self.server.backend = MagicMock()
        self.server.backend.is_faster_whisper.return_value = True
        self.server.finalize_client_meeting_log = MagicMock(
            side_effect=lambda _websocket, interrupted: events.append(f"meeting_log:{interrupted}")
        )
        self.server.client_manager.add_client(ws, client)

        self.server.cleanup(ws)

        self.assertEqual(events, ["asr_finalization", "translation_drain", "meeting_log:False", "session_finalized"])
        payload = json.loads(ws.send.call_args.args[0])
        self.assertEqual(payload["message"], "SESSION_FINALIZED")
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["session_status"], "finished")
        self.assertEqual(payload["asr_finalization"], "completed")
        self.assertEqual(payload["translation_drain"], "timed_out")
        self.assertEqual(payload["translation_timeout_count"], 2)


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
            options = {"uid": "client", "meeting_name": "会议A", "service_mode": "accurate"}

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

    def test_non_accurate_modes_disable_all_hotword_features(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "会议A.txt"), "w", encoding="utf-8") as file:
                file.write("WhisperLive\nOpenAI => 开放人工智能")
            server = TranscriptionServer()
            server.meeting_hotwords = MeetingHotwordStore(directory)
            server.default_hotwords = "Default"
            for mode in ("standard", "conversation", "transcription"):
                with self.subTest(mode=mode):
                    options = {"uid": "client", "meeting_name": "会议A", "service_mode": mode}

                    server.apply_meeting_hotwords(options)
                    server.apply_default_hotwords(options)

                    self.assertNotIn("hotwords", options)
                    self.assertEqual(options["hotword_terms"], [])
                    self.assertEqual(options["hotwords_source"], "none")
                    self.assertFalse(options["hotwords_enabled"])
                    self.assertEqual(options["hotwords_disabled_reason"], "service_mode")
                    self.assertEqual(options["hotwords_count"], 0)
                    self.assertEqual(options["hotwords_file"], "")
                    self.assertEqual(options["translation_glossary"], {})
                    self.assertEqual(options["translation_glossary_count"], 0)
                    self.assertEqual(options["translation_terms"], [])

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
        self.assertEqual(options["translation_glossary"], {})
        self.assertEqual(options["translation_glossary_count"], 0)

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
        self.assertEqual(options["translation_glossary"], {})
        self.assertEqual(options["translation_glossary_count"], 0)


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


class TestTranslationDraftIsolationAndFrontendContract(unittest.TestCase):
    def test_server_filters_drafts_before_admin_and_meeting_logs(self):
        server = TranscriptionServer()
        server.client_manager = ClientManager(max_clients=4, max_connection_time=600)
        server.meeting_logs = MagicMock()
        websocket = MagicMock()
        client = MagicMock()
        client.client_uid = "uid-draft"
        client.meeting_log_session_id = "session-draft"
        options = {
            "uid": "uid-draft",
            "enable_translation": True,
            "target_language": "zh",
        }
        server.client_manager.add_client(websocket, client)
        server.client_manager.register_client_status(
            websocket, client, options, BackendType.FASTER_WHISPER
        )
        draft = {"text": "draft only", "completed": False}
        final = {"text": "final only", "completed": True}

        server.handle_client_segments(websocket, "translated_segments", [draft])
        server.meeting_logs.append_segments.assert_not_called()
        status = server.client_manager.get_client_status_snapshot()["clients"][0]
        self.assertEqual(status["translation_msgs"], 0)
        self.assertEqual(status["last_translation_text"], "")

        server.handle_client_segments(websocket, "translated_segments", [draft, final])
        server.meeting_logs.append_segments.assert_called_once_with(
            "session-draft", "translation", [final]
        )
        status = server.client_manager.get_client_status_snapshot()["clients"][0]
        self.assertEqual(status["translation_msgs"], 1)
        self.assertEqual(status["translation_items"], 1)
        self.assertEqual(status["last_translation_text"], "final only")

    def test_frontend_source_contains_revision_and_final_replacement_guards(self):
        app_path = os.path.join(os.path.dirname(__file__), "..", "web", "app.js")
        with open(app_path, encoding="utf-8") as app_file:
            source = app_file.read()

        self.assertIn("if (translationId) return `translation-id:${translationId}`;", source)
        self.assertIn("if (previousRevision !== undefined && revision <= previousRevision) return;", source)
        self.assertIn("removeDraftTranslationsForSourceIds(sourceIds);", source)
        self.assertIn("if (segment?.completed === false", source)
        self.assertIn("function resetTranslationDraftState(removeDraftSegments = false)", source)
        self.assertIn("function frontendTimeoutTargetLanguage(source)", source)
        self.assertIn("function frontendUnavailableTextForTarget(targetLanguage)", source)
        self.assertIn('translation_warning: "frontend_timeout"', source)
        self.assertNotIn("translation_confidence === 'low'", source)


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
        self.assertNotIn(self.ws, self.cm.client_status)

    def test_recent_snapshot_expires_and_capacity_is_bounded(self):
        self.cm.RECENT_STATUS_TTL_SECONDS = 0
        self.cm.register_client_status(self.ws, self.client, self.options, BackendType.FASTER_WHISPER)
        self.cm.mark_client_disconnected(self.ws)
        self.assertEqual(self.cm.get_client_status_snapshot()["clients"], [])

        self.cm.RECENT_STATUS_TTL_SECONDS = 600
        self.cm.RECENT_STATUS_MAX_RECORDS = 1
        for uid in ("uid-1", "uid-2"):
            websocket = MagicMock()
            client = MagicMock(client_uid=uid)
            self.cm.register_client_status(websocket, client, dict(self.options, uid=uid), BackendType.FASTER_WHISPER)
            self.cm.mark_client_disconnected(websocket)
        snapshots = self.cm.get_client_status_snapshot()["clients"]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["uid"], "uid-2")

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
