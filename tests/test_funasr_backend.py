import json
import queue
import threading
import unittest
from unittest import mock

import numpy as np

from whisper_live.backend.funasr_backend import ServeClientFunASR


class DummyThread:
    def __init__(self, target=None):
        self.target = target

    def start(self):
        return None


class FakeOpenCCConverter:
    def convert(self, text):
        return str(text).replace("繁體", "繁体").replace("臺灣", "台湾")


class TestServeClientFunASR(unittest.TestCase):
    def setUp(self):
        ServeClientFunASR.SINGLE_MODEL = None
        ServeClientFunASR.SINGLE_MODEL_KEY = None
        ServeClientFunASR.FINAL_MODEL = None
        ServeClientFunASR.FINAL_MODEL_KEY = None

    def tearDown(self):
        ServeClientFunASR.SINGLE_MODEL = None
        ServeClientFunASR.SINGLE_MODEL_KEY = None
        ServeClientFunASR.FINAL_MODEL = None
        ServeClientFunASR.FINAL_MODEL_KEY = None

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_initializes_model_and_sends_ready(self, mock_cuda_available):
        websocket = mock.Mock()

        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", object())
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                model="iic/SenseVoiceSmall",
                use_vad=False,
            )

        self.assertEqual(client.device, "cpu")
        websocket.send.assert_called_once()
        payload = json.loads(websocket.send.call_args[0][0])
        self.assertEqual(payload["message"], "SERVER_READY")
        self.assertEqual(payload["backend"], "funasr")

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_extracts_text_and_removes_sensevoice_tags(self, mock_cuda_available):
        websocket = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", object())
            client = ServeClientFunASR(websocket=websocket, client_uid="client", use_vad=False)

        segments = client._extract_segments([{"text": "<|zh|><|NEUTRAL|>你好"}], 2.5)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "你好")
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[0].end, 2.5)

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_completed_segment_enters_translation_queue_simplified(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", object())
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                translation_queue=translation_queue,
                use_vad=False,
            )
        client.opencc_converter = FakeOpenCCConverter()
        client.frames_np = np.ones(4 * client.RATE, dtype=np.float32)

        client.handle_transcription_output(
            [{"text": "繁體中文"}, {"text": "第二句"}],
            duration=4.0,
        )

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "繁体中文")
        self.assertTrue(item["completed"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_retries_without_hotword_when_model_rejects_it(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        transcriber.generate.side_effect = [
            TypeError("unexpected keyword argument hotword"),
            [{"text": "你好"}],
        ]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                hotwords="WhisperLive",
                use_vad=False,
            )

        result = client.transcribe_audio(np.zeros(client.RATE, dtype=np.float32))

        self.assertEqual(result, [{"text": "你好"}])
        self.assertEqual(transcriber.generate.call_count, 2)
        self.assertNotIn("hotword", transcriber.generate.call_args[1])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_vad_finalizes_completed_segment_after_silence(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.return_value = [{"text": "你好世界"}]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                translation_queue=translation_queue,
                use_vad=False,
                min_segment_rms=0.0,
            )

        voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
        silence = np.zeros(client.vad_window_samples, dtype=np.float32)
        client._process_vad_window(voice, 0.0, True)
        client._process_vad_window(voice, 0.5, True)
        client._process_vad_window(silence, 1.0, False)
        client._process_vad_window(silence, 1.5, False)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "你好世界")
        self.assertTrue(item["completed"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_vad_emits_partial_without_translation(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.return_value = [{"text": "临时字幕"}]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                translation_queue=translation_queue,
                use_vad=False,
                min_segment_rms=0.0,
            )

        voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
        for index in range(3):
            client._process_vad_window(voice, index * 0.5, True)

        sent_messages = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
        partials = [msg for msg in sent_messages if msg.get("segments") and not msg["segments"][-1].get("completed")]
        self.assertTrue(partials)
        self.assertTrue(translation_queue.empty())

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_vad_silence_does_not_call_funasr(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(websocket=websocket, client_uid="client", use_vad=False)

        silence = np.zeros(client.vad_window_samples, dtype=np.float32)
        client._process_vad_window(silence, 0.0, False)
        client._process_vad_window(silence, 0.5, False)

        transcriber.generate.assert_not_called()


    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_mode_emits_partial_for_each_voice_chunk(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        transcriber.generate.return_value = [{"text": "实时字幕"}]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                mode="paraformer_streaming",
                final_refine=False,
                use_vad=False,
                min_segment_rms=0.0,
            )

        voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
        client._process_vad_window(voice, 0.0, True)

        sent_messages = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
        partials = [msg for msg in sent_messages if msg.get("segments") and not msg["segments"][-1].get("completed")]
        self.assertTrue(partials)
        generate_kwargs = transcriber.generate.call_args.kwargs
        self.assertFalse(generate_kwargs["is_final"])
        self.assertEqual(generate_kwargs["chunk_size"], [0, 10, 5])
        self.assertIs(generate_kwargs["cache"], client.streaming_cache)

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_enters_translation_queue_after_silence(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.side_effect = [
            [{"text": "你好"}],
            [{"text": "世界"}],
            [{"text": ""}],
        ]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                mode="paraformer_streaming",
                final_refine=False,
                translation_queue=translation_queue,
                use_vad=False,
                min_segment_rms=0.0,
            )

        voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
        silence = np.zeros(client.vad_window_samples, dtype=np.float32)
        client._process_vad_window(voice, 0.0, True)
        client._process_vad_window(voice, 0.6, True)
        client._process_vad_window(silence, 1.2, False)
        client._process_vad_window(silence, 1.8, False)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "你好世界")
        self.assertTrue(item["completed"])
        self.assertTrue(transcriber.generate.call_args.kwargs["is_final"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_cache_is_per_client(self, mock_cuda_available):
        websocket_a = mock.Mock()
        websocket_b = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", mock.Mock())
            client_a = ServeClientFunASR(websocket=websocket_a, client_uid="a", mode="paraformer_streaming", final_refine=False, use_vad=False)
            client_b = ServeClientFunASR(websocket=websocket_b, client_uid="b", mode="paraformer_streaming", final_refine=False, use_vad=False)

        self.assertIsNot(client_a.streaming_cache, client_b.streaming_cache)

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_punctuation_applied_to_final_text(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.side_effect = [[{"text": "你好世界"}], [{"text": ""}]]
        punctuator = mock.Mock()
        punctuator.generate.return_value = [{"text": "你好世界。"}]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            with mock.patch.object(ServeClientFunASR, "create_punc_model", autospec=True) as create_punc:
                create_punc.side_effect = lambda client: setattr(client, "punctuator", punctuator)
                client = ServeClientFunASR(
                    websocket=websocket,
                    client_uid="client",
                    mode="paraformer_streaming",
                    punc_model="model/funasr/ct-punc",
                    final_refine=False,
                    translation_queue=translation_queue,
                    use_vad=False,
                    min_segment_rms=0.0,
                )

        voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
        silence = np.zeros(client.vad_window_samples, dtype=np.float32)
        client._process_vad_window(voice, 0.0, True)
        client._process_vad_window(silence, 0.6, False)
        client._process_vad_window(silence, 1.2, False)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "你好世界。")


    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_sentence_punctuation_finalizes_by_default_after_min_duration(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.side_effect = [[{"text": "你好"}], [{"text": "你好。"}]]
        with mock.patch.object(ServeClientFunASR, "STREAMING_SENTENCE_ENDPOINT_MIN_SECONDS", 1.0):
            with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
                create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
                client = ServeClientFunASR(
                    websocket=websocket,
                    client_uid="client",
                    mode="paraformer_streaming",
                    final_refine=False,
                    translation_queue=translation_queue,
                    use_vad=False,
                    min_segment_rms=0.0,
                )

            voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
            client._process_vad_window(voice, 0.0, True)
            self.assertTrue(translation_queue.empty())
            client._process_vad_window(voice, 0.6, True)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "你好。")
        self.assertTrue(item["completed"])
        self.assertFalse(transcriber.generate.call_args.kwargs["is_final"])


    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_soft_max_finalizes_before_hard_max(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.return_value = [{"text": "这是一个较长的流式文本"}]
        with mock.patch.object(ServeClientFunASR, "STREAMING_SOFT_MAX_SPEECH_SECONDS", 1.6):
            with mock.patch.object(ServeClientFunASR, "STREAMING_SOFT_MAX_MIN_CHARS", 8):
                with mock.patch.object(ServeClientFunASR, "MAX_SPEECH_SECONDS", 8.0):
                    with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
                        create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
                        client = ServeClientFunASR(
                            websocket=websocket,
                            client_uid="client",
                            mode="paraformer_streaming",
                            final_refine=False,
                            translation_queue=translation_queue,
                            use_vad=False,
                            min_segment_rms=0.0,
                        )

                    reasons = []
                    original_commit = client._commit_completed_segments
                    def record_commit(segments, reason, duration):
                        reasons.append(reason)
                        return original_commit(segments, reason, duration)
                    client._commit_completed_segments = record_commit

                    voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
                    client._process_vad_window(voice, 0.0, True)
                    client._process_vad_window(voice, 0.8, True)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "这是一个较长的流式文本")
        self.assertEqual(reasons, ["soft_max_speech"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_soft_max_requires_enough_text(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.return_value = [{"text": "你好"}]
        with mock.patch.object(ServeClientFunASR, "STREAMING_SOFT_MAX_SPEECH_SECONDS", 1.6):
            with mock.patch.object(ServeClientFunASR, "STREAMING_SOFT_MAX_MIN_CHARS", 8):
                with mock.patch.object(ServeClientFunASR, "MAX_SPEECH_SECONDS", 8.0):
                    with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
                        create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
                        client = ServeClientFunASR(
                            websocket=websocket,
                            client_uid="client",
                            mode="paraformer_streaming",
                            final_refine=False,
                            translation_queue=translation_queue,
                            use_vad=False,
                            min_segment_rms=0.0,
                        )

                    voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
                    client._process_vad_window(voice, 0.0, True)
                    client._process_vad_window(voice, 0.8, True)

        self.assertTrue(translation_queue.empty())

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_hard_max_still_fallback_when_soft_max_does_not_trigger(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.return_value = [{"text": "你好"}]
        with mock.patch.object(ServeClientFunASR, "STREAMING_SOFT_MAX_SPEECH_SECONDS", 1.6):
            with mock.patch.object(ServeClientFunASR, "STREAMING_SOFT_MAX_MIN_CHARS", 8):
                with mock.patch.object(ServeClientFunASR, "MAX_SPEECH_SECONDS", 2.4):
                    with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
                        create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
                        client = ServeClientFunASR(
                            websocket=websocket,
                            client_uid="client",
                            mode="paraformer_streaming",
                            final_refine=False,
                            translation_queue=translation_queue,
                            use_vad=False,
                            min_segment_rms=0.0,
                        )

                    reasons = []
                    original_commit = client._commit_completed_segments
                    def record_commit(segments, reason, duration):
                        reasons.append(reason)
                        return original_commit(segments, reason, duration)
                    client._commit_completed_segments = record_commit

                    voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
                    client._process_vad_window(voice, 0.0, True)
                    client._process_vad_window(voice, 0.8, True)
                    client._process_vad_window(voice, 1.6, True)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "你好")
        self.assertEqual(reasons, ["max_speech"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_sentence_punctuation_takes_priority_over_soft_max(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.return_value = [{"text": "这是一个完整句子。"}]
        with mock.patch.object(ServeClientFunASR, "STREAMING_SENTENCE_ENDPOINT_MIN_SECONDS", 1.6):
            with mock.patch.object(ServeClientFunASR, "STREAMING_SOFT_MAX_SPEECH_SECONDS", 1.6):
                with mock.patch.object(ServeClientFunASR, "STREAMING_SOFT_MAX_MIN_CHARS", 8):
                    with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
                        create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
                        client = ServeClientFunASR(
                            websocket=websocket,
                            client_uid="client",
                            mode="paraformer_streaming",
                            final_refine=False,
                            translation_queue=translation_queue,
                            use_vad=False,
                            min_segment_rms=0.0,
                        )

                    reasons = []
                    original_commit = client._commit_completed_segments

                    def record_commit(segments, reason, duration):
                        reasons.append(reason)
                        return original_commit(segments, reason, duration)

                    client._commit_completed_segments = record_commit

                    voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
                    client._process_vad_window(voice, 0.0, True)
                    client._process_vad_window(voice, 0.8, True)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "这是一个完整句子。")
        self.assertEqual(reasons, ["sentence_punctuation"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_short_sentences_are_merged_before_commit(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        transcriber = mock.Mock()
        transcriber.generate.side_effect = [[{"text": "第一句。第二句。"}], [{"text": ""}]]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                mode="paraformer_streaming",
                final_refine=False,
                translation_queue=translation_queue,
                use_vad=False,
                min_segment_rms=0.0,
            )

        voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
        silence = np.zeros(client.vad_window_samples, dtype=np.float32)
        client._process_vad_window(voice, 0.0, True)
        client._process_vad_window(silence, 0.8, False)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "第一句。第二句。")
        self.assertTrue(translation_queue.empty())
        self.assertLess(float(item["start"]), float(item["end"]))

        final_messages = [json.loads(call.args[0]) for call in websocket.send.call_args_list if "segments" in json.loads(call.args[0])]
        final_segments = final_messages[-1]["segments"]
        self.assertEqual(final_segments[-1]["text"], "第一句。第二句。")

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_long_clause_splits_on_weak_boundary(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "FINAL_SPLIT_TARGET_CHARS", 14):
            with mock.patch.object(ServeClientFunASR, "FINAL_SPLIT_MAX_CHARS", 20):
                with mock.patch.object(ServeClientFunASR, "FINAL_SPLIT_MIN_CHARS", 4):
                    with mock.patch.object(ServeClientFunASR, "FINAL_SPLIT_WEAK_BOUNDARIES", ("但是", "所以")):
                        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
                            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
                            client = ServeClientFunASR(
                                websocket=websocket,
                                client_uid="client",
                                mode="paraformer_streaming",
                                final_refine=False,
                                use_vad=False,
                                min_segment_rms=0.0,
                            )

                        parts = client._split_final_text("前面是一段比较长的内容但是后面还有另一段需要保留下来")

        self.assertGreaterEqual(len(parts), 2)
        self.assertEqual("".join(parts), "前面是一段比较长的内容但是后面还有另一段需要保留下来")
        self.assertTrue(any(part.startswith("但是") for part in parts[1:]))

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_split_final_segments_share_utterance_id(self, mock_cuda_available):
        websocket = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", mock.Mock())
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                mode="paraformer_streaming",
                final_refine=False,
                use_vad=False,
            )

        client.speech_start_time = 1.25
        client._begin_utterance()
        partial = client._format_utterance_segment(1.25, 2.0, "临时字幕", completed=False)
        final_segments = client._segments_from_text_parts(1.25, 3.25, ["第一段。", "第二段。"])

        self.assertEqual(len(final_segments), 2)
        self.assertEqual(partial["utterance_id"], final_segments[0]["utterance_id"])
        self.assertEqual(final_segments[0]["utterance_id"], final_segments[1]["utterance_id"])
        client._reset_speech_state()
        self.assertIsNone(client.current_utterance_id)

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_split_final_segments_share_speaker_from_one_inference(self, mock_cuda_available):
        websocket = mock.Mock()
        diarizer = mock.Mock()
        diarizer.identify_speaker.return_value = "SPEAKER_01"
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", mock.Mock())
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                mode="paraformer_streaming",
                final_refine=False,
                use_vad=False,
                min_segment_rms=0.0,
                diarization=diarizer,
            )

        client.speech_buffer = np.ones(client.RATE * 2, dtype=np.float32) * 0.1
        client.speech_start_time = 0.0
        client.streaming_partial_text = "第一段。第二段。"
        client._begin_utterance()
        with mock.patch.object(client, "_split_final_text", return_value=["第一段。", "第二段。"]):
            client._emit_streaming_final(np.empty(0, dtype=np.float32), reason="silence")

        self.assertEqual(
            [segment["speaker"] for segment in client.transcript],
            ["SPEAKER_01", "SPEAKER_01"],
        )
        diarizer.identify_speaker.assert_called_once()

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_partial_segment_does_not_run_diarization(self, mock_cuda_available):
        websocket = mock.Mock()
        diarizer = mock.Mock()
        transcriber = mock.Mock()
        transcriber.generate.return_value = [{"text": "临时字幕"}]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                use_vad=False,
                min_segment_rms=0.0,
                diarization=diarizer,
            )

        client.speech_buffer = np.ones(client.RATE, dtype=np.float32) * 0.1
        client.speech_start_time = 0.0
        client._begin_utterance()
        client._emit_current_speech(completed=False, reason="partial")

        diarizer.identify_speaker.assert_not_called()

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_weak_boundary_list_is_conservative(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "FINAL_SPLIT_TARGET_CHARS", 18):
            with mock.patch.object(ServeClientFunASR, "FINAL_SPLIT_MAX_CHARS", 28):
                with mock.patch.object(ServeClientFunASR, "FINAL_SPLIT_MIN_CHARS", 4):
                    with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
                        create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
                        client = ServeClientFunASR(
                            websocket=websocket,
                            client_uid="client",
                            mode="paraformer_streaming",
                            final_refine=False,
                            use_vad=False,
                            min_segment_rms=0.0,
                        )

                    parts = client._split_final_text("前面是一段比较长的内容其实后面比如说还有然后继续说")

        self.assertEqual(parts, ["前面是一段比较长的内容其实后面比如说还有然后继续说"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_text_normalizes_redundant_punctuation(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                mode="paraformer_streaming",
                final_refine=False,
                use_vad=False,
                min_segment_rms=0.0,
            )

        self.assertEqual(client._normalize_final_text("第一句。，  第二句。。"), "第一句。 第二句。")
        self.assertEqual(client._normalize_final_text("你好，，世界，。继续！！"), "你好，世界。继续！")
        self.assertEqual(client._normalize_final_text("I want to be normal.."), "I want to be normal.")

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_text_repairs_fragmented_punctuation(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                mode="paraformer_streaming",
                final_refine=False,
                use_vad=False,
                min_segment_rms=0.0,
            )

        self.assertEqual(client._repair_fragmented_punctuation("我觉得我的选。选择无比的友谊"), "我觉得我的选择无比的友谊")
        self.assertEqual(client._repair_fragmented_punctuation("后来那些遇到。这种问题"), "后来那些遇到这种问题")
        self.assertEqual(client._repair_fragmented_punctuation("你长。你长什么样子"), "你长什么样子")
        self.assertEqual(client._split_final_text("我觉得我的选。选择无比的友谊"), ["我觉得我的选择无比的友谊"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_text_keeps_standalone_short_sentences(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                mode="paraformer_streaming",
                final_refine=False,
                use_vad=False,
                min_segment_rms=0.0,
            )

        self.assertEqual(client._repair_fragmented_punctuation("是。他说哪有。"), "是。他说哪有。")
        self.assertEqual(client._repair_fragmented_punctuation("好。这个我们继续。"), "好。这个我们继续。")
        self.assertEqual(client._repair_fragmented_punctuation("我很开心。我们继续。"), "我很开心。我们继续。")

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_refine_replaces_streaming_text(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        streaming_model = mock.Mock()
        streaming_model.generate.side_effect = [[{"text": "怕form部署"}], [{"text": ""}]]
        final_model = mock.Mock()
        final_model.generate.return_value = [{"text": "Paraformer 部署"}]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", streaming_model)
            with mock.patch.object(ServeClientFunASR, "create_final_model", autospec=True) as create_final:
                create_final.side_effect = lambda client, single_model=False: setattr(client, "final_transcriber", final_model)
                client = ServeClientFunASR(
                    websocket=websocket,
                    client_uid="client",
                    mode="paraformer_streaming",
                    final_model="model/funasr/SenseVoiceSmall",
                    translation_queue=translation_queue,
                    use_vad=False,
                    min_segment_rms=0.0,
                )

        voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
        silence = np.zeros(client.vad_window_samples, dtype=np.float32)
        client._process_vad_window(voice, 0.0, True)
        client._process_vad_window(silence, 0.8, False)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "Paraformer 部署")
        self.assertTrue(item["completed"])
        final_model.generate.assert_called_once()

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_streaming_final_refine_falls_back_to_streaming_text(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        streaming_model = mock.Mock()
        streaming_model.generate.side_effect = [[{"text": "流式文本"}], [{"text": ""}]]
        final_model = mock.Mock()
        final_model.generate.return_value = [{"text": ""}]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", streaming_model)
            with mock.patch.object(ServeClientFunASR, "create_final_model", autospec=True) as create_final:
                create_final.side_effect = lambda client, single_model=False: setattr(client, "final_transcriber", final_model)
                client = ServeClientFunASR(
                    websocket=websocket,
                    client_uid="client",
                    mode="paraformer_streaming",
                    final_model="model/funasr/SenseVoiceSmall",
                    translation_queue=translation_queue,
                    use_vad=False,
                    min_segment_rms=0.0,
                )

        voice = np.ones(client.vad_window_samples, dtype=np.float32) * 0.1
        silence = np.zeros(client.vad_window_samples, dtype=np.float32)
        client._process_vad_window(voice, 0.0, True)
        client._process_vad_window(silence, 0.8, False)

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "流式文本")
        self.assertTrue(item["completed"])


if __name__ == "__main__":
    unittest.main()
