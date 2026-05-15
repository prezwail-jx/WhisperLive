import json
import queue
import unittest
from unittest import mock

from whisper_live.backend.translation_backend import ServeClientTranslation


class FakeTensorBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, text, return_tensors=None, truncation=None):
        return FakeTensorBatch(input_ids=[1, 2, 3])

    def batch_decode(self, generated_tokens, skip_special_tokens=True):
        return ["translated"]


class FakeModel:
    def to(self, device):
        return self

    def generate(self, **kwargs):
        return [[1, 2, 3]]


class TestServeClientTranslationModelCache(unittest.TestCase):
    def setUp(self):
        ServeClientTranslation._TRANSLATOR_CACHE.clear()
        ServeClientTranslation._TRANSLATOR_INFERENCE_LOCKS.clear()

    def tearDown(self):
        ServeClientTranslation._TRANSLATOR_CACHE.clear()
        ServeClientTranslation._TRANSLATOR_INFERENCE_LOCKS.clear()

    @mock.patch("whisper_live.backend.translation_backend.AutoModelForSeq2SeqLM.from_pretrained")
    @mock.patch("whisper_live.backend.translation_backend.AutoTokenizer.from_pretrained")
    def test_clients_with_same_config_share_translator(self, mock_tokenizer, mock_model):
        mock_tokenizer.return_value = FakeTokenizer()
        mock_model.return_value = FakeModel()

        client_a = ServeClientTranslation(
            client_uid="client-a",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
        )
        client_b = ServeClientTranslation(
            client_uid="client-b",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
        )

        self.assertIs(client_a.translator, client_b.translator)
        self.assertIs(client_a.translator_lock, client_b.translator_lock)
        self.assertEqual(mock_tokenizer.call_count, 2)
        self.assertEqual(mock_model.call_count, 2)

        translated, source_language, target_language = client_b.translate_text("hello", "en")
        self.assertEqual(translated, "translated")
        self.assertEqual(source_language, "en")
        self.assertEqual(target_language, "zh")

    @mock.patch("whisper_live.backend.translation_backend.AutoModelForSeq2SeqLM.from_pretrained")
    @mock.patch("whisper_live.backend.translation_backend.AutoTokenizer.from_pretrained")
    def test_client_cleanup_does_not_clear_shared_translator(self, mock_tokenizer, mock_model):
        mock_tokenizer.return_value = FakeTokenizer()
        mock_model.return_value = FakeModel()

        client_a = ServeClientTranslation(
            client_uid="client-a",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
        )
        client_b = ServeClientTranslation(
            client_uid="client-b",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
        )

        shared_translator = client_b.translator
        client_a.cleanup()

        self.assertIs(client_b.translator, shared_translator)
        translated, source_language, target_language = client_b.translate_text("你好", "zh")
        self.assertEqual(translated, "translated")
        self.assertEqual(source_language, "zh")
        self.assertEqual(target_language, "en")

    @mock.patch("whisper_live.backend.translation_backend.AutoModelForSeq2SeqLM.from_pretrained")
    @mock.patch("whisper_live.backend.translation_backend.AutoTokenizer.from_pretrained")
    def test_different_model_paths_use_different_cached_translators(self, mock_tokenizer, mock_model):
        mock_tokenizer.return_value = FakeTokenizer()
        mock_model.return_value = FakeModel()

        client_a = ServeClientTranslation(
            client_uid="client-a",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
            zh_en_model_path="model/opus-mt-zh-en",
            en_zh_model_path="model/opus-mt-en-zh",
        )
        client_b = ServeClientTranslation(
            client_uid="client-b",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
            zh_en_model_path="model/custom-zh-en",
            en_zh_model_path="model/custom-en-zh",
        )

        self.assertIsNot(client_a.translator, client_b.translator)
        self.assertEqual(mock_tokenizer.call_count, 4)
        self.assertEqual(mock_model.call_count, 4)


class TestServeClientTranslationBuffer(unittest.TestCase):
    def make_client(self, **kwargs):
        with mock.patch.object(ServeClientTranslation, "load_translation_model"):
            client = ServeClientTranslation(
                client_uid="client-buffer",
                websocket=mock.Mock(),
                translation_queue=queue.Queue(),
                **kwargs,
            )
        client.model_loaded = True
        client.translate_text = mock.Mock(
            side_effect=lambda text, source_language: (
                f"translated:{text}",
                source_language,
                "en" if source_language == "zh" else "zh",
            )
        )
        return client

    def get_last_payload(self, client):
        payload = client.websocket.send.call_args[0][0]
        return json.loads(payload)

    def test_short_segment_is_buffered_without_sending(self):
        client = self.make_client()
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "0.500",
            "text": "你好",
            "completed": True,
            "language": "zh",
        })
        client.flush_translation_buffer()

        client.websocket.send.assert_not_called()
        self.assertEqual(len(client.translation_buffer), 1)

    def test_sentence_ending_flushes_buffer(self):
        client = self.make_client()
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "你好。",
            "completed": True,
            "language": "zh",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["start"], "0.000")
        self.assertEqual(segment["end"], "1.000")
        self.assertEqual(segment["text"], "translated:你好。")
        self.assertEqual(segment["source_language"], "zh")
        self.assertEqual(segment["target_language"], "en")
        self.assertEqual(client.translation_buffer, [])

    def test_max_chars_flushes_buffer(self):
        client = self.make_client(translation_max_chars=5)
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "超过最大长度",
            "completed": True,
            "language": "zh",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:超过最大长度")

    def test_max_wait_flushes_buffer_after_min_chars(self):
        client = self.make_client(translation_min_chars=2, translation_max_wait_seconds=1.5)
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "你好",
            "completed": True,
            "language": "zh",
        })
        client.translation_buffer_started_at -= 2.0
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:你好")

    def test_exit_signal_flushes_remaining_buffer(self):
        client = self.make_client()
        client.translation_queue.put({
            "start": "0.000",
            "end": "1.000",
            "text": "还没到阈值",
            "completed": True,
            "language": "zh",
        })
        client.translation_queue.put(None)

        client.process_translation_queue()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:还没到阈值")

    def test_cleanup_flushes_remaining_buffer(self):
        client = self.make_client()
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "清理前剩余",
            "completed": True,
            "language": "zh",
        })

        client.cleanup()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:清理前剩余")
        self.assertEqual(client.translation_buffer, [])

    def test_chinese_segments_are_joined_without_spaces(self):
        client = self.make_client(translation_max_chars=4)
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "0.500",
            "text": "你好",
            "completed": True,
            "language": "zh",
        })
        client.add_segment_to_translation_buffer({
            "start": "0.500",
            "end": "1.000",
            "text": "世界",
            "completed": True,
            "language": "zh",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:你好世界")

    def test_english_segments_are_joined_with_spaces(self):
        client = self.make_client(translation_max_chars=10)
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "0.500",
            "text": "hello",
            "completed": True,
            "language": "en",
        })
        client.add_segment_to_translation_buffer({
            "start": "0.500",
            "end": "1.000",
            "text": "world",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:hello world")
