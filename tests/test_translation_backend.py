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
