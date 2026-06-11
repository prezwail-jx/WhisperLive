import json
import queue
import unittest
from unittest import mock

import torch

from whisper_live.backend.translation_backend import HelsinkiZhEnTranslator, ServeClientTranslation


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


class PlaceholderFakeTokenizer:
    def __init__(self, decoded_text):
        self.decoded_text = decoded_text
        self.last_text = None

    def __call__(self, text, return_tensors=None, truncation=None):
        self.last_text = text
        return FakeTensorBatch(input_ids=[1, 2, 3])

    def batch_decode(self, generated_tokens, skip_special_tokens=True):
        return [self.decoded_text]


class TestHelsinkiZhEnMixedLanguageProtection(unittest.TestCase):
    def test_protects_english_terms_in_chinese_text(self):
        protected_text, terms = HelsinkiZhEnTranslator.protect_english_terms(
            "我现在用 Docker 跑 Whisper small，latency 比 medium 高"
        )

        self.assertIn("ZZX0ZZ", protected_text)
        self.assertIn("ZZX1ZZ", protected_text)
        self.assertIn("ZZX2ZZ", protected_text)
        self.assertIn("ZZX3ZZ", protected_text)
        self.assertEqual(terms["ZZX0ZZ"], "Docker")
        self.assertEqual(terms["ZZX1ZZ"], "Whisper small")
        self.assertEqual(terms["ZZX2ZZ"], "latency")
        self.assertEqual(terms["ZZX3ZZ"], "medium")

    def test_restore_handles_case_and_spaced_placeholder_variants(self):
        restored = HelsinkiZhEnTranslator.restore_english_terms(
            "Use zzx0zz, Z Z X 1 Z Z, XKETERM2X, and XKEPETERM3X.",
            {
                "ZZX0ZZ": "Docker",
                "ZZX1ZZ": "Whisper small",
                "XKEEPTERM2X": "CUDA",
                "XKEEPTERM3X": "TensorRT",
            },
        )

        self.assertEqual(restored, "Use Docker, Whisper small, CUDA, and TensorRT.")

    def test_natural_placeholders_protect_english_terms_in_chinese_text(self):
        protected_text, terms = HelsinkiZhEnTranslator.protect_english_terms_with_natural_placeholders(
            "我现在用 Docker 跑 Whisper small，latency 比 medium 高"
        )

        self.assertIn("第一个术语", protected_text)
        self.assertIn("第二个术语", protected_text)
        self.assertIn("第三个术语", protected_text)
        self.assertIn("第四个术语", protected_text)
        self.assertEqual(terms["第一个术语"], "Docker")
        self.assertEqual(terms["第二个术语"], "Whisper small")
        self.assertEqual(terms["第三个术语"], "latency")
        self.assertEqual(terms["第四个术语"], "medium")

    def test_restore_natural_placeholders_handles_common_translation_variants(self):
        restored = HelsinkiZhEnTranslator.restore_natural_term_placeholders(
            "Use the first word with word 2, third word, and the 4th word.",
            {
                "第一个术语": "Docker",
                "第二个术语": "ACE",
                "第三个术语": "CUDA",
                "第四个术语": "TensorRT",
            },
        )

        self.assertEqual(restored, "Use Docker with ACE, CUDA, and TensorRT.")

    def test_legacy_placeholder_fallback_works_with_natural_term_keys(self):
        restored = HelsinkiZhEnTranslator.restore_english_terms(
            "Use ZZX0ZZ, XKEPETERM1X, and XKETERM2X.",
            {
                "第一个术语": "Docker",
                "第二个术语": "ACE",
                "第三个术语": "CUDA",
            },
        )

        self.assertEqual(restored, "Use Docker, ACE, and CUDA.")

    def test_translate_restores_terms_only_for_zh_en(self):
        translator = HelsinkiZhEnTranslator()
        tokenizer = PlaceholderFakeTokenizer("Use the first term with term 2.")
        translator.tokenizers["zh-en"] = tokenizer
        translator.models["zh-en"] = FakeModel()

        translated, source_language, target_language = translator.translate(
            "我用 Docker 和 ACE",
            "zh",
            "en",
        )

        self.assertEqual(translated, "Use Docker with ACE.")
        self.assertEqual(source_language, "zh")
        self.assertEqual(target_language, "en")
        self.assertIn("第一个术语", tokenizer.last_text)
        self.assertIn("第二个术语", tokenizer.last_text)
        self.assertNotIn("Docker", tokenizer.last_text)
        self.assertNotIn("ZZX0ZZ", tokenizer.last_text)

    def test_pure_chinese_has_no_terms_to_protect(self):
        protected_text, terms = HelsinkiZhEnTranslator.protect_english_terms(
            "这个模型识别中文比较慢"
        )

        self.assertEqual(protected_text, "这个模型识别中文比较慢")
        self.assertEqual(terms, {})


class TestHelsinkiZhEnTranslatorDevice(unittest.TestCase):
    def test_cpu_device_is_explicit(self):
        translator = HelsinkiZhEnTranslator(device="cpu")

        self.assertEqual(translator.device, torch.device("cpu"))

    @mock.patch("whisper_live.backend.translation_backend.torch.cuda.is_available", return_value=True)
    def test_auto_device_keeps_existing_cuda_selection(self, mock_cuda_available):
        translator = HelsinkiZhEnTranslator(device="auto")

        self.assertEqual(translator.device, torch.device("cuda"))

    def test_invalid_device_raises(self):
        with self.assertRaises(ValueError):
            HelsinkiZhEnTranslator(device="mps")


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

    @mock.patch.object(HelsinkiZhEnTranslator, "load")
    def test_cache_key_distinguishes_translation_device(self, mock_load):
        client_cpu = ServeClientTranslation(
            client_uid="client-cpu",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
            translation_device="cpu",
        )
        client_cuda = ServeClientTranslation(
            client_uid="client-cuda",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
            translation_device="cuda",
        )
        client_auto = ServeClientTranslation(
            client_uid="client-auto",
            websocket=mock.Mock(),
            translation_queue=queue.Queue(),
            translation_device="auto",
        )

        self.assertIsNot(client_cpu.translator, client_cuda.translator)
        self.assertIsNot(client_cpu.translator, client_auto.translator)
        self.assertEqual(client_cpu.get_translation_cache_key()[-1], "cpu")
        self.assertEqual(client_cuda.get_translation_cache_key()[-1], "cuda")
        self.assertEqual(client_auto.get_translation_cache_key()[-1], "auto")
        self.assertEqual(mock_load.call_count, 3)


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

    def test_short_translation_buffer_flushes_after_timeout(self):
        client = self.make_client(
            translation_min_chars=12,
            translation_max_chars=100,
            translation_max_wait_seconds=1.0,
        )
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "0.500",
            "text": "And",
            "completed": True,
            "language": "en",
        })
        client.translation_buffer_started_at -= 2.0
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        self.assertEqual(payload["translated_segments"][0]["text"], "translated:And")
        self.assertEqual(client.translation_buffer, [])

    def test_standalone_english_interjection_uses_stable_translation(self):
        client = self.make_client()
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "0.500",
            "text": "OH!",
            "completed": True,
            "language": "en",
            "utterance_id": "client:1:0.000",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "哦")
        self.assertEqual(segment["source_language"], "en")
        self.assertEqual(segment["target_language"], "zh")
        self.assertEqual(segment["utterance_id"], "client:1:0.000")
        client.translate_text.assert_not_called()

    def test_standalone_english_fillers_use_stable_translations(self):
        expected_translations = {
            "uh": "呃",
            "Um.": "呃",
            "hmm...": "嗯",
            "Ah!": "啊",
        }
        for source_text, expected_translation in expected_translations.items():
            with self.subTest(source_text=source_text):
                client = self.make_client()
                client.add_segment_to_translation_buffer({
                    "start": "0.000",
                    "end": "0.500",
                    "text": source_text,
                    "completed": True,
                    "language": "en",
                })
                client.flush_translation_buffer(force=True)

                payload = self.get_last_payload(client)
                self.assertEqual(
                    payload["translated_segments"][0]["text"],
                    expected_translation,
                )
                client.translate_text.assert_not_called()

    def test_english_interjection_with_context_uses_translation_model(self):
        client = self.make_client()
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "Oh, I see.",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:Oh, I see.")
        client.translate_text.assert_called_once_with("Oh, I see.", "en")

    def test_meeting_glossary_overrides_builtin_interjection(self):
        client = self.make_client(translation_glossary={"oh": "噢"})
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "0.500",
            "text": "OH!",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        self.assertEqual(payload["translated_segments"][0]["text"], "噢")
        client.translate_text.assert_not_called()

    def test_glossary_uses_longest_phrase_and_restores_target(self):
        client = self.make_client(translation_glossary={
            "AI": "人工智能",
            "AI model": "指定模型",
        })
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "Use AI model.",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        self.assertEqual(
            payload["translated_segments"][0]["text"],
            "translated:Use 指定模型.",
        )
        protected_text = client.translate_text.call_args[0][0]
        self.assertIn("ZZGLOSSARY0ZZ", protected_text)
        self.assertNotIn("AI model", protected_text)

    def test_english_glossary_does_not_match_inside_word(self):
        client = self.make_client(translation_glossary={"AI": "人工智能"})
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "The SAIL project.",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        self.assertEqual(
            payload["translated_segments"][0]["text"],
            "translated:The SAIL project.",
        )
        client.translate_text.assert_called_once_with("The SAIL project.", "en")

    def test_glossary_marker_loss_falls_back_to_plain_translation(self):
        client = self.make_client(translation_glossary={"OpenAI": "开放人工智能"})
        client.translate_text = mock.Mock(side_effect=[
            ("标记已经丢失", "en", "zh"),
            ("普通整句翻译", "en", "zh"),
        ])
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "Use OpenAI.",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        self.assertEqual(payload["translated_segments"][0]["text"], "普通整句翻译")
        self.assertEqual(client.translate_text.call_count, 2)
        client.translate_text.assert_any_call("Use OpenAI.", "en")

    def test_glossary_exact_match_preserves_cpp_symbols(self):
        client = self.make_client(translation_glossary={"C++": "C Plus Plus"})
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "0.500",
            "text": "C++!",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        self.assertEqual(payload["translated_segments"][0]["text"], "C Plus Plus")
        client.translate_text.assert_not_called()

    def test_translation_preserves_single_utterance_id(self):
        client = self.make_client(translation_max_chars=4)
        for start, end, text in (("0.000", "0.500", "你好"), ("0.500", "1.000", "世界")):
            client.add_segment_to_translation_buffer({
                "start": start,
                "end": end,
                "text": text,
                "completed": True,
                "language": "zh",
                "utterance_id": "client:1:0.000",
            })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["utterance_id"], "client:1:0.000")
        self.assertEqual(segment["source_utterance_ids"], ["client:1:0.000"])

    def test_translation_preserves_multiple_source_utterance_ids(self):
        client = self.make_client(translation_max_chars=4)
        for start, end, text, utterance_id in (
            ("0.000", "0.500", "你好", "client:1:0.000"),
            ("0.500", "1.000", "世界", "client:2:0.500"),
        ):
            client.add_segment_to_translation_buffer({
                "start": start,
                "end": end,
                "text": text,
                "completed": True,
                "language": "zh",
                "utterance_id": utterance_id,
            })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(
            segment["source_utterance_ids"],
            ["client:1:0.000", "client:2:0.500"],
        )
        self.assertNotIn("utterance_id", segment)

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

    def test_english_translation_buffer_trims_boundary_overlap(self):
        client = self.make_client(translation_max_chars=80)
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "this is the end of the first sentence",
            "completed": True,
            "language": "en",
        })
        client.add_segment_to_translation_buffer({
            "start": "1.000",
            "end": "2.000",
            "text": "the first sentence starts cleanly now.",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer(force=True)

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(
            segment["text"],
            "translated:this is the end of the first sentence starts cleanly now.",
        )

    def test_english_translation_buffer_skips_fully_overlapped_segment(self):
        client = self.make_client(translation_max_chars=80)
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "thank you",
            "completed": True,
            "language": "en",
        })
        client.add_segment_to_translation_buffer({
            "start": "1.000",
            "end": "2.000",
            "text": "thank you",
            "completed": True,
            "language": "en",
        })
        client.flush_translation_buffer(force=True)

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:thank you")
