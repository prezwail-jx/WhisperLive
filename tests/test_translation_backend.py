import json
import queue
import threading
import unittest
from unittest import mock

import torch

from whisper_live.backend.translation_backend import HelsinkiZhEnTranslator, NLLBTranslator, ServeClientTranslation


class FakeTensorBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    lang_code_to_id = {"eng_Latn": 1, "zho_Hans": 2}
    unk_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=None):
        return FakeTensorBatch(input_ids=[1, 2, 3])

    def convert_tokens_to_ids(self, token):
        return self.lang_code_to_id.get(token, self.unk_token_id)

    def batch_decode(self, generated_tokens, skip_special_tokens=True):
        return ["translated"]


class FakeModel:
    def __init__(self):
        self.last_generate_kwargs = None

    def to(self, device):
        return self

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
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

    def test_helsinki_generate_uses_realtime_length_limit(self):
        translator = HelsinkiZhEnTranslator()
        translator.tokenizers["en-zh"] = FakeTokenizer()
        model = FakeModel()
        translator.models["en-zh"] = model

        translated, source_language, target_language = translator.translate(
            "hello everyone",
            "en",
            "zh",
        )

        self.assertEqual(translated, "translated")
        self.assertEqual(source_language, "en")
        self.assertEqual(target_language, "zh")
        self.assertEqual(model.last_generate_kwargs["max_new_tokens"], HelsinkiZhEnTranslator.MAX_NEW_TOKENS)
        self.assertEqual(model.last_generate_kwargs["num_beams"], 1)

    def test_nllb_generate_uses_realtime_length_limit(self):
        translator = NLLBTranslator()
        translator.tokenizer = FakeTokenizer()
        translator.model = FakeModel()

        translated, source_language, target_language = translator.translate(
            "hello everyone",
            "en",
            "zh",
        )

        self.assertEqual(translated, "translated")
        self.assertEqual(source_language, "en")
        self.assertEqual(target_language, "zh")
        self.assertEqual(translator.model.last_generate_kwargs["max_new_tokens"], HelsinkiZhEnTranslator.MAX_NEW_TOKENS)
        self.assertEqual(translator.model.last_generate_kwargs["num_beams"], 1)
        self.assertEqual(translator.model.last_generate_kwargs["forced_bos_token_id"], 2)

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


class TestServeClientTranslationOutputGuard(unittest.TestCase):
    def make_client(self):
        with mock.patch.object(ServeClientTranslation, "load_translation_model"):
            client = ServeClientTranslation(
                client_uid="client-guard",
                websocket=mock.Mock(),
                translation_queue=queue.Queue(),
            )
        return client

    def test_guard_rejects_underscore_run(self):
        client = self.make_client()
        translated = "English Technology" + "_" * 30

        guarded = client.guard_translation_output("技术词", translated, "zh", "en")

        self.assertEqual(guarded, "技术词")
        self.assertEqual(
            ServeClientTranslation.translation_output_guard_reason("技术词", translated),
            "underscore_run",
        )

    def test_guard_rejects_too_long_output(self):
        client = self.make_client()
        translated = "word " * 50

        guarded = client.guard_translation_output("短句", translated, "zh", "en")

        self.assertEqual(guarded, "短句")
        self.assertEqual(
            ServeClientTranslation.translation_output_guard_reason("短句", translated),
            "length_ratio",
        )

    def test_guard_rejects_repetitive_output(self):
        client = self.make_client()
        translated = "yes " * 30

        guarded = client.guard_translation_output("对对对", translated, "zh", "en")

        self.assertEqual(guarded, "对对对")
        self.assertEqual(
            ServeClientTranslation.translation_output_guard_reason("对对对", translated),
            "low_unique_word_ratio",
        )

    def test_guard_allows_normal_output(self):
        client = self.make_client()

        guarded = client.guard_translation_output(
            "From my side, the main concern is stability.",
            "从我的角度来看，主要关注点是稳定性。",
            "en",
            "zh",
        )

        self.assertEqual(guarded, "从我的角度来看，主要关注点是稳定性。")
        self.assertIsNone(ServeClientTranslation.translation_output_guard_reason("hello", "你好"))


class TestServeClientTranslationNllbResidualRetry(unittest.TestCase):
    def make_client(self, model_name="nllb_200_600m", translations=None):
        with mock.patch.object(ServeClientTranslation, "load_translation_model"):
            client = ServeClientTranslation(
                client_uid="client-residual",
                websocket=mock.Mock(),
                translation_queue=queue.Queue(),
                model_name=model_name,
                translation_merge_enabled=False,
            )
        client.model_loaded = True
        client.translator = mock.Mock()
        client.translator.translate = mock.Mock(side_effect=translations or [])
        client.translator_lock = threading.Lock()
        return client

    def get_last_payload(self, client):
        payload = client.websocket.send.call_args[0][0]
        return json.loads(payload)

    def test_nllb_zh_to_en_retries_once_when_first_output_has_residual_cjk(self):
        client = self.make_client(translations=[
            ("创新中心联合集粹教育基金会特邀U型理论创始人", "zh", "en"),
            ("The Innovation Center invited the founder of Theory U.", "zh", "en"),
        ])

        translated, source_language, target_language = client.translate_text("创新中心特邀专家", "zh")

        self.assertEqual(translated, "The Innovation Center invited the founder of Theory U.")
        self.assertEqual(source_language, "zh")
        self.assertEqual(target_language, "en")
        self.assertIsNone(client.pending_translation_warning)
        self.assertEqual(client.translator.translate.call_count, 2)

    def test_nllb_zh_to_en_marks_warning_when_retry_still_has_residual_cjk(self):
        client = self.make_client(translations=[
            ("创新中心联合集粹教育基金会特邀U型理论创始人", "zh", "en"),
            ("创新中心仍然没有完成翻译", "zh", "en"),
        ])

        translated, source_language, target_language = client.translate_text("创新中心特邀专家", "zh")

        self.assertEqual(translated, "创新中心仍然没有完成翻译（翻译出错）")
        self.assertEqual(source_language, "zh")
        self.assertEqual(target_language, "en")
        self.assertEqual(client.pending_translation_warning, "residual_cjk")
        self.assertEqual(client.translator.translate.call_count, 2)

    def test_nllb_zh_to_en_flush_adds_warning_metadata_when_retry_still_fails(self):
        client = self.make_client(translations=[
            ("创新中心联合集粹教育基金会特邀U型理论创始人", "zh", "en"),
            ("创新中心仍然没有完成翻译", "zh", "en"),
        ])
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "创新中心特邀专家举办主题讲座。",
            "completed": True,
            "language": "zh",
        })

        client.flush_translation_buffer(force=True)

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "创新中心仍然没有完成翻译（翻译出错）")
        self.assertEqual(segment["translation_warning"], "residual_cjk")

    def test_helsinki_does_not_retry_residual_cjk(self):
        client = self.make_client(
            model_name="helsinki_zh_en",
            translations=[("创新中心联合集粹教育基金会特邀U型理论创始人", "zh", "en")],
        )

        translated, source_language, target_language = client.translate_text("创新中心特邀专家", "zh")

        self.assertEqual(translated, "创新中心联合集粹教育基金会特邀U型理论创始人")
        self.assertEqual(source_language, "zh")
        self.assertEqual(target_language, "en")
        self.assertIsNone(client.pending_translation_warning)
        self.assertEqual(client.translator.translate.call_count, 1)

    def test_nllb_1_3b_alias_uses_nllb_residual_retry(self):
        client = self.make_client(
            model_name="nllb_200_distilled_1_3b",
            translations=[
                ("创新中心联合集粹教育基金会特邀U型理论创始人", "zh", "en"),
                ("The Innovation Center invited the founder of Theory U.", "zh", "en"),
            ],
        )

        translated, source_language, target_language = client.translate_text("创新中心特邀专家", "zh")

        self.assertEqual(translated, "The Innovation Center invited the founder of Theory U.")
        self.assertEqual(source_language, "zh")
        self.assertEqual(target_language, "en")
        self.assertEqual(client.translator.translate.call_count, 2)

    def test_nllb_en_to_zh_does_not_retry_english_residual(self):
        client = self.make_client(translations=[("This remains English.", "en", "zh")])

        translated, source_language, target_language = client.translate_text("This should be translated.", "en")

        self.assertEqual(translated, "This remains English.")
        self.assertEqual(source_language, "en")
        self.assertEqual(target_language, "zh")
        self.assertIsNone(client.pending_translation_warning)
        self.assertEqual(client.translator.translate.call_count, 1)


class TestServeClientTranslationBuffer(unittest.TestCase):
    def make_client(self, **kwargs):
        kwargs.setdefault("translation_merge_enabled", False)
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
            "text": "明天见",
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
            "text": "今天开会。",
            "completed": True,
            "language": "zh",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["start"], "0.000")
        self.assertEqual(segment["end"], "1.000")
        self.assertEqual(segment["text"], "translated:今天开会。")
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
            "text": "明天见",
            "completed": True,
            "language": "zh",
        })
        client.translation_buffer_started_at -= 2.0
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:明天见")

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

    def test_glossary_applies_only_english_sources_for_en_to_zh(self):
        client = self.make_client(translation_glossary={
            "NICE": "长三角国家技术创新中心",
            "Liu Qing": "刘庆",
            "刘庆": "Liu Qing",
            "长三角国家技术创新中心": "NICE",
        })

        translated_text, source_language, target_language = client.translate_with_glossary(
            "NICE works with Liu Qing.",
            "en",
        )

        self.assertEqual(translated_text, "translated:长三角国家技术创新中心 works with 刘庆.")
        self.assertEqual(source_language, "en")
        self.assertEqual(target_language, "zh")

    def test_glossary_applies_only_chinese_sources_for_zh_to_en(self):
        client = self.make_client(translation_glossary={
            "NICE": "长三角国家技术创新中心",
            "Liu Qing": "刘庆",
            "刘庆": "Liu Qing",
            "长三角国家技术创新中心": "NICE",
        })

        translated_text, source_language, target_language = client.translate_with_glossary(
            "刘庆来自长三角国家技术创新中心",
            "zh",
        )

        self.assertEqual(translated_text, "translated:Liu Qing来自NICE")
        self.assertEqual(source_language, "zh")
        self.assertEqual(target_language, "en")

    def test_glossary_exact_match_respects_source_language_direction(self):
        client = self.make_client(translation_glossary={
            "NICE": "长三角国家技术创新中心",
            "长三角国家技术创新中心": "NICE",
        })

        self.assertIsNone(client.translate_with_glossary("NICE", "zh"))

    def test_glossary_exact_match_keeps_legacy_behavior_without_source_language(self):
        client = self.make_client(translation_glossary={"NICE": "长三角国家技术创新中心"})

        translated_text, source_language, target_language = client.translate_with_glossary("NICE", None)

        self.assertEqual(translated_text, "长三角国家技术创新中心")
        self.assertIsNone(source_language)
        self.assertEqual(target_language, "auto")
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

    def test_translation_merge_buffers_completed_translations_until_forced(self):
        client = self.make_client(
            translation_merge_enabled=True,
            translation_merge_max_chars=100,
            translation_merge_max_delay=60,
        )
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
            client.flush_translation_buffer(force=True)

        client.websocket.send.assert_not_called()
        client.flush_merge_buffer(force=True)

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["start"], "0.000")
        self.assertEqual(segment["end"], "1.000")
        self.assertEqual(segment["text"], "translated:你好translated:世界")
        self.assertEqual(segment["source_text"], "你好世界")
        self.assertEqual(segment["source_utterance_ids"], ["client:1:0.000", "client:2:0.500"])
        self.assertNotIn("utterance_id", segment)

    def test_translation_merge_flushes_on_gap(self):
        client = self.make_client(
            translation_merge_enabled=True,
            translation_merge_max_chars=100,
            translation_merge_max_delay=60,
            translation_merge_gap_seconds=0.5,
        )
        client.enqueue_translated_segment({
            "start": "0.000",
            "end": "0.500",
            "text": "hello",
            "completed": True,
            "source_text": "你好",
            "source_language": "zh",
            "target_language": "en",
            "translation_model": "helsinki_zh_en",
            "utterance_id": "client:1:0.000",
        })
        client.enqueue_translated_segment({
            "start": "2.000",
            "end": "2.500",
            "text": "world",
            "completed": True,
            "source_text": "世界",
            "source_language": "zh",
            "target_language": "en",
            "translation_model": "helsinki_zh_en",
            "utterance_id": "client:2:2.000",
        })

        payload = self.get_last_payload(client)
        self.assertEqual(payload["translated_segments"][0]["text"], "hello")
        self.assertEqual(len(client.translation_merge_buffer), 1)
        self.assertEqual(client.translation_merge_buffer[0]["text"], "world")

    def test_translation_merge_disabled_keeps_immediate_send(self):
        client = self.make_client(translation_merge_enabled=False)
        client.enqueue_translated_segment({
            "start": "0.000",
            "end": "0.500",
            "text": "hello",
            "completed": True,
            "source_text": "你好",
            "source_language": "zh",
            "target_language": "en",
            "translation_model": "helsinki_zh_en",
        })

        payload = self.get_last_payload(client)
        self.assertEqual(payload["translated_segments"][0]["text"], "hello")

    def test_auto_language_chinese_segment_is_inferred_for_auto_translation(self):
        client = self.make_client()
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "我们讨论预算。",
            "completed": True,
            "language": "auto",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:我们讨论预算。")
        self.assertEqual(segment["source_language"], "zh")
        self.assertEqual(segment["target_language"], "en")
        client.translate_text.assert_called_once_with("我们讨论预算。", "zh")

    def test_auto_language_chinese_segment_with_english_terms_stays_chinese(self):
        client = self.make_client()
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "我们 review revenue forecast。",
            "completed": True,
            "language": "auto",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:我们 review revenue forecast。")
        self.assertEqual(segment["source_language"], "zh")
        self.assertEqual(segment["target_language"], "en")
        client.translate_text.assert_called_once_with("我们 review revenue forecast。", "zh")

    def test_auto_language_english_segment_is_inferred_for_auto_translation(self):
        client = self.make_client()
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "1.000",
            "text": "The next topic is revenue.",
            "completed": True,
            "language": "auto",
        })
        client.flush_translation_buffer()

        payload = self.get_last_payload(client)
        segment = payload["translated_segments"][0]
        self.assertEqual(segment["text"], "translated:The next topic is revenue.")
        self.assertEqual(segment["source_language"], "en")
        self.assertEqual(segment["target_language"], "zh")
        client.translate_text.assert_called_once_with("The next topic is revenue.", "en")

    def test_auto_language_change_flushes_translation_buffer(self):
        client = self.make_client(translation_min_chars=100, translation_max_chars=100)
        client.add_segment_to_translation_buffer({
            "start": "0.000",
            "end": "0.500",
            "text": "我们讨论预算",
            "completed": True,
            "language": "auto",
        })
        client.add_segment_to_translation_buffer({
            "start": "0.500",
            "end": "1.000",
            "text": "The next topic is revenue.",
            "completed": True,
            "language": "auto",
        })
        client.flush_translation_buffer(force=True)

        self.assertEqual(client.translate_text.call_args_list[0].args, ("我们讨论预算", "zh"))
        self.assertEqual(client.translate_text.call_args_list[1].args, ("The next topic is revenue.", "en"))
        self.assertEqual(client.translated_segments[0]["source_language"], "zh")
        self.assertEqual(client.translated_segments[1]["source_language"], "en")

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

    def test_long_english_segment_is_split_for_realtime_translation(self):
        client = self.make_client()
        segment = {
            "start": 0.0,
            "end": 10.0,
            "text": " ".join(["word"] * 30),
            "completed": True,
            "language": "en",
        }

        split_segments = client.split_realtime_segment(segment)

        self.assertGreater(len(split_segments), 1)
        self.assertTrue(all(len(item["text"]) <= client._REALTIME_MAX_EN_CHARS for item in split_segments))
        self.assertEqual(split_segments[0]["start"], 0.0)
        self.assertEqual(split_segments[-1]["end"], 10.0)

    def test_translation_backlog_drops_old_segments_and_keeps_latest(self):
        client = self.make_client()
        first_segment = {
            "start": 0.0,
            "end": 1.0,
            "text": "first",
            "completed": True,
            "language": "en",
        }
        for index in range(1, 7):
            client.translation_queue.put({
                "start": float(index),
                "end": float(index + 1),
                "text": f"seg{index}",
                "completed": True,
                "language": "en",
            })

        kept, saw_exit_signal = client.drain_translation_backlog(first_segment)

        self.assertFalse(saw_exit_signal)
        self.assertEqual([segment["text"] for segment in kept], ["seg4", "seg5", "seg6"])
        self.assertEqual(client.translation_queue.qsize(), 0)
