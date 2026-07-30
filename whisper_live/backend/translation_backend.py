import json
import logging
import queue
import re
import threading
import time
import unicodedata
from typing import Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from whisper_live.backend.base import ServeClientBase


class HelsinkiZhEnTranslator:
    """Local zh<->en translator backed by two Marian/Helsinki models."""

    SUPPORTED_TARGETS = {"auto", "zh", "en"}
    MAX_NEW_TOKENS = 128
    ENGLISH_TERM_PATTERN = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:[A-Za-z][A-Za-z0-9+#._/-]*"
        r"(?:\s+[A-Za-z][A-Za-z0-9+#._/-]*){0,3})"
        r"(?![A-Za-z0-9])"
    )
    MIN_PROTECTED_ALPHA_CHARS = 2
    PLACEHOLDER_PREFIX = "ZZX"
    PLACEHOLDER_SUFFIX = "ZZ"
    NATURAL_TERM_PLACEHOLDERS = [
        "第一个术语",
        "第二个术语",
        "第三个术语",
        "第四个术语",
        "第五个术语",
        "第六个术语",
        "第七个术语",
        "第八个术语",
        "第九个术语",
        "第十个术语",
    ]
    ENGLISH_ORDINALS = [
        ("first", "1st", "one"),
        ("second", "2nd", "two"),
        ("third", "3rd", "three"),
        ("fourth", "4th", "four"),
        ("fifth", "5th", "five"),
        ("sixth", "6th", "six"),
        ("seventh", "7th", "seven"),
        ("eighth", "8th", "eight"),
        ("ninth", "9th", "nine"),
        ("tenth", "10th", "ten"),
    ]

    def __init__(
        self,
        zh_en_model_path="model/opus-mt-zh-en",
        en_zh_model_path="model/opus-mt-en-zh",
        device="cpu",
    ):
        self.zh_en_model_path = zh_en_model_path
        self.en_zh_model_path = en_zh_model_path
        self.device = self.resolve_device(device)
        self.models = {}
        self.tokenizers = {}

    @staticmethod
    def normalize_device_name(device):
        device = (device or "cpu").strip().lower()
        if device in ("cpu", "cuda", "auto"):
            return device
        if re.fullmatch(r"cuda:\d+", device):
            return device
        raise ValueError(f"Unsupported translation device: {device}")

    @classmethod
    def resolve_device(cls, device):
        device = cls.normalize_device_name(device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def load(self):
        logging.info(f"Loading Helsinki zh-en translation models on device: {self.device}")
        self.tokenizers["zh-en"] = AutoTokenizer.from_pretrained(self.zh_en_model_path)
        self.models["zh-en"] = AutoModelForSeq2SeqLM.from_pretrained(
            self.zh_en_model_path
        ).to(self.device)
        self.tokenizers["en-zh"] = AutoTokenizer.from_pretrained(self.en_zh_model_path)
        self.models["en-zh"] = AutoModelForSeq2SeqLM.from_pretrained(
            self.en_zh_model_path
        ).to(self.device)
        logging.info("Helsinki zh-en translation models loaded successfully")

    @staticmethod
    def normalize_language(language: Optional[str]) -> Optional[str]:
        if not language:
            return None
        language = language.lower().replace("_", "-")
        if language == "zh" or language.startswith("zh-"):
            return "zh"
        if language == "en" or language.startswith("en-"):
            return "en"
        return language

    def resolve_direction(self, source_language: Optional[str], target_language: str):
        source_language = self.normalize_language(source_language)
        target_language = self.normalize_language(target_language) or "auto"

        if target_language not in self.SUPPORTED_TARGETS:
            logging.warning(f"Unsupported target language for Helsinki zh-en translator: {target_language}")
            return None
        if source_language == "zh" and target_language in ("auto", "en"):
            return "zh-en", "en"
        if source_language == "en" and target_language in ("auto", "zh"):
            return "en-zh", "zh"
        if source_language in ("zh", "en") and target_language == source_language:
            return None

        logging.warning(f"Unsupported source language for Helsinki zh-en translator: {source_language}")
        return None

    @staticmethod
    def normalize_model_output(text: str) -> str:
        """Normalize known HTML entities emitted literally by translation models."""
        return str(text or "").replace("&amp;", "&")

    @classmethod
    def should_protect_english_term(cls, term: str) -> bool:
        alpha_chars = [char for char in term if char.isalpha()]
        return len(alpha_chars) >= cls.MIN_PROTECTED_ALPHA_CHARS

    @classmethod
    def make_placeholder(cls, index: int) -> str:
        return f"{cls.PLACEHOLDER_PREFIX}{index}{cls.PLACEHOLDER_SUFFIX}"

    @classmethod
    def make_natural_term_placeholder(cls, index: int) -> str:
        if index < len(cls.NATURAL_TERM_PLACEHOLDERS):
            return cls.NATURAL_TERM_PLACEHOLDERS[index]
        return f"第{index + 1}个术语"

    @classmethod
    def get_natural_term_index(cls, placeholder: str) -> Optional[int]:
        if placeholder in cls.NATURAL_TERM_PLACEHOLDERS:
            return cls.NATURAL_TERM_PLACEHOLDERS.index(placeholder)
        match = re.fullmatch(r"第(\d+)个术语", placeholder)
        if match:
            return max(0, int(match.group(1)) - 1)
        return None

    @staticmethod
    def get_placeholder_index(placeholder: str) -> Optional[str]:
        if placeholder.startswith("ZZX") and placeholder.endswith("ZZ"):
            return placeholder.removeprefix("ZZX").removesuffix("ZZ")
        if placeholder.startswith("XKEEPTERM") and placeholder.endswith("X"):
            return placeholder.removeprefix("XKEEPTERM").removesuffix("X")
        return None

    @classmethod
    def protect_english_terms_with_natural_placeholders(cls, text: str):
        protected_terms = {}

        def replace(match):
            term = match.group(0)
            if not cls.should_protect_english_term(term):
                return term

            placeholder = cls.make_natural_term_placeholder(len(protected_terms))
            protected_terms[placeholder] = term
            return placeholder

        return cls.ENGLISH_TERM_PATTERN.sub(replace, text), protected_terms

    @classmethod
    def restore_natural_term_placeholders(cls, text: str, protected_terms):
        restored_text = text
        for placeholder, term in protected_terms.items():
            index = cls.get_natural_term_index(placeholder)
            restored_text = re.sub(re.escape(placeholder), term, restored_text, flags=re.IGNORECASE)
            if index is None:
                continue

            numeric_index = str(index + 1)
            natural_index = re.escape(placeholder)
            patterns = [
                rf"the\s+{numeric_index}(?:st|nd|rd|th)?\s+term",
                rf"{numeric_index}(?:st|nd|rd|th)?\s+term",
                rf"term\s+{numeric_index}",
                rf"the\s+{numeric_index}(?:st|nd|rd|th)?\s+word",
                rf"{numeric_index}(?:st|nd|rd|th)?\s+word",
                rf"word\s+{numeric_index}",
                natural_index,
            ]

            if index < len(cls.ENGLISH_ORDINALS):
                ordinal, ordinal_number, word_number = cls.ENGLISH_ORDINALS[index]
                patterns.extend([
                    rf"the\s+{ordinal}\s+term",
                    rf"{ordinal}\s+term",
                    rf"the\s+{ordinal_number}\s+term",
                    rf"{ordinal_number}\s+term",
                    rf"term\s+{word_number}",
                    rf"the\s+{ordinal}\s+word",
                    rf"{ordinal}\s+word",
                    rf"the\s+{ordinal_number}\s+word",
                    rf"{ordinal_number}\s+word",
                    rf"word\s+{word_number}",
                ])

            for pattern in patterns:
                restored_text = re.sub(pattern, term, restored_text, flags=re.IGNORECASE)

        return restored_text

    @classmethod
    def has_unresolved_placeholders(cls, text: str) -> bool:
        return bool(re.search(
            r"ZZ\s*X\s*\d+\s*ZZ|"
            r"X\s*K\s*E+\s*P?\s*E?\s*T\s*E\s*R\s*M\s*\d+\s*X|"
            r"第\d+个术语|"
            r"第[一二三四五六七八九十]+个术语",
            text,
            re.IGNORECASE,
        ))

    @classmethod
    def protect_english_terms(cls, text: str):
        protected_terms = {}

        def replace(match):
            term = match.group(0)
            if not cls.should_protect_english_term(term):
                return term

            placeholder = cls.make_placeholder(len(protected_terms))
            protected_terms[placeholder] = term
            return placeholder

        return cls.ENGLISH_TERM_PATTERN.sub(replace, text), protected_terms

    @classmethod
    def restore_english_terms(cls, text: str, protected_terms):
        restored_text = text
        for placeholder, term in protected_terms.items():
            restored_text = re.sub(re.escape(placeholder), term, restored_text, flags=re.IGNORECASE)

            index = cls.get_placeholder_index(placeholder)
            if index is None:
                natural_index = cls.get_natural_term_index(placeholder)
                index = str(natural_index) if natural_index is not None else None
            if index is None:
                continue

            # New placeholder format, in case the model inserts spaces between characters.
            compact_placeholder = re.compile(
                rf"Z\s*Z\s*X\s*{re.escape(index)}\s*Z\s*Z",
                flags=re.IGNORECASE,
            )
            restored_text = compact_placeholder.sub(term, restored_text)

            # Backward compatibility for old placeholders and common model-corrupted variants:
            # XKEEPTERM0X, XKETERM0X, XKEPETERM0X, and spaced forms.
            legacy_placeholder = re.compile(
                rf"X\s*K\s*E+\s*P?\s*E?\s*T\s*E\s*R\s*M\s*{re.escape(index)}\s*X",
                flags=re.IGNORECASE,
            )
            restored_text = legacy_placeholder.sub(term, restored_text)

        if cls.has_unresolved_placeholders(restored_text):
            logging.warning("[MIXED_LANG_PROTECT][WARN] unresolved placeholder in translated text")
        return restored_text

    def translate(self, text: str, source_language: Optional[str], target_language: str, generation_profile=None):
        direction = self.resolve_direction(source_language, target_language)
        if direction is None:
            return text, self.normalize_language(source_language), self.normalize_language(target_language)

        model_key, resolved_target_language = direction
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]
        protected_terms = {}
        text_to_translate = text

        if model_key == "zh-en":
            text_to_translate, protected_terms = self.protect_english_terms_with_natural_placeholders(text)
            if protected_terms:
                logging.info(
                    "[MIXED_LANG_PROTECT] direction=zh-en natural_terms=%d text_len=%d",
                    len(protected_terms),
                    len(text),
                )

        encoded_input = tokenizer(text_to_translate, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded_input,
                max_new_tokens=self.MAX_NEW_TOKENS,
                num_beams=3,
            )
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_text = output[0] if output else text
        if protected_terms:
            translated_text = self.restore_natural_term_placeholders(translated_text, protected_terms)
            translated_text = self.restore_english_terms(translated_text, protected_terms)
        translated_text = self.normalize_model_output(translated_text)
        return (
            translated_text,
            self.normalize_language(source_language),
            resolved_target_language,
        )

    def cleanup(self):
        self.models.clear()
        self.tokenizers.clear()


class NLLBTranslator(HelsinkiZhEnTranslator):
    """Local zh<->en translator backed by NLLB-200 distilled 600M."""

    MAX_NEW_TOKENS = 256

    LANGUAGE_CODES = {
        "zh": "zho_Hans",
        "en": "eng_Latn",
    }

    def __init__(self, model_path="model/NLLB-200-600M", device="cpu"):
        self.model_path = model_path
        self.device = self.resolve_device(device)
        self.tokenizer = None
        self.model = None

    def load(self):
        logging.info("Loading NLLB translation model: %s on device: %s", self.model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, local_files_only=True).to(self.device)
        logging.info("NLLB translation model loaded successfully")

    def resolve_direction(self, source_language: Optional[str], target_language: str):
        source_language = self.normalize_language(source_language)
        target_language = self.normalize_language(target_language) or "auto"

        if target_language not in self.SUPPORTED_TARGETS:
            logging.warning("Unsupported target language for NLLB translator: %s", target_language)
            return None
        if source_language == "zh" and target_language in ("auto", "en"):
            return "zh", "en"
        if source_language == "en" and target_language in ("auto", "zh"):
            return "en", "zh"
        if source_language in ("zh", "en") and target_language == source_language:
            return None

        logging.warning("Unsupported source language for NLLB translator: %s", source_language)
        return None

    def language_token_id(self, language_code: str) -> int:
        if hasattr(self.tokenizer, "lang_code_to_id"):
            token_id = self.tokenizer.lang_code_to_id.get(language_code)
        else:
            token_id = self.tokenizer.convert_tokens_to_ids(language_code)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"NLLB language token not found: {language_code}")
        return token_id

    def generation_kwargs(self, profile=None):
        if profile == "relaxed":
            return {
                "max_new_tokens": 320,
                "num_beams": 3,
                "length_penalty": 1.1,
            }
        return {
            "max_new_tokens": self.MAX_NEW_TOKENS,
            "num_beams": 3,
        }

    def translate(self, text: str, source_language: Optional[str], target_language: str, generation_profile=None):
        direction = self.resolve_direction(source_language, target_language)
        if direction is None:
            return text, self.normalize_language(source_language), self.normalize_language(target_language)

        source_language, resolved_target_language = direction
        source_code = self.LANGUAGE_CODES[source_language]
        target_code = self.LANGUAGE_CODES[resolved_target_language]

        protected_terms = {}
        text_to_translate = text
        if source_language == "zh":
            text_to_translate, protected_terms = self.protect_english_terms_with_natural_placeholders(text)
            if protected_terms:
                logging.info(
                    "[NLLB_MIXED_LANG_PROTECT] direction=zh-en natural_terms=%d text_len=%d",
                    len(protected_terms),
                    len(text),
                )

        self.tokenizer.src_lang = source_code
        encoded_input = self.tokenizer(text_to_translate, return_tensors="pt", truncation=True).to(self.device)
        forced_bos_token_id = self.language_token_id(target_code)
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded_input,
                forced_bos_token_id=forced_bos_token_id,
                **self.generation_kwargs(generation_profile),
            )
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_text = output[0] if output else text
        if protected_terms:
            translated_text = self.restore_natural_term_placeholders(translated_text, protected_terms)
            translated_text = self.restore_english_terms(translated_text, protected_terms)
        translated_text = self.normalize_model_output(translated_text)
        return translated_text, source_language, resolved_target_language

    def translate_batch(self, items, generation_profile=None):
        results = [None] * len(items)
        grouped = {}

        for index, item in enumerate(items):
            text = str(item.get("text") or "")
            source_language = item.get("source_language")
            target_language = item.get("target_language")
            direction = self.resolve_direction(source_language, target_language)
            if direction is None:
                results[index] = (
                    text,
                    self.normalize_language(source_language),
                    self.normalize_language(target_language),
                )
                continue

            resolved_source, resolved_target = direction
            protected_terms = {}
            text_to_translate = text
            if resolved_source == "zh":
                text_to_translate, protected_terms = self.protect_english_terms_with_natural_placeholders(text)
                if protected_terms:
                    logging.info(
                        "[NLLB_MIXED_LANG_PROTECT] direction=zh-en natural_terms=%d text_len=%d",
                        len(protected_terms),
                        len(text),
                    )

            grouped.setdefault((resolved_source, resolved_target), []).append({
                "index": index,
                "text": text,
                "text_to_translate": text_to_translate,
                "protected_terms": protected_terms,
            })

        for (source_language, target_language), group in grouped.items():
            source_code = self.LANGUAGE_CODES[source_language]
            target_code = self.LANGUAGE_CODES[target_language]
            self.tokenizer.src_lang = source_code
            encoded_input = self.tokenizer(
                [entry["text_to_translate"] for entry in group],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            forced_bos_token_id = self.language_token_id(target_code)
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded_input,
                    forced_bos_token_id=forced_bos_token_id,
                    **self.generation_kwargs(generation_profile),
                )
            outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for entry, translated_text in zip(group, outputs):
                protected_terms = entry["protected_terms"]
                if protected_terms:
                    translated_text = self.restore_natural_term_placeholders(translated_text, protected_terms)
                    translated_text = self.restore_english_terms(translated_text, protected_terms)
                translated_text = self.normalize_model_output(translated_text)
                results[entry["index"]] = (translated_text, source_language, target_language)

        return results

    def cleanup(self):
        self.model = None
        self.tokenizer = None


class NLLBTranslationBatchWorker:
    def __init__(self, translator, max_batch_size=8, batch_window_ms=40):
        self.translator = translator
        self.max_batch_size = max(1, int(max_batch_size or 8))
        self.batch_window_seconds = max(0.0, float(batch_window_ms or 0) / 1000.0)
        self.requests = queue.Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, text, source_language, target_language, client_uid, timeout_seconds=10.0, generation_profile=None):
        request = {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
            "generation_profile": generation_profile,
            "client_uid": client_uid,
            "event": threading.Event(),
            "result": None,
            "error": None,
        }
        self.requests.put(request)
        if not request["event"].wait(max(0.1, float(timeout_seconds or 10.0))):
            raise TimeoutError("NLLB batch translation timed out")
        if request["error"] is not None:
            raise request["error"]
        return request["result"]

    def run(self):
        while True:
            first = self.requests.get()
            batch = [first]
            deadline = time.monotonic() + self.batch_window_seconds
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.requests.get(timeout=remaining))
                except queue.Empty:
                    break

            started_at = time.monotonic()
            try:
                profiles = {request.get("generation_profile") for request in batch}
                if len(profiles) == 1:
                    results = self.translator.translate_batch(batch, generation_profile=profiles.pop())
                else:
                    results = [
                        self.translator.translate_batch([request], generation_profile=request.get("generation_profile"))[0]
                        for request in batch
                    ]
                elapsed_ms = (time.monotonic() - started_at) * 1000.0
                logging.info(
                    "[NLLB_BATCH] model=%s device=%s batch_size=%d generate_ms=%.1f",
                    getattr(self.translator, "model_path", ""),
                    getattr(self.translator, "device", ""),
                    len(batch),
                    elapsed_ms,
                )
                for request, result in zip(batch, results):
                    request["result"] = result
                    request["event"].set()
            except Exception as error:
                logging.exception("[NLLB_BATCH_FAILED] batch_size=%d", len(batch))
                for request in batch:
                    request["error"] = error
                    request["event"].set()


class ServeClientTranslation(ServeClientBase):
    """
    Handles translation of completed transcription segments in a separate thread.
    Reads from a queue populated by the transcription backend and sends translated
    segments back to the client via WebSocket.
    """
    _TRANSLATOR_CACHE = {}
    _TRANSLATOR_INFERENCE_LOCKS = {}
    _TRANSLATOR_BATCH_WORKERS = {}
    _TRANSLATOR_CACHE_LOCK = threading.Lock()
    _TRANSLATION_DRAFT_WAKEUP = object()
    _TRANSLATION_DRAIN_SENTINEL = object()
    _READABILITY_BOUNDARY_MARKER = "ZZREADABILITYBOUNDARYZZ"
    _STANDALONE_ENGLISH_INTERJECTIONS = {
        "oh": "哦",
        "uh": "呃",
        "um": "呃",
        "er": "呃",
        "hm": "嗯",
        "hmm": "嗯",
        "mm": "嗯",
        "ah": "啊",
    }
    _FIXED_SHORT_ZH_TRANSLATIONS = {
        "大家好": "Hello everyone.",
        "你好": "Hello.",
        "对": "Yes.",
        "好的": "Okay.",
        "谢谢": "Thank you.",
    }
    _SHORT_ZH_BUFFER_CJK_CHARS = 5
    _SHORT_ZH_BUFFER_WAIT_SECONDS = 3.5
    _ZH_EN_SENTENCE_BUFFER_IDLE_SECONDS = 1.2
    _ZH_EN_SENTENCE_BUFFER_MAX_AUDIO_SECONDS = 8.0
    _ZH_EN_SENTENCE_BUFFER_MAX_GAP_SECONDS = 1.0
    _OUTPUT_GUARD_MAX_LENGTH_RATIO = 4.0
    _OUTPUT_GUARD_MIN_LONG_OUTPUT_CHARS = 160
    _OUTPUT_GUARD_ZH_EN_MAX_LENGTH_RATIO = 6.0
    _OUTPUT_GUARD_ZH_EN_MIN_LONG_OUTPUT_CHARS = 240
    _OUTPUT_GUARD_UNDERSCORE_RUN = 20
    _OUTPUT_GUARD_MIN_REPEAT_WORDS = 24
    _OUTPUT_GUARD_MAX_UNIQUE_WORD_RATIO = 0.28
    _OUTPUT_GUARD_MAX_REPEATED_BIGRAM_COUNT = 8
    _BACKLOG_DROP_THRESHOLD = 5
    _BACKLOG_KEEP_LATEST = 3
    _REALTIME_MAX_EN_CHARS = 280
    _REALTIME_MAX_ZH_CHARS = 120
    _RESIDUAL_CJK_MIN_CHARS = 8
    _RESIDUAL_CJK_MAX_RATIO = 0.25
    _TRANSLATION_UNAVAILABLE_TEXT = "翻译暂不可用"
    _TRANSLATION_UNAVAILABLE_TEXT_EN = "Translation unavailable"
    _SOURCE_ECHO_PROPER_TERM_MAX_CHARS = 32
    _SOURCE_ECHO_PROPER_TERM_MAX_TOKENS = 4
    _UNDERTRANSLATION_MIN_EN_WORDS = 12
    _ZH_EN_UNDERTRANSLATION_MIN_CJK_CHARS = 12
    _ZH_EN_UNDERTRANSLATION_TARGET_WORD_PERCENT = 30
    _UNDERTRANSLATION_TARGET_UNITS_PERCENT = 100
    _UNDERTRANSLATION_REPAIR_SECONDS = 3.0
    _UNDERTRANSLATION_MAX_CHUNKS = 3
    _UNDERTRANSLATION_MIN_CHUNK_WORDS = 6
    _TRANSLATION_COMPARISON_PUNCTUATION = " \t\r\n.,!?;:，。！？；：\"'“”‘’()[]{}"
    _INCOMPLETE_EN_ENDING_WORDS = {
        "a", "an", "and", "are", "as", "at", "because", "but", "by", "for",
        "from", "had", "has", "have", "if", "in", "into", "is", "it", "of",
        "on", "or", "our", "that", "the", "their", "then", "these", "this",
        "those", "to", "was", "were", "when", "where", "which", "while",
        "with", "without", "would",
    }
    _INCOMPLETE_EN_ENDING_PHRASES = {
        "i had", "i have", "it was", "there are", "there is", "we had",
        "we have", "we were", "you can", "you know", "i started",
        "this is how", "i want really", "you have huge", "from the",
        "and they", "but they", "and we", "but we", "assuming you",
        "we can", "they can", "we would", "they would", "we need to",
        "they need to",
    }
    _INCOMPLETE_ZH_ENDING_PHRASES = tuple(sorted({
        "那么", "以及", "并且", "而且", "从而", "但是", "但", "而",
        "我们认为", "我认为", "具体来说", "主要包括", "包括",
        "如果", "因为", "由于", "虽然", "为了", "基于",
        "一方面", "另一方面", "不仅", "不但", "无论", "对于", "关于", "至于",
        "就是", "在于", "意味着", "的话", "为导向", "当中",
    }, key=len, reverse=True))
    _INCOMPLETE_ZH_TRAILING_PUNCTUATION = " \t\r\n，,、：:；;。！？.!?\"'“”‘’）)]}】》"
    _INCOMPLETE_ZH_PAUSE_PUNCTUATION = ("，", ",", "、", "：", ":", "；", ";")

    def __init__(
        self,
        client_uid,
        websocket,
        translation_queue,
        target_language="auto",
        send_last_n_segments=10,
        model_name="helsinki_zh_en",
        zh_en_model_path="model/opus-mt-zh-en",
        en_zh_model_path="model/opus-mt-en-zh",
        nllb_model_path="model/NLLB-200-600M",
        translation_device="cpu",
        translation_min_chars=12,
        translation_max_chars=220,
        translation_max_wait_seconds=3.0,
        translation_incomplete_max_wait_seconds=None,
        translation_context_seconds=0.0,
        translation_sentence_endings="。！？.!?",
        translation_glossary=None,
        translation_terms=None,
        translation_mode="standard",
        translation_merge_enabled=True,
        translation_merge_max_chars=240,
        translation_merge_max_delay=1.8,
        translation_merge_gap_seconds=1.0,
        nllb_batch_translation=False,
        nllb_batch_max_size=8,
        nllb_batch_window_ms=40,
        nllb_batch_timeout_seconds=10.0,
        service_mode="standard",
        source_language=None,
        translation_draft_enabled=False,
        translation_readability_context_enabled=False,
        translation_draft_interval_seconds=1.2,
        translation_draft_min_delta_chars=8,
        translation_draft_max_source_chars=220,
        translation_readability_context_sentences=0,
        translation_readability_context_max_chars=0,
        translation_zh_en_sentence_buffer_enabled=True,
        translation_zh_en_idle_seconds=_ZH_EN_SENTENCE_BUFFER_IDLE_SECONDS,
        translation_zh_en_max_audio_seconds=_ZH_EN_SENTENCE_BUFFER_MAX_AUDIO_SECONDS,
        translation_zh_en_max_gap_seconds=_ZH_EN_SENTENCE_BUFFER_MAX_GAP_SECONDS,
    ):
        """
        Initialize the translation client.
        Args:
            client_uid (str): Unique identifier for the client
            websocket: WebSocket connection to the client
            translation_queue (queue.Queue): Queue containing completed segments to translate
            target_language (str): Target language code or "auto" for zh<->en
            send_last_n_segments (int): Number of recent translated segments to send
            model_name (str): Translation model name to use
        """
        super().__init__(client_uid, websocket, send_last_n_segments)
        self.translation_queue = translation_queue
        self.target_language = target_language
        self.model_name = model_name
        self.zh_en_model_path = zh_en_model_path
        self.en_zh_model_path = en_zh_model_path
        self.nllb_model_path = nllb_model_path
        self.translation_device = HelsinkiZhEnTranslator.normalize_device_name(translation_device)
        self.translation_min_chars = max(0, int(translation_min_chars or 0))
        self.translation_max_chars = max(1, int(translation_max_chars or 220))
        self.translation_max_wait_seconds = max(0.1, float(translation_max_wait_seconds or 3.0))
        if translation_incomplete_max_wait_seconds is None:
            translation_incomplete_max_wait_seconds = self.translation_max_wait_seconds
        self.translation_incomplete_max_wait_seconds = max(
            0.1,
            float(translation_incomplete_max_wait_seconds or self.translation_max_wait_seconds),
        )
        self.translation_context_seconds = max(0.0, float(translation_context_seconds or 0.0))
        self.translation_sentence_endings = translation_sentence_endings
        self.translation_glossary = self.normalize_translation_glossary(translation_glossary)
        self._translation_glossary_sources_cache = {}
        self.translation_terms = list(translation_terms or [])
        self.translation_mode = str(translation_mode or "standard")
        self.translation_merge_enabled = bool(translation_merge_enabled)
        self.translation_merge_max_chars = max(1, int(translation_merge_max_chars or 240))
        self.translation_merge_max_delay = max(0.1, float(translation_merge_max_delay or 1.8))
        self.translation_merge_gap_seconds = max(0.0, float(translation_merge_gap_seconds or 1.0))
        self.nllb_batch_translation = bool(nllb_batch_translation)
        self.nllb_batch_max_size = max(1, int(nllb_batch_max_size or 8))
        self.nllb_batch_window_ms = max(0, int(nllb_batch_window_ms or 0))
        self.nllb_batch_timeout_seconds = max(0.1, float(nllb_batch_timeout_seconds or 10.0))
        self.service_mode = str(service_mode or "standard")
        self.source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        self.translation_draft_enabled = bool(translation_draft_enabled)
        self.translation_readability_context_enabled = bool(translation_readability_context_enabled)
        self.translation_draft_interval_seconds = min(10.0, max(0.5, float(translation_draft_interval_seconds or 1.2)))
        self.translation_draft_min_delta_chars = min(220, max(1, int(translation_draft_min_delta_chars or 8)))
        self.translation_draft_max_source_chars = min(220, max(32, int(translation_draft_max_source_chars or 220)))
        self.translation_readability_context_sentences = min(2, max(0, int(translation_readability_context_sentences or 0)))
        self.translation_readability_context_max_chars = min(220, max(0, int(translation_readability_context_max_chars or 0)))
        if not self.translation_readability_context_enabled or self.translation_readability_context_sentences == 0:
            self.translation_readability_context_sentences = 0
            self.translation_readability_context_max_chars = 0
        self.translation_zh_en_sentence_buffer_enabled = bool(translation_zh_en_sentence_buffer_enabled)
        self.translation_zh_en_idle_seconds = max(0.1, float(translation_zh_en_idle_seconds or self._ZH_EN_SENTENCE_BUFFER_IDLE_SECONDS))
        self.translation_zh_en_max_audio_seconds = max(0.1, float(translation_zh_en_max_audio_seconds or self._ZH_EN_SENTENCE_BUFFER_MAX_AUDIO_SECONDS))
        self.translation_zh_en_max_gap_seconds = max(0.0, float(translation_zh_en_max_gap_seconds or self._ZH_EN_SENTENCE_BUFFER_MAX_GAP_SECONDS))
        self.translation_buffer = []
        self.translation_buffer_started_at = None
        self.translation_buffer_last_added_at = None
        self.translation_buffer_last_source_activity_at = None
        self.translation_merge_buffer = []
        self.translation_merge_started_at = None
        self.translated_segments = []
        self.last_translated_source_text = ""
        self.pending_translation_warning = None
        self.pending_translation_confidence = None
        self.draft_state_lock = threading.Lock()
        self.draft_states = {}
        self.draft_inference_active = False
        self.draft_wakeup_queued = False
        self.last_draft_inference_finished_at = None
        self.readability_context_lock = threading.Lock()
        self.readability_context_history = []
        self.readability_context_direction = None
        self.final_translation_lock = threading.Lock()
        self.pending_final_segments = {}
        self.timed_out_final_keys = set()
        self.translation_drain_completed = threading.Event()
        self.translation_drain_status = None
        self.translation_timeout_count = 0
        self.translator = None
        self.translator_lock = None
        self.batch_worker = None
        self.model_loaded = False
        self.model_load_failure_reason = None
        logging.info(
            "[TRANSLATION_CONFIG] uid=%s model=%s mode=%s context_seconds=%.2f "
            "max_wait=%.2f incomplete_wait=%.2f max_chars=%d merge=%s",
            self.client_uid,
            self.model_name,
            self.translation_mode,
            self.translation_context_seconds,
            self.translation_max_wait_seconds,
            self.translation_incomplete_max_wait_seconds,
            self.translation_max_chars,
            self.translation_merge_enabled,
        )
        self.load_translation_model()

    def readability_context_eligible(self, source_language, target_language=None):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = (
            HelsinkiZhEnTranslator.normalize_language(target_language)
            or self._resolved_target_language(source_language)
        )
        return bool(
            self.service_mode == "accurate"
            and self.translation_readability_context_enabled
            and self.translation_readability_context_sentences > 0
            and self.translation_readability_context_max_chars > 0
            and (source_language, target_language) in (("en", "zh"), ("zh", "en"))
        )

    @staticmethod
    def _trim_readability_context_suffix(text, max_chars):
        value = str(text or "").strip()
        if max_chars <= 0 or len(value) <= max_chars:
            return value
        start = len(value) - max_chars
        suffix = value[start:]
        if start > 0 and value[start - 1].isalnum() and suffix[:1].isalnum():
            boundary = re.search(r"\s+", suffix)
            if boundary and boundary.end() < len(suffix):
                suffix = suffix[boundary.end():]
        return suffix.strip() or value[-max_chars:].strip()

    def readability_context_snapshot(self, source_language, target_language=None):
        if not self.readability_context_eligible(source_language, target_language):
            return []
        direction = self.readability_context_direction_key(source_language, target_language)
        with self.readability_context_lock:
            if self.readability_context_direction != direction:
                return []
            return [unit.copy() for unit in self.readability_context_history]

    def readability_context_direction_key(self, source_language, target_language=None):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = (
            HelsinkiZhEnTranslator.normalize_language(target_language)
            or self._resolved_target_language(source_language)
        )
        return (source_language, target_language)

    def build_readability_context_input(self, current_text, source_language, target_language=None):
        current_text = str(current_text or "").strip()
        history = self.readability_context_snapshot(source_language, target_language)
        if not current_text or not history:
            return None, []
        marker = self._READABILITY_BOUNDARY_MARKER
        if marker.casefold() in current_text.casefold():
            return None, []
        sources = [
            str(unit.get("source_text") or "").strip()
            for unit in history
            if str(unit.get("source_text") or "").strip()
        ]
        if not sources:
            return None, []
        context_text = " ".join(sources)
        context_text = self._trim_readability_context_suffix(
            context_text,
            self.translation_readability_context_max_chars,
        )
        if not context_text or marker.casefold() in context_text.casefold():
            return None, []
        return f"{context_text}\n{marker}\n{current_text}", history

    @classmethod
    def _readability_boundary_pattern(cls):
        return re.compile(
            r"\s*".join(re.escape(char) for char in cls._READABILITY_BOUNDARY_MARKER),
            re.IGNORECASE,
        )

    def extract_readability_current_translation(
        self,
        translated_text,
        current_source_text,
        source_language,
        target_language,
        history,
    ):
        translated_text = str(translated_text or "")
        matches = list(self._readability_boundary_pattern().finditer(translated_text))
        if len(matches) != 1:
            return None, "boundary_missing" if not matches else "boundary_ambiguous"
        current_translation = translated_text[matches[0].end():].strip()
        if not current_translation:
            return None, "current_output_empty"

        normalized_current = self.normalize_translation_comparison_text(current_translation)
        for unit in reversed(history):
            previous_translation = self.normalize_translation_comparison_text(unit.get("text"))
            if (
                len(previous_translation) >= 6
                and normalized_current.startswith(previous_translation)
            ):
                return None, "history_leak"

        failure_reason = self.translation_output_failure_reason(
            current_source_text,
            current_translation,
            source_language,
            target_language,
        )
        if failure_reason:
            return None, failure_reason
        return current_translation, None

    def infer_translation_once_without_final_state(
        self,
        text,
        source_language,
        allow_batch=False,
        generation_profile=None,
    ):
        started_at = time.monotonic()
        source_language, target_language = self._resolved_failure_languages(source_language)
        if not self.model_loaded or not self.translator or not self.translator_lock:
            return (
                None,
                source_language,
                target_language,
                self.model_load_failure_reason or "model_unavailable",
                0.0,
            )
        if self.exit:
            return None, source_language, target_language, "client_exit", 0.0

        if allow_batch and self.should_use_nllb_batch_translation() and self.batch_worker:
            try:
                translated_text, source_language, target_language = self.batch_worker.submit(
                    text,
                    source_language,
                    self.target_language,
                    self.client_uid,
                    timeout_seconds=self.nllb_batch_timeout_seconds,
                    generation_profile=generation_profile,
                )
            except Exception as error:
                return (
                    None,
                    source_language,
                    target_language,
                    self.translation_exception_reason(error),
                    (time.monotonic() - started_at) * 1000.0,
                )
            return (
                translated_text,
                source_language,
                target_language,
                None,
                (time.monotonic() - started_at) * 1000.0,
            )

        try:
            with self.translator_lock:
                translated_text, source_language, target_language = self.translator.translate(
                    text,
                    source_language,
                    self.target_language,
                    generation_profile=generation_profile,
                )
        except Exception as error:
            return (
                None,
                source_language,
                target_language,
                self.translation_exception_reason(error),
                (time.monotonic() - started_at) * 1000.0,
            )
        return (
            translated_text,
            source_language,
            target_language,
            None,
            (time.monotonic() - started_at) * 1000.0,
        )

    def translate_with_readability_context_once(
        self,
        current_text,
        source_language,
        allow_batch,
        path,
    ):
        contextual_input, history = self.build_readability_context_input(
            current_text,
            source_language,
        )
        if contextual_input is None:
            return None, None, None, "context_unavailable", False

        context_chars = len(contextual_input) - len(str(current_text or "")) - len(
            self._READABILITY_BOUNDARY_MARKER
        ) - 2
        logging.info(
            "[TRANSLATION_CONTEXT_USED] uid=%s path=%s history_units=%d "
            "context_chars=%d current_chars=%d",
            self.client_uid,
            path,
            len(history),
            max(0, context_chars),
            len(str(current_text or "")),
        )
        translated_text, resolved_source, target_language, inference_reason, elapsed_ms = (
            self.infer_translation_once_without_final_state(
                contextual_input,
                source_language,
                allow_batch=allow_batch,
            )
        )
        if inference_reason:
            logging.info(
                "[TRANSLATION_CONTEXT_EXTRACTION_FAILED] uid=%s path=%s reason=%s "
                "history_units=%d current_chars=%d elapsed_ms=%.1f",
                self.client_uid,
                path,
                inference_reason,
                len(history),
                len(str(current_text or "")),
                elapsed_ms,
            )
            return None, resolved_source, target_language, inference_reason, True

        current_translation, extraction_reason = self.extract_readability_current_translation(
            translated_text,
            current_text,
            resolved_source,
            target_language,
            history,
        )
        if extraction_reason is None and path == "final":
            extraction_reason = self.contextual_undertranslation_reason(
                current_text,
                current_translation,
                resolved_source,
                target_language,
            )
        if extraction_reason:
            logging.info(
                "[TRANSLATION_CONTEXT_EXTRACTION_FAILED] uid=%s path=%s reason=%s "
                "history_units=%d current_chars=%d output_chars=%d elapsed_ms=%.1f",
                self.client_uid,
                path,
                extraction_reason,
                len(history),
                len(str(current_text or "")),
                len(str(translated_text or "")),
                elapsed_ms,
            )
            return None, resolved_source, target_language, extraction_reason, True
        return current_translation, resolved_source, target_language, None, True

    def record_readability_context(
        self,
        source_text,
        translated_text,
        source_language,
        target_language,
        translation_warning=None,
    ):
        if not self.readability_context_eligible(source_language, target_language):
            return False
        direction = self.readability_context_direction_key(source_language, target_language)
        source_text = str(source_text or "").strip()
        translated_text = str(translated_text or "").strip()
        if (
            not source_text
            or not translated_text
            or translation_warning
            or translated_text in (self._TRANSLATION_UNAVAILABLE_TEXT, self._TRANSLATION_UNAVAILABLE_TEXT_EN)
            or self.normalize_translation_comparison_text(source_text)
            == self.normalize_translation_comparison_text(translated_text)
            or self.translation_output_failure_reason(
                source_text,
                translated_text,
                source_language,
                target_language,
            )
        ):
            return False

        unit = {
            "source_text": source_text,
            "text": translated_text,
        }
        with self.readability_context_lock:
            if self.readability_context_direction != direction:
                if self.readability_context_history:
                    logging.info(
                        "[TRANSLATION_CONTEXT_DIRECTION_RESET] uid=%s previous=%s current=%s cleared=%d",
                        self.client_uid,
                        self.readability_context_direction,
                        direction,
                        len(self.readability_context_history),
                    )
                self.readability_context_direction = direction
                self.readability_context_history.clear()
            self.readability_context_history.append(unit)
            self.readability_context_history = self.readability_context_history[
                -self.translation_readability_context_sentences:
            ]
            history_units = len(self.readability_context_history)
        logging.info(
            "[TRANSLATION_CONTEXT_HISTORY_UPDATED] uid=%s history_units=%d "
            "source_chars=%d translated_chars=%d",
            self.client_uid,
            history_units,
            len(source_text),
            len(translated_text),
        )
        return True

    @staticmethod
    def _translation_draft_change_chars(previous_text, current_text):
        previous = str(previous_text or "")
        current = str(current_text or "")
        prefix_length = 0
        for previous_char, current_char in zip(previous, current):
            if previous_char != current_char:
                break
            prefix_length += 1
        return max(len(previous) - prefix_length, len(current) - prefix_length)

    @staticmethod
    def _trim_translation_draft_suffix(text, max_chars):
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        start = len(value) - max_chars
        suffix = value[start:]
        if start > 0 and value[start - 1].isalnum() and suffix[:1].isalnum():
            boundary = re.search(r"\s+", suffix)
            if boundary and boundary.end() < len(suffix):
                suffix = suffix[boundary.end():]
        return suffix.strip() or value[-max_chars:].strip()

    def _translation_draft_direction_eligible(self, segment):
        if not self.translation_draft_enabled or self.service_mode != "accurate":
            return False
        source_language = HelsinkiZhEnTranslator.normalize_language(
            segment.get("language") or self.source_language
        )
        if source_language != "en" or self._resolved_target_language(source_language) != "zh":
            return False
        return self.infer_text_language(segment.get("text")) == "en"

    def _queue_translation_draft_wakeup_locked(self):
        if self.draft_wakeup_queued or self.exit:
            return
        try:
            self.translation_queue.put_nowait(self._TRANSLATION_DRAFT_WAKEUP)
            self.draft_wakeup_queued = True
        except queue.Full:
            return

    def _clear_translation_draft_wakeup(self):
        with self.draft_state_lock:
            self.draft_wakeup_queued = False

    def observe_asr_segment(self, segment):
        utterance_id = str(segment.get("utterance_id") or "").strip()
        if self.exit:
            return

        now = time.monotonic()
        source_language = self.get_segment_source_language(segment)
        if (
            not segment.get("completed", False)
            and self.zh_en_sentence_buffer_applies(source_language)
        ):
            self.translation_buffer_last_source_activity_at = now

        if not utterance_id or not self.translation_draft_enabled:
            return
        if segment.get("completed", False):
            with self.draft_state_lock:
                state = self.draft_states.get(utterance_id)
                if state is None:
                    return
                state["revision"] += 1
                state["finalized"] = True
                state["pending"] = False
                state["updated_at"] = now
                logging.info(
                    "[TRANSLATION_DRAFT_FINALIZED] uid=%s utterance_id=%s revision=%d in_flight=%s",
                    self.client_uid,
                    utterance_id,
                    state["revision"],
                    str(bool(state["in_flight"])).lower(),
                )
                if not state["in_flight"]:
                    self.draft_states.pop(utterance_id, None)
            return

        text = str(segment.get("text") or "").strip()
        if not text or not re.search(r"[A-Za-z]", text):
            return
        if not self._translation_draft_direction_eligible(segment):
            return
        with self.draft_state_lock:
            state = self.draft_states.get(utterance_id)
            if state is None:
                state = {
                    "utterance_id": utterance_id,
                    "latest_source_text": "",
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "source_language": "en",
                    "revision": 0,
                    "finalized": False,
                    "last_translated_draft_text": "",
                    "last_inference_at": None,
                    "pending": False,
                    "in_flight": False,
                    "updated_at": now,
                }
                self.draft_states[utterance_id] = state
            if state["finalized"] or state["latest_source_text"] == text:
                return
            state["revision"] += 1
            was_pending = state["pending"]
            state["latest_source_text"] = text
            state["start"] = segment.get("start")
            state["end"] = segment.get("end")
            state["source_language"] = "en"
            state["updated_at"] = now
            state["pending"] = self._translation_draft_change_chars(
                state["last_translated_draft_text"],
                text,
            ) >= self.translation_draft_min_delta_chars
            if state["pending"]:
                self._queue_translation_draft_wakeup_locked()
                log_method = logging.debug if was_pending else logging.info
                log_method(
                    "[TRANSLATION_DRAFT_%s] uid=%s utterance_id=%s revision=%d source_chars=%d",
                    "COALESCED" if was_pending else "SCHEDULED",
                    self.client_uid,
                    utterance_id,
                    state["revision"],
                    len(text),
                )

    def translation_draft_wait_timeout(self, default=1.0):
        timeout = default
        if self.translation_buffer and self.zh_en_sentence_buffer_applies(self.get_buffer_source_language()):
            last_activity = self.translation_buffer_last_source_activity_at or self.translation_buffer_last_added_at
            if last_activity is not None:
                remaining = last_activity + self.translation_zh_en_idle_seconds - time.monotonic()
                timeout = min(timeout, max(0.05, remaining))
        if not self.translation_draft_enabled:
            return timeout
        with self.draft_state_lock:
            has_pending = any(
                state["pending"] and not state["finalized"]
                for state in self.draft_states.values()
            )
            if not has_pending or self.last_draft_inference_finished_at is None:
                return timeout
            remaining = (
                self.last_draft_inference_finished_at
                + self.translation_draft_interval_seconds
                - time.monotonic()
            )
        return min(timeout, max(0.05, remaining))

    def claim_ready_translation_draft(self, now=None):
        if (
            not self.translation_draft_enabled
            or self.exit
            or self.translation_buffer
            or not self.translation_queue.empty()
        ):
            return None
        now = time.monotonic() if now is None else float(now)
        with self.draft_state_lock:
            if (
                self.draft_inference_active
                or self.translation_buffer
                or not self.translation_queue.empty()
            ):
                return None
            if (
                self.last_draft_inference_finished_at is not None
                and now - self.last_draft_inference_finished_at < self.translation_draft_interval_seconds
            ):
                return None
            candidates = [
                state for state in self.draft_states.values()
                if state["pending"] and not state["in_flight"] and not state["finalized"]
            ]
            if not candidates:
                return None
            state = max(candidates, key=lambda item: item["updated_at"])
            state["pending"] = False
            state["in_flight"] = True
            state["last_inference_at"] = now
            self.draft_inference_active = True
            return {
                "utterance_id": state["utterance_id"],
                "revision": state["revision"],
                "source_text": self._trim_translation_draft_suffix(
                    state["latest_source_text"],
                    self.translation_draft_max_source_chars,
                ),
                "observed_source_text": state["latest_source_text"],
                "start": state["start"],
                "end": state["end"],
                "source_language": state["source_language"],
                "started_at": now,
            }

    def finish_translation_draft_claim(self, claim, succeeded=False, finished_at=None):
        if not claim:
            return
        finished_at = time.monotonic() if finished_at is None else float(finished_at)
        utterance_id = claim.get("utterance_id")
        revision = claim.get("revision")
        observed_source_text = str(
            claim.get("observed_source_text") or claim.get("source_text") or ""
        )
        with self.draft_state_lock:
            self.draft_inference_active = False
            self.last_draft_inference_finished_at = finished_at
            state = self.draft_states.get(utterance_id)
            if state is None:
                return
            state["in_flight"] = False
            if state["finalized"]:
                self.draft_states.pop(utterance_id, None)
                return
            if state["revision"] == revision:
                if succeeded:
                    state["last_translated_draft_text"] = observed_source_text
                state["pending"] = False
                return
            state["pending"] = self._translation_draft_change_chars(
                state["last_translated_draft_text"],
                state["latest_source_text"],
            ) >= self.translation_draft_min_delta_chars
            if state["pending"]:
                self._queue_translation_draft_wakeup_locked()

    def translation_draft_claim_is_current(self, claim):
        if not claim or self.exit or not self.translation_draft_enabled:
            return False
        if self.service_mode != "accurate" or self._resolved_target_language("en") != "zh":
            return False
        with self.draft_state_lock:
            state = self.draft_states.get(claim.get("utterance_id"))
            return bool(
                state
                and state["in_flight"]
                and not state["finalized"]
                and state["revision"] == claim.get("revision")
            )

    def _prepare_translation_draft_glossary(self, text, source_language):
        if not self.translation_glossary:
            return None, text, []
        eligible_sources = self.glossary_sources_for_language(source_language)
        normalized_text = self._normalize_glossary_lookup_text(text)
        for source in eligible_sources:
            if self._normalize_glossary_lookup_text(source) == normalized_text:
                return self.translation_glossary[source], text, []
        ordered_sources = sorted(eligible_sources, key=len, reverse=True)
        if not ordered_sources:
            return None, text, []
        pattern = re.compile(
            "|".join(self._glossary_term_pattern(source) for source in ordered_sources),
            re.IGNORECASE,
        )
        replacements = []

        def protect(match):
            matched_source = match.group(0)
            target = next(
                self.translation_glossary[source]
                for source in ordered_sources
                if source.casefold() == matched_source.casefold()
            )
            marker = f"ZZGLOSSARY{len(replacements)}ZZ"
            replacements.append((marker, target))
            return marker

        return None, pattern.sub(protect, text), replacements

    def _restore_translation_draft_glossary(self, translated_text, replacements):
        restored_text = str(translated_text or "")
        for index, (_, target) in enumerate(replacements):
            marker_pattern = self._glossary_marker_pattern(index)
            if not marker_pattern.search(restored_text):
                return None
            restored_text = marker_pattern.sub(lambda _: target, restored_text)
        return restored_text

    def translate_draft_text(self, source_text):
        started_at = time.monotonic()
        source_language = "en"
        target_language = self._resolved_target_language(source_language)
        exact_glossary, inference_text, replacements = self._prepare_translation_draft_glossary(
            source_text,
            source_language,
        )
        if exact_glossary is not None:
            translated_text = exact_glossary
        else:
            shortcut = None
            if not replacements:
                shortcut = self.translate_standalone_interjection(
                    source_text,
                    source_language,
                    target_language,
                )
            if shortcut is not None:
                translated_text, source_language, target_language = shortcut
            else:
                translated_text = None
                (
                    contextual_text,
                    contextual_source,
                    contextual_target,
                    contextual_reason,
                    used_context,
                ) = self.translate_with_readability_context_once(
                    inference_text,
                    source_language,
                    allow_batch=False,
                    path="draft",
                )
                if used_context and contextual_reason is None:
                    translated_text = contextual_text
                    source_language = contextual_source
                    target_language = contextual_target
                elif used_context:
                    logging.info(
                        "[TRANSLATION_CONTEXT_FALLBACK] uid=%s path=draft reason=%s "
                        "current_chars=%d",
                        self.client_uid,
                        contextual_reason,
                        len(str(inference_text or "")),
                    )

                if translated_text is None:
                    (
                        translated_text,
                        source_language,
                        target_language,
                        inference_reason,
                        _,
                    ) = self.infer_translation_once_without_final_state(
                        inference_text,
                        source_language,
                        allow_batch=False,
                    )
                    if inference_reason:
                        return (
                            None,
                            inference_reason,
                            (time.monotonic() - started_at) * 1000.0,
                        )

                if replacements:
                    translated_text = self._restore_translation_draft_glossary(
                        translated_text,
                        replacements,
                    )
                    if translated_text is None:
                        return (
                            None,
                            "glossary_marker_missing",
                            (time.monotonic() - started_at) * 1000.0,
                        )
                logging.debug(
                    "[TRANSLATION_DRAFT_INFERENCE] uid=%s source_chars=%d output_chars=%d "
                    "context_used=%s elapsed_ms=%.1f",
                    self.client_uid,
                    len(source_text),
                    len(str(translated_text or "")),
                    str(bool(used_context)).lower(),
                    (time.monotonic() - started_at) * 1000.0,
                )

        failure_reason = self.translation_output_failure_reason(
            source_text,
            translated_text,
            source_language,
            target_language,
        )
        elapsed_ms = (time.monotonic() - started_at) * 1000.0
        if failure_reason:
            return None, failure_reason, elapsed_ms
        return {
            "text": str(translated_text).strip(),
            "source_language": source_language,
            "target_language": target_language,
        }, None, elapsed_ms

    def send_translation_draft_to_client(self, translated_segment):
        segment = translated_segment
        if getattr(self, "segment_post_processor", None) is not None:
            try:
                processed = self.segment_post_processor(translated_segment.copy())
                segment = processed if processed is not None else translated_segment
            except Exception as error:
                logging.error(
                    "[TRANSLATION_DRAFT_POST_PROCESSOR_ERROR] uid=%s utterance_id=%s error=%s",
                    self.client_uid,
                    translated_segment.get("utterance_id"),
                    str(error)[:160],
                )
        try:
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "translated_segments": [segment],
            }))
            return True
        except Exception as error:
            logging.error(
                "[TRANSLATION_DRAFT_SEND_ERROR] uid=%s utterance_id=%s error=%s",
                self.client_uid,
                translated_segment.get("utterance_id"),
                str(error)[:160],
            )
            self.exit = True
            return False

    def process_ready_translation_draft(self):
        claim = self.claim_ready_translation_draft()
        if claim is None:
            return False
        emitted = False
        try:
            result, failure_reason, elapsed_ms = self.translate_draft_text(claim["source_text"])
            if result is None:
                logging.info(
                    "[TRANSLATION_DRAFT_INVALID_DROPPED] uid=%s utterance_id=%s revision=%d "
                    "source_chars=%d reason=%s elapsed_ms=%.1f",
                    self.client_uid,
                    claim["utterance_id"],
                    claim["revision"],
                    len(claim["source_text"]),
                    failure_reason,
                    elapsed_ms,
                )
                return False
            if not self.translation_draft_claim_is_current(claim):
                logging.info(
                    "[TRANSLATION_DRAFT_STALE_DROPPED] uid=%s utterance_id=%s revision=%d "
                    "source_chars=%d output_chars=%d elapsed_ms=%.1f",
                    self.client_uid,
                    claim["utterance_id"],
                    claim["revision"],
                    len(claim["source_text"]),
                    len(result["text"]),
                    elapsed_ms,
                )
                return False
            translated_segment = {
                "start": claim.get("start"),
                "end": claim.get("end"),
                "text": result["text"],
                "completed": False,
                "translation_id": f"draft:{claim['utterance_id']}",
                "revision": claim["revision"],
                "utterance_id": claim["utterance_id"],
                "source_utterance_ids": [claim["utterance_id"]],
                "source_text": claim["source_text"],
                "source_language": result["source_language"],
                "target_language": result["target_language"],
                "translation_model": self.model_name,
            }
            emitted = self.send_translation_draft_to_client(translated_segment)
            if emitted:
                logging.info(
                    "[TRANSLATION_DRAFT_EMITTED] uid=%s utterance_id=%s revision=%d "
                    "source_chars=%d output_chars=%d elapsed_ms=%.1f",
                    self.client_uid,
                    claim["utterance_id"],
                    claim["revision"],
                    len(claim["source_text"]),
                    len(result["text"]),
                    elapsed_ms,
                )
            return emitted
        finally:
            self.finish_translation_draft_claim(claim, succeeded=emitted)

    def get_translation_cache_key(self):
        """Build the process-local cache key for the configured translation model."""
        return (
            self.model_name,
            self.zh_en_model_path,
            self.en_zh_model_path,
            self.nllb_model_path,
            self.translation_device,
        )

    @staticmethod
    def is_nllb_model(model_name):
        return str(model_name or "") in {"nllb_200_600m", "nllb", "nllb_200_distilled_1_3b", "nllb_200_1_3b", "nllb_200_3_3b"}

    def load_translation_model(self):
        """Load the translation model and tokenizer."""
        try:
            self.model_load_failure_reason = None
            cache_key = self.get_translation_cache_key()
            with self._TRANSLATOR_CACHE_LOCK:
                if cache_key not in self._TRANSLATOR_CACHE:
                    if self.model_name == "helsinki_zh_en":
                        translator = HelsinkiZhEnTranslator(
                            zh_en_model_path=self.zh_en_model_path,
                            en_zh_model_path=self.en_zh_model_path,
                            device=self.translation_device,
                        )
                    elif self.is_nllb_model(self.model_name):
                        translator = NLLBTranslator(
                            model_path=self.nllb_model_path,
                            device=self.translation_device,
                        )
                    else:
                        raise ValueError(f"Unsupported translation model provider: {self.model_name}")
                    translator.load()
                    self._TRANSLATOR_CACHE[cache_key] = translator
                    self._TRANSLATOR_INFERENCE_LOCKS[cache_key] = threading.Lock()

                self.translator = self._TRANSLATOR_CACHE[cache_key]
                self.translator_lock = self._TRANSLATOR_INFERENCE_LOCKS[cache_key]
                if self.should_use_nllb_batch_translation():
                    worker_key = (cache_key, self.nllb_batch_max_size, self.nllb_batch_window_ms)
                    if worker_key not in self._TRANSLATOR_BATCH_WORKERS:
                        self._TRANSLATOR_BATCH_WORKERS[worker_key] = NLLBTranslationBatchWorker(
                            self.translator,
                            max_batch_size=self.nllb_batch_max_size,
                            batch_window_ms=self.nllb_batch_window_ms,
                        )
                    self.batch_worker = self._TRANSLATOR_BATCH_WORKERS[worker_key]
            self.model_loaded = True
            logging.info(
                "Translation model loaded successfully. Provider: %s Target language: %s",
                self.model_name,
                self.target_language,
            )
        except Exception as e:
            logging.error(f"Failed to load translation model: {e}")
            self.translator = None
            self.translator_lock = None
            self.model_loaded = False
            reason = self.translation_exception_reason(e)
            self.model_load_failure_reason = reason if reason != "translation_exception" else "model_unavailable"

    def should_use_nllb_batch_translation(self):
        return self.nllb_batch_translation and self.is_nllb_model(self.model_name)

    @staticmethod
    def translation_exception_reason(error):
        if isinstance(error, TimeoutError):
            return "translation_timeout"
        message = str(error or "").lower()
        cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        if (cuda_oom_type and isinstance(error, cuda_oom_type)) or "out of memory" in message:
            return "cuda_oom"
        if isinstance(error, ValueError) and "unsupported translation model" in message:
            return "unsupported_model"
        return "translation_exception"

    def _resolved_failure_languages(self, source_language, target_language=None):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language) or source_language
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language)
        return source_language, target_language or self._resolved_target_language(source_language)

    def translation_unavailable_text(self, target_language=None):
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language)
        if target_language == "en":
            return self._TRANSLATION_UNAVAILABLE_TEXT_EN
        return self._TRANSLATION_UNAVAILABLE_TEXT

    def _log_translation_retry(
        self,
        text,
        translated_text,
        source_language,
        target_language,
        reason,
        path,
        attempt,
        started_at,
        fallback=None,
    ):
        logging.warning(
            "[TRANSLATION_OUTPUT_RETRY] uid=%s model=%s path=%s attempt=%d reason=%s fallback=%s "
            "source_language=%s target_language=%s source_len=%d translated_len=%d elapsed_ms=%.1f "
            "source_preview=%r translated_preview=%r",
            self.client_uid,
            self.model_name,
            path,
            attempt,
            reason,
            fallback or "same_path",
            source_language,
            target_language,
            len(str(text or "")),
            len(str(translated_text or "")),
            (time.monotonic() - started_at) * 1000.0,
            str(text or "").strip()[:120],
            str(translated_text or "").strip()[:160],
        )

    def _finalize_translation_failure(
        self,
        text,
        source_language,
        target_language,
        initial_reason,
        final_reason,
        attempts,
        path,
        started_at,
        translated_text=None,
        error=None,
    ):
        source_language, target_language = self._resolved_failure_languages(source_language, target_language)
        self.pending_translation_warning = final_reason
        logging.warning(
            "[TRANSLATION_OUTPUT_FAILED] uid=%s model=%s path=%s attempts=%d initial_reason=%s "
            "final_reason=%s source_language=%s target_language=%s source_len=%d translated_len=%d "
            "elapsed_ms=%.1f error_type=%s error_preview=%r source_preview=%r translated_preview=%r",
            self.client_uid,
            self.model_name,
            path,
            attempts,
            initial_reason or final_reason,
            final_reason,
            source_language,
            target_language,
            len(str(text or "")),
            len(str(translated_text or "")),
            (time.monotonic() - started_at) * 1000.0,
            type(error).__name__ if error is not None else "none",
            str(error or "")[:160],
            str(text or "").strip()[:120],
            str(translated_text or "").strip()[:160],
        )
        return self.translation_unavailable_text(target_language), source_language, target_language

    def translate_text_with_batch(self, text: str, source_language: Optional[str]):
        if not self.batch_worker:
            return None
        started_at = time.monotonic()
        generate_ms = 0.0
        initial_reason = None
        resolved_source_language, target_language = self._resolved_failure_languages(source_language)
        for attempt in (1, 2):
            attempt_started_at = time.monotonic()
            try:
                translated_text, resolved_source_language, target_language = self.batch_worker.submit(
                    text,
                    source_language,
                    self.target_language,
                    self.client_uid,
                    timeout_seconds=self.nllb_batch_timeout_seconds,
                )
            except Exception as error:
                generate_ms += (time.monotonic() - attempt_started_at) * 1000.0
                reason = self.translation_exception_reason(error)
                initial_reason = initial_reason or reason
                if reason == "translation_timeout" and attempt == 1:
                    self._log_translation_retry(
                        text,
                        "",
                        resolved_source_language,
                        target_language,
                        reason,
                        "batch",
                        attempt,
                        started_at,
                        fallback="direct",
                    )
                    raise
                return self._finalize_translation_failure(
                    text,
                    resolved_source_language,
                    target_language,
                    initial_reason,
                    reason,
                    attempt,
                    "batch",
                    started_at,
                    error=error,
                )

            generate_ms += (time.monotonic() - attempt_started_at) * 1000.0
            reason = self.translation_output_failure_reason(
                text,
                translated_text,
                resolved_source_language,
                target_language,
            )
            if not reason:
                self._log_translation_latency(
                    text,
                    source_language,
                    started_at,
                    generate_ms,
                    0.0,
                    attempt,
                    batch=True,
                )
                return translated_text, resolved_source_language, target_language

            initial_reason = initial_reason or reason
            if attempt == 1:
                self._log_translation_retry(
                    text,
                    translated_text,
                    resolved_source_language,
                    target_language,
                    reason,
                    "batch",
                    attempt,
                    started_at,
                )
                continue
            return self._finalize_translation_failure(
                text,
                resolved_source_language,
                target_language,
                initial_reason,
                reason,
                attempt,
                "batch",
                started_at,
                translated_text=translated_text,
            )

    def _log_translation_latency(
        self,
        text,
        source_language,
        started_at,
        generate_ms,
        lock_wait_ms,
        attempts,
        batch=False,
    ):
        logging.info(
            "[TRANSLATION_LATENCY] uid=%s model=%s device=%s source_language=%s source_len=%d "
            "lock_wait_ms=%.1f generate_ms=%.1f total_ms=%.1f queue_size=%s batch=%s attempts=%d "
            "source_preview=%r",
            self.client_uid,
            self.model_name,
            self.translation_device,
            source_language,
            len(str(text or "")),
            lock_wait_ms,
            generate_ms,
            (time.monotonic() - started_at) * 1000.0,
            self.translation_queue_size(),
            str(bool(batch)).lower(),
            attempts,
            str(text or "").strip()[:80],
        )

    def _translate_text_direct(
        self,
        text,
        source_language,
        max_attempts=2,
        initial_reason=None,
        attempt_offset=0,
        started_at=None,
        generation_profile=None,
    ):
        started_at = started_at or time.monotonic()
        generate_ms = 0.0
        lock_wait_ms = 0.0
        resolved_source_language, target_language = self._resolved_failure_languages(source_language)
        for attempt in range(1, max_attempts + 1):
            lock_started_at = time.monotonic()
            self.translator_lock.acquire()
            lock_wait_ms += (time.monotonic() - lock_started_at) * 1000.0
            attempt_started_at = time.monotonic()
            try:
                translated_text, resolved_source_language, target_language = self.translator.translate(
                    text,
                    source_language,
                    self.target_language,
                    generation_profile=generation_profile,
                )
            except Exception as error:
                generate_ms += (time.monotonic() - attempt_started_at) * 1000.0
                reason = self.translation_exception_reason(error)
                initial_reason = initial_reason or reason
                if reason == "translation_timeout" and attempt < max_attempts:
                    self._log_translation_retry(
                        text,
                        "",
                        resolved_source_language,
                        target_language,
                        reason,
                        "direct",
                        attempt + attempt_offset,
                        started_at,
                    )
                    continue
                return self._finalize_translation_failure(
                    text,
                    resolved_source_language,
                    target_language,
                    initial_reason,
                    reason,
                    attempt + attempt_offset,
                    "direct",
                    started_at,
                    error=error,
                )
            finally:
                self.translator_lock.release()

            generate_ms += (time.monotonic() - attempt_started_at) * 1000.0
            reason = self.translation_output_failure_reason(
                text,
                translated_text,
                resolved_source_language,
                target_language,
            )
            if not reason:
                self._log_translation_latency(
                    text,
                    source_language,
                    started_at,
                    generate_ms,
                    lock_wait_ms,
                    attempt + attempt_offset,
                )
                return translated_text, resolved_source_language, target_language

            initial_reason = initial_reason or reason
            if attempt < max_attempts:
                self._log_translation_retry(
                    text,
                    translated_text,
                    resolved_source_language,
                    target_language,
                    reason,
                    "direct",
                    attempt + attempt_offset,
                    started_at,
                )
                continue
            return self._finalize_translation_failure(
                text,
                resolved_source_language,
                target_language,
                initial_reason,
                reason,
                attempt + attempt_offset,
                "direct",
                started_at,
                translated_text=translated_text,
            )

    def _translate_text_current_only(self, text: str, source_language: Optional[str]):
        """
        Translate a single text segment.

        Args:
            text (str): Text to translate

        Returns:
            str: Translated text or a user-safe unavailable placeholder on failure
        """
        self.pending_translation_warning = None
        self.pending_translation_confidence = None
        if not text.strip():
            return text, source_language, self.target_language

        started_at = time.monotonic()
        if not self.model_loaded or not self.translator or not self.translator_lock:
            reason = self.model_load_failure_reason or "model_unavailable"
            return self._finalize_translation_failure(
                text,
                source_language,
                None,
                reason,
                reason,
                0,
                "none",
                started_at,
            )

        if self.exit:
            return self._finalize_translation_failure(
                text,
                source_language,
                None,
                "client_exit",
                "client_exit",
                0,
                "none",
                started_at,
            )

        if self.should_use_nllb_batch_translation() and self.batch_worker:
            try:
                return self.translate_text_with_batch(text, source_language)
            except TimeoutError:
                return self._translate_text_direct(
                    text,
                    source_language,
                    max_attempts=1,
                    initial_reason="translation_timeout",
                    attempt_offset=1,
                    started_at=started_at,
                )

        return self._translate_text_direct(text, source_language, started_at=started_at)

    def translate_text(self, text: str, source_language: Optional[str]):
        self.pending_translation_warning = None
        self.pending_translation_confidence = None
        if (
            not str(text or "").strip()
            or not self.readability_context_eligible(source_language)
            or not self.readability_context_snapshot(source_language)
        ):
            translated_text, resolved_source, target_language = self._translate_text_current_only(text, source_language)
            return self.repair_en_zh_undertranslation_if_needed(text, translated_text, resolved_source, target_language)

        translated_text, resolved_source, target_language, failure_reason, used_context = (
            self.translate_with_readability_context_once(
                text,
                source_language,
                allow_batch=True,
                path="final",
            )
        )
        if used_context and failure_reason is None:
            risk_reason = self.zh_en_context_risk_reason(text, translated_text, resolved_source, target_language)
            if risk_reason:
                logging.info(
                    "[TRANSLATION_CONTEXT_FALLBACK] uid=%s path=final reason=%s current_chars=%d output_chars=%d",
                    self.client_uid,
                    risk_reason,
                    len(str(text or "")),
                    len(str(translated_text or "")),
                )
                direct_text, direct_source, direct_target = self._translate_text_current_only(text, source_language)
                direct_warning = self.pending_translation_warning
                if direct_warning:
                    logging.warning(
                        "[TRANSLATION_CONTEXT_DIRECT_FAILED] uid=%s reason=%s current_chars=%d output_chars=%d",
                        self.client_uid,
                        direct_warning,
                        len(str(text or "")),
                        len(str(direct_text or "")),
                    )
                else:
                    logging.info(
                        "[TRANSLATION_CONTEXT_DIRECT_USED] uid=%s reason=%s current_chars=%d output_chars=%d",
                        self.client_uid,
                        risk_reason,
                        len(str(text or "")),
                        len(str(direct_text or "")),
                    )
                return self.repair_en_zh_undertranslation_if_needed(text, direct_text, direct_source, direct_target)
            return self.repair_en_zh_undertranslation_if_needed(text, translated_text, resolved_source, target_language)

        if used_context:
            logging.info(
                "[TRANSLATION_CONTEXT_FALLBACK] uid=%s path=final reason=%s current_chars=%d",
                self.client_uid,
                failure_reason,
                len(str(text or "")),
            )
        translated_text, resolved_source, target_language = self._translate_text_current_only(text, source_language)
        return self.repair_en_zh_undertranslation_if_needed(text, translated_text, resolved_source, target_language)

    def en_zh_undertranslation_reason(self, source_text, translated_text, source_language, target_language):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language)
        if (
            self.service_mode != "accurate"
            or not self.is_nllb_model(self.model_name)
            or source_language != "en"
            or target_language != "zh"
        ):
            return None
        source_words = len(self._word_spans(source_text))
        if source_words < self._UNDERTRANSLATION_MIN_EN_WORDS:
            return None
        target_units = self._count_cjk(translated_text) + len(self._word_spans(translated_text))
        if target_units * 100 < source_words * self._UNDERTRANSLATION_TARGET_UNITS_PERCENT:
            return "undertranslation"
        return None

    @classmethod
    def _normalized_anchor_text(cls, text):
        return unicodedata.normalize("NFKC", str(text or "")).casefold()

    @classmethod
    def _numeric_anchors(cls, text):
        return [match.group(0).strip().casefold() for match in re.finditer(r"\d+(?:[.,]\d+)?\s*%?", str(text or ""))]

    def fact_anchors(self, source_text, source_language, target_language):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language)
        source = str(source_text or "")
        anchors = []
        for number in self._numeric_anchors(source):
            anchors.append(("number", number, [number.replace(",", ""), number.replace(".", "")]))

        lower = source.casefold()
        word_equivalents = {
            "bhp": ["必和必拓", "bhp"],
            "rmb": ["人民币", "rmb"],
            "aud": ["澳元", "aud"],
            "australian dollars": ["澳元", "australian dollars"],
            "tons": ["吨", "tons", "tonnes"],
            "tonnes": ["吨", "tonnes", "tons"],
            "ton": ["吨", "ton"],
            "percent": ["%", "百分"],
            "million": ["百万", "million"],
            "billion": ["十亿", "billion"],
            "hundred": ["百", "hundred"],
        }
        for token, equivalents in word_equivalents.items():
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", lower):
                anchors.append(("term", token, equivalents))

        for acronym in re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", source):
            equivalents = word_equivalents.get(acronym.casefold(), [acronym.casefold()])
            anchors.append(("acronym", acronym, equivalents))

        for term in self.translation_terms:
            term = str(term or "").strip()
            if term and self.normalize_translation_comparison_text(term) in self.normalize_translation_comparison_text(source):
                anchors.append(("glossary", term, [term.casefold()]))

        for source_term, target_term in self.translation_glossary.items():
            if self.normalize_translation_comparison_text(source_term) in self.normalize_translation_comparison_text(source):
                anchors.append(("glossary", source_term, [str(target_term or source_term).casefold()]))
        return anchors

    def missing_fact_anchors(self, source_text, translated_text, source_language, target_language):
        anchors = self.fact_anchors(source_text, source_language, target_language)
        output = self._normalized_anchor_text(translated_text)
        missing = []
        for kind, value, equivalents in anchors:
            normalized_equivalents = [self._normalized_anchor_text(item) for item in equivalents if str(item or "").strip()]
            if not any(item and item in output for item in normalized_equivalents):
                missing.append((kind, value))
        return missing

    def translation_completeness_reason(self, source_text, translated_text, source_language, target_language):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language)
        if (
            self.service_mode != "accurate"
            or not self.is_nllb_model(self.model_name)
            or (source_language, target_language) not in (("en", "zh"), ("zh", "en"))
        ):
            return None
        if self.missing_fact_anchors(source_text, translated_text, source_language, target_language):
            return "missing_fact_anchor"
        if source_language == "en" and target_language == "zh":
            return self.en_zh_undertranslation_reason(source_text, translated_text, source_language, target_language)
        source_chars = self._count_cjk(source_text)
        if source_chars < self._ZH_EN_UNDERTRANSLATION_MIN_CJK_CHARS:
            return None
        target_words = len(self._word_spans(translated_text))
        if target_words * 100 < source_chars * self._ZH_EN_UNDERTRANSLATION_TARGET_WORD_PERCENT:
            return "undertranslation"
        clauses = [part for part in re.split(r"[\s，。！？；：,!?;:]+", str(source_text or "")) if part]
        if source_chars >= 24 and len(clauses) >= 3 and target_words < len(clauses) * 2:
            return "undertranslation"
        return None

    def make_translation_candidate(self, text, source_language, target_language, stage, source_text):
        safety_reason = self.translation_output_failure_reason(source_text, text, source_language, target_language)
        completeness_reason = None if safety_reason else self.translation_completeness_reason(
            source_text,
            text,
            source_language,
            target_language,
        )
        missing = [] if safety_reason else self.missing_fact_anchors(source_text, text, source_language, target_language)
        anchors = self.fact_anchors(source_text, source_language, target_language)
        coverage = 1.0 if not anchors else (len(anchors) - len(missing)) / max(1, len(anchors))
        return {
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
            "stage": stage,
            "safety_reason": safety_reason,
            "completeness_reason": completeness_reason,
            "fact_coverage": coverage,
            "units": self._count_cjk(text) + len(self._word_spans(text)),
        }

    def select_best_translation_candidate(self, candidates):
        safe = [candidate for candidate in candidates if candidate and not candidate.get("safety_reason")]
        if not safe:
            return None
        stage_weight = {"strict": 4, "context": 3, "direct": 3, "relaxed": 2, "chunked": 1}
        return max(
            safe,
            key=lambda candidate: (
                not bool(candidate.get("completeness_reason")),
                candidate.get("fact_coverage", 0.0),
                candidate.get("units", 0),
                stage_weight.get(candidate.get("stage"), 0),
            ),
        )

    def repair_en_zh_undertranslation_if_needed(self, source_text, translated_text, source_language, target_language):
        return self.repair_translation_completeness_if_needed(source_text, translated_text, source_language, target_language)

    def repair_translation_completeness_if_needed(self, source_text, translated_text, source_language, target_language):
        if self.pending_translation_warning:
            return translated_text, source_language, target_language
        reason = self.translation_completeness_reason(source_text, translated_text, source_language, target_language)
        if not reason:
            return translated_text, source_language, target_language

        started_at = time.monotonic()
        candidates = [self.make_translation_candidate(translated_text, source_language, target_language, "strict", source_text)]
        logging.info(
            "[TRANSLATION_UNDERTRANSLATION_DETECTED] uid=%s source_words=%d output_units=%d source_chars=%d output_chars=%d",
            self.client_uid,
            len(self._word_spans(source_text)),
            self._count_cjk(translated_text) + len(self._word_spans(translated_text)),
            len(str(source_text or "")),
            len(str(translated_text or "")),
        )

        direct_text, direct_source, direct_target, direct_reason, _ = self.infer_translation_once_without_final_state(
            source_text,
            source_language,
            allow_batch=False,
        )
        if not direct_reason:
            direct_candidate = self.make_translation_candidate(direct_text, direct_source, direct_target, "direct", source_text)
            if not direct_candidate["safety_reason"] and not direct_candidate["completeness_reason"]:
                logging.info(
                    "[TRANSLATION_UNDERTRANSLATION_REPAIRED] uid=%s method=direct elapsed_ms=%.1f",
                    self.client_uid,
                    (time.monotonic() - started_at) * 1000.0,
                )
                return direct_text, direct_source, direct_target
            candidates.append(direct_candidate)

        relaxed_text, relaxed_source, relaxed_target, relaxed_reason, _ = self.infer_translation_once_without_final_state(
            source_text,
            source_language,
            allow_batch=False,
            generation_profile="relaxed",
        )
        if not relaxed_reason:
            relaxed_candidate = self.make_translation_candidate(relaxed_text, relaxed_source, relaxed_target, "relaxed", source_text)
            logging.info(
                "[TRANSLATION_RELAXED_RETRY] uid=%s reason=%s safety=%s completeness=%s",
                self.client_uid,
                reason,
                relaxed_candidate.get("safety_reason"),
                relaxed_candidate.get("completeness_reason"),
            )
            if not relaxed_candidate["safety_reason"] and not relaxed_candidate["completeness_reason"]:
                return relaxed_text, relaxed_source, relaxed_target
            candidates.append(relaxed_candidate)

        if time.monotonic() - started_at <= self._UNDERTRANSLATION_REPAIR_SECONDS:
            chunk_result = self.translate_en_zh_undertranslation_chunks(source_text, source_language, target_language)
            if chunk_result is not None:
                chunk_text, chunk_source, chunk_target = chunk_result
                chunk_candidate = self.make_translation_candidate(chunk_text, chunk_source, chunk_target, "chunked", source_text)
                if not chunk_candidate["safety_reason"] and not chunk_candidate["completeness_reason"]:
                    logging.info(
                        "[TRANSLATION_UNDERTRANSLATION_REPAIRED] uid=%s method=chunked chunks=%d elapsed_ms=%.1f",
                        self.client_uid,
                        len(self.split_en_zh_undertranslation_chunks(source_text)),
                        (time.monotonic() - started_at) * 1000.0,
                    )
                    return chunk_text, chunk_source, chunk_target
                candidates.append(chunk_candidate)

        best = self.select_best_translation_candidate(candidates)
        if best:
            self.pending_translation_confidence = "low"
            self.pending_translation_warning = None
            logging.warning(
                "[TRANSLATION_LOW_CONFIDENCE] uid=%s candidates=%d reason=%s stage=%s coverage=%.3f source_words=%d output_units=%d",
                self.client_uid,
                len(candidates),
                best.get("completeness_reason") or reason,
                best.get("stage"),
                best.get("fact_coverage", 0.0),
                len(self._word_spans(source_text)),
                best.get("units", 0),
            )
            return best["text"], best["source_language"], best["target_language"]

        self.pending_translation_warning = reason or "undertranslation"
        logging.warning(
            "[TRANSLATION_UNDERTRANSLATION_UNRESOLVED] uid=%s candidates=%d warning=%s source_words=%d output_units=%d",
            self.client_uid,
            len(candidates),
            self.pending_translation_warning,
            len(self._word_spans(source_text)),
            self._count_cjk(translated_text) + len(self._word_spans(translated_text)),
        )
        return self.translation_unavailable_text(target_language), source_language, target_language

    @classmethod
    def split_en_zh_undertranslation_chunks(cls, text):
        text = str(text or "").strip()
        if not text:
            return []
        chunks = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text) if part.strip()]
        if len(chunks) <= 1:
            chunks = [part.strip() for part in re.split(r",\s+", text) if part.strip()]
        if len(chunks) <= 1 and cls._count_cjk(text):
            chunks = [part.strip() for part in re.split(r"[，。！？；：,!?;:\s]+", text) if part.strip()]
        if len(chunks) <= 1:
            words = [match.group(0) for match in cls._word_spans(text)]
            if len(words) <= cls._UNDERTRANSLATION_MIN_CHUNK_WORDS:
                return [text]
            target_chunks = min(cls._UNDERTRANSLATION_MAX_CHUNKS, max(2, len(words) // 10))
            chunk_size = max(cls._UNDERTRANSLATION_MIN_CHUNK_WORDS, (len(words) + target_chunks - 1) // target_chunks)
            chunks = [" ".join(words[index:index + chunk_size]) for index in range(0, len(words), chunk_size)]

        while len(chunks) > cls._UNDERTRANSLATION_MAX_CHUNKS:
            shortest_index = min(range(len(chunks)), key=lambda index: len(cls._word_spans(chunks[index])))
            if shortest_index == 0:
                merge_index = 0
            elif shortest_index == len(chunks) - 1:
                merge_index = shortest_index - 1
            else:
                left_words = len(cls._word_spans(chunks[shortest_index - 1]))
                right_words = len(cls._word_spans(chunks[shortest_index + 1]))
                merge_index = shortest_index - 1 if left_words <= right_words else shortest_index
            chunks[merge_index:merge_index + 2] = [cls._join_merge_text(chunks[merge_index:merge_index + 2], "en")]

        index = 0
        while len(chunks) > 1 and index < len(chunks):
            if len(cls._word_spans(chunks[index])) >= cls._UNDERTRANSLATION_MIN_CHUNK_WORDS:
                index += 1
                continue
            if index == 0:
                chunks[0:2] = [cls._join_merge_text(chunks[0:2], "en")]
            else:
                chunks[index - 1:index + 1] = [cls._join_merge_text(chunks[index - 1:index + 1], "en")]
                index -= 1
        return chunks[:cls._UNDERTRANSLATION_MAX_CHUNKS]

    def translate_en_zh_undertranslation_chunks(self, source_text, source_language, target_language):
        if not self.translator or not self.translator_lock or not hasattr(self.translator, "translate_batch"):
            return None
        chunks = self.split_en_zh_undertranslation_chunks(source_text)
        if len(chunks) <= 1:
            return None
        items = [
            {"text": chunk, "source_language": source_language, "target_language": target_language}
            for chunk in chunks
        ]
        try:
            with self.translator_lock:
                results = self.translator.translate_batch(items)
        except Exception as error:
            logging.warning(
                "[TRANSLATION_UNDERTRANSLATION_CHUNK_FAILED] uid=%s chunks=%d reason=%s",
                self.client_uid,
                len(chunks),
                self.translation_exception_reason(error),
            )
            return None
        translated_parts = []
        resolved_source = source_language
        resolved_target = target_language
        if len(results) != len(chunks):
            logging.warning(
                "[TRANSLATION_UNDERTRANSLATION_CHUNK_FAILED] uid=%s chunks=%d results=%d reason=result_count_mismatch",
                self.client_uid,
                len(chunks),
                len(results),
            )
            return None
        for chunk, result in zip(chunks, results):
            translated_part, resolved_source, resolved_target = result
            if self.translation_output_failure_reason(chunk, translated_part, resolved_source, resolved_target):
                return None
            translated_parts.append(str(translated_part or "").strip())
        return "".join(part for part in translated_parts if part), resolved_source, resolved_target

    def zh_en_context_risk_reason(self, source_text, translated_text, source_language, target_language):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language)
        if source_language != "zh" or target_language != "en":
            return None
        source_text = str(source_text or "")
        translated_text = str(translated_text or "")
        source_cjk = self._count_cjk(source_text)
        if 0 < source_cjk <= 24:
            return "short_zh_context_risk"
        if len(translated_text) >= 160 and len(translated_text) > max(source_cjk or len(source_text), 1) * 4:
            return "context_expansion_risk"
        translated_words = self._output_words(translated_text)
        if len(translated_words) >= 4:
            with self.readability_context_lock:
                previous_units = list(self.readability_context_history)
            previous_words = []
            for unit in previous_units:
                previous_words.extend(self._output_words(unit.get("text")))
            previous_ngrams = {
                tuple(previous_words[index:index + 4])
                for index in range(0, max(0, len(previous_words) - 3))
            }
            for index in range(0, len(translated_words) - 3):
                if tuple(translated_words[index:index + 4]) in previous_ngrams:
                    return "context_history_leak"
        return None

    def translation_queue_size(self):
        try:
            return self.translation_queue.qsize()
        except Exception:
            return "unknown"

    @classmethod
    def final_source_key(cls, segment):
        utterance_id = str((segment or {}).get("utterance_id") or "").strip()
        start = cls._segment_time((segment or {}).get("start"))
        end = cls._segment_time((segment or {}).get("end"))
        return (utterance_id, round(start, 3), round(end, 3))

    @classmethod
    def final_source_ids(cls, segment):
        ids = []
        source_ids = (segment or {}).get("source_utterance_ids")
        if isinstance(source_ids, str):
            ids.append(source_ids)
        elif isinstance(source_ids, (list, tuple)):
            ids.extend(source_ids)
        if (segment or {}).get("utterance_id"):
            ids.append((segment or {}).get("utterance_id"))
        return {str(item).strip() for item in ids if str(item or "").strip()}

    @classmethod
    def segment_covers_source(cls, translated_segment, source_segment):
        translated_ids = cls.final_source_ids(translated_segment)
        source_id = str((source_segment or {}).get("utterance_id") or "").strip()
        translated_start = cls._segment_time((translated_segment or {}).get("start"))
        translated_end = cls._segment_time((translated_segment or {}).get("end"))
        source_start = cls._segment_time((source_segment or {}).get("start"))
        source_end = cls._segment_time((source_segment or {}).get("end"))
        time_covers = translated_start <= source_start + 0.001 and translated_end + 0.001 >= source_end
        if source_id:
            return source_id in translated_ids and time_covers
        return time_covers

    def register_pending_final_segment(self, segment):
        if not (segment or {}).get("completed", False):
            return
        key = self.final_source_key(segment)
        with self.final_translation_lock:
            if key in self.timed_out_final_keys:
                return
            self.pending_final_segments[key] = segment.copy()
        logging.info(
            "[TRANSLATION_FINAL_PENDING] uid=%s key=%s queue_size=%s text_preview=%r",
            self.client_uid,
            key,
            self.translation_queue_size(),
            str((segment or {}).get("text") or "").strip()[:80],
        )

    def resolve_pending_final_segments(self, translated_segment):
        resolved = []
        with self.final_translation_lock:
            for key, source_segment in list(self.pending_final_segments.items()):
                if self.segment_covers_source(translated_segment, source_segment):
                    resolved.append(key)
                    self.pending_final_segments.pop(key, None)
            for key in resolved:
                self.timed_out_final_keys.discard(key)
        if resolved:
            logging.info(
                "[TRANSLATION_FINAL_RESOLVED] uid=%s resolved=%d start=%s end=%s warning=%s",
                self.client_uid,
                len(resolved),
                (translated_segment or {}).get("start"),
                (translated_segment or {}).get("end"),
                (translated_segment or {}).get("translation_warning") or "",
            )

    def translated_segment_is_timed_out_late(self, translated_segment):
        if (translated_segment or {}).get("translation_warning") == "translation_drain_timeout":
            return False
        with self.final_translation_lock:
            timed_out = set(self.timed_out_final_keys)
        if not timed_out:
            return False
        translated_ids = self.final_source_ids(translated_segment)
        translated_start = self._segment_time((translated_segment or {}).get("start"))
        translated_end = self._segment_time((translated_segment or {}).get("end"))
        for utterance_id, source_start, source_end in timed_out:
            if utterance_id and utterance_id not in translated_ids:
                continue
            if translated_start <= source_start + 0.001 and translated_end + 0.001 >= source_end:
                return True
        return False

    def build_failed_translation_segment(self, source_segment, reason):
        source_ids = []
        if (source_segment or {}).get("utterance_id"):
            source_ids.append((source_segment or {}).get("utterance_id"))
        source_language = self.get_segment_source_language(source_segment)
        target_language = self._resolved_target_language(source_language)
        failed = {
            "start": (source_segment or {}).get("start"),
            "end": (source_segment or {}).get("end"),
            "text": self.translation_unavailable_text(target_language),
            "completed": True,
            "source_text": str((source_segment or {}).get("text") or "").strip(),
            "source_language": source_language,
            "target_language": target_language,
            "translation_model": self.model_name,
            "translation_warning": reason or "translation_exception",
        }
        if source_ids:
            failed["source_utterance_ids"] = source_ids
            failed["utterance_id"] = source_ids[0]
        return failed

    def emit_failed_translation_for_source(self, source_segment, reason):
        failed = self.build_failed_translation_segment(source_segment, reason)
        self.emit_translated_segment(failed)
        logging.warning(
            "[TRANSLATION_FINAL_FAILED] uid=%s reason=%s key=%s text_preview=%r",
            self.client_uid,
            reason,
            self.final_source_key(source_segment),
            str((source_segment or {}).get("text") or "").strip()[:80],
        )
        return failed

    def emit_timeout_placeholders(self, reason="translation_drain_timeout"):
        with self.final_translation_lock:
            pending = list(self.pending_final_segments.items())
            for key, _segment in pending:
                self.timed_out_final_keys.add(key)
            self.pending_final_segments.clear()
        for _key, source_segment in pending:
            self.emit_failed_translation_for_source(source_segment, reason)
        self.translation_timeout_count += len(pending)
        if pending:
            logging.warning(
                "[TRANSLATION_FINAL_TIMEOUT] uid=%s count=%d reason=%s",
                self.client_uid,
                len(pending),
                reason,
            )
        return len(pending)

    def should_retry_nllb_residual_cjk(self, translated_text, source_language, target_language):
        if not self.is_nllb_model(self.model_name):
            return False
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language) or "auto"
        if source_language != "zh" or target_language != "en":
            return False
        translated_text = str(translated_text or "")
        cjk_count = self._count_cjk(translated_text)
        if cjk_count < self._RESIDUAL_CJK_MIN_CHARS:
            return False
        visible_count = len(re.sub(r"\s+", "", translated_text))
        if visible_count <= 0:
            return False
        return (cjk_count / visible_count) >= self._RESIDUAL_CJK_MAX_RATIO

    @classmethod
    def normalize_translation_comparison_text(cls, text):
        normalized = unicodedata.normalize("NFKC", str(text or ""))
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized.strip(cls._TRANSLATION_COMPARISON_PUNCTUATION).casefold()

    @staticmethod
    def _is_proper_term_token(token):
        token = str(token or "")
        letters = [char for char in token if char.isalpha()]
        uppercase_count = sum(char.isupper() for char in letters)
        return any(char.isdigit() for char in token) or token.isupper() or uppercase_count >= 2

    def _is_exempt_source_echo_term(self, text):
        value = unicodedata.normalize("NFKC", str(text or ""))
        value = re.sub(r"\s+", " ", value).strip()
        value = value.strip(self._TRANSLATION_COMPARISON_PUNCTUATION)
        if not value or self._count_cjk(value) or len(value) > self._SOURCE_ECHO_PROPER_TERM_MAX_CHARS:
            return False

        normalized = self.normalize_translation_comparison_text(value)
        configured_terms = {
            self.normalize_translation_comparison_text(term)
            for term in self.translation_terms
            if str(term or "").strip()
        }
        if normalized in configured_terms:
            return True

        proper_term_pattern = r"[A-Za-z0-9][A-Za-z0-9+#._/-]*(?:\s+[A-Za-z0-9][A-Za-z0-9+#._/-]*){0,3}"
        if not re.fullmatch(proper_term_pattern, value):
            return False
        tokens = value.split()
        return (
            len(tokens) <= self._SOURCE_ECHO_PROPER_TERM_MAX_TOKENS
            and all(self._is_proper_term_token(token) for token in tokens)
        )

    def translation_output_failure_reason(
        self,
        source_text,
        translated_text,
        source_language,
        target_language,
    ):
        translated_text = str(translated_text or "")
        if self._is_hard_drop_hallucination_text(source_text) or self._is_hard_drop_hallucination_text(translated_text):
            return "hard_hallucination_phrase"
        if not translated_text.strip():
            return "empty_output"

        normalized_source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        normalized_target_language = HelsinkiZhEnTranslator.normalize_language(target_language) or "auto"
        source_comparison = self.normalize_translation_comparison_text(source_text)
        translated_comparison = self.normalize_translation_comparison_text(translated_text)
        if (
            normalized_source_language in ("zh", "en")
            and normalized_target_language in ("zh", "en")
            and normalized_source_language != normalized_target_language
            and source_comparison
            and source_comparison == translated_comparison
            and not self._is_exempt_source_echo_term(source_text)
        ):
            return "source_echo"

        if self.should_retry_nllb_residual_cjk(
            translated_text,
            normalized_source_language,
            normalized_target_language,
        ):
            return "residual_cjk"

        return self.translation_output_guard_reason(
            source_text,
            translated_text,
            normalized_source_language,
            normalized_target_language,
        )

    @classmethod
    def contextual_undertranslation_reason(cls, source_text, translated_text, source_language, target_language):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language)
        source_text = str(source_text or "")
        translated_text = str(translated_text or "")
        if source_language == "en" and target_language == "zh":
            source_words = len(cls._word_spans(source_text))
            if source_words < 16:
                return None
            target_units = cls._count_cjk(translated_text) + len(cls._word_spans(translated_text))
            if target_units * 100 < source_words * 65:
                return "context_undertranslation"
        if source_language == "zh" and target_language == "en":
            source_chars = cls._count_cjk(source_text)
            if source_chars < 12:
                return None
            target_words = len(cls._word_spans(translated_text))
            if target_words * 100 < source_chars * 25:
                return "context_undertranslation"
        return None

    @classmethod
    def _output_words(cls, text):
        return [match.group(0).lower() for match in cls._word_spans(text)]

    @classmethod
    def _has_repeated_ngram(cls, words, ngram_size=2):
        if len(words) < ngram_size * cls._OUTPUT_GUARD_MAX_REPEATED_BIGRAM_COUNT:
            return False
        repeated_count = 1
        previous = None
        for index in range(0, len(words) - ngram_size + 1, ngram_size):
            current = tuple(words[index:index + ngram_size])
            if current == previous:
                repeated_count += 1
                if repeated_count >= cls._OUTPUT_GUARD_MAX_REPEATED_BIGRAM_COUNT:
                    return True
            else:
                repeated_count = 1
                previous = current
        return False

    @classmethod
    def translation_output_guard_reason(cls, source_text, translated_text, source_language=None, target_language=None):
        source_text = str(source_text or "")
        translated_text = str(translated_text or "")
        if re.search(rf"_{{{cls._OUTPUT_GUARD_UNDERSCORE_RUN},}}", translated_text):
            return "underscore_run"

        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language)
        min_long_output_chars = cls._OUTPUT_GUARD_MIN_LONG_OUTPUT_CHARS
        max_length_ratio = cls._OUTPUT_GUARD_MAX_LENGTH_RATIO
        if source_language == "zh" and target_language == "en":
            min_long_output_chars = cls._OUTPUT_GUARD_ZH_EN_MIN_LONG_OUTPUT_CHARS
            max_length_ratio = cls._OUTPUT_GUARD_ZH_EN_MAX_LENGTH_RATIO

        if (
            len(translated_text) >= min_long_output_chars
            and len(translated_text) > max(len(source_text), 1) * max_length_ratio
        ):
            return "length_ratio"
        if cls._has_repeated_cjk_phrase(translated_text):
            return "repeated_cjk_phrase"

        words = cls._output_words(translated_text)
        if len(words) >= cls._OUTPUT_GUARD_MIN_REPEAT_WORDS:
            unique_ratio = len(set(words)) / max(1, len(words))
            if unique_ratio <= cls._OUTPUT_GUARD_MAX_UNIQUE_WORD_RATIO:
                return "low_unique_word_ratio"
            if cls._has_repeated_ngram(words):
                return "repeated_ngram"
        return None

    @classmethod
    def _has_repeated_cjk_phrase(cls, text):
        compact = re.sub(r"[^\u3400-\u4dbf\u4e00-\u9fff]+", "", str(text or ""))
        if len(compact) < 16:
            return False
        for phrase_len in range(4, min(12, len(compact) // 4) + 1):
            for start in range(0, len(compact) - phrase_len + 1):
                phrase = compact[start:start + phrase_len]
                if compact.count(phrase) >= 4:
                    return True
        return False

    def guard_translation_output(
        self,
        source_text: str,
        translated_text: str,
        source_language: Optional[str],
        target_language: Optional[str],
    ):
        reason = self.translation_output_failure_reason(
            source_text,
            translated_text,
            source_language,
            target_language,
        )
        if not reason:
            return translated_text
        failed_text, _, _ = self._finalize_translation_failure(
            source_text,
            source_language,
            target_language,
            reason,
            reason,
            1,
            "guard",
            time.monotonic(),
            translated_text=translated_text,
        )
        return failed_text

    @staticmethod
    def normalize_translation_glossary(glossary):
        normalized = {}
        for source, target in dict(glossary or {}).items():
            source = str(source or "").strip()
            target = str(target or "").strip()
            if source and target:
                normalized[source] = target
        return normalized

    @staticmethod
    def _normalize_glossary_lookup_text(text):
        punctuation = " \t\r\n.,!?;:，。！？；：\"'“”‘’()[]{}"
        return str(text or "").strip(punctuation).casefold()

    @staticmethod
    def _glossary_source_language(source):
        source = str(source or "")
        if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", source):
            return "zh"
        if re.search(r"[A-Za-z]", source):
            return "en"
        return None

    def glossary_sources_for_language(self, source_language):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        if source_language not in ("zh", "en"):
            source_language = None
        if source_language in self._translation_glossary_sources_cache:
            return self._translation_glossary_sources_cache[source_language]

        sources = []
        for source in self.translation_glossary:
            glossary_language = self._glossary_source_language(source)
            if source_language is None or glossary_language is None or glossary_language == source_language:
                sources.append(source)
        self._translation_glossary_sources_cache[source_language] = sources
        return sources

    @staticmethod
    def _glossary_term_pattern(source):
        escaped = re.escape(source)
        if source and source[0].isascii() and source[0].isalnum():
            escaped = rf"(?<![A-Za-z0-9]){escaped}"
        if source and source[-1].isascii() and source[-1].isalnum():
            escaped = rf"{escaped}(?![A-Za-z0-9])"
        return escaped

    @classmethod
    def _glossary_marker_pattern(cls, index):
        marker = f"ZZGLOSSARY{index}ZZ"
        return re.compile(r"\s*".join(re.escape(char) for char in marker), re.IGNORECASE)

    def translate_with_glossary(self, text: str, source_language: Optional[str]):
        if not self.translation_glossary:
            return None

        eligible_sources = self.glossary_sources_for_language(source_language)
        if not eligible_sources:
            return None

        normalized_text = self._normalize_glossary_lookup_text(text)
        for source in eligible_sources:
            target = self.translation_glossary[source]
            if self._normalize_glossary_lookup_text(source) == normalized_text:
                logging.info("[TRANSLATION_GLOSSARY_EXACT] source=%r target=%r", text, target)
                return (
                    target,
                    HelsinkiZhEnTranslator.normalize_language(source_language),
                    self._resolved_target_language(source_language),
                )

        ordered_sources = sorted(eligible_sources, key=len, reverse=True)
        if not ordered_sources:
            return None
        pattern = re.compile(
            "|".join(self._glossary_term_pattern(source) for source in ordered_sources),
            re.IGNORECASE,
        )
        replacements = []

        def protect(match):
            matched_source = match.group(0)
            target = next(
                self.translation_glossary[source]
                for source in ordered_sources
                if source.casefold() == matched_source.casefold()
            )
            marker = f"ZZGLOSSARY{len(replacements)}ZZ"
            replacements.append((marker, target))
            return marker

        protected_text = pattern.sub(protect, text)
        if not replacements:
            return None

        translated_text, normalized_source, target_language = self.translate_text(
            protected_text,
            source_language,
        )
        if self.pending_translation_warning and self.pending_translation_warning != "undertranslation":
            return translated_text, normalized_source, target_language
        restored_text = translated_text
        for index, (_, target) in enumerate(replacements):
            marker_pattern = self._glossary_marker_pattern(index)
            if not marker_pattern.search(restored_text):
                logging.warning(
                    "[TRANSLATION_GLOSSARY_FALLBACK] marker=%d source=%r",
                    index,
                    text,
                )
                return None
            restored_text = marker_pattern.sub(lambda _: target, restored_text)

        logging.info(
            "[TRANSLATION_GLOSSARY] matches=%d source=%r",
            len(replacements),
            text,
        )
        return restored_text, normalized_source, target_language

    def _resolved_target_language(self, source_language):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(self.target_language) or "auto"
        if target_language == "auto":
            if source_language == "en":
                return "zh"
            if source_language == "zh":
                return "en"
        return target_language

    @classmethod
    def translate_standalone_interjection(
        cls,
        text: str,
        source_language: Optional[str],
        target_language: str,
    ):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        target_language = HelsinkiZhEnTranslator.normalize_language(target_language) or "auto"
        if source_language != "en" or target_language not in ("auto", "zh"):
            return None

        normalized_text = re.sub(r"^[\W_]+|[\W_]+$", "", str(text or "").strip().lower())
        translated_text = cls._STANDALONE_ENGLISH_INTERJECTIONS.get(normalized_text)
        if translated_text is None:
            return None

        logging.info(
            "[TRANSLATION_INTERJECTION] source=%r translated=%r",
            text,
            translated_text,
        )
        return translated_text, source_language, "zh"

    @staticmethod
    def infer_text_language(text):
        text = str(text or "")
        cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))
        if latin_count >= 4 and cjk_count == 0:
            return "en"
        if cjk_count > 0:
            return "zh"
        if latin_count >= 4:
            return "en"
        return None

    @staticmethod
    def _count_cjk(text):
        return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", str(text or "")))

    @staticmethod
    def _normalized_short_text(text):
        return re.sub(r"^[\W_]+|[\W_]+$", "", str(text or "").strip())

    def should_infer_segment_language(self):
        return self.translation_mode == "mixed_interpretation"

    def get_segment_source_language(self, segment):
        source_language = HelsinkiZhEnTranslator.normalize_language(segment.get("language"))
        if source_language in ("zh", "en"):
            return source_language
        if self.should_infer_segment_language():
            inferred_language = self.infer_text_language(segment.get("text"))
            if inferred_language:
                return inferred_language
        return self.infer_text_language(segment.get("text"))

    def translate_fixed_short_phrase(self, text: str, source_language: Optional[str]):
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        if source_language != "zh" or self._resolved_target_language(source_language) != "en":
            return None
        normalized_text = self._normalized_short_text(text)
        translated = self._FIXED_SHORT_ZH_TRANSLATIONS.get(normalized_text)
        if translated is None:
            return None
        logging.info("[TRANSLATION_FIXED_SHORT] source=%r translated=%r", text, translated)
        return translated, "zh", "en"

    def get_buffer_source_language(self):
        for segment in self.translation_buffer:
            source_language = self.get_segment_source_language(segment)
            if source_language:
                return source_language
        return None

    def join_translation_buffer_text(self):
        source_language = self.get_buffer_source_language()
        texts = [segment.get("text", "").strip() for segment in self.translation_buffer]
        texts = [text for text in texts if text]
        if source_language == "zh":
            return "".join(texts)
        return " ".join(texts)

    def realtime_max_source_chars(self, source_language):
        if source_language == "zh":
            return min(self.translation_max_chars, self._REALTIME_MAX_ZH_CHARS)
        if source_language == "en":
            return min(self.translation_max_chars, self._REALTIME_MAX_EN_CHARS)
        return self.translation_max_chars

    def translation_buffer_audio_seconds(self):
        total = 0.0
        for segment in self.translation_buffer:
            try:
                start = float(segment.get("start"))
                end = float(segment.get("end"))
            except (TypeError, ValueError):
                continue
            total += max(0.0, end - start)
        return total

    @classmethod
    def _split_english_sentence_chunks(cls, text):
        parts = []
        cursor = 0
        for match in re.finditer(r"[^.!?;]+[.!?;]+(?:\s+|$)", text):
            prefix = text[cursor:match.start()].strip()
            if prefix:
                parts.append(prefix)
            parts.append(match.group(0).strip())
            cursor = match.end()
        suffix = text[cursor:].strip()
        if suffix:
            parts.append(suffix)
        return parts or [text]

    @classmethod
    def _pack_text_parts(cls, parts, max_chars, language):
        packed = []
        current = ""
        for part in parts:
            part = str(part or "").strip()
            if not part:
                continue
            candidate = cls._join_merge_text((current, part), language) if current else part
            if current and len(candidate) > max_chars:
                packed.append(current)
                current = part
            else:
                current = candidate
        if current:
            packed.append(current)
        return packed

    @classmethod
    def _split_long_english_text(cls, text, max_chars):
        sentence_parts = cls._pack_text_parts(cls._split_english_sentence_chunks(text), max_chars, "en")
        parts = []
        for sentence_part in sentence_parts:
            if len(sentence_part) <= max_chars:
                parts.append(sentence_part)
                continue
            current = []
            current_len = 0
            for word in sentence_part.split():
                extra = len(word) + (1 if current else 0)
                if current and current_len + extra > max_chars:
                    parts.append(" ".join(current))
                    current = [word]
                    current_len = len(word)
                else:
                    current.append(word)
                    current_len += extra
            if current:
                parts.append(" ".join(current))
        return parts

    def split_realtime_segment(self, segment):
        text = str(segment.get("text", "") or "").strip()
        if not text:
            return []
        source_language = self.get_segment_source_language(segment)
        max_chars = self.realtime_max_source_chars(source_language)
        if len(text) <= max_chars:
            return [segment]

        if source_language == "zh":
            parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        elif source_language == "en":
            parts = self._split_long_english_text(text, max_chars)
        else:
            parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        if len(parts) <= 1:
            return [segment]

        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        duration = max(end - start, 0.0)
        total_chars = max(sum(len(part) for part in parts), 1)
        cursor = start
        split_segments = []
        for index, part in enumerate(parts):
            next_cursor = end if index == len(parts) - 1 else cursor + duration * (len(part) / total_chars)
            split_segment = segment.copy()
            split_segment["text"] = part
            split_segment["start"] = cursor
            split_segment["end"] = next_cursor
            split_segments.append(split_segment)
            cursor = next_cursor
        logging.info(
            "[TRANSLATION_SPLIT] uid=%s source_language=%s source_len=%d parts=%d max_chars=%d",
            self.client_uid,
            source_language,
            len(text),
            len(split_segments),
            max_chars,
        )
        return split_segments

    @classmethod
    def english_text_ends_incomplete(cls, text):
        text = str(text or "").strip()
        if not text:
            return False
        if text.endswith((",", ":", "-", "--", "—")):
            return True
        normalized = re.sub(r"[^A-Za-z0-9\s]+$", "", text).strip().lower()
        if not normalized:
            return False
        words = [match.group(0).lower() for match in cls._word_spans(normalized)]
        if not words:
            return False
        if words[-1] in cls._INCOMPLETE_EN_ENDING_WORDS:
            return True
        max_phrase_words = max(
            len(phrase.split())
            for phrase in cls._INCOMPLETE_EN_ENDING_PHRASES
        )
        for size in range(2, min(max_phrase_words, len(words)) + 1):
            if " ".join(words[-size:]) in cls._INCOMPLETE_EN_ENDING_PHRASES:
                return True
        return False

    @classmethod
    def chinese_text_ends_incomplete(cls, text):
        text = str(text or "").strip()
        if not text:
            return False
        if text.endswith(cls._INCOMPLETE_ZH_PAUSE_PUNCTUATION):
            return True
        normalized = text.rstrip(cls._INCOMPLETE_ZH_TRAILING_PUNCTUATION)
        if not normalized:
            return False
        return any(normalized.endswith(phrase) for phrase in cls._INCOMPLETE_ZH_ENDING_PHRASES)

    def zh_en_sentence_buffer_applies(self, source_language=None):
        if not self.translation_zh_en_sentence_buffer_enabled:
            return False
        source_language = HelsinkiZhEnTranslator.normalize_language(source_language)
        return source_language == "zh" and self._resolved_target_language(source_language) == "en"

    def zh_en_sentence_buffer_idle_elapsed(self):
        last_activity = self.translation_buffer_last_source_activity_at or self.translation_buffer_last_added_at
        if last_activity is None:
            return 0.0
        return max(0.0, time.monotonic() - last_activity)

    def translation_buffer_flush_reason(self, force=False):
        if not self.translation_buffer:
            return None
        if force:
            return "force"

        text = self.join_translation_buffer_text().strip()
        if not text:
            return None
        elapsed = 0.0
        if self.translation_buffer_started_at is not None:
            elapsed = time.monotonic() - self.translation_buffer_started_at
        source_language = self.get_buffer_source_language()
        if self.zh_en_sentence_buffer_applies(source_language):
            if self.translation_buffer_audio_seconds() >= self.translation_zh_en_max_audio_seconds:
                return "zh_en_max_audio"
            if len(text) >= self.realtime_max_source_chars(source_language):
                return "max_chars"
            incomplete = self.chinese_text_ends_incomplete(text)
            if text.endswith(tuple(self.translation_sentence_endings)) and not incomplete:
                return "sentence_end"
            if elapsed >= self.translation_max_wait_seconds:
                return "timeout"
            if self.zh_en_sentence_buffer_idle_elapsed() >= self.translation_zh_en_idle_seconds:
                return "zh_en_idle_timeout"
            return None
        if text.endswith(tuple(self.translation_sentence_endings)):
            return "sentence_end"
        if (
            source_language == "en"
            and self.english_text_ends_incomplete(text)
            and len(text) < self.realtime_max_source_chars(source_language)
        ):
            if elapsed < self.translation_incomplete_max_wait_seconds:
                return None
            return "incomplete_timeout"
        if (
            self.translation_context_seconds > 0
            and self.translation_buffer_audio_seconds() >= self.translation_context_seconds
        ):
            return "context_seconds"
        if len(text) >= self.realtime_max_source_chars(source_language):
            return "max_chars"
        if (
            self.translation_mode == "mixed_interpretation"
            and source_language == "zh"
            and self._count_cjk(text) < self._SHORT_ZH_BUFFER_CJK_CHARS
            and elapsed < min(self._SHORT_ZH_BUFFER_WAIT_SECONDS, self.translation_max_wait_seconds)
        ):
            return None
        if (
            self.translation_buffer_started_at is not None
            and elapsed >= self.translation_max_wait_seconds
        ):
            return "timeout"
        return None

    def should_flush_translation_buffer(self, force=False):
        return self.translation_buffer_flush_reason(force=force) is not None

    @staticmethod
    def _word_spans(text):
        return list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text or "")))

    @classmethod
    def _dedupe_leading_word_overlap(cls, previous_text, current_text, max_words=8):
        current = str(current_text or "")
        previous_words = [m.group(0).lower() for m in cls._word_spans(previous_text)]
        current_matches = cls._word_spans(current)
        current_words = [m.group(0).lower() for m in current_matches]
        max_overlap = min(max_words, len(previous_words), len(current_words))
        for size in range(max_overlap, 0, -1):
            if previous_words[-size:] == current_words[:size]:
                cut_at = current_matches[size - 1].end()
                deduped = current[:current_matches[0].start()] + current[cut_at:].lstrip()
                logging.info(
                    "[TRANSLATION_BUFFER_DEDUPE] overlap_words=%d previous=%r current=%r deduped=%r",
                    size,
                    str(previous_text or "").strip()[-80:],
                    current.strip()[:80],
                    deduped.strip()[:80],
                )
                return deduped
        return current

    def _previous_source_text_for_dedupe(self):
        if self.translation_buffer:
            return self.translation_buffer[-1].get("text", "")
        return self.last_translated_source_text

    def should_flush_translation_buffer_before(self, segment, incoming_language):
        if not self.translation_buffer:
            return None
        current_language = self.get_buffer_source_language()
        if incoming_language and current_language and incoming_language != current_language:
            return "language_switch"
        if not self.zh_en_sentence_buffer_applies(current_language):
            return None

        previous = self.translation_buffer[-1]
        previous_speaker = str(previous.get("speaker") or "").strip()
        incoming_speaker = str((segment or {}).get("speaker") or "").strip()
        if previous_speaker and incoming_speaker and previous_speaker != incoming_speaker:
            return "speaker_switch"

        gap = self._segment_time((segment or {}).get("start")) - self._segment_time(previous.get("end"))
        if gap > self.translation_zh_en_max_gap_seconds:
            return "zh_en_segment_gap"
        return None

    def add_segment_to_translation_buffer(self, segment):
        incoming_language = self.get_segment_source_language(segment)
        split_reason = self.should_flush_translation_buffer_before(segment, incoming_language)
        if split_reason:
            self.flush_translation_buffer(force=True, reason=split_reason)

        segment = segment.copy()
        if incoming_language:
            segment["language"] = incoming_language
        previous_text = self._previous_source_text_for_dedupe()
        if previous_text and incoming_language == "en":
            segment["text"] = self._dedupe_leading_word_overlap(previous_text, segment.get("text", ""))
            if not segment["text"].strip():
                return

        if not self.translation_buffer:
            self.translation_buffer_started_at = time.monotonic()
        self.translation_buffer_last_added_at = time.monotonic()
        if self.zh_en_sentence_buffer_applies(incoming_language):
            self.translation_buffer_last_source_activity_at = self.translation_buffer_last_added_at
        self.translation_buffer.append(segment)

    def flush_translation_buffer(self, force=False, reason=None):
        flush_reason = reason or self.translation_buffer_flush_reason(force=force)
        if flush_reason is None:
            return

        buffered_segments = self.translation_buffer
        original_text = self.join_translation_buffer_text().strip()
        source_language = self.get_buffer_source_language()
        flush_started_at = time.monotonic()
        self.translation_buffer = []
        self.translation_buffer_started_at = None
        self.translation_buffer_last_added_at = None
        self.translation_buffer_last_source_activity_at = None

        if not original_text:
            return

        self.pending_translation_warning = None
        try:
            translation_result = self.translate_fixed_short_phrase(original_text, source_language)
            if translation_result is None:
                translation_result = self.translate_with_glossary(original_text, source_language)
            if translation_result is None:
                translation_result = self.translate_standalone_interjection(
                    original_text,
                    source_language,
                    self.target_language,
                )
            if translation_result is None:
                translation_result = self.translate_text(original_text, source_language)
            translated_text, source_language, target_language = translation_result
        except Exception as error:
            logging.error(
                "[TRANSLATION_FINAL_EXCEPTION] uid=%s reason=%s start=%s end=%s error=%s",
                self.client_uid,
                self.translation_exception_reason(error),
                buffered_segments[0].get("start"),
                buffered_segments[-1].get("end"),
                str(error)[:160],
            )
            for source_segment in buffered_segments:
                self.emit_failed_translation_for_source(source_segment, self.translation_exception_reason(error))
            self.pending_translation_warning = None
            self.pending_translation_confidence = None
            return
        translation_warning = self.pending_translation_warning
        translation_confidence = self.pending_translation_confidence
        if flush_reason == "incomplete_timeout":
            logging.info(
                "[TRANSLATION_INCOMPLETE_TIMEOUT] uid=%s model=%s source_language=%s target_language=%s "
                "source_len=%d translated_len=%d start=%s end=%s",
                self.client_uid,
                self.model_name,
                source_language,
                target_language,
                len(original_text),
                len(str(translated_text or "")),
                buffered_segments[0].get("start"),
                buffered_segments[-1].get("end"),
            )
        self.pending_translation_warning = None
        self.pending_translation_confidence = None
        if not translation_confidence:
            self.record_readability_context(
                original_text,
                translated_text,
                source_language,
                target_language,
                translation_warning=translation_warning,
            )
        self.last_translated_source_text = original_text
        logging.info(
            "[TRANSLATION_FLUSH] uid=%s model=%s source_language=%s target_language=%s "
            "source_len=%d translated_len=%d elapsed_ms=%.1f queue_size=%s flush_reason=%s start=%s end=%s",
            self.client_uid,
            self.model_name,
            source_language,
            target_language,
            len(original_text),
            len(str(translated_text or "")),
            (time.monotonic() - flush_started_at) * 1000.0,
            self.translation_queue_size(),
            flush_reason,
            buffered_segments[0].get("start"),
            buffered_segments[-1].get("end"),
        )

        translated_segment = {
            "start": buffered_segments[0]["start"],
            "end": buffered_segments[-1]["end"],
            "text": translated_text,
            "completed": True,
            "source_text": original_text,
            "source_language": source_language,
            "target_language": target_language,
            "translation_model": self.model_name,
        }
        utterance_ids = list(dict.fromkeys(
            segment.get("utterance_id")
            for segment in buffered_segments
            if segment.get("utterance_id")
        ))
        if translation_warning:
            translated_segment["translation_warning"] = translation_warning
        if translation_confidence:
            translated_segment["translation_confidence"] = translation_confidence
        if utterance_ids:
            translated_segment["source_utterance_ids"] = utterance_ids
        if len(utterance_ids) == 1:
            translated_segment["utterance_id"] = utterance_ids[0]

        self.enqueue_translated_segment(translated_segment)

    @staticmethod
    def _segment_time(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _join_merge_text(cls, values, language=None):
        merged = ""
        for value in values:
            current = str(value or "").strip()
            if not current:
                continue
            if not merged:
                merged = current
                continue
            if language == "zh":
                merged = f"{merged}{current}"
                continue
            needs_space = re.search(r"[A-Za-z0-9]$", merged) and re.search(r"^[A-Za-z0-9]", current)
            merged = f"{merged}{' ' if needs_space else ''}{current}"
        return merged

    def _merge_buffer_text(self):
        target_language = self.translation_merge_buffer[0].get("target_language") if self.translation_merge_buffer else None
        return self._join_merge_text((segment.get("text") for segment in self.translation_merge_buffer), target_language)

    def _merge_buffer_source_text(self):
        source_language = self.translation_merge_buffer[0].get("source_language") if self.translation_merge_buffer else None
        return self._join_merge_text((segment.get("source_text") for segment in self.translation_merge_buffer), source_language)

    def _should_split_merge_buffer_before(self, segment):
        if not self.translation_merge_buffer:
            return False
        previous = self.translation_merge_buffer[-1]
        if previous.get("source_language") != segment.get("source_language"):
            return True
        if previous.get("target_language") != segment.get("target_language"):
            return True
        if previous.get("translation_model") != segment.get("translation_model"):
            return True
        gap = self._segment_time(segment.get("start")) - self._segment_time(previous.get("end"))
        return gap > self.translation_merge_gap_seconds

    def should_flush_merge_buffer(self, force=False):
        if not self.translation_merge_buffer:
            return False
        if force:
            return True
        if len(self._merge_buffer_text()) >= self.translation_merge_max_chars:
            return True
        if self.translation_merge_started_at is not None:
            return time.monotonic() - self.translation_merge_started_at >= self.translation_merge_max_delay
        return False

    def emit_translated_segment(self, translated_segment):
        if self.translated_segment_is_timed_out_late(translated_segment):
            logging.warning(
                "[TRANSLATION_LATE_SUPPRESSED] uid=%s start=%s end=%s ids=%s",
                self.client_uid,
                (translated_segment or {}).get("start"),
                (translated_segment or {}).get("end"),
                sorted(self.final_source_ids(translated_segment)),
            )
            return
        self.resolve_pending_final_segments(translated_segment)
        self.translated_segments.append(translated_segment)
        segments_to_send = self.prepare_translated_segments()
        self.send_translation_to_client(segments_to_send)

    def enqueue_translated_segment(self, translated_segment):
        if not self.translation_merge_enabled:
            self.emit_translated_segment(translated_segment)
            return

        if translated_segment.get("translation_warning"):
            self.flush_merge_buffer(force=True)
            self.emit_translated_segment(translated_segment)
            return

        if self._should_split_merge_buffer_before(translated_segment):
            self.flush_merge_buffer(force=True)

        if not self.translation_merge_buffer:
            self.translation_merge_started_at = time.monotonic()
        self.translation_merge_buffer.append(translated_segment)
        self.flush_merge_buffer()

    def build_merged_translation_segment(self):
        first = self.translation_merge_buffer[0]
        last = self.translation_merge_buffer[-1]
        utterance_ids = list(dict.fromkeys(
            utterance_id
            for segment in self.translation_merge_buffer
            for utterance_id in (segment.get("source_utterance_ids") or ([segment.get("utterance_id")] if segment.get("utterance_id") else []))
            if utterance_id
        ))
        merged_segment = {
            "start": first.get("start"),
            "end": last.get("end"),
            "text": self._merge_buffer_text(),
            "completed": True,
            "source_text": self._merge_buffer_source_text(),
            "source_language": first.get("source_language"),
            "target_language": first.get("target_language"),
            "translation_model": first.get("translation_model"),
        }
        if utterance_ids:
            merged_segment["source_utterance_ids"] = utterance_ids
        if len(utterance_ids) == 1:
            merged_segment["utterance_id"] = utterance_ids[0]
        if any(segment.get("translation_confidence") == "low" for segment in self.translation_merge_buffer):
            merged_segment["translation_confidence"] = "low"
        return merged_segment

    def flush_merge_buffer(self, force=False):
        if not self.should_flush_merge_buffer(force=force):
            return
        item_count = len(self.translation_merge_buffer)
        merged_segment = self.build_merged_translation_segment()
        self.translation_merge_buffer = []
        self.translation_merge_started_at = None
        logging.info(
            "[TRANSLATION_MERGE_FLUSH] uid=%s items=%d text_len=%d start=%s end=%s",
            self.client_uid,
            item_count,
            len(str(merged_segment.get("text") or "")),
            merged_segment.get("start"),
            merged_segment.get("end"),
        )
        self.emit_translated_segment(merged_segment)

    def drain_translation_backlog(self, first_segment):
        queue_size = self.translation_queue_size()
        if not isinstance(queue_size, int) or queue_size <= self._BACKLOG_DROP_THRESHOLD:
            return [first_segment], False

        drained = []
        saw_exit_signal = False
        while True:
            try:
                item = self.translation_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                saw_exit_signal = True
            elif item is self._TRANSLATION_DRAFT_WAKEUP:
                self._clear_translation_draft_wakeup()
            else:
                drained.append(item)
            self.translation_queue.task_done()

        candidates = [first_segment] + drained
        if self.service_mode == "accurate" and self.translation_draft_enabled:
            if len(candidates) > 1:
                logging.info(
                    "[TRANSLATION_BACKLOG_PRESERVED] uid=%s queue_size=%s segments=%d",
                    self.client_uid,
                    queue_size,
                    len(candidates),
                )
            return candidates, saw_exit_signal
        kept = candidates[-self._BACKLOG_KEEP_LATEST:]
        dropped = max(len(candidates) - len(kept), 0)
        if dropped:
            logging.warning(
                "[TRANSLATION_BACKLOG_DROP] uid=%s queue_size=%s dropped=%d kept=%d",
                self.client_uid,
                queue_size,
                dropped,
                len(kept),
            )
        return kept, saw_exit_signal

    def process_translation_segment(self, segment):
        if not segment.get("completed", False):
            return

        logging.info(
            "[TRANSLATION_QUEUE_SEGMENT] uid=%s queue_size=%s start=%s end=%s text_preview=%r",
            self.client_uid,
            self.translation_queue_size(),
            segment.get("start"),
            segment.get("end"),
            str(segment.get("text", "")).strip()[:80],
        )
        for split_segment in self.split_realtime_segment(segment):
            if self.exit:
                break
            self.register_pending_final_segment(split_segment)
            try:
                self.add_segment_to_translation_buffer(split_segment)
                self.flush_translation_buffer()
            except Exception as error:
                logging.error(
                    "[TRANSLATION_SEGMENT_ERROR] uid=%s reason=%s start=%s end=%s error=%s",
                    self.client_uid,
                    self.translation_exception_reason(error),
                    split_segment.get("start"),
                    split_segment.get("end"),
                    str(error)[:160],
                )
                self.emit_failed_translation_for_source(split_segment, self.translation_exception_reason(error))

    def process_translation_queue(self):
        """
        Process segments from the translation queue.
        Continuously reads from the queue until None is received (exit signal).
        """
        logging.info(f"Starting translation processing for client {self.client_uid}")

        while not self.exit:
            try:
                segment = self.translation_queue.get(
                    timeout=self.translation_draft_wait_timeout()
                )

                if segment is None:
                    self.translation_queue.task_done()
                    logging.info(f"Received exit signal for translation client {self.client_uid}")
                    self.flush_translation_buffer(force=True)
                    self.flush_merge_buffer(force=True)
                    break

                if segment is self._TRANSLATION_DRAIN_SENTINEL:
                    self.translation_queue.task_done()
                    logging.info("[TRANSLATION_DRAIN_COMPLETE] uid=%s", self.client_uid)
                    self.flush_translation_buffer(force=True)
                    self.flush_merge_buffer(force=True)
                    self.translation_drain_status = "completed"
                    self.translation_drain_completed.set()
                    break

                if segment is self._TRANSLATION_DRAFT_WAKEUP:
                    self.translation_queue.task_done()
                    self._clear_translation_draft_wakeup()
                    self.flush_translation_buffer()
                    self.flush_merge_buffer()
                    self.process_ready_translation_draft()
                    continue

                segments_to_process, saw_exit_signal = self.drain_translation_backlog(segment)
                if segments_to_process == [segment] and not saw_exit_signal:
                    try:
                        self.process_translation_segment(segment)
                    finally:
                        self.translation_queue.task_done()
                else:
                    self.translation_queue.task_done()
                    for pending_segment in segments_to_process:
                        self.process_translation_segment(pending_segment)
                    if saw_exit_signal:
                        logging.info(f"Received exit signal for translation client {self.client_uid}")
                        self.flush_translation_buffer(force=True)
                        self.flush_merge_buffer(force=True)
                        break

                self.process_ready_translation_draft()

            except queue.Empty:
                self.flush_translation_buffer()
                self.flush_merge_buffer()
                self.process_ready_translation_draft()
                continue
            except Exception as e:
                logging.error(f"Error processing translation queue: {e}")
                continue

        logging.info(f"Translation processing ended for client {self.client_uid}")

    def finalize_translation_drain(self, timeout_seconds=0):
        self.translation_drain_status = None
        self.translation_drain_completed.clear()
        logging.info(
            "[TRANSLATION_DRAIN_START] uid=%s queue_size=%s timeout=%.2f pending=%d",
            self.client_uid,
            self.translation_queue_size(),
            float(timeout_seconds or 0.0),
            len(self.pending_final_segments),
        )
        try:
            self.translation_queue.put(self._TRANSLATION_DRAIN_SENTINEL, timeout=0.5)
        except Exception as error:
            logging.error("[TRANSLATION_DRAIN_SIGNAL_FAILED] uid=%s error=%s", self.client_uid, str(error)[:160])
            self.emit_timeout_placeholders()
            self.translation_drain_status = "timed_out"
            return "timed_out"
        if self.translation_drain_completed.wait(max(0.0, float(timeout_seconds or 0.0))):
            return self.translation_drain_status or "completed"
        logging.warning(
            "[TRANSLATION_DRAIN_TIMEOUT] uid=%s timeout=%.2f pending=%d queue_size=%s",
            self.client_uid,
            float(timeout_seconds or 0.0),
            len(self.pending_final_segments),
            self.translation_queue_size(),
        )
        self.emit_timeout_placeholders()
        self.translation_drain_status = "timed_out"
        self.exit = True
        return "timed_out"

    def prepare_translated_segments(self):
        """
        Prepare the last n translated segments to send to client.

        Returns:
            list: List of recent translated segments
        """
        if len(self.translated_segments) >= self.send_last_n_segments:
            return self.translated_segments[-self.send_last_n_segments:]
        return self.translated_segments[:]

    def send_translation_to_client(self, translated_segments):
        """
        Send translated segments to the client via WebSocket.

        Args:
            translated_segments (list): List of translated segments to send
        """
        if getattr(self, "segment_post_processor", None) is not None:
            processed = []
            for seg in translated_segments:
                try:
                    result = self.segment_post_processor(seg)
                    processed.append(result if result is not None else seg)
                except Exception as e:
                    logging.error(f"[ERROR]: translation segment_post_processor failed: {e}")
                    processed.append(seg)
            translated_segments = processed
        if self.admin_status_callback:
            try:
                self.admin_status_callback(translated_segments)
            except Exception as e:
                logging.error(f"[ERROR]: admin translation status update failed: {e}")
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "translated_segments": translated_segments,
                })
            )
        except Exception as e:
            logging.error(f"[ERROR]: Sending translation data to client: {e}")
            self.exit = True

    def speech_to_text(self):
        """
        Override parent method to handle translation processing.
        This method will be called when the translation thread starts.
        """
        self.process_translation_queue()

    def set_target_language(self, language: str):
        """
        Change the target language for translation.

        Args:
            language (str): New target language code
        """
        self.target_language = language
        logging.info(f"Target language changed to: {language}")

    def cleanup(self):
        """Clean up translation resources."""
        logging.info(f"Cleaning up translation resources for client {self.client_uid}")
        try:
            self.flush_translation_buffer(force=True)
        except Exception as e:
            logging.error(f"Failed to flush translation buffer during cleanup: {e}")
        try:
            self.flush_merge_buffer(force=True)
        except Exception as e:
            logging.error(f"Failed to flush translation merge buffer during cleanup: {e}")
        self.exit = True

        try:
            self.translation_queue.put(None, timeout=1.0)
        except:
            pass

        with self.draft_state_lock:
            self.draft_states.clear()
            self.draft_inference_active = False
            self.draft_wakeup_queued = False
            self.last_draft_inference_finished_at = None
        with self.readability_context_lock:
            self.readability_context_history.clear()
            self.readability_context_direction = None
        self.translated_segments.clear()
        self.translation_buffer.clear()
        self.translation_buffer_started_at = None
        self.translation_merge_buffer.clear()
        self.translation_merge_started_at = None
        self.last_translated_source_text = ""
        self.translator = None
        self.translator_lock = None
