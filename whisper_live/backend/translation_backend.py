import json
import logging
import queue
import re
import threading
import time
from typing import Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from whisper_live.backend.base import ServeClientBase


class HelsinkiZhEnTranslator:
    """Local zh<->en translator backed by two Marian/Helsinki models."""

    SUPPORTED_TARGETS = {"auto", "zh", "en"}
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
        device = (device or "cpu").lower()
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError(f"Unsupported translation device: {device}")
        return device

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

    def translate(self, text: str, source_language: Optional[str], target_language: str):
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
            generated_tokens = model.generate(**encoded_input)
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_text = output[0] if output else text
        if protected_terms:
            translated_text = self.restore_natural_term_placeholders(translated_text, protected_terms)
            translated_text = self.restore_english_terms(translated_text, protected_terms)
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

    def translate(self, text: str, source_language: Optional[str], target_language: str):
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
                max_new_tokens=256,
                num_beams=1,
            )
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_text = output[0] if output else text
        if protected_terms:
            translated_text = self.restore_natural_term_placeholders(translated_text, protected_terms)
            translated_text = self.restore_english_terms(translated_text, protected_terms)
        return translated_text, source_language, resolved_target_language

    def cleanup(self):
        self.model = None
        self.tokenizer = None


class ServeClientTranslation(ServeClientBase):
    """
    Handles translation of completed transcription segments in a separate thread.
    Reads from a queue populated by the transcription backend and sends translated
    segments back to the client via WebSocket.
    """
    _TRANSLATOR_CACHE = {}
    _TRANSLATOR_INFERENCE_LOCKS = {}
    _TRANSLATOR_CACHE_LOCK = threading.Lock()
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
        translation_max_chars=130,
        translation_max_wait_seconds=2.0,
        translation_sentence_endings="。！？.!?",
        translation_glossary=None,
        translation_terms=None,
        translation_mode="standard",
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
        self.translation_min_chars = translation_min_chars
        self.translation_max_chars = translation_max_chars
        self.translation_max_wait_seconds = translation_max_wait_seconds
        self.translation_sentence_endings = translation_sentence_endings
        self.translation_glossary = self.normalize_translation_glossary(translation_glossary)
        self.translation_terms = list(translation_terms or [])
        self.translation_mode = str(translation_mode or "standard")
        self.translation_buffer = []
        self.translation_buffer_started_at = None
        self.translated_segments = []
        self.last_translated_source_text = ""
        self.translator = None
        self.translator_lock = None
        self.model_loaded = False
        self.load_translation_model()

    def get_translation_cache_key(self):
        """Build the process-local cache key for the configured translation model."""
        return (
            self.model_name,
            self.zh_en_model_path,
            self.en_zh_model_path,
            self.nllb_model_path,
            self.translation_device,
        )

    def load_translation_model(self):
        """Load the translation model and tokenizer."""
        try:
            cache_key = self.get_translation_cache_key()
            with self._TRANSLATOR_CACHE_LOCK:
                if cache_key not in self._TRANSLATOR_CACHE:
                    if self.model_name == "helsinki_zh_en":
                        translator = HelsinkiZhEnTranslator(
                            zh_en_model_path=self.zh_en_model_path,
                            en_zh_model_path=self.en_zh_model_path,
                            device=self.translation_device,
                        )
                    elif self.model_name in ("nllb_200_600m", "nllb"):
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

    def translate_text(self, text: str, source_language: Optional[str]):
        """
        Translate a single text segment.

        Args:
            text (str): Text to translate

        Returns:
            str: Translated text or original text if translation fails
        """
        if not self.model_loaded or not self.translator or not text.strip():
            return text, source_language, self.target_language

        try:
            with self.translator_lock:
                return self.translator.translate(text, source_language, self.target_language)
        except Exception as e:
            logging.error(f"Translation failed for text '{text}': {e}")
            return text, source_language, self.target_language

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

        normalized_text = self._normalize_glossary_lookup_text(text)
        for source, target in self.translation_glossary.items():
            if self._normalize_glossary_lookup_text(source) == normalized_text:
                logging.info("[TRANSLATION_GLOSSARY_EXACT] source=%r target=%r", text, target)
                return (
                    target,
                    HelsinkiZhEnTranslator.normalize_language(source_language),
                    self._resolved_target_language(source_language),
                )

        ordered_sources = sorted(self.translation_glossary, key=len, reverse=True)
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
        if self.should_infer_segment_language():
            inferred_language = self.infer_text_language(segment.get("text"))
            if inferred_language:
                return inferred_language
        source_language = HelsinkiZhEnTranslator.normalize_language(segment.get("language"))
        if source_language in ("zh", "en"):
            return source_language
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

    def should_flush_translation_buffer(self, force=False):
        if not self.translation_buffer:
            return False
        if force:
            return True

        text = self.join_translation_buffer_text().strip()
        if not text:
            return False
        elapsed = 0.0
        if self.translation_buffer_started_at is not None:
            elapsed = time.monotonic() - self.translation_buffer_started_at
        source_language = self.get_buffer_source_language()
        if (
            self.translation_mode == "mixed_interpretation"
            and source_language == "zh"
            and self._count_cjk(text) < self._SHORT_ZH_BUFFER_CJK_CHARS
            and elapsed < self._SHORT_ZH_BUFFER_WAIT_SECONDS
        ):
            return False
        if text.endswith(tuple(self.translation_sentence_endings)):
            return True
        if len(text) >= self.translation_max_chars:
            return True
        if (
            self.translation_buffer_started_at is not None
            and elapsed >= self.translation_max_wait_seconds
        ):
            return True
        return False

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

    def add_segment_to_translation_buffer(self, segment):
        incoming_language = self.get_segment_source_language(segment)
        current_language = self.get_buffer_source_language()
        if self.translation_buffer and incoming_language and current_language and incoming_language != current_language:
            self.flush_translation_buffer(force=True)

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
        self.translation_buffer.append(segment)

    def flush_translation_buffer(self, force=False):
        if not self.should_flush_translation_buffer(force=force):
            return

        buffered_segments = self.translation_buffer
        original_text = self.join_translation_buffer_text().strip()
        source_language = self.get_buffer_source_language()
        self.translation_buffer = []
        self.translation_buffer_started_at = None

        if not original_text:
            return

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
        self.last_translated_source_text = original_text

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
        if utterance_ids:
            translated_segment["source_utterance_ids"] = utterance_ids
        if len(utterance_ids) == 1:
            translated_segment["utterance_id"] = utterance_ids[0]

        self.translated_segments.append(translated_segment)
        segments_to_send = self.prepare_translated_segments()
        self.send_translation_to_client(segments_to_send)

    def process_translation_queue(self):
        """
        Process segments from the translation queue.
        Continuously reads from the queue until None is received (exit signal).
        """
        logging.info(f"Starting translation processing for client {self.client_uid}")

        while not self.exit:
            try:
                # Get segment from queue with timeout
                segment = self.translation_queue.get(timeout=1.0)

                # Check for exit signal
                if segment is None:
                    logging.info(f"Received exit signal for translation client {self.client_uid}")
                    self.flush_translation_buffer(force=True)
                    break

                # Only translate completed segments
                if not segment.get("completed", False):
                    self.translation_queue.task_done()
                    continue

                self.add_segment_to_translation_buffer(segment)
                self.flush_translation_buffer()

                self.translation_queue.task_done()

            except queue.Empty:
                self.flush_translation_buffer()
                continue
            except Exception as e:
                logging.error(f"Error processing translation queue: {e}")
                continue

        logging.info(f"Translation processing ended for client {self.client_uid}")

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
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "translated_segments": translated_segments,
                })
            )
            if self.admin_status_callback:
                try:
                    self.admin_status_callback(translated_segments)
                except Exception as e:
                    logging.error(f"[ERROR]: admin translation status update failed: {e}")
        except Exception as e:
            logging.error(f"[ERROR]: Sending translation data to client: {e}")

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
        self.exit = True

        try:
            self.translation_queue.put(None, timeout=1.0)
        except:
            pass

        self.translated_segments.clear()
        self.translation_buffer.clear()
        self.translation_buffer_started_at = None
        self.last_translated_source_text = ""
        self.translator = None
        self.translator_lock = None
