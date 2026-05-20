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

    def __init__(
        self,
        zh_en_model_path="model/opus-mt-zh-en",
        en_zh_model_path="model/opus-mt-en-zh",
    ):
        self.zh_en_model_path = zh_en_model_path
        self.en_zh_model_path = en_zh_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}

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
    def protect_english_terms(cls, text: str):
        protected_terms = {}

        def replace(match):
            term = match.group(0)
            if not cls.should_protect_english_term(term):
                return term

            placeholder = f"XKEEPTERM{len(protected_terms)}X"
            protected_terms[placeholder] = term
            return placeholder

        return cls.ENGLISH_TERM_PATTERN.sub(replace, text), protected_terms

    @staticmethod
    def restore_english_terms(text: str, protected_terms):
        restored_text = text
        for placeholder, term in protected_terms.items():
            restored_text = restored_text.replace(placeholder, term)
            restored_text = restored_text.replace(placeholder.lower(), term)

            index = placeholder.removeprefix("XKEEPTERM").removesuffix("X")
            spaced_placeholder = re.compile(
                rf"X\s*KEEP\s*TERM\s*{re.escape(index)}\s*X",
                flags=re.IGNORECASE,
            )
            restored_text = spaced_placeholder.sub(term, restored_text)

        if "XKEEPTERM" in restored_text.upper():
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
            text_to_translate, protected_terms = self.protect_english_terms(text)
            if protected_terms:
                logging.info(
                    "[MIXED_LANG_PROTECT] direction=zh-en terms=%d text_len=%d",
                    len(protected_terms),
                    len(text),
                )

        encoded_input = tokenizer(text_to_translate, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            generated_tokens = model.generate(**encoded_input)
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_text = output[0] if output else text
        if protected_terms:
            translated_text = self.restore_english_terms(translated_text, protected_terms)
        return (
            translated_text,
            self.normalize_language(source_language),
            resolved_target_language,
        )

    def cleanup(self):
        self.models.clear()
        self.tokenizers.clear()


class ServeClientTranslation(ServeClientBase):
    """
    Handles translation of completed transcription segments in a separate thread.
    Reads from a queue populated by the transcription backend and sends translated
    segments back to the client via WebSocket.
    """
    _TRANSLATOR_CACHE = {}
    _TRANSLATOR_INFERENCE_LOCKS = {}
    _TRANSLATOR_CACHE_LOCK = threading.Lock()
    
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
        translation_min_chars=8,
        translation_max_chars=60,
        translation_max_wait_seconds=1.0,
        translation_sentence_endings="。！？.!?",
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
        self.translation_min_chars = translation_min_chars
        self.translation_max_chars = translation_max_chars
        self.translation_max_wait_seconds = translation_max_wait_seconds
        self.translation_sentence_endings = translation_sentence_endings
        self.translation_buffer = []
        self.translation_buffer_started_at = None
        self.translated_segments = []
        self.translator = None
        self.translator_lock = None
        self.model_loaded = False
        self.load_translation_model()

    def get_translation_cache_key(self):
        """Build the process-local cache key for the configured translation model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            self.model_name,
            self.zh_en_model_path,
            self.en_zh_model_path,
            str(device),
        )
        
    def load_translation_model(self):
        """Load the translation model and tokenizer."""
        try:
            if self.model_name != "helsinki_zh_en":
                raise ValueError(f"Unsupported translation model provider: {self.model_name}")
            cache_key = self.get_translation_cache_key()
            with self._TRANSLATOR_CACHE_LOCK:
                if cache_key not in self._TRANSLATOR_CACHE:
                    translator = HelsinkiZhEnTranslator(
                        zh_en_model_path=self.zh_en_model_path,
                        en_zh_model_path=self.en_zh_model_path,
                    )
                    translator.load()
                    self._TRANSLATOR_CACHE[cache_key] = translator
                    self._TRANSLATOR_INFERENCE_LOCKS[cache_key] = threading.Lock()

                self.translator = self._TRANSLATOR_CACHE[cache_key]
                self.translator_lock = self._TRANSLATOR_INFERENCE_LOCKS[cache_key]
            self.model_loaded = True
            logging.info(f"Translation model loaded successfully. Target language: {self.target_language}")
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

    def get_segment_source_language(self, segment):
        return HelsinkiZhEnTranslator.normalize_language(segment.get("language"))

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
        if text.endswith(tuple(self.translation_sentence_endings)):
            return True
        if len(text) >= self.translation_max_chars:
            return True
        if (
            self.translation_buffer_started_at is not None
            and len(text) >= self.translation_min_chars
            and time.monotonic() - self.translation_buffer_started_at >= self.translation_max_wait_seconds
        ):
            return True
        return False

    def add_segment_to_translation_buffer(self, segment):
        incoming_language = self.get_segment_source_language(segment)
        current_language = self.get_buffer_source_language()
        if self.translation_buffer and incoming_language and current_language and incoming_language != current_language:
            self.flush_translation_buffer(force=True)

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

        translated_text, source_language, target_language = self.translate_text(
            original_text,
            source_language,
        )

        translated_segment = {
            "start": buffered_segments[0]["start"],
            "end": buffered_segments[-1]["end"],
            "text": translated_text,
            "completed": True,
            "source_language": source_language,
            "target_language": target_language,
            "translation_model": self.model_name,
        }

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
        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "translated_segments": translated_segments,
                })
            )
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
        self.translator = None
        self.translator_lock = None
