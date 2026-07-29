import os
import json
import logging
import re
import threading
import time
import torch
import ctranslate2
from huggingface_hub import snapshot_download

from whisper_live.transcriber.transcriber_faster_whisper import WhisperModel
from whisper_live.backend.base import ServeClientBase


class ServeClientFasterWhisper(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()
    SINGLE_MODEL_INIT_LOCK = threading.Lock()
    BATCH_WORKER = None

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="small.en",
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=7,
        cache_path="~/.cache/whisper-live/",
        translation_queue=None,
        hotwords=None,
        hotword_terms=None,
        diarization=None,
        word_timestamps=False,
        min_segment_rms=0.0015,
        max_incomplete_segment_seconds=0.0,
        sentence_completion_min_seconds=0.0,
        min_transcription_chunk_seconds=1.0,
        mixed_interpretation=False,
        mixed_language_retry_enabled=False,
        asr_device_index=0,
        max_pending_audio_seconds=ServeClientBase.MAX_PENDING_AUDIO_SECONDS,
    ):
        """
        Initialize a ServeClient instance.
        The Whisper model is initialized based on the client's language and device availability.
        The transcription thread is started upon initialization. A "SERVER_READY" message is sent
        to the client to indicate that the server is ready.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            task (str, optional): The task type, e.g., "transcribe". Defaults to "transcribe".
            device (str, optional): The device type for Whisper, "cuda" or "cpu". Defaults to None.
            language (str, optional): The language for transcription. Defaults to None.
            client_uid (str, optional): A unique identifier for the client. Defaults to None.
            model (str, optional): The whisper model size. Defaults to 'small.en'
            initial_prompt (str, optional): Prompt for whisper inference. Defaults to None.
            single_model (bool, optional): Whether to instantiate a new model for each client connection. Defaults to False.
            send_last_n_segments (int, optional): Number of most recent segments to send to the client. Defaults to 10.
            no_speech_thresh (float, optional): Segments with no speech probability above this threshold will be discarded. Defaults to 0.45.
            clip_audio (bool, optional): Whether to clip audio with no valid segments. Defaults to False.
            same_output_threshold (int, optional): Number of repeated outputs before considering it as a valid segment. Defaults to 10.

        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            translation_queue,
            diarization,
            word_timestamps,
            min_segment_rms=min_segment_rms,
            max_incomplete_segment_seconds=max_incomplete_segment_seconds,
            sentence_completion_min_seconds=sentence_completion_min_seconds,
            max_pending_audio_seconds=max_pending_audio_seconds,
            min_transcription_chunk_seconds=min_transcription_chunk_seconds,
            stable_utterance_ids=True,
            hotword_terms=hotword_terms,
        )
        self.cache_path = cache_path
        self.model_sizes = [
            "tiny", "tiny.en", "base", "base.en", "small", "small.en",
            "medium", "medium.en", "large-v2", "large-v3", "distil-small.en",
            "distil-medium.en", "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo"
        ]

        self.model_size_or_path = model
        self.language = "en" if self.model_size_or_path and self.model_size_or_path.endswith("en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"threshold": 0.5}
        self.use_vad = use_vad
        self.hotwords = hotwords
        self.mixed_interpretation = bool(mixed_interpretation)
        self.mixed_language_retry_enabled = bool(mixed_language_retry_enabled and self.mixed_interpretation)
        self.asr_device_index = int(asr_device_index or 0)
        self.allow_language_auto_per_chunk = self.mixed_interpretation
        self.current_language = None
        self.current_raw_language = None
        self.current_language_probability = 0.0
        self.current_zh_en_candidates = {}
        self.current_language_trusted = False
        logging.info(
            "[ASR_BUFFER_CONFIG] uid=%s max_pending=%.2f max_incomplete=%.2f sentence_min=%.2f min_chunk=%.2f",
            self.client_uid,
            self.max_pending_audio_seconds,
            self.max_incomplete_segment_seconds,
            self.sentence_completion_min_seconds,
            self.min_transcription_chunk_seconds,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(self.asr_device_index)
            self.compute_type = "float16" if major >= 7 else "float32"
        else:
            self.compute_type = "int8"

        if self.model_size_or_path is None:
            return
        logging.info(f"Using Device={device} index={self.asr_device_index} with precision {self.compute_type}")
    
        try:
            if single_model:
                with ServeClientFasterWhisper.SINGLE_MODEL_INIT_LOCK:
                    if ServeClientFasterWhisper.SINGLE_MODEL is None:
                        logging.info("Loading shared faster-whisper model")
                        self.create_model(device)
                        ServeClientFasterWhisper.SINGLE_MODEL = self.transcriber
                    else:
                        logging.info("Reusing shared faster-whisper model")
                        self.transcriber = ServeClientFasterWhisper.SINGLE_MODEL
            else:
                self.create_model(device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.websocket.send(json.dumps({
                "uid": self.client_uid,
                "status": "ERROR",
                "message": f"Failed to load model: {str(self.model_size_or_path)}"
            }))
            self.websocket.close()
            return

        self.use_vad = use_vad

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "faster_whisper"
                }
            )
        )

    def create_model(self, device):
        """
        Instantiates a new model, sets it as the transcriber. If model is a huggingface model_id
        then it is automatically converted to ctranslate2(faster_whisper) format.
        """
        model_ref = self.model_size_or_path

        if model_ref in self.model_sizes:
            model_to_load = model_ref
        else:
            logging.info(f"Model not in model_sizes")
            if os.path.isdir(model_ref) and ctranslate2.contains_model(model_ref):
                model_to_load = model_ref
            else:
                local_snapshot = snapshot_download(
                    repo_id = model_ref,
                    repo_type = "model",
                )
                if ctranslate2.contains_model(local_snapshot):
                    model_to_load = local_snapshot
                else:
                    cache_root = os.path.expanduser(os.path.join(self.cache_path, "whisper-ct2-models/"))
                    os.makedirs(cache_root, exist_ok=True)
                    safe_name = model_ref.replace("/", "--")
                    ct2_dir = os.path.join(cache_root, safe_name)

                    if not ctranslate2.contains_model(ct2_dir):
                        logging.info(f"Converting '{model_ref}' to CTranslate2 @ {ct2_dir}")
                        ct2_converter = ctranslate2.converters.TransformersConverter(
                            local_snapshot, 
                            copy_files=["tokenizer.json", "preprocessor_config.json"]
                        )
                        ct2_converter.convert(
                            output_dir=ct2_dir,
                            quantization=self.compute_type,
                            force=False,  # skip if already up-to-date
                        )
                    model_to_load = ct2_dir

        logging.info(f"Loading model: {model_to_load}")
        self.transcriber = WhisperModel(
            model_to_load,
            device=device,
            device_index=self.asr_device_index if device == "cuda" else 0,
            compute_type=self.compute_type,
            local_files_only=False,
        )

    def set_language(self, info):
        """
        Updates the language attribute based on the detected language information.

        Args:
            info (object): An object containing the detected language and its probability. This object
                        must have at least two attributes: `language`, a string indicating the detected
                        language, and `language_probability`, a float representing the confidence level
                        of the language detection.
        """
        if info.language_probability > 0.5:
            self.language = info.language
            logging.info(f"Detected language {self.language} with probability {info.language_probability}")
            self.websocket.send(json.dumps(
                {"uid": self.client_uid, "language": self.language, "language_prob": info.language_probability}))

    @staticmethod
    def normalize_mixed_interpretation_language(language):
        language = str(language or "").strip().lower()
        if language == "zh" or language.startswith("zh-"):
            return "zh"
        if language == "en" or language.startswith("en-"):
            return "en"
        return None

    @classmethod
    def zh_en_language_candidates(cls, info):
        candidates = {}
        language_probs = getattr(info, "all_language_probs", None)
        if not isinstance(language_probs, (list, tuple)):
            return candidates
        for item in language_probs:
            try:
                language, probability = item
            except (TypeError, ValueError):
                continue
            language = cls.normalize_mixed_interpretation_language(language)
            if not language:
                continue
            try:
                probability = float(probability)
            except (TypeError, ValueError):
                continue
            candidates[language] = max(candidates.get(language, 0.0), probability)
        return candidates

    def resolve_mixed_interpretation_language(self, info):
        previous_language = self.normalize_mixed_interpretation_language(self.current_language)
        if info is None:
            self.current_raw_language = None
            self.current_language_probability = 0.0
            self.current_zh_en_candidates = {}
            return previous_language

        raw_language = getattr(info, "language", None)
        self.current_raw_language = raw_language
        try:
            self.current_language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        except (TypeError, ValueError):
            self.current_language_probability = 0.0
        candidates = self.zh_en_language_candidates(info)
        self.current_zh_en_candidates = candidates

        detected_language = self.normalize_mixed_interpretation_language(raw_language)
        if detected_language:
            return detected_language

        if candidates:
            resolved_language = max(candidates.items(), key=lambda item: item[1])[0]
            logging.debug(
                "[MIXED_INTERPRETATION_LANGUAGE_CLAMP] uid=%s detected=%s resolved=%s zh=%.3f en=%.3f",
                self.client_uid,
                raw_language,
                resolved_language,
                candidates.get("zh", 0.0),
                candidates.get("en", 0.0),
            )
            return resolved_language

        if raw_language:
            logging.debug(
                "[MIXED_INTERPRETATION_LANGUAGE_IGNORE] uid=%s detected=%s previous=%s",
                self.client_uid,
                raw_language,
                previous_language,
            )
        return previous_language

    @staticmethod
    def _materialize_result(result):
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return list(result)

    @classmethod
    def _candidate_text(cls, result):
        return " ".join(str(getattr(segment, "text", "") or "").strip() for segment in result).strip()

    @staticmethod
    def _candidate_avg_logprob(result):
        values = []
        for segment in result or []:
            value = getattr(segment, "avg_logprob", None)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return sum(values) / len(values) if values else None

    @staticmethod
    def _candidate_no_speech(result):
        values = []
        for segment in result or []:
            value = getattr(segment, "no_speech_prob", None)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return max(values) if values else 0.0

    @staticmethod
    def _text_language_counts(text):
        text = str(text or "")
        return {
            "cjk": len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text)),
            "latin": len(re.findall(r"[A-Za-z]", text)),
        }

    @classmethod
    def _text_language_consistency(cls, text, language):
        language = cls.normalize_mixed_interpretation_language(language)
        counts = cls._text_language_counts(text)
        if not text:
            return 0.0
        if language == "zh":
            if counts["cjk"] <= 0:
                return 0.0
            return counts["cjk"] / max(counts["cjk"] + counts["latin"], 1)
        if language == "en":
            if counts["latin"] < 4 or counts["cjk"] > 0:
                return 0.0
            return counts["latin"] / max(counts["cjk"] + counts["latin"], 1)
        return 0.0

    def _mixed_language_candidate(self, result, info, language):
        result = self._materialize_result(result)
        text = self._candidate_text(result)
        return {
            "result": result,
            "info": info,
            "language": self.normalize_mixed_interpretation_language(language),
            "text": text,
            "consistency": self._text_language_consistency(text, language),
            "avg_logprob": self._candidate_avg_logprob(result),
            "no_speech_prob": self._candidate_no_speech(result),
            "hard_noise": self._is_hard_drop_hallucination_text(text) or self._is_mixed_interpretation_noise_text(text),
        }

    def _should_accept_automatic_language_switch(self, candidate, previous_language):
        language = candidate.get("language")
        if not previous_language or not language or language == previous_language:
            return True
        if candidate.get("hard_noise"):
            return False
        auto_prob = self.current_zh_en_candidates.get(language, self.current_language_probability)
        prev_prob = self.current_zh_en_candidates.get(previous_language, 0.0)
        avg_logprob = candidate.get("avg_logprob")
        if auto_prob >= 0.80 and auto_prob - prev_prob >= 0.30 and candidate.get("consistency", 0.0) >= 0.85:
            return True
        if avg_logprob is not None and avg_logprob > -0.35 and candidate.get("consistency", 0.0) >= 0.95:
            return True
        return False

    def _choose_mixed_language_candidate(self, automatic, retry, previous_language):
        auto_logprob = automatic.get("avg_logprob")
        retry_logprob = retry.get("avg_logprob")
        auto_score = automatic.get("consistency", 0.0) - (0.5 if automatic.get("hard_noise") else 0.0)
        retry_score = retry.get("consistency", 0.0) - (0.5 if retry.get("hard_noise") else 0.0)
        if auto_logprob is not None and retry_logprob is not None:
            auto_score += max(-1.0, min(1.0, auto_logprob + 1.0))
            retry_score += max(-1.0, min(1.0, retry_logprob + 1.0))
        selected = retry if retry_score >= auto_score - 0.15 else automatic
        logging.info(
            "[MIXED_LANGUAGE_RETRY_SELECTED] uid=%s previous=%s automatic=%s retry=%s selected=%s "
            "auto_score=%.3f retry_score=%.3f auto_logprob=%s retry_logprob=%s auto_text=%r retry_text=%r",
            self.client_uid,
            previous_language,
            automatic.get("language"),
            retry.get("language"),
            selected.get("language"),
            auto_score,
            retry_score,
            "none" if auto_logprob is None else f"{auto_logprob:.3f}",
            "none" if retry_logprob is None else f"{retry_logprob:.3f}",
            automatic.get("text", "")[:80],
            retry.get("text", "")[:80],
        )
        return selected

    def _submit_batch_transcription(self, input_sample, language):
        from whisper_live.batch_inference import BatchRequest
        request = BatchRequest(
            audio=input_sample,
            language=language,
            task=self.task,
            initial_prompt=self.initial_prompt,
            hotwords=self.hotwords,
            use_vad=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None,
            word_timestamps=self.word_timestamps,
        )
        ServeClientFasterWhisper.BATCH_WORKER.submit(request)
        request.future.wait(timeout=30)
        if request.error:
            raise request.error
        return self._materialize_result(request.result), request.info

    def _direct_transcription(self, input_sample, language):
        if ServeClientFasterWhisper.SINGLE_MODEL:
            ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()
        try:
            result, info = self.transcriber.transcribe(
                input_sample,
                initial_prompt=self.initial_prompt,
                language=language,
                task=self.task,
                vad_filter=self.use_vad,
                vad_parameters=self.vad_parameters if self.use_vad else None,
                hotwords=self.hotwords,
                word_timestamps=self.word_timestamps,
            )
        finally:
            if ServeClientFasterWhisper.SINGLE_MODEL:
                ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()
        return self._materialize_result(result), info

    def _maybe_retry_mixed_language(self, input_sample, result, info, retry_callback):
        previous_language = self.normalize_mixed_interpretation_language(self.current_language)
        resolved_language = self.resolve_mixed_interpretation_language(info)
        automatic = self._mixed_language_candidate(result, info, resolved_language)
        if (
            not self.mixed_language_retry_enabled
            or not previous_language
            or not resolved_language
            or previous_language == resolved_language
            or self._should_accept_automatic_language_switch(automatic, previous_language)
        ):
            if previous_language and resolved_language and previous_language != resolved_language:
                logging.info(
                    "[MIXED_LANGUAGE_SWITCH_ACCEPTED] uid=%s previous=%s selected=%s prob=%.3f text=%r",
                    self.client_uid,
                    previous_language,
                    resolved_language,
                    self.current_language_probability,
                    automatic.get("text", "")[:80],
                )
            self.current_language = resolved_language
            self.current_language_trusted = bool(resolved_language)
            return automatic["result"]

        logging.info(
            "[MIXED_LANGUAGE_SWITCH_CANDIDATE] uid=%s previous=%s automatic=%s prob=%.3f zh=%.3f en=%.3f text=%r",
            self.client_uid,
            previous_language,
            resolved_language,
            self.current_language_probability,
            self.current_zh_en_candidates.get("zh", 0.0),
            self.current_zh_en_candidates.get("en", 0.0),
            automatic.get("text", "")[:80],
        )
        retry_result, retry_info = retry_callback(input_sample, previous_language)
        retry = self._mixed_language_candidate(retry_result, retry_info, previous_language)
        logging.info(
            "[MIXED_LANGUAGE_RETRY] uid=%s previous=%s retry_text=%r",
            self.client_uid,
            previous_language,
            retry.get("text", "")[:80],
        )
        selected = self._choose_mixed_language_candidate(automatic, retry, previous_language)
        self.current_language = selected.get("language")
        self.current_language_trusted = bool(self.current_language)
        return selected["result"]

    def transcribe_audio(self, input_sample):
        """
        Transcribes the provided audio sample using the configured transcriber instance.

        If the language has not been set, it updates the session's language based on the transcription
        information.

        Args:
            input_sample (np.array): The audio chunk to be transcribed. This should be a NumPy
                                    array representing the audio data.

        Returns:
            The transcription result from the transcriber. The exact format of this result
            depends on the implementation of the `transcriber.transcribe` method but typically
            includes the transcribed text.
        """
        # Batch inference path: submit to central queue and wait
        if ServeClientFasterWhisper.BATCH_WORKER is not None:
            result, info = self._submit_batch_transcription(
                input_sample,
                None if self.mixed_interpretation else self.language,
            )
            if self.mixed_interpretation:
                return self._maybe_retry_mixed_language(input_sample, result, info, self._submit_batch_transcription)
            if not self.mixed_interpretation and self.language is None and info is not None:
                self.set_language(info)
            return result

        # Original lock-based path (backward compatible)
        result, info = self._direct_transcription(
            input_sample,
            language=None if self.mixed_interpretation else self.language,
        )
        if self.mixed_interpretation:
            return self._maybe_retry_mixed_language(input_sample, result, info, self._direct_transcription)
        if not self.mixed_interpretation and self.language is None and info is not None:
            self.set_language(info)
        return result

    def handle_transcription_output(self, result, duration, force_complete_last=False):
        """
        Handle the transcription output, updating the transcript and sending data to the client.

        Args:
            result (str): The result from whisper inference i.e. the list of segments.
            duration (float): Duration of the transcribed audio chunk.
        """
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration, force_complete_last=force_complete_last)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)
