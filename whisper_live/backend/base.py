import json
import logging
import re
import threading
import time
import queue
import numpy as np

from whisper_live import metrics as wl_metrics

try:
    from opencc import OpenCC
except ImportError:
    OpenCC = None


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    client_uid: str
    """A unique identifier for the client."""
    websocket: object
    """The WebSocket connection for the client."""
    send_last_n_segments: int
    """Number of most recent segments to send to the client."""
    no_speech_thresh: float
    """Segments with no speech probability above this threshold will be discarded."""
    clip_audio: bool
    """Whether to clip audio with no valid segments."""
    same_output_threshold: int
    """Number of repeated outputs before considering it as a valid segment."""
    min_segment_rms: float
    """Minimum RMS energy required before accepting an ASR segment."""
    max_incomplete_segment_seconds: float
    """Maximum audio duration to keep reprocessing a single incomplete segment."""

    MAX_TRANSCRIPT_LENGTH = 500
    MAX_TRANSLATION_QUEUE_SIZE = 100
    MAX_PENDING_AUDIO_SECONDS = 8.0
    OPENCC_CONFIG = "t2s"
    OPENCC_UNAVAILABLE_LOGGED = False
    SILENCE_HALLUCINATION_PHRASES = {
        "thank you",
        "thank you very much",
        "thanks",
        "thanks very much",
        "thanks for watching",
        "thank you for watching",
        "bye",
        "bye bye",
        "you",
    }
    GRATITUDE_HALLUCINATION_PHRASES = {
        "thank you",
        "thank you very much",
        "thanks",
        "thanks very much",
        "thanks for watching",
        "thank you for watching",
    }
    MAX_BOUNDARY_DEDUPE_WORDS = 6
    MAX_SHORT_GRATITUDE_SECONDS = 0.5

    def __init__(
        self,
        client_uid,
        websocket,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
        diarization=None,
        word_timestamps=False,
        min_segment_rms=0.0015,
        max_incomplete_segment_seconds=0.0,
        stable_utterance_ids=False,
    ):
        self.client_uid = client_uid
        self.websocket = websocket
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        if min_segment_rms is None:
            min_segment_rms = 0.0015
        self.min_segment_rms = max(0.0, float(min_segment_rms))
        if max_incomplete_segment_seconds is None:
            max_incomplete_segment_seconds = 0.0
        self.max_incomplete_segment_seconds = max(0.0, float(max_incomplete_segment_seconds))
        self.diarization = diarization
        self.word_timestamps = word_timestamps

        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ""
        self.prev_out = ""
        self.exit = False
        self.same_output_count = 0
        self.transcript = []
        self.end_time_for_same_output = None
        self.translation_queue = translation_queue
        self.admin_status_callback = None
        self.opencc_converter = self._create_opencc_converter()
        self.stable_utterance_ids = bool(stable_utterance_ids)
        self.utterance_sequence = 0
        self.current_utterance_id = None

        # Optional post-processing callable for segments.
        # If set, called with a segment dict and must return a segment dict.
        # Allows external projects to plug in custom post-processing
        # (e.g. PII redaction, formatting, diarization) without modifying
        # WhisperLive's core code.
        self.segment_post_processor = None

        # threading
        self.lock = threading.Lock()

    @classmethod
    def _log_opencc_unavailable_once(cls, message):
        if cls.OPENCC_UNAVAILABLE_LOGGED:
            return
        logging.warning(message)
        cls.OPENCC_UNAVAILABLE_LOGGED = True

    def _create_opencc_converter(self):
        if OpenCC is None:
            self._log_opencc_unavailable_once(
                "OpenCC is not installed; ASR text will not be converted to Simplified Chinese."
            )
            return None
        try:
            return OpenCC(self.OPENCC_CONFIG)
        except Exception as e:
            logging.warning("Failed to initialize OpenCC config %s: %s", self.OPENCC_CONFIG, e)
            return None

    def normalize_asr_text(self, text):
        if text is None or self.opencc_converter is None:
            return text
        try:
            return self.opencc_converter.convert(str(text))
        except Exception as e:
            logging.warning("OpenCC conversion failed: %s", e)
            return text

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            try:
                input_sample = input_bytes.copy()
                t0 = time.time()
                result = self.transcribe_audio(input_sample)

                if result is None or self.language is None:
                    self.timestamp_offset += duration
                    time.sleep(0.25)    # wait for voice activity, result is None when no voice activity
                    continue
                wl_metrics.track_transcription_latency(time.time() - t0)
                wl_metrics.track_audio_processed(duration)
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                wl_metrics.track_error("transcription")
                time.sleep(0.01)

    def transcribe_audio(self):
        raise NotImplementedError

    def handle_transcription_output(self, result, duration):
        raise NotImplementedError
    
    def format_segment(self, start, end, text, completed=False, speaker=None, words=None):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.
            speaker (str, optional): Speaker label from diarization.
            words (list, optional): Word-level timestamps and probabilities.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places and the 'text'
                of the transcription.
        """
        seg = {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': self.normalize_asr_text(text),
            'completed': completed,
            'language': getattr(self, "language", None)
        }
        if speaker is not None:
            seg['speaker'] = speaker
        if words is not None:
            seg['words'] = words
        return seg

    def _ensure_utterance_id(self, start):
        if not self.stable_utterance_ids:
            return None
        if self.current_utterance_id is None:
            self.utterance_sequence += 1
            self.current_utterance_id = (
                f"{self.client_uid}:{self.utterance_sequence}:{float(start):.3f}"
            )
        return self.current_utterance_id

    def _attach_utterance_id(self, segment, start, utterance_id=None):
        if not self.stable_utterance_ids:
            return segment
        segment["utterance_id"] = utterance_id or self._ensure_utterance_id(start)
        return segment

    def _finish_utterance(self):
        if self.stable_utterance_ids:
            self.current_utterance_id = None

    def add_frames(self, frame_np):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """
        self.lock.acquire()
        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            self.frames_offset += 30.0
            self.frames_np = self.frames_np[int(30*self.RATE):]
            # check timestamp offset(should be >= self.frame_offset)
            # this basically means that there is no speech as timestamp offset hasnt updated
            # and is less than frame_offset
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)
        self.trim_pending_audio_if_needed()
        self.lock.release()

    def trim_pending_audio_if_needed(self):
        if self.frames_np is None or self.MAX_PENDING_AUDIO_SECONDS <= 0:
            return

        frames_duration = self.frames_np.shape[0] / self.RATE
        processed_seconds = max(0.0, self.timestamp_offset - self.frames_offset)
        pending_seconds = max(0.0, frames_duration - processed_seconds)
        if pending_seconds <= self.MAX_PENDING_AUDIO_SECONDS:
            return

        new_timestamp_offset = self.frames_offset + frames_duration - self.MAX_PENDING_AUDIO_SECONDS
        dropped_seconds = max(0.0, new_timestamp_offset - self.timestamp_offset)
        if dropped_seconds <= 0:
            return

        self.timestamp_offset = new_timestamp_offset
        logging.warning(
            "[REALTIME_DROP] uid=%s pending=%.2fs keep=%.2fs dropped=%.2fs",
            self.client_uid,
            pending_seconds,
            self.MAX_PENDING_AUDIO_SECONDS,
            dropped_seconds,
        )

    def clip_audio_if_no_valid_segment(self):
        """
        Update the timestamp offset based on audio buffer status.
        Clip audio if the current chunk exceeds 30 seconds, this basically implies that
        no valid segment for the last 30 seconds from whisper
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing based on the current offsets.

        Calculates which part of the audio data should be processed next, based on
        the difference between the current timestamp offset and the frame's offset, scaled by
        the audio sample rate (RATE). It then returns this chunk of audio data along with its
        duration in seconds.

        Returns:
            tuple: A tuple containing:
                - input_bytes (np.ndarray): The next chunk of audio data to be processed.
                - duration (float): The duration of the audio chunk in seconds.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles the recent segments of transcribed text, ensuring that only the
        specified number of the most recent segments are included. It also appends the most
        recent segment of text if provided (which is considered incomplete because of the possibility
        of the last word being truncated in the audio chunk).

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def get_audio_chunk_duration(self, input_bytes):
        """
        Calculates the duration of the provided audio chunk.

        Args:
            input_bytes (numpy.ndarray): The audio chunk for which to calculate the duration.

        Returns:
            float: The duration of the audio chunk in seconds.
        """
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.

        This method formats the transcription segments into a JSON object and attempts to send
        this object to the client. If an error occurs during the send operation, it logs the error.

        If a ``segment_post_processor`` callable is set, each segment is passed through it
        before sending. The callable receives a segment dict and must return a segment dict.

        Returns:
            segments (list): A list of transcription segments to be sent to the client.
        """
        if self.segment_post_processor is not None:
            processed = []
            for seg in segments:
                try:
                    result = self.segment_post_processor(seg)
                    processed.append(result if result is not None else seg)
                except Exception as e:
                    logging.error(f"[ERROR]: segment_post_processor failed: {e}")
                    processed.append(seg)
            segments = processed

        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "segments": segments,
                })
            )
            if self.admin_status_callback:
                try:
                    self.admin_status_callback(segments)
                except Exception as e:
                    logging.error(f"[ERROR]: admin status update failed: {e}")
            for seg in segments:
                wl_metrics.track_segment_emitted(completed=seg.get("completed", False))
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def disconnect(self):
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.DISCONNECT
        }))

    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logging.info("Cleaning up.")
        self.exit = True
    
    def get_segment_no_speech_prob(self, segment):
        return getattr(segment, "no_speech_prob", 0)

    def get_segment_start(self, segment):
        return getattr(segment, "start", getattr(segment, "start_ts", 0))

    def get_segment_end(self, segment):
        return getattr(segment, "end", getattr(segment, "end_ts", 0))

    def _audio_rms_for_relative_range(self, start, end, duration):
        if self.min_segment_rms <= 0 or self.frames_np is None:
            return None

        start = max(0.0, min(float(start or 0.0), duration))
        end = max(start, min(float(end or 0.0), duration))
        if end <= start:
            return 0.0

        with self.lock:
            samples_offset = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            start_sample = samples_offset + int(start * self.RATE)
            end_sample = samples_offset + int(end * self.RATE)
            audio_slice = self.frames_np[start_sample:end_sample].copy()

        if audio_slice.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_slice, dtype=np.float64))))

    @classmethod
    def _normalized_phrase(cls, text):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s']+", " ", str(text or "").lower())).strip()

    @classmethod
    def _is_silence_hallucination_phrase(cls, text):
        normalized = cls._normalized_phrase(text)
        return normalized in cls.SILENCE_HALLUCINATION_PHRASES

    @classmethod
    def _is_gratitude_hallucination_phrase(cls, text):
        normalized = cls._normalized_phrase(text)
        return normalized in cls.GRATITUDE_HALLUCINATION_PHRASES

    def _is_probable_gratitude_hallucination(self, text, rel_start, rel_end, has_following_segment):
        if not self._is_gratitude_hallucination_phrase(text):
            return False

        duration = max(0.0, float(rel_end or 0.0) - float(rel_start or 0.0))
        if duration < self.MAX_SHORT_GRATITUDE_SECONDS:
            reason = "short"
        elif has_following_segment:
            reason = "middle"
        else:
            return False

        logging.info(
            "[GRATITUDE_HALLUCINATION_DROP] uid=%s reason=%s duration=%.3fs text=%r",
            self.client_uid,
            reason,
            duration,
            str(text or "").strip()[:80],
        )
        return True

    @staticmethod
    def _word_spans(text):
        return list(re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text or "")))

    @classmethod
    def _dedupe_leading_word_overlap(cls, previous_text, current_text):
        current = str(current_text or "")
        previous_words = [m.group(0).lower() for m in cls._word_spans(previous_text)]
        current_matches = cls._word_spans(current)
        current_words = [m.group(0).lower() for m in current_matches]
        max_overlap = min(cls.MAX_BOUNDARY_DEDUPE_WORDS, len(previous_words), len(current_words))
        for size in range(max_overlap, 0, -1):
            if previous_words[-size:] == current_words[:size]:
                cut_at = current_matches[size - 1].end()
                deduped = current[:current_matches[0].start()] + current[cut_at:].lstrip()
                logging.info(
                    "[BOUNDARY_DEDUPE] overlap_words=%d previous=%r current=%r deduped=%r",
                    size,
                    str(previous_text or "").strip()[-80:],
                    current.strip()[:80],
                    deduped.strip()[:80],
                )
                return deduped
        return current

    def _dedupe_completed_text(self, text):
        if not self.text:
            return text
        return self._dedupe_leading_word_overlap(self.text[-1], text)

    def _is_low_energy_range(self, start, end, duration, text):
        rms = self._audio_rms_for_relative_range(start, end, duration)
        if rms is None:
            return False

        threshold = self.min_segment_rms
        if self._is_silence_hallucination_phrase(text):
            threshold = max(threshold, self.min_segment_rms * 2.5)

        if rms >= threshold:
            return False
        logging.info(
            "[LOW_ENERGY_SEGMENT_DROP] uid=%s rms=%.6f threshold=%.6f text=%r",
            self.client_uid,
            rms,
            threshold,
            str(text or "").strip()[:80],
        )
        return True

    def _identify_speaker(self, segment):
        """Run diarization on a segment's audio slice if diarization is enabled.

        Returns:
            str or None: Speaker label, or None if diarization is disabled or audio unavailable.
        """
        if self.diarization is None or self.frames_np is None:
            return None
        try:
            seg_start = self.get_segment_start(segment)
            seg_end = self.get_segment_end(segment)
            start_sample = int(seg_start * self.RATE)
            end_sample = int(seg_end * self.RATE)
            samples_offset = max(0, int((self.timestamp_offset - self.frames_offset) * self.RATE))
            audio_slice = self.frames_np[samples_offset + start_sample:samples_offset + end_sample]
            if len(audio_slice) < self.RATE * 0.3:
                return None
            return self.diarization.identify_speaker(audio_slice, self.RATE)
        except Exception as e:
            logging.error(f"Diarization error: {e}")
            return None

    def _extract_words(self, segment, time_offset):
        """Extracts word-level timestamps from a segment if word_timestamps is enabled."""
        if not self.word_timestamps:
            return None
        words = getattr(segment, "words", None)
        if not words:
            return None
        return [
            {
                "word": self.normalize_asr_text(w.word),
                "start": "{:.3f}".format(time_offset + w.start),
                "end": "{:.3f}".format(time_offset + w.end),
                "probability": round(w.probability, 4),
            }
            for w in words
        ]

    def update_segments(self, segments, duration):
        """
        Processes the segments from Whisper and updates the transcript.
        Uses helper methods to account for differences between backends.
        
        Args:
            segments (list): List of segments returned by the transcriber.
            duration (float): Duration of the current audio chunk.
        
        Returns:
            dict or None: The last processed segment (if any).
        """
        offset = None
        self.current_out = ''
        last_segment = None

        # Process complete segments only if there are more than one
        # and if the last segment's no_speech_prob is below the threshold.
        if len(segments) > 1 and self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            completed_candidates = segments[:-1]
            completed_utterance_id = None
            if self.stable_utterance_ids and completed_candidates:
                with self.lock:
                    completed_start = self.timestamp_offset + self.get_segment_start(completed_candidates[0])
                completed_utterance_id = self._ensure_utterance_id(completed_start)
            for index, s in enumerate(completed_candidates):
                text_ = s.text
                rel_start = self.get_segment_start(s)
                rel_end = min(duration, self.get_segment_end(s))
                has_following_segment = index < len(completed_candidates) - 1 or len(segments) > 1
                with self.lock:
                    start = self.timestamp_offset + rel_start
                    end = self.timestamp_offset + rel_end
                if start >= end:
                    continue
                if self.get_segment_no_speech_prob(s) > self.no_speech_thresh:
                    continue
                if self._is_probable_gratitude_hallucination(text_, rel_start, rel_end, has_following_segment):
                    offset = rel_end
                    continue
                if self._is_low_energy_range(rel_start, rel_end, duration, text_):
                    offset = rel_end
                    continue
                text_ = self._dedupe_completed_text(text_)
                if not text_.strip():
                    offset = rel_end
                    continue
                self.text.append(text_)
                speaker = self._identify_speaker(s)
                words = self._extract_words(s, self.timestamp_offset)
                completed_segment = self.format_segment(start, end, text_, completed=True, speaker=speaker, words=words)
                self._attach_utterance_id(
                    completed_segment,
                    start,
                    utterance_id=completed_utterance_id,
                )
                self.transcript.append(completed_segment)

                if self.translation_queue:
                    try:
                        self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                    except queue.Full:
                        logging.warning("Translation queue is full, skipping segment")
                offset = rel_end
            self._finish_utterance()

        # Process the last segment if its no_speech_prob is acceptable.
        if self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            rel_start = self.get_segment_start(segments[-1])
            rel_end = min(duration, self.get_segment_end(segments[-1]))
            if self._is_probable_gratitude_hallucination(segments[-1].text, rel_start, rel_end, False):
                offset = rel_end
            elif self._is_low_energy_range(rel_start, rel_end, duration, segments[-1].text):
                offset = rel_end
            else:
                self.current_out += segments[-1].text
                words = self._extract_words(segments[-1], self.timestamp_offset)
                with self.lock:
                    last_segment = self.format_segment(
                        self.timestamp_offset + rel_start,
                        self.timestamp_offset + rel_end,
                        self.current_out,
                        completed=False,
                        words=words
                    )
                    self._attach_utterance_id(
                        last_segment,
                        self.timestamp_offset + rel_start,
                    )

        # Handle repeated output logic.
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_count += 1

            # if we remove the audio because of same output on the nth reptition we might remove the 
            # audio thats not yet transcribed so, capturing the time when it was repeated for the first time
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = self.get_segment_end(segments[-1])
            time.sleep(0.1)  # wait briefly for any new voice activity
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # If the same incomplete segment is repeated too many times,
        # append it to the transcript and update the offset.
        if self.same_output_count > self.same_output_threshold:
            repeated_end = min(duration, self.end_time_for_same_output)
            low_energy_repeated = self._is_low_energy_range(0.0, repeated_end, duration, self.current_out)
            if not low_energy_repeated and (not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower()):
                completed_text = self._dedupe_completed_text(self.current_out)
                if not completed_text.strip():
                    completed_text = ""
                if completed_text:
                    self.text.append(completed_text)
                    with self.lock:
                        completed_segment = self.format_segment(
                            self.timestamp_offset,
                            self.timestamp_offset + repeated_end,
                            completed_text,
                            completed=True
                        )
                        self._attach_utterance_id(
                            completed_segment,
                            self.timestamp_offset,
                        )
                        self.transcript.append(completed_segment)

                        if self.translation_queue:
                            try:
                                self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                            except queue.Full:
                                logging.warning("Translation queue is full, skipping segment")

            self.current_out = ''
            offset = repeated_end
            self.same_output_count = 0
            last_segment = None
            self._finish_utterance()
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        if (
            offset is None
            and last_segment is not None
            and self.max_incomplete_segment_seconds > 0
            and duration >= self.max_incomplete_segment_seconds
            and self.current_out.strip()
        ):
            repeated_end = min(duration, self.get_segment_end(segments[-1]))
            if repeated_end > 0:
                logging.info(
                    "[FORCE_COMPLETE_INCOMPLETE] uid=%s duration=%.2fs threshold=%.2fs text=%r",
                    self.client_uid,
                    duration,
                    self.max_incomplete_segment_seconds,
                    self.current_out.strip()[:80],
                )
                completed_text = self._dedupe_completed_text(self.current_out)
                if completed_text.strip():
                    with self.lock:
                        completed_segment = self.format_segment(
                            self.timestamp_offset,
                            self.timestamp_offset + repeated_end,
                            completed_text,
                            completed=True
                        )
                        self._attach_utterance_id(
                            completed_segment,
                            self.timestamp_offset,
                        )
                        self.transcript.append(completed_segment)

                        if self.translation_queue:
                            try:
                                self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                            except queue.Full:
                                logging.warning("Translation queue is full, skipping segment")

                    self.text.append(completed_text)
                self.current_out = ''
                offset = repeated_end
                self.same_output_count = 0
                last_segment = None
                self._finish_utterance()
                self.end_time_for_same_output = None

        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset
            if last_segment is None:
                self._finish_utterance()

        self._trim_transcript()
        return last_segment

    def _trim_transcript(self):
        """Trims transcript and text lists to prevent unbounded memory growth."""
        if len(self.transcript) > self.MAX_TRANSCRIPT_LENGTH:
            self.transcript = self.transcript[-self.MAX_TRANSCRIPT_LENGTH:]
        if len(self.text) > self.MAX_TRANSCRIPT_LENGTH:
            self.text = self.text[-self.MAX_TRANSCRIPT_LENGTH:]
