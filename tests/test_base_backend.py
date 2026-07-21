import json
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from whisper_live.backend.base import ServeClientBase


class FakeOpenCCConverter:
    def convert(self, text):
        return str(text).replace("繁體", "繁体").replace("臺灣", "台湾")


class ConcreteServeClient(ServeClientBase):
    """Concrete subclass for testing the abstract base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.language = "en"

    def transcribe_audio(self, input_sample):
        return None

    def handle_transcription_output(self, result, duration):
        pass


class TestServeClientBaseInit(unittest.TestCase):
    def test_default_values(self):
        ws = MagicMock()
        client = ConcreteServeClient(client_uid="test-uid", websocket=ws)
        self.assertEqual(client.client_uid, "test-uid")
        self.assertEqual(client.send_last_n_segments, 10)
        self.assertAlmostEqual(client.no_speech_thresh, 0.45)
        self.assertFalse(client.clip_audio)
        self.assertEqual(client.same_output_threshold, 10)
        self.assertAlmostEqual(client.min_segment_rms, 0.0015)
        self.assertAlmostEqual(client.min_transcription_chunk_seconds, 1.0)
        self.assertEqual(client.hotword_match_terms, ())
        self.assertIsNone(client.frames_np)
        self.assertAlmostEqual(client.timestamp_offset, 0.0)
        self.assertFalse(client.exit)
        self.assertEqual(client.transcript, [])

    def test_custom_values(self):
        ws = MagicMock()
        q = queue.Queue()
        client = ConcreteServeClient(
            client_uid="uid2",
            websocket=ws,
            send_last_n_segments=5,
            no_speech_thresh=0.6,
            clip_audio=True,
            same_output_threshold=20,
            min_segment_rms=0.002,
            min_transcription_chunk_seconds=2.5,
            translation_queue=q,
        )
        self.assertEqual(client.send_last_n_segments, 5)
        self.assertAlmostEqual(client.no_speech_thresh, 0.6)
        self.assertTrue(client.clip_audio)
        self.assertEqual(client.same_output_threshold, 20)
        self.assertAlmostEqual(client.min_segment_rms, 0.002)
        self.assertAlmostEqual(client.min_transcription_chunk_seconds, 2.5)
        self.assertIs(client.translation_queue, q)

    def test_hotword_match_terms_are_precomputed(self):
        client = ConcreteServeClient(
            client_uid="uid3",
            websocket=MagicMock(),
            hotword_terms=["ＯｐｅｎＡＩ", "Whisper small", "我"],
        )

        self.assertEqual(client.hotword_match_terms, ("whispersmall", "openai"))


class AutoLanguageClient(ConcreteServeClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_calls = []

    def transcribe_audio(self, input_sample):
        self.exit = True
        return ["segment"]

    def handle_transcription_output(self, result, duration):
        self.output_calls.append((result, duration))


class TestSpeechToTextLanguageGate(unittest.TestCase):
    def test_min_transcription_chunk_waits_for_configured_duration(self):
        client = AutoLanguageClient(client_uid="test", websocket=MagicMock())
        client.min_transcription_chunk_seconds = 2.5
        client.frames_np = np.zeros(16000, dtype=np.float32)

        with patch("whisper_live.backend.base.time.sleep", side_effect=lambda _seconds: setattr(client, "exit", True)):
            client.speech_to_text()

        self.assertEqual(client.output_calls, [])

    def test_auto_language_per_chunk_does_not_drop_result(self):
        client = AutoLanguageClient(client_uid="test", websocket=MagicMock())
        client.language = None
        client.allow_language_auto_per_chunk = True
        client.frames_np = np.zeros(16000, dtype=np.float32)

        client.speech_to_text()

        self.assertEqual(len(client.output_calls), 1)
        self.assertEqual(client.output_calls[0][0], ["segment"])


class TestAddFrames(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_first_frame_initializes_buffer(self):
        frame = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        self.client.add_frames(frame)
        np.testing.assert_array_equal(self.client.frames_np, frame)

    def test_subsequent_frames_concatenated(self):
        frame1 = np.array([0.1, 0.2], dtype=np.float32)
        frame2 = np.array([0.3, 0.4], dtype=np.float32)
        self.client.add_frames(frame1)
        self.client.add_frames(frame2)
        expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        np.testing.assert_array_equal(self.client.frames_np, expected)

    def test_buffer_trimmed_at_45_seconds(self):
        # 45 seconds + 1 sample at 16kHz = 720001 samples
        self.client.frames_np = np.zeros(45 * 16000 + 1, dtype=np.float32)
        self.client.add_frames(np.array([1.0], dtype=np.float32))
        # after trimming 30s, buffer should be ~15s + 1 original + 1 new
        expected_len = (45 * 16000 + 1) - (30 * 16000) + 1
        self.assertEqual(self.client.frames_np.shape[0], expected_len)
        self.assertAlmostEqual(self.client.frames_offset, 30.0)

    def test_timestamp_offset_updated_on_trim(self):
        self.client.frames_np = np.zeros(45 * 16000 + 1, dtype=np.float32)
        self.client.timestamp_offset = 5.0  # behind frames_offset after trim
        self.client.add_frames(np.array([1.0], dtype=np.float32))
        # timestamp_offset should be bumped to at least frames_offset
        self.assertGreaterEqual(self.client.timestamp_offset, self.client.frames_offset)


class TestAddFramesThreadSafety(unittest.TestCase):
    def test_concurrent_add_frames(self):
        ws = MagicMock()
        client = ConcreteServeClient(client_uid="test", websocket=ws)
        errors = []

        def add_many():
            try:
                for _ in range(100):
                    client.add_frames(np.random.randn(160).astype(np.float32))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertIsNotNone(client.frames_np)


class TestGetAudioChunkForProcessing(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_empty_buffer_returns_empty(self):
        self.client.frames_np = np.array([], dtype=np.float32)
        chunk, duration = self.client.get_audio_chunk_for_processing()
        self.assertEqual(duration, 0.0)
        self.assertEqual(chunk.shape[0], 0)

    def test_full_buffer_no_offset(self):
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        self.client.frames_np = audio
        chunk, duration = self.client.get_audio_chunk_for_processing()
        self.assertAlmostEqual(duration, 1.0)
        np.testing.assert_array_equal(chunk, audio)

    def test_with_offset(self):
        audio = np.random.randn(32000).astype(np.float32)  # 2 seconds
        self.client.frames_np = audio
        self.client.timestamp_offset = 1.0  # skip first second
        chunk, duration = self.client.get_audio_chunk_for_processing()
        self.assertAlmostEqual(duration, 1.0)
        self.assertEqual(chunk.shape[0], 16000)


class TestClipAudioIfNoValidSegment(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(
            client_uid="test", websocket=self.ws, clip_audio=True
        )

    def test_clips_when_chunk_exceeds_25s(self):
        # 30 seconds of audio with no valid segments
        self.client.frames_np = np.zeros(30 * 16000, dtype=np.float32)
        self.client.timestamp_offset = 0.0
        self.client.frames_offset = 0.0
        self.client.clip_audio_if_no_valid_segment()
        # offset should have advanced to leave ~5s of remaining audio
        expected_offset = (30 * 16000 / 16000) - 5
        self.assertAlmostEqual(self.client.timestamp_offset, expected_offset, places=1)

    def test_no_clip_when_short(self):
        self.client.frames_np = np.zeros(10 * 16000, dtype=np.float32)
        self.client.timestamp_offset = 0.0
        self.client.frames_offset = 0.0
        self.client.clip_audio_if_no_valid_segment()
        self.assertAlmostEqual(self.client.timestamp_offset, 0.0)


class TestPrepareSegments(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(
            client_uid="test", websocket=self.ws, send_last_n_segments=3
        )

    def test_empty_transcript_no_last(self):
        segments = self.client.prepare_segments()
        self.assertEqual(segments, [])

    def test_empty_transcript_with_last(self):
        last = {"start": "0.000", "end": "1.000", "text": "hello", "completed": False}
        segments = self.client.prepare_segments(last_segment=last)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "hello")

    def test_fewer_than_n_segments(self):
        self.client.transcript = [
            {"start": "0.000", "end": "1.000", "text": "a", "completed": True},
            {"start": "1.000", "end": "2.000", "text": "b", "completed": True},
        ]
        segments = self.client.prepare_segments()
        self.assertEqual(len(segments), 2)

    def test_more_than_n_segments_truncated(self):
        self.client.transcript = [
            {"start": f"{i}.000", "end": f"{i+1}.000", "text": f"seg{i}", "completed": True}
            for i in range(10)
        ]
        segments = self.client.prepare_segments()
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["text"], "seg7")

    def test_last_segment_appended(self):
        self.client.transcript = [
            {"start": "0.000", "end": "1.000", "text": "a", "completed": True},
        ]
        last = {"start": "1.000", "end": "2.000", "text": "in progress", "completed": False}
        segments = self.client.prepare_segments(last_segment=last)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[-1]["text"], "in progress")


class TestFormatSegment(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_format(self):
        seg = self.client.format_segment(1.234, 5.678, "hello world", completed=True)
        self.assertEqual(seg["start"], "1.234")
        self.assertEqual(seg["end"], "5.678")
        self.assertEqual(seg["text"], "hello world")
        self.assertTrue(seg["completed"])

    def test_format_not_completed(self):
        seg = self.client.format_segment(0.0, 1.0, "text")
        self.assertFalse(seg["completed"])

    def test_format_converts_asr_text_to_simplified(self):
        self.client.opencc_converter = FakeOpenCCConverter()
        seg = self.client.format_segment(0.0, 1.0, "繁體中文和臺灣", completed=True)
        self.assertEqual(seg["text"], "繁体中文和台湾")


class TestSendTranscriptionToClient(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test-uid", websocket=self.ws)

    def test_sends_json(self):
        segments = [{"start": "0.000", "end": "1.000", "text": "hi", "completed": True}]
        self.client.send_transcription_to_client(segments)
        self.ws.send.assert_called_once()
        sent = json.loads(self.ws.send.call_args[0][0])
        self.assertEqual(sent["uid"], "test-uid")
        self.assertEqual(len(sent["segments"]), 1)

    def test_send_failure_logged_not_raised(self):
        self.ws.send.side_effect = ConnectionError("broken pipe")
        # should not raise
        self.client.send_transcription_to_client([])


class TestDisconnect(unittest.TestCase):
    def test_sends_disconnect_message(self):
        ws = MagicMock()
        client = ConcreteServeClient(client_uid="uid1", websocket=ws)
        client.disconnect()
        sent = json.loads(ws.send.call_args[0][0])
        self.assertEqual(sent["uid"], "uid1")
        self.assertEqual(sent["message"], "DISCONNECT")


class TestCleanup(unittest.TestCase):
    def test_sets_exit_flag(self):
        ws = MagicMock()
        client = ConcreteServeClient(client_uid="uid1", websocket=ws)
        self.assertFalse(client.exit)
        client.cleanup()
        self.assertTrue(client.exit)


class TestTrimTranscript(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_transcript_trimmed_when_over_max(self):
        self.client.transcript = [
            {"start": f"{i}.000", "end": f"{i+1}.000", "text": f"seg{i}", "completed": True}
            for i in range(self.client.MAX_TRANSCRIPT_LENGTH + 100)
        ]
        self.client._trim_transcript()
        self.assertEqual(len(self.client.transcript), self.client.MAX_TRANSCRIPT_LENGTH)
        self.assertEqual(self.client.transcript[0]["text"], "seg100")

    def test_transcript_not_trimmed_when_under_max(self):
        self.client.transcript = [
            {"start": "0.000", "end": "1.000", "text": "a", "completed": True}
        ]
        self.client._trim_transcript()
        self.assertEqual(len(self.client.transcript), 1)

    def test_text_list_trimmed(self):
        self.client.text = ["word"] * (self.client.MAX_TRANSCRIPT_LENGTH + 50)
        self.client._trim_transcript()
        self.assertEqual(len(self.client.text), self.client.MAX_TRANSCRIPT_LENGTH)


class TestUpdateSegments(unittest.TestCase):
    """Tests for the core update_segments() logic."""

    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(
            client_uid="test",
            websocket=self.ws,
            no_speech_thresh=0.45,
            same_output_threshold=3,
        )
        self.client.frames_np = np.full(16000 * 5, 0.01, dtype=np.float32)

    def _make_segment(self, start, end, text, no_speech_prob=0.0):
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = text
        seg.no_speech_prob = no_speech_prob
        return seg

    def test_single_segment_becomes_last(self):
        segs = [self._make_segment(0.0, 1.0, " hello")]
        last = self.client.update_segments(segs, duration=2.0)
        self.assertIsNotNone(last)
        self.assertIn("hello", last["text"])
        self.assertFalse(last["completed"])
        self.assertEqual(len(self.client.transcript), 0)

    def test_multiple_segments_completes_all_but_last(self):
        segs = [
            self._make_segment(0.0, 1.0, " first"),
            self._make_segment(1.0, 2.0, " second"),
        ]
        last = self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 1)
        self.assertTrue(self.client.transcript[0]["completed"])
        self.assertIn("first", self.client.transcript[0]["text"])
        self.assertIsNotNone(last)
        self.assertIn("second", last["text"])

    def test_high_no_speech_prob_skipped(self):
        segs = [
            self._make_segment(0.0, 1.0, " noise", no_speech_prob=0.9),
            self._make_segment(1.0, 2.0, " also noise", no_speech_prob=0.9),
        ]
        last = self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertIsNone(last)

    def test_segment_with_start_gte_end_skipped(self):
        segs = [
            self._make_segment(1.0, 0.5, " backwards"),
            self._make_segment(1.5, 2.0, " normal"),
        ]
        last = self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertIsNotNone(last)

    def test_repeated_output_triggers_completion(self):
        seg = self._make_segment(0.0, 1.0, " repeated")
        for _ in range(self.client.same_output_threshold + 2):
            last = self.client.update_segments([seg], duration=2.0)
        # after enough repeats, should be added to transcript
        self.assertTrue(len(self.client.transcript) >= 1)

    def test_translation_queue_receives_completed(self):
        q = queue.Queue()
        self.client.translation_queue = q
        segs = [
            self._make_segment(0.0, 1.0, " first"),
            self._make_segment(1.0, 2.0, " second"),
        ]
        self.client.update_segments(segs, duration=3.0)
        self.assertFalse(q.empty())
        item = q.get_nowait()
        self.assertIn("first", item["text"])

    def test_stable_utterance_id_survives_partial_completion(self):
        self.client.stable_utterance_ids = True
        partial = self.client.update_segments(
            [self._make_segment(0.0, 1.0, " first")],
            duration=2.0,
        )
        partial_id = partial["utterance_id"]

        latest = self.client.update_segments(
            [
                self._make_segment(0.0, 1.0, " first"),
                self._make_segment(1.0, 2.0, " second"),
            ],
            duration=3.0,
        )

        self.assertEqual(self.client.transcript[0]["utterance_id"], partial_id)
        self.assertNotEqual(latest["utterance_id"], partial_id)

    def test_split_completed_segments_share_utterance_id(self):
        self.client.stable_utterance_ids = True
        partial = self.client.update_segments(
            [self._make_segment(0.0, 2.0, " combined")],
            duration=3.0,
        )

        self.client.update_segments(
            [
                self._make_segment(0.0, 1.0, " first"),
                self._make_segment(1.0, 2.0, " second"),
                self._make_segment(2.0, 2.5, " next"),
            ],
            duration=3.0,
        )

        completed_ids = [
            segment["utterance_id"]
            for segment in self.client.transcript[-2:]
        ]
        self.assertEqual(completed_ids, [partial["utterance_id"]] * 2)

    def test_stable_utterance_id_reaches_translation_queue(self):
        self.client.stable_utterance_ids = True
        self.client.translation_queue = queue.Queue()
        partial = self.client.update_segments(
            [self._make_segment(0.0, 1.0, " first")],
            duration=2.0,
        )
        self.client.update_segments(
            [
                self._make_segment(0.0, 1.0, " first"),
                self._make_segment(1.0, 2.0, " second"),
            ],
            duration=3.0,
        )

        translated_source = self.client.translation_queue.get_nowait()
        self.assertEqual(translated_source["utterance_id"], partial["utterance_id"])

    def test_translation_queue_receives_simplified_completed_text(self):
        q = queue.Queue()
        self.client.translation_queue = q
        self.client.opencc_converter = FakeOpenCCConverter()
        segs = [
            self._make_segment(0.0, 1.0, " 繁體中文"),
            self._make_segment(1.0, 2.0, " second"),
        ]
        self.client.update_segments(segs, duration=3.0)
        item = q.get_nowait()
        self.assertIn("繁体中文", item["text"])

    def test_low_energy_completed_segment_is_dropped(self):
        q = queue.Queue()
        self.client.translation_queue = q
        self.client.frames_np = np.zeros(16000 * 5, dtype=np.float32)
        segs = [
            self._make_segment(0.0, 1.0, " hotword"),
            self._make_segment(1.0, 2.0, " next"),
        ]
        last = self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertTrue(q.empty())
        self.assertIsNone(last)
        self.assertGreater(self.client.timestamp_offset, 0.0)

    def test_low_energy_incomplete_segment_does_not_repeat(self):
        self.client.frames_np = np.zeros(16000 * 5, dtype=np.float32)
        self.client.prev_out = " hotword"
        seg = self._make_segment(0.0, 1.0, " hotword")
        last = self.client.update_segments([seg], duration=2.0)
        self.assertIsNone(last)
        self.assertEqual(self.client.same_output_count, 0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertGreater(self.client.timestamp_offset, 0.0)

    def test_min_segment_rms_zero_disables_low_energy_filter(self):
        self.client.min_segment_rms = 0
        self.client.frames_np = np.zeros(16000 * 5, dtype=np.float32)
        seg = self._make_segment(0.0, 1.0, " quiet speech")
        last = self.client.update_segments([seg], duration=2.0)
        self.assertIsNotNone(last)
        self.assertIn("quiet speech", last["text"])

    def test_hotword_dominated_completed_segment_with_weak_no_speech_is_dropped(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["OpenAI"])
        segs = [
            self._make_segment(0.0, 1.0, " OpenAI", no_speech_prob=0.40),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        with self.assertLogs(level="INFO") as logs:
            last = self.client.update_segments(segs, duration=3.0)

        self.assertEqual(len(self.client.transcript), 0)
        self.assertIsNotNone(last)
        self.assertIn("HOTWORD_HALLUCINATION_DROP", "\n".join(logs.output))
        self.assertGreater(self.client.timestamp_offset, 0.0)

    def test_hotword_dominated_partial_segment_with_weak_no_speech_waits_for_more_audio(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["OpenAI"])
        seg = self._make_segment(0.0, 1.0, " OpenAI", no_speech_prob=0.40)

        last = self.client.update_segments([seg], duration=2.0)

        self.assertIsNone(last)
        self.assertEqual(self.client.timestamp_offset, 0.0)
        self.assertEqual(self.client.current_out, "")
        self.assertEqual(len(self.client.transcript), 0)

    def test_spoken_hotword_with_strong_evidence_is_retained(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["OpenAI"])
        segs = [
            self._make_segment(0.0, 1.0, " OpenAI", no_speech_prob=0.10),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        self.client.update_segments(segs, duration=3.0)

        self.assertEqual(len(self.client.transcript), 1)
        self.assertIn("OpenAI", self.client.transcript[0]["text"])

    def test_sentence_containing_hotword_is_not_hotword_dominated(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["OpenAI"])
        seg = self._make_segment(0.0, 1.0, " we discuss OpenAI models today", no_speech_prob=0.40)

        last = self.client.update_segments([seg], duration=2.0)

        self.assertIsNotNone(last)
        self.assertIn("OpenAI", last["text"])

    def test_hotword_dominated_low_energy_completed_segment_is_logged_separately(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["OpenAI"])
        self.client.frames_np = np.zeros(16000 * 5, dtype=np.float32)
        segs = [
            self._make_segment(0.0, 1.0, " OpenAI", no_speech_prob=0.10),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        with self.assertLogs(level="INFO") as logs:
            self.client.update_segments(segs, duration=3.0)

        output = "\n".join(logs.output)
        self.assertIn("HOTWORD_HALLUCINATION_DROP", output)
        self.assertIn("reason=low_energy", output)

    def test_timestamp_offset_advances(self):
        segs = [
            self._make_segment(0.0, 1.0, " first"),
            self._make_segment(1.0, 2.0, " second"),
        ]
        self.client.update_segments(segs, duration=3.0)
        self.assertGreater(self.client.timestamp_offset, 0.0)

    def test_boundary_duplicate_words_are_trimmed(self):
        self.client.text = ["This is the end of the first sentence"]
        segs = [
            self._make_segment(0.0, 1.0, " the first sentence starts cleanly now"),
            self._make_segment(1.0, 2.0, " next"),
        ]
        self.client.update_segments(segs, duration=3.0)
        self.assertEqual(
            self.client.transcript[0]["text"].strip(),
            "starts cleanly now",
        )

    def test_low_energy_thank_you_is_dropped_with_stricter_threshold(self):
        self.client.frames_np = np.full(16000 * 5, 0.003, dtype=np.float32)
        segs = [
            self._make_segment(0.0, 1.0, " Thank you."),
            self._make_segment(1.0, 2.0, " next"),
        ]
        self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertGreater(self.client.timestamp_offset, 0.0)

    def test_short_thank_you_is_dropped_even_with_normal_energy(self):
        q = queue.Queue()
        self.client.translation_queue = q
        self.client.frames_np = np.full(16000 * 5, 0.02, dtype=np.float32)
        segs = [
            self._make_segment(0.0, 0.2, " Thank you."),
            self._make_segment(0.2, 1.0, " next"),
        ]
        self.client.update_segments(segs, duration=2.0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertTrue(q.empty())
        self.assertGreater(self.client.timestamp_offset, 0.0)

    def test_middle_thank_you_is_dropped(self):
        q = queue.Queue()
        self.client.translation_queue = q
        self.client.frames_np = np.full(16000 * 5, 0.02, dtype=np.float32)
        segs = [
            self._make_segment(0.0, 1.0, " The second thing you want to do"),
            self._make_segment(1.0, 2.0, " Thank you."),
            self._make_segment(2.0, 3.0, " build relationships"),
        ]
        last = self.client.update_segments(segs, duration=4.0)
        self.assertEqual(len(self.client.transcript), 1)
        self.assertIn("second thing", self.client.transcript[0]["text"])
        self.assertNotIn("Thank you", [item.get("text", "") for item in list(q.queue)])
        self.assertIsNotNone(last)
        self.assertIn("build relationships", last["text"])

    def test_trailing_normal_thank_you_is_kept_as_incomplete(self):
        self.client.frames_np = np.full(16000 * 5, 0.02, dtype=np.float32)
        segs = [self._make_segment(0.0, 1.0, " Thank you.")]
        last = self.client.update_segments(segs, duration=2.0)
        self.assertIsNotNone(last)
        self.assertIn("Thank you", last["text"])
        self.assertEqual(len(self.client.transcript), 0)


class TestGetSegmentHelpers(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.client = ConcreteServeClient(client_uid="test", websocket=self.ws)

    def test_get_segment_no_speech_prob_attr(self):
        seg = MagicMock()
        seg.no_speech_prob = 0.3
        self.assertAlmostEqual(self.client.get_segment_no_speech_prob(seg), 0.3)

    def test_get_segment_no_speech_prob_fallback(self):
        seg = MagicMock(spec=[])  # no attributes
        self.assertEqual(self.client.get_segment_no_speech_prob(seg), 0)

    def test_get_segment_start_uses_start(self):
        seg = MagicMock()
        seg.start = 1.5
        self.assertAlmostEqual(self.client.get_segment_start(seg), 1.5)

    def test_get_segment_end_uses_end(self):
        seg = MagicMock()
        seg.end = 3.0
        self.assertAlmostEqual(self.client.get_segment_end(seg), 3.0)

    def test_get_segment_start_fallback_to_start_ts(self):
        seg = MagicMock(spec=["start_ts"])
        seg.start_ts = 2.0
        self.assertAlmostEqual(self.client.get_segment_start(seg), 2.0)


class TestWordTimestamps(unittest.TestCase):
    """Tests for word-level timestamp extraction."""

    def _make_client(self, word_timestamps=False):
        ws = MagicMock()
        return ConcreteServeClient(
            client_uid="wt-uid", websocket=ws, word_timestamps=word_timestamps
        )

    def _make_word(self, word, start, end, prob):
        w = MagicMock()
        w.word = word
        w.start = start
        w.end = end
        w.probability = prob
        return w

    def _make_segment(self, text, start, end, no_speech_prob=0.0, words=None):
        seg = MagicMock()
        seg.text = text
        seg.start = start
        seg.end = end
        seg.no_speech_prob = no_speech_prob
        seg.words = words
        return seg

    def test_word_timestamps_disabled_by_default(self):
        client = self._make_client()
        self.assertFalse(client.word_timestamps)

    def test_word_timestamps_enabled(self):
        client = self._make_client(word_timestamps=True)
        self.assertTrue(client.word_timestamps)

    def test_extract_words_when_disabled(self):
        client = self._make_client(word_timestamps=False)
        seg = self._make_segment("hello", 0.0, 1.0, words=[self._make_word("hello", 0.0, 0.5, 0.99)])
        result = client._extract_words(seg, 0.0)
        self.assertIsNone(result)

    def test_extract_words_when_enabled(self):
        client = self._make_client(word_timestamps=True)
        words = [
            self._make_word("hello", 0.0, 0.3, 0.95),
            self._make_word("world", 0.4, 0.8, 0.88),
        ]
        seg = self._make_segment("hello world", 0.0, 1.0, words=words)
        result = client._extract_words(seg, 10.0)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["word"], "hello")
        self.assertEqual(result[0]["start"], "10.000")
        self.assertEqual(result[0]["end"], "10.300")
        self.assertEqual(result[0]["probability"], 0.95)
        self.assertEqual(result[1]["word"], "world")
        self.assertEqual(result[1]["start"], "10.400")

    def test_extract_words_no_words_on_segment(self):
        client = self._make_client(word_timestamps=True)
        seg = self._make_segment("hello", 0.0, 1.0, words=None)
        result = client._extract_words(seg, 0.0)
        self.assertIsNone(result)

    def test_format_segment_without_words(self):
        client = self._make_client()
        seg = client.format_segment(0.0, 1.0, "hello")
        self.assertNotIn("words", seg)

    def test_format_segment_with_words(self):
        client = self._make_client(word_timestamps=True)
        words = [{"word": "hello", "start": "0.000", "end": "0.500", "probability": 0.95}]
        seg = client.format_segment(0.0, 1.0, "hello", words=words)
        self.assertIn("words", seg)
        self.assertEqual(len(seg["words"]), 1)
        self.assertEqual(seg["words"][0]["word"], "hello")

    def test_update_segments_includes_words(self):
        client = self._make_client(word_timestamps=True)
        words1 = [self._make_word("hello", 0.0, 0.5, 0.9)]
        words2 = [self._make_word("world", 0.6, 1.0, 0.85)]
        segments = [
            self._make_segment(" hello", 0.0, 0.5, words=words1),
            self._make_segment(" world", 0.6, 1.0, words=words2),
        ]
        last = client.update_segments(segments, 2.0)
        # First segment should be completed (in transcript) with words
        self.assertTrue(len(client.transcript) > 0)
        self.assertIn("words", client.transcript[-1])
        # Last segment should be in-progress with words
        self.assertIsNotNone(last)
        self.assertIn("words", last)

    def test_update_segments_no_words_when_disabled(self):
        client = self._make_client(word_timestamps=False)
        words1 = [self._make_word("hello", 0.0, 0.5, 0.9)]
        words2 = [self._make_word("world", 0.6, 1.0, 0.85)]
        segments = [
            self._make_segment(" hello", 0.0, 0.5, words=words1),
            self._make_segment(" world", 0.6, 1.0, words=words2),
        ]
        last = client.update_segments(segments, 2.0)
        self.assertTrue(len(client.transcript) > 0)
        self.assertNotIn("words", client.transcript[-1])
        self.assertNotIn("words", last)


if __name__ == "__main__":
    unittest.main()
