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

    def handle_transcription_output(self, result, duration, force_complete_last=False):
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
        self.assertAlmostEqual(client.sentence_completion_min_seconds, 0.0)
        self.assertAlmostEqual(client.max_pending_audio_seconds, 8.0)
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
            sentence_completion_min_seconds=4.0,
            min_transcription_chunk_seconds=2.5,
            max_pending_audio_seconds=15.0,
            translation_queue=q,
        )
        self.assertEqual(client.send_last_n_segments, 5)
        self.assertAlmostEqual(client.no_speech_thresh, 0.6)
        self.assertTrue(client.clip_audio)
        self.assertEqual(client.same_output_threshold, 20)
        self.assertAlmostEqual(client.min_segment_rms, 0.002)
        self.assertAlmostEqual(client.sentence_completion_min_seconds, 4.0)
        self.assertAlmostEqual(client.min_transcription_chunk_seconds, 2.5)
        self.assertAlmostEqual(client.max_pending_audio_seconds, 15.0)
        self.assertIs(client.translation_queue, q)

    def test_max_pending_audio_seconds_is_clamped(self):
        low = ConcreteServeClient(client_uid="low", websocket=MagicMock(), max_pending_audio_seconds=-5.0)
        high = ConcreteServeClient(client_uid="high", websocket=MagicMock(), max_pending_audio_seconds=60.0)

        self.assertAlmostEqual(low.max_pending_audio_seconds, 1.0)
        self.assertAlmostEqual(high.max_pending_audio_seconds, 30.0)

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

    def handle_transcription_output(self, result, duration, force_complete_last=False):
        self.output_calls.append((result, duration, force_complete_last))


class CountingTranscriptionClient(ConcreteServeClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transcription_calls = 0

    def transcribe_audio(self, input_sample):
        self.transcription_calls += 1
        return ["segment"]


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

    def test_asr_finalization_completes_when_no_tail_audio(self):
        client = AutoLanguageClient(client_uid="test", websocket=MagicMock())
        client.frames_np = None
        client.request_asr_finalization()

        with patch("whisper_live.backend.base.time.sleep", side_effect=lambda _seconds: setattr(client, "exit", True)):
            client.speech_to_text()

        self.assertEqual(client.asr_finalization_status, "completed")

    def test_asr_finalization_wait_reports_timeout(self):
        client = AutoLanguageClient(client_uid="test", websocket=MagicMock())
        client.request_asr_finalization()

        self.assertEqual(client.wait_for_asr_finalization(timeout=0), "timed_out")

    def test_new_audio_interval_skips_duplicate_audio(self):
        client = CountingTranscriptionClient(
            client_uid="test", websocket=MagicMock(), min_new_audio_seconds=0.25,
        )
        client.frames_np = np.zeros(int(2.5 * client.RATE), dtype=np.float32)
        client.last_transcription_audio_end = 2.5

        with patch("whisper_live.backend.base.time.sleep", side_effect=lambda _seconds: setattr(client, "exit", True)):
            client.speech_to_text()

        self.assertEqual(client.transcription_calls, 0)

    def test_new_audio_interval_allows_sufficient_new_audio(self):
        client = CountingTranscriptionClient(
            client_uid="test", websocket=MagicMock(), min_new_audio_seconds=0.25,
        )
        client.frames_np = np.zeros(int(2.75 * client.RATE), dtype=np.float32)
        client.last_transcription_audio_end = 2.5

        def stop_after_transcription(_input_sample):
            client.transcription_calls += 1
            client.exit = True
            return ["segment"]

        client.transcribe_audio = stop_after_transcription
        client.speech_to_text()

        self.assertEqual(client.transcription_calls, 1)

    def test_finalization_bypasses_new_audio_interval(self):
        client = CountingTranscriptionClient(
            client_uid="test", websocket=MagicMock(), min_new_audio_seconds=0.25,
        )
        client.frames_np = np.zeros(int(2.5 * client.RATE), dtype=np.float32)
        client.last_transcription_audio_end = 2.5
        client.request_asr_finalization()

        def stop_after_transcription(_input_sample):
            client.transcription_calls += 1
            client.exit = True
            return ["segment"]

        client.transcribe_audio = stop_after_transcription
        client.speech_to_text()

        self.assertEqual(client.transcription_calls, 1)
        self.assertEqual(client.asr_finalization_status, "completed")


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

    def test_custom_pending_limit_keeps_audio_under_limit(self):
        client = ConcreteServeClient(
            client_uid="test",
            websocket=self.ws,
            max_pending_audio_seconds=15.0,
        )
        client.add_frames(np.zeros(12 * 16000, dtype=np.float32))

        self.assertAlmostEqual(client.timestamp_offset, 0.0)

    def test_custom_pending_limit_drops_only_excess_audio(self):
        client = ConcreteServeClient(
            client_uid="test",
            websocket=self.ws,
            max_pending_audio_seconds=15.0,
        )

        with self.assertLogs(level="WARNING") as logs:
            client.add_frames(np.zeros(16 * 16000, dtype=np.float32))

        self.assertAlmostEqual(client.timestamp_offset, 1.0)
        self.assertIn("[REALTIME_DROP]", "\n".join(logs.output))
        self.assertIn("keep=15.00s", "\n".join(logs.output))


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

    def test_hard_drop_hallucination_segment_is_removed_before_send(self):
        segments = [
            {"start": "0.000", "end": "1.000", "text": "normal text", "completed": True},
            {"start": "1.000", "end": "2.000", "text": "优优独播剧场——YoYo Television Series Exclusive", "completed": True},
        ]

        self.client.send_transcription_to_client(segments)

        sent = json.loads(self.ws.send.call_args[0][0])
        self.assertEqual([segment["text"] for segment in sent["segments"]], ["normal text"])

    def test_hard_drop_hallucination_matches_partial_phrase(self):
        self.assertTrue(
            self.client._is_hard_drop_hallucination_text("前缀 优优独播剧场 后缀")
        )
        self.assertTrue(
            self.client._is_hard_drop_hallucination_text("YoYo Television Series Exclusive")
        )
        self.assertTrue(
            self.client._is_hard_drop_hallucination_text("今天年纪归宽市原围会提供")
        )
        self.assertTrue(
            self.client._is_hard_drop_hallucination_text("市场—— wears-mêmes request typlaş用比较 Nordic掉")
        )
        self.assertFalse(self.client._is_hard_drop_hallucination_text("normal text"))

    def test_mixed_interpretation_noise_matches_confirmed_phrase_only_in_mixed_mode(self):
        self.assertFalse(self.client._is_mixed_interpretation_noise_text("Hi Ho Zang!"))

        self.client.mixed_interpretation = True
        self.assertTrue(self.client._is_mixed_interpretation_noise_text("Hi Ho Zang!"))
        self.assertFalse(self.client._is_mixed_interpretation_noise_text("使用 X-ray 分析材料"))

    def test_mixed_interpretation_noise_matches_latin_extended_mixed_structure(self):
        text = "市场—— wears-mêmes request typlaş用比较 Nordic掉"
        self.assertFalse(self.client._is_mixed_interpretation_noise_text(text))

        self.client.mixed_interpretation = True
        with self.assertLogs(level="INFO") as logs:
            self.assertTrue(self.client._is_mixed_interpretation_noise_text(text))

        self.assertIn("reason=latin_extended_mixed_structure", "\n".join(logs.output))

    def test_mixed_interpretation_noise_keeps_common_mixed_terms_and_names(self):
        self.client.mixed_interpretation = True

        self.assertFalse(self.client._is_mixed_interpretation_noise_text("Müller教授"))
        self.assertFalse(self.client._is_mixed_interpretation_noise_text("Müller教授与François合作"))
        self.assertFalse(self.client._is_mixed_interpretation_noise_text("使用 X-ray 分析材料"))
        self.assertFalse(self.client._is_mixed_interpretation_noise_text("使用 café 模型"))

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

    def test_force_complete_last_emits_final_segment(self):
        q = queue.Queue()
        self.client.translation_queue = q
        segs = [self._make_segment(0.0, 0.4, " final tail")]

        last = self.client.update_segments(segs, duration=0.5, force_complete_last=True)

        self.assertIsNone(last)
        self.assertEqual(len(self.client.transcript), 1)
        self.assertTrue(self.client.transcript[0]["completed"])
        self.assertEqual(self.client.transcript[0]["text"].strip(), "final tail")
        self.assertEqual(q.get_nowait()["text"].strip(), "final tail")

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

    def test_incomplete_segment_forces_complete_at_configured_duration(self):
        self.client.max_incomplete_segment_seconds = 10.0
        self.client.frames_np = np.full(16000 * 10, 0.01, dtype=np.float32)
        seg = self._make_segment(0.0, 10.0, " long pending speech")

        with self.assertLogs(level="INFO") as logs:
            last = self.client.update_segments([seg], duration=10.0)

        self.assertIsNone(last)
        self.assertEqual(len(self.client.transcript), 1)
        self.assertTrue(self.client.transcript[0]["completed"])
        self.assertAlmostEqual(self.client.timestamp_offset, 10.0)
        self.assertIn("[FORCE_COMPLETE_INCOMPLETE]", "\n".join(logs.output))
        self.assertIn("reason=duration_limit", "\n".join(logs.output))

    def test_sentence_boundary_forces_complete_after_configured_duration(self):
        self.client.max_incomplete_segment_seconds = 12.0
        self.client.sentence_completion_min_seconds = 4.0
        self.client.frames_np = np.full(16000 * 5, 0.01, dtype=np.float32)
        seg = self._make_segment(0.0, 4.0, " complete sentence.")

        with self.assertLogs(level="INFO") as logs:
            last = self.client.update_segments([seg], duration=4.0)

        self.assertIsNone(last)
        self.assertEqual(len(self.client.transcript), 1)
        self.assertTrue(self.client.transcript[0]["completed"])
        self.assertAlmostEqual(self.client.timestamp_offset, 4.0)
        self.assertIn("reason=sentence_boundary", "\n".join(logs.output))

    def test_sentence_boundary_waits_before_configured_duration(self):
        self.client.max_incomplete_segment_seconds = 12.0
        self.client.sentence_completion_min_seconds = 4.0
        self.client.frames_np = np.full(16000 * 3, 0.01, dtype=np.float32)
        seg = self._make_segment(0.0, 3.0, " complete sentence.")

        last = self.client.update_segments([seg], duration=3.0)

        self.assertIsNotNone(last)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertAlmostEqual(self.client.timestamp_offset, 0.0)

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

    def test_repeated_hotword_completed_segment_is_dropped_with_strong_audio(self):
        q = queue.Queue()
        self.client.translation_queue = q
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["国创中心"])
        self.client.frames_np = np.full(16000 * 5, 0.02, dtype=np.float32)
        segs = [
            self._make_segment(0.0, 1.0, "国创中心 国创中心 国创中心", no_speech_prob=0.10),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        with self.assertLogs(level="INFO") as logs:
            last = self.client.update_segments(segs, duration=3.0)

        output = "\n".join(logs.output)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertTrue(q.empty())
        self.assertIsNotNone(last)
        self.assertIn("next", last["text"])
        self.assertIn("reason=consecutive_repeated_hotword", output)
        self.assertIn("repeat_count=3", output)

    def test_prefixed_consecutive_hotword_repetition_is_dropped_without_dominance(self):
        q = queue.Queue()
        self.client.translation_queue = q
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms([
            "长三角国家技术创新中心",
            "国创中心",
        ])
        self.client.frames_np = np.full(16000 * 5, 0.02, dtype=np.float32)
        segs = [
            self._make_segment(
                0.0,
                1.0,
                "三角国家技术创新中心 国创中心 国创中心 国创中心",
                no_speech_prob=0.10,
            ),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        with self.assertLogs(level="INFO") as logs:
            last = self.client.update_segments(segs, duration=3.0)

        output = "\n".join(logs.output)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertTrue(q.empty())
        self.assertIsNotNone(last)
        self.assertIn("next", last["text"])
        self.assertIn("reason=consecutive_repeated_hotword", output)
        self.assertIn("repeat_count=3", output)

    def test_punctuated_consecutive_hotword_repetition_is_dropped(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["国创中心"])
        self.client.frames_np = np.full(16000 * 5, 0.02, dtype=np.float32)
        segs = [
            self._make_segment(0.0, 1.0, "国创中心，国创中心，国创中心", no_speech_prob=0.10),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        with self.assertLogs(level="INFO") as logs:
            self.client.update_segments(segs, duration=3.0)

        self.assertEqual(len(self.client.transcript), 0)
        self.assertIn("reason=consecutive_repeated_hotword", "\n".join(logs.output))

    def test_non_consecutive_hotword_repetition_is_retained(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["国创中心"])
        segs = [
            self._make_segment(0.0, 1.0, "国创中心 正常内容 国创中心 更多内容 国创中心", no_speech_prob=0.10),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        self.client.update_segments(segs, duration=3.0)

        self.assertEqual(len(self.client.transcript), 1)
        self.assertIn("正常内容", self.client.transcript[0]["text"])

    def test_repeated_hotword_partial_segment_waits_for_more_audio(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["国创中心"])
        self.client.frames_np = np.full(16000 * 5, 0.02, dtype=np.float32)
        seg = self._make_segment(0.0, 1.0, "国创中心 国创中心 国创中心", no_speech_prob=0.10)

        last = self.client.update_segments([seg], duration=2.0)

        self.assertIsNone(last)
        self.assertEqual(self.client.current_out, "")
        self.assertEqual(len(self.client.transcript), 0)
        self.assertEqual(self.client.timestamp_offset, 0.0)

    def test_hotword_repeated_twice_is_retained(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["国创中心"])
        segs = [
            self._make_segment(0.0, 1.0, "国创中心 国创中心", no_speech_prob=0.10),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        self.client.update_segments(segs, duration=3.0)

        self.assertEqual(len(self.client.transcript), 1)
        self.assertIn("国创中心", self.client.transcript[0]["text"])

    def test_different_hotwords_do_not_accumulate_as_repetition(self):
        self.client.hotword_match_terms = self.client._prepare_hotword_match_terms(["国创中心", "NICE", "长三角"])
        segs = [
            self._make_segment(0.0, 1.0, "国创中心 NICE 长三角", no_speech_prob=0.10),
            self._make_segment(1.0, 2.0, " next", no_speech_prob=0.0),
        ]

        self.client.update_segments(segs, duration=3.0)

        self.assertEqual(len(self.client.transcript), 1)
        self.assertIn("国创中心", self.client.transcript[0]["text"])

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

    def test_low_energy_short_hao_is_dropped_with_stricter_threshold(self):
        self.client.frames_np = np.full(16000 * 5, 0.003, dtype=np.float32)
        segs = [
            self._make_segment(0.0, 1.0, "好"),
            self._make_segment(1.0, 2.0, " next"),
        ]
        self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 0)
        self.assertGreater(self.client.timestamp_offset, 0.0)

    def test_normal_energy_short_hao_is_kept(self):
        segs = [
            self._make_segment(0.0, 1.0, "好"),
            self._make_segment(1.0, 2.0, " next"),
        ]
        self.client.update_segments(segs, duration=3.0)
        self.assertEqual(len(self.client.transcript), 1)
        self.assertEqual(self.client.transcript[0]["text"], "好")

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


class TestSegmentationProfileV2(unittest.TestCase):
    def setUp(self):
        self.client = ConcreteServeClient(
            client_uid="v2",
            websocket=MagicMock(),
            segmentation_profile_v2=True,
            min_segment_rms=0.028,
            same_output_threshold=9,
            max_incomplete_segment_seconds=12.0,
            sentence_completion_min_seconds=3.0,
        )

    @staticmethod
    def _segment(start, end, text, no_speech_prob=0.0):
        segment = MagicMock()
        segment.start = start
        segment.end = end
        segment.text = text
        segment.no_speech_prob = no_speech_prob
        return segment

    def test_quiet_valid_speech_is_retained(self):
        self.client.frames_np = np.full(16000 * 3, 0.03, dtype=np.float32)
        self.client.update_segments([
            self._segment(0.0, 0.5, " quiet speech"),
            self._segment(0.5, 1.5, " next"),
        ], duration=2.0)

        self.client.flush_pending_completed_segments(force=True)
        self.assertEqual(self.client.transcript[0]["text"].strip(), "quiet speech")

    def test_silence_hallucination_keeps_stricter_energy_gate(self):
        self.client.frames_np = np.full(16000 * 3, 0.02, dtype=np.float32)
        self.client.update_segments([
            self._segment(0.0, 1.1, " Thank you."),
            self._segment(1.1, 2.0, " next"),
        ], duration=2.5)

        self.assertEqual(self.client.transcript, [])

    def test_repeat_threshold_is_not_eager(self):
        self.client.frames_np = np.full(16000 * 3, 0.02, dtype=np.float32)
        segment = self._segment(0.0, 1.0, " repeated words")
        for _ in range(10):
            self.client.update_segments([segment], duration=2.0)

        self.assertEqual(self.client.transcript, [])

    def test_duration_limit_remains_bounded(self):
        self.client.frames_np = np.full(16000 * 13, 0.02, dtype=np.float32)
        self.client.update_segments([self._segment(0.0, 12.0, " long pending speech")], duration=12.0)

        self.assertEqual(len(self.client.transcript), 1)
        self.assertEqual(self.client.transcript[0]["text"].strip(), "long pending speech")

    def test_punctuation_without_trailing_silence_does_not_complete(self):
        self.client.frames_np = np.full(16000 * 4, 0.02, dtype=np.float32)
        segment = self._segment(0.0, 4.0, " complete sentence.")
        for _ in range(3):
            last = self.client.update_segments([segment], duration=4.0)

        self.assertIsNotNone(last)
        self.assertEqual(self.client.transcript, [])

    def test_stability_and_trailing_silence_complete_sentence(self):
        self.client.frames_np = np.concatenate((
            np.full(int(3.4 * 16000), 0.02, dtype=np.float32),
            np.zeros(int(0.6 * 16000), dtype=np.float32),
        ))
        segment = self._segment(0.0, 3.4, " complete sentence.")
        for _ in range(3):
            self.client.update_segments([segment], duration=4.0)

        self.assertEqual(len(self.client.transcript), 1)

    def _completed(self, start, end, text, language="en", speaker=None):
        return {
            "start": f"{start:.3f}", "end": f"{end:.3f}", "text": text,
            "completed": True, "language": language, "speaker": speaker,
        }

    def test_compatible_short_fragments_merge(self):
        self.client._stage_completed_segment(self._completed(0.0, 0.4, "hello"), "test")
        self.client._stage_completed_segment(self._completed(0.45, 1.2, "world"), "test")

        self.assertEqual([segment["text"] for segment in self.client.transcript], ["hello world"])

    def test_terminal_boundary_prevents_merge(self):
        self.client._stage_completed_segment(self._completed(0.0, 0.4, "really?"), "test")
        self.client._stage_completed_segment(self._completed(0.45, 0.8, "yes"), "test")

        self.assertEqual([segment["text"] for segment in self.client.transcript], ["really?"])

    def test_hold_window_release(self):
        self.client._stage_completed_segment(self._completed(0.0, 0.4, "hold me"), "test")
        with patch("whisper_live.backend.base.time.monotonic", return_value=time.monotonic() + 1.0):
            released = self.client.flush_pending_completed_segments()

        self.assertEqual([segment["text"] for segment in released], ["hold me"])

    def test_v2_keeps_subsecond_hold_and_no_audio_interval(self):
        self.assertAlmostEqual(self.client.short_fragment_hold_seconds, 0.7)
        self.assertAlmostEqual(self.client.min_new_audio_seconds, 0.0)


class TestSegmentationProfileV3(unittest.TestCase):
    def setUp(self):
        self.client = ConcreteServeClient(
            client_uid="v3",
            websocket=MagicMock(),
            segmentation_profile_v2=True,
            short_fragment_hold_seconds=2.5,
            min_new_audio_seconds=0.25,
        )

    @staticmethod
    def _completed(start, end, text):
        return {
            "start": f"{start:.3f}", "end": f"{end:.3f}", "text": text,
            "completed": True, "language": "en", "speaker": None,
        }

    def test_v3_holds_short_fragment_for_two_and_a_half_seconds(self):
        self.client._stage_completed_segment(self._completed(0.0, 0.4, "hold me"), "test")
        held_at = self.client.pending_completed_segment["held_at"]

        with patch("whisper_live.backend.base.time.monotonic", return_value=held_at + 2.0):
            self.assertEqual(self.client.flush_pending_completed_segments(), [])
        with patch("whisper_live.backend.base.time.monotonic", return_value=held_at + 2.5):
            released = self.client.flush_pending_completed_segments()

        self.assertEqual([segment["text"] for segment in released], ["hold me"])

    def test_v3_merges_compatible_short_fragments_within_hold_window(self):
        self.client._stage_completed_segment(self._completed(0.0, 0.4, "hello"), "test")
        self.client._stage_completed_segment(self._completed(0.45, 1.2, "world"), "test")

        self.assertEqual([segment["text"] for segment in self.client.transcript], ["hello world"])


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
