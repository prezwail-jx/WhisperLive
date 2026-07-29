import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper

REAL_THREAD = threading.Thread


class DummyThread:
    def __init__(self, target=None):
        self.target = target

    def start(self):
        return None


def segment(text, avg_logprob=-0.6, no_speech_prob=0.0):
    return SimpleNamespace(text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


def info(language, probability=0.6, candidates=None):
    return SimpleNamespace(
        language=language,
        language_probability=probability,
        all_language_probs=candidates,
    )


class TestServeClientFasterWhisperSingleModelInit(unittest.TestCase):
    def setUp(self):
        ServeClientFasterWhisper.SINGLE_MODEL = None
        ServeClientFasterWhisper.BATCH_WORKER = None

    def tearDown(self):
        ServeClientFasterWhisper.SINGLE_MODEL = None
        ServeClientFasterWhisper.BATCH_WORKER = None

    def make_client(self, single_model=True):
        websocket = mock.Mock()
        return ServeClientFasterWhisper(
            websocket=websocket,
            model="model/asr/small",
            client_uid="client",
            single_model=single_model,
        )

    def test_default_max_pending_audio_seconds_is_base_default(self):
        client = ServeClientFasterWhisper(
            websocket=mock.Mock(),
            model=None,
            client_uid="client",
        )

        self.assertAlmostEqual(client.max_pending_audio_seconds, 8.0)

    def test_custom_max_pending_audio_seconds_is_forwarded_to_base(self):
        client = ServeClientFasterWhisper(
            websocket=mock.Mock(),
            model=None,
            client_uid="client",
            max_pending_audio_seconds=15.0,
            sentence_completion_min_seconds=4.0,
            min_transcription_chunk_seconds=2.5,
        )

        self.assertAlmostEqual(client.max_pending_audio_seconds, 15.0)
        self.assertAlmostEqual(client.sentence_completion_min_seconds, 4.0)
        self.assertAlmostEqual(client.min_transcription_chunk_seconds, 2.5)

    @mock.patch("whisper_live.backend.faster_whisper_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.faster_whisper_backend.torch.cuda.is_available", return_value=False)
    def test_concurrent_single_model_clients_load_shared_model_once(self, mock_cuda_available):
        created = []
        barrier = threading.Barrier(2)

        def fake_create_model(client, device):
            time.sleep(0.05)
            client.transcriber = object()
            created.append(client.transcriber)

        clients = []
        errors = []

        def build_client():
            try:
                barrier.wait(timeout=5)
                clients.append(self.make_client(single_model=True))
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(
            ServeClientFasterWhisper,
            "create_model",
            autospec=True,
            side_effect=fake_create_model,
        ):
            threads = [REAL_THREAD(target=build_client) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(clients), 2)
        self.assertEqual(len(created), 1)
        self.assertIs(clients[0].transcriber, ServeClientFasterWhisper.SINGLE_MODEL)
        self.assertIs(clients[1].transcriber, ServeClientFasterWhisper.SINGLE_MODEL)
        self.assertIs(clients[0].transcriber, clients[1].transcriber)

    @mock.patch("whisper_live.backend.faster_whisper_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.faster_whisper_backend.torch.cuda.is_available", return_value=False)
    def test_non_single_model_clients_load_independent_models(self, mock_cuda_available):
        created = []

        def fake_create_model(client, device):
            client.transcriber = object()
            created.append(client.transcriber)

        with mock.patch.object(
            ServeClientFasterWhisper,
            "create_model",
            autospec=True,
            side_effect=fake_create_model,
        ):
            client_a = self.make_client(single_model=False)
            client_b = self.make_client(single_model=False)

        self.assertEqual(len(created), 2)
        self.assertIsNot(client_a.transcriber, client_b.transcriber)
        self.assertIsNone(ServeClientFasterWhisper.SINGLE_MODEL)


    @mock.patch("whisper_live.backend.faster_whisper_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.faster_whisper_backend.torch.cuda.is_available", return_value=False)
    def test_mixed_interpretation_keeps_language_auto_per_chunk(self, mock_cuda_available):
        info = mock.Mock(language="zh", language_probability=0.99, all_language_probs=None)

        def fake_create_model(client, device):
            client.transcriber = mock.Mock()
            client.transcriber.transcribe.return_value = ([], info)

        with mock.patch.object(
            ServeClientFasterWhisper,
            "create_model",
            autospec=True,
            side_effect=fake_create_model,
        ):
            client = ServeClientFasterWhisper(
                websocket=mock.Mock(),
                model="model/asr/small",
                client_uid="client",
                language=None,
                single_model=False,
                mixed_interpretation=True,
            )

        client.transcribe_audio([])

        self.assertIsNone(client.transcriber.transcribe.call_args.kwargs.get("language"))
        self.assertIsNone(client.language)
        self.assertEqual(client.current_language, "zh")

    def test_standard_transcribe_forwards_canonical_hotwords(self):
        client = ServeClientFasterWhisper(
            websocket=mock.Mock(),
            model=None,
            client_uid="client",
            language="en",
            hotwords="Whisper small OpenAI",
        )
        info = mock.Mock()
        client.transcriber = mock.Mock()
        client.transcriber.transcribe.return_value = ([], info)

        client.transcribe_audio([])

        self.assertEqual(
            client.transcriber.transcribe.call_args.kwargs.get("hotwords"),
            "Whisper small OpenAI",
        )

    def test_batch_request_forwards_canonical_hotwords(self):
        captured = []

        class FakeBatchWorker:
            def submit(self, request):
                captured.append(request)
                request.result = []
                request.info = mock.Mock()
                request.future.set()

        ServeClientFasterWhisper.BATCH_WORKER = FakeBatchWorker()
        client = ServeClientFasterWhisper(
            websocket=mock.Mock(),
            model=None,
            client_uid="client",
            language="en",
            initial_prompt="meeting prompt",
            hotwords="Whisper small OpenAI",
        )

        client.transcribe_audio([])

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].initial_prompt, "meeting prompt")
        self.assertEqual(captured[0].hotwords, "Whisper small OpenAI")

    def make_mixed_language_client(self):
        return ServeClientFasterWhisper(
            websocket=mock.Mock(),
            model=None,
            client_uid="client",
            language=None,
            mixed_interpretation=True,
        )

    def test_mixed_interpretation_clamps_non_zh_en_to_candidate_probability(self):
        client = self.make_mixed_language_client()
        info = mock.Mock(
            language="ja",
            language_probability=0.62,
            all_language_probs=[("ja", 0.62), ("zh", 0.31), ("en", 0.07)],
        )

        self.assertEqual(client.resolve_mixed_interpretation_language(info), "zh")

    def test_mixed_interpretation_keeps_previous_language_without_zh_en_candidate(self):
        client = self.make_mixed_language_client()
        client.current_language = "en"
        info = mock.Mock(
            language="ko",
            language_probability=0.71,
            all_language_probs=[("ko", 0.71), ("ja", 0.18)],
        )

        self.assertEqual(client.resolve_mixed_interpretation_language(info), "en")

    def test_mixed_interpretation_drops_non_zh_en_without_previous_language(self):
        client = self.make_mixed_language_client()
        info = mock.Mock(
            language="fr",
            language_probability=0.66,
            all_language_probs=[("fr", 0.66), ("de", 0.12)],
        )

        self.assertIsNone(client.resolve_mixed_interpretation_language(info))


    @mock.patch("whisper_live.backend.faster_whisper_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.faster_whisper_backend.torch.cuda.is_available", return_value=False)
    def test_min_transcription_chunk_seconds_is_configurable(self, mock_cuda_available):
        def fake_create_model(client, device):
            client.transcriber = object()

        with mock.patch.object(
            ServeClientFasterWhisper,
            "create_model",
            autospec=True,
            side_effect=fake_create_model,
        ):
            client = ServeClientFasterWhisper(
                websocket=mock.Mock(),
                model="model/asr/small",
                client_uid="client",
                single_model=False,
                min_transcription_chunk_seconds=2.5,
            )

        self.assertAlmostEqual(client.min_transcription_chunk_seconds, 2.5)

    def test_mixed_language_retry_requires_mixed_interpretation(self):
        client = ServeClientFasterWhisper(
            websocket=mock.Mock(),
            model=None,
            client_uid="client",
            mixed_interpretation=False,
            mixed_language_retry_enabled=True,
        )

        self.assertFalse(client.mixed_language_retry_enabled)

        client = ServeClientFasterWhisper(
            websocket=mock.Mock(),
            model=None,
            client_uid="client",
            mixed_interpretation=True,
            mixed_language_retry_enabled=True,
        )

        self.assertTrue(client.mixed_language_retry_enabled)

    def test_suspicious_switch_retries_previous_language_once(self):
        client = self.make_mixed_language_client()
        client.mixed_language_retry_enabled = True
        client.current_language = "en"
        calls = []

        def retry_callback(input_sample, language):
            calls.append((input_sample, language))
            return [segment("we need to review the plan", -0.55)], info("en", 0.7)

        result = client._maybe_retry_mixed_language(
            [1, 2, 3],
            [segment("我们需要", -0.6)],
            info("zh", 0.55, [("zh", 0.55), ("en", 0.45)]),
            retry_callback,
        )

        self.assertEqual(calls, [([1, 2, 3], "en")])
        self.assertEqual(result[0].text, "we need to review the plan")
        self.assertEqual(client.current_language, "en")

    def test_strong_real_switch_is_accepted_without_retry(self):
        client = self.make_mixed_language_client()
        client.mixed_language_retry_enabled = True
        client.current_language = "en"
        retry_callback = mock.Mock()

        result = client._maybe_retry_mixed_language(
            [],
            [segment("我们今天讨论预算", -0.3)],
            info("zh", 0.92, [("zh", 0.92), ("en", 0.03)]),
            retry_callback,
        )

        self.assertEqual(result[0].text, "我们今天讨论预算")
        self.assertEqual(client.current_language, "zh")
        retry_callback.assert_not_called()

    def test_batch_retry_submits_auto_and_forced_previous_language(self):
        submitted_languages = []

        class FakeBatchWorker:
            def submit(self, request):
                submitted_languages.append(request.language)
                if request.language is None:
                    request.result = [segment("我们需要", -0.6)]
                    request.info = info("zh", 0.55, [("zh", 0.55), ("en", 0.45)])
                else:
                    request.result = [segment("we need to review the plan", -0.55)]
                    request.info = info("en", 0.7)
                request.future.set()

        ServeClientFasterWhisper.BATCH_WORKER = FakeBatchWorker()
        client = self.make_mixed_language_client()
        client.mixed_language_retry_enabled = True
        client.current_language = "en"

        result = client.transcribe_audio([])

        self.assertEqual(submitted_languages, [None, "en"])
        self.assertEqual(result[0].text, "we need to review the plan")


if __name__ == "__main__":
    unittest.main()
