import threading
import time
import unittest
from unittest import mock

from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper

REAL_THREAD = threading.Thread


class DummyThread:
    def __init__(self, target=None):
        self.target = target

    def start(self):
        return None


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


if __name__ == "__main__":
    unittest.main()
