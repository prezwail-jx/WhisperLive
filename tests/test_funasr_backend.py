import json
import queue
import threading
import unittest
from unittest import mock

import numpy as np

from whisper_live.backend.funasr_backend import ServeClientFunASR


class DummyThread:
    def __init__(self, target=None):
        self.target = target

    def start(self):
        return None


class FakeOpenCCConverter:
    def convert(self, text):
        return str(text).replace("繁體", "繁体").replace("臺灣", "台湾")


class TestServeClientFunASR(unittest.TestCase):
    def setUp(self):
        ServeClientFunASR.SINGLE_MODEL = None

    def tearDown(self):
        ServeClientFunASR.SINGLE_MODEL = None

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_initializes_model_and_sends_ready(self, mock_cuda_available):
        websocket = mock.Mock()

        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", object())
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                model="iic/SenseVoiceSmall",
            )

        self.assertEqual(client.device, "cpu")
        websocket.send.assert_called_once()
        payload = json.loads(websocket.send.call_args[0][0])
        self.assertEqual(payload["message"], "SERVER_READY")
        self.assertEqual(payload["backend"], "funasr")

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_extracts_text_and_removes_sensevoice_tags(self, mock_cuda_available):
        websocket = mock.Mock()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", object())
            client = ServeClientFunASR(websocket=websocket, client_uid="client")

        segments = client._extract_segments([{"text": "<|zh|><|NEUTRAL|>你好"}], 2.5)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "你好")
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[0].end, 2.5)

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_completed_segment_enters_translation_queue_simplified(self, mock_cuda_available):
        websocket = mock.Mock()
        translation_queue = queue.Queue()
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", object())
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                translation_queue=translation_queue,
            )
        client.opencc_converter = FakeOpenCCConverter()
        client.frames_np = np.ones(4 * client.RATE, dtype=np.float32)

        client.handle_transcription_output(
            [{"text": "繁體中文"}, {"text": "第二句"}],
            duration=4.0,
        )

        item = translation_queue.get_nowait()
        self.assertEqual(item["text"], "繁体中文")
        self.assertTrue(item["completed"])

    @mock.patch("whisper_live.backend.funasr_backend.threading.Thread", DummyThread)
    @mock.patch("whisper_live.backend.funasr_backend.torch.cuda.is_available", return_value=False)
    def test_retries_without_hotword_when_model_rejects_it(self, mock_cuda_available):
        websocket = mock.Mock()
        transcriber = mock.Mock()
        transcriber.generate.side_effect = [
            TypeError("unexpected keyword argument hotword"),
            [{"text": "你好"}],
        ]
        with mock.patch.object(ServeClientFunASR, "create_model", autospec=True) as create_model:
            create_model.side_effect = lambda client: setattr(client, "transcriber", transcriber)
            client = ServeClientFunASR(
                websocket=websocket,
                client_uid="client",
                hotwords="WhisperLive",
            )

        result = client.transcribe_audio(np.zeros(client.RATE, dtype=np.float32))

        self.assertEqual(result, [{"text": "你好"}])
        self.assertEqual(transcriber.generate.call_count, 2)
        self.assertNotIn("hotword", transcriber.generate.call_args[1])


if __name__ == "__main__":
    unittest.main()
