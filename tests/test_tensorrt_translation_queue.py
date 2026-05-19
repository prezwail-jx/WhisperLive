import queue
import sys
import types
import unittest
from unittest import mock

from whisper_live.backend.base import ServeClientBase


fake_tensorrt_module = types.ModuleType("whisper_live.transcriber.transcriber_tensorrt")
fake_tensorrt_module.WhisperTRTLLM = object
sys.modules.setdefault("whisper_live.transcriber.transcriber_tensorrt", fake_tensorrt_module)

from whisper_live.backend.trt_backend import ServeClientTensorRT


class TestTensorRTTranslationQueue(unittest.TestCase):
    def make_client(self):
        translation_queue = queue.Queue()
        client = ServeClientTensorRT.__new__(ServeClientTensorRT)
        ServeClientBase.__init__(
            client,
            client_uid="trt-client",
            websocket=mock.Mock(),
            translation_queue=translation_queue,
        )
        client.language = "zh"
        return client, translation_queue

    def test_completed_tensorrt_segment_is_queued_for_translation(self):
        client, translation_queue = self.make_client()

        client.update_timestamp_offset("你好", 1.25)

        completed_segment = translation_queue.get_nowait()
        self.assertEqual(completed_segment["text"], "你好 ")
        self.assertEqual(completed_segment["completed"], True)
        self.assertEqual(completed_segment["language"], "zh")
        self.assertEqual(completed_segment["start"], "0.000")
        self.assertEqual(completed_segment["end"], "1.250")
        self.assertEqual(client.timestamp_offset, 1.25)

    def test_repeated_tensorrt_segment_is_not_queued_twice(self):
        client, translation_queue = self.make_client()

        client.update_timestamp_offset("你好", 1.0)
        client.update_timestamp_offset("你好", 1.0)

        self.assertEqual(translation_queue.qsize(), 1)
        self.assertEqual(len(client.transcript), 1)
        self.assertEqual(client.timestamp_offset, 2.0)


if __name__ == "__main__":
    unittest.main()
