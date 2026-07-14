import unittest

from whisper_live.backend.translation_backend import ServeClientTranslation
from whisper_live.server import TranscriptionServer


class TestTranslationModelSelection(unittest.TestCase):
    def test_server_value_recognizes_nllb_3_3b(self):
        self.assertEqual(
            TranscriptionServer._translation_model_value("NLLB-200-3.3B", "model/NLLB-200-3.3B"),
            "nllb_200_3_3b",
        )

    def test_nllb_3_3b_provider_is_supported(self):
        self.assertTrue(ServeClientTranslation.is_nllb_model("nllb_200_3_3b"))


if __name__ == "__main__":
    unittest.main()
