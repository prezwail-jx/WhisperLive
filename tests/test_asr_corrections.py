import os
import tempfile
import unittest

from whisper_live.meeting import (
    AsrTextCorrector,
    MeetingAsrCorrectionStore,
    parse_asr_correction_config,
)
from whisper_live.server import BackendType, TranscriptionServer


class TestAsrCorrections(unittest.TestCase):
    def test_parse_rules_uses_last_duplicate_and_longest_first(self):
        parsed = parse_asr_correction_config(
            "# comment\n派森 => Python\n开放爱爱 => OpenAI\n派森 => Py\n开放 => Open\ninvalid\n"
        )

        self.assertEqual(parsed["count"], 3)
        self.assertEqual(parsed["rules"][0], ("开放爱爱", "OpenAI"))
        self.assertIn(("派森", "Py"), parsed["rules"])

    def test_corrector_replaces_literals_without_regex_semantics(self):
        corrector = AsrTextCorrector([("C++", "C plus plus"), ("派森", "Python")])

        corrected, replacements = corrector.correct("派森 和 C++")

        self.assertEqual(corrected, "Python 和 C plus plus")
        self.assertEqual(replacements, 2)

    def test_corrector_applies_longest_national_innovation_center_rule_first(self):
        corrector = AsrTextCorrector([
            ("二角国创中心地", "长三角国创中心"),
            ("二角国创中心", "长三角国创中心"),
        ])

        corrected, replacements = corrector.correct("二角国创中心地")

        self.assertEqual(corrected, "长三角国创中心")
        self.assertEqual(replacements, 1)

    def test_corrector_does_not_cascade_replacements(self):
        corrector = AsrTextCorrector([("威斯伯", "Whisper"), ("Whisper", "OpenAI Whisper")])

        corrected, replacements = corrector.correct("威斯伯")

        self.assertEqual(corrected, "Whisper")
        self.assertEqual(replacements, 1)

    def test_store_loads_meeting_named_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "例会.txt"), "w", encoding="utf-8") as file:
                file.write("威斯伯 => Whisper\n")

            store = MeetingAsrCorrectionStore(directory)
            corrector, record = store.corrector_for("例会")

        self.assertEqual(record["count"], 1)
        self.assertEqual(record["filename"], "例会.txt")
        self.assertEqual(corrector.correct("威斯伯 部署")[0], "Whisper 部署")

    def test_server_merges_global_and_meeting_rules_with_meeting_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            global_path = os.path.join(directory, "DOMAIN_CORRECTIONS.txt")
            with open(global_path, "w", encoding="utf-8") as file:
                file.write("威斯伯 => Whisper\n派森 => Python\n")
            with open(os.path.join(directory, "例会.txt"), "w", encoding="utf-8") as file:
                file.write("派森 => Py\n中式转化 => 中试转化\n")

            server = TranscriptionServer()
            server.backend = BackendType.FASTER_WHISPER
            server.meeting_asr_corrections = MeetingAsrCorrectionStore(directory)
            server.asr_corrections_file = global_path
            options = {
                "enable_translation": True,
                "language": "zh",
                "target_language": "en",
                "translation_mode": "standard",
                "meeting_name": "例会",
            }

            server.apply_meeting_asr_corrections(options)

        self.assertTrue(options["asr_corrections_enabled"])
        self.assertEqual(dict(options["asr_correction_rules"])["派森"], "Py")
        self.assertEqual(dict(options["asr_correction_rules"])["威斯伯"], "Whisper")
        self.assertIn("DOMAIN_CORRECTIONS.txt", options["asr_corrections_file"])
        self.assertIn("例会.txt", options["asr_corrections_file"])

    def test_server_enables_corrections_for_zh_en_and_bidirectional_only(self):
        server = TranscriptionServer()
        server.backend = BackendType.FASTER_WHISPER

        self.assertTrue(server.asr_corrections_enabled({
            "enable_translation": True,
            "language": "zh",
            "target_language": "en",
            "translation_mode": "standard",
        }))
        self.assertTrue(server.asr_corrections_enabled({
            "enable_translation": True,
            "language": None,
            "target_language": "auto",
            "translation_mode": "mixed_interpretation",
        }))
        self.assertFalse(server.asr_corrections_enabled({
            "enable_translation": True,
            "language": "en",
            "target_language": "zh",
            "translation_mode": "standard",
        }))


if __name__ == "__main__":
    unittest.main()
