import json
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.meeting import (
    MeetingHotwordStore,
    count_hotwords,
    hotword_text_to_prompt,
    normalize_asr_hotwords,
    parse_hotword_config,
)


class TestMeetingHotwordStore(unittest.TestCase):
    def test_list_and_get_scan_txt_files_from_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "会议A.txt"), "w", encoding="utf-8") as file:
                file.write("# comment\n图灵科技\n\nfaster-whisper\n")
            with open(os.path.join(directory, "ignore.md"), "w", encoding="utf-8") as file:
                file.write("ignored")

            store = MeetingHotwordStore(directory)
            meetings = store.list()["meetings"]
            self.assertEqual(len(meetings), 1)
            self.assertEqual(meetings[0]["meeting_name"], "会议A")
            self.assertEqual(meetings[0]["filename"], "会议A.txt")
            self.assertEqual(meetings[0]["count"], 2)

            loaded = store.get("会议A")
            self.assertEqual(loaded["text"], "图灵科技\nfaster-whisper")
            self.assertEqual(loaded["count"], 2)

            missing = store.get("会议B")
            self.assertEqual(missing["count"], 0)
            self.assertEqual(missing["filename"], "")

    def test_count_hotwords_ignores_blank_lines_and_comments(self):
        self.assertEqual(count_hotwords("# c\nACE\n\nDocker"), 2)

    def test_translation_rules_do_not_add_source_to_hotword_prompt(self):
        parsed = parse_hotword_config(
            "# comment\nOpenAI => 开放人工智能\n普通热词\ninvalid =>\n=> invalid\n"
        )

        self.assertEqual(parsed["hotwords"], ["普通热词"])
        self.assertEqual(parsed["translation_glossary"], {"OpenAI": "开放人工智能"})
        self.assertEqual(parsed["count"], 1)
        self.assertEqual(parsed["translation_count"], 1)
        self.assertEqual(
            hotword_text_to_prompt(parsed["text"]),
            "普通热词",
        )

    def test_duplicate_translation_rule_uses_last_target(self):
        parsed = parse_hotword_config("OpenAI => 旧译名\nOpenAI => 新译名")

        self.assertEqual(parsed["translation_glossary"], {"OpenAI": "新译名"})
        self.assertEqual(parsed["translation_count"], 1)

    def test_normalize_asr_hotwords_dedupes_and_preserves_first_spelling(self):
        canonical = normalize_asr_hotwords(terms=[" OpenAI ", "openai", "Whisper   small", "ＡＣＥ"])

        self.assertEqual(canonical["terms"], ["OpenAI", "Whisper small", "ACE"])
        self.assertEqual(canonical["prompt"], "OpenAI Whisper small ACE")
        self.assertEqual(canonical["original_count"], 4)
        self.assertEqual(canonical["accepted_count"], 3)
        self.assertEqual(canonical["rejected_count"], 1)
        self.assertEqual(canonical["validation_reasons"], ["duplicate"])

    def test_normalize_asr_hotwords_limits_terms_and_prompt_without_partial_terms(self):
        canonical = normalize_asr_hotwords(
            terms=["Alpha", "Beta", "Gamma"],
            max_terms=2,
            max_prompt_chars=100,
        )

        self.assertEqual(canonical["terms"], ["Alpha", "Beta"])
        self.assertTrue(canonical["truncated"])
        self.assertEqual(canonical["truncation_reasons"], ["term_count"])

        canonical = normalize_asr_hotwords(
            terms=["Alpha", "Beta Gamma", "Delta"],
            max_terms=10,
            max_prompt_chars=len("Alpha Beta Gamma"),
        )

        self.assertEqual(canonical["terms"], ["Alpha", "Beta Gamma"])
        self.assertTrue(canonical["truncated"])
        self.assertEqual(canonical["truncation_reasons"], ["prompt_chars"])

    def test_normalize_asr_hotwords_rejects_oversized_terms(self):
        canonical = normalize_asr_hotwords(terms=["Valid", "x" * 65, "Next"])

        self.assertEqual(canonical["terms"], ["Valid", "Next"])
        self.assertEqual(canonical["rejected_count"], 1)
        self.assertEqual(canonical["validation_reasons"], ["term_length"])
