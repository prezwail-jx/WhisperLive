import json
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.meeting import MeetingLogStore, SummaryTemplateStore


class TestSummaryTemplateStore(unittest.TestCase):
    def test_confirm_saves_markdown_template_and_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            draft = store.create_draft("weekly.md", "# 周例会\n\n## 概述\n示例内容\n\n## 待办\n- 示例\n")
            definition = store.confirm(draft["draft_id"], "周例会", draft["fields"])
            self.assertEqual(definition["format"], "md")
            template_path = os.path.join(directory, definition["id"], "template.md")
            self.assertTrue(os.path.isfile(template_path))
            with open(template_path, encoding="utf-8") as file:
                self.assertNotIn("示例内容", file.read())
            self.assertEqual(store.list()["templates"][0]["name"], "周例会")

    def test_rejects_template_without_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            with self.assertRaises(ValueError):
                store.create_draft("empty.md", "# 只有主标题\n")

    def test_custom_markdown_replaces_sample_section_content(self):
        summary = {
            "summary_template": "custom",
            "session_id": "session-1",
            "meeting_name": "周例会",
            "custom_template_name": "周例会模板",
            "custom_template_revision": 1,
            "custom_template_markdown": "# {{meeting_name}}\n\n## 概述\n旧示例\n\n## 待办\n- 旧待办\n",
            "custom_template_fields": [
                {"key": "overview", "heading": "概述", "type": "text"},
                {"key": "actions", "heading": "待办", "type": "list"},
            ],
            "template_data": {"overview": "项目按计划推进。", "actions": ["完成联调"]},
        }
        markdown = MeetingLogStore.render_summary_markdown(summary)
        self.assertIn("# 周例会", markdown)
        self.assertIn("项目按计划推进。", markdown)
        self.assertIn("- 完成联调", markdown)
        self.assertNotIn("旧示例", markdown)
        self.assertNotIn("旧待办", markdown)

    def test_custom_markdown_keeps_empty_headings_without_sample_content(self):
        summary = {
            "summary_template": "custom",
            "session_id": "session-empty",
            "meeting_name": "培训记录",
            "custom_template_name": "空栏目模板",
            "custom_template_revision": 1,
            "custom_template_markdown": "# {{meeting_name}}\n\n## 概述\n模板示例概述\n\n## 待办\n| 任务 |\n| --- |\n| 模板示例任务 |\n",
            "custom_template_fields": [
                {"key": "overview", "heading": "概述", "type": "text"},
                {"key": "actions", "heading": "待办", "type": "table", "columns": ["任务"]},
            ],
            "template_data": {"overview": "", "actions": []},
        }

        markdown = MeetingLogStore.render_summary_markdown(summary)

        self.assertIn("## 概述", markdown)
        self.assertIn("## 待办", markdown)
        self.assertNotIn("模板示例概述", markdown)
        self.assertNotIn("模板示例任务", markdown)
        self.assertNotIn("| 任务 |", markdown)
