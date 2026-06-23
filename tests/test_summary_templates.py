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

    def test_blank_parent_with_child_headings_is_structural_container(self):
        markdown = (
            "# 会议纪要\n\n## 二、讨论事项综述\n\n"
            "### 1. 创新创业大赛总体情况及领域决赛安排\n\n"
            "### 2. 研究所筹建方案\n"
        )
        sections = SummaryTemplateStore._extract_sections(markdown)

        self.assertEqual(
            [(section["heading"], section["level"], section["role"]) for section in sections],
            [
                ("二、讨论事项综述", 2, "container"),
                ("1. 创新创业大赛总体情况及领域决赛安排", 3, "field"),
                ("2. 研究所筹建方案", 3, "field"),
            ],
        )

    def test_hierarchy_defaults_use_prose_and_derive_meeting_topics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            markdown = (
                "# 会议纪要\n\n## 一、会议议题\n\n"
                "## 二、讨论事项综述\n\n"
                "### 1. 创新创业大赛安排\n\n"
                "### 2. 其他需讨论汇报事项\n"
            )
            draft = store.create_draft("hierarchy.md", markdown)

        fields = {field["heading"]: field for field in draft["fields"]}
        topics = fields["一、会议议题"]
        competition = fields["1. 创新创业大赛安排"]
        other = fields["2. 其他需讨论汇报事项"]
        self.assertEqual(
            topics["derive_from_fields"],
            [competition["key"], other["key"]],
        )
        self.assertEqual(competition["output_style"], "prose")
        self.assertEqual(other["output_style"], "prose")
        self.assertTrue(other["residual"])

    def test_parent_with_own_body_remains_content_field(self):
        markdown = "## 讨论事项综述\n需要生成总体概述\n\n### 1. 专题\n示例\n"
        sections = SummaryTemplateStore._extract_sections(markdown)

        self.assertEqual(sections[0]["role"], "field")
        self.assertEqual(sections[1]["role"], "field")

    def test_confirm_keeps_structural_parent_without_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            markdown = (
                "# 会议纪要\n\n## 二、讨论事项综述\n\n"
                "### 1. 创新创业大赛总体情况及领域决赛安排\n\n"
                "### 2. 研究所筹建方案\n"
            )
            draft = store.create_draft("hierarchy.md", markdown)

            self.assertEqual(
                [field["heading"] for field in draft["fields"]],
                ["1. 创新创业大赛总体情况及领域决赛安排", "2. 研究所筹建方案"],
            )
            definition = store.confirm(draft["draft_id"], "层级模板", draft["fields"])
            with open(os.path.join(directory, definition["id"], "template.md"), encoding="utf-8") as file:
                saved = file.read()

            self.assertIn("## 二、讨论事项综述", saved)
            self.assertIn("### 1. 创新创业大赛总体情况及领域决赛安排", saved)
            self.assertNotIn("## 二、讨论事项综述\n\n{{", saved)
            self.assertEqual(saved.count("{{"), 2)

    def test_confirm_removes_unconfigured_nested_sample_headings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            markdown = (
                "# 会议纪要\n\n## 讨论综述\n示例内容\n"
                "### 1. 示例议题\n### 2. 另一个示例\n\n## 待办\n- 示例待办\n"
            )
            sections = store._extract_sections(markdown)
            fields = [
                {"key": "overview", "heading": "讨论综述", "label": "讨论综述", "type": "text"},
                {"key": "actions", "heading": "待办", "label": "待办", "type": "list"},
            ]
            draft = store.create_draft("nested.md", markdown, fields)
            definition = store.confirm(draft["draft_id"], "嵌套示例", draft["fields"])
            template_path = os.path.join(directory, definition["id"], "template.md")
            with open(template_path, encoding="utf-8") as file:
                saved = file.read()
            self.assertIn("{{overview}}", saved)
            self.assertIn("{{actions}}", saved)
            self.assertNotIn("示例内容", saved)
            self.assertNotIn("### 1.", saved)
            self.assertNotIn("### 2.", saved)
            self.assertEqual(os.stat(template_path).st_mode & 0o777, 0o644)

    def test_confirm_saves_field_quality_options(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            draft = store.create_draft("info.md", "## 会议信息\n示例\n")
            fields = draft["fields"]
            fields[0].update({
                "type": "table",
                "columns": ["项目", "内容"],
                "required": True,
                "metadata_enrichment": True,
            })
            definition = store.confirm(draft["draft_id"], "会议信息", fields)
            field = definition["fields"][0]
            self.assertTrue(field["required"])
            self.assertTrue(field["metadata_enrichment"])

    def test_confirm_corrects_common_custom_field_types(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            markdown = "## 会议议题\n示例\n\n## 讨论事项综述\n示例\n\n## 决策\n示例\n"
            draft = store.create_draft("types.md", markdown, [
                {"key": "topics", "heading": "会议议题", "label": "会议议题", "type": "text"},
                {"key": "summary", "heading": "讨论事项综述", "label": "讨论事项综述", "type": "evidence_list"},
                {"key": "decisions", "heading": "决策", "label": "决策", "type": "evidence_list"},
            ])
            definition = store.confirm(draft["draft_id"], "类型纠偏", draft["fields"])

        types = {field["heading"]: field["type"] for field in definition["fields"]}
        self.assertEqual(types["会议议题"], "list")
        self.assertEqual(types["讨论事项综述"], "text")
        self.assertEqual(types["决策"], "list")

    def test_rejects_template_without_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            with self.assertRaises(ValueError):
                store.create_draft("empty.md", "# 只有主标题\n")

    def test_delete_soft_hides_template_and_keeps_files(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            draft = store.create_draft("weekly.md", "# 周例会\n\n## 概述\n示例内容\n")
            definition = store.confirm(draft["draft_id"], "周例会", draft["fields"])
            template_dir = os.path.join(directory, definition["id"])

            result = store.delete(definition["id"])

            self.assertTrue(result["deleted"])
            self.assertEqual(result["template_id"], definition["id"])
            self.assertIsNone(store.get(definition["id"]))
            self.assertEqual(store.list()["templates"], [])
            definition_path = os.path.join(template_dir, "definition.json")
            self.assertTrue(os.path.isfile(definition_path))
            with open(definition_path, encoding="utf-8") as file:
                saved = json.load(file)
            self.assertTrue(saved["deleted"])
            self.assertIn("deleted_at", saved)
            self.assertTrue(store.delete(definition["id"])["deleted"])

    def test_delete_missing_template_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            self.assertIsNone(store.delete("missing-template"))

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

    def test_custom_markdown_renders_structured_values_without_json_residue(self):
        summary = {
            "summary_template": "custom",
            "session_id": "session-structured",
            "meeting_name": "周例会",
            "custom_template_name": "结构化模板",
            "custom_template_revision": 1,
            "custom_template_markdown": "# {{meeting_name}}\n\n## 综述\n{{summary}}\n\n## 议题\n{{topics}}\n\n## 表格\n{{rows}}\n",
            "custom_template_fields": [
                {"key": "summary", "heading": "综述", "type": "text"},
                {"key": "topics", "heading": "议题", "type": "list"},
                {"key": "rows", "heading": "表格", "type": "table", "columns": ["事项", "说明"]},
            ],
            "template_data": {
                "summary": {"content": "会议讨论项目推进。"},
                "topics": [{"title": "领域决赛", "content": "安排承办单位"}],
                "rows": [{"事项": {"content": "完成材料"}, "说明": ["本周", "提交"]}],
            },
        }

        markdown = MeetingLogStore.render_summary_markdown(summary)

        self.assertIn("会议讨论项目推进。", markdown)
        self.assertIn("- 领域决赛：安排承办单位", markdown)
        self.assertIn("| 完成材料 | 本周 提交 |", markdown.replace("\n", " "))
        self.assertNotIn("{'", markdown)
        self.assertNotIn('"content"', markdown)

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
