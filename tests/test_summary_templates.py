import json
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.meeting import MeetingLogStore, MeetingSummaryService, SummaryTemplateStore


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

    def test_confirm_reapplies_hierarchy_defaults_missing_from_client_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            markdown = (
                "# 会议纪要\n\n## 会议议题\n\n"
                "## 讨论事项综述\n\n"
                "### 1、创新创业大赛安排\n\n"
                "### 2、其他需讨论汇报事项\n"
            )
            draft = store.create_draft("hierarchy.md", markdown)
            client_fields = [
                {
                    "key": field["key"],
                    "label": field["label"],
                    "heading": field["heading"],
                    "type": "list",
                }
                for field in draft["fields"]
            ]

            definition = store.confirm(draft["draft_id"], "层级模板", client_fields)

        fields = {field["heading"]: field for field in definition["fields"]}
        topics = fields["会议议题"]
        competition = fields["1、创新创业大赛安排"]
        other = fields["2、其他需讨论汇报事项"]
        self.assertEqual(topics["derive_from_fields"], [competition["key"], other["key"]])
        self.assertEqual(competition["type"], "text")
        self.assertEqual(competition["output_style"], "prose")
        self.assertEqual(other["type"], "text")
        self.assertEqual(other["output_style"], "prose")
        self.assertTrue(other["residual"])

    def test_get_repairs_legacy_hierarchy_definition_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SummaryTemplateStore(directory)
            template_dir = os.path.join(directory, "legacy")
            os.makedirs(template_dir)
            markdown = (
                "# 会议纪要\n\n## 会议议题\n\n{{meeting_topic}}\n\n"
                "## 讨论事项综述\n\n"
                "### 1、创新创业大赛安排\n\n{{field_1}}\n\n"
                "### 2、其他需讨论汇报事项\n\n{{field_2}}\n"
            )
            definition = {
                "id": "legacy",
                "name": "旧模板",
                "format": "md",
                "revision": 1,
                "fields": [
                    {"key": "meeting_topic", "heading": "会议议题", "type": "list", "derive_from_fields": []},
                    {"key": "field_1", "heading": "1、创新创业大赛安排", "type": "text", "derive_from_fields": []},
                    {"key": "field_2", "heading": "2、其他需讨论汇报事项", "type": "list", "derive_from_fields": []},
                ],
            }
            with open(os.path.join(template_dir, "template.md"), "w", encoding="utf-8") as file:
                file.write(markdown)
            definition_path = os.path.join(template_dir, "definition.json")
            with open(definition_path, "w", encoding="utf-8") as file:
                json.dump(definition, file, ensure_ascii=False)

            repaired = store.get("legacy")

            fields = {field["key"]: field for field in repaired["fields"]}
            self.assertEqual(fields["meeting_topic"]["derive_from_fields"], ["field_1", "field_2"])
            self.assertEqual(fields["field_2"]["type"], "text")
            self.assertEqual(fields["field_2"]["output_style"], "prose")
            self.assertTrue(fields["field_2"]["residual"])
            with open(definition_path, encoding="utf-8") as file:
                self.assertEqual(json.load(file)["fields"][0]["derive_from_fields"], [])

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

    def test_custom_markdown_marks_residual_generation_failure(self):
        summary = {
            "summary_template": "custom",
            "session_id": "session-residual-error",
            "meeting_name": "周例会",
            "custom_template_name": "层级模板",
            "custom_template_markdown": "## 其他事项\n{{other}}\n",
            "custom_template_fields": [
                {"key": "other", "heading": "其他事项", "type": "text"},
            ],
            "template_data": {"other": ""},
            "summary_quality": {
                "residual_generation_errors": [
                    {"key": "other", "label": "其他事项", "error": "offline"},
                ],
            },
        }

        markdown = MeetingLogStore.render_summary_markdown(summary)

        self.assertIn("## 其他事项", markdown)
        self.assertIn("该专题生成失败，请重新生成总结", markdown)
        self.assertNotIn("{{other}}", markdown)

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

    def test_custom_markdown_renders_topic_and_point_structures(self):
        summary = {
            "summary_template": "custom",
            "session_id": "session-topic",
            "meeting_name": "AI 演讲",
            "custom_template_markdown": "# {{meeting_name}}\n\n## 议题\n{{topics}}\n\n## 观点\n{{points}}\n",
            "custom_template_fields": [
                {"key": "topics", "heading": "议题", "type": "list"},
                {"key": "points", "heading": "观点", "type": "list"},
            ],
            "template_data": {
                "topics": [
                    {
                        "topic": "AI对计算范式的颠覆",
                        "evidence_timestamp": "[242.699 - 246.299]",
                        "evidence_quote": "Artificial Intelligence has reinvented computing",
                    }
                ],
                "points": [
                    {"point": "AI扩展人类潜力"},
                    "• {'topic': 'NVIDIA的自我革新与AI发展', 'evidence_timestamp': '[57.861 - 65.141]', 'evidence_quote': 'For 33 years, NVIDIA had reinvented itself over and over again'}",
                ],
            },
        }

        markdown = MeetingLogStore.render_summary_markdown(summary)

        self.assertIn("- AI对计算范式的颠覆", markdown)
        self.assertIn("- AI扩展人类潜力", markdown)
        self.assertIn("- NVIDIA的自我革新与AI发展", markdown)
        self.assertNotIn("topic：", markdown)
        self.assertNotIn("point：", markdown)
        self.assertNotIn("evidence_timestamp", markdown)

    def test_custom_evidence_list_accepts_topic_and_timestamp(self):
        service = MeetingSummaryService()
        payload = {
            "source_segments": [
                {
                    "start": 242.699,
                    "end": 246.299,
                    "text": "Artificial Intelligence has reinvented computing from human coding to machine learning",
                }
            ]
        }
        items, rejected = service._evidence_items(
            [
                {
                    "topic": "AI对计算范式的颠覆",
                    "evidence_time": "[242.699 - 246.299]",
                    "evidence_quote": "Artificial Intelligence has reinvented computing",
                }
            ],
            payload,
            limit=8,
        )

        self.assertEqual(rejected, 0)
        self.assertEqual(items[0]["text"], "AI对计算范式的颠覆")
        self.assertEqual(items[0]["evidence_start"], 242.699)
        self.assertEqual(items[0]["evidence_end"], 246.299)

    def test_standard_markdown_renders_literal_list_items_without_dict_residue(self):
        summary = {
            "summary_template": "auto",
            "session_id": "session-standard-list",
            "meeting_name": "AI 演讲",
            "template_data": {
                "topics": [
                    "• {'point': 'AI正在彻底改变计算方式。', 'evidence_time': '[242.699 - 246.299]', 'evidence_quote': 'Artificial Intelligence has reinvented computing from human coding to machine learning.'}",
                ],
            },
        }

        markdown = MeetingLogStore.render_summary_markdown(summary)

        self.assertIn("- AI正在彻底改变计算方式。", markdown)
        self.assertNotIn("{'point'", markdown)
        self.assertNotIn("evidence_time", markdown)

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
