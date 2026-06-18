import json
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

from whisper_live.meeting import MeetingLogStore, MeetingSummaryService, SummaryGenerationError


class TestMeetingSummaryService(unittest.TestCase):
    def test_extract_meeting_text_prefers_source_segments(self):
        service = MeetingSummaryService(startup_command="")
        payload = {"source_segments": [{"start": "0", "end": "1", "text": "hello"}], "translation_segments": [{"text": "你好"}]}
        text = service.extract_meeting_text(payload)
        self.assertIn("hello", text)
        self.assertNotIn("你好", text)

    def test_extract_meeting_text_does_not_fall_back_to_translation(self):
        service = MeetingSummaryService(startup_command="")
        payload = {"source_segments": [], "translation_segments": [{"text": "translation only"}]}
        self.assertEqual(service.extract_meeting_text(payload), "")

    def test_call_chat_disables_thinking_and_limits_output(self):
        service = MeetingSummaryService(startup_command="")
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "<think>hidden</think>{\"overview\":\"ok\"}"}, "finish_reason": "stop"}]
        }).encode("utf-8")
        response.__enter__.return_value = response
        with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
            content = service.call_chat([{"role": "user", "content": "test"}])
        request_payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(request_payload["max_tokens"], 1536)
        self.assertEqual(request_payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(content, {"content": '{"overview":"ok"}', "finish_reason": "stop"})

    def test_request_json_retries_truncated_response_with_compact_prompt(self):
        service = MeetingSummaryService(startup_command="")
        responses = [
            {"content": '{"overview":"cut', "finish_reason": "length"},
            {"content": '{"overview":"ok"}', "finish_reason": "stop"},
        ]
        with mock.patch.object(service, "call_chat", side_effect=responses) as call_chat:
            data = service.request_json([{"role": "user", "content": "test"}])
        self.assertEqual(data["overview"], "ok")
        self.assertEqual(call_chat.call_count, 2)
        retry_messages = call_chat.call_args_list[1].args[0]
        self.assertIn("更精简的完整 JSON", retry_messages[-1]["content"])

    def test_request_json_raises_after_two_invalid_responses(self):
        service = MeetingSummaryService(startup_command="")
        with mock.patch.object(service, "call_chat", return_value={"content": '{"overview":', "finish_reason": "stop"}):
            with self.assertRaises(SummaryGenerationError) as caught:
                service.request_json([{"role": "user", "content": "test"}])
        self.assertEqual(caught.exception.code, "summary_response_invalid_json")

    def test_split_text_uses_configured_character_budget(self):
        service = MeetingSummaryService(startup_command="", max_chars_per_chunk=2000)
        text = "\n".join(["x" * 900, "y" * 900, "z" * 900])
        chunks = service.split_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertLessEqual(max(map(len, chunks)), 2000)

    def test_normalize_summary_filters_unverified_decisions_and_actions(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-1",
            "meeting_name": "项目会",
            "source_segments": [{"start": 10.0, "end": 20.0, "text": "确认周五前由张三完成接口联调。"}],
        }
        data = {
            "meeting_type": "project_meeting",
            "overview": "确认接口联调安排。",
            "decisions": [
                {"text": "周五前完成联调", "evidence_start": 10, "evidence_end": 20, "evidence_quote": "确认周五前由张三完成接口联调。"},
                {"text": "上线延期", "evidence_start": 10, "evidence_end": 20, "evidence_quote": "原文不存在"},
            ],
            "action_items": [
                {"task": "完成接口联调", "owner": "张三", "deadline": "周五前", "status": None,
                 "evidence_start": 10, "evidence_end": 20, "evidence_quote": "周五前由张三完成接口联调"},
            ],
        }
        summary = service.normalize_summary(data, payload, template="project_meeting")
        self.assertEqual(len(summary["decisions"]), 1)
        self.assertEqual(len(summary["action_items"]), 1)
        self.assertEqual(summary["summary_quality"]["filtered_unverified_count"], 1)
        self.assertEqual(summary["summary_template"], "project_meeting")

    def test_training_template_has_independent_sections_and_evidence(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-2",
            "meeting_name": "培训",
            "source_segments": [{"start": 0, "end": 10, "text": "人类容易高估过去的变化，低估未来的变化。"}],
        }
        data = {
            "meeting_type": "training_speech",
            "overview": "讨论时间认知偏差。",
            "thesis": "心理学需要关注未来。",
            "outline": ["时间认知偏差", "未来导向"],
            "key_points": [{"text": "人类低估未来变化", "evidence_start": 0, "evidence_end": 10,
                            "evidence_quote": "人类容易高估过去的变化，低估未来的变化。"}],
            "arguments": ["通过过去与未来变化感知的对比展开论证"],
            "cases": [],
            "data_points": [],
            "notable_quotes": [],
            "takeaways": ["应主动培养未来导向"],
            "asr_uncertainties": [],
            "timeline": [{"text": "时间认知偏差", "evidence_start": 0, "evidence_end": 10,
                          "evidence_quote": "人类容易高估过去的变化，低估未来的变化。"}],
        }
        summary = service.normalize_summary(data, payload, template="training_speech")
        markdown = MeetingLogStore.render_summary_markdown(summary)
        self.assertEqual(summary["summary_quality"]["evidence_count"], 2)
        self.assertIn("## 核心主旨", markdown)
        self.assertIn("## 核心观点", markdown)
        self.assertIn("## 内容时间线", markdown)
        self.assertNotIn("## 关键决策", markdown)
        self.assertNotIn("## 待办事项", markdown)

    def test_each_template_uses_its_own_prompt_schema(self):
        service = MeetingSummaryService(startup_command="")
        self.assertIn("project_status", service.prompt_for_template("project_meeting"))
        self.assertIn("pain_points", service.prompt_for_template("customer_interview"))
        self.assertIn("notable_quotes", service.prompt_for_template("training_speech"))
        self.assertIn("disagreements", service.prompt_for_template("discussion"))

    def test_training_template_enforces_balanced_item_limits(self):
        service = MeetingSummaryService(startup_command="")
        segments = [{"start": 0, "end": 10, "text": "有效原文证据。"}]
        evidence_items = [
            {"text": f"观点{index}", "evidence_start": 0, "evidence_end": 10, "evidence_quote": "有效原文证据。"}
            for index in range(12)
        ]
        summary = service.normalize_summary({
            "meeting_type": "training_speech",
            "outline": [f"章节{index}" for index in range(12)],
            "key_points": evidence_items,
            "timeline": evidence_items,
        }, {"source_segments": segments}, template="training_speech")
        self.assertEqual(len(summary["template_data"]["outline"]), 8)
        self.assertEqual(len(summary["template_data"]["key_points"]), 6)
        self.assertEqual(len(summary["template_data"]["timeline"]), 8)

    def test_stage_prompt_only_requests_selected_fields(self):
        prompt = MeetingSummaryService.prompt_for_stage(
            "training_speech", "timeline", ("timeline",)
        )
        self.assertIn('"timeline"', prompt)
        self.assertNotIn('"key_points"', prompt)
        self.assertIn("禁止输出其他字段", prompt)

    def test_stage_truncation_falls_back_to_single_field_requests(self):
        service = MeetingSummaryService(startup_command="")
        truncated = SummaryGenerationError(
            "summary_response_truncated", "summary model response was truncated"
        )
        with mock.patch.object(
            service,
            "request_json",
            side_effect=[truncated, {"overview": "概述"}, {"thesis": "主旨"}],
        ) as request_json:
            result = service._request_stage_json(
                [
                    {"role": "system", "content": "stage"},
                    {"role": "user", "content": "source"},
                ],
                "training_speech",
                "foundation",
                ("overview", "thesis"),
                1024,
            )
        self.assertEqual(result, {"overview": "概述", "thesis": "主旨"})
        self.assertEqual(request_json.call_count, 3)
        self.assertIn("foundation.overview", request_json.call_args_list[1].args[0][0]["content"])

    def test_generate_uses_staged_pipeline_before_normalization(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-stage",
            "meeting_name": "培训",
            "source_segments": [
                {"start": 0, "end": 10, "text": "人类容易低估未来的变化。"}
            ],
        }
        staged_data = {
            "meeting_type": "training_speech",
            "overview": "讨论未来变化。",
            "thesis": "应重视未来变化。",
            "key_points": [
                {
                    "text": "人类容易低估未来变化",
                    "evidence_start": 0,
                    "evidence_end": 10,
                    "evidence_quote": "人类容易低估未来的变化。",
                }
            ],
        }
        with mock.patch.object(service, "ensure_ready"), mock.patch.object(
            service, "generate_staged", return_value=staged_data
        ) as generate_staged, mock.patch.object(service, "schedule_idle_shutdown"):
            summary = service.generate(payload, template="training_speech")
        generate_staged.assert_called_once()
        self.assertEqual(summary["overview"], "讨论未来变化。")
        self.assertEqual(summary["summary_quality"]["evidence_count"], 1)

    def test_validate_template_rejects_unknown_template(self):
        service = MeetingSummaryService(startup_command="")
        with self.assertRaises(ValueError):
            service.validate_template("unknown")

    def test_merge_summary_tree_merges_in_groups(self):
        service = MeetingSummaryService(startup_command="")
        summaries = [{"overview": str(index)} for index in range(9)]
        with mock.patch.object(
            service,
            "merge_summaries",
            side_effect=lambda group, _payload, _template="auto": {"overview": ",".join(item["overview"] for item in group)},
        ) as merge:
            result = service.merge_summary_tree(summaries, {})
        self.assertEqual(result["overview"], "0,1,2,3,4,5,6,7,8")
        self.assertEqual(merge.call_count, 3)

    def test_custom_fields_require_source_evidence_for_all_types(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "source_segments": [
                {"start": 10, "end": 20, "text": "会议确认周五完成接口联调，并记录当前进度正常。"}
            ]
        }
        fields = [
            {"key": "overview", "label": "概述", "type": "text"},
            {"key": "conclusion", "label": "结论", "type": "text"},
            {"key": "topics", "label": "议题", "type": "list"},
            {"key": "decisions", "label": "决策", "type": "evidence_list"},
            {"key": "actions", "label": "待办", "type": "table", "columns": ["任务", "截止时间"]},
        ]
        evidence = {
            "evidence_start": 10,
            "evidence_end": 20,
            "evidence_quote": "会议确认周五完成接口联调",
        }
        data = {
            "overview": {"text": "会议讨论接口联调安排。", **evidence},
            "conclusion": "模板示例中的结论",
            "topics": [
                {"text": "接口联调", **evidence},
                {"text": "前端展示优化", "evidence_start": 10, "evidence_end": 20, "evidence_quote": "原文不存在"},
            ],
            "decisions": [{"text": "周五完成接口联调", **evidence}],
            "actions": [
                {"任务": "完成接口联调", "截止时间": "周五", **evidence},
                {"任务": "优化翻译链路", "截止时间": "下周", "evidence_start": 10, "evidence_end": 20, "evidence_quote": "原文不存在"},
            ],
        }

        normalized, evidence_count, filtered = service._normalize_custom_data(
            data, payload, fields
        )

        self.assertEqual(normalized["overview"], "会议讨论接口联调安排。")
        self.assertEqual(normalized["conclusion"], "模板示例中的结论")
        self.assertEqual(normalized["topics"], ["接口联调", "前端展示优化"])
        self.assertEqual(len(normalized["decisions"]), 1)
        self.assertEqual(normalized["actions"], [
            {"任务": "完成接口联调", "截止时间": "周五"},
            {"任务": "优化翻译链路", "截止时间": "下周"},
        ])
        self.assertEqual(evidence_count, 1)
        self.assertEqual(filtered, 0)

    def test_custom_normalization_extracts_text_from_structured_values(self):
        service = MeetingSummaryService(startup_command="")
        payload = {"source_segments": []}
        fields = [
            {"key": "summary", "label": "讨论事项综述", "type": "text"},
            {"key": "topics", "label": "会议议题", "type": "list"},
            {"key": "actions", "label": "行动项", "type": "table", "columns": ["任务", "负责人"]},
        ]
        data = {
            "summary": {"content": "会议讨论项目推进。"},
            "topics": [
                {"title": "领域决赛", "content": "安排承办单位"},
                "{\"text\": \"海沃斯光电处置\"}",
            ],
            "actions": [
                {"任务": {"content": "完成材料"}, "负责人": ["张三", "李四"]},
            ],
        }

        normalized, evidence_count, filtered = service._normalize_custom_data(data, payload, fields)

        self.assertEqual(normalized["summary"], "会议讨论项目推进。")
        self.assertEqual(normalized["topics"], ["领域决赛：安排承办单位", "海沃斯光电处置"])
        self.assertEqual(normalized["actions"], [{"任务": "完成材料", "负责人": "张三\n李四"}])
        self.assertEqual(evidence_count, 0)
        self.assertEqual(filtered, 0)

    def test_custom_evidence_list_rejects_malformed_items_without_json_residue(self):
        service = MeetingSummaryService(startup_command="")
        payload = {"source_segments": [{"start": 1, "end": 2, "text": "会议确认通过方案。"}]}
        fields = [{"key": "evidence", "label": "原文依据", "type": "evidence_list"}]
        data = {"evidence": [{"content": "通过方案", "evidence_start": 1, "evidence_end": 2, "evidence_quote": "不存在"}]}

        normalized, evidence_count, filtered = service._normalize_custom_data(data, payload, fields)

        self.assertEqual(normalized["evidence"], [])
        self.assertEqual(evidence_count, 0)
        self.assertEqual(filtered, 1)

    def test_session_time_converts_utc_to_configured_timezone(self):
        service = MeetingSummaryService(startup_command="")
        with mock.patch.dict(os.environ, {"TZ": "Asia/Shanghai"}):
            parsed = service._parse_session_time("2026-06-15T06:41:12.342Z")
        self.assertEqual(parsed.strftime("%Y-%m-%d %H:%M:%S"), "2026-06-15 14:41:12")

    def test_custom_metadata_uses_full_source_range(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "meeting_name": "PengKaiping",
            "model": "model/funasr/paraformer-zh-streaming",
            "source_segments": [
                {"start": "10.100", "end": "22.400", "text": "第一段", "language": "zh",
                 "session_started_at": "2026-06-15T06:41:12.342"},
                {"start": "903.700", "end": "908.800", "text": "最后一段", "language": "en",
                 "session_started_at": "2026-06-15T06:41:12.342"},
            ],
        }
        fields = [{
            "key": "info", "type": "table", "columns": ["项目", "内容"],
            "metadata_enrichment": True,
        }]
        enriched = service._enrich_custom_metadata({"info": [
            {"项目": "会议基本信息", "内容": "会议名称：PengKaiping；时间：10.100 - 22.400"},
            {"项目": "会议时间", "内容": "10.100 - 22.400"},
            {"项目": "主持人", "内容": "彭凯平"},
        ]}, payload, fields)
        rows = {row["项目"]: row["内容"] for row in enriched["info"]}
        self.assertEqual(rows["会议名称"], "PengKaiping")
        self.assertEqual(rows["开始时间"], "2026-06-15 06:41:22")
        self.assertEqual(rows["结束时间"], "2026-06-15 06:56:21")
        self.assertEqual(rows["时长"], "14分59秒")
        self.assertEqual(rows["语言"], "zh, en")
        self.assertEqual(rows["主持人"], "彭凯平")
        self.assertNotIn("会议基本信息", rows)
        self.assertNotIn("会议时间", rows)

    def test_custom_truncation_splits_into_single_fields(self):
        service = MeetingSummaryService(startup_command="")
        fields = [
            {"key": "overview", "label": "概述", "type": "text"},
            {"key": "topics", "label": "议题", "type": "list"},
        ]
        truncated = SummaryGenerationError(
            "summary_response_truncated", "summary model response was truncated"
        )
        with mock.patch.object(
            service,
            "request_json",
            side_effect=[truncated, {"overview": {}}, {"topics": []}],
        ) as request_json:
            result = service._request_custom_fields(
                [{"role": "system", "content": "prompt"}, {"role": "user", "content": "source"}],
                {"id": "custom"}, fields, "custom-test",
            )
        self.assertEqual(result, {"overview": {}, "topics": []})
        self.assertEqual(request_json.call_count, 3)

    def test_custom_quality_rejects_empty_required_field(self):
        service = MeetingSummaryService(startup_command="")
        fields = [
            {"key": "overview", "label": "讨论综述", "type": "text", "required": True},
            {"key": "topics", "label": "议题", "type": "list", "required": False},
        ]
        issues, missing = service._custom_quality_issues(
            {"overview": "", "topics": ["议题"]}, fields, evidence_count=0, filtered=0
        )
        self.assertEqual(missing, [{"key": "overview", "label": "讨论综述"}])
        self.assertEqual(issues, ["required_fields_empty"])

    def test_custom_quality_does_not_require_evidence_for_plain_fields(self):
        service = MeetingSummaryService(startup_command="")
        fields = [
            {"key": "overview", "label": "综述", "type": "text", "required": True},
            {"key": "topics", "label": "议题", "type": "list", "required": True},
            {"key": "info", "label": "基本信息", "type": "table", "required": False},
        ]
        issues, missing = service._custom_quality_issues(
            {"overview": "有效综述", "topics": ["议题一"], "info": [{"项目": "会议名称", "内容": "周会"}]},
            fields,
            evidence_count=0,
            filtered=20,
        )
        self.assertEqual(issues, [])
        self.assertEqual(missing, [])

    def test_custom_quality_still_requires_evidence_for_evidence_list(self):
        service = MeetingSummaryService(startup_command="")
        fields = [
            {"key": "decisions", "label": "决策", "type": "evidence_list", "required": True},
        ]
        issues, missing = service._custom_quality_issues(
            {"decisions": []}, fields, evidence_count=0, filtered=3
        )
        self.assertIn("required_fields_empty", issues)
        self.assertIn("no_valid_evidence", issues)
        self.assertIn("high_evidence_rejection", issues)
        self.assertEqual(missing, [{"key": "decisions", "label": "决策"}])

    def test_required_text_fields_are_generated_alone(self):
        service = MeetingSummaryService(startup_command="")
        fields = [
            {"key": "info", "type": "table"},
            {"key": "overview", "type": "text", "required": True},
            {"key": "topics", "type": "list"},
        ]
        groups = service._custom_field_groups(fields)
        self.assertEqual([[field["key"] for field in group] for group in groups], [["info"], ["overview"], ["topics"]])

    def test_custom_prompt_treats_template_examples_as_non_facts(self):
        service = MeetingSummaryService(startup_command="")
        fields = [
            {"key": "overview", "label": "概述", "type": "text", "description": "总结实时语音识别效果"},
            {"key": "topics", "label": "议题", "type": "list", "description": "提取议题"},
            {"key": "actions", "label": "待办", "type": "table", "description": "提取待办", "columns": ["任务"]},
        ]
        prompt = service._custom_prompt({}, fields)

        self.assertIn("不是会议事实", prompt)
        self.assertIn("没有依据时", prompt)
        self.assertIn('"overview":{"text":""}', prompt)
        self.assertIn('"topics":[""]', prompt)
        self.assertIn('"actions":[{"任务":""}]', prompt)
        self.assertIn("只有 evidence_list 字段必须提供", prompt)
        self.assertIn("禁止在字段内容里重复输出字段标题", prompt)
        self.assertIn("每个编号聚合一个议题", prompt)
        self.assertIn("覆盖整场会议", prompt)
        self.assertIn("不要输出时间戳", prompt)
        self.assertIn("编号组织输出", prompt)
        self.assertIn("禁止时间戳", prompt)
        self.assertIn("逐segment摘要", prompt)

    def test_custom_merge_prompt_uses_numbered_topic_structure(self):
        service = MeetingSummaryService(startup_command="")
        fields = [{"key": "key_items", "label": "讨论事项综述", "type": "text"}]
        prompt = service._custom_prompt({}, fields, merge=True)

        self.assertIn("按编号议题去重归并", prompt)
        self.assertIn("编号议题段落", prompt)
        self.assertIn("覆盖整场会议", prompt)

    def test_normalize_custom_text_cleans_markdown_bullets_and_heading(self):
        service = MeetingSummaryService(startup_command="")
        fields = [{"key": "key_items", "label": "讨论事项综述", "heading": "二、讨论事项综述", "type": "text"}]
        normalized, evidence_count, filtered = service._normalize_custom_data(
            {"key_items": {"text": "• 讨论事项综述：\n• 华境融资事项\n  • 结论：走线上办公流程\n- 南京概念验证"}},
            {"source_segments": []},
            fields,
        )

        self.assertEqual(
            normalized["key_items"],
            "- 华境融资事项\n  - 结论：走线上办公流程\n- 南京概念验证",
        )
        self.assertEqual(evidence_count, 0)
        self.assertEqual(filtered, 0)

    def test_normalize_custom_topics_filters_fragments(self):
        service = MeetingSummaryService(startup_command="")
        fields = [{"key": "topics", "label": "会议议题", "type": "list"}]
        normalized, evidence_count, filtered = service._normalize_custom_data(
            {"topics": ["华境融资", "华境融资", "嗯", "---", {"text": "南京概念验证项目"}]},
            {"source_segments": []},
            fields,
        )

        self.assertEqual(normalized["topics"], ["华境融资", "南京概念验证项目"])
        self.assertEqual(evidence_count, 0)
        self.assertEqual(filtered, 0)

    def test_custom_text_quality_detects_timeline_dump_and_repetition(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        body = "\n".join(
            f"- [{index}.100 - {index + 1}.200] 讨论华境融资相关事宜，涉及股东沟通及表决。"
            for index in range(20)
        )
        issues = service._custom_text_quality_issues(body, field)

        self.assertIn("timeline_dump", issues)
        self.assertIn("high_repetition", issues)
        self.assertIn("missing_numbered_topics", issues)

    def test_custom_text_quality_detects_single_bucket_bullet_list(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        body = "- 领域决赛安排\n" + "\n".join(
            f"  - 子事项{index}：继续讨论相关事项。" for index in range(14)
        )
        issues = service._custom_text_quality_issues(body, field)

        self.assertIn("bad_grouping_structure", issues)
        self.assertIn("missing_numbered_topics", issues)

    def test_custom_text_quality_accepts_numbered_topic_summary(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        body = "\n".join([
            "1. 领域决赛安排 会议讨论了项目数量、承办单位、费用和评审流程。",
            "2. 马新元团队转研究所事项 会议讨论了团队方向调整和后续立项流程。",
            "3. 海沃斯光电处理 会议讨论了债务、诉讼、清算和扬州协调安排。",
            "4. 院内跟投管理办法 会议同步了有限合伙平台和分配原则。",
            "5. 华境融资与三期项目 会议讨论了股东表决、子公司和投资安排。",
            "6. 南京概念验证事项 会议讨论了指标、项目来源和事务部协同。",
        ])

        self.assertEqual(service._custom_text_quality_issues(body, field), [])

    def test_custom_text_quality_uses_soft_target_for_numbered_topics(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        four_topics = "\n".join(
            f"{index}. 议题{index} 会议讨论了相关背景、进展、结论和后续安排。"
            for index in range(1, 5)
        )
        twelve_topics = "\n".join(
            f"{index}. 议题{index} 会议讨论了相关背景、进展、结论和后续安排。"
            for index in range(1, 13)
        )
        twenty_four_topics = "\n".join([
            "1. 领域决赛组织安排 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "2. 项目筛选复审流程 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "3. 专家邀请评审机制 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "4. 团队转研究所事项 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "5. 公益项目平台承接 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "6. 海沃斯债务处理 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "7. 扬州协调会议安排 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "8. 院内跟投管理办法 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "9. 四川院项目进展 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "10. EDA合作方向评估 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "11. 航天抗辐射合作 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "12. 华境融资表决 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "13. 子公司设立事项 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "14. 南京概念验证 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "15. 江北项目引进 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "16. 无线微款项验收 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "17. 研发协议盖章 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "18. 第三方测试报告 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "19. 审计合规材料 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "20. 经营能力说明 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "21. 投资流程安排 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "22. 供应链技术分析 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "23. 团队协作分工 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "24. 下半年工作计划 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
        ])

        self.assertNotIn("bad_topic_count", service._custom_text_quality_issues(four_topics, field))
        self.assertNotIn("bad_topic_count", service._custom_text_quality_issues(twelve_topics, field))
        issues = service._custom_text_quality_issues(twenty_four_topics, field)
        self.assertNotIn("high_repetition", issues)
        self.assertNotIn("bad_topic_count", service._custom_text_quality_blocking(issues, twenty_four_topics))

    def test_generate_custom_rewrites_bad_summary_text_before_save(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-custom",
            "meeting_name": "周会",
            "source_segments": [{"start": 0, "end": 10, "text": "会议讨论领域决赛、华境融资和南京概念验证。"}],
        }
        definition = {
            "id": "template-1",
            "name": "模板",
            "fields": [{"key": "summary", "label": "讨论事项综述", "type": "text", "required": True}],
        }
        bad = "- [1.000 - 2.000] 讨论华境融资相关事宜。\n" * 8
        rewritten = "\n".join([
            "1. 领域决赛安排 会议讨论了项目筛选和评审流程。",
            "2. 华境融资事项 会议讨论了股东表决和项目推进。",
            "3. 南京概念验证 会议讨论了项目指标和协同安排。",
            "4. 无线微半导体验收 会议讨论了拨付款项和审计问题。",
            "5. 四川院项目进展 会议讨论了项目推进和资源安排。",
            "6. 院内跟投管理办法 会议同步了平台和分配原则。",
        ])
        with mock.patch.object(service, "ensure_ready"), \
             mock.patch.object(service, "_generate_custom_fields", return_value={"summary": {"text": bad}}), \
             mock.patch.object(service, "request_json", return_value={"summary": {"text": rewritten}}) as request_json, \
             mock.patch.object(service, "schedule_idle_shutdown"):
            summary = service.generate_custom(payload, definition)

        request_json.assert_called_once()
        self.assertEqual(summary["template_data"]["summary"], rewritten)

    def test_custom_rewrite_limits_source_context_and_output_budget(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        rewritten = "\n".join([
            "1. 领域决赛安排 会议讨论了项目筛选和评审流程。",
            "2. 华境融资事项 会议讨论了股东表决和项目推进。",
            "3. 南京概念验证 会议讨论了项目指标和协同安排。",
            "4. 无线微半导体验收 会议讨论了拨付款项和审计问题。",
            "5. 四川院项目进展 会议讨论了项目推进和资源安排。",
            "6. 院内跟投管理办法 会议同步了平台和分配原则。",
        ])
        with mock.patch.object(service, "request_json", return_value={"summary": {"text": rewritten}}) as request_json:
            result = service._custom_rewrite_text_field(
                {"meeting_name": "周会"},
                {"id": "template-1"},
                field,
                "坏" * 5000,
                "源" * 20000,
            )

        messages = request_json.call_args.args[0]
        user_content = messages[1]["content"]
        self.assertEqual(result, rewritten)
        self.assertEqual(request_json.call_args.kwargs["max_tokens"], 1600)
        self.assertLessEqual(user_content.count("坏"), 2500)
        self.assertIn("会议原文前中后摘录", user_content)
        self.assertLessEqual(user_content.count("源"), 5400)

    def test_custom_rewrite_retries_without_source_on_context_error(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        rewritten = "\n".join([
            "1. 领域决赛安排 会议讨论了项目筛选和评审流程。",
            "2. 华境融资事项 会议讨论了股东表决和项目推进。",
            "3. 南京概念验证 会议讨论了项目指标和协同安排。",
            "4. 无线微半导体验收 会议讨论了拨付款项和审计问题。",
            "5. 四川院项目进展 会议讨论了项目推进和资源安排。",
            "6. 院内跟投管理办法 会议同步了平台和分配原则。",
        ])
        context_error = RuntimeError("maximum context length exceeded")
        with mock.patch.object(
            service,
            "request_json",
            side_effect=[context_error, {"summary": {"text": rewritten}}],
        ) as request_json:
            result = service._custom_rewrite_text_field(
                {"meeting_name": "周会"},
                {"id": "template-1"},
                field,
                "坏" * 5000,
                "源" * 20000,
            )

        self.assertEqual(result, rewritten)
        self.assertEqual(request_json.call_count, 2)
        first_user = request_json.call_args_list[0].args[0][1]["content"]
        second_user = request_json.call_args_list[1].args[0][1]["content"]
        self.assertIn("会议原文前中后摘录", first_user)
        self.assertNotIn("会议原文前中后摘录", second_user)
        self.assertLessEqual(second_user.count("坏"), 3000)

    def test_generate_custom_uses_degraded_rewrite_after_context_error(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-custom",
            "meeting_name": "周会",
            "source_segments": [{"start": 0, "end": 10, "text": "会议讨论领域决赛、华境融资和南京概念验证。"}],
        }
        definition = {
            "id": "template-1",
            "name": "模板",
            "fields": [{"key": "summary", "label": "讨论事项综述", "type": "text", "required": True}],
        }
        bad = "- [1.000 - 2.000] 讨论华境融资相关事宜。\n" * 8
        rewritten = "\n".join([
            "1. 领域决赛安排 会议讨论了项目筛选和评审流程。",
            "2. 华境融资事项 会议讨论了股东表决和项目推进。",
            "3. 南京概念验证 会议讨论了项目指标和协同安排。",
            "4. 无线微半导体验收 会议讨论了拨付款项和审计问题。",
            "5. 四川院项目进展 会议讨论了项目推进和资源安排。",
            "6. 院内跟投管理办法 会议同步了平台和分配原则。",
        ])
        context_error = RuntimeError("maximum context length exceeded")
        with mock.patch.object(service, "ensure_ready"), \
             mock.patch.object(service, "_generate_custom_fields", return_value={"summary": {"text": bad}}), \
             mock.patch.object(service, "request_json", side_effect=[context_error, {"summary": {"text": rewritten}}]), \
             mock.patch.object(service, "schedule_idle_shutdown"):
            summary = service.generate_custom(payload, definition)

        self.assertEqual(summary["template_data"]["summary"], rewritten)

    def test_generate_custom_accepts_bad_topic_count_after_rewrite(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-custom",
            "meeting_name": "周会",
            "source_segments": [{"start": 0, "end": 10, "text": "会议讨论领域决赛、华境融资和南京概念验证。"}],
        }
        definition = {
            "id": "template-1",
            "name": "模板",
            "fields": [{"key": "summary", "label": "讨论事项综述", "type": "text", "required": True}],
        }
        bad = "- [1.000 - 2.000] 讨论华境融资相关事宜。\n" * 8
        three_topics = "\n".join([
            "1. 领域决赛安排 会议讨论了项目筛选和评审流程。",
            "2. 华境融资事项 会议讨论了股东表决和项目推进。",
            "3. 南京概念验证 会议讨论了项目指标和协同安排。",
        ])
        with mock.patch.object(service, "ensure_ready"), \
             mock.patch.object(service, "_generate_custom_fields", return_value={"summary": {"text": bad}}), \
             mock.patch.object(service, "request_json", return_value={"summary": {"text": three_topics}}), \
             mock.patch.object(service, "schedule_idle_shutdown"):
            summary = service.generate_custom(payload, definition)

        self.assertEqual(summary["template_data"]["summary"], three_topics)

    def test_generate_custom_compacts_too_many_topics_after_rewrite(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-custom",
            "meeting_name": "周会",
            "source_segments": [{"start": 0, "end": 10, "text": "会议讨论多个事项。"}],
        }
        definition = {
            "id": "template-1",
            "name": "模板",
            "fields": [{"key": "summary", "label": "讨论事项综述", "type": "text", "required": True}],
        }
        bad = "- [1.000 - 2.000] 讨论相关事宜。\n" * 8
        too_many = "\n".join(
            f"{index}. 议题{index} 会议讨论了相关背景、进展、结论和后续安排。"
            for index in range(1, 21)
        )
        compacted = "\n".join([
            "1. 领域决赛安排 会议合并讨论了项目筛选、评审流程和承办安排。",
            "2. 华境融资事项 会议合并讨论了股东表决、子公司设立和投资推进。",
            "3. 南京概念验证 会议合并讨论了项目指标、来源筛选和部门协同。",
            "4. 无线微半导体验收 会议合并讨论了第二笔款项、审计问题和里程碑验收。",
            "5. 四川院项目进展 会议合并讨论了项目推进、资源协调和后续安排。",
            "6. 院内跟投管理办法 会议合并讨论了平台设置、分配原则和管理要求。",
            "7. 海沃斯光电处理 会议合并讨论了债务诉讼、清算方案和地方协调。",
            "8. 公益活动安排 会议合并讨论了活动组织、参与人员和执行节奏。",
            "9. 团队转研究所事项 会议合并讨论了团队方向调整和后续立项流程。",
            "10. 合作协议进展 会议合并讨论了协议推进、责任分工和待确认事项。",
        ])
        with mock.patch.object(service, "ensure_ready"), \
             mock.patch.object(service, "_generate_custom_fields", return_value={"summary": {"text": bad}}), \
             mock.patch.object(service, "request_json", side_effect=[{"summary": {"text": too_many}}, {"summary": {"text": compacted}}]) as request_json, \
             mock.patch.object(service, "schedule_idle_shutdown"):
            summary = service.generate_custom(payload, definition)

        self.assertEqual(request_json.call_count, 2)
        self.assertEqual(summary["template_data"]["summary"], compacted)

    def test_generate_custom_allows_many_topics_when_structure_is_good(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-custom",
            "meeting_name": "周会",
            "source_segments": [{"start": 0, "end": 10, "text": "会议讨论多个事项。"}],
        }
        definition = {
            "id": "template-1",
            "name": "模板",
            "fields": [{"key": "summary", "label": "讨论事项综述", "type": "text", "required": True}],
        }
        bad = "- [1.000 - 2.000] 讨论相关事宜。\n" * 8
        many_topics = "\n".join([
            "1. 领域决赛组织安排 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "2. 项目筛选复审流程 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "3. 专家邀请评审机制 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "4. 团队转研究所事项 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "5. 公益项目平台承接 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "6. 海沃斯债务处理 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "7. 扬州协调会议安排 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "8. 院内跟投管理办法 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "9. 四川院项目进展 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "10. EDA合作方向评估 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "11. 航天抗辐射合作 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "12. 华境融资表决 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "13. 子公司设立事项 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "14. 南京概念验证 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "15. 江北项目引进 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "16. 无线微款项验收 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "17. 研发协议盖章 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "18. 第三方测试报告 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "19. 审计合规材料 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "20. 经营能力说明 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "21. 投资流程安排 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "22. 供应链技术分析 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "23. 团队协作分工 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
            "24. 下半年工作计划 会议围绕该事项的背景、关键进展、处理结论和后续责任安排进行了说明。",
        ])
        with mock.patch.object(service, "ensure_ready"), \
             mock.patch.object(service, "_generate_custom_fields", return_value={"summary": {"text": bad}}), \
             mock.patch.object(service, "request_json", return_value={"summary": {"text": many_topics}}), \
             mock.patch.object(service, "schedule_idle_shutdown"):
            summary = service.generate_custom(payload, definition)

        self.assertEqual(summary["template_data"]["summary"], many_topics)

    def test_custom_text_quality_detects_restarted_numbering_template_and_title_only(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        body = "\n".join([
            "1. 领域决赛安排 会议讨论了项目筛选和评审流程。",
            "2. 华境融资事项 会议讨论了股东表决和项目推进。",
            "1. 议题标题：南京概念验证",
            "2. 议题标题：无线微半导体验收",
            "3. 议题标题：四川院项目进展",
            "4. 议题标题：院内跟投管理办法",
        ])
        issues = service._custom_text_quality_issues(body, field)

        self.assertIn("restarted_numbering", issues)
        self.assertIn("template_residue", issues)
        self.assertIn("title_only_topics", issues)

    def test_custom_text_quality_detects_duplicate_topic_titles(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        body = "\n".join([
            "1. 项目协调与法律处理 会议讨论了海沃斯项目的法律意见和债务安排。",
            "2. 项目推进与团队协作 会议讨论了团队配合和后续执行。",
            "3. 项目协调与法律处理 会议再次讨论了同一项目的协调安排。",
            "4. 南京概念验证事项 会议讨论了项目筛选和资金安排。",
        ])

        self.assertIn("duplicate_topic_titles", service._custom_text_quality_issues(body, field))

    def test_generate_custom_derives_topic_list_from_numbered_summary(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-custom",
            "meeting_name": "周会",
            "source_segments": [{"start": 0, "end": 10, "text": "会议讨论领域决赛、华境融资、南京概念验证和无线微半导体验收。"}],
        }
        definition = {
            "id": "template-1",
            "name": "模板",
            "fields": [
                {"key": "topics", "label": "会议议题", "type": "list"},
                {"key": "summary", "label": "讨论事项综述", "type": "text", "required": True},
            ],
        }
        numbered = "\n".join([
            "1. 领域决赛安排 会议讨论了项目筛选和评审流程。",
            "2. 华境融资事项 会议讨论了股东表决和项目推进。",
            "3. 南京概念验证 会议讨论了项目指标和协同安排。",
            "4. 无线微半导体验收 会议讨论了第二笔款项和里程碑验收。",
            "5. 四川院项目进展 会议讨论了项目推进和资源安排。",
        ])
        stale_topics = ["领域决赛安排", "马新元团队转研究所事项"]
        with mock.patch.object(service, "ensure_ready"), \
             mock.patch.object(service, "_generate_custom_fields", return_value={"topics": stale_topics, "summary": {"text": numbered}}), \
             mock.patch.object(service, "schedule_idle_shutdown"):
            summary = service.generate_custom(payload, definition)

        self.assertEqual(summary["template_data"]["topics"][0], "领域决赛安排")
        self.assertIn("无线微半导体验收", summary["template_data"]["topics"])

    def test_custom_text_quality_detail_includes_numbered_count(self):
        service = MeetingSummaryService(startup_command="")
        field = {"key": "summary", "label": "讨论事项综述", "type": "text"}
        body = "\n".join([
            "1. 领域决赛安排 会议讨论了项目筛选和评审流程。",
            "2. 华境融资事项 会议讨论了股东表决和项目推进。",
            "3. 南京概念验证 会议讨论了项目指标和协同安排。",
        ])
        detail = service._custom_text_quality_detail(body, field)

        self.assertEqual(detail["issues"], ["bad_topic_count"])
        self.assertEqual(detail["numbered_count"], 3)
        self.assertEqual(detail["allowed_min"], 4)
        self.assertEqual(detail["soft_target_max"], 12)

    def test_generate_custom_rejects_bad_summary_after_rewrite(self):
        service = MeetingSummaryService(startup_command="")
        payload = {
            "session_id": "session-custom",
            "meeting_name": "周会",
            "source_segments": [{"start": 0, "end": 10, "text": "会议讨论领域决赛、华境融资和南京概念验证。"}],
        }
        definition = {
            "id": "template-1",
            "name": "模板",
            "fields": [{"key": "summary", "label": "讨论事项综述", "type": "text", "required": True}],
        }
        bad = "- [1.000 - 2.000] 讨论华境融资相关事宜。\n" * 8
        with mock.patch.object(service, "ensure_ready"), \
             mock.patch.object(service, "_generate_custom_fields", return_value={"summary": {"text": bad}}), \
             mock.patch.object(service, "request_json", return_value={"summary": {"text": bad}}), \
             mock.patch.object(service, "schedule_idle_shutdown"):
            with self.assertRaises(SummaryGenerationError) as caught:
                service.generate_custom(payload, definition)

        self.assertEqual(caught.exception.code, "summary_quality_insufficient")
        self.assertIn("custom_text_quality_insufficient", caught.exception.details["issues"])
        self.assertEqual(caught.exception.details["text_quality_fields"][0]["key"], "summary")
