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
        self.assertEqual(normalized["conclusion"], "")
        self.assertEqual(normalized["topics"], ["接口联调"])
        self.assertEqual(len(normalized["decisions"]), 1)
        self.assertEqual(normalized["actions"], [{"任务": "完成接口联调", "截止时间": "周五"}])
        self.assertEqual(evidence_count, 4)
        self.assertEqual(filtered, 3)

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
        self.assertIn('"overview":{"text":"","evidence_start":0', prompt)
        self.assertIn('"topics":[{"text":"","evidence_start":0', prompt)
        self.assertIn('"actions":[{"任务":"","evidence_start":0', prompt)
