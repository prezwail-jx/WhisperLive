import os
import asyncio
import time
import threading
import queue
import json
import functools
import logging
import re
import shutil
import tempfile
import subprocess
import shlex
import urllib.error
import urllib.request
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse, JSONResponse, FileResponse
import uvicorn
from faster_whisper import WhisperModel
import torch

from enum import Enum

from whisper_live import metrics as wl_metrics
from typing import List, Optional
import numpy as np
from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed
from whisper_live.vad import VoiceActivityDetector
from whisper_live.backend.base import ServeClientBase

logging.basicConfig(level=logging.INFO)


def parse_hotword_config(text):
    hotwords = []
    translation_glossary = {}
    normalized_lines = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" not in line:
            hotwords.append(line)
            normalized_lines.append(line)
            continue

        source, target = (part.strip() for part in line.split("=>", 1))
        if not source or not target:
            logging.warning("Ignoring invalid hotword translation rule: %r", line)
            continue
        hotwords.append(source)
        translation_glossary[source] = target
        normalized_lines.append(f"{source} => {target}")

    return {
        "text": "\n".join(normalized_lines),
        "hotwords": hotwords,
        "translation_glossary": translation_glossary,
        "count": len(hotwords),
        "translation_count": len(translation_glossary),
    }


def normalize_hotword_text(text):
    return parse_hotword_config(text)["text"]


def hotword_text_to_prompt(text):
    parsed = parse_hotword_config(text)
    if not parsed["hotwords"]:
        return None
    return " ".join(parsed["hotwords"])


def count_hotwords(text):
    return parse_hotword_config(text)["count"]


class SummaryGenerationError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class MeetingSummaryService:
    MAX_MERGE_BATCH_SIZE = 4
    MAX_CONTEXT_RETRIES = 2
    CONTEXT_ERROR_MARKERS = ("maximum context length", "input_tokens", "context length")
    TEMPLATE_CONFIGS = {
        "auto": {
            "meeting_type": "other",
            "prompt": """自动识别内容类型并生成通用可信总结。
输出 JSON：
{
  "meeting_type": "project_meeting|customer_interview|training_speech|discussion|other",
  "overview": "不超过200字",
  "topics": [],
  "key_points": [{"text":"", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "decisions": [{"text":"", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "action_items": [{"task":"", "owner":null, "deadline":null, "status":null, "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "risks": [],
  "open_questions": [],
  "follow_ups": []
}
只保留原文实际存在的栏目；关键观点、决策和待办必须提供证据。""",
        },
        "project_meeting": {
            "meeting_type": "project_meeting",
            "prompt": """这是项目会议。重点分析项目状态、进展、决策、执行责任和阻塞，不要输出培训或演讲式启示。
输出 JSON：
{
  "meeting_type": "project_meeting",
  "overview": "项目目标、当前状态和本次结果，不超过200字",
  "project_status": "原文明示的整体状态，没有则为空字符串",
  "progress": [{"text":"已完成或正在推进的事项", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "decisions": [{"text":"明确决定", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "action_items": [{"task":"明确任务", "owner":null, "deadline":null, "status":null, "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "blockers": [{"text":"明确阻塞", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "risks": [],
  "open_questions": [],
  "next_steps": []
}
进展、决策、待办和阻塞必须提供原文证据；不得生成整理纪要等建议性待办。""",
        },
        "customer_interview": {
            "meeting_type": "customer_interview",
            "prompt": """这是客户访谈。重点还原客户背景、真实需求、痛点、场景、反馈和异议，不要把客户表达改写成内部决策。
输出 JSON：
{
  "meeting_type": "customer_interview",
  "overview": "客户是谁、主要诉求和访谈结论，不超过200字",
  "customer_profile": [],
  "needs": [{"text":"客户明确需求", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "pain_points": [{"text":"客户明确痛点", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "use_cases": [{"text":"使用场景", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "feedback": [{"text":"产品或方案反馈", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "objections": [{"text":"顾虑或异议", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "action_items": [{"task":"双方明确约定的后续动作", "owner":null, "deadline":null, "status":null, "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "open_questions": [],
  "follow_ups": []
}
需求、痛点、场景、反馈、异议和待办必须提供原文证据。""",
        },
        "training_speech": {
            "meeting_type": "training_speech",
            "prompt": """这是培训或演讲。重点提炼知识结构、论证逻辑、案例、数据和启示，不得虚构项目决策、负责人或待办。
输出 JSON：
{
  "meeting_type": "training_speech",
  "overview": "主题、主旨和整体结论，不超过250字",
  "thesis": "演讲核心主张",
  "outline": ["内容章节或论述顺序"],
  "key_points": [{"text":"核心观点", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "arguments": ["主要论证逻辑"],
  "cases": [{"text":"案例、研究或人物观点", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "data_points": [{"text":"关键数字或研究结果", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "notable_quotes": [{"text":"值得保留的原话", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "takeaways": ["听众可直接理解和应用的启示"],
  "asr_uncertainties": ["疑似识别错误的人名、术语、数字或残句"],
  "timeline": [{"text":"章节主题", "evidence_start":0, "evidence_end":0, "evidence_quote":""}]
}
核心观点、案例、数据、原话和时间线必须提供原文证据。启示只能概括讲者已表达的内容。""",
        },
        "discussion": {
            "meeting_type": "discussion",
            "prompt": """这是讨论会或头脑风暴。重点区分不同观点、共识、分歧、决策和未决问题，不要把讨论中的提议误写成已决定事项。
输出 JSON：
{
  "meeting_type": "discussion",
  "overview": "讨论主题、主要观点和当前结论，不超过200字",
  "discussion_topics": [],
  "viewpoints": [{"speaker":null, "text":"观点", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "consensus": [{"text":"明确共识", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "disagreements": [{"text":"明确分歧", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "decisions": [{"text":"最终明确决定", "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "action_items": [{"task":"明确任务", "owner":null, "deadline":null, "status":null, "evidence_start":0, "evidence_end":0, "evidence_quote":""}],
  "open_questions": [],
  "follow_ups": []
}
观点、共识、分歧、决策和待办必须提供原文证据。""",
        },
    }
    MEETING_TYPES = {"project_meeting", "customer_interview", "training_speech", "discussion", "other"}
    TEMPLATE_LIMITS = {
        "auto": {"topics": 6, "key_points": 6, "decisions": 5, "action_items": 6, "risks": 5, "open_questions": 5, "follow_ups": 5},
        "project_meeting": {"progress": 6, "decisions": 5, "action_items": 8, "blockers": 5, "risks": 5, "open_questions": 5, "next_steps": 5},
        "customer_interview": {"customer_profile": 5, "needs": 6, "pain_points": 6, "use_cases": 5, "feedback": 5, "objections": 5, "action_items": 6, "open_questions": 5, "follow_ups": 5},
        "training_speech": {"outline": 8, "key_points": 6, "arguments": 5, "cases": 4, "data_points": 4, "notable_quotes": 3, "takeaways": 5, "asr_uncertainties": 5, "timeline": 8},
        "discussion": {"discussion_topics": 6, "viewpoints": 8, "consensus": 5, "disagreements": 5, "decisions": 5, "action_items": 6, "open_questions": 5, "follow_ups": 5},
    }
    FIELD_SCHEMAS = {
        "meeting_type": '"meeting_type":"project_meeting|customer_interview|training_speech|discussion|other"',
        "overview": '"overview":"内容概述"',
        "topics": '"topics":["核心议题"]',
        "key_points": '"key_points":[{"text":"核心观点","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "decisions": '"decisions":[{"text":"明确决定","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "action_items": '"action_items":[{"task":"明确任务","owner":null,"deadline":null,"status":null,"evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "risks": '"risks":["原文明示的风险"]',
        "open_questions": '"open_questions":["尚未解决的问题"]',
        "follow_ups": '"follow_ups":["原文明示的后续安排"]',
        "project_status": '"project_status":"原文明示的整体状态"',
        "progress": '"progress":[{"text":"项目进展","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "blockers": '"blockers":[{"text":"明确阻塞","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "next_steps": '"next_steps":["原文明示的下一步"]',
        "customer_profile": '"customer_profile":["客户背景"]',
        "needs": '"needs":[{"text":"客户需求","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "pain_points": '"pain_points":[{"text":"客户痛点","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "use_cases": '"use_cases":[{"text":"使用场景","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "feedback": '"feedback":[{"text":"产品反馈","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "objections": '"objections":[{"text":"顾虑或异议","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "thesis": '"thesis":"核心主张"',
        "outline": '"outline":["章节或论述顺序"]',
        "arguments": '"arguments":["论证逻辑"]',
        "cases": '"cases":[{"text":"案例或研究","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "data_points": '"data_points":[{"text":"关键数据","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "notable_quotes": '"notable_quotes":[{"text":"重要原话","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "takeaways": '"takeaways":["讲者已经表达的启示"]',
        "asr_uncertainties": '"asr_uncertainties":["疑似识别错误"]',
        "timeline": '"timeline":[{"text":"章节主题","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "discussion_topics": '"discussion_topics":["讨论议题"]',
        "viewpoints": '"viewpoints":[{"speaker":null,"text":"观点","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "consensus": '"consensus":[{"text":"明确共识","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
        "disagreements": '"disagreements":[{"text":"明确分歧","evidence_start":0,"evidence_end":0,"evidence_quote":"原文"}]',
    }
    TEMPLATE_STAGES = {
        "auto": (
            ("foundation", ("meeting_type", "overview", "topics"), 768),
            ("evidence", ("key_points", "decisions"), 1024),
            ("execution", ("action_items", "risks", "open_questions", "follow_ups"), 1024),
        ),
        "project_meeting": (
            ("foundation", ("meeting_type", "overview", "project_status"), 768),
            ("progress", ("progress", "decisions", "blockers"), 1100),
            ("execution", ("action_items", "risks", "open_questions", "next_steps"), 1024),
        ),
        "customer_interview": (
            ("foundation", ("meeting_type", "overview", "customer_profile"), 768),
            ("needs", ("needs", "pain_points", "use_cases"), 1100),
            ("feedback", ("feedback", "objections", "action_items"), 1100),
            ("follow_up", ("open_questions", "follow_ups"), 768),
        ),
        "training_speech": (
            ("foundation", ("meeting_type", "overview", "thesis", "outline", "arguments"), 1024),
            ("evidence", ("key_points", "cases", "data_points"), 1200),
            ("quotes", ("notable_quotes", "takeaways", "asr_uncertainties"), 1024),
            ("timeline", ("timeline",), 1100),
        ),
        "discussion": (
            ("foundation", ("meeting_type", "overview", "discussion_topics"), 768),
            ("viewpoints", ("viewpoints", "consensus", "disagreements"), 1200),
            ("execution", ("decisions", "action_items", "open_questions", "follow_ups"), 1100),
        ),
    }
    COMPACT_RETRY_PROMPT = """上一次输出被截断或不是完整 JSON。请重新输出更精简的完整 JSON：
1. 只保留最重要内容，各数组条目数量减半。
2. overview 不超过120字，thesis 不超过80字，普通文本条目不超过50字。
3. evidence_quote 截取能证明结论的最短连续原文，不超过80字。
4. 删除重复观点、重复案例和重复证据。
5. 必须闭合所有 JSON 对象和数组；不要输出 JSON 以外的任何字符。"""

    BASE_PROMPT = """你是可信会议内容分析助手。只能依据输入的带时间戳原文，不得使用外部知识补全，不得猜测或编造。
通用规则：
1. 输出必须是严格 JSON，不要 Markdown、代码块、解释、分析过程或思考内容。
2. 不确定的人名、术语、数字保持谨慎，不要擅自纠正。
3. 数组没有内容时返回 []，字符串没有内容时返回空字符串，未知负责人、时间和状态返回 null。
4. 所有要求证据的对象必须使用输入方括号中的时间，并原样摘录 evidence_quote；无法找到对应原文就不要输出该项。
5. 区分事实、观点、建议、提议、共识、决定和任务，不要混淆。
6. 忽略明显重复、口头语和无意义残句，但不得因此改变原意。"""


    def __init__(self, base_url="http://127.0.0.1:8001/v1", model="qwen3-4b-awq",
                 startup_command="bash scripts/start_summary_llm_service.sh", timeout=600,
                 ready_timeout=300, max_chars_per_chunk=4000, idle_shutdown_seconds=600):
        self.base_url = str(base_url or "").rstrip("/")
        self.model = model or "qwen3-4b-awq"
        self.startup_command = startup_command or ""
        self.timeout = int(timeout or 600)
        self.ready_timeout = int(ready_timeout or 300)
        self.max_chars_per_chunk = int(max_chars_per_chunk or 4000)
        self.idle_shutdown_seconds = int(idle_shutdown_seconds or 600)
        self.lock = threading.Lock()
        self.process = None
        self.started_by_us = False
        self.shutdown_timer = None

    def is_ready(self):
        if not self.base_url:
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def ensure_ready(self):
        if self.is_ready():
            return
        with self.lock:
            if self.is_ready():
                return
            if not (self.process and self.process.poll() is None):
                if not self.startup_command:
                    raise RuntimeError("summary LLM service is not running and no startup command is configured")
                logging.info("Starting summary LLM service: %s", self.startup_command)
                self.process = subprocess.Popen(shlex.split(self.startup_command), cwd=os.getcwd())
                self.started_by_us = True
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if self.is_ready():
                return
            if self.process and self.process.poll() is not None:
                raise RuntimeError(f"summary LLM service exited with code {self.process.returncode}")
            time.sleep(2)
        raise TimeoutError("summary LLM service did not become ready in time")

    def schedule_idle_shutdown(self):
        if not self.started_by_us or self.idle_shutdown_seconds <= 0:
            return
        with self.lock:
            if self.shutdown_timer:
                self.shutdown_timer.cancel()
            self.shutdown_timer = threading.Timer(self.idle_shutdown_seconds, self.shutdown_if_idle)
            self.shutdown_timer.daemon = True
            self.shutdown_timer.start()

    def shutdown_if_idle(self):
        with self.lock:
            process = self.process
            self.process = None
            self.shutdown_timer = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()

    def call_chat(self, messages, max_tokens=1536):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"summary LLM request failed: HTTP {exc.code}: {detail}") from exc
        choice = data["choices"][0]
        return {
            "content": self._strip_thinking(choice["message"]["content"]),
            "finish_reason": choice.get("finish_reason") or "",
        }

    def request_json(self, messages, max_tokens=None, context="summary"):
        last_error = None
        for attempt in range(2):
            request_messages = list(messages)
            if attempt:
                request_messages.append({"role": "system", "content": self.COMPACT_RETRY_PROMPT})
            if max_tokens is None:
                response = self.call_chat(request_messages)
            else:
                response = self.call_chat(request_messages, max_tokens=max_tokens)
            if response.get("finish_reason") == "length":
                last_error = SummaryGenerationError(
                    "summary_response_truncated",
                    "summary model response was truncated",
                )
            else:
                try:
                    return self._parse_json(response.get("content"))
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    last_error = SummaryGenerationError(
                        "summary_response_invalid_json",
                        "summary model returned invalid JSON",
                    )
                    last_error.__cause__ = exc
            if attempt == 0:
                logging.warning(
                    "%s context=%s; retrying with compact output constraints",
                    last_error.code,
                    context,
                )
        raise last_error

    @staticmethod
    def _strip_thinking(text):
        content = str(text or "")
        while "<think>" in content and "</think>" in content:
            start = content.find("<think>")
            end = content.find("</think>", start)
            if end < 0:
                break
            content = content[:start] + content[end + len("</think>"):]
        return content.strip()

    @classmethod
    def _is_context_error(cls, exc):
        message = str(exc).lower()
        return any(marker in message for marker in cls.CONTEXT_ERROR_MARKERS)

    @staticmethod
    def _parse_json(text):
        content = str(text or "").strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start:end + 1])
            raise

    @staticmethod
    def _list(value, limit=None):
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
        elif isinstance(value, str) and value.strip():
            items = [value.strip()]
        else:
            items = []
        return items[:limit] if limit else items

    @classmethod
    def validate_template(cls, template):
        template = str(template or "auto").strip()
        if template not in cls.TEMPLATE_CONFIGS:
            raise ValueError(f"unsupported summary template: {template}")
        return template

    @classmethod
    def prompt_for_template(cls, template):
        template = cls.validate_template(template)
        limits = ", ".join(f"{key}最多{value}条" for key, value in cls.TEMPLATE_LIMITS[template].items())
        return (
            f"{cls.BASE_PROMPT}\n\n当前模板：{template}\n"
            f"{cls.TEMPLATE_CONFIGS[template]['prompt']}\n数量限制：{limits}。"
        )

    @classmethod
    def prompt_for_stage(cls, template, stage_name, fields, merge=False):
        template = cls.validate_template(template)
        schema = ",\n  ".join(cls.FIELD_SCHEMAS[field] for field in fields)
        limits = ", ".join(
            f"{field}最多{cls.TEMPLATE_LIMITS[template][field]}条"
            for field in fields
            if field in cls.TEMPLATE_LIMITS[template]
        )
        template_intro = cls.TEMPLATE_CONFIGS[template]["prompt"].split("输出 JSON：", 1)[0].strip()
        task = "合并同一会议的分块结果，去重并保留最重要内容" if merge else "分析当前原文分块"
        return (
            f"{cls.BASE_PROMPT}\n\n"
            f"当前模板：{template}；当前阶段：{stage_name}。\n"
            f"{template_intro}\n{task}。只输出以下字段，禁止输出其他字段：\n"
            f"{{\n  {schema}\n}}\n"
            f"{limits + '。' if limits else ''}"
            "字符串应简洁；证据引用使用能证明结论的最短连续原文，不超过80字。"
        )

    @staticmethod
    def _float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_evidence_text(value):
        return re.sub(r"[\s，。！？!?；;：:、,.\"'（）()【】\[\]\-—]+", "", str(value or "")).lower()

    def _validate_evidence(self, item, payload):
        start = self._float(item.get("evidence_start"))
        end = self._float(item.get("evidence_end"))
        quote = str(item.get("evidence_quote") or "").strip()
        if start is None or end is None or end < start or not quote:
            return None
        matching = []
        for segment in payload.get("source_segments") or []:
            if not isinstance(segment, dict):
                continue
            segment_start = self._float(segment.get("start"))
            segment_end = self._float(segment.get("end"))
            if segment_start is None or segment_end is None:
                continue
            if segment_start <= end and segment_end >= start:
                body = str(segment.get("text") or "").strip()
                if body:
                    matching.append(body)
        source_text = self._normalize_evidence_text("".join(matching))
        quote_text = self._normalize_evidence_text(quote)
        if len(quote_text) < 2 or not source_text:
            return None
        if quote_text not in source_text and source_text not in quote_text:
            return None
        return {"evidence_start": start, "evidence_end": end, "evidence_quote": quote}

    def _evidence_items(self, value, payload, text_key="text", extra_keys=(), limit=None):
        accepted, filtered = [], 0
        if not isinstance(value, list):
            return accepted, filtered
        for item in value[:limit] if limit else value:
            if not isinstance(item, dict):
                filtered += 1
                continue
            body = str(item.get(text_key) or "").strip()
            evidence = self._validate_evidence(item, payload)
            if not body or not evidence:
                filtered += 1
                continue
            normalized = {text_key: body}
            for key in extra_keys:
                normalized[key] = str(item.get(key)).strip() if item.get(key) not in (None, "") else None
            normalized.update(evidence)
            accepted.append(normalized)
        return accepted, filtered

    def _actions(self, value, payload, limit=None):
        return self._evidence_items(
            value,
            payload,
            text_key="task",
            extra_keys=("owner", "deadline", "status"),
            limit=limit,
        )

    def _normalize_template_data(self, data, payload, template):
        filtered = 0
        evidence_count = 0
        limits = self.TEMPLATE_LIMITS[template]

        def limited_list(field):
            return self._list(data.get(field), limits.get(field))

        def evidence(field, text_key="text", extra_keys=()):
            nonlocal filtered, evidence_count
            items, rejected = self._evidence_items(
                data.get(field), payload, text_key, extra_keys, limit=limits.get(field)
            )
            filtered += rejected
            evidence_count += len(items)
            return items

        def actions():
            nonlocal filtered, evidence_count
            items, rejected = self._actions(data.get("action_items"), payload, limit=limits.get("action_items"))
            filtered += rejected
            evidence_count += len(items)
            return items

        if template == "project_meeting":
            decisions = evidence("decisions")
            action_items = actions()
            template_data = {
                "project_status": str(data.get("project_status") or "").strip(),
                "progress": evidence("progress"),
                "decisions": decisions,
                "action_items": action_items,
                "blockers": evidence("blockers"),
                "risks": limited_list("risks"),
                "open_questions": limited_list("open_questions"),
                "next_steps": limited_list("next_steps"),
            }
        elif template == "customer_interview":
            action_items = actions()
            decisions = []
            template_data = {
                "customer_profile": limited_list("customer_profile"),
                "needs": evidence("needs"),
                "pain_points": evidence("pain_points"),
                "use_cases": evidence("use_cases"),
                "feedback": evidence("feedback"),
                "objections": evidence("objections"),
                "action_items": action_items,
                "open_questions": limited_list("open_questions"),
                "follow_ups": limited_list("follow_ups"),
            }
        elif template == "training_speech":
            decisions = []
            action_items = []
            template_data = {
                "thesis": str(data.get("thesis") or "").strip(),
                "outline": limited_list("outline"),
                "key_points": evidence("key_points"),
                "arguments": limited_list("arguments"),
                "cases": evidence("cases"),
                "data_points": evidence("data_points"),
                "notable_quotes": evidence("notable_quotes"),
                "takeaways": limited_list("takeaways"),
                "asr_uncertainties": limited_list("asr_uncertainties"),
                "timeline": evidence("timeline"),
            }
        elif template == "discussion":
            decisions = evidence("decisions")
            action_items = actions()
            template_data = {
                "discussion_topics": limited_list("discussion_topics"),
                "viewpoints": evidence("viewpoints", extra_keys=("speaker",)),
                "consensus": evidence("consensus"),
                "disagreements": evidence("disagreements"),
                "decisions": decisions,
                "action_items": action_items,
                "open_questions": limited_list("open_questions"),
                "follow_ups": limited_list("follow_ups"),
            }
        else:
            decisions = evidence("decisions")
            action_items = actions()
            template_data = {
                "topics": limited_list("topics"),
                "key_points": evidence("key_points"),
                "decisions": decisions,
                "action_items": action_items,
                "risks": limited_list("risks"),
                "open_questions": limited_list("open_questions"),
                "follow_ups": limited_list("follow_ups"),
            }
        return template_data, decisions, action_items, evidence_count, filtered

    def normalize_summary(self, data, payload, raw_text=None, template="auto"):
        if not isinstance(data, dict):
            data = {"overview": str(raw_text or data or "").strip()}
        template = self.validate_template(template)
        template_data, decisions, action_items, evidence_count, filtered = self._normalize_template_data(
            data, payload, template
        )
        meeting_type = str(data.get("meeting_type") or "").strip()
        if meeting_type not in self.MEETING_TYPES:
            meeting_type = self.TEMPLATE_CONFIGS[template]["meeting_type"]
        return {
            "session_id": payload.get("session_id") or "",
            "meeting_name": payload.get("meeting_name") or payload.get("client_name") or "",
            "generated_at": MeetingLogStore.now_iso(),
            "model": self.model,
            "summary_template": template,
            "meeting_type": meeting_type,
            "overview": str(data.get("overview") or "").strip(),
            "template_data": template_data,
            # Keep these common fields for API compatibility with existing consumers.
            "topics": template_data.get("topics", template_data.get("discussion_topics", [])),
            "decisions": decisions,
            "action_items": action_items,
            "risks": template_data.get("risks", []),
            "follow_ups": template_data.get("follow_ups", template_data.get("next_steps", [])),
            "open_questions": template_data.get("open_questions", []),
            "summary_quality": {
                "source_segment_count": len(payload.get("source_segments") or []),
                "evidence_count": evidence_count,
                "filtered_unverified_count": filtered,
            },
        }

    def extract_meeting_text(self, payload):
        lines = []
        for segment in payload.get("source_segments") or []:
            if not isinstance(segment, dict):
                continue
            body = str(segment.get("text") or "").strip()
            if body:
                lines.append(f"[{segment.get('start', '')} - {segment.get('end', '')}] {body}")
        return "\n".join(lines).strip()

    def split_text(self, text):
        return self.split_text_with_limit(text, self.max_chars_per_chunk)

    @staticmethod
    def split_text_with_limit(text, max_chars):
        max_chars = max(2000, int(max_chars))
        chunks, current, size = [], [], 0
        for line in text.splitlines():
            line_size = len(line) + 1
            if current and size + line_size > max_chars:
                chunks.append("\n".join(current))
                current, size = [], 0
            if line_size > max_chars:
                chunks.extend(line[index:index + max_chars] for index in range(0, len(line), max_chars))
                continue
            current.append(line)
            size += line_size
        if current:
            chunks.append("\n".join(current))
        return chunks

    def summarize_text(self, text, payload, template="auto", context_retry=0):
        messages = [
            {"role": "system", "content": self.prompt_for_template(template)},
            {"role": "user", "content": f"会议名称：{payload.get('meeting_name') or payload.get('client_name') or '未命名会议'}\n\n会议原文记录：\n{text}"},
        ]
        try:
            data = self.request_json(messages)
        except RuntimeError as exc:
            if not self._is_context_error(exc) or len(text) < 2000 or context_retry >= self.MAX_CONTEXT_RETRIES:
                raise
            smaller_chunks = self.split_text_with_limit(text, max(2000, int(len(text) * 0.8)))
            if len(smaller_chunks) < 2:
                raise
            logging.warning("Summary input exceeded model context; retrying with %d smaller chunks", len(smaller_chunks))
            return self.merge_summary_tree(
                [self.summarize_text(chunk, payload, template, context_retry + 1) for chunk in smaller_chunks],
                payload,
                template,
            )
        return self.normalize_summary(data, payload, template=template)

    @staticmethod
    def _summary_for_merge(summary):
        return {
            "meeting_type": summary.get("meeting_type") or "other",
            "overview": summary.get("overview") or "",
            **(summary.get("template_data") or {}),
        }

    def merge_summaries(self, summaries, payload, template="auto"):
        merge_inputs = [self._summary_for_merge(item) for item in summaries]
        data = self.request_json([
            {"role": "system", "content": self.prompt_for_template(template)},
            {"role": "user", "content": "下面是同一场会议的分段总结。请去重合并，保留原有证据时间和原文引用，不要创造新证据。\n" + json.dumps(merge_inputs, ensure_ascii=False, separators=(",", ":"))},
        ])
        return self.normalize_summary(data, payload, template=template)

    def merge_summary_tree(self, summaries, payload, template="auto"):
        current = list(summaries)
        batch_size = 2 if template == "training_speech" else self.MAX_MERGE_BATCH_SIZE
        while len(current) > 1:
            next_level = []
            for index in range(0, len(current), batch_size):
                group = current[index:index + batch_size]
                next_level.append(group[0] if len(group) == 1 else self.merge_summaries(group, payload, template))
            current = next_level
        return current[0]

    @staticmethod
    def _stage_data(data, fields):
        if not isinstance(data, dict):
            return {}
        return {field: data[field] for field in fields if field in data}

    def _request_stage_json(self, messages, template, stage_name, fields, max_tokens, merge=False):
        context = f"template={template} stage={stage_name} mode={'merge' if merge else 'extract'}"
        try:
            return self._stage_data(
                self.request_json(messages, max_tokens=max_tokens, context=context),
                fields,
            )
        except SummaryGenerationError as exc:
            if len(fields) == 1 or exc.code not in {
                "summary_response_truncated", "summary_response_invalid_json"
            }:
                raise
            logging.warning(
                "%s context=%s; splitting stage into %d field requests",
                exc.code,
                context,
                len(fields),
            )
            combined = {}
            for field in fields:
                field_messages = [
                    {
                        "role": "system",
                        "content": self.prompt_for_stage(
                            template, f"{stage_name}.{field}", (field,), merge=merge
                        ),
                    },
                    *messages[1:],
                ]
                field_data = self.request_json(
                    field_messages,
                    max_tokens=min(max_tokens, 768),
                    context=f"{context} field={field}",
                )
                combined.update(self._stage_data(field_data, (field,)))
            return combined

    def _merge_stage_results(self, results, template, stage_name, fields, max_tokens):
        results = [self._stage_data(item, fields) for item in results if item]
        if not results:
            return {}
        if len(results) == 1:
            return results[0]
        messages = [
            {
                "role": "system",
                "content": self.prompt_for_stage(template, stage_name, fields, merge=True),
            },
            {
                "role": "user",
                "content": (
                    "下面是同一会议按原文顺序得到的分块结果。请去重合并，"
                    "不得创造新的事实或证据：\n"
                    + json.dumps(results, ensure_ascii=False, separators=(",", ":"))
                ),
            },
        ]
        return self._request_stage_json(
            messages, template, stage_name, fields, max_tokens, merge=True
        )

    def _summarize_stage_text(
            self, text, payload, template, stage_name, fields, max_tokens, context_retry=0):
        messages = [
            {
                "role": "system",
                "content": self.prompt_for_stage(template, stage_name, fields),
            },
            {
                "role": "user",
                "content": (
                    f"会议名称：{payload.get('meeting_name') or payload.get('client_name') or '未命名会议'}"
                    f"\n\n会议原文记录：\n{text}"
                ),
            },
        ]
        try:
            return self._request_stage_json(
                messages, template, stage_name, fields, max_tokens
            )
        except SummaryGenerationError:
            raise
        except RuntimeError as exc:
            if (
                not self._is_context_error(exc)
                or len(text) < 2000
                or context_retry >= self.MAX_CONTEXT_RETRIES
            ):
                raise
            smaller_chunks = self.split_text_with_limit(text, max(2000, int(len(text) * 0.8)))
            if len(smaller_chunks) < 2:
                raise
            logging.warning(
                "Summary stage input exceeded model context: template=%s stage=%s chunks=%d",
                template,
                stage_name,
                len(smaller_chunks),
            )
            return self._merge_stage_results(
                [
                    self._summarize_stage_text(
                        chunk, payload, template, stage_name, fields, max_tokens, context_retry + 1
                    )
                    for chunk in smaller_chunks
                ],
                template,
                stage_name,
                fields,
                max_tokens,
            )

    def generate_staged(self, chunks, payload, template):
        combined = {}
        for stage_name, fields, max_tokens in self.TEMPLATE_STAGES[template]:
            logging.info(
                "Generating meeting summary stage: template=%s stage=%s chunks=%d fields=%s",
                template,
                stage_name,
                len(chunks),
                ",".join(fields),
            )
            chunk_results = [
                self._summarize_stage_text(
                    chunk, payload, template, stage_name, fields, max_tokens
                )
                for chunk in chunks
            ]
            combined.update(
                self._merge_stage_results(
                    chunk_results, template, stage_name, fields, max_tokens
                )
            )
        return combined

    def generate(self, payload, template="auto"):
        template = self.validate_template(template)
        text = self.extract_meeting_text(payload)
        if not text:
            raise ValueError("meeting log has no completed source segments")
        self.ensure_ready()
        chunks = self.split_text(text)
        logging.info(
            "Generating staged meeting summary from source only: segments=%d chars=%d chunks=%d template=%s",
            len(payload.get("source_segments") or []), len(text), len(chunks), template,
        )
        data = self.generate_staged(chunks, payload, template)
        summary = self.normalize_summary(data, payload, template=template)
        self.schedule_idle_shutdown()
        return summary


class MeetingLogStore:
    UNSAFE_FILENAME_CHARS = set('/\\:*?"<>|')

    def __init__(self, directory="logs"):
        self.directory = directory or "logs"
        self.lock = threading.Lock()
        self.sessions = {}
        os.makedirs(self.directory, exist_ok=True)
        self._load_existing_sessions()

    def _load_existing_sessions(self):
        loaded = 0
        for root, directories, filenames in os.walk(self.directory):
            directories[:] = [name for name in directories if not name.endswith("-summaries")]
            for filename in filenames:
                if not filename.endswith(".json") or filename.endswith("-summary.json"):
                    continue
                json_path = os.path.join(root, filename)
                try:
                    with open(json_path, "r", encoding="utf-8") as file:
                        payload = json.load(file)
                except (OSError, ValueError):
                    continue
                if not isinstance(payload, dict) or not payload.get("session_id"):
                    continue
                if not isinstance(payload.get("source_segments"), list):
                    continue
                session_id = str(payload["session_id"])
                current = self.sessions.get(session_id)
                if current and str(current["payload"].get("updated_at") or "") >= str(payload.get("updated_at") or ""):
                    continue
                stem, _ext = os.path.splitext(json_path)
                md_path = f"{stem}.md"
                self.sessions[session_id] = {
                    "payload": payload,
                    "source_keys": {self.segment_key(item) for item in payload.get("source_segments") or [] if isinstance(item, dict)},
                    "translation_keys": {self.segment_key(item) for item in payload.get("translation_segments") or [] if isinstance(item, dict)},
                    "json_path": json_path,
                    "md_path": md_path,
                    "json_filename": filename,
                    "md_filename": os.path.basename(md_path),
                }
                loaded += 1
        if loaded:
            logging.info("Restored %d meeting log sessions from %s", loaded, self.directory)

    @classmethod
    def safe_name(cls, value):
        name = str(value or "").strip() or "meeting"
        cleaned = "".join("_" if char in cls.UNSAFE_FILENAME_CHARS or ord(char) < 32 else char for char in name)
        cleaned = cleaned.strip(" ._") or "meeting"
        return cleaned[:80]

    @classmethod
    def safe_timestamp(cls, value):
        raw = str(value or "").strip() or cls.timestamp_for_filename()
        cleaned = "".join("_" if char in cls.UNSAFE_FILENAME_CHARS or ord(char) < 32 else char for char in raw)
        cleaned = cleaned.replace(":", "-").replace(".", "-").strip(" ._")
        return cleaned[:80] or cls.timestamp_for_filename()

    @staticmethod
    def timestamp_for_filename():
        return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())

    @staticmethod
    def now_iso():
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def segment_key(segment):
        return f"{segment.get('session_id', '')}|{segment.get('start', '')}|{segment.get('end', '')}|{segment.get('text', '')}"

    def session_paths(self, payload):
        meeting_dir = self.safe_name(payload.get("meeting_name") or payload.get("client_name") or "meeting")
        session_id = self.safe_name(payload.get("session_id") or payload.get("uid") or "session")
        started_at = self.safe_timestamp(payload.get("session_started_at") or payload.get("created_at"))
        directory = os.path.join(self.directory, meeting_dir)
        stem = f"{started_at}-{session_id}"
        return directory, stem, os.path.join(directory, f"{stem}.json"), os.path.join(directory, f"{stem}.md")

    def build_payload_from_options(self, options, backend=None):
        session_id = options.get("session_id") or options.get("uid")
        created_at = options.get("session_started_at") or self.now_iso()
        return {
            "meeting_id": options.get("meeting_id") or session_id,
            "session_id": session_id,
            "uid": options.get("uid"),
            "client_instance_id": options.get("client_instance_id") or "",
            "client_name": options.get("client_name") or options.get("meeting_name") or f"Client-{str(options.get('uid', ''))[:8]}",
            "meeting_name": options.get("meeting_name") or "",
            "hotwords_count": int(options.get("hotwords_count") or count_hotwords(options.get("hotwords"))),
            "hotwords_file": options.get("hotwords_file") or "",
            "created_at": created_at,
            "updated_at": created_at,
            "exported_at": None,
            "server": options.get("server") or "",
            "backend": backend.value if isinstance(backend, BackendType) else str(backend or options.get("backend") or ""),
            "model": options.get("model"),
            "source_language": options.get("language"),
            "translation_mode": options.get("target_language", "auto"),
            "status": "active",
            "source_segments": [],
            "translation_segments": [],
        }

    def save(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("meeting log payload must be a JSON object")
        meeting_name = payload.get("meeting_name") or payload.get("client_name") or "meeting"
        filename_stem = f"{self.safe_name(meeting_name)}-{self.timestamp_for_filename()}"
        with self.lock:
            filename = f"{filename_stem}.json"
            path = os.path.join(self.directory, filename)
            suffix = 1
            while os.path.exists(path):
                filename = f"{filename_stem}-{suffix}.json"; path = os.path.join(self.directory, filename); suffix += 1
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2); file.write("\n")
        return {"saved": True, "filename": filename, "path": path}

    def start_session(self, options, backend=None):
        payload = self.build_payload_from_options(options, backend=backend)
        session_id = payload.get("session_id")
        if not session_id:
            raise ValueError("session_id is required")
        directory, _stem, json_path, md_path = self.session_paths(payload)
        with self.lock:
            os.makedirs(directory, exist_ok=True)
            record = {"payload": payload, "source_keys": set(), "translation_keys": set(), "json_path": json_path, "md_path": md_path,
                      "json_filename": os.path.basename(json_path), "md_filename": os.path.basename(md_path)}
            self.sessions[session_id] = record
            self._write_record(record)
        return self.session_info(session_id)

    def append_segments(self, session_id, kind, segments):
        if not session_id or kind not in ("source", "translation"):
            return None
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            target = "source_segments" if kind == "source" else "translation_segments"
            key_name = "source_keys" if kind == "source" else "translation_keys"
            changed = False
            for segment in segments or []:
                if not isinstance(segment, dict) or not segment.get("completed"):
                    continue
                text = str(segment.get("text") or "").strip()
                if not text:
                    continue
                normalized = dict(segment); normalized["text"] = text; normalized.setdefault("session_id", session_id)
                key = self.segment_key(normalized)
                if key in record[key_name]:
                    continue
                record[key_name].add(key); record["payload"][target].append(normalized); changed = True
            if changed:
                record["payload"]["updated_at"] = self.now_iso()
                self._sort_segments(record["payload"][target])
                self._write_record(record)
            return self.session_info(session_id)

    def finish_session(self, session_id):
        if not session_id:
            return None
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            record["payload"]["status"] = "finished"
            record["payload"]["exported_at"] = self.now_iso()
            record["payload"]["updated_at"] = record["payload"]["exported_at"]
            self._write_record(record)
            return self.session_info(session_id)

    @staticmethod
    def _clone_payload(payload):
        return json.loads(json.dumps(payload, ensure_ascii=False))

    def get_session_payload(self, session_id):
        with self.lock:
            record = self.sessions.get(session_id)
            return self._clone_payload(record["payload"]) if record else None

    @staticmethod
    def summary_paths_for_record(record):
        stem, _ext = os.path.splitext(record["json_path"])
        return f"{stem}-summary.json", f"{stem}-summary.md"

    @staticmethod
    def summary_version_directory(record):
        stem, _ext = os.path.splitext(record["json_path"])
        return f"{stem}-summaries"

    @staticmethod
    def _atomic_write(path, content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix=".summary-", dir=os.path.dirname(path), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(content)
            os.replace(temporary_path, path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    def _version_entries(self, record):
        directory = self.summary_version_directory(record)
        entries = []
        if not os.path.isdir(directory):
            return entries
        for filename in sorted(os.listdir(directory)):
            match = re.fullmatch(r"v(\d{4})\.json", filename)
            if not match:
                continue
            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    summary = json.load(file)
            except (OSError, ValueError) as exc:
                logging.warning("Failed to read summary version %s: %s", path, exc)
                continue
            version = int(match.group(1))
            entries.append({
                "version": version,
                "template": summary.get("summary_template") or "legacy",
                "meeting_type": summary.get("meeting_type") or "other",
                "generated_at": summary.get("generated_at"),
                "json_filename": filename,
                "md_filename": f"v{version:04d}.md",
            })
        return entries

    def _ensure_legacy_summary_version(self, record):
        entries = self._version_entries(record)
        if entries:
            return entries
        summary_json, summary_md = self.summary_paths_for_record(record)
        if not os.path.isfile(summary_json):
            return []
        try:
            with open(summary_json, "r", encoding="utf-8") as file:
                summary = json.load(file)
        except (OSError, ValueError) as exc:
            logging.warning("Failed to import legacy summary %s: %s", summary_json, exc)
            return []
        summary = dict(summary)
        summary["version"] = 1
        summary.setdefault("summary_template", "legacy")
        directory = self.summary_version_directory(record)
        version_json = os.path.join(directory, "v0001.json")
        version_md = os.path.join(directory, "v0001.md")
        self._atomic_write(version_json, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if os.path.isfile(summary_md):
            with open(summary_md, "r", encoding="utf-8") as file:
                markdown = file.read()
        else:
            markdown = self.render_summary_markdown(summary)
        self._atomic_write(version_md, markdown)
        return self._version_entries(record)

    def session_info(self, session_id):
        record = self.sessions.get(session_id)
        if not record:
            return None
        summary = self.summary_info(session_id)
        return {
            "session_id": session_id,
            "meeting_name": record["payload"].get("meeting_name") or record["payload"].get("client_name") or "",
            "created_at": record["payload"].get("created_at"),
            "updated_at": record["payload"].get("updated_at"),
            "json_path": record["json_path"],
            "md_path": record["md_path"],
            "json_filename": record["json_filename"],
            "md_filename": record["md_filename"],
            "summary_json_path": summary["json_path"],
            "summary_md_path": summary["md_path"],
            "summary_json_filename": summary["json_filename"],
            "summary_md_filename": summary["md_filename"],
            "has_summary": summary["has_summary"],
            "latest_summary_version": summary["latest_version"],
            "source_count": len(record["payload"].get("source_segments", [])),
            "translation_count": len(record["payload"].get("translation_segments", [])),
            "status": record["payload"].get("status"),
        }

    def list_sessions(self):
        with self.lock:
            sessions = [self.session_info(session_id) for session_id in self.sessions]
        sessions = [item for item in sessions if item]
        sessions.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {"sessions": sessions}

    def get_session_file(self, session_id, file_format="md"):
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            if file_format == "json":
                return record["json_path"], "application/json", record["json_filename"]
            return record["md_path"], "text/markdown; charset=utf-8", record["md_filename"]

    def write_summary(self, session_id, summary):
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            entries = self._ensure_legacy_summary_version(record)
            version = max((item["version"] for item in entries), default=0) + 1
            summary = dict(summary)
            summary["version"] = version
            summary_json, summary_md = self.summary_paths_for_record(record)
            directory = self.summary_version_directory(record)
            version_json = os.path.join(directory, f"v{version:04d}.json")
            version_md = os.path.join(directory, f"v{version:04d}.md")
            json_content = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            markdown_content = self.render_summary_markdown(summary)
            self._atomic_write(version_json, json_content)
            self._atomic_write(version_md, markdown_content)
            self._atomic_write(summary_json, json_content)
            self._atomic_write(summary_md, markdown_content)
        return self.summary_info(session_id)

    def summary_info(self, session_id):
        record = self.sessions.get(session_id)
        if not record:
            return None
        summary_json, summary_md = self.summary_paths_for_record(record)
        versions = self._version_entries(record)
        latest = versions[-1] if versions else None
        return {
            "session_id": session_id,
            "json_path": summary_json,
            "md_path": summary_md,
            "json_filename": os.path.basename(summary_json),
            "md_filename": os.path.basename(summary_md),
            "has_summary": os.path.isfile(summary_json) and os.path.isfile(summary_md),
            "latest_version": latest["version"] if latest else None,
            "latest_template": latest["template"] if latest else None,
            "versions": versions,
        }

    def get_summary_file(self, session_id, file_format="md", version=None):
        with self.lock:
            record = self.sessions.get(session_id)
            if not record:
                return None
            if version is None:
                summary_json, summary_md = self.summary_paths_for_record(record)
                if file_format == "json":
                    return summary_json, "application/json", os.path.basename(summary_json)
                return summary_md, "text/markdown; charset=utf-8", os.path.basename(summary_md)
            try:
                version = int(version)
            except (TypeError, ValueError):
                return None
            if version < 1:
                return None
            directory = self.summary_version_directory(record)
            if file_format == "json":
                path = os.path.join(directory, f"v{version:04d}.json")
                return path, "application/json", os.path.basename(path)
            path = os.path.join(directory, f"v{version:04d}.md")
            return path, "text/markdown; charset=utf-8", os.path.basename(path)

    @staticmethod
    def _sort_segments(segments):
        segments.sort(key=lambda item: (str(item.get("session_started_at") or ""), float(item.get("start") or 0)))

    def _write_record(self, record):
        payload = record["payload"]
        with open(record["json_path"], "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2); file.write("\n")
        with open(record["md_path"], "w", encoding="utf-8") as file:
            file.write(self.render_markdown(payload))

    @staticmethod
    def render_markdown(payload):
        lines = [f"# {payload.get('meeting_name') or payload.get('client_name') or 'Meeting Log'}", "", "## 会议信息",
                 f"- Session ID: {payload.get('session_id') or ''}", f"- Client: {payload.get('client_name') or ''}",
                 f"- Started: {payload.get('created_at') or ''}", f"- Updated: {payload.get('updated_at') or ''}",
                 f"- Backend: {payload.get('backend') or ''}", f"- Model: {payload.get('model') or ''}",
                 f"- Source language: {payload.get('source_language') or 'auto'}", "", "## 原文记录"]
        for segment in payload.get("source_segments", []):
            lines.append(f"- [{segment.get('start', '')} - {segment.get('end', '')}] {segment.get('text', '')}")
        lines.extend(["", "## 翻译记录"])
        for segment in payload.get("translation_segments", []):
            lines.append(f"- [{segment.get('start', '')} - {segment.get('end', '')}] {segment.get('text', '')}")
        lines.extend(["", "## AI 总结", "", "_待生成_", ""])
        return "\n".join(lines)

    @staticmethod
    def render_summary_markdown(summary):
        def evidence_suffix(item):
            if not isinstance(item, dict) or not item.get("evidence_quote"):
                return ""
            return f"\n  - 依据 [{item.get('evidence_start', '')} - {item.get('evidence_end', '')}]：{item.get('evidence_quote')}"

        def append_text(lines, title, value):
            value = str(value or "").strip()
            if value:
                lines.extend(["", f"## {title}", value])

        def append_list(lines, title, values):
            values = values or []
            if not values:
                return
            lines.extend(["", f"## {title}"])
            for item in values:
                lines.append(f"- {item}")

        def append_evidence(lines, title, values, text_key="text", formatter=None):
            values = values or []
            if not values:
                return
            lines.extend(["", f"## {title}"])
            for item in values:
                if not isinstance(item, dict):
                    lines.append(f"- {item}")
                    continue
                body = formatter(item) if formatter else item.get(text_key) or ""
                lines.append(f"- {body}{evidence_suffix(item)}")

        template = summary.get("summary_template") or "auto"
        data = summary.get("template_data") or {}
        if not data:
            data = {
                "topics": summary.get("topics") or [],
                "decisions": summary.get("decisions") or [],
                "action_items": summary.get("action_items") or [],
                "risks": summary.get("risks") or [],
                "follow_ups": summary.get("follow_ups") or [],
                "open_questions": summary.get("open_questions") or [],
            }

        lines = [
            f"# {summary.get('meeting_name') or '会议总结'}", "", "## 会议信息",
            f"- Session ID: {summary.get('session_id') or ''}",
            f"- Version: {summary.get('version') or ''}",
            f"- Template: {template}",
            f"- Meeting type: {summary.get('meeting_type') or ''}",
            f"- Generated: {summary.get('generated_at') or ''}",
            f"- Model: {summary.get('model') or ''}",
        ]
        append_text(lines, "内容概述", summary.get("overview") or "未明确")

        action_formatter = lambda item: (
            f"{item.get('task') or '未明确'}（负责人：{item.get('owner') or '未明确'}；"
            f"时间：{item.get('deadline') or '未明确'}；状态：{item.get('status') or '未明确'}）"
        )

        if template == "project_meeting":
            append_text(lines, "项目状态", data.get("project_status"))
            append_evidence(lines, "项目进展", data.get("progress"))
            append_evidence(lines, "关键决策", data.get("decisions"))
            append_evidence(lines, "待办事项", data.get("action_items"), text_key="task", formatter=action_formatter)
            append_evidence(lines, "阻塞事项", data.get("blockers"))
            append_list(lines, "风险", data.get("risks"))
            append_list(lines, "未决问题", data.get("open_questions"))
            append_list(lines, "下一步", data.get("next_steps"))
        elif template == "customer_interview":
            append_list(lines, "客户画像", data.get("customer_profile"))
            append_evidence(lines, "客户需求", data.get("needs"))
            append_evidence(lines, "客户痛点", data.get("pain_points"))
            append_evidence(lines, "使用场景", data.get("use_cases"))
            append_evidence(lines, "产品反馈", data.get("feedback"))
            append_evidence(lines, "顾虑与异议", data.get("objections"))
            append_evidence(lines, "约定事项", data.get("action_items"), text_key="task", formatter=action_formatter)
            append_list(lines, "待确认问题", data.get("open_questions"))
            append_list(lines, "后续跟进", data.get("follow_ups"))
        elif template == "training_speech":
            append_text(lines, "核心主旨", data.get("thesis"))
            append_list(lines, "内容结构", data.get("outline"))
            append_evidence(lines, "核心观点", data.get("key_points"))
            append_list(lines, "论证逻辑", data.get("arguments"))
            append_evidence(lines, "案例与研究", data.get("cases"))
            append_evidence(lines, "关键数据", data.get("data_points"))
            append_evidence(lines, "重要原话", data.get("notable_quotes"))
            append_list(lines, "核心启示", data.get("takeaways"))
            append_evidence(lines, "内容时间线", data.get("timeline"))
            append_list(lines, "疑似识别问题", data.get("asr_uncertainties"))
        elif template == "discussion":
            append_list(lines, "讨论议题", data.get("discussion_topics"))
            append_evidence(
                lines,
                "主要观点",
                data.get("viewpoints"),
                formatter=lambda item: f"{item.get('speaker') or '发言人未明确'}：{item.get('text') or ''}",
            )
            append_evidence(lines, "已达成共识", data.get("consensus"))
            append_evidence(lines, "主要分歧", data.get("disagreements"))
            append_evidence(lines, "关键决策", data.get("decisions"))
            append_evidence(lines, "待办事项", data.get("action_items"), text_key="task", formatter=action_formatter)
            append_list(lines, "未决问题", data.get("open_questions"))
            append_list(lines, "后续跟进", data.get("follow_ups"))
        else:
            append_list(lines, "核心议题", data.get("topics"))
            append_evidence(lines, "关键观点", data.get("key_points"))
            append_evidence(lines, "关键决策", data.get("decisions"))
            append_evidence(lines, "待办事项", data.get("action_items"), text_key="task", formatter=action_formatter)
            append_list(lines, "风险与问题", data.get("risks"))
            append_list(lines, "未决问题", data.get("open_questions"))
            append_list(lines, "后续跟进", data.get("follow_ups"))

        quality = summary.get("summary_quality") or {}
        lines.extend([
            "", "## 证据校验",
            f"- 原文片段数：{quality.get('source_segment_count', 0)}",
            f"- 已验证带证据内容数：{quality.get('evidence_count', 0)}",
            f"- 已过滤无依据内容数：{quality.get('filtered_unverified_count', 0)}",
            "",
        ])
        return "\n".join(lines)



class MeetingHotwordStore:
    def __init__(self, directory="config/hotwords.d"):
        self.directory = directory
        self.lock = threading.Lock()

    @staticmethod
    def normalize_name(meeting_name):
        return str(meeting_name or "").strip()

    @staticmethod
    def _empty_record(meeting_name):
        return {
            "meeting_name": meeting_name,
            "filename": "",
            "text": "",
            "count": 0,
            "translation_count": 0,
            "translation_glossary": {},
            "updated_at": None,
        }

    def _safe_path(self, meeting_name):
        name = self.normalize_name(meeting_name)
        if not name:
            raise ValueError("meeting_name is required")
        if os.path.basename(name) != name or "/" in name or "\\" in name:
            raise ValueError("meeting_name must match a hotword txt filename")
        return name, os.path.join(self.directory, f"{name}.txt")

    def _record_from_file(self, meeting_name, path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as file:
                text = file.read()
        parsed = parse_hotword_config(text)
        filename = os.path.basename(path)
        return {
            "meeting_name": meeting_name,
            "filename": filename,
            "path": path,
            "text": parsed["text"],
            "count": parsed["count"],
            "translation_count": parsed["translation_count"],
            "translation_glossary": parsed["translation_glossary"],
            "updated_at": os.path.getmtime(path),
        }

    def _scan(self):
        if not self.directory:
            return []
        os.makedirs(self.directory, exist_ok=True)
        records = []
        for filename in os.listdir(self.directory):
            if not filename.endswith(".txt"):
                continue
            path = os.path.join(self.directory, filename)
            if not os.path.isfile(path):
                continue
            meeting_name = os.path.splitext(filename)[0]
            try:
                record = self._record_from_file(meeting_name, path)
            except Exception as exc:
                logging.warning("Failed to load meeting hotwords from %s: %s", path, exc)
                continue
            records.append(record)
        records.sort(key=lambda item: item["meeting_name"])
        return records

    def list(self):
        with self.lock:
            return {"directory": self.directory, "meetings": self._scan()}

    def get(self, meeting_name):
        name, path = self._safe_path(meeting_name)
        with self.lock:
            if not os.path.isfile(path):
                return self._empty_record(name)
            return self._record_from_file(name, path)


class ClientManager:
    def __init__(self, max_clients=4, max_connection_time=600):
        """
        Initializes the ClientManager with specified limits on client connections and connection durations.

        Args:
            max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
            max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                                 to 600 seconds (10 minutes).
        """
        self.clients = {}
        self.start_times = {}
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time
        self.lock = threading.Lock()
        self.client_status = {}

    @staticmethod
    def _latest_segment_text(segments):
        if not segments:
            return ""
        for segment in reversed(segments):
            text = segment.get("text", "") if isinstance(segment, dict) else ""
            if text:
                return text
        return ""

    def register_client_status(self, websocket, client, options, backend):
        now = time.time()
        uid = getattr(client, "client_uid", options.get("uid"))
        status = {
            "uid": uid,
            "client_instance_id": options.get("client_instance_id") or "",
            "client_name": options.get("client_name") or options.get("meeting_name") or f"Client-{str(uid)[:8]}",
            "meeting_name": options.get("meeting_name") or "",
            "hotwords_file": options.get("hotwords_file") or "",
            "hotwords_count": int(options.get("hotwords_count") or count_hotwords(options.get("hotwords"))),
            "hotwords_locked": True,
            "connected": True,
            "connected_at": now,
            "disconnected_at": None,
            "backend": backend.value if isinstance(backend, BackendType) else str(backend),
            "language": options.get("language"),
            "model": options.get("model"),
            "translation_enabled": bool(options.get("enable_translation", False)),
            "target_language": options.get("target_language", "auto"),
            "segment_msgs": 0,
            "segment_items": 0,
            "translation_msgs": 0,
            "translation_items": 0,
            "last_activity_at": now,
            "last_source_text": "",
            "last_translation_text": "",
        }
        with self.lock:
            instance_id = status.get("client_instance_id")
            if instance_id:
                for old_websocket, old_status in list(self.client_status.items()):
                    if old_websocket is websocket:
                        continue
                    if old_status.get("client_instance_id") != instance_id:
                        continue
                    if old_status.get("connected"):
                        continue
                    del self.client_status[old_websocket]
            self.client_status[websocket] = status

    def update_client_message(self, websocket, message_type, segments):
        now = time.time()
        text = self._latest_segment_text(segments)
        with self.lock:
            status = self.client_status.get(websocket)
            if not status:
                return
            if message_type == "segments":
                status["segment_msgs"] += 1
                status["segment_items"] += len(segments or [])
                if text:
                    status["last_source_text"] = text
            elif message_type == "translated_segments":
                status["translation_msgs"] += 1
                status["translation_items"] += len(segments or [])
                if text:
                    status["last_translation_text"] = text
            status["last_activity_at"] = now

    def mark_client_disconnected(self, websocket):
        now = time.time()
        with self.lock:
            status = self.client_status.get(websocket)
            if status:
                status["connected"] = False
                status["disconnected_at"] = now
                status["last_activity_at"] = now

    def get_client_status_snapshot(self):
        now = time.time()
        with self.lock:
            statuses = [dict(status) for status in self.client_status.values()]
        for status in statuses:
            connected_at = status.get("connected_at") or now
            last_activity_at = status.get("last_activity_at") or connected_at
            status["connected_seconds"] = round((status.get("disconnected_at") or now) - connected_at, 3)
            status["last_activity_seconds_ago"] = round(now - last_activity_at, 3)
        statuses.sort(key=lambda item: item.get("connected_at", 0), reverse=True)
        return {"server_time": now, "clients": statuses}

    def delete_disconnected_client_status(self, uid):
        with self.lock:
            for websocket, status in list(self.client_status.items()):
                if status.get("uid") != uid:
                    continue
                if status.get("connected"):
                    return "connected"
                del self.client_status[websocket]
                return "deleted"
        return "not_found"

    def add_client(self, websocket, client):
        """
        Adds a client and their connection start time to the tracking dictionaries.

        Args:
            websocket: The websocket associated with the client to add.
            client: The client object to be added and tracked.
        """
        with self.lock:
            self.clients[websocket] = client
            self.start_times[websocket] = time.time()

    def get_client(self, websocket):
        """
        Retrieves a client associated with the given websocket.

        Args:
            websocket: The websocket associated with the client to retrieve.

        Returns:
            The client object if found, False otherwise.
        """
        with self.lock:
            if websocket in self.clients:
                return self.clients[websocket]
            return False

    def remove_client(self, websocket):
        """
        Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
        client if necessary.

        Args:
            websocket: The websocket associated with the client to be removed.
        """
        with self.lock:
            client = self.clients.pop(websocket, None)
            self.start_times.pop(websocket, None)
        if client:
            client.cleanup()

    def get_wait_time(self):
        """
        Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

        Returns:
            The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
        """
        with self.lock:
            wait_time = None
            for start_time in self.start_times.values():
                current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
                if wait_time is None or current_client_time_remaining < wait_time:
                    wait_time = current_client_time_remaining
        return wait_time / 60 if wait_time is not None else 0

    def is_server_full(self, websocket, options):
        """
        Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

        Args:
            websocket: The websocket of the client attempting to connect.
            options: A dictionary of options that may include the client's unique identifier.

        Returns:
            True if the server is full, False otherwise.
        """
        with self.lock:
            if len(self.clients) >= self.max_clients:
                wait_time = None
                for start_time in self.start_times.values():
                    remaining = self.max_connection_time - (time.time() - start_time)
                    if wait_time is None or remaining < wait_time:
                        wait_time = remaining
                wait_time_minutes = wait_time / 60 if wait_time is not None else 0
                response = {"uid": options["uid"], "status": "WAIT", "message": wait_time_minutes}
                websocket.send(json.dumps(response))
                return True
            return False

    def is_client_timeout(self, websocket):
        """
        Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

        Args:
            websocket: The websocket associated with the client to check.

        Returns:
            True if the client's connection time has exceeded the maximum limit, False otherwise.
        """
        with self.lock:
            elapsed_time = time.time() - self.start_times[websocket]
            client = self.clients.get(websocket)
        if elapsed_time >= self.max_connection_time and client:
            client.disconnect()
            logging.warning(f"Client with uid '{client.client_uid}' disconnected due to overtime.")
            return True
        return False


class BackendType(Enum):
    FASTER_WHISPER = "faster_whisper"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    MLX_WHISPER = "mlx_whisper"
    FUNASR = "funasr"

    @staticmethod
    def valid_types() -> List[str]:
        return [backend_type.value for backend_type in BackendType]

    @staticmethod
    def is_valid(backend: str) -> bool:
        return backend in BackendType.valid_types()

    def is_faster_whisper(self) -> bool:
        return self == BackendType.FASTER_WHISPER

    def is_tensorrt(self) -> bool:
        return self == BackendType.TENSORRT
    
    def is_openvino(self) -> bool:
        return self == BackendType.OPENVINO

    def is_mlx_whisper(self) -> bool:
        return self == BackendType.MLX_WHISPER

    def is_funasr(self) -> bool:
        return self == BackendType.FUNASR


class TranscriptionServer:
    RATE = 16000
    LOCAL_ASR_MODEL_ROOT = "model/asr"
    LOCAL_ASR_MODEL_NAMES = {
        "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large-v3-turbo", "large-v3",
    }

    def __init__(self):
        self.client_manager = None
        self.no_voice_activity_chunks = 0
        self.use_vad = True
        self.single_model = False
        self.batch_config = None
        self.raw_pcm_input = False
        self.segment_post_processor = None
        self.default_hotwords = None
        self.translation_device = "cpu"
        self.meeting_hotwords = MeetingHotwordStore()
        self.meeting_logs = MeetingLogStore()
        self.meeting_summary = MeetingSummaryService()

    @staticmethod
    def load_hotwords_file(path):
        if not path:
            return None
        if not os.path.isfile(path):
            logging.warning(f"Hotwords file not found: {path}")
            return None

        hotwords = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip()
                if not word or word.startswith("#"):
                    continue
                hotwords.append(word)

        if not hotwords:
            logging.info(f"Hotwords file is empty: {path}")
            return None

        return " ".join(hotwords)

    def apply_meeting_hotwords(self, options):
        meeting_name = options.get("meeting_name")
        if not meeting_name or not self.meeting_hotwords:
            return
        stored = self.meeting_hotwords.get(meeting_name)
        prompt = hotword_text_to_prompt(stored.get("text"))
        if prompt and not options.get("hotwords"):
            options["hotwords"] = prompt
        if prompt or stored.get("translation_glossary"):
            options["hotwords_count"] = stored.get("count") or count_hotwords(stored.get("text"))
            options["hotwords_file"] = stored.get("filename") or ""
            options["hotwords_locked"] = True
            options["translation_glossary"] = dict(stored.get("translation_glossary") or {})
            options["translation_glossary_count"] = int(stored.get("translation_count") or 0)

    def apply_default_hotwords(self, options):
        if options.get("hotwords"):
            return
        if self.default_hotwords:
            options["hotwords"] = self.default_hotwords

    def get_admin_clients_payload(self):
        if not self.client_manager:
            return {"server_time": time.time(), "clients": []}
        return self.client_manager.get_client_status_snapshot()

    def handle_client_segments(self, websocket, message_type, segments):
        if self.client_manager:
            self.client_manager.update_client_message(websocket, message_type, segments)
        client = self.client_manager.get_client(websocket) if self.client_manager else None
        session_id = getattr(client, "meeting_log_session_id", None)
        kind = "translation" if message_type == "translated_segments" else "source"
        try:
            self.meeting_logs.append_segments(session_id, kind, segments)
        except Exception as exc:
            logging.error("Failed to append meeting log segments: %s", exc)

    def finalize_client_meeting_log(self, websocket):
        client = self.client_manager.get_client(websocket) if self.client_manager else None
        session_id = getattr(client, "meeting_log_session_id", None)
        try:
            return self.meeting_logs.finish_session(session_id)
        except Exception as exc:
            logging.error("Failed to finalize meeting log: %s", exc)
            return None

    def generate_meeting_summary(self, session_id, template="auto"):
        info = self.meeting_logs.session_info(session_id)
        if not info:
            raise KeyError("meeting log session not found")
        if info.get("status") != "finished":
            raise RuntimeError("请停止会议后再生成总结")
        template = self.meeting_summary.validate_template(template)
        payload = self.meeting_logs.get_session_payload(session_id)
        summary = self.meeting_summary.generate(payload, template=template)
        result = self.meeting_logs.write_summary(session_id, summary)
        return {"generated": True, "summary": result, "data": summary}

    def _default_cors_origins(self, websocket_port):
        return [
            f"http://localhost:{websocket_port}",
            f"http://127.0.0.1:{websocket_port}",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

    def resolve_asr_model_path(self, model):
        if model in self.LOCAL_ASR_MODEL_NAMES:
            local_model = os.path.join(self.LOCAL_ASR_MODEL_ROOT, model)
            if os.path.isdir(local_model):
                return local_model
        return model

    def resolve_funasr_model_path(self, client_model, server_model):
        server_model = server_model or "iic/SenseVoiceSmall"
        client_model = str(client_model or "").strip()
        if not client_model:
            return server_model
        if client_model in self.LOCAL_ASR_MODEL_NAMES or client_model.startswith("model/asr/"):
            return server_model
        if client_model == "iic/SenseVoiceSmall" and server_model != "iic/SenseVoiceSmall":
            return server_model
        if (
            client_model.startswith("model/funasr/")
            or client_model.startswith("/")
            or client_model.startswith("~")
            or "/" in client_model
        ):
            return client_model
        return server_model

    def initialize_client(
        self, websocket, options, faster_whisper_custom_model_path,
        whisper_tensorrt_path, trt_multilingual, trt_py_session=False,
        funasr_model=None, funasr_device="auto",
        funasr_mode="sensevoice", funasr_punc_model=None, funasr_vad_model=None,
        funasr_final_model="model/funasr/SenseVoiceSmall", funasr_final_device=None,
        funasr_final_refine=True,
    ):
        client: Optional[ServeClientBase] = None

        # Check if client wants translation
        enable_translation = options.get("enable_translation", False)
        
        # Create translation queue if translation is enabled
        translation_queue = None
        translation_client = None
        translation_thread = None
        
        if enable_translation:
            target_language = options.get("target_language", "auto")
            translation_device = options.get("translation_device", self.translation_device)
            translation_queue = queue.Queue(maxsize=ServeClientBase.MAX_TRANSLATION_QUEUE_SIZE)
            from whisper_live.backend.translation_backend import ServeClientTranslation
            translation_client = ServeClientTranslation(
                client_uid=options["uid"],
                websocket=websocket,
                translation_queue=translation_queue,
                target_language=target_language,
                send_last_n_segments=options.get("send_last_n_segments", 10),
                model_name=options.get("translation_provider", "helsinki_zh_en"),
                zh_en_model_path=options.get("zh_en_model_path", "model/opus-mt-zh-en"),
                translation_device=translation_device,
                en_zh_model_path=options.get("en_zh_model_path", "model/opus-mt-en-zh"),
                translation_glossary=options.get("translation_glossary"),
            )
            
            # Start translation thread
            translation_thread = threading.Thread(
                target=translation_client.speech_to_text,
                daemon=True
            )
            translation_thread.start()
            
            logging.info(f"Translation enabled for client {options['uid']} with target language: {target_language}")

        if self.backend.is_tensorrt():
            try:
                from whisper_live.backend.trt_backend import ServeClientTensorRT
                client = ServeClientTensorRT(
                    websocket,
                    multilingual=trt_multilingual,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=whisper_tensorrt_path,
                    single_model=self.single_model,
                    use_py_session=trt_py_session,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    translation_queue=translation_queue,
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                )
                logging.info("Running TensorRT backend.")
            except Exception as e:
                logging.error(f"TensorRT-LLM not supported: {e}")
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "TensorRT-LLM not supported on Server yet. "
                               "Reverting to available backend: 'faster_whisper'"
                }))
                self.backend = BackendType.FASTER_WHISPER
        
        if self.backend.is_openvino():
            try:
                from whisper_live.backend.openvino_backend import ServeClientOpenVINO
                client = ServeClientOpenVINO(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                )
                logging.info("Running OpenVINO backend.")
            except Exception as e:
                logging.error(f"OpenVINO not supported: {e}")
                self.backend = BackendType.FASTER_WHISPER
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "WARNING",
                    "message": "OpenVINO not supported on Server yet. "
                                "Reverting to available backend: 'faster_whisper'"
                }))

        if self.backend.is_mlx_whisper():
            try:
                from whisper_live.backend.mlx_whisper_backend import ServeClientMLXWhisper
                client = ServeClientMLXWhisper(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    initial_prompt=options.get("initial_prompt"),
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    translation_queue=translation_queue,
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                )
                logging.info("Running MLX Whisper backend.")
            except Exception as e:
                logging.error(f"MLX Whisper not supported: {e}")
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "ERROR",
                    "message": str(e)
                }))
                websocket.close()
                raise

        if self.backend.is_funasr():
            try:
                from whisper_live.backend.funasr_backend import ServeClientFunASR
                if funasr_mode == "paraformer_streaming":
                    options["model"] = funasr_model or "model/funasr/paraformer-zh-streaming"
                else:
                    options["model"] = self.resolve_funasr_model_path(options.get("model"), funasr_model)
                client = ServeClientFunASR(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    device=funasr_device,
                    mode=funasr_mode,
                    punc_model=funasr_punc_model,
                    vad_model=funasr_vad_model,
                    final_model=funasr_final_model,
                    final_device=funasr_final_device,
                    final_refine=funasr_final_refine,
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 3),
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                    max_incomplete_segment_seconds=options.get("max_incomplete_segment_seconds", 6.0),
                    use_vad=self.use_vad,
                    translation_queue=translation_queue,
                    hotwords=options.get("hotwords"),
                )
                logging.info("Running FunASR backend.")
            except Exception as e:
                logging.error(f"FunASR not supported: {e}")
                self.client_uid = options["uid"]
                websocket.send(json.dumps({
                    "uid": self.client_uid,
                    "status": "ERROR",
                    "message": str(e)
                }))
                websocket.close()
                raise

        try:
            if self.backend.is_faster_whisper():
                from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
                # model is of the form namespace/repo_name and not a filesystem path
                if faster_whisper_custom_model_path is not None:
                    logging.info(f"Using custom model {faster_whisper_custom_model_path}")
                    options["model"] = faster_whisper_custom_model_path
                else:
                    options["model"] = self.resolve_asr_model_path(options["model"])
                client = ServeClientFasterWhisper(
                    websocket,
                    language=options["language"],
                    task=options["task"],
                    client_uid=options["uid"],
                    model=options["model"],
                    initial_prompt=options.get("initial_prompt"),
                    vad_parameters=options.get("vad_parameters"),
                    use_vad=self.use_vad,
                    single_model=self.single_model,
                    send_last_n_segments=options.get("send_last_n_segments", 10),
                    no_speech_thresh=options.get("no_speech_thresh", 0.45),
                    clip_audio=options.get("clip_audio", False),
                    same_output_threshold=options.get("same_output_threshold", 10),
                    min_segment_rms=options.get("min_segment_rms", 0.0015),
                    max_incomplete_segment_seconds=options.get("max_incomplete_segment_seconds", 0.0),
                    cache_path=self.cache_path,
                    translation_queue=translation_queue,
                    hotwords=options.get("hotwords"),
                    diarization=self._create_diarizer(options),
                    word_timestamps=options.get("word_timestamps", False),
                )

                logging.info("Running faster_whisper backend.")

                # Start batch inference worker on first client (after model is loaded)
                if (self.batch_config is not None
                        and ServeClientFasterWhisper.BATCH_WORKER is None
                        and ServeClientFasterWhisper.SINGLE_MODEL is not None):
                    from whisper_live.batch_inference import BatchInferenceWorker
                    worker = BatchInferenceWorker(
                        transcriber=ServeClientFasterWhisper.SINGLE_MODEL,
                        **self.batch_config,
                    )
                    worker.start()
                    ServeClientFasterWhisper.BATCH_WORKER = worker
        except Exception as e:
            logging.error(e)
            return

        if client is None:
            raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")

        # Attach segment post-processor if configured
        if self.segment_post_processor is not None:
            client.segment_post_processor = self.segment_post_processor

        if translation_client:
            client.translation_client = translation_client
            client.translation_thread = translation_thread

        if translation_client:
            translation_client.admin_status_callback = functools.partial(
                self.handle_client_segments, websocket, "translated_segments"
            )
        client.admin_status_callback = functools.partial(
            self.handle_client_segments, websocket, "segments"
        )
        try:
            log_info = self.meeting_logs.start_session(options, backend=self.backend)
            client.meeting_log_session_id = log_info.get("session_id") if log_info else options.get("session_id") or options.get("uid")
        except Exception as exc:
            logging.error("Failed to start meeting log session: %s", exc)
            client.meeting_log_session_id = options.get("session_id") or options.get("uid")
        self.client_manager.add_client(websocket, client)
        self.client_manager.register_client_status(websocket, client, options, self.backend)

    def _create_diarizer(self, options):
        """Create a SpeakerDiarizer if the client requested diarization.

        Returns:
            SpeakerDiarizer or None
        """
        if not options.get("enable_diarization", False):
            return None
        try:
            from whisper_live.diarization import SpeakerDiarizer
            return SpeakerDiarizer(
                similarity_threshold=options.get("diarization_threshold", 0.55),
                max_speakers=options.get("max_speakers", 10),
                hf_token=options.get("hf_token"),
            )
        except ImportError:
            logging.warning("pyannote.audio not installed; diarization disabled")
            return None

    def get_audio_from_websocket(self, websocket):
        """
        Receives audio buffer from websocket and creates a numpy array out of it.

        Args:
            websocket: The websocket to receive audio from.

        Returns:
            A numpy array containing the audio.
        """
        frame_data = websocket.recv()
        if frame_data == b"END_OF_AUDIO":
            return False
        if self.raw_pcm_input:
            audio_np = np.frombuffer(frame_data, dtype=np.int16)
            return audio_np.astype(np.float32) / 32768.0
        return np.frombuffer(frame_data, dtype=np.float32)

    def handle_new_connection(self, websocket, faster_whisper_custom_model_path,
                              whisper_tensorrt_path, trt_multilingual, trt_py_session=False,
                              funasr_model=None, funasr_device="auto",
                              funasr_mode="sensevoice", funasr_punc_model=None, funasr_vad_model=None,
                              funasr_final_model="model/funasr/SenseVoiceSmall", funasr_final_device=None,
                              funasr_final_refine=True):
        try:
            logging.info("New client connected")
            options = websocket.recv()
            options = json.loads(options)
            self.apply_meeting_hotwords(options)
            self.apply_default_hotwords(options)

            self.use_vad = options.get('use_vad')
            if self.client_manager.is_server_full(websocket, options):
                wl_metrics.track_connection_rejected(reason="full")
                websocket.close()
                return False  # Indicates that the connection should not continue

            if self.backend.is_tensorrt() and self.use_vad:
                self.vad_detector = VoiceActivityDetector(frame_rate=self.RATE)
            self.initialize_client(websocket, options, faster_whisper_custom_model_path,
                                   whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session,
                                   funasr_model=funasr_model, funasr_device=funasr_device,
                                   funasr_mode=funasr_mode, funasr_punc_model=funasr_punc_model,
                                   funasr_vad_model=funasr_vad_model,
                                   funasr_final_model=funasr_final_model,
                                   funasr_final_device=funasr_final_device,
                                   funasr_final_refine=funasr_final_refine)
            wl_metrics.track_connection_opened()
            return True
        except json.JSONDecodeError:
            logging.error("Failed to decode JSON from client")
            return False
        except ConnectionClosed:
            logging.info("Connection closed by client")
            return False
        except Exception as e:
            logging.error(f"Error during new connection initialization: {str(e)}")
            return False

    def process_audio_frames(self, websocket):
        frame_np = self.get_audio_from_websocket(websocket)
        client = self.client_manager.get_client(websocket)
        if frame_np is False:
            if self.backend.is_tensorrt():
                client.set_eos(True)
            return False

        if self.backend.is_tensorrt():
            voice_active = self.voice_activity(websocket, frame_np)
            if voice_active:
                self.no_voice_activity_chunks = 0
                client.set_eos(False)
            if self.use_vad and not voice_active:
                return True

        client.add_frames(frame_np)
        return True

    def recv_audio(self,
                   websocket,   
                   backend: BackendType = BackendType.FASTER_WHISPER,
                   faster_whisper_custom_model_path=None,
                   whisper_tensorrt_path=None,
                   trt_multilingual=False,
                   trt_py_session=False,
                   funasr_model=None,
                   funasr_device="auto",
                   funasr_mode="sensevoice",
                   funasr_punc_model=None,
                   funasr_vad_model=None,
                   funasr_final_model="model/funasr/SenseVoiceSmall",
                   funasr_final_device=None,
                   funasr_final_refine=True):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client
        over a WebSocket connection. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.
        If the maximum number of clients is reached, the method sends a
        "WAIT" status to the client, indicating that they should wait
        until a slot is available.
        If a client's connection exceeds the maximum allowed time, it will
        be disconnected, and the client's resources will be cleaned up.

        Args:
            websocket (WebSocket): The WebSocket connection for the client.
            backend (str): The backend to run the server with.
            faster_whisper_custom_model_path (str): path to custom faster whisper model.
            whisper_tensorrt_path (str): Required for tensorrt backend.
            trt_multilingual(bool): Only used for tensorrt, True if multilingual model.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        self.backend = backend
        if not self.handle_new_connection(websocket, faster_whisper_custom_model_path,
                                          whisper_tensorrt_path, trt_multilingual, trt_py_session=trt_py_session,
                                          funasr_model=funasr_model, funasr_device=funasr_device,
                                          funasr_mode=funasr_mode, funasr_punc_model=funasr_punc_model,
                                          funasr_vad_model=funasr_vad_model,
                                          funasr_final_model=funasr_final_model,
                                          funasr_final_device=funasr_final_device,
                                          funasr_final_refine=funasr_final_refine):
            return

        try:
            while not self.client_manager.is_client_timeout(websocket):
                if not self.process_audio_frames(websocket):
                    break
        except ConnectionClosed:
            logging.info("Connection closed by client")
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
        finally:
            if self.client_manager.get_client(websocket):
                self.cleanup(websocket)
                websocket.close()
            wl_metrics.track_connection_closed()
            del websocket

    def run(self,
            host,
            port=9090,
            backend="tensorrt",
            faster_whisper_custom_model_path=None,
            whisper_tensorrt_path=None,
            funasr_model=None,
            funasr_mode="sensevoice",
            funasr_punc_model=None,
            funasr_vad_model=None,
            funasr_final_model="model/funasr/SenseVoiceSmall",
            funasr_final_device=None,
            funasr_final_refine=True,
            funasr_device="auto",
            trt_multilingual=False,
            trt_py_session=False,
            single_model=False,
            max_clients=4,
            max_connection_time=600,
            cache_path="~/.cache/whisper-live/",
            rest_port=8000,
            enable_rest=False,
            cors_origins: Optional[str] = None,
            batch_enabled=False,
            batch_max_size=8,
            batch_window_ms=50,
            raw_pcm_input=False,
            metrics_port: int = 0,
            hotwords_file=None,
            translation_device="cpu",
            meeting_hotwords_dir="config/hotwords.d",
            meeting_logs_dir="logs",
            summary_base_url="http://127.0.0.1:8001/v1",
            summary_model="qwen3-8b-awq",
            summary_startup_command="bash scripts/start_summary_llm_service.sh",
            summary_timeout=600,
            summary_ready_timeout=300,
            summary_max_chars_per_chunk=4000,
            summary_idle_shutdown_seconds=600,
            segment_post_processor=None):
        """
        Run the transcription server.

        Args:
            host (str): The host address to bind the server.
            port (int): The port number to bind the server.
            batch_enabled (bool): Enable cross-client GPU batch inference for
                the faster_whisper backend. When enabled, ``single_model`` is
                forced to True and a ``BatchInferenceWorker`` is started after
                the first client connects. Defaults to False.
            batch_max_size (int): Maximum number of requests per GPU batch.
                Defaults to 8.
            batch_window_ms (int): Maximum time in milliseconds to wait for
                the batch to fill after the first request arrives. Defaults
                to 50.
            segment_post_processor (callable, optional): A callable that receives
                a transcription segment dict and returns a modified segment dict.
                Applied to every segment before sending to the client. Useful for
                plugging in custom post-processing (e.g. formatting, redaction).
                Defaults to None.
        """
        self.cache_path = cache_path
        self.raw_pcm_input = raw_pcm_input
        self.translation_device = translation_device
        self.meeting_hotwords = MeetingHotwordStore(meeting_hotwords_dir)
        self.meeting_logs = MeetingLogStore(meeting_logs_dir)
        self.meeting_summary = MeetingSummaryService(
            base_url=summary_base_url, model=summary_model, startup_command=summary_startup_command,
            timeout=summary_timeout, ready_timeout=summary_ready_timeout,
            max_chars_per_chunk=summary_max_chars_per_chunk, idle_shutdown_seconds=summary_idle_shutdown_seconds,
        )
        self.default_hotwords = self.load_hotwords_file(hotwords_file)
        if self.default_hotwords:
            logging.info(
                "Loaded %d default hotword tokens from %s",
                len(self.default_hotwords.split()),
                hotwords_file,
            )

        if max_clients < 1:
            raise ValueError(f"max_clients must be >= 1, got {max_clients}")
        if max_connection_time <= 0:
            raise ValueError(f"max_connection_time must be > 0, got {max_connection_time}")
        if batch_enabled and batch_max_size < 1:
            raise ValueError(f"batch_max_size must be >= 1, got {batch_max_size}")
        if batch_enabled and batch_window_ms < 0:
            raise ValueError(f"batch_window_ms must be >= 0, got {batch_window_ms}")

        self.segment_post_processor = segment_post_processor
        self.client_manager = ClientManager(max_clients, max_connection_time)
        if faster_whisper_custom_model_path is not None and not os.path.exists(faster_whisper_custom_model_path):
            if "/" not in faster_whisper_custom_model_path:
                raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path or HuggingFace model.")
        if whisper_tensorrt_path is not None and not os.path.exists(whisper_tensorrt_path):
            raise ValueError(f"TensorRT model '{whisper_tensorrt_path}' is not a valid path.")

        # Batch inference config
        if batch_enabled:
            single_model = True  # Batch mode requires shared model
            self.batch_config = {
                'max_batch_size': batch_max_size,
                'batch_window_ms': batch_window_ms,
            }
            logging.info(f"Batch inference enabled (max_batch={batch_max_size}, window={batch_window_ms}ms)")
        else:
            self.batch_config = None

        if single_model:
            if faster_whisper_custom_model_path or whisper_tensorrt_path or backend == BackendType.FUNASR.value:
                logging.info("Custom model option was provided. Switching to single model mode.")
                self.single_model = True
                # TODO: load model initially
            else:
                logging.info("Single model mode currently only works with custom models.")
        if not BackendType.is_valid(backend):
            raise ValueError(f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}")

        # Start Prometheus metrics endpoint if port is specified
        if metrics_port > 0:
            wl_metrics.start_metrics_server(metrics_port)

        # Admin status API is always available on rest_port. The OpenAI-compatible
        # REST API is added to the same app when enable_rest is true.
        app = FastAPI(title="WhisperLive Admin API")
        origins = [o.strip() for o in cors_origins.split(',')] if cors_origins else self._default_cors_origins(port)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/admin/clients")
        async def admin_clients():
            return self.get_admin_clients_payload()

        @app.delete("/admin/clients/{uid}")
        async def delete_admin_client(uid: str):
            result = self.client_manager.delete_disconnected_client_status(uid)
            if result == "deleted":
                return {"deleted": True, "uid": uid}
            if result == "connected":
                return JSONResponse(
                    status_code=409,
                    content={"deleted": False, "uid": uid, "error": "client is still connected"},
                )
            return JSONResponse(
                status_code=404,
                content={"deleted": False, "uid": uid, "error": "client not found"},
            )

        @app.get("/admin/hotwords")
        async def list_admin_hotwords():
            return self.meeting_hotwords.list()

        @app.get("/admin/hotwords/{meeting_name}")
        async def get_admin_hotwords(meeting_name: str):
            try:
                return self.meeting_hotwords.get(meeting_name)
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

        @app.post("/admin/meeting-logs")
        async def save_admin_meeting_log(request: Request):
            try:
                payload = await request.json()
                return self.meeting_logs.save(payload)
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"saved": False, "error": "request body must be valid JSON"})
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"saved": False, "error": str(exc)})
            except Exception as exc:
                logging.error("Failed to save meeting log: %s", exc)
                return JSONResponse(status_code=500, content={"saved": False, "error": str(exc)})

        @app.get("/admin/meeting-logs")
        async def list_admin_meeting_logs():
            return self.meeting_logs.list_sessions()

        @app.get("/admin/meeting-logs/{session_id}")
        async def download_admin_meeting_log(session_id: str, format: str = "md"):
            result = self.meeting_logs.get_session_file(session_id, "json" if format.lower() == "json" else "md")
            if not result or not os.path.isfile(result[0]):
                return JSONResponse(status_code=404, content={"error": "meeting log not found"})
            return FileResponse(result[0], media_type=result[1], filename=result[2])

        @app.get("/admin/meeting-logs/{session_id}/info")
        async def get_admin_meeting_log_info(session_id: str):
            info = self.meeting_logs.session_info(session_id)
            return info if info else JSONResponse(status_code=404, content={"error": "meeting log session not found"})

        @app.post("/admin/meeting-logs/{session_id}/summary")
        async def generate_admin_meeting_summary(session_id: str, request: Request):
            try:
                raw_body = await request.body()
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                if not isinstance(body, dict):
                    raise ValueError("request body must be a JSON object")
                return await asyncio.to_thread(
                    self.generate_meeting_summary,
                    session_id,
                    body.get("template") or "auto",
                )
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"generated": False, "error": "request body must be valid JSON"})
            except KeyError as exc:
                return JSONResponse(status_code=404, content={"generated": False, "error": str(exc)})
            except SummaryGenerationError as exc:
                logging.error("Meeting summary generation failed: %s", exc.code)
                return JSONResponse(
                    status_code=502,
                    content={"generated": False, "error": str(exc), "error_code": exc.code},
                )
            except RuntimeError as exc:
                return JSONResponse(status_code=409, content={"generated": False, "error": str(exc)})
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"generated": False, "error": str(exc)})
            except Exception as exc:
                logging.error("Failed to generate meeting summary: %s", exc)
                return JSONResponse(status_code=500, content={"generated": False, "error": str(exc)})

        @app.get("/admin/meeting-logs/{session_id}/summary")
        async def download_admin_meeting_summary(session_id: str, format: str = "md", version: Optional[int] = None):
            result = self.meeting_logs.get_summary_file(
                session_id,
                "json" if format.lower() == "json" else "md",
                version=version,
            )
            if not result or not os.path.isfile(result[0]):
                return JSONResponse(status_code=404, content={"error": "meeting summary not found"})
            return FileResponse(result[0], media_type=result[1], filename=result[2])

        @app.get("/admin/meeting-logs/{session_id}/summary/info")
        async def get_admin_meeting_summary_info(session_id: str):
            info = self.meeting_logs.summary_info(session_id)
            return info if info else JSONResponse(status_code=404, content={"error": "meeting log session not found"})

        if enable_rest:
            @app.post("/v1/audio/transcriptions")
            async def transcribe(
                file: UploadFile,
                model: str = Form(default="whisper-1"),
                language: Optional[str] = Form(default=None),
                prompt: Optional[str] = Form(default=None),
                response_format: str = Form(default="json"),
                temperature: float = Form(default=0.0),
                timestamp_granularities: Optional[List[str]] = Form(default=None),
                # Stubs for unsupported OpenAI params
                chunking_strategy: Optional[str] = Form(default=None),
                include: Optional[List[str]] = Form(default=None),
                known_speaker_names: Optional[List[str]] = Form(default=None),
                known_speaker_references: Optional[List[str]] = Form(default=None),
                stream: bool = Form(default=False),
                hotwords: Optional[str] = Form(default=None),
            ):
                if stream:
                    wl_metrics.track_rest_request(endpoint="transcriptions", status=400)
                    return JSONResponse({"error": "Streaming not supported in this backend."}, status_code=400)
                if chunking_strategy or known_speaker_names or known_speaker_references:
                    logging.warning("Diarization/chunking params ignored; not supported.")

                supported_formats = ["json", "text", "srt", "verbose_json", "vtt"]
                if response_format not in supported_formats:
                    wl_metrics.track_rest_request(endpoint="transcriptions", status=400)
                    return JSONResponse({"error": f"Unsupported response_format. Supported: {supported_formats}"}, status_code=400)

                if model != "whisper-1":
                    logging.warning(f"Model '{model}' requested; using 'small' as fallback.")
                model_name = faster_whisper_custom_model_path or self.resolve_asr_model_path("small")

                try:
                    suffix = os.path.splitext(file.filename)[1] or ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        tmp_path = tmp.name

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    compute_type = "float16" if device == "cuda" else "int8"

                    transcriber = WhisperModel(model_name, device=device, compute_type=compute_type)
                    segments, info = transcriber.transcribe(
                        tmp_path,
                        language=language,
                        initial_prompt=prompt,
                        temperature=temperature,
                        vad_filter=False,
                        word_timestamps=(timestamp_granularities and "word" in timestamp_granularities),
                        hotwords=hotwords,
                    )

                    text = " ".join([s.text.strip() for s in segments])
                    os.unlink(tmp_path)

                    if response_format == "text":
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return PlainTextResponse(text)
                    elif response_format == "json":
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return {"text": text}
                    elif response_format == "verbose_json":
                        verbose = {
                            "task": "transcribe",
                            "language": info.language,
                            "duration": info.duration,
                            "text": text,
                            "segments": []
                        }
                        for seg in segments:
                            seg_dict = {
                                "id": seg.id,
                                "seek": seg.seek,
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text.strip(),
                                "tokens": seg.tokens,
                                "temperature": seg.temperature,
                                "avg_logprob": seg.avg_logprob,
                                "compression_ratio": seg.compression_ratio,
                                "no_speech_prob": seg.no_speech_prob
                            }
                            if timestamp_granularities and "word" in timestamp_granularities:
                                seg_dict["words"] = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words]
                            verbose["segments"].append(seg_dict)
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return verbose
                    elif response_format in ["srt", "vtt"]:
                        output = []
                        for i, seg in enumerate(segments, 1):
                            start = f"{int(seg.start // 3600):02}:{int((seg.start % 3600) // 60):02}:{seg.start % 60:06.3f}"
                            end = f"{int(seg.end // 3600):02}:{int((seg.end % 3600) // 60):02}:{seg.end % 60:06.3f}"
                            if response_format == "srt":
                                output.append(f"{i}\n{start.replace('.', ',')} --> {end.replace('.', ',')}\n{seg.text.strip()}\n")
                            else:  # vtt
                                output.append(f"{start} --> {end}\n{seg.text.strip()}\n")
                        wl_metrics.track_rest_request(endpoint="transcriptions", status=200)
                        return PlainTextResponse("\n".join(output))
                except Exception as e:
                    wl_metrics.track_rest_request(endpoint="transcriptions", status=500)
                    wl_metrics.track_error("rest_transcription")
                    return JSONResponse({"error": str(e)}, status_code=500)

        threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": rest_port, "log_level": "info"},
            daemon=True
        ).start()
        if enable_rest:
            logging.info(f"OpenAI-compatible REST API started on http://0.0.0.0:{rest_port}")
        logging.info(f"Admin API available at http://0.0.0.0:{rest_port}/admin/clients")

        # Original WebSocket server (always supported)
        with serve(
            functools.partial(
                self.recv_audio,
                backend=BackendType(backend),
                faster_whisper_custom_model_path=faster_whisper_custom_model_path,
                whisper_tensorrt_path=whisper_tensorrt_path,
                funasr_model=funasr_model,
                funasr_mode=funasr_mode,
                funasr_punc_model=funasr_punc_model,
                funasr_vad_model=funasr_vad_model,
                funasr_final_model=funasr_final_model,
                funasr_final_device=funasr_final_device,
                funasr_final_refine=funasr_final_refine,
                funasr_device=funasr_device,
                trt_multilingual=trt_multilingual,
                trt_py_session=trt_py_session,
            ),
            host,
            port
        ) as server:
            server.serve_forever()

    def voice_activity(self, websocket, frame_np):
        """
        Evaluates the voice activity in a given audio frame and manages the state of voice activity detection.

        This method uses the configured voice activity detection (VAD) model to assess whether the given audio frame
        contains speech. If the VAD model detects no voice activity for more than three consecutive frames,
        it sets an end-of-speech (EOS) flag for the associated client. This method aims to efficiently manage
        speech detection to improve subsequent processing steps.

        Args:
            websocket: The websocket associated with the current client. Used to retrieve the client object
                    from the client manager for state management.
            frame_np (numpy.ndarray): The audio frame to be analyzed. This should be a NumPy array containing
                                    the audio data for the current frame.

        Returns:
            bool: True if voice activity is detected in the current frame, False otherwise. When returning False
                after detecting no voice activity for more than three consecutive frames, it also triggers the
                end-of-speech (EOS) flag for the client.
        """
        if not self.vad_detector(frame_np):
            self.no_voice_activity_chunks += 1
            if self.no_voice_activity_chunks > 3:
                client = self.client_manager.get_client(websocket)
                if not client.eos:
                    client.set_eos(True)
                time.sleep(0.1)    # Sleep 100m; wait some voice activity.
            return False
        return True

    def cleanup(self, websocket):
        """
        Cleans up resources associated with a given client's websocket.

        Args:
            websocket: The websocket associated with the client to be cleaned up.
        """
        client = self.client_manager.get_client(websocket)
        if client:
            if hasattr(client, 'translation_client') and client.translation_client:
                client.translation_client.cleanup()
                
            # Wait for translation thread to finish
            if hasattr(client, 'translation_thread') and client.translation_thread:
                client.translation_thread.join(timeout=2.0)
            self.finalize_client_meeting_log(websocket)
            self.client_manager.mark_client_disconnected(websocket)
            self.client_manager.remove_client(websocket)
