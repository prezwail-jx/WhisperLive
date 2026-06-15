import json
import logging
import os
import re
import threading
import time
import uuid

from .common import atomic_write, now_iso


class SummaryTemplateStore:
    MAX_FILE_BYTES = 2 * 1024 * 1024
    MAX_FIELDS = 30
    MAX_TABLE_COLUMNS = 10
    DRAFT_TTL_SECONDS = 24 * 60 * 60
    FIELD_TYPES = {"text", "list", "evidence_list", "table"}
    SAFE_ID_PATTERN = re.compile(r"[^a-z0-9_-]+")

    def __init__(self, directory="config/summary_templates"):
        self.directory = directory or "config/summary_templates"
        self.lock = threading.Lock()
        self.drafts = {}
        os.makedirs(self.directory, exist_ok=True)

    @classmethod
    def _safe_id(cls, value):
        value = cls.SAFE_ID_PATTERN.sub("-", str(value or "").strip().lower()).strip("-_")
        return value[:64] or "summary-template"

    @staticmethod
    def _field_key(value, fallback):
        value = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip()).strip("_").lower()
        if not value or value[0].isdigit():
            value = fallback
        return value[:64]

    @staticmethod
    def _extract_sections(markdown):
        sections = []
        for index, line in enumerate(str(markdown or "").splitlines()):
            match = re.match(r"^(#{2,6})\s+(.+?)\s*$", line)
            if match:
                sections.append({"heading": match.group(2).strip(), "level": len(match.group(1)), "line_index": index})
        return sections

    @staticmethod
    def _fallback_fields(sections):
        fields, used = [], set()
        for index, section in enumerate(sections):
            if len(fields) >= SummaryTemplateStore.MAX_FIELDS:
                break
            base = SummaryTemplateStore._field_key(section["heading"], f"field_{index + 1}")
            key, suffix = base, 2
            while key in used:
                key = f"{base}_{suffix}"
                suffix += 1
            used.add(key)
            fields.append({
                "key": key,
                "label": section["heading"],
                "type": "list" if any(word in section["heading"] for word in ("事项", "要点", "问题", "风险", "结论", "议题")) else "text",
                "description": f"根据会议原文填写‘{section['heading']}’",
                "heading": section["heading"],
                "columns": [],
            })
        return fields

    @staticmethod
    def _sanitize_fields(fields, sections):
        headings = {section["heading"] for section in sections}
        cleaned, used, used_headings = [], set(), set()
        for index, field in enumerate(fields or []):
            if not isinstance(field, dict) or len(cleaned) >= SummaryTemplateStore.MAX_FIELDS:
                continue
            heading = str(field.get("heading") or field.get("label") or "").strip()
            if heading not in headings or heading in used_headings:
                continue
            used_headings.add(heading)
            key = SummaryTemplateStore._field_key(field.get("key"), f"field_{index + 1}")
            if key in used:
                continue
            used.add(key)
            field_type = str(field.get("type") or "text").strip()
            if field_type not in SummaryTemplateStore.FIELD_TYPES:
                field_type = "text"
            columns = []
            if field_type == "table":
                for column in field.get("columns") or []:
                    column = str(column or "").strip()
                    if column and column not in columns:
                        columns.append(column[:50])
                    if len(columns) >= SummaryTemplateStore.MAX_TABLE_COLUMNS:
                        break
                if not columns:
                    columns = ["内容"]
            cleaned.append({
                "key": key,
                "label": str(field.get("label") or heading).strip()[:100],
                "type": field_type,
                "description": str(field.get("description") or f"根据会议原文填写‘{heading}’").strip()[:300],
                "heading": heading,
                "columns": columns,
            })
        return cleaned

    @staticmethod
    def _sanitize_markdown(markdown, fields):
        replacements = {str(field.get("heading") or ""): "{{" + str(field.get("key") or "") + "}}" for field in fields}
        lines = str(markdown or "").splitlines()
        output, index = [], 0
        while index < len(lines):
            match = re.match(r"^(#{2,6})\s+(.+?)\s*$", lines[index])
            if not match:
                output.append(lines[index])
                index += 1
                continue
            heading = match.group(2).strip()
            output.append(lines[index])
            if heading in replacements:
                output.extend(["", replacements[heading]])
            index += 1
            while index < len(lines):
                next_heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", lines[index])
                if next_heading:
                    break
                index += 1
        return "\n".join(output).rstrip() + "\n"

    def _purge_drafts(self):
        cutoff = time.time() - self.DRAFT_TTL_SECONDS
        self.drafts = {key: value for key, value in self.drafts.items() if value.get("created_at", 0) >= cutoff}

    def create_draft(self, filename, markdown, fields=None):
        sections = self._extract_sections(markdown)
        if not sections:
            raise ValueError("Markdown 模板至少需要一个二级或更低级标题")
        fields = self._sanitize_fields(fields, sections) or self._fallback_fields(sections)
        draft_id = str(uuid.uuid4())
        draft = {"draft_id": draft_id, "filename": os.path.basename(filename or "template.md"), "markdown": markdown, "sections": sections, "fields": fields, "created_at": time.time()}
        with self.lock:
            self._purge_drafts()
            self.drafts[draft_id] = draft
        return {key: value for key, value in draft.items() if key != "markdown"}

    def confirm(self, draft_id, name, fields):
        with self.lock:
            self._purge_drafts()
            draft = self.drafts.get(draft_id)
            if not draft:
                raise KeyError("template analysis draft not found or expired")
            cleaned = self._sanitize_fields(fields, draft["sections"])
            if not cleaned:
                raise ValueError("模板至少需要一个有效字段")
            base_id = self._safe_id(name or os.path.splitext(draft["filename"])[0])
            template_id, suffix = base_id, 2
            while os.path.exists(os.path.join(self.directory, template_id)):
                template_id = f"{base_id}-{suffix}"
                suffix += 1
            template_dir = os.path.join(self.directory, template_id)
            os.makedirs(template_dir, exist_ok=False)
            definition = {"id": template_id, "name": str(name or template_id).strip()[:100], "format": "md", "revision": 1, "created_at": now_iso(), "fields": cleaned}
            sanitized_markdown = self._sanitize_markdown(draft["markdown"], cleaned)
            atomic_write(os.path.join(template_dir, "template.md"), sanitized_markdown)
            atomic_write(os.path.join(template_dir, "definition.json"), json.dumps(definition, ensure_ascii=False, indent=2) + "\n")
            self.drafts.pop(draft_id, None)
        return definition

    def get(self, template_id):
        if not str(template_id or "").strip():
            return None
        template_id = self._safe_id(template_id)
        definition_path = os.path.join(self.directory, template_id, "definition.json")
        template_path = os.path.join(self.directory, template_id, "template.md")
        if not os.path.isfile(definition_path) or not os.path.isfile(template_path):
            return None
        try:
            with open(definition_path, "r", encoding="utf-8") as file:
                definition = json.load(file)
            with open(template_path, "r", encoding="utf-8") as file:
                definition["markdown"] = file.read()
        except (OSError, ValueError) as exc:
            logging.warning("Failed to load summary template %s: %s", template_id, exc)
            return None
        return definition

    def list(self):
        templates = []
        if os.path.isdir(self.directory):
            for template_id in sorted(os.listdir(self.directory)):
                definition = self.get(template_id)
                if definition:
                    templates.append({key: definition.get(key) for key in ("id", "name", "format", "revision", "created_at", "fields")})
        return {"templates": templates}
