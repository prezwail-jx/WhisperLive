import logging
import os
import re
import threading
import unicodedata


MAX_ASR_HOTWORD_TERMS = 20
MAX_ASR_HOTWORD_TERM_CHARS = 64
MAX_ASR_HOTWORD_PROMPT_CHARS = 180


def _unique_reasons(reasons):
    return list(dict.fromkeys(reasons))


def normalize_asr_hotword_term(value):
    term = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", term).strip()


def normalize_asr_hotwords(
    value=None,
    terms=None,
    max_terms=MAX_ASR_HOTWORD_TERMS,
    max_term_chars=MAX_ASR_HOTWORD_TERM_CHARS,
    max_prompt_chars=MAX_ASR_HOTWORD_PROMPT_CHARS,
):
    if terms is None:
        if isinstance(value, dict):
            terms = value.get("terms") or value.get("hotwords") or []
        elif isinstance(value, (list, tuple, set)):
            terms = value
        else:
            terms = parse_hotword_config(value).get("hotwords") or []

    candidates = []
    invalid_reasons = []
    for item in terms or []:
        term = normalize_asr_hotword_term(item)
        if not term:
            continue
        candidates.append(term)

    accepted = []
    seen = set()
    prompt = ""
    reasons = []

    for term in candidates:
        if len(term) > max_term_chars:
            invalid_reasons.append("term_length")
            continue
        key = term.casefold()
        if key in seen:
            invalid_reasons.append("duplicate")
            continue
        if len(accepted) >= max_terms:
            reasons.append("term_count")
            break
        next_prompt = term if not prompt else f"{prompt} {term}"
        if len(next_prompt) > max_prompt_chars:
            reasons.append("prompt_chars")
            break
        accepted.append(term)
        seen.add(key)
        prompt = next_prompt

    original_count = len(candidates)
    accepted_count = len(accepted)
    truncation_reasons = _unique_reasons(reasons)
    validation_reasons = _unique_reasons(invalid_reasons)
    return {
        "terms": accepted,
        "prompt": prompt or None,
        "original_count": original_count,
        "accepted_count": accepted_count,
        "rejected_count": max(0, original_count - accepted_count),
        "truncated": bool(truncation_reasons),
        "truncation_reasons": truncation_reasons,
        "validation_reasons": validation_reasons,
        "max_terms": max_terms,
        "max_term_chars": max_term_chars,
        "max_prompt_chars": max_prompt_chars,
    }


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
    return normalize_asr_hotwords(terms=parsed["hotwords"])["prompt"]


def count_hotwords(text):
    return parse_hotword_config(text)["count"]


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
