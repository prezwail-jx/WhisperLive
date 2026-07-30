import logging
import os
import re
import threading


def parse_asr_correction_config(text):
    rules = {}
    normalized_lines = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" not in line:
            logging.warning("Ignoring invalid ASR correction rule: %r", line)
            continue
        source, target = (part.strip() for part in line.split("=>", 1))
        if not source or not target:
            logging.warning("Ignoring invalid ASR correction rule: %r", line)
            continue
        rules[source] = target
        normalized_lines.append(f"{source} => {target}")
    ordered_rules = sorted(rules.items(), key=lambda item: len(item[0]), reverse=True)
    return {
        "text": "\n".join(normalized_lines),
        "rules": ordered_rules,
        "count": len(ordered_rules),
    }


class AsrTextCorrector:
    def __init__(self, rules=None):
        self.rules = tuple(sorted(rules or [], key=lambda item: len(item[0]), reverse=True))
        self.replacements = dict(self.rules)
        self.pattern = None
        if self.rules:
            self.pattern = re.compile("|".join(re.escape(source) for source, _target in self.rules))

    def correct(self, text):
        original = str(text or "")
        if self.pattern is None:
            return original, 0
        replacements = 0

        def replace(match):
            nonlocal replacements
            replacements += 1
            return self.replacements.get(match.group(0), match.group(0))

        corrected = self.pattern.sub(replace, original)
        return corrected, replacements


class MeetingAsrCorrectionStore:
    def __init__(self, directory="config/asr_corrections.d"):
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
            "rules": [],
            "count": 0,
            "updated_at": None,
        }

    def _safe_path(self, meeting_name):
        name = self.normalize_name(meeting_name)
        if not name:
            raise ValueError("meeting_name is required")
        if os.path.basename(name) != name or "/" in name or "\\" in name:
            raise ValueError("meeting_name must match an ASR correction txt filename")
        return name, os.path.join(self.directory, f"{name}.txt")

    def _record_from_file(self, meeting_name, path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="utf-8-sig") as file:
                text = file.read()
        parsed = parse_asr_correction_config(text)
        return {
            "meeting_name": meeting_name,
            "filename": os.path.basename(path),
            "path": path,
            "text": parsed["text"],
            "rules": parsed["rules"],
            "count": parsed["count"],
            "updated_at": os.path.getmtime(path),
        }

    def get_file(self, path, name="global"):
        if not path:
            return self._empty_record(name)
        path = os.path.expanduser(str(path))
        with self.lock:
            if not os.path.isfile(path):
                logging.warning("ASR correction file not found: %s", path)
                return self._empty_record(name)
            return self._record_from_file(name, path)

    def get(self, meeting_name):
        name, path = self._safe_path(meeting_name)
        with self.lock:
            if not os.path.isfile(path):
                return self._empty_record(name)
            return self._record_from_file(name, path)

    def corrector_for(self, meeting_name):
        record = self.get(meeting_name)
        return AsrTextCorrector(record.get("rules") or []), record
