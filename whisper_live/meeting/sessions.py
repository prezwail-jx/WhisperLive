import copy
from datetime import datetime, timezone

SESSION_ACTIVE = "active"
SESSION_INTERRUPTED = "interrupted"
SESSION_FINISHED = "finished"


def parse_iso_timestamp(value):
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def seconds_between(start, end):
    start_dt = parse_iso_timestamp(start)
    end_dt = parse_iso_timestamp(end)
    if not start_dt or not end_dt:
        return 0.0
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    return max(0.0, (end_dt - start_dt).total_seconds())


def can_resume_payload(payload, options):
    if not payload or payload.get("status") == SESSION_FINISHED:
        return False, "session is finished"
    expected = str(payload.get("client_instance_id") or "")
    current = str(options.get("client_instance_id") or "")
    if expected and current and expected != current:
        return False, "client_instance_id mismatch"
    return True, ""


def apply_timeline_offset(segment, offset):
    try:
        offset = float(offset or 0)
    except (TypeError, ValueError):
        offset = 0.0
    if not offset:
        return dict(segment) if isinstance(segment, dict) else segment
    if not isinstance(segment, dict):
        return segment
    adjusted = copy.deepcopy(segment)
    for key in ("start", "end"):
        try:
            adjusted[key] = "{:.3f}".format(float(adjusted.get(key) or 0) + offset)
        except (TypeError, ValueError):
            pass
    for word in adjusted.get("words") or []:
        if not isinstance(word, dict):
            continue
        for key in ("start", "end"):
            try:
                word[key] = float(word.get(key) or 0) + offset
            except (TypeError, ValueError):
                pass
    return adjusted


def apply_timeline_offset_to_segments(segments, offset):
    return [apply_timeline_offset(segment, offset) for segment in segments or []]
