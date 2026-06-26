import hashlib
import re


class TranscriptRevisionConflict(ValueError):
    pass


def _segment_id(session_id, kind, index, segment):
    raw = "|".join([
        str(session_id or ""), kind, str(index), str(segment.get("start") or ""),
        str(segment.get("end") or ""), str(segment.get("original_text") or segment.get("text") or ""),
    ])
    return f"{kind}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _speaker_id(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "").strip()).strip("-_").lower()[:64]


def normalize_transcript(payload):
    payload.setdefault("transcript_revision", 0)
    payload.setdefault("transcript_edits", [])
    payload.setdefault("translation_stale", False)
    payload.setdefault("summary_stale", False)
    if not isinstance(payload.get("speakers"), list):
        payload["speakers"] = []
    speakers = {
        str(item.get("speaker_id")): item
        for item in payload["speakers"]
        if isinstance(item, dict) and item.get("speaker_id")
    }
    for kind, key in (("source", "source_segments"), ("translation", "translation_segments")):
        if not isinstance(payload.get(key), list):
            payload[key] = []
        for index, segment in enumerate(payload[key]):
            if not isinstance(segment, dict):
                continue
            segment.setdefault("segment_id", _segment_id(payload.get("session_id"), kind, index, segment))
            segment.setdefault("original_text", str(segment.get("text") or ""))
            legacy = str(segment.get("speaker") or "").strip()
            speaker_id = str(segment.get("speaker_id") or "").strip()
            if not speaker_id and legacy:
                speaker_id = _speaker_id(legacy)
                segment["speaker_id"] = speaker_id
            if speaker_id and speaker_id not in speakers:
                item = {"speaker_id": speaker_id, "name": legacy or speaker_id}
                payload["speakers"].append(item)
                speakers[speaker_id] = item
    return payload


def transcript_view(payload):
    normalize_transcript(payload)
    return {
        "session_id": payload.get("session_id"),
        "status": payload.get("status"),
        "transcript_revision": int(payload.get("transcript_revision") or 0),
        "translation_stale": bool(payload.get("translation_stale")),
        "summary_stale": bool(payload.get("summary_stale")),
        "speakers": [dict(item) for item in payload["speakers"]],
        "segments": [dict(item) for item in payload["source_segments"]],
    }


def _check_revision(payload, expected_revision):
    current = int(payload.get("transcript_revision") or 0)
    try:
        expected = int(expected_revision)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_revision is required") from exc
    if expected != current:
        raise TranscriptRevisionConflict(
            f"transcript revision changed: expected {expected}, current {current}"
        )


def _record(payload, action, details, now):
    payload["transcript_revision"] = int(payload.get("transcript_revision") or 0) + 1
    payload["transcript_updated_at"] = now
    payload["transcript_edits"].append({
        "revision": payload["transcript_revision"],
        "action": action,
        "details": details,
        "updated_at": now,
    })


def _speaker(payload, speaker_id):
    return next((item for item in payload["speakers"] if item.get("speaker_id") == speaker_id), None)


def update_segment(payload, segment_id, text, speaker_id, expected_revision, now):
    normalize_transcript(payload)
    _check_revision(payload, expected_revision)
    segment = next((item for item in payload["source_segments"] if item.get("segment_id") == segment_id), None)
    if not segment:
        raise KeyError("transcript segment not found")
    normalized_text = str(text if text is not None else segment.get("text") or "").strip()
    if not normalized_text:
        raise ValueError("segment text cannot be empty")
    if len(normalized_text) > 10000:
        raise ValueError("segment text is too long")
    normalized_speaker = str(speaker_id or "").strip() or None
    if normalized_speaker and not _speaker(payload, normalized_speaker):
        raise ValueError("speaker not found")
    old_text = str(segment.get("text") or "")
    old_speaker = segment.get("speaker_id") or None
    if old_text == normalized_text and old_speaker == normalized_speaker:
        return False
    segment["text"] = normalized_text
    if normalized_speaker:
        segment["speaker_id"] = normalized_speaker
    else:
        segment.pop("speaker_id", None)
        segment.pop("speaker", None)
    _record(payload, "update_segment", {
        "segment_id": segment_id,
        "old_text": old_text,
        "new_text": normalized_text,
        "old_speaker_id": old_speaker,
        "new_speaker_id": normalized_speaker,
    }, now)
    payload["summary_stale"] = True
    if old_text != normalized_text:
        payload["translation_stale"] = True
    return True


def create_speaker(payload, name, expected_revision, now):
    normalize_transcript(payload)
    _check_revision(payload, expected_revision)
    name = str(name or "").strip()
    if not name:
        raise ValueError("speaker name cannot be empty")
    if len(name) > 80:
        raise ValueError("speaker name is too long")
    if any(str(item.get("name") or "").strip() == name for item in payload["speakers"]):
        raise ValueError("speaker name already exists")
    used = {item.get("speaker_id") for item in payload["speakers"]}
    index = 1
    while f"speaker-{index}" in used:
        index += 1
    item = {"speaker_id": f"speaker-{index}", "name": name}
    payload["speakers"].append(item)
    _record(payload, "create_speaker", dict(item), now)
    payload["summary_stale"] = True
    return item


def rename_speaker(payload, speaker_id, name, expected_revision, now):
    normalize_transcript(payload)
    _check_revision(payload, expected_revision)
    item = _speaker(payload, speaker_id)
    if not item:
        raise KeyError("speaker not found")
    name = str(name or "").strip()
    if not name:
        raise ValueError("speaker name cannot be empty")
    if len(name) > 80:
        raise ValueError("speaker name is too long")
    old_name = str(item.get("name") or "")
    if old_name == name:
        return False
    item["name"] = name
    _record(payload, "rename_speaker", {
        "speaker_id": speaker_id, "old_name": old_name, "new_name": name,
    }, now)
    payload["summary_stale"] = True
    return True


def merge_speakers(payload, source_id, target_id, expected_revision, now):
    normalize_transcript(payload)
    _check_revision(payload, expected_revision)
    if source_id == target_id:
        raise ValueError("cannot merge a speaker into itself")
    if not _speaker(payload, source_id) or not _speaker(payload, target_id):
        raise KeyError("speaker not found")
    changed = 0
    for segment in payload["source_segments"]:
        if segment.get("speaker_id") == source_id:
            segment["speaker_id"] = target_id
            changed += 1
    payload["speakers"] = [item for item in payload["speakers"] if item.get("speaker_id") != source_id]
    _record(payload, "merge_speakers", {
        "source_speaker_id": source_id,
        "target_speaker_id": target_id,
        "changed_segments": changed,
    }, now)
    payload["summary_stale"] = True
    return True
