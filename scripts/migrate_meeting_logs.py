#!/usr/bin/env python3
"""Safely inventory and migrate historical WhisperLive meeting logs."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_files(json_path):
    stem = json_path.with_suffix("")
    files = [json_path]
    for suffix in (".md", "-summary.json", "-summary.md"):
        candidate = Path(f"{stem}{suffix}")
        if candidate.is_file():
            files.append(candidate)
    versions = Path(f"{stem}-summaries")
    if versions.is_dir():
        files.extend(path for path in versions.rglob("*") if path.is_file())
    return sorted(files)


def file_set(json_path):
    result = {}
    for path in session_files(json_path):
        relative = path.relative_to(json_path.parent)
        result[str(relative)] = file_digest(path)
    return result


def invalid_companion_json(json_path):
    for path in session_files(json_path):
        if path == json_path or path.suffix != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            return path, str(error)
        if not isinstance(payload, dict):
            return path, "JSON companion is not an object"
    return None


def is_session_record(path):
    return path.suffix == ".json" and not path.name.endswith("-summary.json") and not any(
        parent.name.endswith("-summaries") for parent in path.parents
    )


def inventory_tree(root, label, missing_is_empty=False):
    root = Path(root)
    inventory = {"label": label, "sessions": {}, "invalid_records": []}
    if not root.exists():
        if not missing_is_empty:
            inventory["invalid_records"].append({"path": str(root), "reason": "directory does not exist"})
        return inventory
    if not root.is_dir():
        inventory["invalid_records"].append({"path": str(root), "reason": "not a directory"})
        return inventory

    for path in sorted(root.rglob("*.json")):
        if not is_session_record(path):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            inventory["invalid_records"].append({"path": str(path), "reason": str(error)})
            continue
        if not isinstance(payload, dict):
            inventory["invalid_records"].append({"path": str(path), "reason": "JSON record is not an object"})
            continue
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            inventory["invalid_records"].append({"path": str(path), "reason": "missing session_id"})
            continue
        if not isinstance(payload.get("source_segments"), list):
            inventory["invalid_records"].append({"path": str(path), "reason": "missing source_segments list"})
            continue
        companion_error = invalid_companion_json(path)
        if companion_error:
            invalid_path, reason = companion_error
            inventory["invalid_records"].append({"path": str(invalid_path), "reason": reason})
            continue
        try:
            files = file_set(path)
        except OSError as error:
            inventory["invalid_records"].append({"path": str(path), "reason": str(error)})
            continue
        record = {
            "path": path,
            "destination_dir": path.parent.relative_to(root),
            "files": files,
        }
        inventory["sessions"].setdefault(session_id, []).append(record)
    return inventory


def inventory_summary(inventory):
    return {
        "label": inventory["label"],
        "session_count": len(inventory["sessions"]),
        "record_count": sum(len(records) for records in inventory["sessions"].values()),
        "invalid_records": inventory["invalid_records"],
    }


def make_plan(source_one, source_two, destination):
    inventories = [
        inventory_tree(source_one, "source_one"),
        inventory_tree(source_two, "source_two"),
        inventory_tree(destination, "destination", missing_is_empty=True),
    ]
    by_session = {}
    for inventory in inventories:
        for session_id, records in inventory["sessions"].items():
            by_session.setdefault(session_id, []).extend((inventory["label"], record) for record in records)

    copies = []
    duplicates = []
    conflicts = []
    for session_id in sorted(by_session):
        records = by_session[session_id]
        source_records = [(label, record) for label, record in records if label != "destination"]
        destination_records = [(label, record) for label, record in records if label == "destination"]
        if not source_records:
            continue
        reference_files = source_records[0][1]["files"]
        if any(record["files"] != reference_files for _label, record in source_records[1:]):
            conflicts.append({"session_id": session_id, "reason": "source file sets differ", "paths": [str(record["path"]) for _label, record in source_records]})
            continue
        if len(destination_records) > 1 or any(record["files"] != reference_files for _label, record in destination_records):
            conflicts.append({"session_id": session_id, "reason": "destination file set differs", "paths": [str(record["path"]) for _label, record in records]})
            continue
        if destination_records:
            duplicates.append({"session_id": session_id, "paths": [str(record["path"]) for _label, record in records]})
            continue
        source_label, source_record = source_records[0]
        destination_dir = Path(destination) / source_record["destination_dir"]
        target_paths = [destination_dir / relative for relative in source_record["files"]]
        existing_paths = [str(path) for path in target_paths if path.exists()]
        if existing_paths:
            conflicts.append({"session_id": session_id, "reason": "destination paths already exist", "paths": existing_paths})
            continue
        copies.append({
            "session_id": session_id,
            "source": str(source_record["path"].parent),
            "destination": str(destination_dir),
            "files": sorted(source_record["files"]),
            "source_label": source_label,
        })
        if len(source_records) > 1:
            duplicates.append({"session_id": session_id, "paths": [str(record["path"]) for _label, record in source_records]})

    return {
        "mode": "dry-run",
        "inventories": [inventory_summary(inventory) for inventory in inventories],
        "copy_count": len(copies),
        "copies": copies,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }


def execute_plan(plan):
    for copy in plan["copies"]:
        source_dir = Path(copy["source"])
        destination_dir = Path(copy["destination"])
        for relative in copy["files"]:
            source = source_dir / relative
            destination = destination_dir / relative
            if not source.is_file():
                raise FileNotFoundError(f"source file changed after dry-run: {source}")
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite destination file: {destination}")
    for copy in plan["copies"]:
        source_dir = Path(copy["source"])
        destination_dir = Path(copy["destination"])
        for relative in copy["files"]:
            source = source_dir / relative
            destination = destination_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    plan["mode"] = "execute"
    return plan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inventory and safely migrate WhisperLive meeting-log trees.")
    parser.add_argument("source_one", help="First source meeting-log tree")
    parser.add_argument("source_two", help="Second source meeting-log tree")
    parser.add_argument("destination", help="Empty staging/shared meeting-log tree")
    parser.add_argument("--execute", action="store_true", help="Copy planned sessions; default is dry-run only")
    parser.add_argument("--report", help="Write the JSON report to this path as well as stdout")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args.source_one, args.source_two, args.destination)
    if args.execute:
        execute_plan(plan)
    report = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
