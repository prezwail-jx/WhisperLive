import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "migrate_meeting_logs.py"
SPEC = importlib.util.spec_from_file_location("migrate_meeting_logs", SCRIPT_PATH)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


class TestMeetingLogMigration(unittest.TestCase):
    def write_session(self, root, directory, stem, session_id, text="hello", summary=False):
        session_dir = Path(root) / directory
        session_dir.mkdir(parents=True, exist_ok=True)
        payload = {"session_id": session_id, "source_segments": [{"text": text}]}
        (session_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
        (session_dir / f"{stem}.md").write_text(f"# {text}\n", encoding="utf-8")
        if summary:
            (session_dir / f"{stem}-summary.json").write_text(json.dumps({"session_id": session_id}), encoding="utf-8")
            (session_dir / f"{stem}-summary.md").write_text("# summary\n", encoding="utf-8")
            versions = session_dir / f"{stem}-summaries"
            versions.mkdir()
            (versions / "v0001.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
            (versions / "v0001.md").write_text("# version\n", encoding="utf-8")

    def test_unique_session_is_planned_then_copied_with_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            source_one = Path(directory) / "gpu0"
            source_two = Path(directory) / "gpu1"
            destination = Path(directory) / "shared"
            self.write_session(source_one, "weekly", "2026-session-a", "session-a", summary=True)

            plan = migration.make_plan(source_one, source_two, destination)

            self.assertEqual(plan["mode"], "dry-run")
            self.assertEqual(plan["copy_count"], 1)
            self.assertFalse(destination.exists())
            with redirect_stdout(io.StringIO()):
                result = migration.main([str(source_one), str(source_two), str(destination), "--execute"])
            self.assertEqual(result, 0)
            copied = destination / "weekly"
            self.assertTrue((copied / "2026-session-a.json").is_file())
            self.assertTrue((copied / "2026-session-a.md").is_file())
            self.assertTrue((copied / "2026-session-a-summary.json").is_file())
            self.assertTrue((copied / "2026-session-a-summaries" / "v0001.md").is_file())

    def test_identical_duplicate_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            source_one = Path(directory) / "gpu0"
            source_two = Path(directory) / "gpu1"
            destination = Path(directory) / "shared"
            self.write_session(source_one, "weekly", "session", "same")
            self.write_session(source_two, "weekly", "session", "same")

            plan = migration.make_plan(source_one, source_two, destination)

            self.assertEqual(plan["copy_count"], 1)
            self.assertEqual(plan["duplicate_count"], 1)
            self.assertEqual(plan["conflict_count"], 0)

    def test_different_duplicate_is_reported_without_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source_one = Path(directory) / "gpu0"
            source_two = Path(directory) / "gpu1"
            destination = Path(directory) / "shared"
            self.write_session(source_one, "weekly", "session", "same", text="first")
            self.write_session(source_two, "weekly", "session", "same", text="second")

            plan = migration.make_plan(source_one, source_two, destination)

            self.assertEqual(plan["copy_count"], 0)
            self.assertEqual(plan["conflict_count"], 1)
            self.assertFalse(destination.exists())

    def test_malformed_json_is_reported_and_not_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            source_one = Path(directory) / "gpu0"
            source_two = Path(directory) / "gpu1"
            destination = Path(directory) / "shared"
            source_one.mkdir()
            (source_one / "broken.json").write_text("{broken", encoding="utf-8")

            plan = migration.make_plan(source_one, source_two, destination)

            invalid = plan["inventories"][0]["invalid_records"]
            self.assertEqual(plan["copy_count"], 0)
            self.assertEqual(len(invalid), 1)
            self.assertIn("broken.json", invalid[0]["path"])

    def test_malformed_summary_is_reported_and_not_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            source_one = Path(directory) / "gpu0"
            source_two = Path(directory) / "gpu1"
            destination = Path(directory) / "shared"
            self.write_session(source_one, "weekly", "session", "session-a", summary=True)
            (source_one / "weekly" / "session-summary.json").write_text("{broken", encoding="utf-8")

            plan = migration.make_plan(source_one, source_two, destination)

            self.assertEqual(plan["copy_count"], 0)
            self.assertIn("session-summary.json", plan["inventories"][0]["invalid_records"][0]["path"])


if __name__ == "__main__":
    unittest.main()
