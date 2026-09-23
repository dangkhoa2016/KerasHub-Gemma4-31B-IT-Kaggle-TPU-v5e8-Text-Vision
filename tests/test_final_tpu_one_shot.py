from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from final_tpu_one_shot import (  # noqa: E402
    assert_source_identity,
    package_evidence,
    poll_job,
    read_cgroup_snapshot,
    request_json,
    runtime_identity_from_paths,
    sha256_file,
)


class OneShotUtilityTests(unittest.TestCase):
    def test_reads_runtime_identity_from_explicit_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hostname").write_text("host-a\n")
            (root / "boot_id").write_text("boot-a\n")
            (root / "pid1").write_text("1\n")

            identity = runtime_identity_from_paths(
                hostname_path=root / "hostname",
                boot_id_path=root / "boot_id",
                pid1_path=root / "pid1",
            )

        self.assertEqual(identity["hostname"], "host-a")
        self.assertEqual(identity["boot_id"], "boot-a")
        self.assertEqual(identity["pid1"], "1")

    def test_reads_memory_and_oom_cgroup_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "memory.current").write_text("123\n")
            (root / "memory.max").write_text("456\n")
            (root / "memory.events").write_text(
                "low 0\nhigh 1\nmax 2\noom 3\noom_kill 4\noom_group_kill 5\n"
            )

            snapshot = read_cgroup_snapshot(root)

        self.assertEqual(snapshot["memory_current"], 123)
        self.assertEqual(snapshot["memory_max"], 456)
        self.assertEqual(snapshot["memory_events"]["oom"], 3)
        self.assertEqual(snapshot["memory_events"]["oom_kill"], 4)
        self.assertEqual(snapshot["memory_events"]["oom_group_kill"], 5)

    def test_hashes_file_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.txt"
            path.write_text("payload\n")
            expected = hashlib.sha256(b"payload\n").hexdigest()
            self.assertEqual(sha256_file(path), expected)

    def test_source_identity_requires_exact_clean_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            import subprocess

            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Test"],
                check=True,
            )
            (repo / "file.txt").write_text("ok\n")
            subprocess.run(["git", "-C", str(repo), "add", "file.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-qm", "initial"],
                check=True,
            )
            expected = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()

            result = assert_source_identity(repo, expected)
            self.assertTrue(result["git_sha_exact"])
            self.assertTrue(result["worktree_clean"])

            (repo / "file.txt").write_text("dirty\n")
            with self.assertRaisesRegex(RuntimeError, "clean"):
                assert_source_identity(repo, expected)

    def test_packages_only_allowed_compact_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "00-context.txt").write_text("context\n")
            (root / "11-memory-before.txt").write_text("memory\n")
            (root / ".env").write_text("secret\n")
            (root / "private.log").write_text("private\n")
            (root / "restart_secret.txt").write_text("secret\n")
            (root / "state").mkdir()
            (root / "state" / "worker.pid").write_text("123\n")

            sums_path = package_evidence(root)
            manifest = sums_path.read_text()

        self.assertIn("00-context.txt", manifest)
        self.assertIn("11-memory-before.txt", manifest)
        self.assertNotIn(".env", manifest)
        self.assertNotIn("private.log", manifest)
        self.assertNotIn("restart_secret.txt", manifest)
        self.assertNotIn("worker.pid", manifest)


class FakeApiHandler(BaseHTTPRequestHandler):
    polls = 0

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/result/"):
            type(self).polls += 1
            status = "completed" if type(self).polls >= 2 else "processing"
            body = {"status": status, "output": "done" if status == "completed" else None}
        else:
            body = {"ready": True}
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        return


class HttpUtilityTests(unittest.TestCase):
    def setUp(self):
        FakeApiHandler.polls = 0
        self.server = HTTPServer(("127.0.0.1", 0), FakeApiHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    def test_request_json_and_poll_job(self):
        status, payload = request_json(self.base_url, "/health/ready")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ready"])

        result = poll_job(
            self.base_url,
            "job-1",
            headers={},
            timeout=2,
            interval=0.01,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output"], "done")

    def test_poll_job_times_out(self):
        with self.assertRaises(TimeoutError):
            poll_job(
                self.base_url,
                "job-1",
                headers={},
                timeout=0.01,
                interval=0.01,
            )


if __name__ == "__main__":
    unittest.main()
