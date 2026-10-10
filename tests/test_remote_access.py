from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from s4d import remote_access as access  # noqa: E402


class RemoteAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".remote-access-test-", dir=REPO)
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.secret = "boundary-only-password-!$`"
        self.credentials = self.repo / "fake-credentials"
        self.write_credentials()
        main = patch.object(access, "main_repository", return_value=self.repo)
        main.start()
        self.addCleanup(main.stop)
        self.client = access.RemoteClient(self.credentials, self.repo)

    def write_credentials(self, **overrides):
        values = {"SSH_HOST": access.REMOTE_HOST, "SSH_USER": access.REMOTE_USER,
                  "SSH_PASSWORD": self.secret, "REMOTE_ROOT": "~/kaist/sunho"}
        values.update(overrides)
        self.credentials.write_text("\n".join(k + "=" + shlex.quote(v) for k, v in values.items()) + "\n")
        self.credentials.chmod(0o600)

    @staticmethod
    def result(code=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(["boundary-fixture"], code, stdout, stderr)

    def test_credentials_are_data_not_executable_shell(self):
        text = "$(touch never-created) ' literal # value"
        self.write_credentials(SSH_PASSWORD=text)
        credentials = access.read_credentials(self.credentials)
        self.assertEqual(credentials.password, text)
        self.assertNotIn(text, repr(credentials))
        self.assertFalse((self.repo / "never-created").exists())
        self.assertEqual(credentials.root, access.REMOTE_ROOT)
        self.assertNotIn(self.secret, repr(self.client))

    def test_credentials_reject_permissions_unknown_fields_and_targets(self):
        self.credentials.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "accessible"):
            access.read_credentials(self.credentials)
        self.write_credentials(SSH_HOST="192.168.10.23")
        with self.assertRaisesRegex(ValueError, "not authorized"):
            access.read_credentials(self.credentials)
        self.write_credentials(REMOTE_ROOT="~/elsewhere")
        with self.assertRaisesRegex(ValueError, "root"):
            access.read_credentials(self.credentials)
        self.write_credentials()
        with self.credentials.open("a") as stream:
            stream.write("EXTRA=" + self.secret + "\n")
        with self.assertRaises(ValueError) as error:
            access.read_credentials(self.credentials)
        self.assertNotIn(self.secret, str(error.exception))

    def test_duplicate_or_malformed_credentials_do_not_echo_values(self):
        for tail in ("SSH_USER=compu\n", "SSH_PASSWORD='" + self.secret + "\n"):
            self.write_credentials()
            with self.credentials.open("a") as stream:
                stream.write(tail)
            with self.assertRaises(ValueError) as error:
                access.read_credentials(self.credentials)
            self.assertNotIn(self.secret, str(error.exception))

    def test_remote_paths_are_contained_and_shell_quoted(self):
        self.assertEqual(self.client.remote_path("data/task.hdf5"), access.REMOTE_ROOT + "/data/task.hdf5")
        for path in ("../other", "/home/compu/kaist/sunho-other/a", "~/other", "runs/a\nb", "runs/../a"):
            with self.assertRaises(ValueError):
                self.client.remote_path(path)
        for path in (access.REMOTE_ROOT, ""):
            with self.assertRaises(ValueError):
                self.client.remote_path(path)
        malicious = "data/a'; touch /outside; 'b"
        with patch.object(self.client, "checked_ssh", return_value=self.result()) as run:
            target = self.client.ensure_remote_path(malicious, create_parent=True)
        command = run.call_args.args[0]
        components = command.split("for component in ", 1)[1].split("; do ", 1)[0]
        self.assertEqual(shlex.split(components), ["data", "a'; touch ", "outside; 'b"])
        self.assertIn('[ ! -L "$current" ]', command)
        self.assertIn('realpath -m', command)
        self.assertEqual(target, access.REMOTE_ROOT + "/" + malicious)

    def test_shell_path_checks_actually_refuse_symlinks(self):
        root = self.repo / "remote-root"
        root.mkdir()
        outside = self.repo / "outside"
        outside.mkdir()
        (root / "escape").symlink_to(outside, target_is_directory=True)
        self.client.root = str(root)

        def local_shell(command, **kwargs):
            result = subprocess.run(["sh", "-c", command], capture_output=True, text=True, cwd=self.repo)
            if result.returncode:
                raise access.RemoteCommandError("Rejected local boundary fixture path")
            return result

        with patch.object(self.client, "checked_ssh", side_effect=local_shell):
            with self.assertRaises(access.RemoteCommandError):
                self.client.ensure_remote_path("escape/new-dir/file", create_parent=True)
            self.assertFalse((outside / "new-dir").exists())
            self.client.ensure_remote_path("safe/new-dir/file", create_parent=True)
            self.assertTrue((root / "safe/new-dir").is_dir())
            self.client.ensure_remote_path("data/$(touch marker)/file", create_parent=True)
            self.assertFalse((self.repo / "marker").exists())
            self.assertTrue((root / "data/$(touch marker)").is_dir())

    def test_remote_symlink_rejection_happens_before_transfer(self):
        source = self.repo / "source.bin"
        source.write_bytes(b"data")
        with patch.object(self.client, "checked_ssh", side_effect=access.RemoteCommandError("unsafe symlink")), \
                patch.object(self.client, "_rsync") as transfer:
            with self.assertRaises(access.RemoteCommandError):
                self.client.rsync_upload(source, "data/source.bin")
        transfer.assert_not_called()

    def test_local_symlinks_and_outside_state_are_rejected(self):
        source = self.repo / "source.bin"
        source.write_bytes(b"data")
        link = self.repo / "link.bin"
        link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.client._local_path(link)
        with self.assertRaisesRegex(ValueError, "outside"):
            access.RemoteClient(self.credentials, self.repo, state_dir=self.repo.parent / "outside-access")

    def test_ssh_password_is_child_environment_only_and_output_is_redacted(self):
        response = self.result(stdout=self.secret, stderr=self.secret)
        with patch.object(self.client, "check_tunnel"), patch.object(access.subprocess, "run", return_value=response) as run:
            result = self.client.ssh("true", wait=False)
        args = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertNotIn(self.secret, shlex.join(args))
        self.assertEqual(environment["SSHPASS"], self.secret)
        self.assertNotIn("SSH_PASSWORD", environment)
        self.assertEqual(result.stdout, "[REDACTED]")
        self.assertEqual(result.stderr, "[REDACTED]")
        self.assertIn("ControlMaster=auto", args)
        self.assertIn("ControlPersist=600", args)
        self.assertIn("StrictHostKeyChecking=accept-new", args)
        self.assertIn("PubkeyAuthentication=no", args)
        self.assertIn("/dev/null", args)
        self.assertNotIn("-p", args)

    def test_credentials_cannot_be_sent_as_commands_or_files(self):
        with patch.object(access.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                self.client.ssh("echo " + self.secret)
            with self.assertRaises(ValueError):
                self.client._rsync(self.secret, "data/file")
        run.assert_not_called()
        with patch.object(self.client, "ensure_remote_path") as validate:
            with self.assertRaisesRegex(ValueError, "credential"):
                self.client.rsync_upload(self.credentials, "data/credentials")
        validate.assert_not_called()

    def test_transport_loss_does_not_retry_nonidempotent_command(self):
        with patch.object(self.client, "check_tunnel"), \
                patch.object(access.subprocess, "run", return_value=self.result(255)) as run:
            with self.assertRaisesRegex(access.RemoteUnreachable, "unknown"):
                self.client.ssh("launch-once", wait=False)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(access.RemoteUnreachable.exit_code, 75)

    def test_only_idempotent_ssh_is_retried(self):
        with patch.object(self.client, "check_tunnel"), \
                patch.object(access.subprocess, "run", side_effect=[self.result(255), self.result(stdout="ok")]) as run:
            self.assertEqual(self.client.ssh("inspect-only", wait=False, idempotent=True).stdout, "ok")
        self.assertEqual(run.call_count, 2)
        with patch.object(self.client, "check_tunnel"), \
                patch.object(access.subprocess, "run", return_value=self.result(9)) as run:
            self.assertEqual(self.client.ssh("inspect-only", idempotent=True).returncode, 9)
        self.assertEqual(run.call_count, 1)

    def test_timeout_has_unknown_outcome_without_secret_leak(self):
        failure = subprocess.TimeoutExpired(["ssh"], 1, output=self.secret)
        with patch.object(self.client, "check_tunnel"), patch.object(access.subprocess, "run", side_effect=failure):
            with self.assertRaises(access.RemoteUnreachable) as error:
                self.client.ssh("launch-once", wait=False)
        self.assertNotIn(self.secret, str(error.exception))

    def test_outage_events_preserve_start_end_and_duration(self):
        with patch.object(access.socket, "create_connection", side_effect=ConnectionRefusedError), \
                patch.object(access.time, "time", return_value=100):
            with self.assertRaises(access.RemoteUnreachable):
                self.client.check_tunnel(wait=False)
        with patch.object(access.socket, "create_connection"), patch.object(access.time, "time", return_value=180):
            self.client.check_tunnel(wait=False)
        events = [json.loads(line) for line in (self.client.network_dir / "network.jsonl").read_text().splitlines()]
        self.assertEqual([e["event"] for e in events], ["REMOTE_UNREACHABLE", "REMOTE_RECONNECTED"])
        self.assertEqual(events[-1]["duration_seconds"], 80)

    def test_tunnel_backoff_is_bounded_at_ten_minutes(self):
        clock = [0.0]
        sleeps = []

        def sleep(delay):
            sleeps.append(delay)
            clock[0] += delay

        with patch.object(access.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(access.time, "sleep", side_effect=sleep), \
                patch.object(access.socket, "create_connection", side_effect=ConnectionRefusedError):
            with self.assertRaisesRegex(access.RemoteUnreachable, "VPN"):
                self.client.check_tunnel()
        self.assertEqual(sleeps[:4], [5, 10, 20, 40])
        self.assertLessEqual(max(sleeps), 60)
        self.assertEqual(sum(sleeps), 600)

    def test_rsync_flags_and_environment_are_safe_and_resumable(self):
        with patch.object(self.client, "check_tunnel"), \
                patch.object(access.subprocess, "run", side_effect=[self.result(12), self.result()]) as run:
            self.client._rsync("explicit-source", "explicit-target", wait=False)
        args = run.call_args.args[0]
        for flag in ("--partial", "--append-verify", "--checksum", "--protect-args"):
            self.assertIn(flag, args)
        self.assertNotIn(self.secret, shlex.join(args))
        self.assertEqual(run.call_args.kwargs["env"]["SSHPASS"], self.secret)
        self.assertEqual(run.call_count, 2)

    def test_upload_is_checksum_addressed_and_atomic(self):
        source = self.repo / "source.bin"
        source.write_bytes(b"new-content")
        digest = access.sha256(source)
        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "_remote_checksum", side_effect=[None, digest, digest]), \
                patch.object(self.client, "_rsync") as transfer, \
                patch.object(self.client, "checked_ssh", return_value=self.result()) as publish:
            self.client.rsync_upload(source, "data/source.bin")
        self.assertTrue(transfer.call_args.args[1].endswith(".s4d-partial-" + digest))
        self.assertIn("mv -T --", publish.call_args.args[0])
        self.assertFalse(publish.call_args.kwargs["idempotent"])

    def test_bad_upload_checksum_never_publishes(self):
        source = self.repo / "source.bin"
        source.write_bytes(b"data")
        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "_remote_checksum", side_effect=[None, "0" * 64]), \
                patch.object(self.client, "_rsync"), patch.object(self.client, "checked_ssh") as publish:
            with self.assertRaisesRegex(access.RemoteCommandError, "checksum"):
                self.client.rsync_upload(source, "data/source.bin")
        publish.assert_not_called()

    def test_download_keeps_previous_file_until_verified(self):
        target = self.repo / "results.json"
        target.write_bytes(b"old")
        content = b"new"
        source = self.repo / "hash-source"
        source.write_bytes(content)
        digest = access.sha256(source)

        def transfer(remote, partial, **kwargs):
            self.assertEqual(target.read_bytes(), b"old")
            Path(partial).write_bytes(content)

        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "_remote_checksum", return_value=digest), \
                patch.object(self.client, "_rsync", side_effect=transfer):
            self.client.rsync_download("runs/probe/results.json", target)
        self.assertEqual(target.read_bytes(), content)

    def test_changed_download_cannot_replace_previous_results(self):
        target = self.repo / "results.json"
        target.write_bytes(b"old")

        def transfer(remote, partial, **kwargs):
            Path(partial).write_bytes(b"wrong")

        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "_remote_checksum", side_effect=["a" * 64, "b" * 64]), \
                patch.object(self.client, "_rsync", side_effect=transfer):
            with self.assertRaises(access.RemoteCommandError):
                self.client.rsync_download("runs/probe/results.json", target)
        self.assertEqual(target.read_bytes(), b"old")

    def test_data_manifest_validates_all_entries_before_transfer(self):
        source = self.repo / "split.json"
        source.write_text("{}")
        good = {"source": str(source), "target": "data/metaworld/splatter4d_v1/splits/hammer.json",
                "sha256": access.sha256(source)}
        for second in (good, {**good, "target": "runs/wrong.json"}, {**good, "sha256": "0" * 64}):
            with patch.object(self.client, "rsync_upload") as upload:
                with self.assertRaises(ValueError):
                    self.client.sync_data({"files": [good, second]})
            upload.assert_not_called()

    def test_data_identity_excludes_source_paths_and_input_order(self):
        one = self.repo / "one.json"
        two = self.repo / "two.json"
        one.write_text("one")
        two.write_text("two")
        entries = [{"source": str(one), "target": "data/one.json"}, {"source": str(two), "target": "data/two.json"}]
        with patch.object(self.client, "rsync_upload"):
            first = self.client.sync_data({"files": entries})
            second = self.client.sync_data({"files": entries[::-1]})
        self.assertEqual(first["identity_sha256"], second["identity_sha256"])
        self.assertEqual(len(first["files"]), 2)

    def test_dirty_unpushed_and_stale_code_cannot_transfer(self):
        sha, tip = "a" * 40, "b" * 40
        for mode in ("dirty", "unpushed", "stale"):
            def git(*args, mode=mode, **kwargs):
                if args[0] == "status":
                    return " M source.py" if mode == "dirty" else ""
                if args[0] == "remote":
                    return "https://github.com/sunho001215/splatter_vae.git"
                if args[0] == "-c":
                    return tip + "\trefs/heads/splatter4d"
                if args[0] == "merge-base":
                    if mode == "unpushed":
                        raise access.RemoteCommandError("Not an ancestor")
                    return ""
                if args[0] == "rev-parse":
                    return sha
                raise AssertionError(args)

            with patch.object(self.client, "_git", side_effect=git), patch.object(self.client, "rsync_upload") as upload:
                with self.assertRaises(access.RemoteCommandError):
                    self.client.sync_code(sha)
            upload.assert_not_called()

    def test_operational_untracked_files_are_not_dirty_release_sources(self):
        status = "\x00".join("?? " + name for name in ("runs/", ".cache/", "outputs/", "experiments/HOLD")) + "\x00"
        with patch.object(self.client, "_git", side_effect=[status, access.RemoteCommandError("past status check")]):
            with self.assertRaisesRegex(access.RemoteCommandError, "past status check"):
                self.client.sync_code("a" * 40)
        for status in ("?? source.py\x00", " M runs/tracked.json\x00", "?? experiments/unknown.py\x00"):
            with patch.object(self.client, "_git", return_value=status):
                with self.assertRaisesRegex(access.RemoteCommandError, "dirty sources"):
                    self.client.sync_code("a" * 40)

    def test_code_release_accepts_ancestor_and_records_external_proof(self):
        sha, tip = "a" * 40, "b" * 40

        def git(*args, **kwargs):
            if args[0] == "status":
                return ""
            if args[0] == "remote":
                return "https://github.com/sunho001215/splatter_vae.git"
            if args[0] == "-c":
                return tip + "\trefs/heads/splatter4d"
            if args[0] == "rev-parse":
                return tip if args[-1] == "refs/remotes/origin/splatter4d" else sha
            if args[:2] == ("bundle", "create"):
                Path(args[2]).write_bytes(b"boundary-only-bundle")
            return ""

        with (
            patch.object(self.client, "_git", side_effect=git) as provenance,
            patch.object(
                self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p),
            ) as paths,
            patch.object(self.client, "rsync_upload") as upload,
            patch.object(self.client, "checked_ssh", return_value=self.result()) as checkout,
        ):
            code = self.client.sync_code(sha)
        self.assertEqual(code, access.REMOTE_ROOT + "/releases/" + sha + "/code")
        self.assertIn("fetch --quiet", checkout.call_args_list[0].args[0])
        self.assertIn("status --porcelain", checkout.call_args.args[0])
        for mountpoint in ("runs", "outputs", ".cache"):
            self.assertTrue(any(call.args == (code + "/" + mountpoint,) and
                                call.kwargs == {"create_parent": True, "directory": True}
                                for call in paths.call_args_list))
        self.assertTrue(any(call.args == ("merge-base", "--is-ancestor", sha, tip) for call in provenance.call_args_list))
        proof = json.loads(upload.call_args_list[-1].args[0].read_text())
        self.assertEqual(proof["commit"], sha)
        self.assertEqual(proof["origin_commit"], tip)
        self.assertEqual(proof["verified_origin_commit"], tip)
        self.assertTrue(proof["verified_ancestor"])
        self.assertTrue(upload.call_args_list[-1].args[1].endswith("/provenance.json"))

    def test_result_manifest_refuses_replay_traversal_and_duplicates(self):
        digest = "a" * 64
        for relative in ("../outside.json", "replay/data.json", "a/replay/data.json", "/outside.json", "a\\b.json"):
            with self.assertRaises(access.RemoteCommandError):
                self.client._result_manifest(digest + "  ./" + relative + "\n", ["replay"])
        duplicate = (digest + "  ./metrics.json\n") * 2
        with self.assertRaises(access.RemoteCommandError):
            self.client._result_manifest(duplicate, [])

    def test_final_results_require_stable_checksums_and_exclude_replay_checkpoints(self):
        run_id = "probe"
        local = self.repo / "runs" / run_id
        local.mkdir(parents=True)
        metric = local / "metrics.json"
        metric.write_text('{"step":1}')
        text = access.sha256(metric) + "  ./metrics.json\n"
        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "checked_ssh", return_value=self.result(stdout=text)) as manifest, \
                patch.object(self.client, "rsync_download") as download:
            result = self.client.sync_results(run_id, final=True)
        self.assertTrue(result["verified"])
        self.assertTrue(result["final"])
        self.assertEqual(result["files"], {"metrics.json": access.sha256(metric)})
        command = manifest.call_args.args[0]
        self.assertIn("-name replay", command)
        self.assertIn("-name checkpoints", command)
        self.assertIn("-name exit_code", command)
        download.assert_not_called()

    def test_default_results_include_only_literal_root_encoder_export_weights(self):
        remote = self.repo / "remote-run"
        remote.mkdir()
        weights = ["encoder.pt", "latest.pt", "arbitrary.pt", "nested/encoder.pt",
                   "snapshots/encoder.pt", "checkpoints/latest.pt", "replay/encoder.pt"]
        for relative in weights:
            path = remote / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())

        def manifest(command, **kwargs):
            return subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)

        def download(source, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(source).read_bytes())

        with patch.object(self.client, "ensure_remote_path", return_value=str(remote)), \
                patch.object(self.client, "checked_ssh", side_effect=manifest) as snapshots, \
                patch.object(self.client, "rsync_download", side_effect=download):
            result = self.client.sync_results("baseline", checkpoints=False, final=True)
            self.assertEqual(result["files"], {"encoder.pt": access.sha256(remote / "encoder.pt")})
            self.assertNotIn("-name '*.pt'", snapshots.call_args.args[0])
            self.assertIn("-path './encoder.pt'", snapshots.call_args.args[0])
            with_checkpoints = self.client.sync_results("baseline", checkpoints=True, final=True)
        self.assertEqual(set(with_checkpoints["files"]), set(weights) - {"replay/encoder.pt"})
        self.assertFalse((self.repo / "runs/baseline/replay/encoder.pt").exists())

    def test_changed_final_results_leave_unverified_marker(self):
        run_id = "changed"
        content = self.repo / "seed.json"
        content.write_text("before")
        before = access.sha256(content) + "  ./metrics.json\n"
        after = "0" * 64 + "  ./metrics.json\n"

        def download(remote, local, **kwargs):
            local.write_text("before")

        responses = [self.result(stdout=before), self.result(stdout=after)]
        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "checked_ssh", side_effect=responses), \
                patch.object(self.client, "rsync_download", side_effect=download):
            with self.assertRaisesRegex(access.RemoteCommandError, "changed"):
                self.client.sync_results(run_id, final=True)
        marker = json.loads((self.client.network_dir / "results" / (run_id + ".json")).read_text())
        self.assertFalse(marker["verified"])
        self.assertFalse(marker["final"])

    def test_result_sync_wait_false_propagates_to_all_access(self):
        run_id = "nonblocking"
        source = self.repo / "fixture.json"
        source.write_text("data")
        text = access.sha256(source) + "  ./metrics.json\n"

        def download(remote, local, **kwargs):
            self.assertFalse(kwargs["wait"])
            local.write_text("data")

        with (
            patch.object(
                self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p),
            ) as validate,
            patch.object(self.client, "checked_ssh", return_value=self.result(stdout=text)) as snapshots,
            patch.object(self.client, "rsync_download", side_effect=download),
        ):
            self.client.sync_results(run_id, final=True, wait=False)
        self.assertFalse(validate.call_args.kwargs["wait"])
        self.assertTrue(all(not call.kwargs["wait"] for call in snapshots.call_args_list))

    def test_shell_wrappers_check_tunnel_without_reading_credentials(self):
        shim = self.repo / "fake-python"
        log = self.repo / "wrapper-calls"
        shim.write_text('#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$3" >> "$WRAPPER_LOG"\n')
        shim.chmod(0o700)
        for script, command in (("ssh.sh", "ssh"), ("sync_code.sh", "sync-code"),
                                ("sync_data.sh", "sync-data"), ("sync_results.sh", "sync-results")):
            log.unlink(missing_ok=True)
            environment = dict(os.environ, S4D_REMOTE_PYTHON=str(shim), WRAPPER_LOG=str(log),
                               S4D_REMOTE_CREDENTIALS=str(self.repo / "does-not-exist"))
            subprocess.run(["bash", str(REPO / "scripts/remote" / script), "fixture"], env=environment, check=True)
            self.assertEqual(log.read_text().splitlines(), ["check-tunnel", command])

    def test_empty_final_result_snapshot_is_not_accepted(self):
        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "checked_ssh", return_value=self.result()):
            with self.assertRaisesRegex(access.RemoteCommandError, "empty"):
                self.client.sync_results("empty", final=True)


if __name__ == "__main__":
    unittest.main()
