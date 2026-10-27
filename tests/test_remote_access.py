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
from s4d.gpu_guard import validate_runtime_path  # noqa: E402


class RemoteAccessTests(unittest.TestCase):
    def setUp(self):
        temp_root = validate_runtime_path(Path(os.environ.get("S4D_PYTEST_TMP", str(REPO / ".pytest_tmp"))))
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix=".remote-access-test-", dir=temp_root)
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

    def test_release_carries_each_pinned_submodule_and_verifies_the_checkout(self):
        def run(cwd, *args):
            subprocess.run(["git", "-C", str(cwd), "-c", "protocol.file.allow=always", "-c", "user.name=t", "-c",
                            "user.email=t@t", *args], capture_output=True, check=True)

        upstream = self.repo / "upstream"
        upstream.mkdir()
        run(upstream, "init", "-q")
        (upstream / "model.py").write_text("WIDTH = 1\n")
        run(upstream, "add", "model.py")
        run(upstream, "commit", "-qm", "first")
        pinned = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        superproject = self.repo / "super"
        superproject.mkdir()
        run(superproject, "init", "-q")
        run(superproject, "submodule", "add", "-q", str(upstream), "third_party/upstream")
        run(superproject, "commit", "-qm", "pin")
        sha = subprocess.run(["git", "-C", str(superproject), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        self.client.repo = superproject
        code = access.REMOTE_ROOT + "/releases/" + sha + "/code"
        with patch.object(self.client, "rsync_upload") as upload, \
                patch.object(self.client, "checked_ssh", return_value=self.result()) as ssh:
            released = self.client._sync_submodules(sha, code, "git -C " + code)
        self.assertEqual(released["third_party/upstream"]["commit"], pinned)
        bundle = upload.call_args.args[0]
        subprocess.run(["git", "-C", str(upstream), "bundle", "verify", str(bundle)], capture_output=True, check=True)
        fetch, verify = ssh.call_args_list[0].args[0], ssh.call_args_list[-1].args[0]
        self.assertIn(code + "/third_party/upstream", fetch)
        self.assertIn("checkout --quiet --detach " + pinned, fetch)
        self.assertIn("status --porcelain --untracked-files=all", fetch)
        self.assertIn("submodule status", verify)
        self.client.repo = self.repo
        (superproject / "third_party/upstream/.git").unlink() if (superproject / "third_party/upstream/.git").is_file() \
            else None
        self.client.repo = superproject
        with patch.object(self.client, "rsync_upload"), patch.object(self.client, "checked_ssh"):
            with self.assertRaisesRegex(access.RemoteCommandError, "not checked out"):
                self.client._sync_submodules(sha, code, "git -C " + code)

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
                patch.object(self.client, "download_batch") as download:
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

        def download(source, local, wanted, name, **kwargs):
            for relative in wanted:
                destination = local / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((Path(source) / relative).read_bytes())

        with patch.object(self.client, "ensure_remote_path", return_value=str(remote)), \
                patch.object(self.client, "checked_ssh", side_effect=manifest) as snapshots, \
                patch.object(self.client, "download_batch", side_effect=download):
            result = self.client.sync_results("baseline", checkpoints=False, final=True)
            self.assertEqual(result["files"], {"encoder.pt": access.sha256(remote / "encoder.pt")})
            self.assertNotIn("-name '*.pt'", snapshots.call_args.args[0])
            self.assertIn("-path './encoder.pt'", snapshots.call_args.args[0])
            self.assertIn("-path './encoder_init.pt'", snapshots.call_args.args[0])
            with_checkpoints = self.client.sync_results("baseline", checkpoints=True, final=True)
        self.assertEqual(set(with_checkpoints["files"]), set(weights) - {"replay/encoder.pt"})
        self.assertFalse((self.repo / "runs/baseline/replay/encoder.pt").exists())

    def test_changed_final_results_leave_unverified_marker(self):
        run_id = "changed"
        content = self.repo / "seed.json"
        content.write_text("before")
        before = access.sha256(content) + "  ./metrics.json\n"
        after = "0" * 64 + "  ./metrics.json\n"

        def download(remote, local, wanted, name, **kwargs):
            (local / "metrics.json").write_text("before")

        responses = [self.result(stdout=before), self.result(stdout=after)]
        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "checked_ssh", side_effect=responses), \
                patch.object(self.client, "download_batch", side_effect=download):
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

        def download(remote, local, wanted, name, **kwargs):
            self.assertFalse(kwargs["wait"])
            (local / "metrics.json").write_text("data")

        with (
            patch.object(
                self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p),
            ) as validate,
            patch.object(self.client, "checked_ssh", return_value=self.result(stdout=text)) as snapshots,
            patch.object(self.client, "download_batch", side_effect=download),
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

    def remote_fixture(self, files):
        remote = self.repo / "remote-run"
        for relative, content in files.items():
            path = remote / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return remote

    def fake_rsync(self, remote, calls):
        """Stand-in for _rsync: copies the --files-from list from the local stand-in remote into the target."""
        def transfer(source, target, extra=None, wait=True, **options):
            calls.append((source, target, list(extra or []), wait, options.get("priority")))
            assert options["append"] is False and options["accept"] == (0, 23, 24)
            listing = Path([arg for arg in extra if arg.startswith("--files-from=")][0].split("=", 1)[1])
            for relative in listing.read_text().splitlines():
                destination = Path(target) / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((remote / relative).read_bytes())
        return transfer

    def test_batched_download_uses_one_connection_and_places_only_verified_files(self):
        remote = self.remote_fixture({"eval/step_1/a.png": b"a", "eval/step_1/b.png": b"b", "metrics.jsonl": b"m"})
        local = self.repo / "runs" / "batched"
        local.mkdir(parents=True)
        digest = lambda relative: access.sha256(remote / relative)  # noqa: E731
        wanted = {"eval/step_1/a.png": digest("eval/step_1/a.png"), "eval/step_1/b.png": digest("eval/step_1/b.png"),
                  "metrics.jsonl": "0" * 64}  # metrics.jsonl changed after the manifest: it must not be placed
        calls = []
        with patch.object(self.client, "_rsync", side_effect=self.fake_rsync(remote, calls)):
            placed = self.client.download_batch(str(remote), local, wanted, "batched-media", wait=False)
        self.assertEqual(len(calls), 1)
        source, target, extra, wait, priority = calls[0]
        self.assertEqual(priority, "dependency")
        self.assertTrue(source.endswith(str(remote) + "/") and not wait)
        self.assertEqual(Path(target), self.client.network_dir / "staging" / "batched-media")
        self.assertEqual(placed, ["eval/step_1/a.png", "eval/step_1/b.png"])
        self.assertEqual((local / "eval/step_1/b.png").read_bytes(), b"b")
        self.assertFalse((local / "metrics.jsonl").exists())
        self.assertTrue((self.client.network_dir / "staging" / "batched-media" / "metrics.jsonl").exists())
        with self.assertRaises(ValueError):
            self.client.download_batch(str(remote), local, {"x.png": "0" * 64}, "../escape")

    def test_batched_rsync_uses_delta_transfer_and_tolerates_vanished_files(self):
        local = self.repo / "runs" / "flags"
        local.mkdir(parents=True)
        with patch.object(self.client, "check_tunnel"), \
                patch.object(access.subprocess, "run", return_value=self.result(24)) as run:
            placed = self.client.download_batch("/remote/runs/flags", local, {"gone.png": "0" * 64}, "flags-media")
        args = run.call_args.args[0]
        self.assertEqual(run.call_count, 1)  # exit 24 (vanished files) is not retried or raised
        self.assertNotIn("--append-verify", args)  # a stale staged copy must not be skipped by append mode
        for flag in ("--partial", "--checksum", "--protect-args"):
            self.assertIn(flag, args)
        self.assertTrue(any(arg.startswith("--files-from=") for arg in args))
        self.assertEqual(placed, [])
        with patch.object(self.client, "check_tunnel"), \
                patch.object(access.subprocess, "run", return_value=self.result(23)):
            self.client.download_batch("/remote/runs/flags", local, {"gone.png": "0" * 64}, "flags-media")
        with patch.object(self.client, "check_tunnel"), \
                patch.object(access.subprocess, "run", return_value=self.result(5)):
            with self.assertRaises(access.RemoteCommandError):
                self.client.download_batch("/remote/runs/flags", local, {"gone.png": "0" * 64}, "flags-media")
        with patch.object(self.client, "check_tunnel"), \
                patch.object(access.subprocess, "run", return_value=self.result(24)):
            with self.assertRaises(access.RemoteCommandError):  # single-file transfers keep the strict behaviour
                self.client._rsync("explicit-source", "explicit-target")

    def test_essential_and_media_scopes_split_the_result_set_with_separate_records_and_locks(self):
        remote = self.remote_fixture({"metrics.jsonl": b"m", "encoder.pt": b"w", "eval/step_1/motion.png": b"p",
                                      "eval/step_1/summary.json": b"{}", "media/clip.mp4": b"v",
                                      "exports/encoder.pt": b"e", "exports/preview.png": b"i"})

        def manifest(command, **kwargs):
            return subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)

        calls = []
        with patch.object(self.client, "ensure_remote_path", return_value=str(remote)), \
                patch.object(self.client, "checked_ssh", side_effect=manifest), \
                patch.object(self.client, "_rsync", side_effect=self.fake_rsync(remote, calls)):
            essential = self.client.sync_results("split", final=True, scope="essential")
            self.assertEqual(set(essential["files"]), {"metrics.jsonl", "encoder.pt", "eval/step_1/summary.json",
                                                       "exports/encoder.pt"})
            self.assertFalse((self.repo / "runs/split/eval/step_1/motion.png").exists())
            media = self.client.sync_results("split", final=True, scope="media")
            self.assertEqual(set(media["files"]), {"eval/step_1/motion.png", "media/clip.mp4", "exports/preview.png"})
            everything = self.client.sync_results("split", final=True)
        # essential: one dependency batch; media: bulk chunks (the video alone, the images together); the "all" pass
        # finds everything already in place
        self.assertEqual([call[4] for call in calls], ["dependency", "bulk", "bulk"])
        self.assertEqual(set(everything["files"]), set(essential["files"]) | set(media["files"]))
        self.assertEqual((essential["scope"], media["scope"], everything["scope"]), ("essential", "media", "all"))
        record = json.loads((self.client.network_dir / "results-media" / "split.json").read_text())
        self.assertTrue(record["verified"] and record["final"] and record["scope"] == "media")
        self.assertEqual(json.loads((self.client.network_dir / "results" / "split.json").read_text())["scope"], "all")
        self.assertTrue((self.client.state_dir / "media-split.lock").exists())
        self.assertTrue((self.client.state_dir / "results-split.lock").exists())
        with self.assertRaises(ValueError):
            self.client.sync_results("split", scope="images")

    def test_final_media_sync_of_a_run_without_media_is_complete(self):
        with patch.object(self.client, "ensure_remote_path", side_effect=lambda p, **kw: self.client.remote_path(p)), \
                patch.object(self.client, "checked_ssh", return_value=self.result()):
            result = self.client.sync_results("no-media", final=True, scope="media")
            self.assertTrue(result["verified"] and result["final"] and result["files"] == {})
            with self.assertRaisesRegex(access.RemoteCommandError, "empty"):
                self.client.sync_results("no-media", final=True, scope="essential")

    def test_downloads_are_budgeted_and_rate_limited_while_uploads_are_not(self):
        with patch.object(self.client, "check_tunnel"), \
                patch.object(self.client, "_legacy_downloads", return_value=[]), \
                patch.object(access.subprocess, "run", return_value=self.result()) as run:
            self.client._rsync(f"{self.client.target}:/remote/runs/x/a.json", "local-target")
            self.assertIn("--bwlimit=600", run.call_args.args[0])
            self.assertIn("--timeout=300", run.call_args.args[0])
            with self.assertRaises(ValueError):  # callers cannot raise or bypass the budget
                self.client._rsync(f"{self.client.target}:/remote/runs/x/a.json", "local-target", extra=["--bwlimit=0"])
            self.client._rsync("local-source", f"{self.client.target}:/remote/runs/x/a.json")
            self.assertFalse(any(arg.startswith("--bwlimit") for arg in run.call_args.args[0]))
            (self.client.state_dir / "download-budget.json").write_text(json.dumps({"kib_per_s": 300}))
            self.client._rsync(f"{self.client.target}:/remote/runs/x/a.json", "local-target", priority="dependency")
            self.assertIn("--bwlimit=300", run.call_args.args[0])
        (self.client.state_dir / "download-budget.json").write_text(json.dumps({"kib_per_s": 0}))
        with self.assertRaises(ValueError):
            self.client.download_budget_kib()
        with self.assertRaises(ValueError):
            with self.client.download_slot("urgent"):
                pass

    def test_one_download_at_a_time_and_dependency_transfers_go_before_bulk(self):
        import threading
        order, held = [], threading.Event()
        with patch.object(self.client, "_legacy_downloads", return_value=[]):
            def bulk():
                with self.client.download_slot("bulk"):
                    order.append("bulk")

            def dependency():
                with self.client.download_slot("dependency"):
                    held.set()
                    order.append("dependency-start")
                    worker.start()
                    worker.join(0.5)
                    self.assertTrue(worker.is_alive())  # the budget is exclusive: bulk waits for this slot
                    order.append("dependency-end")

            worker = threading.Thread(target=bulk)
            dependency()
            worker.join(5)
            # a dependency transfer only announced (waiting for the budget) also holds bulk back
            with (self.client.state_dir / "download-priority.lock").open("a") as announce:
                access.fcntl.flock(announce, access.fcntl.LOCK_SH)
                second = threading.Thread(target=bulk)
                second.start()
                second.join(0.5)
                self.assertTrue(second.is_alive())
                access.fcntl.flock(announce, access.fcntl.LOCK_UN)
            second.join(5)
        self.assertEqual(order, ["dependency-start", "dependency-end", "bulk", "bulk"])

    def test_new_downloads_wait_for_transfers_started_without_a_limit(self):
        legacy = subprocess.Popen(["rsync", "-c", "sleep 30", f"{self.client.target}:/remote/runs/x/", "/local/x/"],
                                  executable="/bin/sh")
        self.addCleanup(legacy.kill)
        limited = subprocess.Popen(["rsync", "-c", "sleep 30", "--bwlimit=600", f"{self.client.target}:/remote/runs/y/",
                                    "/local/y/"], executable="/bin/sh")
        self.addCleanup(limited.kill)
        upload = subprocess.Popen(["rsync", "-c", "sleep 30", "/local/file", f"{self.client.target}:/remote/runs/z/"],
                                  executable="/bin/sh")
        self.addCleanup(upload.kill)
        found = self.client._legacy_downloads()
        self.assertIn(legacy.pid, found)
        self.assertNotIn(limited.pid, found)
        self.assertNotIn(upload.pid, found)  # uploads (remote destination) are not downloads
        waits = []
        with patch.object(self.client, "_legacy_downloads", side_effect=[[legacy.pid], [legacy.pid], []]), \
                patch.object(access.time, "sleep", side_effect=waits.append):
            with self.client.download_slot("bulk") as kib:
                self.assertEqual(kib, 600)
        self.assertEqual(waits, [5.0, 5.0])
        with patch.object(self.client, "_legacy_downloads", return_value=[legacy.pid]), \
                patch.object(access, "LEGACY_WAIT_SECONDS", 0.0), patch.object(access.time, "sleep"):
            with self.assertRaisesRegex(access.RemoteCommandError, "without a limit"):
                with self.client.download_slot("dependency"):
                    pass

    def test_bulk_batches_release_the_budget_between_short_chunks(self):
        local = self.repo / "runs" / "chunks"
        local.mkdir(parents=True)
        wanted = {f"eval/{i:03d}.png": "0" * 64 for i in range(100)}
        wanted.update({"checkpoints/step_1.pt": "0" * 64, "checkpoints/step_2.pt": "0" * 64, "media/clip.mp4": "0" * 64})
        listings = []

        def transfer(source, target, extra=None, wait=True, **options):
            listing = Path([arg for arg in extra if arg.startswith("--files-from=")][0].split("=", 1)[1])
            listings.append((options["priority"], listing.read_text().splitlines()))

        with patch.object(self.client, "_rsync", side_effect=transfer):
            self.client.download_batch("/remote/runs/chunks", local, wanted, "chunks-media", priority="bulk")
            self.client.download_batch("/remote/runs/chunks", local, wanted, "chunks-essential")
        bulk = [files for priority, files in listings if priority == "bulk"]
        self.assertEqual([len(files) for files in bulk], [1, 1, 1, access.BULK_CHUNK_FILES, access.BULK_CHUNK_FILES, 4])
        self.assertEqual(sorted(f for files in bulk for f in files), sorted(wanted))
        self.assertEqual([(priority, len(files)) for priority, files in listings][-1], ("dependency", len(wanted)))

    def test_final_essential_sync_can_take_only_the_required_checkpoints(self):
        remote = self.remote_fixture({"metrics.jsonl": b"m", "checkpoints/step_0100000.pt": b"a",
                                      "checkpoints/step_0200000.pt": b"b", "snapshots/latest.pt": b"s",
                                      "eval/step_1/motion.png": b"p"})

        def manifest(command, **kwargs):
            return subprocess.run(["sh", "-c", command], capture_output=True, text=True, check=True)

        calls = []
        with patch.object(self.client, "ensure_remote_path", return_value=str(remote)), \
                patch.object(self.client, "checked_ssh", side_effect=manifest), \
                patch.object(self.client, "_rsync", side_effect=self.fake_rsync(remote, calls)):
            needed = self.client.sync_results("weights", checkpoints=True, final=True, scope="essential",
                                              only_checkpoints=["checkpoints/step_0200000.pt"])
            rest = self.client.sync_results("weights", final=True, scope="checkpoints")
            with self.assertRaises(ValueError):
                self.client.sync_results("weights", checkpoints=True, final=True, scope="essential",
                                         only_checkpoints=["../outside.pt"])
            with self.assertRaises(ValueError):
                self.client.sync_results("weights", final=True, scope="media", only_checkpoints=["checkpoints/a.pt"])
        self.assertEqual(set(needed["files"]), {"metrics.jsonl", "checkpoints/step_0200000.pt"})
        self.assertEqual(needed["checkpoints"], "required")
        self.assertEqual(set(rest["files"]), {"checkpoints/step_0100000.pt", "checkpoints/step_0200000.pt",
                                              "snapshots/latest.pt"})
        self.assertEqual([call[4] for call in calls], ["dependency", "bulk", "bulk"])  # step_0200000 already local
        record = json.loads((self.client.network_dir / "results-checkpoints" / "weights.json").read_text())
        self.assertTrue(record["verified"] and record["final"] and record["scope"] == "checkpoints")

    def test_bulk_backs_off_when_a_dependency_is_announced_after_it_won_the_budget(self):
        import threading
        calls = {"n": 0}
        real_flock = access.fcntl.flock
        announced = (self.client.state_dir / "download-priority.lock").open("a")
        self.addCleanup(announced.close)

        def flock(handle, operation):
            # the first time bulk tries the non-blocking announce check, a dependency has just announced itself
            if operation == access.fcntl.LOCK_EX | access.fcntl.LOCK_NB and calls["n"] == 0:
                calls["n"] += 1
                real_flock(announced, access.fcntl.LOCK_SH)
                threading.Timer(0.3, lambda: real_flock(announced, access.fcntl.LOCK_UN)).start()
            return real_flock(handle, operation)

        with patch.object(self.client, "_legacy_downloads", return_value=[]), \
                patch.object(access.fcntl, "flock", side_effect=flock):
            started = access.time.monotonic()
            with self.client.download_slot("bulk"):
                waited = access.time.monotonic() - started
        self.assertEqual(calls["n"], 1)
        self.assertGreaterEqual(waited, 0.25)  # it released the budget and waited for the announced dependency


if __name__ == "__main__":
    unittest.main()
