from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import DEFAULT_ROBOMIMIC_PYTHON, ISAACLAB_ROOT
from .policy_observations import PolicyMetadata

ROBOMIMIC_IMPORT_CHECK = "import robomimic, torch"


def guess_latest_checkpoint() -> str:
    """Return the newest robomimic checkpoint under IsaacLab logs.

    There are no inputs; the output is a checkpoint path string or empty string.
    This exists to make the extension useful out of the box when the user has
    recent local training runs.
    """

    checkpoint_paths = []
    logs_root = ISAACLAB_ROOT / "logs" / "robomimic"
    if logs_root.exists():
        checkpoint_paths = sorted(logs_root.rglob("*.pth"), key=lambda path: path.stat().st_mtime, reverse=True)
    return str(checkpoint_paths[0]) if checkpoint_paths else ""


def iter_robomimic_python_candidates(configured_python: str | None) -> list[str]:
    """Build a de-duplicated list of Python interpreters to test.

    Input is an optional configured interpreter path; the output is candidate
    paths. This exists so users can configure a specific env while the extension
    can still discover common Isaac/host Python locations.
    """

    if configured_python:
        return [configured_python]

    repo_root = Path(__file__).resolve().parents[4]
    raw_candidates = []
    for pattern in (
        "_build/*/release/python.sh",
        "_build/*/debug/python.sh",
        "_build/*/release/kit/python/bin/python3",
        "_build/*/debug/kit/python/bin/python3",
    ):
        raw_candidates.extend(str(path) for path in sorted(repo_root.glob(pattern)))

    current_executable = Path(sys.executable)
    if current_executable.name.startswith("python"):
        raw_candidates.append(str(current_executable))

    for executable_name in ("python3", "python"):
        resolved_path = shutil.which(executable_name)
        if resolved_path:
            raw_candidates.append(resolved_path)

    candidates = []
    seen = set()
    for candidate in raw_candidates:
        normalized_candidate = str(Path(candidate).expanduser())
        if normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        candidates.append(normalized_candidate)
    return candidates


@lru_cache(maxsize=8)
def resolve_robomimic_python_executable(configured_python: str | None) -> str:
    """Find a Python executable that can import torch and robomimic.

    Input is an optional configured interpreter path; the output is an executable
    path. This exists to fail fast with diagnostic context before launching the
    long-lived inference subprocess.
    """

    failures = []
    for candidate in iter_robomimic_python_candidates(configured_python):
        try:
            completed = subprocess.run(
                [candidate, "-c", ROBOMIMIC_IMPORT_CHECK],
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except FileNotFoundError:
            reason = "interpreter was not found"
        except OSError as exc:
            reason = str(exc)
        except subprocess.TimeoutExpired:
            reason = "module check timed out"
        else:
            if completed.returncode == 0:
                return candidate
            reason = completed.stderr.strip() or completed.stdout.strip() or f"returned exit code {completed.returncode}"

        failures.append(f"{candidate} ({reason})")
        if configured_python:
            break

    if configured_python:
        raise RuntimeError(
            "Configured robomimic Python is missing dependencies. "
            f"Checked {configured_python}: {failures[0] if failures else 'unknown error'}"
        )

    checked_paths = ", ".join(failures) if failures else "no candidates found"
    raise RuntimeError(
        "Could not find a Python interpreter with both torch and robomimic installed. "
        "Set IILAB_ROBOMIMIC_PYTHON to a working interpreter if needed. "
        f"Checked: {checked_paths}"
    )


@dataclass
class RobomimicInferenceWorker:
    """Line-delimited JSON client for the robomimic subprocess worker."""

    checkpoint_path: str
    norm_factor_min: float | None = None
    norm_factor_max: float | None = None
    python_executable: str = DEFAULT_ROBOMIMIC_PYTHON
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _metadata: PolicyMetadata | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start the robomimic subprocess and read policy metadata.

        There are no inputs beyond instance fields; there is no output.
        This exists to isolate ML dependencies from the Isaac process and avoid
        loading robomimic/torch directly into the extension runtime.
        """

        worker_script_path = Path(__file__).with_name("robomimic_worker.py")
        resolved_python_executable = resolve_robomimic_python_executable(self.python_executable)
        command = [resolved_python_executable, str(worker_script_path), "--checkpoint", self.checkpoint_path]
        if self.norm_factor_min is not None and self.norm_factor_max is not None:
            command.extend(
                ["--norm-factor-min", str(self.norm_factor_min), "--norm-factor-max", str(self.norm_factor_max)]
            )

        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            ready_message = self._read_message()
            if ready_message.get("status") != "ready":
                raise RuntimeError(ready_message.get("message", "Robomimic worker failed to start."))
            self._metadata = PolicyMetadata.from_worker_message(ready_message)
        except Exception:
            self.close()
            raise

    def reset_episode(self) -> None:
        """Tell the worker to reset policy recurrent/episode state.

        There are no inputs or outputs. This exists to keep simulation resets and
        robomimic policy episode boundaries aligned.
        """

        self._request({"cmd": "reset"})

    def infer(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Run one policy inference request in the worker subprocess.

        Input is an observation dict; the output is a flat action array.
        This exists as the only synchronous inference boundary used by the
        simulation controller.
        """

        response = self._request({"cmd": "infer", "obs": {key: value.tolist() for key, value in observation.items()}})
        action = np.asarray(response.get("action", []), dtype=np.float32)
        if action.ndim != 1:
            action = action.reshape(-1)
        return action

    def close(self) -> None:
        """Close or terminate the worker subprocess.

        There are no inputs or outputs. This exists so policy stops, errors, and
        extension shutdown release subprocess resources reliably.
        """

        if self._process is None:
            return

        try:
            self._request({"cmd": "close"})
        except Exception:
            pass

        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

        self._process = None
        self._metadata = None

    @property
    def metadata(self) -> PolicyMetadata:
        """Return metadata reported by a running worker.

        There are no inputs; the output is PolicyMetadata.
        This exists to prevent callers from using metadata before the worker is
        started successfully.
        """

        if self._metadata is None:
            raise RuntimeError("Robomimic worker metadata is not available because the worker is not running.")
        return self._metadata

    def _request(self, payload: dict) -> dict:
        """Send one JSON command to the worker and read the JSON response.

        Input is a payload dictionary; the output is a response dictionary.
        This exists to keep line-delimited JSON protocol details out of policy
        controller code.
        """

        process = self._require_process()
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()
        return self._read_message()

    def _read_message(self) -> dict:
        """Read the next JSON worker message, preserving diagnostics.

        There are no inputs; the output is a decoded message dictionary.
        This exists to tolerate library stdout chatter while still reporting the
        last skipped lines when the worker fails.
        """

        process = self._require_process()
        assert process.stdout is not None
        skipped_output_lines: list[str] = []

        while True:
            line = process.stdout.readline()
            if not line:
                stderr_output = ""
                if process.poll() is not None and process.stderr is not None:
                    stderr_output = process.stderr.read().strip()

                diagnostic_output = "\n".join(skipped_output_lines).strip()
                if stderr_output and diagnostic_output:
                    raise RuntimeError(f"{stderr_output}\nWorker stdout:\n{diagnostic_output}")
                if stderr_output:
                    raise RuntimeError(stderr_output)
                if diagnostic_output:
                    raise RuntimeError(f"Robomimic worker exited unexpectedly.\nWorker stdout:\n{diagnostic_output}")
                raise RuntimeError("Robomimic worker exited unexpectedly.")

            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                message = json.loads(stripped_line)
            except json.JSONDecodeError:
                skipped_output_lines.append(stripped_line)
                skipped_output_lines = skipped_output_lines[-20:]
                continue

            if message.get("status") == "error":
                error_message = message.get("message", "Robomimic worker reported an error.")
                diagnostic_output = "\n".join(skipped_output_lines).strip()
                if diagnostic_output:
                    error_message = f"{error_message}\nWorker stdout:\n{diagnostic_output}"
                raise RuntimeError(error_message)
            return message

    def _require_process(self) -> subprocess.Popen[str]:
        """Return the running subprocess or raise a clear error.

        There are no inputs; the output is the Popen instance.
        This exists so protocol helpers fail consistently when the worker has not
        been started or was already closed.
        """

        if self._process is None:
            raise RuntimeError("Robomimic worker is not running.")
        return self._process
