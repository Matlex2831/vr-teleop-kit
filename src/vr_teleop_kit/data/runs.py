"""Launch and track long-running jobs (training, export, rollout).

Training an ACT policy takes hours; a rollout moves a real arm. Neither
belongs inside a web request, and neither should die because the relay
was restarted to pick up a code change. So every job is a DETACHED
subprocess (`start_new_session=True`, its own process group) that owns a
directory:

    <runs_dir>/<run_id>/
        spec.json      what was asked for
        run.json       pid, argv, status, timestamps
        run.log        stdout + stderr, appended live
        metrics.jsonl  one JSON object per logged step (training only)
        train/         the job's own output_dir (checkpoints)

The page reattaches by reading those files, so state survives a relay
restart with the full log and metric history intact.

Exit status is written by the runner itself into `result.json`, not
inferred by us: after a relay restart the server is no longer the
process's parent and cannot reap it, so "pid is gone" alone cannot tell
a clean finish from a crash. A dead pid with no `result.json` is
reported as `died`, which is exactly what it means.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Runner modules, by job kind. Each takes a single argument: the path to
# its spec.json.
RUNNERS = {
    "record": "vr_teleop_kit.data.record_runner",
    "train": "vr_teleop_kit.data.train_runner",
    "export": "vr_teleop_kit.data.export_runner",
    "rollout": "vr_teleop_kit.data.rollout_runner",
}

# Seconds to wait for a clean SIGINT shutdown before escalating. LeRobot's
# ProcessSignalHandler and the training loop both handle KeyboardInterrupt,
# and training needs a moment to finish writing a checkpoint.
STOP_GRACE_S = 20.0


def runs_dir() -> Path:
    """Where run directories live.

    `VR_TELEOP_RUNS` overrides; otherwise `./outputs/runs` relative to
    the process's working directory, which is the repo root in every
    documented way of starting the relay. The resolved path is reported
    by `/api/env` so it is never a guess.
    """
    env = os.environ.get("VR_TELEOP_RUNS")
    base = Path(env).expanduser() if env else Path.cwd() / "outputs" / "runs"
    return base


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def new_run_id(kind: str, name: str = "") -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return f"{kind}-{stamp}" + (f"-{slug}" if slug else "")


def run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"bad run id: {run_id!r}")
    return runs_dir() / run_id


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _pid_alive(pid: int, run_id: str) -> bool:
    """Alive AND still the process we launched.

    Checking only `kill(pid, 0)` would call a recycled pid our training
    job — on a workstation that spawns processes all day, that is a real
    way to show a finished run as running forever.
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return True  # no procfs: fall back to the kill(0) answer
    return run_id in cmdline


def launch(kind: str, spec: dict, name: str = "", env: dict | None = None) -> dict:
    """Start a job. Returns its run record."""
    if kind not in RUNNERS:
        raise ValueError(f"unknown job kind {kind!r}")
    run_id = new_run_id(kind, name)
    rdir = run_dir(run_id)
    rdir.mkdir(parents=True, exist_ok=True)

    spec = dict(spec)
    spec["run_id"] = run_id
    spec["run_dir"] = str(rdir)
    spec_path = rdir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    argv = [sys.executable, "-m", RUNNERS[kind], str(spec_path)]
    log_path = rdir / "run.log"

    record = {
        "id": run_id,
        "kind": kind,
        "name": name,
        "spec": spec,
        "argv": argv,
        "cwd": str(Path.cwd()),
        "started_at": _now(),
        "status": "running",
        "pid": None,
    }

    child_env = dict(os.environ)
    if env:
        child_env.update({k: str(v) for k, v in env.items()})
    # Unbuffered, so the log pane shows progress as it happens instead of
    # in 4 KB bursts.
    child_env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(log_path, "ab") as log:
            log.write(f"$ {' '.join(argv)}\n".encode())
            log.flush()
            proc = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(Path.cwd()),
                env=child_env,
                # Own session/process group: the job outlives the relay,
                # and stop() can signal the whole group without ever
                # reaching the server itself.
                start_new_session=True,
            )
    except Exception as e:
        record["status"] = "launch_failed"
        record["error"] = str(e)
        (rdir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        raise

    record["pid"] = proc.pid
    (rdir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load(run_id: str) -> dict:
    """Run record with its live status resolved."""
    rdir = run_dir(run_id)
    record = _read_json(rdir / "run.json")
    if record is None:
        raise FileNotFoundError(f"no run {run_id}")
    result = _read_json(rdir / "result.json")
    alive = _pid_alive(int(record.get("pid") or 0), run_id)

    if result is not None:
        record["status"] = result.get("status", "done")
        record["exit_code"] = result.get("exit_code")
        record["finished_at"] = result.get("finished_at")
        record["error"] = result.get("error") or record.get("error")
    elif alive:
        record["status"] = "running"
    elif record.get("status") == "running":
        # Pid gone, no result written: killed hard, OOM, or a crash the
        # runner could not survive long enough to report.
        record["status"] = "died"

    record["alive"] = alive
    log = rdir / "run.log"
    record["log_size"] = log.stat().st_size if log.exists() else 0
    metrics = rdir / "metrics.jsonl"
    record["metrics_size"] = metrics.stat().st_size if metrics.exists() else 0
    record["output_dir"] = str(rdir / "train")
    return record


def list_runs(limit: int = 100) -> list[dict]:
    base = runs_dir()
    if not base.exists():
        return []
    out = []
    for rdir in base.iterdir():
        if not rdir.is_dir() or not (rdir / "run.json").exists():
            continue
        try:
            out.append(load(rdir.name))
        except Exception:
            continue
    # Newest first by start time, not by directory name: run ids are
    # prefixed with the kind, so sorting by name would group every
    # rollout above every train regardless of when they ran.
    out.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return out[:limit]


def stop(run_id: str) -> dict:
    """SIGINT the job's process group, escalating to SIGTERM.

    SIGINT first because LeRobot's training loop and `ProcessSignalHandler`
    both treat it as "wind down": training saves a checkpoint, a rollout
    returns the arm to its initial pose. SIGKILL is never sent from here —
    a half-killed rollout leaves an arm under torque in an unknown pose.
    """
    record = load(run_id)
    pid = int(record.get("pid") or 0)
    if not record.get("alive"):
        return record
    try:
        os.killpg(os.getpgid(pid), signal.SIGINT)
    except (ProcessLookupError, PermissionError, OSError) as e:
        record["error"] = f"could not signal: {e}"
        return record

    deadline = time.time() + STOP_GRACE_S
    while time.time() < deadline:
        time.sleep(0.25)
        if not _pid_alive(pid, run_id):
            break
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass

    record = load(run_id)
    record["stop_requested"] = True
    return record


def tail_log(run_id: str, offset: int = 0, max_bytes: int = 200_000) -> dict:
    """Incremental read. The page passes back the offset it last saw, so
    a multi-hour log is never re-sent."""
    path = run_dir(run_id) / "run.log"
    if not path.exists():
        return {"offset": 0, "text": "", "size": 0}
    size = path.stat().st_size
    offset = max(0, min(int(offset), size))
    # A client that fell far behind gets the tail, not a 50 MB response.
    if size - offset > max_bytes:
        offset = size - max_bytes
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
    return {"offset": offset + len(data), "size": size, "text": data.decode("utf-8", "replace")}


def read_metrics(run_id: str, offset: int = 0) -> dict:
    """Incremental JSONL read. Only whole lines are consumed, so a record
    caught mid-write is picked up on the next poll rather than dropped."""
    path = run_dir(run_id) / "metrics.jsonl"
    if not path.exists():
        return {"offset": 0, "rows": [], "size": 0}
    size = path.stat().st_size
    offset = max(0, min(int(offset), size))
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    text = data.decode("utf-8", "replace")
    consumed = len(data)
    if text and not text.endswith("\n"):
        cut = text.rfind("\n")
        if cut == -1:
            return {"offset": offset, "rows": [], "size": size}
        consumed = len(text[: cut + 1].encode("utf-8"))
        text = text[: cut + 1]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return {"offset": offset + consumed, "size": size, "rows": rows}


def checkpoints(run_id: str) -> list[dict]:
    """Saved checkpoints of a training run, newest step first.

    LeRobot writes `<output_dir>/checkpoints/<step>/pretrained_model` plus
    a `last` symlink; the rollout launcher wants the model directory, so
    that is what is returned.
    """
    out_dir = run_dir(run_id) / "train" / "checkpoints"
    if not out_dir.exists():
        return []
    found = []
    for entry in out_dir.iterdir():
        model = entry / "pretrained_model"
        if not model.is_dir():
            continue
        found.append({
            "name": entry.name,
            "step": int(entry.name) if entry.name.isdigit() else None,
            "path": str(model),
            "is_link": entry.is_symlink(),
            "modified": entry.stat().st_mtime,
        })
    found.sort(key=lambda c: (c["step"] is None, -(c["step"] or 0), c["name"]))
    return found


def write_result(run_dir_path: str | Path, status: str, exit_code: int = 0, error: str = "") -> None:
    """Called by a runner in its `finally` — this is what makes a
    finished run distinguishable from a killed one."""
    payload = {
        "status": status,
        "exit_code": exit_code,
        "error": error,
        "finished_at": _now(),
    }
    Path(run_dir_path, "result.json").write_text(json.dumps(payload, indent=2) + "\n")
