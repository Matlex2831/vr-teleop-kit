"""HTTP API behind the `/data` page.

Mounted on the relay's FastAPI app, which is also the real-time teleop
path. Two consequences shape this module:

  1. Every handler that touches parquet is a plain `def`, not `async
     def`. FastAPI runs those in a worker thread, so a multi-megabyte
     read can never stall the event loop that forwards `xr_frame`
     messages to the arm.

  2. Anything that STARTS a process is refused for non-loopback clients
     unless the relay was started with `--allow-remote-control`. The
     relay is documented to run with `--host 0.0.0.0` so a Quest can
     reach it over the LAN, and "launch a training job" / "drive the
     arm" are not things that should come with that.

Repo-ids are passed as query parameters rather than path segments: they
contain a slash, and a `{repo_id:path}` route would shadow every
sub-resource under it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import FileResponse

from . import catalog, review as review_mod, runs

router = APIRouter(prefix="/api")

# Set by the relay's main(): allows POSTs from non-loopback clients.
ALLOW_REMOTE_CONTROL = False

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def set_allow_remote_control(value: bool) -> None:
    global ALLOW_REMOTE_CONTROL
    ALLOW_REMOTE_CONTROL = bool(value)


def _require_local(request: Request) -> None:
    if ALLOW_REMOTE_CONTROL:
        return
    host = request.client.host if request.client else None
    if host not in _LOOPBACK:
        raise HTTPException(
            status_code=403,
            detail=(
                f"launching jobs is limited to this machine (request came from {host}). "
                "Restart the relay with --allow-remote-control to permit it from the LAN."
            ),
        )


def _wrap(fn, *args, **kwargs):
    """Turn the module's exceptions into HTTP answers a human can act on."""
    try:
        return fn(*args, **kwargs)
    except catalog.DataDependencyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except catalog.DatasetBusyError as e:
        # 409, not 500: nothing is broken, the dataset is just still being
        # written. The page shows this as a banner, not an error.
        raise HTTPException(status_code=409, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - unexpected
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


# ── environment ──────────────────────────────────────────────────────────

_gpu_cache: list | None = None


def _gpus() -> list[str]:
    """GPU names via nvidia-smi rather than torch: importing torch into
    the relay process would cost seconds and hundreds of MB for a string
    the page shows once."""
    global _gpu_cache
    if _gpu_cache is not None:
        return _gpu_cache
    names: list[str] = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            names = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            names = []
    _gpu_cache = names
    return names


def _port_holders(device: str) -> list[dict]:
    """Which processes hold a device open.

    The serial bus is a single-holder resource: `haller_hmi` sitting on
    /dev/ttyACM0 with torque enabled is exactly the kind of thing that
    turns a rollout launch into corrupted Feetech packets, and it is
    much friendlier to say so before the arm is involved.
    """
    holders = []
    try:
        target = os.path.realpath(device)
    except OSError:
        return holders
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        fd_dir = proc / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    if os.path.realpath(fd) == target:
                        cmdline = (proc / "cmdline").read_bytes().decode("utf-8", "replace")
                        holders.append({
                            "pid": int(proc.name),
                            "cmd": cmdline.replace("\x00", " ").strip()[:200],
                        })
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return holders


@router.get("/env")
def get_env(request: Request) -> dict:
    """What the page needs for preflight warnings."""
    # Imported here, not at module scope: the relay server imports this
    # module, so a top-level import back into it would be circular.
    try:
        from ..relay import server as relay_server
        relay_cameras = [
            {"id": s.id, "label": s.label, "device": s.device}
            for s in relay_server.CAMERA_SPECS
        ]
    except Exception:
        relay_cameras = []

    default_port = os.environ.get("SO101_PORT", "/dev/ttyACM0")
    host = request.client.host if request.client else None
    return {
        "hf_home": str(catalog.hf_home()),
        "runs_dir": str(runs.runs_dir()),
        "cwd": os.getcwd(),
        "relay_cameras": relay_cameras,
        "gpus": _gpus(),
        "serial": {
            "port": default_port,
            "exists": Path(default_port).exists(),
            "holders": _port_holders(default_port) if Path(default_port).exists() else [],
        },
        "client_host": host,
        "is_local": host in _LOOPBACK,
        "allow_remote_control": ALLOW_REMOTE_CONTROL,
        "urdf": _resolved_urdf(),
    }


# ── datasets ─────────────────────────────────────────────────────────────

def _resolved_urdf() -> str:
    """The URDF the recorder would actually load — SO101_URDF if set,
    otherwise whatever the conventional-location search turns up."""
    try:
        from ..ik.so101_model import resolve_so101_urdf_path
        return str(resolve_so101_urdf_path())
    except Exception:
        return ""


@router.get("/datasets")
def get_datasets() -> dict:
    return {"datasets": _wrap(catalog.list_datasets)}


@router.get("/dataset")
def get_dataset(repo_id: str = Query(...)) -> dict:
    return _wrap(catalog.dataset_detail, repo_id)


@router.get("/dataset/trace")
def get_trace(repo_id: str = Query(...), episode: int = Query(...)) -> dict:
    return _wrap(catalog.episode_trace, repo_id, episode)


@router.get("/dataset/video")
def get_video(
    repo_id: str = Query(...),
    key: str = Query(...),
    chunk: int = Query(0),
    file: int = Query(0),
) -> FileResponse:
    path = _wrap(catalog.video_path, repo_id, key, chunk, file)
    # Starlette's FileResponse answers Range requests, which is what lets
    # the player seek straight to an episode inside the concatenated mp4
    # instead of downloading the whole file first.
    return FileResponse(path, media_type="video/mp4")


@router.get("/dataset/split")
def get_split(
    repo_id: str = Query(...),
    eval_split: float = Query(0.0),
    seed: int = Query(42),
    mode: str = Query("random"),
) -> dict:
    """Which kept episodes will be held out for eval loss.

    Computed here rather than in the page so there is exactly one
    implementation of the rule the trainer will actually follow.
    """
    def _apply():
        detail = catalog.dataset_detail(repo_id)
        return catalog.plan_eval_split(
            detail["episodes"], detail["keep_list"], eval_split, seed, mode)

    return _wrap(_apply)


@router.post("/dataset/review")
def post_review(payload: dict = Body(...)) -> dict:
    repo_id = payload.get("repo_id")
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")

    def _apply():
        root = catalog.dataset_root(repo_id)
        detail = catalog.dataset_detail(repo_id)
        episode = int(payload["episode"])
        review_mod.set_status(
            root,
            episode,
            str(payload.get("status", review_mod.UNSET)),
            payload.get("note"),
            fingerprint=detail["fingerprint"],
            # Stamp the episode's length so this mark stays verifiable
            # if the dataset is ever pruned and renumbered.
            episode_frames=detail["episode_frames"].get(episode),
        )
        return catalog.dataset_detail(repo_id)

    return _wrap(_apply)


@router.post("/dataset/review/bulk")
def post_review_bulk(payload: dict = Body(...)) -> dict:
    """Apply many marks at once: the auto-grade action, or a reset."""
    repo_id = payload.get("repo_id")
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    action = payload.get("action")

    def _apply():
        root = catalog.dataset_root(repo_id)
        detail = catalog.dataset_detail(repo_id)
        fingerprint = detail["fingerprint"]
        if action == "clear":
            review_mod.clear(root, fingerprint=fingerprint)
        elif action == "auto":
            # Only FAILs are auto-rejected. SUSPECT means "look at this",
            # and a heuristic that quietly discarded a struggle-but-
            # successful demonstration would be making the operator's
            # judgement call for them.
            marks = {
                ep["episode_index"]: (
                    review_mod.REJECT if ep["verdict"] == "FAIL" else review_mod.KEEP
                )
                for ep in detail["episodes"]
                if ep["verdict"] in ("FAIL", "PASS")
            }
            review_mod.set_many(root, marks, fingerprint=fingerprint,
                                episode_frames=detail["episode_frames"])
        else:
            marks = {int(k): str(v) for k, v in (payload.get("marks") or {}).items()}
            if not marks:
                raise ValueError("nothing to apply: pass marks, or action=auto|clear")
            review_mod.set_many(root, marks, fingerprint=fingerprint,
                                episode_frames=detail["episode_frames"])
        return catalog.dataset_detail(repo_id)

    return _wrap(_apply)


@router.post("/dataset/export")
def post_export(request: Request, payload: dict = Body(...)) -> dict:
    """Write a pruned copy as a background job (it re-encodes video)."""
    _require_local(request)
    repo_id = payload.get("repo_id")
    new_repo_id = (payload.get("new_repo_id") or "").strip()
    if not repo_id or not new_repo_id:
        raise HTTPException(status_code=400, detail="repo_id and new_repo_id are required")

    def _apply():
        detail = catalog.dataset_detail(repo_id)
        keep = set(detail["keep_list"])
        drop = [e["episode_index"] for e in detail["episodes"] if e["episode_index"] not in keep]
        if not drop:
            raise ValueError("no episodes are rejected — the copy would be identical")
        if len(keep) == 0:
            raise ValueError("every episode is rejected — there would be nothing to export")
        return runs.launch(
            "export",
            {
                "mode": "copy",
                "repo_id": repo_id,
                "new_repo_id": new_repo_id,
                "delete_episodes": drop,
            },
            name=new_repo_id.replace("/", "-"),
        )

    return _wrap(_apply)


@router.post("/dataset/delete_episodes")
def post_delete_episodes(request: Request, payload: dict = Body(...)) -> dict:
    """Permanently drop the rejected episodes from the dataset itself.

    Unlike the export above, this rewrites the dataset in place. LeRobot
    keeps the previous version as `<name>_old` unless `keep_backup` is
    false, which is the one path here with nothing to fall back on — the
    UI requires a typed confirmation for it.
    """
    _require_local(request)
    repo_id = payload.get("repo_id")
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")

    def _apply():
        detail = catalog.dataset_detail(repo_id)
        keep = set(detail["keep_list"])
        drop = [e["episode_index"] for e in detail["episodes"] if e["episode_index"] not in keep]
        if not drop:
            raise ValueError("no episodes are rejected — nothing would be deleted")
        if not keep:
            raise ValueError(
                "every episode is rejected — a dataset cannot be emptied this way. "
                "Delete the whole dataset directory instead if that is what you want."
            )
        # The client sends back what it believes it is deleting; if the
        # dataset changed since the page loaded, refuse rather than delete
        # a different set of episodes than the one that was confirmed.
        expected = payload.get("expect_episodes")
        if expected is not None and sorted(int(e) for e in expected) != sorted(drop):
            raise ValueError(
                f"the rejected set changed since you confirmed (now {sorted(drop)}). "
                "Reload the page and check the marks before deleting."
            )
        return runs.launch(
            "export",
            {
                "mode": "in_place",
                "repo_id": repo_id,
                "delete_episodes": drop,
                "keep_backup": bool(payload.get("keep_backup", True)),
            },
            name="prune-" + repo_id.replace("/", "-"),
        )

    return _wrap(_apply)


@router.post("/dataset/rename")
def post_rename(request: Request, payload: dict = Body(...)) -> dict:
    """Rename a dataset. In v3.0 the directory path IS the identity — no
    repo_id is stored inside meta/ — so this is a move, and any `_old`
    backup moves with it."""
    _require_local(request)
    repo_id = payload.get("repo_id")
    new_repo_id = payload.get("new_repo_id")
    if not repo_id or not new_repo_id:
        raise HTTPException(status_code=400, detail="repo_id and new_repo_id are required")

    def _apply():
        _refuse_if_busy(repo_id, "rename")
        return catalog.rename_dataset(repo_id, new_repo_id)

    return _wrap(_apply)


@router.post("/dataset/delete")
def post_delete_dataset(request: Request, payload: dict = Body(...)) -> dict:
    """Delete a whole dataset directory. Irreversible, so the caller must
    echo the dataset's own name back — a stray click cannot satisfy it."""
    _require_local(request)
    repo_id = payload.get("repo_id")
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    if (payload.get("confirm") or "").strip() != repo_id:
        raise HTTPException(
            status_code=400,
            detail="confirm must repeat the dataset name exactly",
        )

    def _apply():
        _refuse_if_busy(repo_id, "delete")
        return catalog.delete_dataset(repo_id)

    return _wrap(_apply)


def _refuse_if_busy(repo_id: str, verb: str) -> None:
    """Block moving or removing a dataset something is still writing to.

    A recording session appends to its dataset for as long as it runs;
    renaming the directory out from under it loses the session.
    """
    for run in runs.list_runs(limit=50):
        if run.get("status") != "running":
            continue
        spec = run.get("spec") or {}
        if spec.get("repo_id") == repo_id:
            raise ValueError(
                f"cannot {verb} {repo_id}: run {run['id']} ({run['kind']}) is using it. "
                "Stop that run first."
            )


# ── cameras ──────────────────────────────────────────────────────────────

_camera_cache: dict | None = None


@router.get("/cameras")
def get_cameras(refresh: bool = Query(False)) -> dict:
    """Cameras available for recording.

    Discovery runs in a short-lived subprocess: finding RealSense devices
    means importing pyrealsense2, and the relay is the teleop latency
    path — it does not import that, lerobot, or torch.
    """
    global _camera_cache
    if _camera_cache is not None and not refresh:
        return _camera_cache
    try:
        out = subprocess.run(
            [sys.executable, "-m", "vr_teleop_kit.data.record_runner", "--list-cameras"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        _camera_cache = json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        _camera_cache = {"cameras": [], "error": f"{type(e).__name__}: {e}"}
    # Which of them this relay is holding open right now — a camera the
    # relay has opened cannot also be recorded from.
    _camera_cache["relay_open"] = _relay_open_cameras()
    return _camera_cache


def _relay_open_cameras() -> list[dict]:
    """Cameras the relay has actually OPENED (not merely configured).

    The relay opens its v4l2 devices lazily, on the first WebRTC request,
    so a configured camera is only a conflict once someone has actually
    turned streaming on in the headset.
    """
    try:
        from ..relay import server as relay_server
        specs = {s.id: s for s in relay_server.CAMERA_SPECS}
        return [
            {"id": cid, "device": specs[cid].device}
            for cid in relay_server.CAMERA_READERS
            if cid in specs
        ]
    except Exception:
        return []


# ── runs ─────────────────────────────────────────────────────────────────

@router.get("/runs")
def get_runs(limit: int = Query(100)) -> dict:
    return {"runs": _wrap(runs.list_runs, limit)}


@router.get("/run")
def get_run(id: str = Query(...)) -> dict:
    return _wrap(runs.load, id)


@router.get("/run/log")
def get_run_log(id: str = Query(...), offset: int = Query(0)) -> dict:
    return _wrap(runs.tail_log, id, offset)


@router.get("/run/metrics")
def get_run_metrics(id: str = Query(...), offset: int = Query(0)) -> dict:
    return _wrap(runs.read_metrics, id, offset)


@router.get("/run/checkpoints")
def get_run_checkpoints(id: str = Query(...)) -> dict:
    return {"checkpoints": _wrap(runs.checkpoints, id)}


@router.post("/run/train")
def post_train(request: Request, payload: dict = Body(...)) -> dict:
    _require_local(request)
    repo_id = payload.get("repo_id")
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")

    def _apply():
        detail = catalog.dataset_detail(repo_id)
        # Train on the kept set unless the caller overrides it explicitly.
        episodes = payload.get("episodes")
        eval_split = float(payload.get("eval_split", 0.0))
        eval_mode = payload.get("eval_mode", "random")
        eval_seed = int(payload.get("eval_seed", 42))
        plan = catalog.plan_eval_split(
            detail["episodes"], detail["keep_list"], eval_split, eval_seed, eval_mode)
        if episodes is None:
            # The ORDER matters, not just the membership: LeRobot holds out
            # the tail of whatever list it is handed, so this is what makes
            # the eval set a random sample instead of the newest episodes.
            episodes = plan["order"]
        if not episodes:
            raise ValueError("no episodes left to train on — every one is rejected")
        if eval_split > 0 and not plan["train"]:
            raise ValueError(
                f"eval_split={eval_split} would leave no episodes to train on "
                f"({len(detail['keep_list'])} kept)"
            )
        spec = {
            "repo_id": repo_id,
            "episodes": [int(e) for e in episodes],
            "total_episodes": detail["total_episodes"],
            # Carried so a later rollout can prefill the camera mapping and
            # task wording from what the policy was actually trained on —
            # a rollout whose observation space differs from the recording
            # is a policy being shown a world it has never seen.
            "camera_keys": detail["video_keys"],
            "task_text": (detail["tasks"] or [""])[0],
            "policy_type": payload.get("policy_type", "act"),
            "steps": int(payload.get("steps", 100_000)),
            "batch_size": int(payload.get("batch_size", 8)),
            "eval_split": eval_split,
            "eval_mode": eval_mode,
            "eval_seed": eval_seed,
            "eval_episodes": plan["eval"],
            "train_episodes": plan["train"],
            "eval_steps": int(payload.get("eval_steps", 0)),
            "max_eval_samples": int(payload.get("max_eval_samples", 0)),
            "save_freq": int(payload.get("save_freq", 20_000)),
            # Scale the logging interval to the run so the loss chart has
            # roughly 200 points whatever the length. LeRobot's fixed
            # default of 200 means a 40-step debug run logs nothing at
            # all and the chart stays blank the whole way through.
            "log_freq": int(payload.get("log_freq")
                            or max(1, min(200, int(payload.get("steps", 100_000)) // 200))),
            "num_workers": int(payload.get("num_workers", 4)),
            "device": payload.get("device", "cuda"),
            "job_name": payload.get("job_name") or "",
            "extra_args": payload.get("extra_args") or [],
        }
        return runs.launch("train", spec, name=spec["job_name"] or spec["policy_type"])

    return _wrap(_apply)


@router.post("/run/record")
def post_record(request: Request, payload: dict = Body(...)) -> dict:
    """Start (or resume) a recording session.

    Only one can run at a time: they share the arm and the cameras, and a
    second one would fail deep inside the hardware setup rather than here.
    """
    _require_local(request)
    repo_id = (payload.get("repo_id") or "").strip()
    task = (payload.get("task") or "").strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    if not task:
        raise HTTPException(
            status_code=400,
            detail="a task description is required — it is stored with every episode "
                   "and is what the policy is conditioned on",
        )

    def _apply():
        catalog.validate_repo_id(repo_id)
        for run in runs.list_runs(limit=20):
            if run.get("kind") == "record" and run.get("status") == "running":
                raise ValueError(
                    f"a recording session is already running ({run['id']}). "
                    "Stop it before starting another."
                )
        exists = catalog.dataset_root(repo_id).exists()
        resume = bool(payload.get("resume", exists))
        if exists and not resume:
            raise ValueError(
                f"{repo_id} already exists — resume it to add episodes, or choose "
                "another name. Recording over it is not offered."
            )
        cameras = [str(c) for c in (payload.get("cameras") or []) if str(c).strip()]
        if not cameras:
            raise ValueError("at least one camera is required — a policy needs to see")
        # The recorder builds the IK model during its safety prep, after
        # it has already connected to the arm. Resolve the URDF here so a
        # missing one is a message in the browser instead of a traceback
        # from a subprocess that just powered the servos up and down.
        from ..ik.so101_model import resolve_so101_urdf_path
        try:
            resolve_so101_urdf_path(payload.get("urdf_path") or None)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        spec = {
            "repo_id": repo_id,
            "task": task,
            "resume": resume,
            "num_episodes": int(payload.get("num_episodes", 20)),
            "fps": int(payload.get("fps", 30)),
            "episode_time_s": float(payload.get("episode_time_s", 60)),
            "reset_time_s": payload.get("reset_time_s"),
            "cameras": cameras,
            "cam_width": int(payload.get("cam_width", 640)),
            "cam_height": int(payload.get("cam_height", 480)),
            "cam_fps": int(payload.get("cam_fps", 30)),
            "port": payload.get("port", os.environ.get("SO101_PORT", "/dev/ttyACM0")),
            "robot_id": payload.get("robot_id", "so101"),
            "hand": payload.get("hand", "right"),
            "ws_url": payload.get("ws_url", ""),
            "urdf_path": payload.get("urdf_path", ""),
            "joint_signs": payload.get("joint_signs", ""),
            "no_start_gate": bool(payload.get("no_start_gate", False)),
            "no_sounds": bool(payload.get("no_sounds", False)),
            "display_data": bool(payload.get("display_data", False)),
            "push_to_hub": bool(payload.get("push_to_hub", False)),
            "skip_calibration_check": bool(payload.get("skip_calibration_check", False)),
            "dry_run": bool(payload.get("dry_run", False)),
            "extra_args": payload.get("extra_args") or [],
        }
        return runs.launch("record", spec, name=repo_id.split("/")[-1])

    return _wrap(_apply)


@router.post("/run/rollout")
def post_rollout(request: Request, payload: dict = Body(...)) -> dict:
    """Run a trained checkpoint on the real arm. Guarded, and never
    open-ended: `duration_s` is required."""
    _require_local(request)
    policy_path = payload.get("policy_path")
    if not policy_path:
        raise HTTPException(status_code=400, detail="policy_path is required")
    if not Path(policy_path).is_dir():
        raise HTTPException(status_code=400, detail=f"no checkpoint at {policy_path}")
    duration = float(payload.get("duration_s") or 0)
    if duration <= 0:
        raise HTTPException(
            status_code=400,
            detail="duration_s must be > 0 — a rollout launched from a browser is never open-ended",
        )

    def _apply():
        spec = {
            "policy_path": policy_path,
            "task": payload.get("task", ""),
            "duration_s": duration,
            "fps": float(payload.get("fps", 30)),
            "port": payload.get("port", os.environ.get("SO101_PORT", "/dev/ttyACM0")),
            "robot_id": payload.get("robot_id", "so101"),
            "cameras": payload.get("cameras") or [],
            "cam_width": int(payload.get("cam_width", 640)),
            "cam_height": int(payload.get("cam_height", 480)),
            "cam_fps": int(payload.get("cam_fps", 30)),
            "joint_signs": payload.get("joint_signs", "1,1,1,1,1"),
            "skip_calibration_check": bool(payload.get("skip_calibration_check", False)),
            "device": payload.get("device", "cuda"),
            "source_run": payload.get("source_run", ""),
            "dry_run": bool(payload.get("dry_run", False)),
            "extra_args": payload.get("extra_args") or [],
        }
        return runs.launch("rollout", spec, name=payload.get("name") or "rollout")

    return _wrap(_apply)


@router.post("/run/stop")
def post_stop(request: Request, payload: dict = Body(...)) -> dict:
    _require_local(request)
    run_id = payload.get("id")
    if not run_id:
        raise HTTPException(status_code=400, detail="id is required")
    return _wrap(runs.stop, run_id)
