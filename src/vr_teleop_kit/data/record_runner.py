"""Subprocess entry point for a recording session started from the web UI.

Runs `examples/record_so101.py` — the same script, the same safety
sequence — with the arguments the page collected, after checking the two
things that reliably ruin a session before it starts:

  * something else holding the servo bus (`haller_hmi` sits on
    /dev/ttyACM0 with torque enabled; two processes on one Feetech bus
    corrupt each other's packets);
  * the relay holding a camera the recorder needs. The RealSense colour
    node cannot be shared, and this is not hypothetical — a whole
    session was recorded as purple sludge before that was understood.

Also serves camera discovery (`--list-cameras`) for the form, because
finding RealSense devices means importing pyrealsense2 and the relay
must not carry that import.

The session is driven from the headset (B / Y on the controllers) or
from the page, which sends `record_control` messages over the relay —
see `start_vr_episode_buttons` in record_so101.py. Note that stdin is
not a terminal here, so the pynput keyboard shortcuts are unavailable;
that is exactly why the page has its own End / Re-record / Stop buttons.

Usage:
    python -m vr_teleop_kit.data.record_runner SPEC.json
    python -m vr_teleop_kit.data.record_runner --list-cameras
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import traceback
from pathlib import Path

# Stable v4l path of a RealSense colour stream (index2 is the only node
# cv2 can open on the D455).
REALSENSE_RGB_GLOB = "/dev/v4l/by-id/usb-Intel_R__RealSense*video-index2"


def _examples_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "examples"


# ── camera discovery ─────────────────────────────────────────────────────

def _v4l2_devices() -> list[dict]:
    """Every /dev/videoN with its kernel-reported name and stable path.

    Reads /sys rather than opening the devices: opening a v4l2 node to
    ask what it is would fight with whatever is already streaming from it.
    """
    out = []
    by_id: dict[str, str] = {}
    for link in glob.glob("/dev/v4l/by-id/*"):
        try:
            by_id[os.path.realpath(link)] = link
        except OSError:
            continue
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        node = os.path.basename(path)
        try:
            name = Path(f"/sys/class/video4linux/{node}/name").read_text().strip()
        except OSError:
            name = node
        # Only capture-capable nodes are useful; metadata nodes report no
        # capture format and would just clutter the picker.
        try:
            caps = Path(f"/sys/class/video4linux/{node}/device/../class").read_text()
        except OSError:
            caps = ""
        out.append({
            "kind": "v4l2",
            "device": path,
            "stable_path": by_id.get(os.path.realpath(path)),
            "name": name,
            "source": by_id.get(os.path.realpath(path)) or path,
        })
    return out


def _realsense_devices() -> list[dict]:
    try:
        from lerobot.cameras.realsense import RealSenseCamera
    except Exception:
        return []
    try:
        found = RealSenseCamera.find_cameras()
    except Exception:
        return []
    out = []
    for cam in found:
        serial = str(cam.get("id"))
        out.append({
            "kind": "realsense",
            "device": None,
            "name": str(cam.get("name") or "RealSense") + f" ({serial})",
            "serial": serial,
            "source": f"rs:{serial}",
        })
    return out


def list_cameras() -> dict:
    """RealSense devices first — on this rig `auto` means the RealSense,
    and it must go through librealsense rather than v4l2 (the UYVY node
    decodes dark and magenta, which is what a policy would then train
    on)."""
    rs = _realsense_devices()
    v4l2 = _v4l2_devices()
    rs_nodes = {os.path.realpath(p) for p in glob.glob(REALSENSE_RGB_GLOB)}
    for cam in v4l2:
        real = os.path.realpath(cam["device"])
        cam["is_realsense_node"] = real in rs_nodes
        if cam["is_realsense_node"]:
            cam["warning"] = (
                "This is the RealSense's v4l2 node. Recording through it produces "
                "dark, magenta frames — use the RealSense entry above instead."
            )
    cameras = []
    if rs:
        cameras.append({
            "kind": "auto", "name": "RealSense (auto, native driver)", "source": "auto",
        })
    cameras.extend(rs)
    cameras.extend(v4l2)
    return {"cameras": cameras, "realsense_count": len(rs)}


# ── preflight ────────────────────────────────────────────────────────────

def port_holders(device: str) -> list[str]:
    out: list[str] = []
    try:
        target = os.path.realpath(device)
    except OSError:
        return out
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or proc.name == str(os.getpid()):
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.path.realpath(fd) == target:
                        cmd = (proc / "cmdline").read_bytes().decode("utf-8", "replace")
                        out.append(f"pid {proc.name}: {cmd.replace(chr(0), ' ').strip()[:160]}")
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return out


def camera_conflicts(specs: list[str]) -> list[str]:
    """Requested cameras that another process already holds open."""
    problems = []
    rs_nodes = {os.path.realpath(p) for p in glob.glob(REALSENSE_RGB_GLOB)}
    for spec in specs:
        _name, _sep, source = spec.partition("=")
        targets: list[str] = []
        if source in ("auto",) or source.startswith("rs:"):
            targets = sorted(rs_nodes)
        elif source.startswith("/dev/"):
            targets = [os.path.realpath(source)]
        for target in targets:
            for holder in port_holders(target):
                problems.append(f"{spec}: {target} is open — {holder}")
    return problems


# ── argv ─────────────────────────────────────────────────────────────────

def build_argv(spec: dict) -> list[str]:
    argv = [
        str(_examples_dir() / "record_so101.py"),
        f"--repo-id={spec['repo_id']}",
        f"--task={spec['task']}",
        f"--num-episodes={int(spec.get('num_episodes', 20))}",
        f"--fps={int(spec.get('fps', 30))}",
        f"--episode-time-s={float(spec.get('episode_time_s', 60))}",
        f"--port={spec.get('port', '/dev/ttyACM0')}",
        f"--id={spec.get('robot_id', 'so101')}",
        f"--hand={spec.get('hand', 'right')}",
        f"--cam-width={int(spec.get('cam_width', 640))}",
        f"--cam-height={int(spec.get('cam_height', 480))}",
        f"--cam-fps={int(spec.get('cam_fps', 30))}",
    ]
    if spec.get("ws_url"):
        argv.append(f"--ws-url={spec['ws_url']}")
    if spec.get("urdf_path"):
        argv.append(f"--urdf-path={spec['urdf_path']}")
    if spec.get("joint_signs"):
        argv.append(f"--joint-signs={spec['joint_signs']}")
    if spec.get("reset_time_s") is not None:
        argv.append(f"--reset-time-s={float(spec['reset_time_s'])}")
    for cam in spec.get("cameras") or []:
        argv.append(f"--camera={cam}")
    if spec.get("resume"):
        argv.append("--resume")
    if spec.get("no_start_gate"):
        argv.append("--no-start-gate")
    if spec.get("no_sounds"):
        argv.append("--no-sounds")
    if spec.get("display_data"):
        argv.append("--display-data")
    if spec.get("push_to_hub"):
        argv.append("--push-to-hub")
    if spec.get("skip_calibration_check"):
        argv.append("--skip-calibration-check")
    for extra in spec.get("extra_args") or []:
        argv.append(str(extra))
    return argv


def main() -> int:
    raw = sys.argv[1:]
    if "--list-cameras" in raw:
        print(json.dumps(list_cameras()))
        return 0

    dry_run = "--dry-run" in raw
    args = [a for a in raw if not a.startswith("--")]
    if not args:
        print("usage: python -m vr_teleop_kit.data.record_runner SPEC.json [--dry-run]")
        return 2

    spec = json.loads(Path(args[0]).read_text())
    dry_run = dry_run or bool(spec.get("dry_run"))
    run_dir = Path(spec["run_dir"])

    from . import runs

    status, exit_code, error = "done", 0, ""
    try:
        port = spec.get("port", "/dev/ttyACM0")
        if not Path(port).exists():
            raise SystemExit(f"no serial device at {port}")
        holders = port_holders(port)
        if holders:
            raise SystemExit(
                f"{port} is already open by:\n  " + "\n  ".join(holders) +
                "\nStop it first — two processes on one Feetech bus corrupt each other's packets."
            )
        conflicts = camera_conflicts(spec.get("cameras") or [])
        if conflicts:
            raise SystemExit(
                "a camera this session needs is already in use:\n  " + "\n  ".join(conflicts) +
                "\nThe RealSense colour node cannot be shared. If the relay is holding it, "
                "turn camera streaming off in the headset (or restart the relay without "
                "CAM_TOP) — in passthrough you see the real scene anyway."
            )
        if not (spec.get("task") or "").strip():
            raise SystemExit("a task description is required — it is stored with every episode")

        argv = build_argv(spec)
        print("record argv: python " + " ".join(argv), flush=True)
        if dry_run:
            print("\ndry run: preflight passed, nothing was started.", flush=True)
            return 0

        # record_so101.py imports its siblings by module name, the way it
        # does when run as a script from examples/.
        sys.path.insert(0, str(_examples_dir()))
        sys.argv = argv
        import record_so101

        record_so101.main()
    except KeyboardInterrupt:
        status, exit_code, error = "stopped", 130, "interrupted"
        print("\ninterrupted", flush=True)
    except SystemExit as e:
        code = e.code
        if isinstance(code, str):
            print(code, flush=True)
            status, exit_code, error = "failed", 1, code.splitlines()[0]
        else:
            exit_code = int(code or 0)
            status = "done" if exit_code == 0 else "failed"
            error = "" if exit_code == 0 else f"exited with {exit_code}"
    except Exception as e:
        status, exit_code, error = "failed", 1, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        runs.write_result(run_dir, status, exit_code, error)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
