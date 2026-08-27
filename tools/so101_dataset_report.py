"""Per-episode quality report for a recorded LeRobotDataset.

Recording demonstrations is easy; noticing that a third of them are
unusable is not. This reads a dataset written by
`examples/record_so101.py` (or plain `lerobot-record`) and prints one
line of verdict per episode, so you can drop the failures before they
teach a policy the wrong thing.

What it checks per episode, all from the recorded arrays — no video
decoding needed for the verdict:

  grasp        the gripper must CLOSE and then REOPEN. A pick-and-place
               that never closes the jaws did not pick anything up; a
               run of several close/open cycles is a struggle with
               retries.
  motion       total joint sweep. Near zero means the operator never
               engaged the clutch and the arm sat at rest the whole
               episode.
  tracking     mean |action - state|. This separates OPERATOR failures
               from HARDWARE failures: if commands were issued and the
               arm did not follow, the servos are the problem, not the
               demonstration.
  share        fraction of the whole dataset this episode occupies. One
               long fumbling episode can dominate a small dataset and
               drag the policy toward imitating the fumbling.

Verdicts are heuristics tuned for a single pick-and-place task; they
flag episodes worth LOOKING at, they do not overrule your eyes. Pass
`--contact-sheets DIR` to also write one filmstrip PNG per episode
(needs ffmpeg) and actually look at the flagged ones.

Run:
    python tools/so101_dataset_report.py --repo-id local/so101_pick_cube
    python tools/so101_dataset_report.py --repo-id local/... --contact-sheets /tmp/sheets

The same verdicts drive the relay's review page (http://localhost:8443/data),
where you can watch each episode next to its traces and mark the failures
without deleting anything. To prune from here instead:
    lerobot-edit-dataset --repo_id local/so101_pick_cube \\
        --new_repo_id local/so101_pick_cube \\
        --operation.type=delete_episodes --operation.episode_indices='[1,3,4]'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

# The verdict heuristics live in the package so this report and the web
# review page (`/data`) grade episodes with one implementation — two
# copies that drifted would be worse than no verdict at all.
from vr_teleop_kit.data.grade import (  # noqa: E402
    DOMINANT_SHARE,
    GRIP_CLOSED_BELOW,
    GRIP_OPEN_ABOVE,
    GRIPPER_IDX,
    JOINTS,
    STILL_TOTAL_DEG,
    TRACKING_FAIL_DEG,
    analyse,
)


def dataset_root(repo_id: str | None, root: str | None) -> Path:
    if root:
        return Path(root).expanduser()
    if not repo_id:
        raise SystemExit("pass --repo-id (or --root)")
    base = os.environ.get("HF_LEROBOT_HOME") or (Path.home() / ".cache/huggingface/lerobot")
    return Path(base).expanduser() / repo_id


def load_frames(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no data parquet under {root}/data — is the path right?")
    try:
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    except Exception as e:
        raise SystemExit(
            f"could not read {root}/data ({e}).\n"
            "If a recording session is still running, stop it first (Esc) — the "
            "parquet is only complete once the session finishes."
        ) from e


def load_episode_meta(root: Path) -> pd.DataFrame | None:
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not files:
        return None
    try:
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    except Exception:
        return None


def write_contact_sheet(video: Path, start_s: float, end_s: float, out: Path, tiles: int = 6) -> bool:
    """Filmstrip of `tiles` frames spanning [start_s, end_s)."""
    span = max(end_s - start_s, 1e-3)
    step = span / tiles
    # Seek before -i so ffmpeg jumps rather than decoding from the top.
    expr = "+".join(f"eq(n\\,{int(i * step * 30)})" for i in range(tiles))
    cmd = [
        "ffmpeg", "-v", "error", "-ss", str(start_s), "-t", str(span), "-i", str(video),
        "-vf", f"select='{expr}',scale=320:240,tile={min(tiles,3)}x{(tiles + 2)//3}",
        "-frames:v", "1", "-y", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out.exists()
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo-id", default=None, help="e.g. local/so101_pick_cube")
    ap.add_argument("--root", default=None, help="dataset directory (overrides --repo-id)")
    ap.add_argument("--contact-sheets", default=None, metavar="DIR",
                    help="also write one filmstrip PNG per episode into DIR (needs ffmpeg)")
    args = ap.parse_args()

    root = dataset_root(args.repo_id, args.root)
    if not root.exists():
        raise SystemExit(f"no dataset at {root}")
    info = json.load(open(root / "meta" / "info.json"))
    fps = int(info.get("fps", 30))
    df = load_frames(root)
    total = len(df)

    print(f"\ndataset : {root}")
    print(f"episodes: {info.get('total_episodes')}   frames: {info.get('total_frames')}   fps: {fps}")

    # Cross-check the video against the metadata. A mismatch means stale
    # frames are sitting in the video file (it happens when a dataset
    # directory is reused), and frame indices may not line up.
    vids = sorted(glob.glob(str(root / "videos" / "**" / "*.mp4"), recursive=True))
    if vids:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", vids[0]],
                capture_output=True, text=True, check=True).stdout.strip()
            nb = int(out)
            status = "OK" if nb == info.get("total_frames") else "MISMATCH — stale frames in the video file"
            print(f"video   : {nb} frames vs {info.get('total_frames')} in metadata  [{status}]")
        except Exception:
            pass

    ep_meta = load_episode_meta(root)
    rows = []
    print()
    for ep, sub in df.groupby("episode_index"):
        r = analyse(sub, fps, total)
        r["episode"] = int(ep)
        rows.append(r)
        mark = {"PASS": "PASS   ", "SUSPECT": "SUSPECT", "FAIL": "FAIL   "}[r["verdict"]]
        print(f"  ep {int(ep):<3d} {mark}  {r['frames']:5d} fr / {r['seconds']:5.1f}s "
              f"({r['share']*100:4.1f}%)  grip {r['grip_min']:5.1f}-{r['grip_max']:5.1f} "
              f"closes={r['closes']}  track={r['tracking']:.2f}°")
        print(f"          {r['why']}")
        print("          sweep " + " ".join(
            f"{n.split('_')[0][:5]}:{v:.0f}°" for n, v in zip(JOINTS, r["sweep"])))

    good = [r["episode"] for r in rows if r["verdict"] == "PASS"]
    susp = [r["episode"] for r in rows if r["verdict"] == "SUSPECT"]
    bad = [r["episode"] for r in rows if r["verdict"] == "FAIL"]
    print(f"\n  PASS    {len(good):2d}  {good}")
    print(f"  SUSPECT {len(susp):2d}  {susp}   (look at these before deciding)")
    print(f"  FAIL    {len(bad):2d}  {bad}")

    if bad or susp:
        drop = sorted(bad + susp)
        print(f"\n  To drop the failures (backs the dataset up first):\n"
              f"    lerobot-edit-dataset --repo_id {args.repo_id or root.name} \\\n"
              f"        --new_repo_id {args.repo_id or root.name} \\\n"
              f"        --operation.type=delete_episodes "
              f"--operation.episode_indices='{drop}'")

    if args.contact_sheets:
        outdir = Path(args.contact_sheets).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)
        if not vids:
            print("\n  no video in the dataset — skipping contact sheets")
        elif ep_meta is None:
            print("\n  no episode metadata — skipping contact sheets")
        else:
            tcol = [c for c in ep_meta.columns if c.endswith("from_timestamp")]
            if not tcol:
                print("\n  episode metadata has no video timestamps — skipping contact sheets")
            else:
                from_c = tcol[0]
                to_c = from_c.replace("from_timestamp", "to_timestamp")
                print()
                for _, m in ep_meta.iterrows():
                    ep = int(m["episode_index"])
                    out = outdir / f"ep{ep:03d}.png"
                    ok = write_contact_sheet(Path(vids[0]), float(m[from_c]), float(m[to_c]), out)
                    print(f"  {'wrote' if ok else 'FAILED'} {out}")

    if any(r["verdict"] == "FAIL" and "did not follow" in r["why"] for r in rows):
        print("\n  NOTE: a tracking failure is a HARDWARE symptom, not a bad demo — "
              "run tools/so101_joint_diag.py before recording more.")


if __name__ == "__main__":
    main()
