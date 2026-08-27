"""Smoke test for the /data page's dataset layer — no hardware needed.

Runs against a throwaway dataset tree in a temp directory, so it never
touches real recordings.

The symlink case is here on purpose. Moving datasets out of `~/.cache`
and leaving a compatibility symlink behind is the obvious way to do it,
and it broke the page once: `hf_home()` returned the unresolved link
while `dataset_root()` returned the resolved target, so `relative_to`
raised "is not in the subpath of" on two spellings of one directory.
Nothing about that is visible until someone actually has a symlinked
home, which is exactly when it matters.

Run:
    python tools/smoke_test_dataui.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FPS = 30
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{'  — ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(label)


def make_dataset(root: Path, n_episodes: int = 2) -> None:
    """Episode 0 is a clean grasp, episode 1 the arm never moving; any
    further episodes repeat the clean grasp with distinct lengths."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.top" / "chunk-000").mkdir(parents=True)

    rows, ep_meta = [], []
    cursor = 0
    for ep in range(n_episodes):
        moving = ep != 1
        # Distinct lengths, so a renumbering is detectable per mark.
        n = 90 + ep
        state = np.zeros((n, 6), dtype=np.float32)
        state[:, 5] = 100.0  # gripper open
        if moving:
            state[:, 0] = np.linspace(0, 40, n)          # a real joint sweep
            state[30:60, 5] = 10.0                       # close, then reopen
        action = state + 0.1
        for i in range(n):
            rows.append({
                "action": action[i], "observation.state": state[i],
                "timestamp": i / FPS, "frame_index": i, "episode_index": ep,
                "index": cursor + i, "task_index": 0,
            })
        ep_meta.append({
            "episode_index": ep, "tasks": ["Test task"], "length": n,
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": cursor, "dataset_to_index": cursor + n,
            "videos/observation.images.top/chunk_index": 0,
            "videos/observation.images.top/file_index": 0,
            "videos/observation.images.top/from_timestamp": cursor / FPS,
            "videos/observation.images.top/to_timestamp": (cursor + n) / FPS,
        })
        cursor += n

    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(ep_meta).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task": ["Test task"]}).set_index("task").assign(task_index=0).to_parquet(
        root / "meta" / "tasks.parquet")
    (root / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4").write_bytes(b"\0" * 64)
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": FPS, "robot_type": "so_follower",
        "total_episodes": n_episodes, "total_frames": cursor, "total_tasks": 1,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [6], "names": ["a"] * 6},
            "observation.state": {"dtype": "float32", "shape": [6], "names": [
                "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]},
            "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }))


def run_checks(label: str, home: Path) -> None:
    """Every check that must hold whatever spelling of the home is used."""
    os.environ["HF_LEROBOT_HOME"] = str(home)
    for mod in [m for m in list(sys.modules) if m.startswith("vr_teleop_kit.data")]:
        del sys.modules[mod]
    from vr_teleop_kit.data import catalog

    print(f"\n{label}  (HF_LEROBOT_HOME={home})")
    listed = catalog.list_datasets()
    check("lists the dataset", [d["repo_id"] for d in listed] == ["local/smoke"],
          str([d["repo_id"] for d in listed]))

    # The regression: detail derives repo_id via relative_to(hf_home()).
    detail = catalog.dataset_detail("local/smoke")
    check("dataset_detail resolves repo_id", detail["repo_id"] == "local/smoke",
          detail["repo_id"])
    check("grades both episodes",
          [e["verdict"] for e in detail["episodes"]] == ["PASS", "FAIL"],
          str([e["verdict"] for e in detail["episodes"]]))
    check("flags the still arm",
          "never moved" in detail["episodes"][1]["why"], detail["episodes"][1]["why"])
    check("episode labels are 1-based",
          [e["label"] for e in detail["episodes"]] == [1, 2])

    trace = catalog.episode_trace("local/smoke", 0)
    check("trace returns samples", len(trace["t"]) == 90, str(len(trace["t"])))
    check("video path resolves",
          catalog.video_path("local/smoke", "observation.images.top", 0, 0).exists())

    for bad in ("../../etc", "local/../../etc"):
        try:
            catalog.dataset_root(bad)
            check(f"refuses traversal {bad!r}", False, "was allowed")
        except ValueError:
            check(f"refuses traversal {bad!r}", True)

    # Review marks round-trip to disk.
    root = catalog.dataset_root("local/smoke")
    from vr_teleop_kit.data import review
    detail = catalog.dataset_detail("local/smoke")
    review.set_status(root, 1, review.REJECT, note="still arm",
                      episode_frames=detail["episode_frames"][1])
    reread = catalog.dataset_detail("local/smoke")
    check("reject mark persists", reread["episodes"][1]["status"] == "reject")
    check("keep list excludes it", reread["keep_list"] == [0], str(reread["keep_list"]))

    # Appending episodes must NOT invalidate marks: --resume leaves every
    # existing index exactly where it was. Warning here on every session
    # would train people to ignore the warning that matters.
    frames = dict(reread["episode_frames"])
    frames[2] = 120  # a newly recorded episode
    marks = review.load(root)
    check("append does not flag marks", not review.is_stale(marks, {}, frames))

    # A prune renumbers survivors, so the episode at a marked index is a
    # different one — that must be caught.
    pruned = {0: reread["episode_frames"][0], 1: 999}
    check("prune flags the moved mark", review.is_stale(marks, {}, pruned),
          str(review.stale_marks(marks, pruned)))
    check("prune names only the affected mark",
          review.stale_marks(marks, pruned) == [1])
    review.clear(root)


def check_split(tmp: Path) -> None:
    """The eval split must be a random SAMPLE, not the newest episodes.

    LeRobot holds out the tail of the episode list it is handed, and
    episode order is not neutral: an operator improves across a session,
    and a resumed dataset ends with a different day's conditions. Holding
    out the last N therefore validates on the best, most recent
    demonstrations while training on the earliest — so the ORDER passed
    to LeRobot is shuffled instead.
    """
    home = tmp / "split" / "lerobot"
    make_dataset(home / "local" / "many", n_episodes=10)
    os.environ["HF_LEROBOT_HOME"] = str(home)
    for mod in [m for m in list(sys.modules) if m.startswith("vr_teleop_kit.data")]:
        del sys.modules[mod]
    from vr_teleop_kit.data import catalog

    print("\nevaluation split")
    d = catalog.dataset_detail("local/many")
    keep = d["keep_list"]

    recent = catalog.plan_eval_split(d["episodes"], keep, 0.2, 0, "recent")
    check("'recent' holds out the newest", recent["eval"] == [8, 9], str(recent["eval"]))

    rnd = catalog.plan_eval_split(d["episodes"], keep, 0.2, 42, "random")
    check("'random' holds out the same COUNT", len(rnd["eval"]) == 2, str(rnd["eval"]))
    check("'random' is not the newest", rnd["eval"] != [8, 9], str(rnd["eval"]))
    check("train and eval are disjoint",
          not (set(rnd["train"]) & set(rnd["eval"])))
    check("every kept episode is used once",
          sorted(rnd["train"] + rnd["eval"]) == sorted(keep))
    check("order is what LeRobot receives",
          sorted(rnd["order"]) == sorted(keep) and len(rnd["order"]) == len(keep))
    # The tail of the order is exactly the eval set — this is the property
    # that makes the shuffle work at all.
    check("eval set is the tail of the order",
          sorted(rnd["order"][-len(rnd["eval"]):]) == rnd["eval"],
          f"order={rnd['order']} eval={rnd['eval']}")

    same = catalog.plan_eval_split(d["episodes"], keep, 0.2, 42, "random")
    check("same seed reproduces the split", same["eval"] == rnd["eval"])
    draws = {tuple(catalog.plan_eval_split(d["episodes"], keep, 0.2, s, "random")["eval"])
             for s in range(8)}
    check("different seeds give different draws", len(draws) > 1, f"{len(draws)} distinct")

    check("eval_split=0 holds out nothing",
          catalog.plan_eval_split(d["episodes"], keep, 0.0, 42, "random")["eval"] == [])


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dataui-smoke-"))
    try:
        real = tmp / "real" / "lerobot"
        make_dataset(real / "local" / "smoke")
        run_checks("direct path", real)

        # The case that broke: home is a symlink to the real tree.
        link = tmp / "cache" / "lerobot"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(real)
        run_checks("through a symlink", link)

        check_split(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("HF_LEROBOT_HOME", None)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
