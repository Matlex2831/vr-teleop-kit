"""Reading recorded LeRobotDatasets for the review UI.

Deliberately implemented with pandas/pyarrow alone. This module is
imported by the relay server, which is the real-time path for teleop —
pulling `lerobot` in would drag torch, torchcodec and a multi-second
import into a process whose job is to forward `xr_frame` messages with
minimal latency. Everything here is plain parquet and JSON reads.

Dataset layout (LeRobot v3.0), which is what shapes the API below:

    <root>/meta/info.json                     features, fps, totals
    <root>/meta/tasks.parquet                 task strings
    <root>/meta/episodes/chunk-*/file-*.parquet
                                              per-episode index ranges,
                                              video timestamps, stats
    <root>/data/chunk-*/file-*.parquet        all episodes' frames
    <root>/videos/<key>/chunk-*/file-*.mp4    all episodes' video

Note that v3.0 packs MANY episodes into ONE parquet and ONE mp4 — an
episode is a *slice*, identified by `dataset_from_index`/`to_index` for
the frames and `from_timestamp`/`to_timestamp` for the video. That is
why the player can show a single episode without any transcoding: it
just seeks.
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

from . import review as review_mod
from .grade import JOINTS, analyse

# Directories under HF_LEROBOT_HOME that are not datasets.
_NOT_DATASETS = {"calibration", "hub"}

# Cap on the number of points sent to the browser for one episode's
# trace. A 60 s episode at 30 fps is 1800 samples; drawing more pixels
# than the chart has is wasted bandwidth on every episode click.
TRACE_MAX_POINTS = 600


class DataDependencyError(RuntimeError):
    """pandas/pyarrow missing — the API turns this into a clear message."""


class DatasetBusyError(RuntimeError):
    """The data parquet is mid-write, so the dataset cannot be read yet.

    This is the normal state of a dataset while `record_so101.py` is
    running: LeRobot finalises the parquet when the session ends, and
    until then pyarrow sees a truncated file. It is a "come back in a
    minute", not a failure, and the page says so rather than showing a
    stack trace.
    """


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - depends on install extras
        raise DataDependencyError(
            "reading datasets needs pandas + pyarrow: "
            "uv pip install 'vr-teleop-kit[dataui]'"
        ) from e
    return pd


def hf_home() -> Path:
    """The dataset cache root, always fully resolved.

    Resolving here rather than at each call site is load-bearing: the
    default location is commonly a symlink (moving datasets out of
    `~/.cache` and leaving a compat link behind is the obvious way to do
    it). A resolved dataset root then has no common prefix with an
    unresolved base, and `Path.relative_to` fails with "is not in the
    subpath of" on two spellings of the same directory.
    """
    base = os.environ.get("HF_LEROBOT_HOME")
    if base:
        return Path(base).expanduser().resolve()
    return (Path.home() / ".cache/huggingface/lerobot").resolve()


def dataset_root(repo_id: str) -> Path:
    """Resolve a repo-id to its directory, refusing to escape the cache.

    `repo_id` arrives from a URL path, so `../` traversal is a real
    concern: this endpoint would happily serve any mp4 on the machine.
    """
    base = hf_home()
    root = (base / repo_id).resolve()
    if not (root == base or base in root.parents):
        raise ValueError(f"repo_id escapes the dataset cache: {repo_id!r}")
    return root


# ── discovery ────────────────────────────────────────────────────────────

def _info(root: Path) -> dict | None:
    try:
        return json.loads((root / "meta" / "info.json").read_text())
    except Exception:
        return None


def _dir_size(root: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _fingerprint(info: dict) -> dict:
    return {
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
    }


def _video_keys(info: dict) -> list[str]:
    return [
        k for k, f in (info.get("features") or {}).items()
        if f.get("dtype") == "video"
    ]


def list_datasets() -> list[dict]:
    """Every dataset under HF_LEROBOT_HOME, newest first.

    Both `name/meta/info.json` and `namespace/name/meta/info.json` are
    matched: the hf_user/name convention is what `record_so101.py` uses,
    but a bare directory is still a valid dataset.
    """
    base = hf_home()
    if not base.exists():
        return []
    seen: set[Path] = set()
    out: list[dict] = []
    for pattern in ("*/meta/info.json", "*/*/meta/info.json"):
        for path in base.glob(pattern):
            root = path.parent.parent
            if root in seen or root.name in _NOT_DATASETS:
                continue
            if root.parent != base and root.parent.name in _NOT_DATASETS:
                continue
            seen.add(root)
            info = _info(root)
            if info is None:
                continue
            repo_id = str(root.relative_to(base))
            total_eps = int(info.get("total_episodes", 0))
            fps = int(info.get("fps", 30)) or 30
            rev = review_mod.load(root)
            out.append({
                "repo_id": repo_id,
                "root": str(root),
                "total_episodes": total_eps,
                "total_frames": int(info.get("total_frames", 0)),
                "fps": fps,
                "seconds": int(info.get("total_frames", 0)) / fps,
                "robot_type": info.get("robot_type"),
                "codebase_version": info.get("codebase_version"),
                "video_keys": _video_keys(info),
                "tasks": _tasks(root),
                "size_bytes": _dir_size(root),
                "modified": path.stat().st_mtime,
                "review": review_mod.counts(rev, total_eps),
                # The card only has totals to work with; the precise
                # per-mark check runs when the dataset is opened.
                "review_stale": review_mod.is_stale(rev, _fingerprint(info)),
                # A dataset whose backup twin exists was pruned in place by
                # `lerobot-edit-dataset`; the UI de-emphasises the leftover.
                "is_backup": repo_id.endswith("_old"),
            })
    out.sort(key=lambda d: d["modified"], reverse=True)
    return out


def _tasks(root: Path) -> list[str]:
    pd = _pandas()
    path = root / "meta" / "tasks.parquet"
    if not path.exists():
        return []
    try:
        df = pd.read_parquet(path)
    except Exception:
        return []
    # The task string is the index in v3.0's tasks.parquet.
    if df.index.name == "task":
        return [str(t) for t in df.index.tolist()]
    if "task" in df.columns:
        return [str(t) for t in df["task"].tolist()]
    return []


# ── per-dataset detail ───────────────────────────────────────────────────

def _parquets(root: Path, *parts: str) -> list[str]:
    return sorted(glob.glob(str(root.joinpath(*parts) / "**" / "*.parquet"), recursive=True))


def _stamp(root: Path) -> tuple:
    """Cheap change-detector for the cache: every data/meta parquet's
    size and mtime. Recording appends files, so this moves whenever the
    dataset does."""
    files = _parquets(root, "data") + _parquets(root, "meta", "episodes")
    out = []
    for f in files:
        try:
            st = os.stat(f)
            out.append((f, st.st_size, st.st_mtime))
        except OSError:
            pass
    return tuple(out)


_detail_cache: dict[str, tuple[tuple, float, dict]] = {}

# Loaded frame tables, keyed by dataset root. Clicking through episodes
# asks for one trace at a time, and re-reading the whole parquet on every
# click would make a 25-episode review crawl. Bounded to a couple of
# datasets so a long browsing session cannot grow without limit.
_frames_cache: dict[str, tuple[tuple, object]] = {}
_FRAMES_CACHE_MAX = 2

_GRADE_COLUMNS = ["action", "observation.state", "episode_index"]


def _load_frames(root: Path, columns: list[str] | None = None):
    pd = _pandas()
    files = _parquets(root, "data")
    if not files:
        raise FileNotFoundError(f"no data parquet under {root}/data")
    try:
        return pd.concat(
            [pd.read_parquet(f, columns=columns) for f in files], ignore_index=True
        )
    except Exception as e:
        raise DatasetBusyError(
            "this dataset cannot be read yet — a recording session is most likely "
            "still running. Stop it with Esc (not Ctrl-C, which loses the current "
            "episode); the parquet is only complete once the session finishes."
        ) from e


def _frames(root: Path):
    """Cached action/state/episode_index table for one dataset."""
    key = str(root)
    stamp = _stamp(root)
    hit = _frames_cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    df = _load_frames(root, columns=_GRADE_COLUMNS)
    if len(_frames_cache) >= _FRAMES_CACHE_MAX:
        _frames_cache.pop(next(iter(_frames_cache)))
    _frames_cache[key] = (stamp, df)
    return df


def _load_episode_meta(root: Path):
    pd = _pandas()
    files = _parquets(root, "meta", "episodes")
    if not files:
        return None
    try:
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    except Exception:
        return None


def dataset_detail(repo_id: str, use_cache: bool = True) -> dict:
    """Info plus one row per episode: length, task, video slice, grade.

    Grading reads the whole data parquet (action + state only), which is
    a few MB even for a long session — the video, which is the bulk of a
    dataset, is never touched.
    """
    root = dataset_root(repo_id)
    info = _info(root)
    if info is None:
        raise FileNotFoundError(f"no dataset at {root}")

    stamp = _stamp(root)
    cached = _detail_cache.get(repo_id)
    if use_cache and cached and cached[0] == stamp:
        detail = dict(cached[2])
    else:
        detail = _build_detail(root, info)
        _detail_cache[repo_id] = (stamp, time.time(), detail)
        detail = dict(detail)

    # Review marks are cheap and change often — always read them fresh so
    # a mark never appears to be lost behind a cached grade.
    rev = review_mod.load(root)
    fingerprint = _fingerprint(info)
    episodes = []
    for ep in detail["episodes"]:
        ep = dict(ep)
        ep["status"] = review_mod.status_of(rev, ep["episode_index"])
        ep["note"] = review_mod.note_of(rev, ep["episode_index"])
        episodes.append(ep)
    detail["episodes"] = episodes
    detail["review"] = review_mod.counts(rev, detail["total_episodes"])
    # Validate each mark against the episode now at its index, rather
    # than against the dataset's totals: appending with --resume changes
    # the totals but moves nothing.
    frames_by_index = {e["episode_index"]: e["frames"] for e in episodes}
    detail["episode_frames"] = frames_by_index
    detail["stale_episodes"] = review_mod.stale_marks(rev, frames_by_index)
    detail["review_stale"] = bool(detail["stale_episodes"])
    detail["keep_list"] = review_mod.keep_list(rev, detail["total_episodes"])
    detail["fingerprint"] = fingerprint
    return detail


def _build_detail(root: Path, info: dict) -> dict:
    fps = int(info.get("fps", 30)) or 30
    total_frames = int(info.get("total_frames", 0))
    video_keys = _video_keys(info)
    ep_meta = _load_episode_meta(root)

    df = _frames(root)
    total = len(df)

    meta_by_ep: dict[int, dict] = {}
    if ep_meta is not None:
        for _, row in ep_meta.iterrows():
            meta_by_ep[int(row["episode_index"])] = row

    episodes = []
    for ep, sub in df.groupby("episode_index"):
        ep = int(ep)
        r = analyse(sub, fps, total or total_frames)
        row = meta_by_ep.get(ep)
        tasks = []
        videos = {}
        if row is not None:
            raw_tasks = row.get("tasks")
            if raw_tasks is not None:
                tasks = [str(t) for t in list(raw_tasks)]
            for key in video_keys:
                col = f"videos/{key}"
                if f"{col}/from_timestamp" in row:
                    videos[key] = {
                        "chunk_index": int(row[f"{col}/chunk_index"]),
                        "file_index": int(row[f"{col}/file_index"]),
                        "from_timestamp": float(row[f"{col}/from_timestamp"]),
                        "to_timestamp": float(row[f"{col}/to_timestamp"]),
                    }
        episodes.append({
            "episode_index": ep,
            # Oscar counts episodes 1-based in conversation; the UI shows
            # both so "episode 3" can never delete index 3 by accident.
            "label": ep + 1,
            "frames": r["frames"],
            "seconds": r["seconds"],
            "share": r["share"],
            "verdict": r["verdict"],
            "why": r["why"],
            "closes": r["closes"],
            "reopened": r["reopened"],
            "grip_min": r["grip_min"],
            "grip_max": r["grip_max"],
            "tracking": r["tracking"],
            "sweep_total": r["sweep_total"],
            "sweep": [float(v) for v in r["sweep"]],
            "tasks": tasks,
            "videos": videos,
        })
    episodes.sort(key=lambda e: e["episode_index"])

    return {
        "repo_id": str(root.relative_to(hf_home())),
        "root": str(root),
        "fps": fps,
        "total_episodes": int(info.get("total_episodes", len(episodes))),
        "total_frames": total_frames,
        "seconds": total_frames / fps,
        "robot_type": info.get("robot_type"),
        "codebase_version": info.get("codebase_version"),
        "video_keys": video_keys,
        "joints": list(JOINTS),
        "tasks": _tasks(root),
        "features": {
            k: {"dtype": f.get("dtype"), "shape": f.get("shape"), "names": f.get("names")}
            for k, f in (info.get("features") or {}).items()
        },
        "episodes": episodes,
    }


# ── train / eval split ───────────────────────────────────────────────────

def plan_eval_split(
    episodes: list[dict],
    keep_list: list[int],
    eval_split: float,
    seed: int = 42,
    mode: str = "random",
) -> dict:
    """Decide which kept episodes are held out for eval loss.

    LeRobot's `make_train_eval_datasets` groups the episode list it is
    given by task and holds out the LAST `ceil(n * eval_split)` of each
    group. It never sorts that list — `LeRobotDataset` stores it as
    given — so the ORDER passed in `--dataset.episodes` is what decides
    the split, and shuffling it here is enough to randomise the holdout
    without touching LeRobot.

    That matters because episode order is not neutral: an operator gets
    better across a session, and a resumed dataset ends with a different
    day's lighting and object placement. Holding out the last episodes
    therefore validates on the best, most recent demonstrations while
    training on the earliest ones — the two halves are not samples from
    the same distribution, and the eval curve stops meaning what it
    looks like it means.

    `mode="recent"` keeps the chronological behaviour for the case where
    validating on the newest conditions is the actual intent.

    Returns the episode ORDER to pass to LeRobot, plus the train and eval
    sets that order will produce.
    """
    import math
    import random

    kept = set(keep_list)
    order = [e["episode_index"] for e in episodes if e["episode_index"] in kept]
    if mode == "random":
        random.Random(int(seed)).shuffle(order)

    # Mirror factory.py exactly: group by task in the order given, then
    # slice each group's tail.
    task_of = {e["episode_index"]: (e["tasks"][0] if e.get("tasks") else "") for e in episodes}
    groups: dict[str, list[int]] = {}
    for ep in order:
        groups.setdefault(task_of.get(ep, ""), []).append(ep)

    train, val = [], []
    for eps in groups.values():
        n_eval = math.ceil(len(eps) * eval_split) if eval_split > 0 else 0
        train.extend(eps[: len(eps) - n_eval])
        val.extend(eps[len(eps) - n_eval:])

    return {
        "order": order,
        "train": sorted(train),
        "eval": sorted(val),
        "mode": mode,
        "seed": int(seed),
        "eval_split": eval_split,
    }


# ── dataset lifecycle ────────────────────────────────────────────────────
# Rename is a directory move: LeRobot v3.0's meta/info.json carries no
# repo_id of its own, so the path IS the identity. Delete is an rmtree.
# Both are cheap enough to run inline, unlike a prune (which re-encodes).

_REPO_ID_RE = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def validate_repo_id(repo_id: str) -> str:
    """Accept `name` or `namespace/name`, nothing that could climb out."""
    repo_id = (repo_id or "").strip().strip("/")
    if not repo_id:
        raise ValueError("dataset name cannot be empty")
    if not _REPO_ID_RE.match(repo_id):
        raise ValueError(
            f"{repo_id!r} is not a valid dataset name — use letters, digits, "
            ". _ - and at most one / (e.g. local/so101_pick_cube)"
        )
    return repo_id


def _forget(root: Path) -> None:
    """Drop cached reads for a dataset whose files just moved or vanished."""
    _frames_cache.pop(str(root), None)
    for key in [k for k, v in _detail_cache.items() if v[2].get("root") == str(root)]:
        _detail_cache.pop(key, None)


def rename_dataset(repo_id: str, new_repo_id: str) -> dict:
    new_repo_id = validate_repo_id(new_repo_id)
    root = dataset_root(repo_id)
    new_root = dataset_root(new_repo_id)
    if not root.exists():
        raise FileNotFoundError(f"no dataset at {root}")
    if new_root == root:
        raise ValueError("that is already the dataset's name")
    if new_root.exists():
        raise ValueError(f"{new_repo_id} already exists — pick another name")

    new_root.parent.mkdir(parents=True, exist_ok=True)
    root.rename(new_root)
    _forget(root)
    # An in-place prune leaves a `<name>_old` sibling; keep the pair together
    # so the backup does not end up orphaned under the previous name.
    backup = root.with_name(root.name + "_old")
    moved_backup = None
    if backup.exists():
        new_backup = new_root.with_name(new_root.name + "_old")
        if not new_backup.exists():
            backup.rename(new_backup)
            _forget(backup)
            moved_backup = str(new_backup)
    return {"repo_id": new_repo_id, "root": str(new_root), "moved_backup": moved_backup}


def delete_dataset(repo_id: str) -> dict:
    """Remove a dataset directory outright. There is no undo."""
    import shutil

    root = dataset_root(repo_id)
    if not root.exists():
        raise FileNotFoundError(f"no dataset at {root}")
    if root == hf_home():
        raise ValueError("refusing to delete the dataset cache root")
    size = _dir_size(root)
    shutil.rmtree(root)
    _forget(root)
    return {"repo_id": repo_id, "root": str(root), "freed_bytes": size}


# ── per-episode trace ────────────────────────────────────────────────────

def episode_trace(repo_id: str, episode: int, max_points: int = TRACE_MAX_POINTS) -> dict:
    """Downsampled action/state per joint for the timeline chart.

    Downsampling is by stride, not by averaging: the point of this chart
    is to show whether the gripper closed and whether the arm followed,
    and an average would smear exactly the short transitions that answer
    both questions.
    """
    root = dataset_root(repo_id)
    info = _info(root)
    if info is None:
        raise FileNotFoundError(f"no dataset at {root}")
    fps = int(info.get("fps", 30)) or 30

    import numpy as np

    df = _frames(root)
    sub = df[df["episode_index"] == int(episode)]
    if sub.empty:
        raise KeyError(f"episode {episode} not in {repo_id}")

    state = np.stack(sub["observation.state"].to_numpy())
    action = np.stack(sub["action"].to_numpy())
    n = len(state)
    stride = max(1, n // max_points)
    idx = np.arange(0, n, stride)

    names = (info.get("features", {}).get("observation.state", {}) or {}).get("names") or []
    if not names:
        names = [f"j{i}" for i in range(state.shape[1])]

    return {
        "episode_index": int(episode),
        "fps": fps,
        "frames": n,
        "seconds": n / fps,
        "names": [str(x) for x in names],
        "t": [float(i) / fps for i in idx],
        "state": [[float(v) for v in state[idx, j]] for j in range(state.shape[1])],
        "action": [[float(v) for v in action[idx, j]] for j in range(action.shape[1])],
    }


# ── video ────────────────────────────────────────────────────────────────

def video_path(repo_id: str, key: str, chunk_index: int = 0, file_index: int = 0) -> Path:
    """Path of the mp4 holding a given video key's chunk/file.

    The template lives in info.json (`video_path`), so this follows the
    dataset's own convention rather than assuming the default layout.
    """
    root = dataset_root(repo_id)
    info = _info(root)
    if info is None:
        raise FileNotFoundError(f"no dataset at {root}")
    if key not in _video_keys(info):
        raise KeyError(f"{key!r} is not a video feature of {repo_id}")
    template = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    rel = template.format(video_key=key, chunk_index=int(chunk_index), file_index=int(file_index))
    path = (root / rel).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("video path escapes the dataset root")
    if not path.exists():
        raise FileNotFoundError(f"no video at {path}")
    return path
