"""Keep/reject marks for recorded episodes — a sidecar, not a rewrite.

Deciding an episode is bad and *destroying* it are two different acts,
and a review UI that fuses them is one misclick away from losing a
demonstration that took a minute of your life to record. So marks live
in `review.json` at the dataset root and change nothing else: training
simply passes the kept set as `--dataset.episodes`, and writing a
pruned dataset is a separate, explicit export.

The file sits at the dataset ROOT, not under `meta/` — that directory
belongs to LeRobot's own loaders and is not ours to litter.

Layout:

    {
      "version": 1,
      "updated": "2026-08-26T18:40:00Z",
      "fingerprint": {"total_episodes": 5, "total_frames": 4066},
      "episodes": {"1": {"status": "reject", "note": "missed grasp"}}
    }

Only marked episodes are stored; anything absent is `unset`, which
counts as KEEP. That way a fresh dataset trains on everything, and the
file stays a record of decisions you actually made.

Each mark also records the LENGTH of the episode it was made about,
which is what makes a mark verifiable later. `delete_episodes` renumbers
the survivors 0..n-1, so marks written before a prune silently describe
*different* episodes afterwards — a "reject" can become a keep-worthy
demonstration. Comparing each mark's recorded length against the episode
now at that index catches it per mark.

Comparing dataset TOTALS cannot do this job, and the earlier version
that tried got it wrong in both directions: appending episodes with
`--resume` changes the totals without renumbering anything (a false
alarm on every recording session), while a single later click would
overwrite the stored totals and hide a real renumbering. Lengths are
per-episode and are not disturbed by an append.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path

REVIEW_FILENAME = "review.json"
REVIEW_VERSION = 1

KEEP = "keep"
REJECT = "reject"
UNSET = "unset"
STATUSES = (KEEP, REJECT, UNSET)


def review_path(root: Path | str) -> Path:
    return Path(root) / REVIEW_FILENAME


def load(root: Path | str) -> dict:
    """Read the sidecar. A missing or corrupt file reads as empty — a
    review file is an annotation, never a thing worth failing a page load
    over."""
    path = review_path(root)
    if not path.exists():
        return {"version": REVIEW_VERSION, "updated": None, "fingerprint": {}, "episodes": {}}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {"version": REVIEW_VERSION, "updated": None, "fingerprint": {}, "episodes": {}}
    data.setdefault("version", REVIEW_VERSION)
    data.setdefault("episodes", {})
    data.setdefault("fingerprint", {})
    data.setdefault("updated", None)
    return data


def save(root: Path | str, data: dict) -> None:
    """Atomic write — a half-written review.json read by a concurrent
    page load would look like corruption and reset every mark."""
    path = review_path(root)
    data["version"] = REVIEW_VERSION
    data["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".review-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def set_status(
    root: Path | str,
    episode: int,
    status: str,
    note: str | None = None,
    fingerprint: dict | None = None,
    episode_frames: int | None = None,
) -> dict:
    """Mark one episode. `unset` with no note removes the entry entirely
    so the file only ever holds real decisions."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    data = load(root)
    key = str(int(episode))
    entry = dict(data["episodes"].get(key) or {})
    entry["status"] = status
    if note is not None:
        entry["note"] = note
    if episode_frames is not None:
        entry["frames"] = int(episode_frames)
    if status == UNSET and not entry.get("note"):
        data["episodes"].pop(key, None)
    else:
        data["episodes"][key] = entry
    if fingerprint:
        data["fingerprint"] = fingerprint
    save(root, data)
    return data


def set_many(
    root: Path | str,
    marks: dict[int, str],
    fingerprint: dict | None = None,
    note: str | None = None,
    episode_frames: dict[int, int] | None = None,
) -> dict:
    """Apply several marks in one write (the auto-grade bulk action)."""
    data = load(root)
    for episode, status in marks.items():
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        key = str(int(episode))
        entry = dict(data["episodes"].get(key) or {})
        entry["status"] = status
        if note is not None:
            entry["note"] = note
        if episode_frames and int(episode) in episode_frames:
            entry["frames"] = int(episode_frames[int(episode)])
        if status == UNSET and not entry.get("note"):
            data["episodes"].pop(key, None)
        else:
            data["episodes"][key] = entry
    if fingerprint:
        data["fingerprint"] = fingerprint
    save(root, data)
    return data


def clear(root: Path | str, fingerprint: dict | None = None) -> dict:
    data = load(root)
    data["episodes"] = {}
    if fingerprint:
        data["fingerprint"] = fingerprint
    save(root, data)
    return data


def status_of(data: dict, episode: int) -> str:
    return (data.get("episodes", {}).get(str(int(episode))) or {}).get("status", UNSET)


def note_of(data: dict, episode: int) -> str:
    return (data.get("episodes", {}).get(str(int(episode))) or {}).get("note", "")


def keep_list(data: dict, total_episodes: int) -> list[int]:
    """Episode indices that training should see: everything not rejected."""
    return [i for i in range(total_episodes) if status_of(data, i) != REJECT]


def stale_marks(data: dict, episode_frames: dict[int, int]) -> list[int]:
    """Marked episodes that are no longer the episode that was marked.

    `episode_frames` maps the dataset's CURRENT episode indices to their
    lengths. A mark is suspect when the episode it names has a different
    length than it did when marked, or has vanished off the end — both
    are what renumbering after a prune looks like from here.

    Marks written before lengths were recorded carry no `frames` and can
    only be checked for pointing past the end.
    """
    suspect = []
    for key, entry in (data.get("episodes") or {}).items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        current = episode_frames.get(idx)
        if current is None:
            suspect.append(idx)          # marked an episode that is gone
            continue
        recorded = entry.get("frames")
        if recorded is not None and int(recorded) != int(current):
            suspect.append(idx)
    return sorted(suspect)


def is_stale(data: dict, fingerprint: dict, episode_frames: dict[int, int] | None = None) -> bool:
    """True when at least one mark can no longer be trusted.

    Deliberately NOT "the dataset changed size": recording more episodes
    with `--resume` appends them, leaving every existing index exactly
    where it was, and warning about that on every session would train
    people to ignore the warning that matters.
    """
    if not data.get("episodes"):
        return False
    if episode_frames is not None:
        return bool(stale_marks(data, episode_frames))
    # Legacy fallback for review files written before per-mark lengths:
    # only a SHRINK implies episodes were removed and renumbered.
    stored = data.get("fingerprint") or {}
    if not stored:
        return False
    return any(
        k in stored and fingerprint.get(k, 0) < stored[k]
        for k in ("total_episodes", "total_frames")
    )


def counts(data: dict, total_episodes: int) -> dict:
    kept = rejected = unset = 0
    for i in range(total_episodes):
        s = status_of(data, i)
        if s == REJECT:
            rejected += 1
        elif s == KEEP:
            kept += 1
        else:
            unset += 1
    return {"keep": kept, "reject": rejected, "unset": unset, "train": kept + unset}
