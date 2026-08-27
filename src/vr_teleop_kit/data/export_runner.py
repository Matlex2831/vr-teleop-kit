"""Subprocess entry point for pruning rejected episodes.

Two modes, both driven by LeRobot's own `delete_episodes`:

  copy      (default) write a NEW dataset containing only the kept
            episodes. The source is never touched, so a review decision
            can be revisited afterwards.
  in_place  rewrite the dataset itself, permanently dropping the
            rejected episodes. LeRobot moves the original aside to
            `<name>_old` first; `keep_backup: false` then deletes that
            too, which is the only genuinely unrecoverable path here and
            is gated behind a typed confirmation in the UI.

This runs as a job rather than inside a request because it re-encodes
video. A v3.0 dataset packs every episode into one mp4, so dropping an
episode from the middle means re-encoding the file around the hole —
minutes of AV1 at the dataset's own crf/preset settings, not
milliseconds.

Two things about the result are worth knowing, and the UI says both:

  * episodes are RENUMBERED 0..n-1, so "episode 4" before the prune is
    not "episode 4" afterwards;
  * review marks are therefore not carried over — a copy starts with
    none, and an in-place prune clears the source's own marks, because
    keeping them would silently attach old decisions to new episodes.

Usage:
    python -m vr_teleop_kit.data.export_runner /path/to/spec.json
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv[1:]
    if not args:
        print("usage: python -m vr_teleop_kit.data.export_runner SPEC.json")
        return 2

    spec = json.loads(Path(args[0]).read_text())
    run_dir = Path(spec["run_dir"])
    repo_id = spec["repo_id"]
    mode = spec.get("mode", "copy")
    new_repo_id = spec.get("new_repo_id")
    keep_backup = bool(spec.get("keep_backup", True))
    drop = sorted({int(e) for e in spec.get("delete_episodes") or []})

    if not drop:
        print("nothing to drop — every episode is kept")
        return 2
    if mode == "copy":
        if not new_repo_id:
            print("copy mode needs a new_repo_id")
            return 2
        if new_repo_id == repo_id:
            print("refusing to export onto the source dataset — pick a new repo-id")
            return 2
        print(f"export {repo_id} -> {new_repo_id}, dropping episodes {drop}", flush=True)
    else:
        print(
            f"PERMANENTLY deleting episodes {drop} from {repo_id}"
            + (" (a _old backup is kept)" if keep_backup else " (no backup kept)"),
            flush=True,
        )
    if dry_run:
        return 0

    from . import catalog, review, runs

    status, exit_code, error = "done", 0, ""
    try:
        if mode == "copy":
            _export_copy(repo_id, new_repo_id, drop)
        else:
            _prune_in_place(repo_id, drop, keep_backup)
            # The survivors are renumbered, so every stored mark now points
            # at a different episode than the one it was made about.
            review.clear(catalog.dataset_root(repo_id))
            print("review marks cleared (episodes were renumbered)", flush=True)
    except KeyboardInterrupt:
        status, exit_code, error = "stopped", 130, "interrupted"
    except Exception as e:
        status, exit_code, error = "failed", 1, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        runs.write_result(run_dir, status, exit_code, error)
    return exit_code


def _export_copy(repo_id: str, new_repo_id: str, drop: list[int]) -> None:
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from . import catalog

    output_dir = catalog.hf_home() / new_repo_id
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists — choose a different name rather than "
            "overwriting a dataset"
        )

    dataset = LeRobotDataset(repo_id, root=str(catalog.dataset_root(repo_id)))
    new_dataset = delete_episodes(
        dataset, episode_indices=drop, output_dir=output_dir, repo_id=new_repo_id,
    )
    print(
        f"wrote {new_repo_id}: {new_dataset.meta.total_episodes} episodes, "
        f"{new_dataset.meta.total_frames} frames -> {output_dir}",
        flush=True,
    )
    print(
        f"note: episodes were renumbered 0..{new_dataset.meta.total_episodes - 1}; "
        "the copy has no review marks yet.",
        flush=True,
    )


def _prune_in_place(repo_id: str, drop: list[int], keep_backup: bool) -> None:
    """Rewrite the dataset without the rejected episodes.

    Delegates to `lerobot-edit-dataset`'s own in-place handler so the
    backup-then-replace dance matches what that CLI does — including
    moving the original to `<name>_old` before writing. Deleting that
    backup is a separate, explicit step.
    """
    import shutil

    from lerobot.scripts.lerobot_edit_dataset import (
        DeleteEpisodesConfig,
        EditDatasetConfig,
        handle_delete_episodes,
    )

    from . import catalog

    root = catalog.dataset_root(repo_id)
    cfg = EditDatasetConfig(
        operation=DeleteEpisodesConfig(episode_indices=drop),
        repo_id=repo_id,
        root=str(root),
        new_repo_id=repo_id,
        new_root=str(root),
    )
    handle_delete_episodes(cfg)

    backup = root.with_name(root.name + "_old")
    if keep_backup:
        if backup.exists():
            print(f"previous version kept at {backup}", flush=True)
        return
    if backup.exists():
        # Only reached once the rewrite above succeeded, so this discards a
        # backup of a dataset that has already been replaced intact.
        shutil.rmtree(backup)
        print(f"backup {backup.name} deleted — this prune is not recoverable", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
