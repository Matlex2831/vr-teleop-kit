"""Subprocess entry point for a training run launched from the web UI.

Runs LeRobot's own `lerobot-train` — same argument parsing, same code
path — with one addition: a logging handler that records every metric
the training loop reports into `metrics.jsonl`, so the page can draw
train and eval loss without wandb, a second server, or an account.

The handler is the reason this wrapper exists. LeRobot logs progress
with `logging.info(train_tracker)`, passing the `MetricsTracker`
*object* as the log message; the terminal only ever sees its `__str__`,
which rounds through `format_big_number` — `step:1K` for step 1000, and
`step:12K` for anything from 11,500 to 12,499. Scraping that back out
of stdout would give a chart with a made-up x-axis. Reaching for
`record.msg.to_dict()` instead yields the exact numbers the trainer
actually holds.

Usage (the UI does this via `runs.launch`):
    python -m vr_teleop_kit.data.train_runner /path/to/spec.json
    python -m vr_teleop_kit.data.train_runner /path/to/spec.json --dry-run
"""

from __future__ import annotations

import json
import logging
import re
import sys
import traceback
from pathlib import Path

# `step 1200: eval_loss=0.0431` — the one metric the trainer reports as a
# formatted string rather than through the tracker.
_EVAL_RE = re.compile(r"step\s+(\d+):\s*eval_loss=([0-9.eE+-]+)")
# `Train/eval split: 4 train, 1 eval (eval_split=0.2, 1 tasks)`
_SPLIT_RE = re.compile(r"Train/eval split:\s*(\d+)\s+train,\s*(\d+)\s+eval")


class JsonlMetricsHandler(logging.Handler):
    """Append every training metric to `metrics.jsonl` as it is logged."""

    def __init__(self, path: Path) -> None:
        super().__init__(level=logging.INFO)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # line buffered
        try:
            from lerobot.utils.logging_utils import MetricsTracker
            self._tracker_cls = MetricsTracker
        except Exception:  # pragma: no cover - lerobot layout change
            self._tracker_cls = ()

    def _write(self, row: dict) -> None:
        self._fh.write(json.dumps(row) + "\n")

    def emit(self, record: logging.LogRecord) -> None:
        # A logging handler must never take the training run down with it.
        try:
            msg = record.msg
            if self._tracker_cls and isinstance(msg, self._tracker_cls):
                row = {"kind": "train"}
                row.update(msg.to_dict())
                self._write(row)
                return
            if not isinstance(msg, str):
                return
            text = record.getMessage()
            m = _EVAL_RE.search(text)
            if m:
                self._write({
                    "kind": "eval",
                    "steps": int(m.group(1)),
                    "eval_loss": float(m.group(2)),
                })
                return
            m = _SPLIT_RE.search(text)
            if m:
                self._write({
                    "kind": "split",
                    "train_episodes": int(m.group(1)),
                    "eval_episodes": int(m.group(2)),
                })
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def build_argv(spec: dict) -> list[str]:
    """Translate a UI spec into `lerobot-train` arguments.

    Only flags the UI actually offers are emitted; `extra_args` is passed
    through verbatim so anything LeRobot supports stays reachable without
    this form having to grow a field for it.
    """
    run_dir = Path(spec["run_dir"])
    output_dir = run_dir / "train"

    argv = ["lerobot-train", f"--dataset.repo_id={spec['repo_id']}"]

    episodes = spec.get("episodes")
    if episodes is not None:
        # The kept set from the review page. Rejected episodes are not
        # deleted — they are simply never handed to the trainer.
        argv.append("--dataset.episodes=" + json.dumps([int(e) for e in episodes], separators=(",", ":")))

    eval_split = float(spec.get("eval_split") or 0.0)
    if eval_split > 0:
        argv.append(f"--dataset.eval_split={eval_split}")

    argv += [
        f"--policy.type={spec.get('policy_type', 'act')}",
        f"--policy.device={spec.get('device', 'cuda')}",
        "--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--job_name={spec.get('job_name') or spec['run_id']}",
        f"--steps={int(spec.get('steps', 100_000))}",
        f"--batch_size={int(spec.get('batch_size', 8))}",
        f"--log_freq={int(spec.get('log_freq', 200))}",
        f"--save_freq={int(spec.get('save_freq', 20_000))}",
        f"--num_workers={int(spec.get('num_workers', 4))}",
        "--wandb.enable=false",
    ]

    eval_steps = int(spec.get("eval_steps") or 0)
    if eval_steps > 0 and eval_split > 0:
        argv.append(f"--eval_steps={eval_steps}")
        # Without a cap, every eval pass walks the ENTIRE held-out set,
        # decoding video for each frame. On a few thousand held-out
        # frames that is longer than the training steps between passes.
        max_eval = int(spec.get("max_eval_samples") or 0)
        if max_eval > 0:
            argv.append(f"--max_eval_samples={max_eval}")

    # Environment-based policy evaluation needs a simulator; there isn't
    # one here, and leaving it at its 20k default would stall the run.
    argv.append("--env_eval_freq=0")

    for extra in spec.get("extra_args") or []:
        argv.append(str(extra))
    return argv


def main() -> int:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    if not args:
        print("usage: python -m vr_teleop_kit.data.train_runner SPEC.json [--dry-run]")
        return 2

    spec = json.loads(Path(args[0]).read_text())
    run_dir = Path(spec["run_dir"])
    argv = build_argv(spec)

    if dry_run:
        print(" ".join(argv))
        return 0

    from . import runs

    status, exit_code, error = "done", 0, ""
    handler = JsonlMetricsHandler(run_dir / "metrics.jsonl")
    try:
        import lerobot.scripts.lerobot_train as lerobot_train

        # init_logging() replaces the root handlers, so attach ours after
        # it has run — which is the first thing train() does.
        logging.getLogger().addHandler(handler)
        _install_handler_after_init_logging(handler)

        print("training argv:", " ".join(argv), flush=True)
        sys.argv = argv
        lerobot_train.main()
    except KeyboardInterrupt:
        status, exit_code, error = "stopped", 130, "interrupted"
        print("\ninterrupted — stopping", flush=True)
    except SystemExit as e:
        exit_code = int(e.code or 0)
        status = "done" if exit_code == 0 else "failed"
        error = "" if exit_code == 0 else f"exited with {exit_code}"
    except Exception as e:
        status, exit_code, error = "failed", 1, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        handler.close()
        runs.write_result(run_dir, status, exit_code, error)
    return exit_code


def _install_handler_after_init_logging(handler: logging.Handler) -> None:
    """Re-attach `handler` whenever LeRobot resets the root logger.

    `lerobot.utils.utils.init_logging()` clears the root handlers and
    installs its own, which would silently drop ours and leave the run
    with an empty metrics file — the failure would only show up as a
    blank chart an hour into training.
    """
    try:
        from lerobot.utils import utils as lerobot_utils
    except Exception:  # pragma: no cover
        return
    original = lerobot_utils.init_logging

    def patched(*args, **kwargs):
        result = original(*args, **kwargs)
        root = logging.getLogger()
        if handler not in root.handlers:
            root.addHandler(handler)
        return result

    lerobot_utils.init_logging = patched
    # train() imported the symbol directly, so patch it there too.
    try:
        import lerobot.scripts.lerobot_train as lerobot_train
        if getattr(lerobot_train, "init_logging", None) is original:
            lerobot_train.init_logging = patched
    except Exception:  # pragma: no cover
        pass


if __name__ == "__main__":
    raise SystemExit(main())
