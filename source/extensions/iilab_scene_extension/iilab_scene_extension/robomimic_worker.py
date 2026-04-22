from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import traceback

import numpy as np
import torch


def _emit(message: dict) -> None:
    print(json.dumps(message), flush=True)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Robomimic policy inference worker.")
    parser.add_argument("--checkpoint", required=True, help="Path to the robomimic checkpoint.")
    parser.add_argument("--norm-factor-min", type=float, default=None)
    parser.add_argument("--norm-factor-max", type=float, default=None)
    return parser


def _unnormalize_action(actions: np.ndarray, norm_factor_min: float | None, norm_factor_max: float | None) -> np.ndarray:
    if norm_factor_min is None or norm_factor_max is None:
        return actions
    return ((actions + 1.0) * (norm_factor_max - norm_factor_min)) / 2.0 + norm_factor_min


def _mute_stdout():
    return contextlib.redirect_stdout(io.StringIO())


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    try:
        with _mute_stdout():
            import robomimic.utils.file_utils as file_utils

            policy, ckpt_dict = file_utils.policy_from_checkpoint(ckpt_path=args.checkpoint, device=torch.device("cpu"))
            policy.start_episode()
            shape_meta = ckpt_dict["shape_metadata"]
        _emit(
            {
                "status": "ready",
                "obs_shapes": shape_meta.get("all_shapes", {}),
                "obs_keys": shape_meta.get("all_obs_keys", []),
                "ac_dim": int(shape_meta.get("ac_dim", 0)),
            }
        )
    except Exception:
        _emit({"status": "error", "message": traceback.format_exc()})
        return 1

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
            command = message.get("cmd")

            if command == "reset":
                with _mute_stdout():
                    policy.start_episode()
                _emit({"status": "ok"})
            elif command == "infer":
                observation = {
                    key: torch.tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32)
                    for key, value in message.get("obs", {}).items()
                }
                with _mute_stdout():
                    policy_output = policy(observation)
                action = np.asarray(policy_output, dtype=np.float32).reshape(-1)
                action = _unnormalize_action(action, args.norm_factor_min, args.norm_factor_max)
                _emit({"status": "ok", "action": action.tolist()})
            elif command == "close":
                _emit({"status": "ok"})
                return 0
            else:
                raise RuntimeError(f"Unsupported worker command: {command}")
        except Exception:
            _emit({"status": "error", "message": traceback.format_exc()})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
