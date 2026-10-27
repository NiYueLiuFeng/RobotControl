"""Control the UR5e from a remote JSON-over-WebSocket policy server.

Loop: read observations -> POST `{"images": ..., "state": ...}` -> receive
`{"action": [[...], [...]]}` -> execute the action chunk on the robot.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from ur5e_control.inference.json_protocol import build_request, parse_actions
from ur5e_control.inference.json_ws_client import JsonWebSocketClient
from ur5e_control.runtime.config import (
    RuntimeConfig,
    load_runtime_config,
    resolve_repo_path,
)
from ur5e_control.runtime.env import build_robot_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Runtime YAML config. Defaults to configs/hardware/ur5e_d455.yaml.",
    )
    parser.add_argument("--server-url", default=None, help="Policy WebSocket URL.")
    parser.add_argument("--api-key", default=None, help="Optional policy API key.")
    parser.add_argument("--task", default=None, help="Task text override.")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Stop after this many actions."
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Skip the ENTER prompt before the loop."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_runtime_config(args.config)
    task = args.task or _task_text(config)

    bundle = build_robot_env(config)
    client = JsonWebSocketClient(
        args.server_url or config.vla.server_url,
        args.api_key or config.vla.api_key,
        config.vla.request_timeout_s,
    )
    try:
        _move_to_start(bundle.env, config.robot.reset_joints)
        if not args.no_wait:
            input("press ENTER to start policy control loop...")
        print("Starting UR5e policy control. Press Ctrl+C to stop.")
        steps = _run(bundle.env, client, config, task, args.max_steps)
        print(f"Stopped after executing {steps} action(s).")
    except KeyboardInterrupt:
        print("Stopping UR5e policy control.")
    finally:
        client.close()
        bundle.close()


def _run(
    env: Any,
    client: JsonWebSocketClient,
    config: RuntimeConfig,
    task: str,
    max_steps: int | None,
) -> int:
    executed = 0
    while max_steps is None or executed < max_steps:
        request = build_request(
            env.get_obs(),
            repo_id=config.vla.repo_id,
            task=task,
        )
        actions = parse_actions(client.request(request))[: config.vla.execute_horizon]
        for action in actions:
            env.step(_robot_command(action, config.vla.threadhold))
            executed += 1
            if max_steps is not None and executed >= max_steps:
                return executed
    return executed


def _robot_command(action: np.ndarray, gripper_threshold: float | None) -> np.ndarray:
    """Convert a policy action into a UR5e joint command (gripper 0 or 1000)."""

    command = np.asarray(action, dtype=float).copy()
    threshold = 0.5 if gripper_threshold is None else float(gripper_threshold)
    command[6] = 1000.0 if command[6] >= threshold else 0.0
    return command


def _move_to_start(env: Any, joints: Any, steps: int = 100) -> None:
    """Move the arm to the configured start pose in small joint steps."""

    target = np.asarray(joints, dtype=float)
    current = np.asarray(env.get_obs()["joint_positions"], dtype=float)
    if current.shape != target.shape:
        return
    for point in np.linspace(current, target, steps):
        env.step(point)


def _task_text(config: RuntimeConfig) -> str:
    from ur5e_control.data_collection.tasks import load_task_file

    task_file = resolve_repo_path(config.task.file)
    return load_task_file(str(task_file)).name


if __name__ == "__main__":
    main()
