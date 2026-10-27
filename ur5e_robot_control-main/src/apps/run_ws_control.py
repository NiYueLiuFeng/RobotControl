"""Control the UR5e from a remote JSON-over-WebSocket policy server.

Each loop: read cameras + joint state -> send ``{"images": ..., "state": ...}``
-> receive ``{"action": [[...], ...]}`` -> execute the action chunk on the UR5e.
"""

from __future__ import annotations

import argparse
import base64
import json
from typing import Any

import numpy as np

from ur5e_control.data_collection.tasks import load_task_file
from ur5e_control.inference.websocket import connect_websocket
from ur5e_control.runtime.config import load_runtime_config, resolve_repo_path
from ur5e_control.runtime.env import build_robot_env

# Name sent to the policy server -> observation key produced by RobotEnv.
IMAGE_FIELDS = {
    "third_person": "third_left_top_rgb",
    "wrist": "wrist_rgb",
}
ACTION_DIM = 7


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
    vla = config.vla
    task = args.task or load_task_file(str(resolve_repo_path(config.task.file))).name

    bundle = build_robot_env(config)
    connection = connect_websocket(
        args.server_url or vla.server_url,
        args.api_key or vla.api_key,
        vla.request_timeout_s,
    )
    try:
        _move_to_start(bundle.env, config.robot.reset_joints)
        if not args.no_wait:
            input("press ENTER to start policy control loop...")
        print("Starting UR5e policy control. Press Ctrl+C to stop.")
        steps = _run(
            bundle.env,
            connection,
            repo_id=vla.repo_id,
            task=task,
            execute_horizon=vla.execute_horizon,
            threshold=vla.threshold,
            max_steps=args.max_steps,
        )
        print(f"Stopped after executing {steps} action(s).")
    except KeyboardInterrupt:
        print("Stopping UR5e policy control.")
    finally:
        connection.close()
        bundle.close()


def _run(
    env: Any,
    connection: Any,
    *,
    repo_id: str,
    task: str,
    execute_horizon: int,
    threshold: float | None,
    max_steps: int | None,
) -> int:
    """Run the request/execute loop and return how many commands were sent."""

    executed = 0
    while max_steps is None or executed < max_steps:
        request = _build_request(env.get_obs(), repo_id=repo_id, task=task)
        response = _request(connection, request)
        for action in _parse_actions(response)[:execute_horizon]:
            env.step(_robot_command(action, threshold))
            executed += 1
            if max_steps is not None and executed >= max_steps:
                break
    return executed


def _request(connection: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one JSON request and parse the JSON response."""

    connection.send(json.dumps(payload))
    response = json.loads(connection.recv())
    if not isinstance(response, dict):
        raise RuntimeError(
            f"policy server returned {type(response).__name__}, expected an object"
        )
    if response.get("ok") is False:
        raise RuntimeError(str(response.get("error", "policy server error")))
    return response


def _build_request(
    obs: dict[str, Any], *, repo_id: str, task: str, jpeg_quality: int = 90
) -> dict[str, Any]:
    """Build the observation body uploaded to the policy server."""

    return {
        "repo_id": repo_id,
        "task": task,
        "state": [float(value) for value in obs["joint_positions"]],
        "images": {
            name: _encode_jpeg(obs[key], jpeg_quality)
            for name, key in IMAGE_FIELDS.items()
        },
    }


def _parse_actions(response: dict[str, Any]) -> np.ndarray:
    """Read `{"action": [[...], [...]]}` into a validated `(N, 7)` array."""

    if "action" not in response:
        raise KeyError("policy response is missing 'action'")

    actions = np.asarray(response["action"], dtype=float)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            "policy response 'action' must be a list of "
            f"{ACTION_DIM}-value actions, got shape {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("policy response 'action' must contain finite values")
    return actions


def _encode_jpeg(image: Any, quality: int = 90) -> str:
    """Encode an RGB `HxWx3` frame as a base64 JPEG string."""

    import cv2  # imported lazily to keep module imports hardware-free

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {rgb.shape}")

    bgr = np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)
    ok, buffer = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("failed to encode camera image as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _robot_command(action: np.ndarray, threshold: float | None) -> np.ndarray:
    """Convert a policy action into a UR5e joint command (gripper 0 or 1000)."""

    command = np.asarray(action, dtype=float).copy()
    limit = 0.5 if threshold is None else float(threshold)
    command[6] = 1000.0 if command[6] >= limit else 0.0
    return command


def _move_to_start(env: Any, joints: Any, steps: int = 100) -> None:
    """Move the arm to the configured start pose in small joint steps."""

    target = np.asarray(joints, dtype=float)
    current = np.asarray(env.get_obs()["joint_positions"], dtype=float)
    if current.shape != target.shape:
        return
    for point in np.linspace(current, target, steps):
        env.step(point)


if __name__ == "__main__":
    main()
