"""Request/response fields for the simple JSON policy WebSocket protocol.

Only `build_request` and `parse_actions` describe the wire format. Adjust those
two functions when the remote API changes.
"""

from __future__ import annotations

import base64
from typing import Any, Mapping

import numpy as np

# Logical image name sent to the server -> observation key produced by RobotEnv.
DEFAULT_IMAGE_FIELDS = {
    "third_person": "third_left_top_rgb",
    "wrist": "wrist_rgb",
}

ACTION_DIM = 7


def build_request(
    env_obs: Mapping[str, Any],
    *,
    repo_id: str,
    task: str,
    image_fields: Mapping[str, str] = DEFAULT_IMAGE_FIELDS,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Build the observation body uploaded to the policy server."""

    return {
        "repo_id": repo_id,
        "task": task,
        "state": [float(value) for value in env_obs["joint_positions"]],
        "images": {
            name: encode_jpeg_base64(env_obs[env_key], jpeg_quality)
            for name, env_key in image_fields.items()
        },
    }


def parse_actions(response: Mapping[str, Any]) -> np.ndarray:
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


def encode_jpeg_base64(image: Any, quality: int = 90) -> str:
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
