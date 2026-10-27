# Third-Party Dependencies

Third-party sources are **not** tracked in this repository. Only the directory
layout is kept so the expected install locations stay visible.

| Directory | Purpose | Install |
| --- | --- | --- |
| `DynamixelSDK/` | GELLO leader-arm Dynamixel driver | `git submodule update --init third_party/DynamixelSDK` then `uv pip install -e third_party/DynamixelSDK/python` |
| `mujoco_menagerie/` | UR5e model for the optional SpaceMouse backend | `git submodule update --init third_party/mujoco_menagerie` |

Contents of these directories are ignored by `.gitignore`; the directory names
are preserved through placeholder files (`.gitkeep`).
