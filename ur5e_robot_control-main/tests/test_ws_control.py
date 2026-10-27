import base64
import json
import unittest
from types import SimpleNamespace

import numpy as np


class FakeConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class FakeEnv:
    def __init__(self, obs):
        self.obs = obs
        self.commands = []

    def get_obs(self):
        return self.obs

    def step(self, action):
        self.commands.append(np.asarray(action, dtype=float))
        return self.obs


class WSPolicyControlTest(unittest.TestCase):
    def test_build_request_encodes_state_and_images(self):
        from ur5e_control.inference.json_protocol import build_request

        obs = {
            "joint_positions": np.array([0, 1, 2, 3, 4, 5, 1], dtype=float),
            "third_left_top_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_rgb": np.full((8, 8, 3), 255, dtype=np.uint8),
        }

        request = build_request(obs, repo_id="repo", task="pick the pen")

        self.assertEqual(request["repo_id"], "repo")
        self.assertEqual(request["task"], "pick the pen")
        self.assertEqual(request["state"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0])
        self.assertEqual(set(request["images"]), {"third_person", "wrist"})
        # JPEG frames start with the SOI marker after base64 decoding.
        self.assertTrue(
            base64.b64decode(request["images"]["wrist"]).startswith(b"\xff\xd8")
        )

    def test_parse_actions_accepts_action_chunk(self):
        from ur5e_control.inference.json_protocol import parse_actions

        actions = parse_actions(
            {"action": [[0, 1, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5, 6, 0]]}
        )

        self.assertEqual(actions.shape, (2, 7))

    def test_parse_actions_rejects_wrong_dimension(self):
        from ur5e_control.inference.json_protocol import parse_actions

        with self.assertRaisesRegex(ValueError, "7-value actions"):
            parse_actions({"action": [[0, 1, 2]]})

    def test_client_sends_json_and_reads_action(self):
        from ur5e_control.inference.json_ws_client import JsonWebSocketClient

        connection = FakeConnection([json.dumps({"action": [[0] * 7]})])
        client = JsonWebSocketClient(
            "ws://policy", connection_factory=lambda *_: connection
        )

        response = client.request({"state": [0] * 7})
        client.close()

        self.assertEqual(json.loads(connection.sent[0]), {"state": [0] * 7})
        self.assertEqual(response, {"action": [[0] * 7]})
        self.assertTrue(connection.closed)

    def test_loop_executes_returned_action_chunk(self):
        from apps.run_ws_control import _run

        class FakeClient:
            def request(self, payload):
                return {"action": [[0.1] * 6 + [0.9], [0.2] * 6 + [0.1]]}

        config = SimpleNamespace(
            vla=SimpleNamespace(repo_id="repo", execute_horizon=2, threadhold=0.7)
        )
        env = FakeEnv(
            {
                "joint_positions": np.zeros(7),
                "third_left_top_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                "wrist_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        )

        executed = _run(env, FakeClient(), config, "pick", max_steps=2)

        self.assertEqual(executed, 2)
        self.assertEqual(len(env.commands), 2)
        self.assertEqual(env.commands[0][-1], 1000.0)
        self.assertEqual(env.commands[1][-1], 0.0)


if __name__ == "__main__":
    unittest.main()
