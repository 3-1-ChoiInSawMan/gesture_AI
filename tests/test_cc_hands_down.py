"""Exercise the actual socket handler without loading model weights."""

import ast
import asyncio
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock


class Disconnected(Exception):
    pass


class HandsDownTests(unittest.IsolatedAsyncioTestCase):
    def make_handler(self, events, consume, *, lowered=True):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "router/cc.py").read_text(encoding="utf-8"))
        handler = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "jamak")
        handler.decorator_list = []
        frame = object()
        payload = SimpleNamespace(shape=(88,))
        ws = SimpleNamespace(
            accept=AsyncMock(), query_params={},
            receive_json=AsyncMock(side_effect=events), send_json=AsyncMock(),
        )
        namespace = dict(
            WebSocket=object, WebSocketDisconnect=Disconnected, asyncio=asyncio,
            deque=deque, uuid4=lambda: "session", logger=Mock(),
            np=SimpleNamespace(zeros=lambda *args, **kwargs: frame, float32=float,
                               ndarray=object, asarray=lambda *args, **kwargs: payload),
            WINDOW_SIZE=30, INPUT_SIZE=88, CC_SMOOTHING_WINDOW=7,
            CC_SILENCE_TIMEOUT_SECONDS=1, CC_HANDS_DOWN_MIN_FRAMES=6,
            _extract_keypoints_payload=lambda data: data["keypoints"],
            _is_valid_frame=lambda frame: True, _are_hands_lowered=lambda frame: lowered,
            _send_debug=AsyncMock(), _flush_words=AsyncMock(return_value=[]),
            _consume_frame=consume, CC_NONE_MIN_PREDICTIONS=6,
        )
        exec(compile(ast.Module(body=[handler], type_ignores=[]), "<cc handler>", "exec"), namespace)
        return namespace["jamak"], ws, namespace

    async def test_lowered_frames_never_reach_classifier(self):
        consume = Mock(side_effect=AssertionError("lowered hands reached classifier"))
        handler, ws, _ = self.make_handler(
            [{"keypoints": [1]}] * 40 + [Disconnected()], consume
        )
        await handler(ws)
        consume.assert_not_called()

    async def test_timeout_clears_previous_sequence_and_votes(self):
        snapshots = []

        def consume(frame, **state):
            snapshots.append((list(state["seq_buffer"]),
                              list(state["valid_flag_buffer"]),
                              list(state["pred_history"])))
            state["seq_buffer"].append(frame)
            state["valid_flag_buffer"].append(True)
            state["pred_history"].append(1)
            return ["hello"], frame, {"reason": "gesture"}

        handler, ws, namespace = self.make_handler(
            [{"keypoints": [1]}, asyncio.TimeoutError(),
             {"keypoints": [1]}, Disconnected()], consume, lowered=False
        )
        await handler(ws)
        self.assertEqual(snapshots, [([], [], []), ([], [], [])])
        self.assertEqual(ws.send_json.await_count, 2)
        self.assertEqual(namespace["_flush_words"].call_args_list[0].args[2], [["hello"]])


if __name__ == "__main__":
    unittest.main()
