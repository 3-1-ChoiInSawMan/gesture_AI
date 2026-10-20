"""Exercise the actual STT handler without loading Whisper or GPU weights."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock


class FakeAudioBuffer:
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.data = bytearray()
        self.pending = False

    def append(self, chunk):
        if len(self.data) + len(chunk) > self.max_bytes:
            raise BufferError()
        self.data.extend(chunk)
        self.pending = True

    def take_audio(self):
        result = bytes(self.data)
        self.clear()
        return result

    def clear(self):
        self.data.clear()
        self.pending = False


class STTUtteranceTests(unittest.IsolatedAsyncioTestCase):
    def make_handler(self, events, *, transcribe=None, limit=1024):
        source = Path(__file__).resolve().parents[1] / "router" / "cc_stt.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)]
        for function in functions:
            function.decorator_list = []
        ws = SimpleNamespace(
            accept=AsyncMock(), query_params={}, receive=AsyncMock(side_effect=events),
            send_text=AsyncMock(), close=AsyncMock(),
        )
        transcribe = transcribe or Mock(return_value="recognized text")
        namespace = dict(
            asyncio=asyncio, json=json, WebSocket=object, WebSocketDisconnect=ConnectionError,
            uuid4=lambda: "test-session", logger=Mock(), STT_SILENCE_TIMEOUT_SECONDS=1,
            STT_MAX_UTTERANCE_BYTES=limit, _transcribe_audio=transcribe,
            STTAudioBuffer=FakeAudioBuffer,
        )
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        return namespace["stt_cc"], ws, transcribe

    async def test_long_utterance_infers_once_at_timeout_and_keeps_all_chunks(self):
        chunks = [bytes([i]) for i in range(30)]
        events = [{"type": "websocket.receive", "bytes": chunk} for chunk in chunks]
        events += [asyncio.TimeoutError(), asyncio.TimeoutError(), {"type": "websocket.disconnect"}]
        handler, ws, transcribe = self.make_handler(events)

        async def receive():
            event = events.pop(0)
            if isinstance(event, Exception):
                raise event
            if event.get("bytes"):
                transcribe.assert_not_called()
            return event

        ws.receive.side_effect = receive
        await handler(ws)
        transcribe.assert_called_once_with(b"".join(chunks))
        ws.send_text.assert_awaited_once_with("recognized text")

    async def test_explicit_end_separates_utterances_and_empty_end_is_ignored(self):
        end = {"type": "websocket.receive", "text": '{"type":"end_of_utterance"}'}
        handler, ws, transcribe = self.make_handler([
            {"type": "websocket.receive", "bytes": b"first"}, end, end,
            {"type": "websocket.receive", "bytes": b"second"}, end,
            {"type": "websocket.disconnect"},
        ])
        await handler(ws)
        self.assertEqual([call.args[0] for call in transcribe.call_args_list], [b"first", b"second"])
        self.assertEqual(ws.send_text.await_count, 2)

    async def test_failed_inference_is_not_retried(self):
        transcribe = Mock(side_effect=ValueError("bad audio"))
        handler, ws, _ = self.make_handler([
            {"type": "websocket.receive", "bytes": b"audio"},
            asyncio.TimeoutError(), asyncio.TimeoutError(), {"type": "websocket.disconnect"},
        ], transcribe=transcribe)
        await handler(ws)
        transcribe.assert_called_once_with(b"audio")
        ws.send_text.assert_not_awaited()

    async def test_disconnect_discards_unfinished_audio(self):
        handler, ws, transcribe = self.make_handler([
            {"type": "websocket.receive", "bytes": b"audio"}, {"type": "websocket.disconnect"},
        ])
        await handler(ws)
        transcribe.assert_not_called()

    async def test_buffer_limit_closes_without_partial_inference(self):
        handler, ws, transcribe = self.make_handler([
            {"type": "websocket.receive", "bytes": b"too much audio"},
        ], limit=4)
        await handler(ws)
        transcribe.assert_not_called()
        self.assertEqual(ws.close.await_args.kwargs["code"], 1009)


if __name__ == "__main__":
    unittest.main()
