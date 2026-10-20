"""One hands-up segment must produce at most one confirmed word."""
import ast
import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock


class GestureSegmentTests(unittest.IsolatedAsyncioTestCase):
    def make_flush(self):
        tree = ast.parse((Path(__file__).resolve().parents[1]/'router/cc.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_flush_words')
        generate = Mock(return_value='sentence')
        ns = dict(WebSocket=object, Counter=Counter, asyncio=asyncio, logger=Mock(),
                  generate_sentence_from_words=generate, store_final_sentence=Mock(),
                  WebSocketDisconnect=RuntimeError)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual flush>','exec'),ns)
        return ns['_flush_words'], generate

    async def test_fluctuating_predictions_emit_only_winner(self):
        flush, generate = self.make_flush()
        ws = SimpleNamespace(send_json=AsyncMock())
        result = await flush(ws,'session',[[w] for w in ['저','저','의견','저','저']],emit=True)
        self.assertEqual(result,[])
        generate.assert_called_once_with([['저']])
        messages = [c.args[0] for c in ws.send_json.call_args_list]
        self.assertEqual([m['word'] for m in messages if m['type']=='word'],['저'])

    async def test_ambiguous_or_brief_gestures_are_discarded(self):
        for predictions in ([],['저'],['저','저'],['저','의견']*3):
            flush, generate = self.make_flush()
            ws = SimpleNamespace(send_json=AsyncMock())
            self.assertEqual(await flush(ws,'session',[[w] for w in predictions],emit=True),[])
            generate.assert_not_called()
            ws.send_json.assert_not_called()

    async def test_disconnect_does_not_send_word(self):
        flush, generate = self.make_flush()
        ws = SimpleNamespace(send_json=AsyncMock())
        await flush(ws,'session',[['저']]*4,emit=False)
        generate.assert_called_once_with([['저']])
        ws.send_json.assert_not_called()

    async def test_repeated_gestures_each_emit_once(self):
        flush, generate = self.make_flush()
        ws = SimpleNamespace(send_json=AsyncMock())
        for _ in range(2):
            await flush(ws,'session',[['저']]*4,emit=True)
        self.assertEqual(generate.call_count,2)
        self.assertEqual(sum(c.args[0]['type']=='word' for c in ws.send_json.call_args_list),2)
