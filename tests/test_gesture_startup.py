"""Regression: startup padding must not cause zero-shoulder inference failures."""
import ast
from collections import deque
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
from util.gesture_preprocessing import prepare_sequence


class StartupFrameTests(unittest.TestCase):
    def test_first_prediction_after_invalid_startup(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'router/cc.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {'_is_valid_frame', '_consume_frame'}]
        calls = []
        def predict(sequence):
            transformed = prepare_sequence(sequence, sequence_normalization=False)
            self.assertTrue(np.isfinite(transformed).all())
            calls.append(transformed)
            return [(0, 1.0)]
        ns = dict(np=np, deque=deque, INPUT_SIZE=88, WINDOW_SIZE=30,
                  CC_MIN_VALID_FRAMES=24, CC_PRED_EVERY_N_FRAMES=1,
                  _predict_sequence=predict, idx2label={0:'None'},
                  select_word_candidates=Mock(return_value=(None, {'reason':'none_label'})))
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<actual router>', 'exec'), ns)
        valid = np.random.default_rng(42).random(88, dtype=np.float32)
        valid[84:] = [.4,.6,.6,.6]
        self.assertTrue(ns['_is_valid_frame'](valid))
        for invalid_count in (0, 6, 25, 51):
            with self.subTest(invalid_count=invalid_count):
                calls.clear()
                seq, flags, history = deque(maxlen=30), deque(maxlen=30), deque(maxlen=7)
                last = np.zeros(88, dtype=np.float32)
                frames = [np.zeros(88,dtype=np.float32)]*invalid_count + [valid]*30
                for count, frame in enumerate(frames, 1):
                    _, last, debug = ns['_consume_frame'](frame, seq_buffer=seq,
                        valid_flag_buffer=flags, pred_history=history,
                        last_valid_framevec=last, frame_count=count)
                    observed = count-invalid_count
                    if observed < 24:
                        self.assertEqual(len(calls),0)
                    if invalid_count >= 6 and observed == 24:
                        self.assertEqual(len(calls),1)
                        self.assertEqual(sum(flags),24)
                self.assertGreater(len(calls),0)
                # Later tracking loss reuses the last actual frame safely.
                ns['_consume_frame'](np.zeros(88,dtype=np.float32), seq_buffer=seq,
                    valid_flag_buffer=flags, pred_history=history,
                    last_valid_framevec=last, frame_count=count+1)
                prepare_sequence(list(seq), sequence_normalization=False)


if __name__ == '__main__':
    unittest.main()
