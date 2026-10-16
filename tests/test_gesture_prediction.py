import unittest
from collections import deque

from util.config import CC_SMOOTHING_WINDOW
from util.gesture_prediction import is_none_label, select_word_candidates


class GesturePredictionTests(unittest.TestCase):
    def setUp(self):
        self.labels = {0: "hello", 1: "meeting", 2: "none"}
        self.history = deque([0] * CC_SMOOTHING_WINDOW, maxlen=CC_SMOOTHING_WINDOW)

    def test_none_stops_output_despite_old_gesture_votes(self):
        words, debug = select_word_candidates(
            [(2, 0.95), (0, 0.04), (1, 0.01)], self.labels, self.history
        )
        self.assertIsNone(words)
        self.assertEqual(list(self.history), [])
        self.assertFalse(debug["accepted"])
        self.assertEqual(debug["reason"], "none_label")

    def test_low_confidence_does_not_reemit_old_gesture(self):
        words, debug = select_word_candidates([(1, 0.4)], self.labels, self.history)
        self.assertIsNone(words)
        self.assertFalse(self.history)
        self.assertEqual(debug["reason"], "low_confidence")

    def test_none_is_excluded_from_alternative_words(self):
        words, debug = select_word_candidates(
            [(0, 0.85), (2, 0.1), (1, 0.05)], self.labels, self.history
        )
        self.assertEqual(words, ["hello", "meeting"])
        self.assertNotIn("none", debug["candidates"])

    def test_gesture_resumes_without_votes_from_before_none(self):
        select_word_candidates([(2, 0.9)], self.labels, self.history)
        words, _ = select_word_candidates([(1, 0.9)], self.labels, self.history)
        self.assertEqual(words, ["meeting"])
        self.assertEqual(list(self.history), [1])

    def test_unknown_label_is_not_replaced_with_old_word(self):
        words, debug = select_word_candidates([(99, 0.99)], self.labels, self.history)
        self.assertIsNone(words)
        self.assertFalse(self.history)
        self.assertEqual(debug["reason"], "unknown_label")

    def test_empty_predictions_clear_old_votes(self):
        words, _ = select_word_candidates([], self.labels, self.history)
        self.assertIsNone(words)
        self.assertFalse(self.history)

    def test_existing_model_without_none_keeps_gesture_predictions(self):
        words, _ = select_word_candidates(
            [(0, 0.95), (1, 0.05)], {0: "hello", 1: "meeting"}, self.history
        )
        self.assertEqual(words, ["hello", "meeting"])

    def test_none_aliases_are_normalized(self):
        for label in ["none", " NONE ", "idle", "no_gesture"]:
            with self.subTest(label=label):
                self.assertTrue(is_none_label(label))
                words, _ = select_word_candidates([(2, 0.9)], {2: label}, self.history)
                self.assertIsNone(words)
        self.assertFalse(is_none_label("hello"))
        self.assertFalse(is_none_label(None))

    def test_nonfinite_confidence_is_rejected(self):
        words, _ = select_word_candidates([(0, float("nan"))], self.labels, self.history)
        self.assertIsNone(words)
        self.assertFalse(self.history)


if __name__ == "__main__":
    unittest.main()
