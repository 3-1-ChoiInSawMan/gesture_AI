import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from util.stt_filter import filter_hallucination


class STTFilterTests(unittest.TestCase):
    def test_extract_text_filters_segments_before_joining(self):
        source = Path(__file__).resolve().parents[1] / "router" / "cc_stt.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_extract_text"
        )
        namespace = {"filter_hallucination": filter_hallucination}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        extract = namespace["_extract_text"]
        self.assertEqual(extract([SimpleNamespace(text="Thanks for watching.")]), "")
        self.assertEqual(extract([
            SimpleNamespace(text="회의를 시작합니다."),
            SimpleNamespace(text="구독과 좋아요는 사랑입니다."),
            SimpleNamespace(text="의견을 말씀해주세요."),
        ]), "회의를 시작합니다. 의견을 말씀해주세요.")

    def test_filters_listed_korean_phrase_with_punctuation_and_spacing(self):
        self.assertEqual(filter_hallucination(" 오늘도 시청해 주셔서 감사합니다! "), "")

    def test_filters_foreign_phrase_case_insensitively(self):
        self.assertEqual(filter_hallucination("Thanks for watching."), "")

    def test_preserves_normal_speech_containing_a_listed_phrase(self):
        text = "오늘도 시청해주셔서 감사합니다라는 문구가 나왔어요."
        self.assertEqual(filter_hallucination(text), text)

    def test_preserves_short_fillers_and_normal_speech(self):
        for text in ("아", "음", "어", "감사합니다", "회의를 시작하겠습니다.", ""):
            with self.subTest(text=text):
                self.assertEqual(filter_hallucination(text), text)


if __name__ == "__main__":
    unittest.main()
