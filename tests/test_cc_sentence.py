import os
import unittest
import urllib.error
from unittest.mock import patch

from util.service import cc_service


class SentenceNoneFilteringTests(unittest.TestCase):
    def test_none_only_never_calls_llm(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model"}), patch.object(
            cc_service, "_ollama_chat"
        ) as chat:
            for candidates in [[], [["NONE"]], [[" none "], ["IDLE"], ["no_gesture"]]]:
                with self.subTest(candidates=candidates):
                    self.assertEqual(cc_service.generate_sentence_from_words(candidates), "")
            chat.assert_not_called()

    def test_mixed_words_reach_llm_without_none(self):
        words = [["NONE"], ["hello", "none", "hi"], ["meeting"], ["idle"]]
        with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model"}), patch.object(
            cc_service, "_ollama_chat", return_value="hello meeting"
        ) as chat:
            self.assertEqual(cc_service.generate_sentence_from_words(words), "hello meeting")
            prompt = chat.call_args.kwargs["user_prompt"]
            self.assertIn("hello meeting", prompt)
            self.assertIn("hi", prompt)
            self.assertNotIn("none", prompt.casefold())
            self.assertNotIn("idle", prompt.casefold())
        self.assertEqual(words[0], ["NONE"])
        self.assertEqual(words[1], ["hello", "none", "hi"])

    def test_none_top1_does_not_promote_backup_gesture(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model"}), patch.object(
            cc_service, "_ollama_chat"
        ) as chat:
            self.assertEqual(cc_service.generate_sentence_from_words([["NONE", "hello"]]), "")
            chat.assert_not_called()

    def test_missing_model_fallback_excludes_none(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": ""}), patch.object(
            cc_service, "_ollama_chat"
        ) as chat:
            self.assertEqual(
                cc_service.generate_sentence_from_words([["NONE"], [" hello "], ["meeting", "none"]]),
                "hello meeting",
            )
            chat.assert_not_called()

    def test_llm_failure_fallback_excludes_none(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model"}), patch.object(
            cc_service, "_ollama_chat", side_effect=urllib.error.URLError("offline")
        ):
            self.assertEqual(
                cc_service.generate_sentence_from_words([["none"], ["hello"]]), "hello"
            )

    def test_direct_prompt_and_fallback_filter_none(self):
        words = [["NONE", "unwanted"], ["hello", "idle", "hi"], ["no_gesture"]]
        prompt = cc_service.build_sentence_prompt(words)
        self.assertIn("hello", prompt)
        self.assertIn("hi", prompt)
        for ignored in ["none", "idle", "no_gesture", "unwanted"]:
            self.assertNotIn(ignored, prompt.casefold())
        self.assertEqual(cc_service._fallback_from_candidates(words), "hello")

    def test_empty_words_do_not_trigger_llm(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "test-model"}), patch.object(
            cc_service, "_ollama_chat"
        ) as chat:
            self.assertEqual(cc_service.generate_sentence_from_words([[], [" ", "NONE"]]), "")
            chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
