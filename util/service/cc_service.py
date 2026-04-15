import os
from datetime import UTC, datetime

from openai import OpenAI

from util.config import CC_SENTENCE_SYSTEM_PROMPT
from util.mongo_connect import col

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None


def build_sentence_prompt(words: list[str]) -> str:
    joined_words = ", ".join(words)
    return (
        "다음은 시간 순서대로 인식된 수어 단어들입니다.\n"
        "이 단어들을 참고해 가장 자연스러운 한국어 문장 한 문장만 반환하세요.\n"
        f"단어 목록: {joined_words}"
    )


def generate_sentence_from_words(words: list[str]) -> str:
    fallback_sentence = " ".join(words).strip()
    model_name = os.getenv("MODEL")

    if not fallback_sentence:
        return ""

    if not client or not model_name:
        return fallback_sentence

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": CC_SENTENCE_SYSTEM_PROMPT},
                {"role": "user", "content": build_sentence_prompt(words)},
            ],
        )
    except Exception:
        return fallback_sentence

    sentence = response.output_text.strip()
    return sentence or fallback_sentence


def store_final_sentence(session_id: str, sentence: str, words: list[str]) -> None:
    if not sentence:
        return

    try:
        col.insert_one(
            {
                "session_id": session_id,
                "text": sentence,
                "source": "cc",
                "is_final": True,
                "words": words,
                "created_at": datetime.now(UTC),
            }
        )
    except Exception:
        return
