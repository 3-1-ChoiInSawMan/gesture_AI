import os
from datetime import UTC, datetime

from util.ssl_config import configure_system_truststore

configure_system_truststore()

from openai import OpenAI

from util.config import CC_SENTENCE_SYSTEM_PROMPT
from util.loadLogger import logger
from util.mongo_connect import col

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None


def _fallback_from_candidates(word_candidates: list[list[str]]) -> str:
    return " ".join(candidates[0] for candidates in word_candidates if candidates).strip()


def build_sentence_prompt(word_candidates: list[list[str]]) -> str:
    candidate_lines = "\n".join(
        f"{index}. 후보: {', '.join(candidates)}"
        for index, candidates in enumerate(word_candidates, start=1)
        if candidates
    )
    return (
        "다음은 시간 순서대로 인식된 수어별 top 후보 단어입니다.\n"
        "각 번호는 하나의 수어를 의미하며, 후보는 가능성이 높은 순서입니다.\n"
        "문맥상 가장 자연스러운 후보를 골라 한국어 문장 한 문장으로 복원하세요.\n"
        f"{candidate_lines}"
    )


def generate_sentence_from_words(word_candidates: list[list[str]]) -> str:
    fallback_sentence = _fallback_from_candidates(word_candidates)
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
                {"role": "user", "content": build_sentence_prompt(word_candidates)},
            ],
        )
    except Exception as exc:
        logger.warning("OpenAI sentence generation failed: %s", exc)
        return fallback_sentence

    sentence = response.output_text.strip()
    return sentence or fallback_sentence


def store_final_sentence(
    session_id: str,
    sentence: str,
    word_candidates: list[list[str]],
) -> None:
    if not sentence:
        return

    try:
        col.insert_one(
            {
                "session_id": session_id,
                "text": sentence,
                "source": "cc",
                "is_final": True,
                "words": word_candidates,
                "top_words": [candidates[0] for candidates in word_candidates if candidates],
                "created_at": datetime.now(UTC),
            }
        )
    except Exception as exc:
        logger.warning("Failed to store CC sentence: %s", exc)
        return
