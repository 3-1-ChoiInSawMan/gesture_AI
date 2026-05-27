import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime

from util.loadLogger import logger

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_PATH = "/api/chat"
CC_SENTENCE_SYSTEM_PROMPT = (
    "당신은 수어 인식 후보 단어들을 자연스러운 한국어 문장으로 복원하는 비서입니다. "
    "각 번호마다 시간 순서대로 top 후보 단어들이 주어집니다. "
    "후보 중 문맥상 가장 그럴듯한 단어를 고르고, 조사와 어미를 자연스럽게 보완해 "
    "한국어 문장 한 문장만 반환하세요. 설명, 따옴표, 번호, 불필요한 형식은 쓰지 마세요."
)


def _fallback_from_candidates(word_candidates: list[list[str]]) -> str:
    return " ".join(candidates[0] for candidates in word_candidates if candidates).strip()


def build_sentence_prompt(word_candidates: list[list[str]]) -> str:
    candidate_lines = "\n".join(
        f"{index}. 후보: {', '.join(candidates)}"
        for index, candidates in enumerate(word_candidates, start=1)
        if candidates
    )
    return (
        "다음은 시간 순서대로 인식된 수어 후보 단어입니다.\n"
        "각 번호에서 후보 하나를 선택하고, 필요한 조사와 어미를 보완해서 "
        "가장 자연스러운 한국어 한 문장으로 만들어 주세요.\n"
        "단어를 그대로 나열하지 말고 문장처럼 다듬어 주세요.\n"
        f"{candidate_lines}"
    )


def _ollama_chat(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    payload = {
        "model": model_name,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0.2,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        f"{base_url}{OLLAMA_CHAT_PATH}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data.get("message", {}).get("content", "").strip()


def generate_sentence_from_words(word_candidates: list[list[str]]) -> str:
    fallback_sentence = _fallback_from_candidates(word_candidates)
    model_name = os.getenv("OLLAMA_MODEL")

    if not fallback_sentence:
        return ""

    if not model_name:
        logger.warning("OLLAMA_MODEL is not configured; using fallback sentence.")
        return fallback_sentence

    try:
        sentence = _ollama_chat(
            model_name=model_name,
            system_prompt=CC_SENTENCE_SYSTEM_PROMPT,
            user_prompt=build_sentence_prompt(word_candidates),
        )
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning("Ollama sentence generation failed: %s", exc)
        return fallback_sentence

    return sentence or fallback_sentence


def store_final_sentence(
    session_id: str,
    sentence: str,
    word_candidates: list[list[str]],
) -> None:
    if not sentence:
        return
