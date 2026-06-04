import json
import os
import time
import urllib.error
import urllib.request

from util.loadLogger import logger

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_PATH = "/api/chat"
CC_SENTENCE_SYSTEM_PROMPT = (
    "당신은 수어 인식 후보 단어들을 자연스러운 한국어 문장으로 복원하는 비서입니다. "
    "각 번호마다 시간 순서대로 top3 후보 단어들이 주어집니다. "
    "후보 중 문맥상 가장 그럴듯한 단어를 고르고, 조사와 어미를 자연스럽게 보완해 "
    "한국어 문장 한 문장만 반환하세요. 설명, 따옴표, 번호, 불필요한 형식은 쓰지 마세요."
)


class OllamaRequestError(Exception):
    pass


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


def _ollama_option_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("%s must be an integer; using %s.", name, default)
        return default


def _ollama_option_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("%s must be a number; using %s.", name, default)
        return default


def _ollama_think_value() -> bool | str | None:
    value = os.getenv("OLLAMA_THINK")
    if value is None or not value.strip():
        return None

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    if normalized in {"high", "medium", "low"}:
        return normalized

    logger.warning("OLLAMA_THINK must be true, false, high, medium, or low; omitting it.")
    return None


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _duration_seconds(data: dict, key: str) -> float | None:
    duration = data.get(key)
    if not isinstance(duration, (int, float)):
        return None
    return duration / 1_000_000_000


def _log_ollama_metrics(data: dict, wall_seconds: float, *, source: str) -> None:
    logger.info(
        "Ollama %s latency wall=%.3fs total=%s load=%s prompt_eval=%s eval=%s",
        source,
        wall_seconds,
        _format_duration(_duration_seconds(data, "total_duration")),
        _format_duration(_duration_seconds(data, "load_duration")),
        _format_duration(_duration_seconds(data, "prompt_eval_duration")),
        _format_duration(_duration_seconds(data, "eval_duration")),
    )


def _format_duration(duration: float | None) -> str:
    if duration is None:
        return "n/a"
    return f"{duration:.3f}s"


def _ollama_chat(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    source: str,
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))
    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    payload = {
        "model": model_name,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": _ollama_option_float("OLLAMA_TEMPERATURE", 0.0),
            "num_predict": _ollama_option_int("OLLAMA_NUM_PREDICT", 64),
            "num_ctx": _ollama_option_int("OLLAMA_NUM_CTX", 4096),
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    think = _ollama_think_value()
    if think is not None:
        payload["think"] = think

    request = urllib.request.Request(
        f"{base_url}{OLLAMA_CHAT_PATH}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = _read_http_error_body(exc)
        detail = f": {body}" if body else ""
        raise OllamaRequestError(f"Ollama HTTP {exc.code} {exc.reason}{detail}") from exc

    _log_ollama_metrics(data, time.perf_counter() - started_at, source=source)

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
            source="sentence",
        )
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        OllamaRequestError,
    ) as exc:
        logger.warning("Ollama sentence generation failed: %s", exc)
        return fallback_sentence

    return sentence or fallback_sentence


def warmup_sentence_model() -> None:
    model_name = os.getenv("OLLAMA_MODEL")

    if not model_name:
        logger.info("OLLAMA_MODEL is not configured; skipping Ollama warmup.")
        return

    try:
        _ollama_chat(
            model_name=model_name,
            system_prompt="Reply with OK only.",
            user_prompt="warmup",
            source="warmup",
        )
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        OllamaRequestError,
    ) as exc:
        logger.warning("Ollama warmup failed: %s", exc)
        return

    logger.info("Ollama warmup complete")


def store_final_sentence(
    session_id: str,
    sentence: str,
    word_candidates: list[list[str]],
) -> None:
    if not sentence:
        return
