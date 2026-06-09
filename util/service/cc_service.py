import json
import os
import time
import urllib.error
import urllib.request

from util.loadLogger import logger

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_PATH = "/api/chat"
CC_SENTENCE_SYSTEM_PROMPT = (
    "당신은 수어 인식 단어를 문장으로 만드는 한국어 전문가입니다. "
    "주어진 단어들을 조사와 어미를 추가하여 자연스러운 한국어 한 문장으로만 변환합니다. "
    "문장만 반환하고 설명이나 주석은 절대 포함하지 마세요."
)


class OllamaRequestError(Exception):
    pass


def _fallback_from_candidates(word_candidates: list[list[str]]) -> str:
    return " ".join(candidates[0] for candidates in word_candidates if candidates).strip()


def build_sentence_prompt(word_candidates: list[list[str]]) -> str:
    """
    프롬프트 구성:
    - 주요 단어: 각 수어마다 top1 단어 (가장 유력한)
    - 백업 단어: top2/3 단어들 (필요시 참고)
    
    이렇게 하면 모델이 top1을 우선으로 사용하고, 
    필요한 경우에만 백업을 참고하여 문맥상 맞는 선택을 할 수 있음
    """
    # 주요 단어: 각 그룹의 첫 번째(top1)
    main_words = [
        candidates[0] 
        for candidates in word_candidates 
        if candidates
    ]
    
    # 백업 단어: 각 그룹의 2번째 이후(top2/3)
    backup_items = []
    for idx, candidates in enumerate(word_candidates):
        if len(candidates) > 1:
            main_word = candidates[0]
            alternatives = ", ".join(candidates[1:])
            backup_items.append(f"- {main_word} 대신: {alternatives}")
    
    backup_section = "\n".join(backup_items) if backup_items else ""
    
    prompt = (
        "다음 주요 단어들을 사용하여 자연스러운 한국어 문장 한 문장을 만들어주세요.\n"
        "주요 단어를 최우선으로 사용하되, 문맥상 필요하면 백업 단어를 참고해도 됩니다.\n\n"
        "주요 단어:\n"
        f"{' '.join(main_words)}\n"
    )
    
    if backup_section:
        prompt += f"\n백업 단어 (필요시만 사용):\n{backup_section}\n"
    
    prompt += "\n한 문장만 반환하세요."
    return prompt


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

def _ollama_keep_alive_value(value: str):
    value = value.strip()

    # JSON number로 보내야 하는 값
    if value in {"-1", "0"}:
        return int(value)

    # 60, 3600 같은 초 단위 숫자도 number로 보낼 수 있음
    if value.lstrip("-").isdigit():
        return int(value)

    # 30m, 5m, 24h, -1m 같은 duration 문자열
    return value

def _ollama_chat(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    source: str,
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL).rstrip("/")
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30"))
    keep_alive = _ollama_keep_alive_value(os.getenv("OLLAMA_KEEP_ALIVE", "30m"))

    payload = {
        "model": model_name,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": _ollama_option_float("OLLAMA_TEMPERATURE", 0.3),
            "num_predict": _ollama_option_int("OLLAMA_NUM_PREDICT", 512),
            "num_ctx": _ollama_option_int("OLLAMA_NUM_CTX", 4096),
        },
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ],
    }

    think = _ollama_think_value()
    if think is not None:
        payload["think"] = think
    else:
        payload["think"] = False

    request = urllib.request.Request(
        f"{base_url}{OLLAMA_CHAT_PATH}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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

    logger.info("Ollama %s full response: %s", source, json.dumps(data, ensure_ascii=False))
    response_text = data.get("message", {}).get("content", "").strip()
    logger.info("Ollama %s raw response: %r", source, response_text)
    return response_text


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
