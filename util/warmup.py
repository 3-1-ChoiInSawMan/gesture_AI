import requests

def warmup_ollama():
    payload = {
        "model": "qwen3:8b",
        "messages": [
            {"role": "user", "content": "안녕"}
        ],
        "think": False,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 1,
            "num_ctx": 4096,
            "temperature": 0
        }
    }

    requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=60
    )