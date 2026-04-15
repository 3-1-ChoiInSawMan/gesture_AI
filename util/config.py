WINDOW_SIZE = 60
INPUT_SIZE = 88
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 29

STT_WINDOW_SIZE = 15

SYSTEM_PROMPT = (
    "당신은 회의/대화 기록을 한 줄로 압축하는 비서다. "
    "핵심 주제와 결론만 50자 내외의 한국어 한 문장으로 요약하라. "
    "불필요한 수식어와 추측은 넣지 마라."
)

CC_SENTENCE_SYSTEM_PROMPT = (
    "당신은 수어 인식 단어 목록을 이어 자연스러운 한국어 문장으로 복원하는 비서다. "
    "단어들의 시간 순서를 유지한 채 가장 자연스러운 문장 한 문장만 반환하라. "
    "설명, 따옴표, 번호, 불필요한 수식어는 넣지 마라."
)

TEXT_KEYS = (
    "message",
    "text",
    "content",
    "subtitle",
    "utterance",
    "transcript",
    "summary",
)
SPEAKER_KEYS = ("speaker", "role", "type", "source", "sender", "user")
LIST_KEYS = ("messages", "conversation", "contents", "items", "segments")
SILENCE_TIMEOUT_SECONDS = 2.0
