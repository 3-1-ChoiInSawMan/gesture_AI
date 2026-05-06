WINDOW_SIZE = 60
INPUT_SIZE = 88
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 32

CC_PRED_EVERY_N_FRAMES = 2
CC_CONF_THRESHOLD = 0.70
CC_SMOOTHING_WINDOW = 7
CC_MIN_VALID_FRAMES = 45
CC_HANDS_DOWN_MARGIN = 0.05
CC_HANDS_DOWN_RATIO = 0.75
CC_HANDS_DOWN_MIN_FRAMES = 6

STT_WINDOW_SIZE = 15
STT_LANGUAGE = "ko"
STT_INFERENCE_INTERVAL_SECONDS = 0.5

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

LABEL2IDX = {
    '감사합니다': 0, '괜찮다': 1, '내일': 2, '너': 3, '네': 4, '누가': 5,
    '만들다': 6, '만족': 7, '많이': 8, '말하다': 9, '뭐': 10, '빠르게': 11,
    '시작': 12, '아니요': 13, '안녕하세요': 14, '어디': 15, '어떻게': 16,
    '언제': 17, '오늘': 18, '왜': 19, '의견': 20, '이해': 21, '자리': 22,
    '잠시': 23, '저': 24, '조금': 25, '좋다': 26, '죄송합니다': 27,
    '지금': 28, '질문': 29, '학교': 30, '회의': 31
}
