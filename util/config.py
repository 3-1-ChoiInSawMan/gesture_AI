WINDOW_SIZE = 30
INPUT_SIZE = 88
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 5

CC_CONF_THRESHOLD = 0.70
CC_TOP_K = 3
CC_PRED_EVERY_N_FRAMES = 1
CC_SMOOTHING_WINDOW = 7
CC_MIN_VALID_FRAMES = 24
CC_HANDS_DOWN_MARGIN = 0.05
CC_HANDS_DOWN_RATIO = 0.75
CC_HANDS_DOWN_MIN_FRAMES = 6
CC_NO_GESTURE_MIN_FRAMES = 6
CC_SILENCE_TIMEOUT_SECONDS = 1.0

STT_WINDOW_SIZE = 15
STT_LANGUAGE = "ko"
STT_INFERENCE_INTERVAL_SECONDS = 0.5

SYSTEM_PROMPT = (
    "당신은 회의/대화 기록을 한 줄로 압축하는 비서다. "
    "핵심 주제와 결론만 50자 내외의 한국어 한 문장으로 요약하라. "
    "불필요한 수식어와 추측은 넣지 마라."
)

CC_SENTENCE_SYSTEM_PROMPT = (
    "당신은 수어 인식 후보 단어들을 자연스러운 한국어 문장으로 복원하는 비서입니다. "
    # "각 수어마다 시간 순서대로 top3 후보 단어들이 주어집니다. "
    # "후보 중 문맥상 가장 알맞은 단어를 고르고, 가장 자연스러운 한국어 문장 한 문장만 반환하세요. "
    "각 수어마다 시간 순서대로 단어가 주어집니다. "
    "주어진 단어들을 조사와 어미를 추가하여 자연스러운 한국어 한 문장으로만 변환하세요. "
    "설명, 따옴표, 번호, 불필요한 형식은 넣지 마세요."
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

LABEL2IDX = {'감사합니다': 0, '시간': 1, '안녕하세요': 2, '의견': 3, '회의': 4}