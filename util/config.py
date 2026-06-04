WINDOW_SIZE = 30
INPUT_SIZE = 88
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 48

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
    "각 수어마다 시간 순서대로 top3 후보 단어들이 주어집니다. "
    "후보 중 문맥상 가장 알맞은 단어를 고르고, 가장 자연스러운 한국어 문장 한 문장만 반환하세요. "
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

LABEL2IDX = {'가다': 0, '감기': 1, '감사합니다_고맙습니다': 2, '괜찮다': 3, '기분_느낌': 4, '나쁘다_좋지않다': 5,
             '내일_익일': 6, '너_당신': 7, '네_알겠습니다': 8, '누가': 9, '느리게_천천히': 10, '만들다_생성하다': 11, 
             '만족_완료': 12, '많이_엄청_크게': 13, '말하다': 14, '망신당하다_쪽팔린다': 15, '면접': 16, '뭐': 17, 
             '보수_고치다_수정하다': 18, '빠르게_빨리': 19, '사무실': 20, '시간_일정': 21, '시작': 22, '아니요_아닙니다': 23, 
             '아빠_아버지': 24, '아쉽다_아깝다': 25, '안녕하세요_반갑습니다': 26, '어디': 27, '어떻게': 28, '언제': 29, 
             '엄마_어머니': 30, '오늘_금일': 31, '왜': 32, '의견_견해': 33, '이해_이해하다': 34, '있다_존재하다_가지고있다': 35, 
             '자리_좌석': 36, '잠시_잠시만': 37, '저_나': 38, '조금_약간': 39, '좋다_나쁘지않다': 40, '죄송합니다_미안합니다': 41, 
             '중요한': 42, '지금_바로': 43, '지루하다': 44, '질문': 45, '학교': 46, '회의_미팅': 47}
