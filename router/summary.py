import os
from fastapi import APIRouter, HTTPException

from util.ssl_config import configure_system_truststore

configure_system_truststore()

from openai import OpenAI

from schema.subtitleSchema import SummaryRequest, SummaryResponse
from util.loadMessage import load_message
from util.config import SYSTEM_PROMPT

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

@router.post('/summary', response_model = SummaryResponse)
def summary_meeting(payload: SummaryRequest) -> SummaryResponse:
    model_name = os.getenv("MODEL")

    if not client:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY가 설정되지 않았습니다.",
        )

    if not model_name:
        raise HTTPException(
            status_code=500,
            detail="MODEL이 설정되지 않았습니다.",
        )

    messages = load_message(session_id=payload.session_id, limit=payload.limit)

    if not messages:
        raise HTTPException(
            status_code=404,
            detail="컬렉션 비어있음",
        )

    conversation = "\n".join(f"- {message}" for message in messages)

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "다음 대화 기록을 보고 한 줄 요약만 반환하세요.\n"
                        f"{conversation}"
                    ),
                },
            ],
        )
    except Exception as E:
        raise HTTPException(
            status_code=502,
            detail=f"요약 생성 중 OpenAI 호출에 실패했습니다: {E}",
        ) from E

    summary_text = response.output_text.strip()
    if not summary_text:
        raise HTTPException(
            status_code=502,
            detail="OpenAI 응답에서 요약 텍스트를 추출하지 못했습니다.",
        )

    return SummaryResponse(
        summary=summary_text,
        source_count=len(messages),
        session_id=payload.session_id,
    )
