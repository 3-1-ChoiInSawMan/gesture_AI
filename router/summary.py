import os
from fastapi import FastAPI, APIRouter
from openai import OpenAI

router = APIRouter(prefix='/ai')
Client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

@router.websocket('/summary')
def summary_meeting():
    pass