import os
from fastapi import FastAPI, APIRouter
from openai import OpenAI
from util.mongo_connect import client, db, col

router = APIRouter()
Client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

@router.post('/summary')
def summary_meeting():
    with col.find() as c:
        for doc in c:
            print(doc)