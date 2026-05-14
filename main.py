from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from router import health, cc, summary, cc_stt, root
from pymongo import MongoClient

APP = FastAPI()
APP.include_router(root.router)
APP.include_router(health.router)
APP.include_router(cc.router)
APP.include_router(cc_stt.router)
# APP.include_router(summary.router) // 회의록 요약은 Spring에서