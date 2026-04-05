from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from router import health, cc, cc_stt, summary
from pymongo import MongoClient

APP = FastAPI()
APP.include_router(health.router)
APP.include_router(cc.router)
APP.include_router(cc_stt.router)
APP.include_router(summary.router)