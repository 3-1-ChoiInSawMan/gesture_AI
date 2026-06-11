import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from router import health, cc, summary, cc_stt, root
from util.loadLogger import logger
from util.service.cc_service import warmup_sentence_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI startup warmup starting")
    await asyncio.to_thread(warmup_sentence_model)
    yield


APP = FastAPI(lifespan=lifespan)
APP.include_router(root.router)
APP.include_router(cc.router)
APP.include_router(cc_stt.router)
# APP.include_router(summary.router) // 회의록 요약은 Spring에서
