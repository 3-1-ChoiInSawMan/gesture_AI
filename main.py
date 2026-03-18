from fastapi import FastAPI
from router import health, cc

APP = FastAPI()
APP.include_router(health.router)
APP.include_router(cc.router)