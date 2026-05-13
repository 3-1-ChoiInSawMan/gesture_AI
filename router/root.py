from fastapi import FastAPI, APIRouter

router = APIRouter()

@router.get('/')
def root():
    return {'status':'ok'}