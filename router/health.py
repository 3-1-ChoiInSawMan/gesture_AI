from fastapi import APIRouter

router = APIRouter()

@router.get('/health')
def healthCheck():
    return {"status": "ok"}