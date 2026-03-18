from fastapi import APIRouter

router = APIRouter(prefix= '/ai')

@router.get('/health')
def healthCheck():
    return {"status": "ok"}