import httpx
from fastapi import APIRouter

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("/local")
async def get_local_models():
    """Checks local Ollama service to fetch downloaded models."""
    ollama_url = "http://127.0.0.1:11434/api/tags"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(ollama_url)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"running": True, "models": models}
    except Exception:
        pass
    return {"running": False, "models": []}
