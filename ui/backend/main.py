"""
Legacy entrypoint wrapper for backwards-compatibility.
Forwards FastAPI execution to the modular app structure.
"""
from ui.backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui.backend.main:app", host="127.0.0.1", port=8000, reload=True)
