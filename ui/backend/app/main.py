from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ui.backend.app.config import ALLOWED_ORIGINS
from ui.backend.app.routers import eval, models, results, tasks

app = FastAPI(
    title="LM Evaluation Harness API",
    description="Industry-standard benchmark launcher API wrapper.",
    version="1.0.0"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(eval.router)
app.include_router(models.router)
app.include_router(results.router)
app.include_router(tasks.router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui.backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
