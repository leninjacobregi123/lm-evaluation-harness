from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ui.backend.app.models.eval import EvalRequest
from ui.backend.app.services.process import process_manager

router = APIRouter(prefix="/api/eval", tags=["evaluation"])

@router.post("/start")
def start_eval(req: EvalRequest):
    """Launches lm-eval in a non-blocking background subprocess."""
    run_id = process_manager.start_evaluation(req)
    return {"run_id": run_id, "status": "running"}

@router.get("/status/{run_id}")
def get_run_status(run_id: str):
    """Retrieves current execution status of a run."""
    return process_manager.get_run_status(run_id)

@router.post("/cancel/{run_id}")
def cancel_eval(run_id: str):
    """Safely terminates an active evaluation run."""
    process_manager.cancel_evaluation(run_id)
    return {"run_id": run_id, "status": "cancelled"}

@router.get("/stream/{run_id}")
def stream_run_logs(run_id: str):
    """Streams stdout/stderr lines of the running subprocess as Server-Sent Events (SSE)."""
    return StreamingResponse(
        process_manager.get_log_generator(run_id),
        media_type="text/event-stream"
    )
