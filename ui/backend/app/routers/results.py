from fastapi import APIRouter
from ui.backend.app.services.results import ResultsService

router = APIRouter(prefix="/api/results", tags=["results"])

@router.get("")
def list_results():
    """Traverses results directory and returns structured summaries of past runs."""
    return ResultsService.list_results()

@router.get("/file/{path:path}")
def get_result_file(path: str):
    """Fetches full contents of a results JSON/JSONL file."""
    return ResultsService.get_result_file(path)
