from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

@router.get("")
def get_tasks():
    """Returns available tasks, groups, and tags registered in lm-eval."""
    try:
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        return {
            "tasks": tm.all_subtasks,
            "groups": tm.all_groups,
            "tags": tm.all_tags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load task manager: {str(e)}")
