from typing import List, Optional, Dict
from pydantic import BaseModel

class EvalRequest(BaseModel):
    model: str
    model_args: str
    tasks: List[str]
    limit: Optional[float] = None
    num_fewshot: Optional[int] = None
    apply_chat_template: bool = False
    log_samples: bool = True
    api_keys: Optional[Dict[str, str]] = None
    is_pipeline: Optional[bool] = False
    pipeline_runs: Optional[int] = 7
    resume_pipeline: Optional[bool] = False
