import os
import uuid
import json
import asyncio
import threading
import subprocess
from typing import Dict, List, Any, AsyncGenerator
from fastapi import HTTPException
from ui.backend.app.config import (
    PROJECT_ROOT,
    RESULTS_DIR,
    VENV_PYTHON,
    PIPELINE_SCRIPT,
    APIKEY_FALLBACK_FILES
)
from ui.backend.app.models.eval import EvalRequest

class ProcessManager:
    def __init__(self):
        # In-memory storage for active subprocesses and their log stream buffers
        self.active_runs: Dict[str, subprocess.Popen] = {}
        self.run_logs: Dict[str, List[str]] = {}
        self.run_statuses: Dict[str, str] = {}  # "running", "completed", "failed", "cancelled"
        self.run_output_files: Dict[str, str] = {}

    def _monitor_subprocess(self, run_id: str, proc: subprocess.Popen):
        """Monitors a subprocess, reads its stdout to avoid buffer deadlocks, and updates status."""
        try:
            # Read stdout line by line until process exits
            for line in iter(proc.stdout.readline, ""):
                self.run_logs[run_id].append(line)
                
            return_code = proc.wait()
            if return_code == 0:
                self.run_statuses[run_id] = "completed"
            elif self.run_statuses[run_id] == "running":
                self.run_statuses[run_id] = "failed"
                
            exit_msg = f"--- Process exited with code {return_code} (Status: {self.run_statuses[run_id]}) ---\n"
            self.run_logs[run_id].append(exit_msg)
        except Exception as e:
            self.run_statuses[run_id] = "failed"
            self.run_logs[run_id].append(f"[System Error] Monitor thread crashed: {str(e)}\n")
        finally:
            if proc.stdout:
                proc.stdout.close()

    def start_evaluation(self, req: EvalRequest) -> str:
        """Launches lm-eval or the pipeline in a non-blocking background subprocess."""
        run_id = str(uuid.uuid4())
        
        if req.is_pipeline:
            # Parse model args
            args_dict = {}
            for part in req.model_args.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    args_dict[k.strip()] = v.strip()
                    
            cmd = [
                str(VENV_PYTHON), str(PIPELINE_SCRIPT),
                "--runs", str(req.pipeline_runs or 7),
                "--model", args_dict.get("model", "qwen3:4b"),
                "--model_type", req.model,
                "--base_url", args_dict.get("base_url", "http://127.0.0.1:11434/v1/completions"),
                "--tokenizer", args_dict.get("tokenizer", "gpt2"),
                "--tasks", ",".join(req.tasks),
                "--num_fewshot", str(req.num_fewshot if req.num_fewshot is not None else 5),
                "--gen_kwargs", "temperature=0.3,top_p=0.95,do_sample=True"
            ]
            if req.resume_pipeline:
                cmd.append("--resume")
            if req.limit is not None:
                cmd += ["--limit", str(req.limit)]
                
            output_path = os.path.join(RESULTS_DIR, f"pipeline_{args_dict.get('model', 'qwen3_4b').replace(':', '_')}")
            self.run_output_files[run_id] = output_path
        else:
            cmd = [
                str(VENV_PYTHON), "-m", "lm_eval",
                "--model", req.model,
                "--model_args", req.model_args,
                "--tasks", ",".join(req.tasks),
            ]
            
            if req.limit is not None:
                cmd += ["--limit", str(req.limit)]
            if req.num_fewshot is not None:
                cmd += ["--num_fewshot", str(req.num_fewshot)]
            if req.apply_chat_template:
                cmd.append("--apply_chat_template")
            if req.log_samples:
                cmd.append("--log_samples")
            cmd.append("--confirm_run_unsafe_code")

            output_path = os.path.join(RESULTS_DIR, f"run_{run_id}")
            cmd += ["--output_path", output_path]
            self.run_output_files[run_id] = output_path
        
        print(f"Launching command: {' '.join(cmd)}")
        
        try:
            sub_env = os.environ.copy()
            sub_env["HF_ALLOW_CODE_EVAL"] = "1"
            sub_env["PYTHONUNBUFFERED"] = "1"
            
            local_apikey = None
            for filename in APIKEY_FALLBACK_FILES:
                p = PROJECT_ROOT / filename
                if p.exists() and p.is_file():
                    try:
                        with open(p) as f:
                            local_apikey = f.read().strip()
                            break
                    except Exception:
                        pass

            if req.api_keys:
                if req.api_keys.get("openai"):
                    sub_env["OPENAI_API_KEY"] = req.api_keys["openai"]
                elif local_apikey:
                    sub_env["OPENAI_API_KEY"] = local_apikey
                
                if req.api_keys.get("anthropic"):
                    sub_env["ANTHROPIC_API_KEY"] = req.api_keys["anthropic"]
                if req.api_keys.get("hf"):
                    sub_env["HF_TOKEN"] = req.api_keys["hf"]
            elif local_apikey:
                sub_env["OPENAI_API_KEY"] = local_apikey

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                env=sub_env
            )
            self.run_logs[run_id] = []
            self.run_statuses[run_id] = "running"
            self.active_runs[run_id] = proc
            
            # Start background thread to consume stdout and monitor process status
            monitor_thread = threading.Thread(
                target=self._monitor_subprocess, 
                args=(run_id, proc),
                daemon=True
            )
            monitor_thread.start()
            
            return run_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start evaluation run: {str(e)}")

    def cancel_evaluation(self, run_id: str):
        """Safely terminates an active evaluation run."""
        proc = self.active_runs.get(run_id)
        if not proc:
            raise HTTPException(status_code=404, detail="Active process not found")
            
        try:
            proc.terminate()
            # Wait up to 3 seconds for graceful exit, then kill if needed
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                
            self.run_statuses[run_id] = "cancelled"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to cancel process: {str(e)}")

    def get_run_status(self, run_id: str) -> dict:
        """Retrieves current run status, updating state if completed."""
        if run_id not in self.run_statuses:
            raise HTTPException(status_code=404, detail="Run ID not found")
            
        proc = self.active_runs.get(run_id)
        if proc:
            poll = proc.poll()
            if poll is not None:
                if poll == 0:
                    self.run_statuses[run_id] = "completed"
                elif self.run_statuses[run_id] == "running":
                    self.run_statuses[run_id] = "failed"
                    
        return {
            "run_id": run_id,
            "status": self.run_statuses[run_id],
            "has_logs": len(self.run_logs.get(run_id, [])) > 0
        }

    async def get_log_generator(self, run_id: str) -> AsyncGenerator[str, None]:
        """Streams stdout/stderr lines of the running subprocess as Server-Sent Events (SSE)."""
        if run_id not in self.run_logs:
            raise HTTPException(status_code=404, detail="Run logs not found")
            
        sent_count = 0
        while True:
            current_logs = list(self.run_logs[run_id])
            if sent_count < len(current_logs):
                for i in range(sent_count, len(current_logs)):
                    line = current_logs[i]
                    yield f"data: {json.dumps({'log': line, 'status': self.run_statuses[run_id]})}\n\n"
                sent_count = len(current_logs)
                
            if self.run_statuses[run_id] in ["completed", "failed", "cancelled"] and sent_count >= len(self.run_logs[run_id]):
                break
                
            await asyncio.sleep(0.25)

# Singleton service instance
process_manager = ProcessManager()
