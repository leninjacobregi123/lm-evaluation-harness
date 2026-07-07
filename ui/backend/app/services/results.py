import os
import json
from fastapi import HTTPException
from ui.backend.app.config import RESULTS_DIR

class ResultsService:
    @staticmethod
    def list_results() -> list:
        """Traverses results directory and returns structured summaries of past runs."""
        results = []
        if not os.path.exists(RESULTS_DIR):
            return []
            
        for root, dirs, files in os.walk(RESULTS_DIR):
            dirs.sort()
            files.sort()
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, RESULTS_DIR)
                
                if file.endswith(".json"):
                    try:
                        with open(full_path) as f:
                            data = json.load(f)
                        
                        is_samples = "samples_" in file
                        results.append({
                            "filename": file,
                            "relative_path": relative_path,
                            "date": data.get("date", os.path.getmtime(full_path)),
                            "model": data.get("config", {}).get("model", "Unknown"),
                            "tasks": list(data.get("results", {}).keys()) if not is_samples else [],
                            "summary": data.get("results", {}) if not is_samples else {},
                            "is_samples": is_samples
                        })
                    except Exception:
                        continue
                        
                elif file.endswith(".jsonl"):
                    is_samples = "samples_" in file
                    if not is_samples:
                        continue
                    
                    # To get the model name for a samples file, we look for a .json file in the same directory
                    model = "Unknown"
                    try:
                        for sibling in os.listdir(root):
                            if sibling.endswith(".json") and not sibling.startswith("samples_"):
                                sibling_path = os.path.join(root, sibling)
                                with open(sibling_path) as sf:
                                    sibling_data = json.load(sf)
                                    model = sibling_data.get("config", {}).get("model", "Unknown")
                                    break
                    except Exception:
                        pass
                    
                    results.append({
                        "filename": file,
                        "relative_path": relative_path,
                        "date": os.path.getmtime(full_path),
                        "model": model,
                        "tasks": [],
                        "summary": {},
                        "is_samples": True
                    })
                            
        # Sort by date descending
        results.sort(key=lambda x: x.get("date", 0), reverse=True)
        return results

    @staticmethod
    def get_result_file(path: str) -> dict:
        """Fetches full contents of a results JSON/JSONL file."""
        full_path = os.path.abspath(os.path.join(RESULTS_DIR, path))
        # Prevent directory traversal attacks
        if not full_path.startswith(str(RESULTS_DIR)):
            raise HTTPException(status_code=403, detail="Access denied")
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        try:
            if full_path.endswith(".jsonl"):
                # Parse task from filename: samples_<task>_<timestamp>.jsonl
                name = os.path.basename(full_path)
                parts = name.split('_')
                task = "_".join(parts[1:-1]) if len(parts) > 2 else "unknown"
                
                samples_list = []
                with open(full_path) as f:
                    for line in f:
                        if line.strip():
                            samples_list.append(json.loads(line))
                return {task: samples_list}
                
            with open(full_path) as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")
