#!/usr/bin/env python3
"""
===============================================================================
  LM Evaluation Harness — Multi-Run Benchmarking Pipeline
===============================================================================

  Model:       qwen3:4b (via Ollama local-completions)
  Tasks:       gsm8k, truthfulqa_gen, humaneval
  Iterations:  7 runs with different seeds
  Output:      Individual run results + aggregated statistics

  Estimated time: ~14-20 hours total (7 × ~2-3 hrs per full run)

  Usage:
    python benchmark_pipeline.py              # Run all 7 iterations
    python benchmark_pipeline.py --runs 3     # Override number of runs
    python benchmark_pipeline.py --aggregate  # Only aggregate existing results
    python benchmark_pipeline.py --resume     # Resume from last completed run
===============================================================================
"""

import os
import sys
import json
import time
import subprocess
import argparse
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "qwen3:4b"
MODEL_TYPE = "local-completions"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1/completions"
TOKENIZER = "Qwen/Qwen3-4B"

# Generation-compatible tasks only (Ollama doesn't support logprobs)
TASKS = ["gsm8k", "truthfulqa_gen", "humaneval"]

# Seeds for each run — different seeds produce different few-shot selections
# and sampling outcomes
SEEDS = [0, 42, 1234, 2024, 7777, 31415, 99999]

# Generation kwargs: small temperature for controlled variance across runs
# temperature=0.0 → deterministic (identical results every run)
# temperature=0.3 → slight sampling variation (recommended for multi-run avg)
GEN_KWARGS = "temperature=0.3,top_p=0.95,do_sample=True"

NUM_RUNS = 7
NUM_FEWSHOT = 5  # Standard few-shot count
LIMIT = None

# Paths
PROJECT_ROOT = Path(__file__).parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PIPELINE_RESULTS_DIR = PROJECT_ROOT / "pipeline_results" / f"{MODEL_NAME.replace(':', '_')}"

# ── Color output helpers ─────────────────────────────────────────────────────

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

def log(msg, color=Colors.CYAN):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{Colors.DIM}[{timestamp}]{Colors.END} {color}{msg}{Colors.END}")

def log_header(msg):
    width = 70
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'═' * width}")
    print(f"  {msg}")
    print(f"{'═' * width}{Colors.END}\n")

def log_success(msg):
    log(f"✅ {msg}", Colors.GREEN)

def log_error(msg):
    log(f"❌ {msg}", Colors.RED)

def log_warn(msg):
    log(f"⚠️  {msg}", Colors.YELLOW)

# ── Pre-flight checks ───────────────────────────────────────────────────────

def preflight_checks() -> bool:
    """Verify connectivity and model availability."""
    log_header("PRE-FLIGHT CHECKS")

    # Check venv python
    if not VENV_PYTHON.exists():
        log_error(f"Virtual environment python not found at {VENV_PYTHON}")
        return False
    log_success(f"Python: {VENV_PYTHON}")

    # Check connectivity based on endpoint
    is_local = "127.0.0.1" in OLLAMA_BASE_URL or "localhost" in OLLAMA_BASE_URL
    
    if is_local:
        # Check Ollama connectivity
        try:
            import urllib.request
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if MODEL_NAME not in models:
                    log_error(f"Model '{MODEL_NAME}' not found in Ollama. Available: {models}")
                    return False
                log_success(f"Ollama running, model '{MODEL_NAME}' available")
        except Exception as e:
            log_error(f"Cannot connect to Ollama at 127.0.0.1:11434 — {e}")
            return False

        # Warm up the model with a tiny request to load it into memory
        log("Warming up model (loading into VRAM)...", Colors.YELLOW)
        try:
            import urllib.request
            warm_data = json.dumps({"model": MODEL_NAME, "prompt": "Hi", "stream": False}).encode()
            warm_req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=warm_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(warm_req, timeout=60) as resp:
                resp.read()
            log_success("Model warmed up and loaded into memory")
        except Exception as e:
            log_warn(f"Warm-up request failed (non-fatal): {e}")
    else:
        # External API endpoint check (like college LiteLLM)
        log(f"External API endpoint detected: {OLLAMA_BASE_URL}. Checking connection...", Colors.YELLOW)
        try:
            import urllib.request
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(
                OLLAMA_BASE_URL,
                method="GET"
            )
            
            # Find local key to authenticate the probe
            token = os.environ.get("OPENAI_API_KEY")
            if not token:
                for filename in ["apikey", "college key"]:
                    p = PROJECT_ROOT / filename
                    if p.exists() and p.is_file():
                        try:
                            with open(p) as f:
                                token = f.read().strip()
                                break
                        except Exception:
                            pass
            if token:
                req.add_header("Authorization", f"Bearer {token}")
                
            try:
                with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                    resp.read()
                log_success(f"Reachable external API: {OLLAMA_BASE_URL}")
            except urllib.error.HTTPError as he:
                if he.code in [200, 401, 405, 400]:
                    log_success(f"Reachable external API (HTTP {he.code}): {OLLAMA_BASE_URL}")
                else:
                    log_warn(f"External API returned status {he.code}. Proceeding anyway...")
        except Exception as e:
            log_error(f"Cannot connect to external API at {OLLAMA_BASE_URL} — {e}")
            return False

    # Create results directory
    PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_success(f"Results directory: {PIPELINE_RESULTS_DIR}")

    return True

# ── Single run execution ────────────────────────────────────────────────────

def run_single_evaluation(run_index: int, seed: int) -> Optional[Path]:
    """Execute a single lm-eval run and return the output directory path."""

    run_id = f"run_{run_index+1:02d}_seed{seed}"
    output_dir = PIPELINE_RESULTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log_header(f"RUN {run_index+1}/{NUM_RUNS}  •  Seed={seed}  •  ID={run_id}")
    log(f"Tasks: {', '.join(TASKS)}")
    log(f"Output: {output_dir}")

    cmd = [
        str(VENV_PYTHON), "-m", "lm_eval",
        "--model", MODEL_TYPE,
        "--model_args", f"model={MODEL_NAME},base_url={OLLAMA_BASE_URL},tokenizer={TOKENIZER},tokenized_requests=False",
        "--tasks", ",".join(TASKS),
        "--num_fewshot", str(NUM_FEWSHOT),
        "--seed", str(seed),
        "--gen_kwargs", GEN_KWARGS,
        "--output_path", str(output_dir),
        "--log_samples",
        "--confirm_run_unsafe_code",
    ]

    if "chat" in MODEL_TYPE:
        cmd += ["--apply_chat_template"]

    if LIMIT is not None:
        cmd += ["--limit", str(LIMIT)]

    # Set up environment and enable telemetry tracking
    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["LMEVAL_TELEMETRY_PATH"] = str(output_dir / "api_telemetry.json")
    env["LMEVAL_ENABLE_STREAMING"] = "1"

    # Fallback to local 'apikey' or 'college key' file if not present in env
    if "OPENAI_API_KEY" not in env:
        for filename in ["apikey", "college key"]:
            p = PROJECT_ROOT / filename
            if p.exists() and p.is_file():
                try:
                    with open(p) as f:
                        env["OPENAI_API_KEY"] = f.read().strip()
                        break
                except Exception:
                    pass

    log(f"Command: {' '.join(cmd[:6])} ...")

    start_time = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(PROJECT_ROOT),
        )

        # Stream output in real-time
        last_progress_line = ""
        for line in iter(proc.stdout.readline, ""):
            stripped = line.strip()
            # Show progress bars and important lines
            if "Requesting API:" in stripped or "%" in stripped:
                # Overwrite progress bar in place
                print(f"\r  {Colors.DIM}{stripped[:100]}{Colors.END}", end="", flush=True)
                last_progress_line = stripped
            elif any(kw in stripped for kw in ["INFO", "WARNING", "ERROR", "Tasks", "---", "|"]):
                if last_progress_line:
                    print()  # Newline after progress bar
                    last_progress_line = ""
                print(f"  {Colors.DIM}{stripped}{Colors.END}")

        if last_progress_line:
            print()

        return_code = proc.wait()
        elapsed = time.time() - start_time

        if return_code == 0:
            log_success(f"Run {run_index+1} completed in {timedelta(seconds=int(elapsed))}")
            return output_dir
        else:
            log_error(f"Run {run_index+1} failed with exit code {return_code} after {timedelta(seconds=int(elapsed))}")
            return None

    except KeyboardInterrupt:
        log_warn("Interrupted by user! Terminating current run...")
        proc.terminate()
        proc.wait(timeout=5)
        return None
    except Exception as e:
        log_error(f"Run {run_index+1} crashed: {e}")
        return None

# ── Results parsing ──────────────────────────────────────────────────────────

def find_results_file(run_dir: Path) -> Optional[Path]:
    """Find the newest results JSON file in a run output directory."""
    candidate_files = []
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            if f.startswith("results_") and f.endswith(".json"):
                candidate_files.append(Path(root) / f)
    if not candidate_files:
        return None
    # Return the file with the latest modification time
    candidate_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidate_files[0]

def parse_run_results(results_file: Path) -> Dict[str, Dict[str, float]]:
    """Parse a results JSON file and extract metrics per task."""
    with open(results_file) as f:
        data = json.load(f)

    results = data.get("results", {})
    parsed = {}

    for task_name, metrics in results.items():
        parsed[task_name] = {}
        for metric_key, value in metrics.items():
            # Skip stderr entries and non-numeric values
            if metric_key.endswith(",none") or metric_key.endswith("_stderr"):
                continue
            if isinstance(value, (int, float)):
                # Clean up metric name
                clean_key = metric_key.replace(",none", "").replace(",flexible-extract", "_flex").replace(",strict-match", "_strict")
                parsed[task_name][clean_key] = value

    return parsed

# ── Aggregation ──────────────────────────────────────────────────────────────

def get_percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_val = sorted(values)
    idx = int(len(sorted_val) * pct)
    idx = min(max(0, idx), len(sorted_val) - 1)
    return sorted_val[idx]

def aggregate_results(run_dirs: List[Path]) -> Dict:
    """Aggregate metrics and telemetry across all runs and compute statistics."""

    log_header("AGGREGATING RESULTS ACROSS ALL RUNS")

    all_run_results = []
    all_telemetry_records = []
    for run_dir in run_dirs:
        results_file = find_results_file(run_dir)
        if results_file:
            parsed = parse_run_results(results_file)
            all_run_results.append({"dir": str(run_dir), "results": parsed})
            log_success(f"Parsed metrics: {run_dir.name}")
        else:
            log_warn(f"No results file found in {run_dir.name}")
            
        # Parse telemetry
        telemetry_file = run_dir / "api_telemetry.json"
        if telemetry_file.exists():
            try:
                with open(telemetry_file) as f:
                    records = json.load(f)
                    all_telemetry_records.extend(records)
                log_success(f"Parsed telemetry: {run_dir.name} ({len(records)} requests)")
            except Exception as e:
                log_warn(f"Failed to read telemetry file in {run_dir.name}: {e}")

    if not all_run_results:
        log_error("No results to aggregate!")
        return {}

    # Collect all metrics per task across runs
    task_metrics: Dict[str, Dict[str, List[float]]] = {}

    for run_data in all_run_results:
        for task_name, metrics in run_data["results"].items():
            if task_name not in task_metrics:
                task_metrics[task_name] = {}
            for metric_key, value in metrics.items():
                if metric_key not in task_metrics[task_name]:
                    task_metrics[task_name][metric_key] = []
                task_metrics[task_name][metric_key].append(value)

    # Compute telemetry statistics
    telemetry_stats = {}
    if all_telemetry_records:
        ttfts = [r["ttft"] for r in all_telemetry_records if r.get("ttft") is not None]
        trts = [r["trt"] for r in all_telemetry_records if r.get("trt") is not None]
        itls = [r["itl"] for r in all_telemetry_records if r.get("itl") is not None and r["itl"] > 0]
        throughputs = [r["throughput"] for r in all_telemetry_records if r.get("throughput") is not None and r["throughput"] > 0]
        
        telemetry_stats = {
            "num_requests": len(all_telemetry_records),
            "trt": {
                "mean": round(statistics.mean(trts), 4) if trts else 0.0,
                "p95": round(get_percentile(trts, 0.95), 4) if trts else 0.0,
                "p99": round(get_percentile(trts, 0.99), 4) if trts else 0.0,
                "min": round(min(trts), 4) if trts else 0.0,
                "max": round(max(trts), 4) if trts else 0.0,
            },
            "ttft": {
                "mean": round(statistics.mean(ttfts), 4) if ttfts else 0.0,
                "p95": round(get_percentile(ttfts, 0.95), 4) if ttfts else 0.0,
                "p99": round(get_percentile(ttfts, 0.99), 4) if ttfts else 0.0,
                "min": round(min(ttfts), 4) if ttfts else 0.0,
                "max": round(max(ttfts), 4) if ttfts else 0.0,
            },
            "itl": {
                "mean": round(statistics.mean(itls), 4) if itls else 0.0,
                "p95": round(get_percentile(itls, 0.95), 4) if itls else 0.0,
                "p99": round(get_percentile(itls, 0.99), 4) if itls else 0.0,
                "min": round(min(itls), 4) if itls else 0.0,
                "max": round(max(itls), 4) if itls else 0.0,
            },
            "throughput": {
                "mean": round(statistics.mean(throughputs), 4) if throughputs else 0.0,
                "p95": round(get_percentile(throughputs, 0.95), 4) if throughputs else 0.0,
                "min": round(min(throughputs), 4) if throughputs else 0.0,
                "max": round(max(throughputs), 4) if throughputs else 0.0,
            }
        }

    # Compute statistics
    aggregated = {
        "model": MODEL_NAME,
        "num_runs": len(all_run_results),
        "seeds": SEEDS[:len(all_run_results)],
        "gen_kwargs": GEN_KWARGS,
        "tasks": {},
        "telemetry": telemetry_stats,
        "timestamp": datetime.now().isoformat(),
    }

    for task_name, metrics in task_metrics.items():
        aggregated["tasks"][task_name] = {}
        for metric_key, values in metrics.items():
            n = len(values)
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if n > 1 else 0.0
            aggregated["tasks"][task_name][metric_key] = {
                "mean": round(mean, 4),
                "stdev": round(stdev, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "n": n,
                "values": [round(v, 4) for v in values],
            }

    return aggregated

def print_aggregated_report(aggregated: Dict):
    """Print a beautiful formatted report of aggregated results."""

    log_header(f"BENCHMARK REPORT — {aggregated['model']}  ({aggregated['num_runs']} runs)")

    print(f"  {Colors.BOLD}Model:{Colors.END}      {aggregated['model']}")
    print(f"  {Colors.BOLD}Runs:{Colors.END}       {aggregated['num_runs']}")
    print(f"  {Colors.BOLD}Seeds:{Colors.END}      {aggregated['seeds']}")
    print(f"  {Colors.BOLD}Gen Args:{Colors.END}   {aggregated['gen_kwargs']}")
    print()

    # Print per-task results table
    for task_name, metrics in aggregated["tasks"].items():
        print(f"  {Colors.BOLD}{Colors.CYAN}┌─── {task_name} ───{'─' * max(0, 50 - len(task_name))}{Colors.END}")

        for metric_key, stats in sorted(metrics.items()):
            mean_str = f"{stats['mean']:.4f}"
            stdev_str = f"±{stats['stdev']:.4f}"
            range_str = f"[{stats['min']:.4f} — {stats['max']:.4f}]"

            # Color-code the mean value
            if "acc" in metric_key or "exact_match" in metric_key:
                val_color = Colors.GREEN if stats["mean"] > 0.5 else Colors.YELLOW if stats["mean"] > 0.2 else Colors.RED
            else:
                val_color = Colors.BLUE

            print(f"  {Colors.DIM}│{Colors.END}  {metric_key:30s}  {val_color}{mean_str}{Colors.END}  {Colors.DIM}{stdev_str:10s}  {range_str}{Colors.END}")

        print(f"  {Colors.DIM}└{'─' * 68}{Colors.END}")
        print()

    # Print key summary metrics
    print(f"  {Colors.BOLD}{Colors.GREEN}╔══ KEY METRICS SUMMARY ════════════════════════════════════════╗{Colors.END}")

    key_metrics = []
    for task_name, metrics in aggregated["tasks"].items():
        for metric_key, stats in metrics.items():
            if any(km in metric_key for km in ["exact_match", "bleu_acc", "rouge1_acc", "pass@1"]):
                key_metrics.append((task_name, metric_key, stats))

    for task_name, metric_key, stats in key_metrics:
        bar_len = int(stats["mean"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pct = stats["mean"] * 100
        print(f"  {Colors.GREEN}║{Colors.END}  {task_name:16s} {metric_key:20s} {bar} {Colors.BOLD}{pct:.1f}%{Colors.END} ±{stats['stdev']*100:.1f}%")

    print(f"  {Colors.BOLD}{Colors.GREEN}╚═══════════════════════════════════════════════════════════════╝{Colors.END}")
    print()

def save_aggregated_report(aggregated: Dict):
    """Save aggregated results as JSON and Markdown."""

    # Save detailed pipeline JSON
    json_path = PIPELINE_RESULTS_DIR / "aggregated_results.json"
    with open(json_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    log_success(f"JSON report saved: {json_path}")

    # Build standard lm-eval compatible schema for easy dashboard auto-visualization
    compatible_results = {}
    for task_name, metrics in aggregated["tasks"].items():
        compatible_results[task_name] = {}
        for metric_key, stats in metrics.items():
            compatible_results[task_name][metric_key] = stats["mean"]
            compatible_results[task_name][metric_key + "_stderr"] = stats["stdev"]

    dashboard_json = {
        "date": time.time(),
        "config": {
            "model": f"{MODEL_NAME} (Pipeline Average - {aggregated['num_runs']} Runs)",
            "model_args": f"model={MODEL_NAME},runs={aggregated['num_runs']},tasks={','.join(TASKS)}",
        },
        "results": compatible_results,
        "is_pipeline": True,
        "pipeline_details": aggregated
    }

    # Save to global eval_results/ for dashboard detection
    global_results_dir = PROJECT_ROOT / "eval_results"
    global_results_dir.mkdir(parents=True, exist_ok=True)
    model_safe_name = MODEL_NAME.replace(":", "_")
    global_json_path = global_results_dir / f"pipeline_{model_safe_name}_aggregated.json"
    
    with open(global_json_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    log_success(f"Dashboard-compatible JSON saved: {global_json_path}")

    # Save Markdown report
    md_path = PIPELINE_RESULTS_DIR / "BENCHMARK_REPORT.md"
    with open(md_path, "w") as f:
        f.write(f"# Benchmark Report — {aggregated['model']}\n\n")
        f.write(f"**Date:** {aggregated['timestamp']}\n\n")
        f.write(f"**Runs:** {aggregated['num_runs']} | **Seeds:** {aggregated['seeds']} | **Gen Args:** `{aggregated['gen_kwargs']}`\n\n")
        f.write("---\n\n")

        for task_name, metrics in aggregated["tasks"].items():
            f.write(f"## {task_name}\n\n")
            f.write("| Metric | Mean | Std Dev | Min | Max | N |\n")
            f.write("|--------|------|---------|-----|-----|---|\n")
            for metric_key, stats in sorted(metrics.items()):
                f.write(f"| {metric_key} | **{stats['mean']:.4f}** | ±{stats['stdev']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['n']} |\n")
            f.write("\n")

        # Key metrics summary
        f.write("---\n\n## Key Metrics Summary\n\n")
        f.write("| Task | Metric | Mean Score | Std Dev |\n")
        f.write("|------|--------|-----------|----------|\n")
        for task_name, metrics in aggregated["tasks"].items():
            for metric_key, stats in sorted(metrics.items()):
                if any(km in metric_key for km in ["exact_match", "bleu_acc", "rouge1_acc", "pass@1"]):
                    f.write(f"| {task_name} | {metric_key} | **{stats['mean']*100:.1f}%** | ±{stats['stdev']*100:.1f}% |\n")

        # System & API Performance Telemetry summary
        telemetry = aggregated.get("telemetry")
        if telemetry:
            f.write("\n---\n\n## System & API Performance Telemetry\n\n")
            f.write(f"**Total API Requests Profiled:** {telemetry['num_requests']}\n\n")
            f.write("| Performance Metric | Mean | p95 | p99 | Min | Max |\n")
            f.write("|--------------------|------|-----|-----|-----|-----|\n")
            f.write(f"| **Time to First Token (TTFT)** | {telemetry['ttft']['mean']:.3f}s | {telemetry['ttft']['p95']:.3f}s | {telemetry['ttft']['p99']:.3f}s | {telemetry['ttft']['min']:.3f}s | {telemetry['ttft']['max']:.3f}s |\n")
            f.write(f"| **Inter-Token Latency (ITL)** | {telemetry['itl']['mean']:.3f}s | {telemetry['itl']['p95']:.3f}s | {telemetry['itl']['p99']:.3f}s | {telemetry['itl']['min']:.3f}s | {telemetry['itl']['max']:.3f}s |\n")
            f.write(f"| **Total Response Time (TRT)** | {telemetry['trt']['mean']:.3f}s | {telemetry['trt']['p95']:.3f}s | {telemetry['trt']['p99']:.3f}s | {telemetry['trt']['min']:.3f}s | {telemetry['trt']['max']:.3f}s |\n")
            f.write(f"| **Throughput (Tokens/sec)** | {telemetry['throughput']['mean']:.2f} t/s | {telemetry['throughput']['p95']:.2f} t/s | N/A | {telemetry['throughput']['min']:.2f} t/s | {telemetry['throughput']['max']:.2f} t/s |\n")
            f.write("\n")

    # Also save MD report to global eval_results/ for easy download
    global_md_path = global_results_dir / f"pipeline_{model_safe_name}_report.md"
    try:
        import shutil
        shutil.copy2(md_path, global_md_path)
        log_success(f"Global Markdown report copied to: {global_md_path}")
    except Exception as e:
        log_warn(f"Failed to copy Markdown report to global folder: {e}")

    log_success(f"Markdown report saved: {md_path}")

# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    global MODEL_NAME, MODEL_TYPE, OLLAMA_BASE_URL, TOKENIZER, TASKS, GEN_KWARGS, NUM_RUNS, NUM_FEWSHOT, PIPELINE_RESULTS_DIR, LIMIT

    parser = argparse.ArgumentParser(description="Multi-run LM benchmarking pipeline")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of evaluation runs")
    parser.add_argument("--aggregate", action="store_true", help="Only aggregate existing results (skip running)")
    parser.add_argument("--resume", action="store_true", help="Resume from last completed run")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name / ID in Ollama")
    parser.add_argument("--model_type", type=str, default=MODEL_TYPE, help="Harness model type wrapper")
    parser.add_argument("--base_url", type=str, default=OLLAMA_BASE_URL, help="Ollama completions endpoint URL")
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER, help="Tokenizer override path/ID")
    parser.add_argument("--tasks", type=str, default=",".join(TASKS), help="Comma-separated evaluation tasks")
    parser.add_argument("--num_fewshot", type=int, default=NUM_FEWSHOT, help="Few-shot prompts count")
    parser.add_argument("--gen_kwargs", type=str, default=GEN_KWARGS, help="Generation kwargs override")
    parser.add_argument("--limit", type=float, default=None, help="Limit number of samples per task")
    args = parser.parse_args()

    NUM_RUNS = args.runs
    MODEL_NAME = args.model
    MODEL_TYPE = args.model_type
    OLLAMA_BASE_URL = args.base_url
    TOKENIZER = args.tokenizer
    TASKS = [t.strip() for t in args.tasks.split(",") if t.strip()]
    NUM_FEWSHOT = args.num_fewshot
    GEN_KWARGS = args.gen_kwargs
    LIMIT = args.limit
    
    # Re-calculate result directory in case model name changed
    PIPELINE_RESULTS_DIR = PROJECT_ROOT / "pipeline_results" / f"{MODEL_NAME.replace(':', '_')}"

    log_header("LM EVALUATION HARNESS — MULTI-RUN BENCHMARK PIPELINE")
    print(f"  {Colors.BOLD}Model:{Colors.END}       {MODEL_NAME}")
    print(f"  {Colors.BOLD}Tasks:{Colors.END}       {', '.join(TASKS)}")
    print(f"  {Colors.BOLD}Runs:{Colors.END}        {NUM_RUNS}")
    print(f"  {Colors.BOLD}Seeds:{Colors.END}       {SEEDS[:NUM_RUNS]}")
    print(f"  {Colors.BOLD}Gen Kwargs:{Colors.END}  {GEN_KWARGS}")
    print(f"  {Colors.BOLD}Few-shot:{Colors.END}    {NUM_FEWSHOT}")
    print()

    # ── Aggregate-only mode ──
    if args.aggregate:
        run_dirs = sorted([d for d in PIPELINE_RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if not run_dirs:
            log_error(f"No run directories found in {PIPELINE_RESULTS_DIR}")
            sys.exit(1)

        aggregated = aggregate_results(run_dirs)
        if aggregated:
            print_aggregated_report(aggregated)
            save_aggregated_report(aggregated)
        return

    # ── Pre-flight ──
    if not preflight_checks():
        log_error("Pre-flight checks failed. Aborting.")
        sys.exit(1)

    # ── Determine starting point ──
    start_index = 0
    completed_dirs = []
    if args.resume:
        existing = sorted([d for d in PIPELINE_RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")])
        for d in existing:
            if find_results_file(d):
                completed_dirs.append(d)
        start_index = len(completed_dirs)
        if start_index > 0:
            log_success(f"Resuming from run {start_index + 1} ({start_index} completed runs found)")
        else:
            log("No previous completed runs found, starting fresh")

    # ── Execute runs ──
    pipeline_start = time.time()
    successful_dirs = list(completed_dirs)  # Include previously completed runs

    for i in range(start_index, NUM_RUNS):
        seed = SEEDS[i] if i < len(SEEDS) else SEEDS[0] + i * 111

        remaining = NUM_RUNS - i
        if successful_dirs:
            avg_time = (time.time() - pipeline_start) / max(1, i - start_index) if i > start_index else 0
            if avg_time > 0:
                eta = timedelta(seconds=int(avg_time * remaining))
                log(f"Estimated time remaining: {eta}", Colors.YELLOW)

        result_dir = run_single_evaluation(i, seed)
        if result_dir:
            successful_dirs.append(result_dir)
        else:
            log_warn(f"Run {i+1} did not produce results. Continuing with remaining runs...")

    # ── Aggregate ──
    total_time = time.time() - pipeline_start
    log_header(f"PIPELINE COMPLETE — {len(successful_dirs)}/{NUM_RUNS} runs succeeded in {timedelta(seconds=int(total_time))}")

    if successful_dirs:
        aggregated = aggregate_results(successful_dirs)
        if aggregated:
            print_aggregated_report(aggregated)
            save_aggregated_report(aggregated)
            log_success(f"All reports saved to: {PIPELINE_RESULTS_DIR}")
    else:
        log_error("No successful runs to aggregate!")
        sys.exit(1)

if __name__ == "__main__":
    main()
