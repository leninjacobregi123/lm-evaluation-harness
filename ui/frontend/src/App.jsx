import React, { useState, useEffect, useRef } from "react";
import {
  Play, Settings, BarChart2, History, Terminal, StopCircle,
  CheckCircle, XCircle, Key, Database, Search, RefreshCw,
  FolderOpen, Info, Clock, Code, Download, FileText,
  ChevronRight, Zap, AlertTriangle
} from "lucide-react";

import SectionHeader from "./components/SectionHeader";
import FormGroup from "./components/FormGroup";
import TaskTypeBadge from "./components/TaskTypeBadge";
import TaskPicker from "./components/TaskPicker";
import ModelConfigPanel from "./components/ModelConfigPanel";
import CLIPreviewCard from "./components/CLIPreviewCard";
import LeaderboardChart from "./components/LeaderboardChart";

/* ────────────────────────────────────────────────────────── */
/*  MAIN APP                                                    */
/* ────────────────────────────────────────────────────────── */

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState("new_eval");
  const [pipelineRuns, setPipelineRuns] = useState("7");
  const [resumePipeline, setResumePipeline] = useState(false);

  // API Config (LocalStorage)
  const [apiKeys, setApiKeys] = useState(() => {
    const saved = localStorage.getItem("lm_eval_api_keys");
    return saved ? JSON.parse(saved) : { openai: "", anthropic: "", hf: "" };
  });

  // Task & Model Registries
  const [tasksRegistry, setTasksRegistry] = useState({ tasks: [], groups: [], tags: [] });
  const [localModels, setLocalModels] = useState([]);
  const [ollamaRunning, setOllamaRunning] = useState(false);
  const [loadingRegistries, setLoadingRegistries] = useState(true);

  // Evaluation Form States
  const [modelType, setModelType] = useState("local-chat-completions");
  const [customModelName, setCustomModelName] = useState("");
  const [selectedLocalModel, setSelectedLocalModel] = useState("");
  const [modelArgs, setModelArgs] = useState("");
  const [selectedTasks, setSelectedTasks] = useState([]);
  const [evalMode, setEvalMode] = useState("quick");
  const [limit, setLimit] = useState("5");
  const [numFewshot, setNumFewshot] = useState("");
  const [applyChatTemplate, setApplyChatTemplate] = useState(true);
  const [logSamples, setLogSamples] = useState(true);
  const [taskSearch, setTaskSearch] = useState("");
  const [taskCategory, setTaskCategory] = useState("tasks");

  // Running State / Logs
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState("idle");
  const [logLines, setLogLines] = useState([]);
  const terminalEndRef = useRef(null);

  // Past Results
  const [resultsList, setResultsList] = useState([]);
  const [loadingResults, setLoadingResults] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const [resultDetails, setResultDetails] = useState(null);
  const [samplesList, setSamplesList] = useState([]);
  const [samplesFilter, setSamplesFilter] = useState("all");

  // ── Effects ──
  useEffect(() => {
    fetchRegistries();
    fetchResults();
  }, []);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logLines]);

  useEffect(() => {
    if (activeTab === "pipeline" && selectedTasks.length === 0) {
      setSelectedTasks(["gsm8k", "truthfulqa_gen", "humaneval"]);
    }
  }, [activeTab]);

  // ── Business Logic ──
  const saveApiKeys = (keys) => {
    setApiKeys(keys);
    localStorage.setItem("lm_eval_api_keys", JSON.stringify(keys));
  };

  const fetchRegistries = async () => {
    setLoadingRegistries(true);
    try {
      const tRes = await fetch("http://127.0.0.1:8000/api/tasks");
      if (tRes.ok) {
        const tData = await tRes.json();
        setTasksRegistry(tData);
      }
      const mRes = await fetch("http://127.0.0.1:8000/api/models/local");
      if (mRes.ok) {
        const mData = await mRes.json();
        setOllamaRunning(mData.running);
        setLocalModels(mData.models);
        if (mData.models.length > 0) {
          setSelectedLocalModel(mData.models[0]);
        }
      }
    } catch (e) {
      console.error("Failed to load registries from backend. Is server running on port 8000?", e);
    } finally {
      setLoadingRegistries(false);
    }
  };

  const fetchResults = async () => {
    setLoadingResults(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/results");
      if (res.ok) {
        const data = await res.json();
        setResultsList(data);
      }
    } catch (e) {
      console.error("Failed to load past runs results:", e);
    } finally {
      setLoadingResults(false);
    }
  };

  const getTokenizerForModel = (modelName) => {
    const name = (modelName || "").toLowerCase();
    if (name.includes("qwen")) return "Qwen/Qwen2.5-7B-Instruct";
    if (name.includes("llama")) return "unsloth/llama-3-8b-Instruct";
    return "gpt2";
  };

  const getCompiledModelArgs = () => {
    if (modelArgs && modelArgs.trim()) return modelArgs.trim();
    const parts = [];
    if (modelType === "local-chat-completions" || modelType === "local-completions") {
      const mName = selectedLocalModel || customModelName || "qwen2.5:0.5b";
      parts.push(`model=${mName}`);
      const endpoint = modelType === "local-chat-completions" ? "/v1/chat/completions" : "/v1/completions";
      parts.push(`base_url=http://127.0.0.1:11434${endpoint}`);
      parts.push(`tokenizer=${getTokenizerForModel(mName)}`);
      parts.push("tokenized_requests=False");
    } else if (modelType === "openai" || modelType === "openai-chat-completions") {
      parts.push(`model=${customModelName || "gpt-4o"}`);
    } else if (modelType === "hf") {
      parts.push(`pretrained=${customModelName || "Qwen/Qwen2.5-0.5B"}`);
      parts.push("dtype=float16");
    } else {
      parts.push(`model=${customModelName}`);
    }
    return parts.join(",");
  };

  const getHardwareGuideline = () => {
    const mName = (selectedLocalModel || customModelName || "qwen2.5:0.5b").toLowerCase();
    if (modelType === "openai" || modelType === "openai-chat-completions") {
      return { level: "api", text: "Cloud API Model: Runs on remote servers. No local GPU/VRAM memory footprint. Zero risk of Out of Memory (OOM) errors." };
    }
    if (mName.includes("0.5b") || mName.includes("500m")) {
      return { level: "safe", text: "🟢 Tiny Model (~0.5B params): Requires ~1GB-2GB VRAM. Fully safe and extremely fast on your 4GB VRAM GPU." };
    }
    if (mName.includes("1.5b") || mName.includes("1b") || mName.includes("1.3b")) {
      return { level: "safe", text: "🟢 Small Model (~1B - 1.5B params): Requires ~2GB-3GB VRAM. Safely fits inside your 4GB VRAM GPU." };
    }
    if (mName.includes("3b") || mName.includes("4b")) {
      return { level: "tight", text: "🟡 Medium Model (~3B - 4B params): Requires ~4GB-6GB VRAM. Will be very tight. Running through Ollama is recommended as it offloads excess layers to CPU system memory, avoiding crashes." };
    }
    if (mName.includes("7b") || mName.includes("8b") || mName.includes("latest")) {
      return { level: "danger", text: "🔴 Large Model (~7B - 8B params): Requires ~14GB+ VRAM in 16-bit. Exceeds your 4GB VRAM! If running locally, you must run via Ollama (which will offload heavily to CPU RAM, causing slow generation) or use API models." };
    }
    if (mName.includes("70b") || mName.includes("72b") || mName.includes("405b")) {
      return { level: "danger", text: "🔴 Extreme Model (>70B params): Requires 140GB+ VRAM. Local execution is impossible on your computer. Please evaluate this model via cloud APIs." };
    }
    return { level: "info", text: "⚪ Hardware Check: Ensure model parameters match your VRAM capacity. For 4GB VRAM, prioritize models under 2B parameters." };
  };

  const calculateAverageScore = (summary) => {
    if (!summary || Object.keys(summary).length === 0) return null;
    let total = 0, count = 0;
    Object.keys(summary).forEach(taskKey => {
      const taskMetrics = summary[taskKey];
      const metricKeys = Object.keys(taskMetrics).filter(
        m => !m.toLowerCase().includes("stderr") && typeof taskMetrics[m] === "number"
      );
      if (metricKeys.length > 0) {
        let bestKey = metricKeys[0];
        const preferences = ["exact_match", "acc_norm", "acc", "f1", "bleu", "rouge"];
        for (const pref of preferences) {
          const found = metricKeys.find(m => m.toLowerCase().includes(pref));
          if (found) { bestKey = found; break; }
        }
        total += taskMetrics[bestKey];
        count += 1;
      }
    });
    return count > 0 ? total / count : null;
  };

  const exportToMarkdown = () => {
    if (!selectedResult) return;
    const avg = calculateAverageScore(selectedResult.summary);
    const dateStr = new Date(selectedResult.date * 1000).toLocaleString();
    let content = `# LM Evaluation Harness Run Report\n\n`;
    content += `* **Model Evaluated**: \`${selectedResult.model}\`\n`;
    content += `* **Evaluation Date**: ${dateStr}\n`;
    if (avg !== null) content += `* **Aggregate Point Value (Avg Score)**: **${(avg * 100).toFixed(1)}%**\n`;
    content += `* **Run ID/Filename**: \`${selectedResult.filename}\`\n\n`;
    content += `## Benchmark Metrics Summary\n\n`;
    content += `| Task / Dataset | Metric / Filter | Score |\n`;
    content += `| :--- | :--- | :---: |\n`;
    Object.keys(selectedResult.summary).forEach(taskKey => {
      const taskMetrics = selectedResult.summary[taskKey];
      Object.keys(taskMetrics).filter(m => !m.includes("_stderr")).forEach(metricKey => {
        const val = taskMetrics[metricKey];
        content += `| \`${taskKey}\` | \`${metricKey}\` | **${(val * 100).toFixed(1)}%** |\n`;
      });
    });
    content += `\n*Report dynamically generated by LM-Eval Dashboard.*\n`;
    const blob = new Blob([content], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `report_${selectedResult.model.replace(/[:/]/g, "_")}_${selectedResult.filename.replace(".json", "")}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToCSV = () => {
    if (!selectedResult) return;
    let csvContent = "Task,Metric,Score\n";
    Object.keys(selectedResult.summary).forEach(taskKey => {
      const taskMetrics = selectedResult.summary[taskKey];
      Object.keys(taskMetrics).filter(m => !m.includes("_stderr")).forEach(metricKey => {
        const val = taskMetrics[metricKey];
        csvContent += `"${taskKey}","${metricKey}",${(val * 100).toFixed(2)}\n`;
      });
    });
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `metrics_${selectedResult.model.replace(/[:/]/g, "_")}_${selectedResult.filename.replace(".json", "")}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleStartEvaluation = async (e) => {
    e.preventDefault();
    if (selectedTasks.length === 0) { alert("Please select at least one evaluation task!"); return; }

    let activeModelType = modelType;
    let activeModelArgs = getCompiledModelArgs();

    const LOGLIKELIHOOD_PATTERNS = ["mmlu", "arc", "hellaswag", "winogrande", "piqa", "boolq", "openbookqa", "sciq"];
    const isOllamaBackend = modelType === "local-chat-completions" || modelType === "local-completions";
    const loglikelihoodTasks = selectedTasks.filter(task => LOGLIKELIHOOD_PATTERNS.some(p => task.toLowerCase().includes(p)));
    const generationTasks = selectedTasks.filter(task => !LOGLIKELIHOOD_PATTERNS.some(p => task.toLowerCase().includes(p)));

    if (isOllamaBackend && loglikelihoodTasks.length > 0) {
      if (generationTasks.length > 0) {
        const removeConfirm = window.confirm(
          `⚠️ Ollama Limitation: The following tasks require log-probability scoring which Ollama does NOT support:\n\n` +
          `  ❌ ${loglikelihoodTasks.join(", ")}\n\n` +
          `The following generation-based tasks CAN run on Ollama:\n\n` +
          `  ✅ ${generationTasks.join(", ")}\n\n` +
          `Would you like to automatically remove the incompatible tasks and run only the compatible ones?`
        );
        if (removeConfirm) { setSelectedTasks(generationTasks); } else { return; }
      } else {
        alert(
          `❌ Ollama Limitation: All your selected tasks (${loglikelihoodTasks.join(", ")}) require log-probability scoring, which Ollama's API does not support.\n\n` +
          `Please select generation-based tasks instead (e.g. gsm8k, humaneval, truthfulqa) or switch to HuggingFace Transformers backend.`
        );
        return;
      }
    }

    if (modelType === "local-chat-completions") {
      setModelType("local-completions");
      activeModelType = "local-completions";
      const endpoint = "/v1/completions";
      const parts = [];
      const mName = selectedLocalModel || "qwen3:4b";
      parts.push(`model=${mName}`);
      parts.push(`base_url=http://127.0.0.1:11434${endpoint}`);
      parts.push(`tokenizer=${getTokenizerForModel(mName)}`);
      parts.push("tokenized_requests=False");
      activeModelArgs = parts.join(",");
    }

    setLogLines(["[System] Initializing subprocess...", "[System] Compiling model configurations..."]);
    setRunStatus("running");
    setActiveTab("terminal");

    const tasksToRun = (isOllamaBackend && loglikelihoodTasks.length > 0) ? generationTasks : selectedTasks;
    if (tasksToRun.length === 0) { alert("No compatible tasks remaining to run."); return; }

    const payload = {
      model: activeModelType.startsWith("openai") ? "openai" : activeModelType === "hf" ? "hf" : activeModelType,
      model_args: activeModelArgs,
      tasks: tasksToRun,
      limit: limit ? parseFloat(limit) : null,
      num_fewshot: numFewshot ? parseInt(numFewshot) : null,
      apply_chat_template: applyChatTemplate,
      log_samples: logSamples,
      api_keys: apiKeys
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/eval/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server error starting process");
      }
      const data = await response.json();
      const newRunId = data.run_id;
      setRunId(newRunId);

      const eventSource = new EventSource(`http://127.0.0.1:8000/api/eval/stream/${newRunId}`);
      eventSource.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.log) setLogLines(prev => [...prev, payload.log]);
        if (payload.status) {
          setRunStatus(payload.status);
          if (payload.status === "completed" || payload.status === "failed") {
            eventSource.close();
            fetchResults();
          }
        }
      };
      eventSource.onerror = (err) => {
        console.error("SSE stream error", err);
        setRunStatus("failed");
        eventSource.close();
      };
    } catch (err) {
      setLogLines(prev => [...prev, `[System Error] ${err.message}`]);
      setRunStatus("failed");
    }
  };

  const handleStartPipeline = async (e) => {
    e.preventDefault();
    if (selectedTasks.length === 0) { alert("Please select at least one evaluation task!"); return; }

    let activeModelType = modelType;
    let activeModelArgs = getCompiledModelArgs();

    const LOGLIKELIHOOD_PATTERNS = ["mmlu", "arc", "hellaswag", "winogrande", "piqa", "boolq", "openbookqa", "sciq"];
    const isOllamaBackend = modelType === "local-chat-completions" || modelType === "local-completions";
    const loglikelihoodTasks = selectedTasks.filter(task => LOGLIKELIHOOD_PATTERNS.some(p => task.toLowerCase().includes(p)));
    const generationTasks = selectedTasks.filter(task => !LOGLIKELIHOOD_PATTERNS.some(p => task.toLowerCase().includes(p)));

    if (isOllamaBackend && loglikelihoodTasks.length > 0) {
      if (generationTasks.length > 0) {
        const removeConfirm = window.confirm(
          `⚠️ Ollama Limitation: The following tasks require log-probability scoring which Ollama does NOT support:\n\n` +
          `  ❌ ${loglikelihoodTasks.join(", ")}\n\n` +
          `The following generation-based tasks CAN run on Ollama:\n\n` +
          `  ✅ ${generationTasks.join(", ")}\n\n` +
          `Would you like to automatically remove the incompatible tasks and run only the compatible ones?`
        );
        if (removeConfirm) { setSelectedTasks(generationTasks); } else { return; }
      } else {
        alert(
          `❌ Ollama Limitation: All your selected tasks (${loglikelihoodTasks.join(", ")}) require log-probability scoring, which Ollama's API does not support.\n\n` +
          `Please select generation-based tasks instead (e.g. gsm8k, humaneval, truthfulqa) or switch to HuggingFace Transformers backend.`
        );
        return;
      }
    }

    if (modelType === "local-chat-completions") {
      setModelType("local-completions");
      activeModelType = "local-completions";
      const endpoint = "/v1/completions";
      const parts = [];
      const mName = selectedLocalModel || "qwen3:4b";
      parts.push(`model=${mName}`);
      parts.push(`base_url=http://127.0.0.1:11434${endpoint}`);
      parts.push(`tokenizer=${getTokenizerForModel(mName)}`);
      parts.push("tokenized_requests=False");
      activeModelArgs = parts.join(",");
    }

    setLogLines(["[System] Initializing subprocess...", "[System] Compiling pipeline configurations..."]);
    setRunStatus("running");
    setActiveTab("terminal");

    const tasksToRun = (isOllamaBackend && loglikelihoodTasks.length > 0) ? generationTasks : selectedTasks;
    if (tasksToRun.length === 0) { alert("No compatible tasks remaining to run."); return; }

    const payload = {
      model: activeModelType.startsWith("openai") ? "openai" : activeModelType === "hf" ? "hf" : activeModelType,
      model_args: activeModelArgs,
      tasks: tasksToRun,
      limit: limit ? parseFloat(limit) : null,
      num_fewshot: numFewshot ? parseInt(numFewshot) : null,
      apply_chat_template: applyChatTemplate,
      log_samples: logSamples,
      api_keys: apiKeys,
      is_pipeline: true,
      pipeline_runs: parseInt(pipelineRuns) || 7,
      resume_pipeline: resumePipeline
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/api/eval/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server error starting pipeline process");
      }
      const data = await response.json();
      const newRunId = data.run_id;
      setRunId(newRunId);

      const eventSource = new EventSource(`http://127.0.0.1:8000/api/eval/stream/${newRunId}`);
      eventSource.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.log) setLogLines(prev => [...prev, payload.log]);
        if (payload.status) {
          setRunStatus(payload.status);
          if (payload.status === "completed" || payload.status === "failed") {
            eventSource.close();
            fetchResults();
          }
        }
      };
      eventSource.onerror = (err) => {
        console.error("SSE stream error", err);
        setRunStatus("failed");
        eventSource.close();
      };
    } catch (err) {
      setLogLines(prev => [...prev, `[System Error] ${err.message}`]);
      setRunStatus("failed");
    }
  };

  const handleCancelRun = async () => {
    if (!runId) return;
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/eval/cancel/${runId}`, { method: "POST" });
      if (res.ok) {
        setRunStatus("cancelled");
        setLogLines(prev => [...prev, "\n[System] Process terminated by user request.\n"]);
      }
    } catch (err) {
      console.error("Error cancelling process:", err);
    }
  };

  const viewResultDetails = async (resultSummary) => {
    setSelectedResult(resultSummary);
    setResultDetails(null);
    setSamplesList([]);
    setSamplesFilter("all");
    try {
      const fileRes = await fetch(`http://127.0.0.1:8000/api/results/file/${resultSummary.relative_path}`);
      if (fileRes.ok) {
        const data = await fileRes.json();
        setResultDetails(data);
      }
      const runId = resultSummary.filename.replace("results_", "").replace(".json", "");
      const samplesFile = resultsList.find(
        r => r.is_samples && r.filename.includes(runId) && r.model === resultSummary.model
      );
      if (samplesFile) {
        const sRes = await fetch(`http://127.0.0.1:8000/api/results/file/${samplesFile.relative_path}`);
        if (sRes.ok) {
          const sData = await sRes.json();
          const allSamples = [];
          Object.keys(sData).forEach(taskKey => {
            sData[taskKey].forEach(sample => allSamples.push({ ...sample, task: taskKey }));
          });
          setSamplesList(allSamples);
        }
      }
    } catch (e) {
      console.error("Failed loading result details:", e);
    }
  };

  const getTasksByCategory = () => {
    if (taskCategory === "groups") return tasksRegistry.groups || [];
    if (taskCategory === "tags") return tasksRegistry.tags || [];
    return tasksRegistry.tasks || [];
  };

  const filteredTasks = getTasksByCategory().filter(t =>
    t.toLowerCase().includes(taskSearch.toLowerCase())
  );

  // ── Topbar config per tab ──
  const topbarConfig = {
    new_eval:  { icon: <Play size={17} />,     title: "New Evaluation Run",      sub: "Configure and launch a benchmarking session" },
    pipeline:  { icon: <Zap size={17} />,      title: "Multi-Run Pipeline",      sub: "Run multiple seeds for statistically stable scores" },
    terminal:  { icon: <Terminal size={17} />, title: "Live Console",            sub: "Real-time subprocess stdout/stderr stream" },
    results:   { icon: <BarChart2 size={17} />, title: "Results & History",       sub: "Browse, compare, and export past evaluation runs" },
    keys:      { icon: <Key size={17} />,      title: "Credentials Wallet",      sub: "Securely store API keys in browser localStorage" },
  };
  const tb = topbarConfig[activeTab] || topbarConfig["new_eval"];

  // ── CLI preview strings ──
  const newEvalCLI =
    `.venv/bin/lm-eval --model ${modelType.startsWith("openai") ? "openai" : modelType === "hf" ? "hf" : modelType} \\\n` +
    `  --model_args ${getCompiledModelArgs()} \\\n` +
    `  --tasks ${selectedTasks.length > 0 ? selectedTasks.join(",") : "[no tasks selected]"}` +
    (limit ? ` \\\n  --limit ${limit}` : "") +
    (numFewshot ? ` \\\n  --num_fewshot ${numFewshot}` : "") +
    (applyChatTemplate ? " \\\n  --apply_chat_template" : "") +
    (logSamples ? " \\\n  --log_samples" : "") +
    "\n  --output_path ./eval_results/run_UUID";

  const pipelineCLI =
    `python3 benchmark_pipeline.py \\\n` +
    `  --runs ${pipelineRuns} \\\n` +
    `  --model ${selectedLocalModel || customModelName || "qwen3:4b"} \\\n` +
    `  --model_type ${modelType.startsWith("openai") ? "openai" : modelType === "hf" ? "hf" : modelType} \\\n` +
    `  --tasks ${selectedTasks.length > 0 ? selectedTasks.join(",") : "[no tasks selected]"}` +
    (numFewshot ? ` \\\n  --num_fewshot ${numFewshot}` : "") +
    (resumePipeline ? " \\\n  --resume" : "");

  const hwGuide = getHardwareGuideline();

  // ── Popular benchmarks ──
  const newEvalBenchmarks = [
    { id: "mmlu",          label: "MMLU",          type: "groups" },
    { id: "gsm8k",         label: "GSM8K",         type: "tasks" },
    { id: "arc_challenge", label: "ARC-Challenge",  type: "tasks" },
    { id: "hellaswag",     label: "HellaSwag",      type: "tasks" },
    { id: "truthfulqa",    label: "TruthfulQA",     type: "groups" },
    { id: "humaneval",     label: "HumanEval",      type: "tasks" },
  ];

  const pipelineBenchmarks = [
    { id: "gsm8k",      label: "GSM8K",      type: "tasks" },
    { id: "truthfulqa", label: "TruthfulQA", type: "groups" },
    { id: "humaneval",  label: "HumanEval",  type: "tasks" },
  ];

  // ── Terminal line classifier ──
  const getLineClass = (line) => {
    if (line.startsWith("[System")) return "log-system";
    if (/^\d{4}-/.test(line))       return "log-date";
    if (/error|exception/i.test(line)) return "log-error";
    if (/warn/i.test(line))         return "log-warning";
    return "log-output";
  };

  return (
    <div className="app-container">
      {/* ═══════════════════════════════════════════════
           SIDEBAR
         ═══════════════════════════════════════════════ */}
      <nav className="sidebar">
        <div className="sidebar-top">
          {/* Logo */}
          <div className="logo">
            <div className="logo-icon">
              <BarChart2 size={18} />
            </div>
            <div className="logo-text">
              <span className="logo-name">LM-Eval Dashboard</span>
              <span className="logo-badge">EleutherAI v0.4</span>
            </div>
          </div>

          {/* Nav items */}
          <ul className="nav-list">
            <li className={`nav-item${activeTab === "new_eval" ? " active" : ""}`} onClick={() => setActiveTab("new_eval")}>
              <Play size={16} className="nav-icon" />
              <span className="nav-label">New Run</span>
            </li>
            <li className={`nav-item${activeTab === "pipeline" ? " active" : ""}`} onClick={() => setActiveTab("pipeline")}>
              <Zap size={16} className="nav-icon" />
              <span className="nav-label">Multi-Run Pipeline</span>
            </li>
            <li className={`nav-item${activeTab === "terminal" ? " active" : ""}`} onClick={() => setActiveTab("terminal")}>
              <Terminal size={16} className="nav-icon" />
              <span className="nav-label">Live Console</span>
              {runStatus === "running" && <span className="nav-running-dot" title="Running" />}
            </li>
            <li className={`nav-item${activeTab === "results" ? " active" : ""}`} onClick={() => setActiveTab("results")}>
              <History size={16} className="nav-icon" />
              <span className="nav-label">Results & History</span>
            </li>
            <li className={`nav-item${activeTab === "keys" ? " active" : ""}`} onClick={() => setActiveTab("keys")}>
              <Key size={16} className="nav-icon" />
              <span className="nav-label">Credentials</span>
            </li>
          </ul>
        </div>

        {/* Footer */}
        <div className="sidebar-footer">
          <div className="status-pill">
            <span className={`status-dot ${ollamaRunning ? "connected" : "disconnected"}`} />
            <span>Ollama</span>
            <span className={`status-label ${ollamaRunning ? "connected" : "disconnected"}`}>
              {ollamaRunning ? "Connected" : "Offline"}
            </span>
          </div>
          <button
            className="btn btn-secondary"
            style={{ width: "100%", fontSize: 12, padding: "8px 12px" }}
            onClick={fetchRegistries}
            disabled={loadingRegistries}
          >
            <RefreshCw size={13} className={loadingRegistries ? "animate-spin" : ""} />
            {loadingRegistries ? "Scanning…" : "Refresh Registries"}
          </button>
        </div>
      </nav>

      {/* ═══════════════════════════════════════════════
           MAIN WRAPPER (topbar + content)
         ═══════════════════════════════════════════════ */}
      <div className="main-wrapper">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-left">
            <div className="topbar-icon">{tb.icon}</div>
            <div className="topbar-title">
              <span className="topbar-page-name">{tb.title}</span>
              <span className="topbar-subtitle">{tb.sub}</span>
            </div>
          </div>
          <div className="topbar-right">
            {/* Ollama status pill in topbar */}
            <div className={`badge ${ollamaRunning ? "badge-success" : "badge-danger"}`}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "currentColor", display: "inline-block" }} />
              Ollama {ollamaRunning ? "Connected" : "Offline"}
            </div>
            {runStatus === "running" && (
              <div className="badge badge-running">
                <span className="nav-running-dot" style={{ width: 6, height: 6, margin: 0 }} />
                Running
              </div>
            )}
            <button
              className="btn btn-secondary"
              style={{ padding: "7px 13px", fontSize: 12 }}
              onClick={() => { fetchRegistries(); fetchResults(); }}
            >
              <RefreshCw size={13} />
              Refresh
            </button>
          </div>
        </header>

        {/* Scrollable content */}
        <main className="main-content">

          {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
               TAB: NEW EVALUATION RUN
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          {activeTab === "new_eval" && (
            <div className="animate-fade-in">
              <form onSubmit={handleStartEvaluation} style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 24 }}>
                {/* Left: Model Config + Inference Settings */}
                <div className="glass-panel" style={{ padding: 26, display: "flex", flexDirection: "column", gap: 20 }}>
                  <ModelConfigPanel
                    modelType={modelType} setModelType={setModelType}
                    localModels={localModels}
                    selectedLocalModel={selectedLocalModel} setSelectedLocalModel={setSelectedLocalModel}
                    customModelName={customModelName} setCustomModelName={setCustomModelName}
                    modelArgs={modelArgs} setModelArgs={setModelArgs}
                    headerLabel="Model Backend Wrapper"
                  />

                  <div style={{ borderTop: "1px solid var(--glass-border)", paddingTop: 20 }}>
                    <SectionHeader icon={<Settings size={15} />}>Inference Settings</SectionHeader>
                  </div>

                  {/* Eval mode toggle */}
                  <FormGroup label="Evaluation Run Mode">
                    <div className="pill-toggle">
                      <button
                        type="button"
                        className={`pill-toggle-btn${evalMode === "quick" ? " active" : ""}`}
                        onClick={() => { setEvalMode("quick"); setLimit("5"); }}
                      >
                        <Zap size={13} /> Quick Test (Limit: 5)
                      </button>
                      <button
                        type="button"
                        className={`pill-toggle-btn${evalMode === "complete" ? " active" : ""}`}
                        onClick={() => { setEvalMode("complete"); setLimit(""); }}
                      >
                        <CheckCircle size={13} /> Complete Run (Full Dataset)
                      </button>
                    </div>
                  </FormGroup>

                  {evalMode === "complete" && (
                    <div className="alert alert-warning">
                      <AlertTriangle size={15} />
                      <span><strong>Warning:</strong> Running a complete evaluation tests the entire dataset. This can take minutes to several hours depending on hardware and model size.</span>
                    </div>
                  )}

                  {/* Fewshot + Limit grid */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                    <FormGroup label="Few-Shot Prompt Count">
                      <input
                        type="number"
                        placeholder="e.g. 5"
                        value={numFewshot}
                        onChange={e => setNumFewshot(e.target.value)}
                      />
                    </FormGroup>
                    <FormGroup label="Sample Limit per Task">
                      <input
                        type="number"
                        placeholder="No limit"
                        value={limit}
                        disabled={evalMode === "complete"}
                        onChange={e => setLimit(e.target.value)}
                        style={{ opacity: evalMode === "complete" ? 0.45 : 1 }}
                      />
                    </FormGroup>
                  </div>

                  {/* Checkboxes */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    <label className="checkbox-container">
                      <input type="checkbox" checked={applyChatTemplate} onChange={e => setApplyChatTemplate(e.target.checked)} />
                      <div className="checkbox-mark" />
                      <span style={{ fontSize: 13, color: "var(--text-2)" }}>Apply Chat Template to Input Prompts</span>
                    </label>
                    <label className="checkbox-container">
                      <input type="checkbox" checked={logSamples} onChange={e => setLogSamples(e.target.checked)} />
                      <div className="checkbox-mark" />
                      <span style={{ fontSize: 13, color: "var(--text-2)" }}>Log Output Samples (JSONL format)</span>
                    </label>
                  </div>
                </div>

                {/* Right: Task Picker */}
                <div className="glass-panel" style={{ padding: 26, display: "flex", flexDirection: "column", gap: 16 }}>
                  <TaskPicker
                    tasksRegistry={tasksRegistry}
                    loadingRegistries={loadingRegistries}
                    taskCategory={taskCategory} setTaskCategory={setTaskCategory}
                    taskSearch={taskSearch} setTaskSearch={setTaskSearch}
                    selectedTasks={selectedTasks} setSelectedTasks={setSelectedTasks}
                    popularBenchmarks={newEvalBenchmarks}
                  />

                  <div className="mt-auto" style={{ paddingTop: 16 }}>
                    <button
                      type="submit"
                      className="btn-launch"
                      disabled={selectedTasks.length === 0}
                    >
                      <Play size={17} />
                      Launch Evaluation Run
                    </button>
                  </div>
                </div>

                {/* Full-width CLI + Hardware */}
                <CLIPreviewCard cliCommand={newEvalCLI} hwGuide={hwGuide} />
              </form>
            </div>
          )}

          {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
               TAB: PIPELINE
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          {activeTab === "pipeline" && (
            <div className="animate-fade-in">
              <form onSubmit={handleStartPipeline} style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 24 }}>
                {/* Left: Model + Pipeline Settings */}
                <div className="glass-panel" style={{ padding: 26, display: "flex", flexDirection: "column", gap: 20 }}>
                  <ModelConfigPanel
                    modelType={modelType} setModelType={setModelType}
                    localModels={localModels}
                    selectedLocalModel={selectedLocalModel} setSelectedLocalModel={setSelectedLocalModel}
                    customModelName={customModelName} setCustomModelName={setCustomModelName}
                    modelArgs={modelArgs} setModelArgs={setModelArgs}
                    headerLabel="Model Backend Wrapper"
                  />

                  <div style={{ borderTop: "1px solid var(--glass-border)", paddingTop: 20 }}>
                    <SectionHeader icon={<Settings size={15} />}>Pipeline Settings</SectionHeader>
                  </div>

                  {/* Runs count + Fewshot grid */}
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                    <FormGroup label="Number of Runs (Iterations)">
                      <input
                        type="number"
                        min="1" max="7"
                        value={pipelineRuns}
                        onChange={e => setPipelineRuns(e.target.value)}
                      />
                    </FormGroup>
                    <FormGroup label="Few-Shot Prompt Count">
                      <input
                        type="number"
                        placeholder="e.g. 5"
                        value={numFewshot}
                        onChange={e => setNumFewshot(e.target.value)}
                      />
                    </FormGroup>
                  </div>

                  {/* Checkboxes */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    <label className="checkbox-container">
                      <input type="checkbox" checked={resumePipeline} onChange={e => setResumePipeline(e.target.checked)} />
                      <div className="checkbox-mark" />
                      <span style={{ fontSize: 13, color: "var(--text-1)", fontWeight: 500 }}>Resume Pipeline (skip completed runs)</span>
                    </label>
                    <span style={{ fontSize: 11, color: "var(--text-4)", marginLeft: 27, marginTop: -4, lineHeight: 1.4 }}>
                      Scans output folders and resumes from the first incomplete run.
                    </span>
                    <label className="checkbox-container" style={{ marginTop: 6 }}>
                      <input type="checkbox" checked={applyChatTemplate} onChange={e => setApplyChatTemplate(e.target.checked)} />
                      <div className="checkbox-mark" />
                      <span style={{ fontSize: 13, color: "var(--text-2)" }}>Apply Chat Template to Input Prompts</span>
                    </label>
                    <label className="checkbox-container">
                      <input type="checkbox" checked={logSamples} onChange={e => setLogSamples(e.target.checked)} />
                      <div className="checkbox-mark" />
                      <span style={{ fontSize: 13, color: "var(--text-2)" }}>Log Output Samples (JSONL format)</span>
                    </label>
                  </div>

                  <div className="alert alert-info">
                    <Info size={14} />
                    <span>
                      <strong>Pipeline Behavior:</strong> Runs {pipelineRuns} independent times using random seeds [0, 42, 1234, 2024, 7777, 31415, 99999].
                      Generations use <code>temperature=0.3</code> and <code>top_p=0.95</code> to generate variance.
                    </span>
                  </div>
                </div>

                {/* Right: Task Picker */}
                <div className="glass-panel" style={{ padding: 26, display: "flex", flexDirection: "column", gap: 16 }}>
                  <TaskPicker
                    tasksRegistry={tasksRegistry}
                    loadingRegistries={loadingRegistries}
                    taskCategory={taskCategory} setTaskCategory={setTaskCategory}
                    taskSearch={taskSearch} setTaskSearch={setTaskSearch}
                    selectedTasks={selectedTasks} setSelectedTasks={setSelectedTasks}
                    popularBenchmarks={pipelineBenchmarks}
                  />

                  <div className="mt-auto" style={{ paddingTop: 16 }}>
                    <button
                      type="submit"
                      className="btn-launch"
                      disabled={selectedTasks.length === 0}
                    >
                      <Zap size={17} />
                      Launch Benchmarking Pipeline
                    </button>
                  </div>
                </div>

                {/* Full-width CLI + Hardware */}
                <CLIPreviewCard cliCommand={pipelineCLI} hwGuide={hwGuide} />
              </form>
            </div>
          )}

          {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
               TAB: LIVE CONSOLE / TERMINAL
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          {activeTab === "terminal" && (
            <div className="animate-fade-in" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {/* Terminal header with controls */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  {runStatus === "running" && (
                    <div className="badge badge-running">
                      <span className="nav-running-dot" style={{ width: 6, height: 6, margin: 0 }} />
                      Live — Streaming
                    </div>
                  )}
                  {runStatus === "completed" && (
                    <div className="badge badge-success"><CheckCircle size={11} /> Completed</div>
                  )}
                  {runStatus === "failed" && (
                    <div className="badge badge-danger"><XCircle size={11} /> Failed</div>
                  )}
                  {runStatus === "cancelled" && (
                    <div className="badge badge-warning"><StopCircle size={11} /> Cancelled</div>
                  )}
                  {runStatus === "idle" && (
                    <div className="badge" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid var(--glass-border)", color: "var(--text-3)" }}>
                      Idle — No active run
                    </div>
                  )}
                  {runId && (
                    <span style={{ fontSize: 11, color: "var(--text-4)", fontFamily: "var(--font-mono)" }}>
                      run_id: {runId}
                    </span>
                  )}
                </div>
                {runStatus === "running" && (
                  <button onClick={handleCancelRun} className="btn btn-danger" style={{ gap: 7 }}>
                    <StopCircle size={15} />
                    Terminate Run
                  </button>
                )}
              </div>

              {/* Terminal window */}
              <div className="terminal-window">
                <div className="terminal-topbar">
                  <span className="terminal-dot red" />
                  <span className="terminal-dot yellow" />
                  <span className="terminal-dot green" />
                  <span className="terminal-title">Evaluation Console — lm-eval subprocess</span>
                  {runStatus === "running" && (
                    <div className="badge badge-running" style={{ marginLeft: "auto", fontSize: 10 }}>
                      <RefreshCw size={10} className="animate-spin" /> Live
                    </div>
                  )}
                </div>
                <div className="terminal-body">
                  {logLines.length > 0 ? (
                    logLines.map((line, idx) => (
                      <div key={idx} className={`terminal-line ${getLineClass(line)}`}>
                        {line}
                      </div>
                    ))
                  ) : (
                    <div className="terminal-empty">
                      <Terminal size={46} />
                      <h3>No Active Run</h3>
                      <p>Navigate to <strong>New Run</strong> or <strong>Pipeline</strong> to start an evaluation. Logs will stream here in real time.</p>
                    </div>
                  )}
                  <div ref={terminalEndRef} />
                </div>
              </div>
            </div>
          )}

          {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
               TAB: RESULTS & HISTORY
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          {activeTab === "results" && (
            <div className="animate-fade-in" style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 24, alignItems: "start" }}>
              {/* ── Left: Runs List ── */}
              <div className="glass-panel" style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)", display: "flex", alignItems: "center", gap: 8 }}>
                    <History size={15} style={{ color: "var(--primary-light)" }} />
                    Past Runs
                  </span>
                  <button className="btn btn-secondary" style={{ padding: "5px 10px", fontSize: 11 }} onClick={fetchResults}>
                    <RefreshCw size={12} className={loadingResults ? "animate-spin" : ""} />
                  </button>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: "70vh", overflowY: "auto" }}>
                  {loadingResults ? (
                    <div className="empty-state" style={{ padding: 30 }}>
                      <RefreshCw size={24} className="animate-spin" />
                      <p>Reading records…</p>
                    </div>
                  ) : resultsList.filter(r => !r.is_samples).length > 0 ? (
                    resultsList.filter(r => !r.is_samples).map(run => {
                      const avg = calculateAverageScore(run.summary);
                      return (
                        <div
                          key={run.filename}
                          className={`run-list-item${selectedResult?.filename === run.filename ? " active" : ""}`}
                          onClick={() => viewResultDetails(run)}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                            <div style={{ flex: 1, minWidth: 0 }}>
                              <div className="run-model-name">{run.model}</div>
                              {avg !== null && (
                                <div className="run-avg-score">Avg: {(avg * 100).toFixed(1)}%</div>
                              )}
                            </div>
                            <div className="run-date">
                              <Clock size={10} />
                              {new Date(run.date * 1000).toLocaleDateString()}
                            </div>
                          </div>
                          <div className="run-tasks-wrap">
                            {run.tasks.slice(0, 4).map(t => (
                              <span key={t} className="run-task-chip">{t}</span>
                            ))}
                            {run.tasks.length > 4 && (
                              <span className="run-task-chip">+{run.tasks.length - 4}</span>
                            )}
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <div className="empty-state">
                      <FolderOpen size={32} />
                      <p>No results found in eval_results directory.</p>
                    </div>
                  )}
                </div>
              </div>

              {/* ── Right: Detail Panel ── */}
              {selectedResult ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
                  {/* Score card */}
                  {(() => {
                    const avg = calculateAverageScore(selectedResult.summary);
                    return avg !== null ? (
                      <div className="score-card">
                        <div>
                          <div className="score-label">Aggregate Benchmark Score</div>
                          <div className="score-number">{(avg * 100).toFixed(1)}%</div>
                          <div className="score-sub">
                            Average across {Object.keys(selectedResult.summary).length} evaluated dataset(s).
                          </div>
                        </div>
                        {/* Circular SVG progress */}
                        <div style={{ position: "relative", width: 88, height: 88, flexShrink: 0 }}>
                          <svg width="88" height="88" viewBox="0 0 88 88" style={{ transform: "rotate(-90deg)" }}>
                            <circle cx="44" cy="44" r="36" stroke="rgba(255,255,255,0.05)" strokeWidth="7" fill="transparent" />
                            <circle
                              cx="44" cy="44" r="36"
                              stroke="url(#scoreGrad)"
                              strokeWidth="7"
                              fill="transparent"
                              strokeDasharray="226.2"
                              strokeDashoffset={226.2 - (226.2 * avg)}
                              strokeLinecap="round"
                              style={{ transition: "stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)" }}
                            />
                            <defs>
                              <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#6366f1" />
                                <stop offset="100%" stopColor="#8b5cf6" />
                              </linearGradient>
                            </defs>
                          </svg>
                          <div style={{
                            position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 15, fontWeight: 800, color: "var(--text-1)"
                          }}>
                            {Math.round(avg * 100)}%
                          </div>
                        </div>
                      </div>
                    ) : null;
                  })()}

                  {/* Metrics Table */}
                  <div className="glass-panel" style={{ padding: 24 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: 20 }}>
                      <div>
                        <h2 style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.4px", color: "var(--text-1)", marginBottom: 3 }}>
                          {selectedResult.model}
                        </h2>
                        <span style={{ fontSize: 11.5, color: "var(--text-3)", fontFamily: "var(--font-mono)" }}>
                          {selectedResult.filename}
                        </span>
                      </div>
                      <div style={{ display: "flex", gap: 8 }}>
                        <button className="btn btn-secondary" style={{ fontSize: 12, padding: "6px 12px" }} onClick={exportToMarkdown}>
                          <Download size={13} /> Export MD
                        </button>
                        <button className="btn btn-secondary" style={{ fontSize: 12, padding: "6px 12px" }} onClick={exportToCSV}>
                          <FileText size={13} /> Export CSV
                        </button>
                      </div>
                    </div>

                    <table className="metrics-table">
                      <thead>
                        <tr>
                          <th>Task / Dataset</th>
                          <th>Metric / Filter</th>
                          <th style={{ textAlign: "right" }}>Score</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.keys(selectedResult.summary).map(taskKey => {
                          const taskMetrics = selectedResult.summary[taskKey];
                          return Object.keys(taskMetrics)
                            .filter(m => !m.includes("_stderr") && m !== "name" && m !== "alias" && m !== "sample_len")
                            .map(metricKey => {
                              const val = taskMetrics[metricKey];
                              const isPct = ["acc", "exact_match", "f1", "bleu", "rouge", "pass"].some(k => metricKey.toLowerCase().includes(k));
                              return (
                                <tr key={`${taskKey}-${metricKey}`}>
                                  <td className="task-cell">{taskKey}</td>
                                  <td>{metricKey}</td>
                                  <td className="score-cell">
                                    {isPct ? `${(val * 100).toFixed(1)}%` : (typeof val === "number" ? val.toFixed(2) : val)}
                                  </td>
                                </tr>
                              );
                            });
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Bar Chart */}
                  <div className="glass-panel" style={{ padding: 24 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 20 }}>
                      <BarChart2 size={15} style={{ color: "var(--primary-light)" }} />
                      <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)" }}>Accuracy Chart Comparison</span>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                      {Object.keys(selectedResult.summary).map(taskKey => {
                        const taskMetrics = selectedResult.summary[taskKey];
                        const primaryMetric = Object.keys(taskMetrics).find(m => !m.includes("_stderr") && m !== "name" && m !== "alias" && m !== "sample_len" && typeof taskMetrics[m] === "number") || "";
                        const val = taskMetrics[primaryMetric] || 0;
                        const isPct = ["acc", "exact_match", "f1", "bleu", "rouge", "pass"].some(k => primaryMetric.toLowerCase().includes(k));
                        return (
                          <div key={taskKey} className="metric-bar-row">
                            <div className="metric-bar-header">
                              <span className="metric-bar-task">{taskKey} <span style={{ color: "var(--text-4)", fontSize: 11 }}>({primaryMetric})</span></span>
                              <span className="metric-bar-val">{isPct ? `${(val * 100).toFixed(1)}%` : (typeof val === "number" ? val.toFixed(2) : val)}</span>
                            </div>
                            {isPct && (
                              <div className="metric-bar-track">
                                <div
                                  className="metric-bar-fill"
                                  style={{ width: `${Math.min(100, Math.max(0, val * 100))}%` }}
                                />
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Leaderboard Chart */}
                  <LeaderboardChart 
                    resultsList={resultsList}
                    selectedResult={selectedResult}
                    calculateAverageScore={calculateAverageScore}
                  />

                  {/* Telemetry panel */}
                  {selectedResult.pipeline_details?.telemetry && (
                    <div className="glass-panel" style={{ padding: 24 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 6 }}>
                        <Zap size={15} style={{ color: "var(--primary-light)" }} />
                        <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)" }}>System & API Performance Telemetry</span>
                      </div>
                      <p style={{ fontSize: 12, color: "var(--text-3)", marginBottom: 20 }}>
                        Aggregated across <strong style={{ color: "var(--text-1)" }}>{selectedResult.pipeline_details.telemetry.num_requests}</strong> API requests.
                      </p>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 16 }}>
                        {selectedResult.pipeline_details.telemetry.ttft && (
                          <div className="telemetry-card">
                            <div className="telemetry-label">Time to First Token (TTFT)</div>
                            <div className="telemetry-value">{selectedResult.pipeline_details.telemetry.ttft.mean.toFixed(3)}s</div>
                            <div className="telemetry-grid">
                              <div>p95: <strong>{selectedResult.pipeline_details.telemetry.ttft.p95.toFixed(3)}s</strong></div>
                              <div>p99: <strong>{selectedResult.pipeline_details.telemetry.ttft.p99.toFixed(3)}s</strong></div>
                              <div>min: {selectedResult.pipeline_details.telemetry.ttft.min.toFixed(3)}s</div>
                              <div>max: {selectedResult.pipeline_details.telemetry.ttft.max.toFixed(3)}s</div>
                            </div>
                          </div>
                        )}
                        {selectedResult.pipeline_details.telemetry.itl && (
                          <div className="telemetry-card">
                            <div className="telemetry-label">Inter-Token Latency (ITL)</div>
                            <div className="telemetry-value">{(selectedResult.pipeline_details.telemetry.itl.mean * 1000).toFixed(1)}ms</div>
                            <div className="telemetry-grid">
                              <div>p95: <strong>{(selectedResult.pipeline_details.telemetry.itl.p95 * 1000).toFixed(1)}ms</strong></div>
                              <div>p99: <strong>{(selectedResult.pipeline_details.telemetry.itl.p99 * 1000).toFixed(1)}ms</strong></div>
                              <div>min: {(selectedResult.pipeline_details.telemetry.itl.min * 1000).toFixed(1)}ms</div>
                              <div>max: {(selectedResult.pipeline_details.telemetry.itl.max * 1000).toFixed(1)}ms</div>
                            </div>
                          </div>
                        )}
                        {selectedResult.pipeline_details.telemetry.trt && (
                          <div className="telemetry-card">
                            <div className="telemetry-label">Total Response Time (TRT)</div>
                            <div className="telemetry-value">{selectedResult.pipeline_details.telemetry.trt.mean.toFixed(3)}s</div>
                            <div className="telemetry-grid">
                              <div>p95: <strong>{selectedResult.pipeline_details.telemetry.trt.p95.toFixed(3)}s</strong></div>
                              <div>p99: <strong>{selectedResult.pipeline_details.telemetry.trt.p99.toFixed(3)}s</strong></div>
                              <div>min: {selectedResult.pipeline_details.telemetry.trt.min.toFixed(3)}s</div>
                              <div>max: {selectedResult.pipeline_details.telemetry.trt.max.toFixed(3)}s</div>
                            </div>
                          </div>
                        )}
                        {selectedResult.pipeline_details.telemetry.throughput && (
                          <div className="telemetry-card">
                            <div className="telemetry-label">Token Throughput</div>
                            <div className="telemetry-value" style={{ color: "var(--success)" }}>
                              {selectedResult.pipeline_details.telemetry.throughput.mean.toFixed(2)} t/s
                            </div>
                            <div className="telemetry-grid">
                              <div>p95: <strong>{selectedResult.pipeline_details.telemetry.throughput.p95.toFixed(2)} t/s</strong></div>
                              <div>&nbsp;</div>
                              <div>min: {selectedResult.pipeline_details.telemetry.throughput.min.toFixed(2)} t/s</div>
                              <div>max: {selectedResult.pipeline_details.telemetry.throughput.max.toFixed(2)} t/s</div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Sample Inspector */}
                  {samplesList.length > 0 && (
                    <div className="glass-panel" style={{ padding: 24 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                        <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)", display: "flex", alignItems: "center", gap: 8 }}>
                          <FileText size={15} style={{ color: "var(--primary-light)" }} />
                          Sample Inspector
                        </span>
                        <select
                          style={{ width: "auto", padding: "6px 32px 6px 12px", fontSize: 12 }}
                          value={samplesFilter}
                          onChange={e => setSamplesFilter(e.target.value)}
                        >
                          <option value="all">All Samples ({samplesList.length})</option>
                          <option value="correct">Only Correct</option>
                          <option value="incorrect">Only Incorrect</option>
                        </select>
                      </div>

                      <div style={{ display: "flex", flexDirection: "column", gap: 14, maxHeight: "500px", overflowY: "auto" }}>
                        {samplesList
                          .filter(sample => {
                            const isCorrect = sample.exact_match === 1 || sample.exact_match === true;
                            if (samplesFilter === "correct") return isCorrect;
                            if (samplesFilter === "incorrect") return !isCorrect;
                            return true;
                          })
                          .map((sample, index) => {
                            const isCorrect = sample.exact_match === 1 || sample.exact_match === true;
                            return (
                              <div key={index} className={`sample-card ${isCorrect ? "correct" : "incorrect"}`}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                                  <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-3)" }}>
                                    {sample.task} · ID: {sample.doc_id}
                                  </span>
                                  <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, fontWeight: 700, color: isCorrect ? "var(--success)" : "var(--danger)" }}>
                                    {isCorrect ? <CheckCircle size={13} /> : <XCircle size={13} />}
                                    {isCorrect ? "Correct" : "Incorrect"}
                                  </span>
                                </div>
                                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                  <div>
                                    <span className="sample-field-label">Question</span>
                                    <div className="sample-field-content">
                                      {sample.doc?.question || sample.doc?.prompt || sample.doc?.context || JSON.stringify(sample.doc)}
                                    </div>
                                  </div>
                                  <div>
                                    <span className="sample-field-label">Target Solution</span>
                                    <div className="sample-field-content" style={{ color: "var(--success)", fontWeight: 600 }}>
                                      {sample.target}
                                    </div>
                                  </div>
                                  <div>
                                    <span className="sample-field-label">Model Answer</span>
                                    <div className="sample-field-content">
                                      {Array.isArray(sample.resps) ? sample.resps[0][0] : sample.resps || JSON.stringify(sample.resps)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="glass-panel" style={{ padding: 0 }}>
                  <div className="empty-state">
                    <FolderOpen size={44} />
                    <h3>No Result Selected</h3>
                    <p>Click a run from the list on the left to inspect its detailed metrics, charts, and individual samples.</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
               TAB: CREDENTIALS WALLET
             ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
          {activeTab === "keys" && (
            <div className="animate-fade-in" style={{ maxWidth: 600, display: "flex", flexDirection: "column", gap: 24 }}>
              <div className="glass-panel" style={{ padding: 28, display: "flex", flexDirection: "column", gap: 22 }}>
                <SectionHeader icon={<Key size={15} />}>API Credentials</SectionHeader>

                <FormGroup label="OpenAI API Key">
                  <div style={{ position: "relative" }}>
                    <input
                      type="password"
                      placeholder="sk-proj-..."
                      value={apiKeys.openai}
                      onChange={e => saveApiKeys({ ...apiKeys, openai: e.target.value })}
                    />
                  </div>
                </FormGroup>

                <FormGroup label="Anthropic API Key">
                  <input
                    type="password"
                    placeholder="sk-ant-..."
                    value={apiKeys.anthropic}
                    onChange={e => saveApiKeys({ ...apiKeys, anthropic: e.target.value })}
                  />
                </FormGroup>

                <FormGroup label="Hugging Face Token (HF_TOKEN)">
                  <input
                    type="password"
                    placeholder="hf_..."
                    value={apiKeys.hf}
                    onChange={e => saveApiKeys({ ...apiKeys, hf: e.target.value })}
                  />
                </FormGroup>

                <div className="alert alert-info" style={{ marginTop: 4 }}>
                  <Info size={15} />
                  <span>
                    The backend automatically reads these saved keys and maps them to environment variables
                    (e.g. <code>OPENAI_API_KEY</code>, <code>ANTHROPIC_API_KEY</code>) before launching subprocesses.
                    Keys are stored only in your browser's <code>localStorage</code> and never transmitted to third-party servers.
                  </span>
                </div>
              </div>

              {/* Security note */}
              <div className="glass-panel" style={{ padding: 20 }}>
                <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                  <div style={{ width: 36, height: 36, borderRadius: "var(--r-md)", background: "rgba(16,185,129,0.1)", border: "1px solid rgba(16,185,129,0.2)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <CheckCircle size={16} style={{ color: "var(--success)" }} />
                  </div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)", marginBottom: 4 }}>Local Storage — No Server Persistence</div>
                    <div style={{ fontSize: 12, color: "var(--text-3)", lineHeight: 1.5 }}>
                      Your credentials are encrypted by your browser and stored locally. They are sent only to the local backend server at <code>127.0.0.1:8000</code> when launching evaluations — never to external services directly from the UI.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
