import React from "react";
import { Database, AlertTriangle } from "lucide-react";
import SectionHeader from "./SectionHeader";
import FormGroup from "./FormGroup";

export default function ModelConfigPanel({
  modelType,
  setModelType,
  localModels,
  selectedLocalModel,
  setSelectedLocalModel,
  customModelName,
  setCustomModelName,
  modelArgs,
  setModelArgs,
  headerLabel
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <SectionHeader icon={<Database size={15} />}>{headerLabel || "Model Backend"}</SectionHeader>

      <FormGroup label="Model Type / Framework">
        <select value={modelType} onChange={e => setModelType(e.target.value)}>
          <option value="local-chat-completions">Ollama / Chat Completions (local-chat-completions)</option>
          <option value="local-completions">Ollama / Text Completions (local-completions)</option>
          <option value="openai-chat-completions">OpenAI Chat API (gpt-4o, etc.)</option>
          <option value="hf">HuggingFace Transformers (Local GPU/CPU)</option>
        </select>
      </FormGroup>

      {(modelType === "local-chat-completions" || modelType === "local-completions") && (
        <FormGroup label="Local Ollama Model">
          {localModels.length > 0 ? (
            <select value={selectedLocalModel} onChange={e => setSelectedLocalModel(e.target.value)}>
              {localModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          ) : (
            <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--danger)", fontSize: 13, padding: "10px 0" }}>
              <AlertTriangle size={14} />
              No local models found. Is Ollama running?
            </div>
          )}
        </FormGroup>
      )}

      {(!modelType.startsWith("local") || localModels.length === 0) && (
        <FormGroup label={modelType === "hf" ? "HF Model ID (e.g. Qwen/Qwen2.5-0.5B)" : "Model Name / Identifier"}>
          <input
            type="text"
            placeholder={modelType === "hf" ? "Qwen/Qwen2.5-0.5B" : "gpt-4o"}
            value={customModelName}
            onChange={e => setCustomModelName(e.target.value)}
          />
        </FormGroup>
      )}

      <FormGroup
        label="Manual Model Args Override (Optional)"
        hint="Leave blank — args compile automatically from your selection."
      >
        <input
          type="text"
          placeholder="e.g. dtype=float16,trust_remote_code=True"
          value={modelArgs}
          onChange={e => setModelArgs(e.target.value)}
        />
      </FormGroup>
    </div>
  );
}
