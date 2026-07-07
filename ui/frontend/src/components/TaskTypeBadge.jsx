import React from "react";

export default function TaskTypeBadge({ taskName }) {
  const LOGLIKELIHOOD_PATTERNS = ["mmlu", "arc", "hellaswag", "winogrande", "piqa", "boolq", "openbookqa", "sciq"];
  const isLikelihood = LOGLIKELIHOOD_PATTERNS.some(p => taskName.toLowerCase().includes(p));
  return (
    <span className={`task-badge ${isLikelihood ? "task-badge-likelihood" : "task-badge-generation"}`}>
      {isLikelihood ? "Likelihood" : "Generation"}
    </span>
  );
}
