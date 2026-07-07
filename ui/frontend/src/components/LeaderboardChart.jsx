import React, { useState } from "react";
import { BarChart2, TrendingUp, Calendar } from "lucide-react";
import SectionHeader from "./SectionHeader";

export default function LeaderboardChart({ resultsList, selectedResult, calculateAverageScore }) {
  const [selectedTask, setSelectedTask] = useState("Aggregate Average");

  // Extract all unique tasks evaluated across all history records
  const allTasks = new Set();
  resultsList.forEach(run => {
    if (!run.is_samples && run.summary) {
      Object.keys(run.summary).forEach(task => allTasks.add(task));
    }
  });
  const taskOptions = ["Aggregate Average", ...Array.from(allTasks)];

  // Compile matching runs and scores
  const comparisonData = resultsList
    .filter(run => !run.is_samples && run.summary)
    .map(run => {
      let score = null;
      let metricName = "";

      if (selectedTask === "Aggregate Average") {
        score = calculateAverageScore(run.summary);
        metricName = "Average Accuracy";
      } else if (run.summary[selectedTask]) {
        const taskMetrics = run.summary[selectedTask];
        const primaryMetric = Object.keys(taskMetrics).find(m => !m.includes("_stderr") && m !== "name" && m !== "alias" && m !== "sample_len" && typeof taskMetrics[m] === "number") || "";
        score = typeof taskMetrics[primaryMetric] === "number" ? taskMetrics[primaryMetric] : null;
        metricName = primaryMetric;
      }

      return {
        filename: run.filename,
        model: run.model,
        date: run.date,
        score: score,
        metric: metricName
      };
    })
    .filter(item => item.score !== null)
    // Sort descending by score
    .sort((a, b) => b.score - a.score);

  return (
    <div className="glass-panel" style={{ padding: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
        <SectionHeader icon={<TrendingUp size={15} />}>Historical Leaderboard Comparison</SectionHeader>
        
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11.5, color: "var(--text-3)", fontWeight: 600 }}>COMPARE BY:</span>
          <select 
            value={selectedTask} 
            onChange={e => setSelectedTask(e.target.value)}
            style={{ width: "auto", padding: "6px 12px", fontSize: 12 }}
          >
            {taskOptions.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>
      </div>

      {comparisonData.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {comparisonData.map((item, index) => {
            const isCurrent = selectedResult?.filename === item.filename;
            const percentage = Math.min(100, Math.max(0, item.score * 100));
            const formattedDate = new Date(item.date * 1000).toLocaleDateString();

            return (
              <div 
                key={item.filename} 
                className={`metric-bar-row${isCurrent ? " active" : ""}`}
                style={{
                  padding: "10px 14px",
                  borderRadius: "var(--r-md)",
                  background: isCurrent ? "rgba(99, 102, 241, 0.08)" : "rgba(255, 255, 255, 0.01)",
                  border: `1px solid ${isCurrent ? "var(--primary)" : "transparent"}`,
                  transition: "all 0.2s var(--ease)"
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ 
                      fontSize: 11, 
                      fontWeight: 800, 
                      color: index === 0 ? "#f59e0b" : "var(--text-3)",
                      background: index === 0 ? "rgba(245, 158, 11, 0.12)" : "rgba(255,255,255,0.03)",
                      width: 20,
                      height: 20,
                      borderRadius: "50%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center"
                    }}>
                      {index + 1}
                    </span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-1)" }}>
                      {item.model}
                    </span>
                    {isCurrent && (
                      <span className="badge badge-info" style={{ fontSize: 9, padding: "1px 5px", verticalAlign: "middle" }}>
                        Active Run
                      </span>
                    )}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{ fontSize: 11, color: "var(--text-3)", display: "flex", alignItems: "center", gap: 4 }}>
                      <Calendar size={11} /> {formattedDate}
                    </span>
                    <span style={{ fontSize: 13, fontWeight: 700, color: isCurrent ? "var(--primary-light)" : "var(--success)" }}>
                      {percentage.toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="metric-bar-track" style={{ height: 6, background: "rgba(255,255,255,0.02)" }}>
                  <div
                    className="metric-bar-fill"
                    style={{ 
                      width: `${percentage}%`,
                      height: "100%",
                      background: isCurrent 
                        ? "linear-gradient(90deg, var(--primary) 0%, var(--primary-light) 100%)" 
                        : "linear-gradient(90deg, var(--violet) 0%, var(--primary) 100%)",
                      borderRadius: 99
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div style={{ color: "var(--text-3)", textAlign: "center", padding: 20 }}>
          No historical comparison data found.
        </div>
      )}
    </div>
  );
}
