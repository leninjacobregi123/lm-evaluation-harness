import React from "react";
import { BarChart2, Info, CheckCircle, Search, RefreshCw } from "lucide-react";
import SectionHeader from "./SectionHeader";
import TaskTypeBadge from "./TaskTypeBadge";

export default function TaskPicker({
  tasksRegistry,
  loadingRegistries,
  taskCategory,
  setTaskCategory,
  taskSearch,
  setTaskSearch,
  selectedTasks,
  setSelectedTasks,
  popularBenchmarks
}) {
  const getTasksByCategory = () => {
    if (taskCategory === "groups") return tasksRegistry.groups || [];
    if (taskCategory === "tags") return tasksRegistry.tags || [];
    return tasksRegistry.tasks || [];
  };

  const filteredTasks = getTasksByCategory().filter(t =>
    t.toLowerCase().includes(taskSearch.toLowerCase())
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <SectionHeader icon={<BarChart2 size={15} />}>Select Evaluation Tasks</SectionHeader>

      {/* Category tabs */}
      <div className="category-tabs">
        {[
          { id: "tasks",  label: `Subtasks (${tasksRegistry.tasks?.length || 0})` },
          { id: "groups", label: `Groups (${tasksRegistry.groups?.length || 0})` },
          { id: "tags",   label: `Tags (${tasksRegistry.tags?.length || 0})` },
        ].map(cat => (
          <button
            key={cat.id}
            type="button"
            className={`category-tab${taskCategory === cat.id ? " active" : ""}`}
            onClick={() => setTaskCategory(cat.id)}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Category guide */}
      <div className="info-panel">
        <div className="info-panel-title"><Info size={13} /> Category Definitions</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, color: "var(--text-2)" }}>
          <div>• <strong>Subtasks</strong>: Isolated tests (e.g. <code>mmlu_anatomy</code>, <code>gsm8k</code>).</div>
          <div>• <strong>Groups</strong>: Bundled subtasks — selecting <code>mmlu</code> runs all 57 subjects at once.</div>
          <div>• <strong>Tags</strong>: Dataset labels used for leaderboards (e.g. <code>leaderboard</code>).</div>
        </div>
      </div>

      {/* Popular benchmarks */}
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--text-3)" }}>
          ⭐ Popular Benchmarks
        </span>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {popularBenchmarks.map(item => {
            const isSelected = selectedTasks.includes(item.id);
            return (
              <button
                key={item.id}
                type="button"
                className={`benchmark-chip${isSelected ? " selected" : ""}`}
                onClick={() => {
                  setTaskCategory(item.type);
                  if (isSelected) {
                     setSelectedTasks(selectedTasks.filter(t => t !== item.id));
                  } else {
                     setSelectedTasks([...selectedTasks, item.id]);
                  }
                }}
              >
                {isSelected && <CheckCircle size={11} />}
                {item.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected chips */}
      {selectedTasks.length > 0 && (
        <div className="selected-tasks-box">
          <span className="selected-tasks-label">Active Selection — {selectedTasks.length} task(s)</span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {selectedTasks.map(t => (
              <span key={t} className="task-chip">
                {t}
                <span
                  className="task-chip-remove"
                  onClick={() => setSelectedTasks(selectedTasks.filter(st => st !== t))}
                >×</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Search */}
      <div className="search-wrap">
        <Search size={15} className="search-icon" />
        <input
          type="text"
          placeholder="Search benchmarks..."
          value={taskSearch}
          onChange={e => setTaskSearch(e.target.value)}
        />
      </div>

      {/* Task list */}
      <div className="task-list">
        {loadingRegistries ? (
          <div style={{ color: "var(--text-3)", textAlign: "center", padding: 20, fontSize: 12 }}>
            <RefreshCw size={18} style={{ animation: "spin 1s linear infinite", margin: "0 auto 8px", display: "block" }} />
            Scanning task index…
          </div>
        ) : taskSearch.trim() === "" ? (
          <div className="empty-state" style={{ padding: "20px 10px", minHeight: "auto" }}>
            <Search size={22} />
            <p>Search above to find tasks, or click a popular benchmark.</p>
          </div>
        ) : filteredTasks.length > 0 ? (
          filteredTasks.map(task => (
            <label key={task} className="task-list-item checkbox-container" style={{ padding: "7px 10px", borderRadius: "var(--r-sm)" }}>
              <input
                type="checkbox"
                checked={selectedTasks.includes(task)}
                onChange={e => {
                  if (e.target.checked) {
                    setSelectedTasks([...selectedTasks, task]);
                  } else {
                    setSelectedTasks(selectedTasks.filter(t => t !== task));
                  }
                }}
              />
              <div className="checkbox-mark" />
              <span className="task-name">{task}</span>
              <TaskTypeBadge taskName={task} />
            </label>
          ))
        ) : (
          <div style={{ color: "var(--text-3)", textAlign: "center", padding: 20, fontSize: 12 }}>
            No matching tasks found.
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="selection-footer">
        <span className="selection-count">{selectedTasks.length} task(s) selected</span>
        {selectedTasks.length > 0 && (
          <span className="selection-clear" onClick={() => setSelectedTasks([])}>Clear all</span>
        )}
      </div>
    </div>
  );
}
