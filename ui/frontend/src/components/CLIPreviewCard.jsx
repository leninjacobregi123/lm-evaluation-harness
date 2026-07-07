import React from "react";
import { Code, Info } from "lucide-react";
import SectionHeader from "./SectionHeader";

export default function CLIPreviewCard({ cliCommand, hwGuide }) {
  return (
    <div className="glass-panel" style={{ padding: 26, gridColumn: "1 / span 2", display: "flex", flexDirection: "column", gap: 16 }}>
      <SectionHeader icon={<Code size={15} />}>Live Configuration Preview & Hardware Advisor</SectionHeader>

      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <span style={{ fontSize: 11.5, fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase", color: "var(--text-3)" }}>
          Equivalent CLI Command
        </span>
        <div className="cli-box">{cliCommand}</div>
      </div>

      <div className={`hw-guide level-${hwGuide.level}`}>
        <Info size={16} style={{ flexShrink: 0, marginTop: 1 }} />
        <span>{hwGuide.text}</span>
      </div>
    </div>
  );
}
