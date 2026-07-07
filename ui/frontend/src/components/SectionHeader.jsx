import React from "react";

export default function SectionHeader({ icon, children }) {
  return (
    <div className="section-header">
      {icon}
      <span>{children}</span>
    </div>
  );
}
