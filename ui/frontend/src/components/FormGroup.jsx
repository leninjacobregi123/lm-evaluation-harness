import React from "react";

export default function FormGroup({ label, hint, children }) {
  return (
    <div className="form-group">
      {label && <label className="form-label">{label}</label>}
      {children}
      {hint && <span className="form-hint">{hint}</span>}
    </div>
  );
}
