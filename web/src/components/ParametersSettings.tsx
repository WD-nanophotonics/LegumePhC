import { useEffect, useState } from "react";
import type { Parameter } from "../types";

interface Props {
  parameters: Parameter[];
  values: Record<string, number>;
  error: { index: number; field: string; message: string } | null;
  onCommit: (parameters: Parameter[]) => void;
}

/** Global Definitions › Parameters: names usable in every geometry expression. */
export function ParametersSettings({ parameters, values, error, onCommit }: Props) {
  const [rows, setRows] = useState<Parameter[]>(parameters);
  useEffect(() => setRows(parameters), [parameters]);
  const edit = (index: number, key: keyof Parameter, value: string) => setRows(rows.map((row, i) => (i === index ? { ...row, [key]: value } : row)));
  const commit = (next = rows) => onCommit(next.filter((row) => row.name.trim() || row.expression.trim()));
  return (
    <div className="form">
      <div className="section-title first">Parameters</div>
      <div className="hint">Use these names in any geometry field, e.g. radius <code>r0</code> or centre <code>a1/2</code>. Changing a value re-evaluates the whole component.</div>
      <table className="sites-table params-table">
        <thead><tr><th>Name</th><th>Expression</th><th>Value</th><th>Description</th><th /></tr></thead>
        <tbody>
          {rows.map((row, index) => {
            const bad = (field: string) => (error && error.index === index && error.field === field ? "bad" : "");
            const cell = (key: keyof Parameter) => (
              <td><input className={bad(key)} value={row[key] ?? ""} onChange={(e) => edit(index, key, e.target.value)}
                onBlur={() => commit()} onKeyDown={(e) => e.key === "Enter" && commit()} /></td>
            );
            return (
              <tr key={index}>
                {cell("name")}
                {cell("expression")}
                <td className="value">{values[row.name] !== undefined ? Number(values[row.name].toPrecision(8)) : "—"}</td>
                {cell("description")}
                <td className="row-actions"><button title="Remove parameter" onClick={() => commit(rows.filter((_, i) => i !== index))}>✕</button></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="button-row small"><button onClick={() => setRows([...rows, { name: `p${rows.length + 1}`, expression: "0", description: "" }])}>＋ Add parameter</button></div>
      {error && <div className="error-box">{error.message}</div>}
    </div>
  );
}
