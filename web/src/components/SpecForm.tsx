// Generic renderers for the declarative FieldSpec / StyleSpec contracts.
import { useEffect, useState } from "react";
import type { CalculationParameters, PlotStyle } from "../types";
import type { FieldContext, FieldSpec, StyleSpec } from "../operations/types";

function fmt(value: unknown): string {
  return value == null ? "" : String(value);
}

/** Text input that only commits on blur/Enter, validating with `parse`. */
function Committed<T>({ label, value, parse, onCommit, readOnly, placeholder, hint, type }: {
  label: string; value: string; parse: (text: string) => T; onCommit: (v: T) => void;
  readOnly?: boolean; placeholder?: string; hint?: string; type?: string;
}) {
  const [text, setText] = useState(value);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { setText(value); setError(null); }, [value]);
  const commit = () => {
    if (text === value) return;
    try {
      onCommit(parse(text));
      setError(null);
    } catch (exc) {
      setError((exc as Error).message);
    }
  };
  return (
    <label>{label}
      <input className={error ? "bad" : ""} value={text} type={type} readOnly={readOnly} placeholder={placeholder}
        onChange={(e) => setText(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === "Enter" && commit()} />
      {error ? <span className="field-error">{error}</span> : hint ? <span className="hint">{hint}</span> : null}
    </label>
  );
}

const parseNumber = (min?: number, integer = false) => (text: string) => {
  const value = Number(text);
  if (!text.trim() || !Number.isFinite(value)) throw new Error("enter a number");
  if (integer && !Number.isInteger(value)) throw new Error("enter a whole number");
  if (min !== undefined && value < min) throw new Error(`must be ≥ ${min}`);
  return value;
};

const parseList = (text: string): number[] | null => {
  const values = text.split(/[,\s]+/).filter(Boolean).map(Number);
  if (values.some((v) => !Number.isFinite(v))) throw new Error("enter comma-separated numbers");
  return values.length ? values : null;
};

const parseCenters = (text: string): number[][] => {
  const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
  return lines.map((line, i) => {
    const pair = line.split(/[,\s]+/).filter(Boolean).map(Number);
    if (pair.length !== 2 || pair.some((v) => !Number.isFinite(v))) throw new Error(`line ${i + 1}: expected “qₓ, qᵧ”`);
    return pair;
  });
};

function CentersField({ label, value, onCommit }: { label: string; value: number[][]; onCommit: (v: number[][]) => void }) {
  const initial = (value ?? []).map((c) => c.join(", ")).join("\n");
  const [text, setText] = useState(initial);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => setText(initial), [initial]);
  return (
    <label>{label}
      <textarea rows={5} className={error ? "bad" : ""} value={text} onChange={(e) => setText(e.target.value)}
        onBlur={() => { try { onCommit(parseCenters(text)); setError(null); } catch (exc) { setError((exc as Error).message); } }} />
      {error && <span className="field-error">{error}</span>}
    </label>
  );
}

export function FieldForm({ fields, parameters, onChange, context }: {
  fields: FieldSpec[]; parameters: CalculationParameters; onChange: (patch: Partial<CalculationParameters>) => void; context: FieldContext;
}) {
  const p = parameters;
  const render = (field: FieldSpec) => {
    if (field.visible && !field.visible(p)) return null;
    const readOnly = field.readOnly?.(p);
    const value = (p as Record<string, unknown>)[field.key];
    switch (field.kind) {
      case "row":
        return <div className="row" key={field.key}>{field.fields.map(render)}</div>;
      case "note":
        return <div className="hint" key={field.key}>{field.label}</div>;
      case "number":
      case "int":
        return <Committed key={field.key} label={field.label} value={fmt(value)} readOnly={readOnly} hint={field.hint}
          parse={parseNumber(field.min, field.kind === "int")} onCommit={(v) => onChange({ [field.key]: v })} />;
      case "select":
        return (
          <label key={field.key}>{field.label}
            <select value={fmt(value)} disabled={readOnly} onChange={(e) => onChange({ [field.key]: e.target.value })}>
              {field.options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
            </select>
          </label>
        );
      case "list":
        return <Committed key={field.key} label={field.label} placeholder={field.placeholder} value={((value as number[] | null) ?? []).join(", ")}
          parse={parseList} onCommit={(v) => onChange({ [field.key]: v })} />;
      case "qpoint":
        return (
          <div className="row" key={field.key}>
            <Committed label="qₓ (reduced)" value={fmt(p.qpoint[0])} parse={parseNumber()} onCommit={(v) => onChange({ qpoint: [v, p.qpoint[1]] })} />
            <Committed label="qᵧ (reduced)" value={fmt(p.qpoint[1])} parse={parseNumber()} onCommit={(v) => onChange({ qpoint: [p.qpoint[0], v] })} />
          </div>
        );
      case "centers":
        return <CentersField key={field.key} label={field.label} value={p.centers} onCommit={(v) => onChange({ centers: v })} />;
      case "berrySource":
        return (
          <label key={field.key}>{field.label}
            <select value={fmt(value)} onChange={(e) => onChange({ [field.key]: e.target.value || null })}>
              <option value="">No source selected</option>
              {context.berrySources.map((entry) => (
                <option key={entry.id} value={entry.record_reference.path}>{entry.record_reference.path.split("/").pop()}</option>
              ))}
            </select>
            {field.hint && <span className="hint">{field.hint}</span>}
          </label>
        );
    }
  };
  return <>{fields.map(render)}</>;
}

function LimitsField({ label, value, onCommit }: { label: string; value: unknown; onCommit: (v: [number, number] | null) => void }) {
  const current = Array.isArray(value) ? value.map(String) : ["", ""];
  const [low, setLow] = useState(current[0]);
  const [high, setHigh] = useState(current[1]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { setLow(current[0]); setHigh(current[1]); }, [current[0], current[1]]);
  const commit = () => {
    try {
      const a = low.trim() ? parseNumber()(low) : null;
      const b = high.trim() ? parseNumber()(high) : null;
      if ((a === null) !== (b === null)) throw new Error("both limits are required");
      if (a !== null && b !== null && a >= b) throw new Error("minimum must be smaller than maximum");
      onCommit(a === null || b === null ? null : [a, b]);
      setError(null);
    } catch (exc) {
      setError((exc as Error).message);
    }
  };
  return (
    <label>{label}
      <span className="row">
        <input value={low} placeholder="auto" onChange={(e) => setLow(e.target.value)} onBlur={commit} />
        <input value={high} placeholder="auto" onChange={(e) => setHigh(e.target.value)} onBlur={commit} />
      </span>
      {error && <span className="field-error">{error}</span>}
    </label>
  );
}

export function StyleForm({ specs, style, onChange }: { specs: StyleSpec[]; style: PlotStyle; onChange: (patch: PlotStyle) => void }) {
  const checks = specs.filter((spec) => spec.kind === "check");
  const others = specs.filter((spec) => spec.kind !== "check");
  return (
    <>
      {others.map((spec) => {
        const value = style[spec.key];
        switch (spec.kind) {
          case "select":
            return (
              <label key={spec.key}>{spec.label}
                <select value={fmt(value)} onChange={(e) => onChange({ [spec.key]: e.target.value })}>
                  {spec.options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
                </select>
              </label>
            );
          case "number":
            return <Committed key={spec.key} label={spec.label} value={fmt(value)} parse={parseNumber(spec.min)} onCommit={(v) => onChange({ [spec.key]: v })} />;
          case "optionalNumber":
            return <Committed key={spec.key} label={spec.label} value={fmt(value)} placeholder="auto"
              parse={(text) => (text.trim() ? parseNumber()(text) : null)} onCommit={(v) => onChange({ [spec.key]: v })} />;
          case "text":
            return <Committed key={spec.key} label={spec.label} value={fmt(value)} parse={(t) => t} onCommit={(v) => onChange({ [spec.key]: v })} />;
          case "limits":
            return <LimitsField key={spec.key} label={spec.label} value={value} onCommit={(v) => onChange({ [spec.key]: v })} />;
          default:
            return null;
        }
      })}
      {checks.length > 0 && (
        <div className="checks">
          {checks.map((spec) => (
            <label key={spec.key} className="check">
              <input type="checkbox" checked={Boolean(style[spec.key] ?? false)} onChange={(e) => onChange({ [spec.key]: e.target.checked })} />{spec.label}
            </label>
          ))}
        </div>
      )}
    </>
  );
}
