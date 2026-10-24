import { useEffect, useRef, useState } from "react";
import type { RecordViewData, ResultEntry } from "../types";

interface Props {
  view: RecordViewData | null;
  result: ResultEntry | null;
  selected: number | null;
  pins: number[];
  onSelect: (index: number, pin: boolean) => void;
  progress: { label: string; fraction: number | null; running: boolean };
  log: string[];
}

function format(value: unknown): string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (value === 0) return "0";
    const magnitude = Math.abs(value);
    return magnitude < 1e-5 || magnitude >= 1e6 ? value.toExponential(5) : String(Number(value.toPrecision(6)));
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value ?? "");
}

const MAX_ROWS = 2000;

export function BottomPanel({ view, result, selected, pins, onSelect, progress, log }: Props) {
  const [tab, setTab] = useState<"inspector" | "details" | "log">("inspector");
  const logEnd = useRef<HTMLDivElement>(null);
  const selectedRow = useRef<HTMLTableRowElement>(null);
  useEffect(() => logEnd.current?.scrollIntoView({ block: "nearest" }), [log.length]);
  useEffect(() => selectedRow.current?.scrollIntoView({ block: "nearest" }), [selected]);
  const copy = () => {
    if (!view || selected == null) return;
    const row = view.rows[selected];
    void navigator.clipboard.writeText(Object.entries(row.values).map(([k, v]) => `${k}: ${format(v)}`).join("\n"));
  };
  const summary = view?.summary as Record<string, unknown> | undefined;
  const qualification = summary?.qualification as Record<string, unknown> | undefined;
  return (
    <section className="bottom">
      <div className="tabs">
        <button className={tab === "inspector" ? "active" : ""} onClick={() => setTab("inspector")}>Data inspector</button>
        <button className={tab === "details" ? "active" : ""} onClick={() => setTab("details")}>Result details</button>
        <button className={tab === "log" ? "active" : ""} onClick={() => setTab("log")}>Progress / log</button>
        <div className="progress">
          <div className={`bar ${progress.running && progress.fraction == null ? "indeterminate" : ""}`}>
            <div style={{ width: `${Math.round((progress.fraction ?? (progress.running ? 0.3 : 0)) * 100)}%` }} />
          </div>
          <span>{progress.label}</span>
        </div>
      </div>
      <div className="bottom-body">
        {tab === "inspector" && (
          view ? (
            <>
              <div className="inspector-tools">
                <span className="muted">{view.rows.length} raw samples · click a row or point to select, Shift+click to pin</span>
                <button disabled={selected == null} onClick={copy}>Copy values</button>
              </div>
              <table className="inspector">
                <thead><tr><th>#</th>{view.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
                <tbody>
                  {view.rows.slice(0, MAX_ROWS).map((row, index) => (
                    <tr key={index} ref={index === selected ? selectedRow : undefined}
                      className={`${index === selected ? "selected" : ""} ${pins.includes(index) ? "pinned" : ""}`}
                      onClick={(e) => onSelect(index, e.shiftKey)}>
                      <td>{pins.includes(index) ? "📌" : index}</td>
                      {view.columns.map((column) => <td key={column}>{format(row.values[column])}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
              {view.rows.length > MAX_ROWS && <div className="muted">Showing first {MAX_ROWS} rows; export for the full record.</div>}
            </>
          ) : <div className="empty">No result selected.</div>
        )}
        {tab === "details" && (
          view && result ? (
            <div className="cards">
              <div className="card"><h4>Record</h4><code>{result.record_reference.path}</code>
                <div className="muted">operation: {view.operation}</div>
                {Object.entries(result.record_reference.identity ?? {}).filter(([k]) => k !== "affine").map(([k, v]) => (
                  <div key={k} className="kv"><span>{k}</span><span>{typeof v === "object" ? JSON.stringify(v) : String(v)}</span></div>
                ))}
              </div>
              {qualification && (
                <div className="card"><h4>Qualification</h4>
                  {Object.entries(qualification).filter(([, v]) => typeof v !== "object").map(([k, v]) => (
                    <div key={k} className="kv"><span>{k}</span><span className={String(v).includes("QUALIFIED") && !String(v).includes("UN") ? "ok" : String(v).includes("WITHHELD") || String(v).includes("UNQUALIFIED") ? "warn" : ""}>{String(v)}</span></div>
                  ))}
                </div>
              )}
              <div className="card"><h4>Summary</h4>
                {Object.entries(summary ?? {}).filter(([k, v]) => k !== "qualification" && typeof v !== "object").map(([k, v]) => (
                  <div key={k} className="kv"><span>{k}</span><span>{format(v)}</span></div>
                ))}
              </div>
              <div className="card wide"><h4>Calculation snapshot</h4><pre>{JSON.stringify(result.calculation_snapshot, null, 2)}</pre></div>
            </div>
          ) : <div className="empty">No result selected.</div>
        )}
        {tab === "log" && (
          <div className="log">
            {log.map((line, index) => <div key={index}>{line}</div>)}
            <div ref={logEnd} />
          </div>
        )}
      </div>
    </section>
  );
}
