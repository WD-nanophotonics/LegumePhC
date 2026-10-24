import type { Calculation, RecordViewData, ResultEntry } from "../types";

interface Props {
  study: Calculation | null;
  solution: ResultEntry | null;
  view: RecordViewData | null;
  onAddPlot: () => void;
}

function formatValue(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

/** Dataset = one study's immutable solution record. */
export function DatasetSettings({ study, solution, view, onAddPlot }: Props) {
  if (!study || !solution) return <div className="empty">This study has no solution yet.</div>;
  const qualification = (view?.summary?.qualification ?? null) as Record<string, unknown> | null;
  return (
    <div className="form">
      <div className="section-title first">{study.name} / Solution</div>
      <div className="kv"><span>Solved</span><span>{solution.solved_at ? new Date(solution.solved_at).toLocaleString() : "—"}</span></div>
      <div className="kv"><span>Samples</span><span>{view?.rows.length ?? "…"}</span></div>
      {Object.entries(solution.record_reference.identity ?? {}).filter(([key]) => key !== "affine").map(([key, value]) => (
        <div key={key} className="kv"><span>{key}</span><span>{formatValue(value)}</span></div>
      ))}
      {qualification && (
        <>
          <div className="section-title">Qualification (as recorded)</div>
          {Object.entries(qualification).filter(([, v]) => typeof v !== "object").map(([key, value]) => {
            const text = String(value);
            const tone = text.includes("UNQUALIFIED") || text.includes("WITHHELD") ? "warn" : text.includes("QUALIFIED") ? "ok" : "";
            return <div key={key} className="kv"><span>{key}</span><span className={tone}>{text}</span></div>;
          })}
        </>
      )}
      <div className="section-title">Record</div>
      <code className="path">{solution.record_reference.path}</code>
      <div className="button-row"><button onClick={onAddPlot}>＋ Add plot group for this dataset</button></div>
    </div>
  );
}
