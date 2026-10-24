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
      {view?.valley && view.valley.bands.length > 0 && (
        <>
          <div className="section-title">Valley Chern · {view.valley.valley} half BZ</div>
          <table className="sites-table valley-table">
            <thead><tr><th>Band</th><th>C_v (source grid)</th><th>C_v (clipped)</th><th>Qualified</th><th>Min gap</th></tr></thead>
            <tbody>
              {view.valley.bands.map((band) => {
                const value = (estimate: typeof band.SOURCE_GRID_MIDPOINT) => estimate.valley_chern === null ? <span className="warn">not reported</span> : estimate.valley_chern.toFixed(4);
                return (
                  <tr key={band.band_zero_based}>
                    <td>{band.band_zero_based + 1}</td>
                    <td>{value(band.SOURCE_GRID_MIDPOINT)}</td>
                    <td>{value(band.CLIPPED_CELL)}</td>
                    <td>{band.SOURCE_GRID_MIDPOINT.qualified_samples}/{band.SOURCE_GRID_MIDPOINT.total_samples}</td>
                    <td>{band.min_gap.toFixed(4)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="hint">LegumePhC curvature sign (+arg∏⟨u|u′⟩/area, = −Ω of Xiao–Chang–Niu). A band is reported only when every sample qualifies.</div>
        </>
      )}
      <div className="section-title">Record</div>
      <code className="path">{solution.record_reference.path}</code>
      <div className="button-row"><button onClick={onAddPlot}>＋ Add plot group for this dataset</button></div>
    </div>
  );
}
