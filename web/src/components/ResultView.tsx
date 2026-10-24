import { useMemo } from "react";
import type Plotly from "plotly.js";
import { Plot } from "./Plot";
import { OPERATION_BY_ID } from "../operations/definitions";
import { frequencyColumn } from "../operations/renderers";
import type { PlotStyle, RecordViewData } from "../types";

interface Props {
  view: RecordViewData | null;
  style: PlotStyle;
  selected: number | null;
  pins: number[];
  onSelect: (index: number, pin: boolean) => void;
  message?: string | null;
}

function ScalarCard({ view }: { view: RecordViewData }) {
  const row = view.rows[0];
  if (!row) return <div className="empty">This record has no samples.</div>;
  const label = frequencyColumn(view);
  return (
    <div className="scalar-result">
      <div className="scalar-value">{Number(row.values[label]).toPrecision(8)}</div>
      <div className="scalar-label">{label} · band {String(row.values.Band)} · q = ({String(row.values["qₓ"])}, {String(row.values["qᵧ"])})</div>
    </div>
  );
}

export function ResultView({ view, style, selected, pins, onSelect, message }: Props) {
  const op = view ? OPERATION_BY_ID[view.operation] : undefined;
  const built = useMemo(() => (view && op ? op.render(view, style) : null), [view, op, style]);
  const data = useMemo(() => {
    if (!view || !built) return [];
    const marks = [...new Set([...(selected != null ? [selected] : []), ...pins])].filter((i) => i < view.rows.length);
    if (!marks.length) return built.data;
    return [
      ...built.data,
      {
        type: "scatter", mode: "text+markers", name: "selection", showlegend: false, hoverinfo: "skip",
        x: marks.map((i) => view.rows[i].x), y: marks.map((i) => view.rows[i].y),
        text: marks.map((i) => (pins.includes(i) ? "📌" : "")), textposition: "top center",
        marker: { size: 16, color: "rgba(0,0,0,0)", line: { width: 2.5, color: "#f59e0b" } },
      } as Plotly.Data,
    ];
  }, [view, built, selected, pins]);
  const layout = useMemo(() => ({ ...(built?.layout ?? {}), title: { text: String(style.title ?? "") } }), [built, style.title]);
  if (message) return <div className="empty">{message}</div>;
  if (!view) return <div className="empty">Select a result in the project tree, or run a calculation.</div>;
  if (!op) return <div className="empty">No renderer for operation “{view.operation}”; see the Data inspector.</div>;
  if (!built) return <ScalarCard view={view} />;
  return (
    <Plot
      data={data}
      layout={layout}
      onClickPoint={(point, event) => {
        const index = Number(point.customdata);
        if (Number.isInteger(index)) onSelect(index, event.shiftKey);
      }}
    />
  );
}
