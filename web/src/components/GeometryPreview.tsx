import { useMemo, useState } from "react";
import type Plotly from "plotly.js";
import { Plot } from "./Plot";
import type { PreviewData } from "../types";

const LABELS: Record<string, string> = {
  unit_cell: "Unit cell",
  motif_array: "Motif array",
  lattice_sites: "Lattice sites",
  reciprocal_bz: "Reciprocal BZ",
  epsilon: "Epsilon",
};

function closed(points: number[][]): { x: number[]; y: number[] } {
  const all = [...points, points[0]];
  return { x: all.map((p) => p[0]), y: all.map((p) => p[1]) };
}

function motifOutline(motif: NonNullable<PreviewData["motifs"]>[number]): { x: number[]; y: number[] } {
  if (motif.kind === "polygon" && motif.vertices) return closed(motif.vertices);
  const n = 64;
  const rx = motif.kind === "ellipse" ? (motif.width ?? 0) / 2 : motif.radius ?? 0;
  const ry = motif.kind === "ellipse" ? (motif.height ?? 0) / 2 : motif.radius ?? 0;
  const phi = ((motif.angle_degrees ?? 0) * Math.PI) / 180;
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i <= n; i++) {
    const t = (2 * Math.PI * i) / n;
    const px = rx * Math.cos(t);
    const py = ry * Math.sin(t);
    x.push(motif.center[0] + px * Math.cos(phi) - py * Math.sin(phi));
    y.push(motif.center[1] + px * Math.sin(phi) + py * Math.cos(phi));
  }
  return { x, y };
}

function traces(data: PreviewData): Plotly.Data[] {
  const out: Plotly.Data[] = [];
  if (data.epsilon && data.x_grid && data.y_grid) {
    const x = data.x_grid.flat();
    const y = data.y_grid.flat();
    const eps = data.epsilon.flat();
    out.push({
      type: "scatter", mode: "markers", x, y,
      marker: { color: eps, colorscale: "Viridis", size: 4, symbol: "square", colorbar: { title: "ε", thickness: 12 } },
      hovertemplate: "x/a=%{x:.4f}<br>y/a=%{y:.4f}<br>ε=%{marker.color:.4g}<extra></extra>",
      name: "ε",
    });
  }
  if (data.points && (data.view === "lattice_sites" || data.view === "reciprocal_bz")) {
    out.push({
      type: "scatter", mode: "markers", x: data.points.map((p) => p[0]), y: data.points.map((p) => p[1]),
      marker: { size: 8, color: "#8ecae6", line: { color: "#219ebc", width: 1 } }, name: "sites",
    });
  }
  if (data.bz_vertices) {
    out.push({ type: "scatter", mode: "lines", ...closed(data.bz_vertices), line: { color: "#6b7280", width: 2 }, name: "1st BZ" } as Plotly.Data);
  }
  if (data.motifs && data.view === "unit_cell") {
    data.motifs.forEach((motif, index) => {
      out.push({ type: "scatter", mode: "lines", ...motifOutline(motif), line: { color: "#ffffff", width: 1.5 }, name: `site ${index + 1}`, hoverinfo: "skip" });
    });
  }
  if (data.direct_basis && (data.view === "lattice_sites" || data.view === "unit_cell")) {
    const b = data.direct_basis;
    out.push({ type: "scatter", mode: "lines", x: [0, b[0][0]], y: [0, b[1][0]], line: { color: "#d62828", width: 2.5 }, name: "a₁" });
    out.push({ type: "scatter", mode: "lines", x: [0, b[0][1]], y: [0, b[1][1]], line: { color: "#1d4ed8", width: 2.5 }, name: "a₂" });
  }
  return out;
}

export function GeometryPreview({ previews, error }: { previews: Record<string, PreviewData> | null; error: string | null }) {
  const [view, setView] = useState("unit_cell");
  const data = previews?.[view];
  const plotData = useMemo(() => (data ? traces(data) : []), [data]);
  const layout = useMemo<Partial<Plotly.Layout>>(() => ({
    showlegend: false,
    xaxis: { scaleanchor: "y", title: { text: view === "reciprocal_bz" ? "kₓ a" : "x / a" } },
    yaxis: { title: { text: view === "reciprocal_bz" ? "kᵧ a" : "y / a" } },
  }), [view]);
  return (
    <div className="geometry-preview">
      <div className="subtabs">
        {Object.entries(LABELS).map(([key, label]) => (
          <button key={key} className={key === view ? "active" : ""} onClick={() => setView(key)}>{label}</button>
        ))}
        {data?.point_group !== undefined && <span className="badge">point group: {data.point_group ?? "not verified (generic basis)"}</span>}
      </div>
      {error ? <div className="error-box">Preview failed: {error}</div> : <Plot data={plotData} layout={layout} />}
    </div>
  );
}
