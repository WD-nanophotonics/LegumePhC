// Plot-family renderers. An operation picks one (or supplies its own); they
// draw only the immutable raw rows returned by /api/records/view.
import type Plotly from "plotly.js";
import { colorAt, colorscale } from "../components/Plot";
import type { PlotStyle, RecordViewData } from "../types";
import type { Rendered } from "./types";

const PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"];

function num(value: unknown): number {
  return typeof value === "number" ? value : Number.NaN;
}

function limits(value: unknown): [number, number] | undefined {
  return Array.isArray(value) && value.length === 2 ? [Number(value[0]), Number(value[1])] : undefined;
}

/** The first column (in preference order) that the record actually has. */
export function valueColumn(view: RecordViewData, preferred: string[]): string {
  return preferred.find((key) => view.columns.includes(key))
    ?? view.columns.find((key) => key.startsWith("Normalized frequency") || key.startsWith("Frequency ("))
    ?? "";
}

export function frequencyColumn(view: RecordViewData): string {
  return valueColumn(view, []);
}

/** Lines (or markers) per band along a path; x is the sample index. */
export function renderBandLines(view: RecordViewData, style: PlotStyle, options: { markersOnly?: boolean } = {}): Rendered {
  const rows = view.rows;
  const column = frequencyColumn(view);
  const bands = new Map<number, number[]>();
  rows.forEach((row, index) => {
    const band = num(row.values.Band);
    bands.set(band, [...(bands.get(band) ?? []), index]);
  });
  const mode = options.markersOnly
    ? "markers"
    : [style.band_line !== false ? "lines" : "", style.band_markers ? "markers" : ""].filter(Boolean).join("+") || "lines";
  const data: Plotly.Data[] = [...bands.entries()].map(([band, indices], k) => ({
    type: "scatter",
    mode: mode as Plotly.PlotData["mode"],
    name: `Band ${band}`,
    x: indices.map((i) => rows[i].x),
    y: indices.map((i) => rows[i].y),
    customdata: indices as unknown as Plotly.Datum[],
    line: { width: Number(style.linewidth ?? 1.5), color: PALETTE[k % PALETTE.length] },
    marker: { size: Math.max(4, Number(style.marker_size ?? 4)) * (options.markersOnly ? 2.5 : 1), color: PALETTE[k % PALETTE.length] },
    hovertemplate: `Band ${band}<br>${column}=%{y:.6g}<extra></extra>`,
  }));
  const ticks = new Map<number, string>();
  rows.forEach((row) => typeof row.values.Path === "string" && ticks.set(row.x, String(row.values.Path)));
  return {
    data,
    layout: {
      showlegend: Boolean(style.legend ?? true),
      xaxis: { title: { text: String(style.x_label || (ticks.size ? "Path" : "")) }, tickvals: [...ticks.keys()], ticktext: [...ticks.values()], showgrid: Boolean(style.grid ?? true), range: limits(style.x_limits) },
      yaxis: { title: { text: String(style.y_label || column) }, showgrid: Boolean(style.grid ?? true), range: limits(style.y_limits) },
    },
  };
}

export interface MapOptions {
  /** Preferred value columns, first match wins. */
  columns: string[];
  /** Real-space map (x/a, y/a) instead of reciprocal (qₓ, qᵧ). */
  spatial?: boolean;
  /** Draw `view.sample_cells` as filled tiles when available. */
  sampleCells?: boolean;
  /** Mark rows whose `Qualified` value is false. */
  qualification?: boolean;
}

/** Colour-mapped scatter over a 2D domain, optionally tiled by sample cells. */
export function renderMap(view: RecordViewData, style: PlotStyle, options: MapOptions): Rendered {
  const rows = view.rows;
  const column = valueColumn(view, options.columns);
  const cmap = String(style.cmap ?? "RdBu_r");
  const values = rows.map((row) => num(row.values[column]));
  const finite = values.filter(Number.isFinite);
  const vmin = style.berry_vmin != null ? Number(style.berry_vmin) : Math.min(...finite);
  const vmax = style.berry_vmax != null ? Number(style.berry_vmax) : Math.max(...finite);
  const data: Plotly.Data[] = [];
  const interpolate = style.berry_render_mode === "linear_interpolation";
  const cells = options.sampleCells && !interpolate && view.sample_cells?.length === rows.length ? view.sample_cells : undefined;
  if (cells) {
    cells.forEach((polygon, index) => {
      if (!polygon?.length || !Number.isFinite(values[index])) return;
      const ring = [...polygon, polygon[0]];
      data.push({
        type: "scatter", mode: "lines", fill: "toself", hoverinfo: "skip", showlegend: false,
        x: ring.map((p) => p[0]), y: ring.map((p) => p[1]),
        fillcolor: colorAt(cmap, values[index], vmin, vmax), line: { width: 0.3, color: "rgba(0,0,0,0.25)" },
      });
    });
  } else if (options.sampleCells && interpolate) {
    data.push({
      type: "contour", x: rows.map((r) => r.x), y: rows.map((r) => r.y), z: values,
      colorscale: colorscale(cmap), zmin: vmin, zmax: vmax, showscale: false, contours: { coloring: "heatmap" }, line: { width: 0 },
      hoverinfo: "skip", connectgaps: true,
    } as Plotly.Data);
  }
  const showMarkers = !cells || Boolean(style.show_sample_centers);
  data.push({
    type: rows.length > 2000 ? "scattergl" : "scatter",
    mode: "markers",
    name: column,
    x: rows.map((r) => r.x),
    y: rows.map((r) => r.y),
    customdata: rows.map((_, i) => i) as unknown as Plotly.Datum[],
    marker: {
      color: values, colorscale: colorscale(cmap), cmin: vmin, cmax: vmax,
      size: showMarkers ? Math.max(5, Number(style.marker_size ?? 4) * 2) : 10,
      opacity: showMarkers ? 1 : 0.001,
      symbol: options.spatial ? "square" : "circle",
      line: { width: showMarkers && cells ? 0.6 : 0, color: "#111" },
      showscale: Boolean(style.colorbar ?? true),
      colorbar: { title: column, thickness: 12 },
    },
    showlegend: false,
    hovertemplate: `${options.spatial ? "x" : "qₓ"}=%{x:.6g}<br>${options.spatial ? "y" : "qᵧ"}=%{y:.6g}<br>${column}=%{marker.color:.6g}<extra></extra>`,
  });
  if (options.qualification) {
    const failed = rows.map((row, i) => (row.values.Qualified === false ? i : -1)).filter((i) => i >= 0);
    if (failed.length) {
      data.push({
        type: "scatter", mode: "markers", name: "not qualified", x: failed.map((i) => rows[i].x), y: failed.map((i) => rows[i].y),
        marker: { symbol: "x-thin", size: 9, line: { width: 1.5, color: "#111" } }, hoverinfo: "skip",
      });
    }
  }
  const outline = view.arrays.domain_outline as number[][] | undefined;
  if (outline?.length) {
    const ring = [...outline, outline[0]];
    data.push({ type: "scatter", mode: "lines", name: "domain", x: ring.map((p) => p[0]), y: ring.map((p) => p[1]), line: { color: "#555", width: 1.5 }, hoverinfo: "skip" });
  }
  return {
    data,
    layout: {
      showlegend: Boolean(style.legend ?? true),
      legend: { orientation: "h", x: 0, y: -0.18 },
      xaxis: { title: { text: String(style.x_label || (options.spatial ? "x / a" : "qₓ (reduced)")) }, scaleanchor: "y", showgrid: Boolean(style.grid ?? true), range: limits(style.x_limits) },
      yaxis: { title: { text: String(style.y_label || (options.spatial ? "y / a" : "qᵧ (reduced)")) }, showgrid: Boolean(style.grid ?? true), range: limits(style.y_limits) },
    },
  };
}
