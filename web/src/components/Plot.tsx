import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export interface PlotProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  onClickPoint?: (point: Plotly.PlotDatum, event: MouseEvent) => void;
  className?: string;
}

function themeLayout(): Partial<Plotly.Layout> {
  const style = getComputedStyle(document.documentElement);
  const fg = style.getPropertyValue("--fg").trim() || "#1b1f24";
  const grid = style.getPropertyValue("--grid").trim() || "#e3e6ea";
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: fg, family: "Inter, Segoe UI, system-ui, sans-serif", size: 12 },
    xaxis: { gridcolor: grid, zerolinecolor: grid },
    yaxis: { gridcolor: grid, zerolinecolor: grid },
    margin: { l: 56, r: 16, t: 28, b: 44 },
  };
}

function merge(base: Partial<Plotly.Layout>, extra: Partial<Plotly.Layout>): Partial<Plotly.Layout> {
  const result: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(extra)) {
    const current = result[key];
    result[key] = current && typeof current === "object" && value && typeof value === "object" && !Array.isArray(value)
      ? { ...(current as object), ...(value as object) }
      : value;
  }
  return result as Partial<Plotly.Layout>;
}

/** Thin imperative wrapper around Plotly; re-renders with Plotly.react. */
export function Plot({ data, layout = {}, onClickPoint, className }: PlotProps) {
  const host = useRef<HTMLDivElement>(null);
  const clickRef = useRef(onClickPoint);
  clickRef.current = onClickPoint;

  useEffect(() => {
    const element = host.current;
    if (!element) return;
    Plotly.react(element, data, merge(themeLayout(), { autosize: true, ...layout }), {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    });
  }, [data, layout]);

  useEffect(() => {
    const element = host.current as (HTMLDivElement & { on?: (name: string, cb: (e: Plotly.PlotMouseEvent) => void) => void }) | null;
    if (!element?.on) return;
    element.on("plotly_click", (event: Plotly.PlotMouseEvent) => {
      const point = event.points?.[0];
      if (point && clickRef.current) clickRef.current(point, event.event as MouseEvent);
    });
    return () => {
      (element as unknown as { removeAllListeners?: (name: string) => void }).removeAllListeners?.("plotly_click");
    };
  }, [data]);

  useEffect(() => {
    const element = host.current;
    return () => {
      if (element) Plotly.purge(element);
    };
  }, []);

  return <div ref={host} className={className ?? "plot"} />;
}

const SCALES: Record<string, [number, string][]> = {
  RdBu_r: [[0, "#2166ac"], [0.25, "#67a9cf"], [0.5, "#f7f7f7"], [0.75, "#ef8a62"], [1, "#b2182b"]],
  coolwarm: [[0, "#3b4cc0"], [0.5, "#dddddd"], [1, "#b40426"]],
  viridis: [[0, "#440154"], [0.25, "#3b528b"], [0.5, "#21918c"], [0.75, "#5ec962"], [1, "#fde725"]],
  plasma: [[0, "#0d0887"], [0.25, "#7e03a8"], [0.5, "#cc4778"], [0.75, "#f89540"], [1, "#f0f921"]],
  magma: [[0, "#000004"], [0.25, "#51127c"], [0.5, "#b73779"], [0.75, "#fc8961"], [1, "#fcfdbf"]],
};

export function colorscale(name: string): [number, string][] {
  return SCALES[name] ?? SCALES.viridis;
}

function hex(value: string): [number, number, number] {
  const n = parseInt(value.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Map a value to an rgb() colour on a named scale (for filled polygons). */
export function colorAt(name: string, value: number, vmin: number, vmax: number): string {
  const scale = colorscale(name);
  const t = vmax > vmin ? Math.min(1, Math.max(0, (value - vmin) / (vmax - vmin))) : 0.5;
  for (let i = 1; i < scale.length; i++) {
    const [t1, c1] = scale[i];
    if (t <= t1) {
      const [t0, c0] = scale[i - 1];
      const f = (t - t0) / (t1 - t0 || 1);
      const a = hex(c0);
      const b = hex(c1);
      const mix = a.map((channel, k) => Math.round(channel + (b[k] - channel) * f));
      return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
    }
  }
  return scale[scale.length - 1][1];
}
