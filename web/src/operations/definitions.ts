// One entry per Studio operation. Adding an operation = one OperationDef here
// plus its backend implementation (see ./types.ts).
import type Plotly from "plotly.js";
import type { CalculationParameters } from "../types";
import { renderBandLines, renderMap } from "./renderers";
import type { FieldSpec, OperationDef, StyleSpec } from "./types";

const isSingleBerry = (p: CalculationParameters) => p.berry_target_mode === "single_band";

// ---- shared field groups --------------------------------------------------
const solver = (options: { numeigReadOnly?: boolean } = {}): FieldSpec => ({
  kind: "row", key: "solver", label: "",
  fields: [
    { kind: "number", key: "gmax", label: "gmax", min: 0 },
    { kind: "int", key: "numeig", label: "Eigenmodes", min: 1, readOnly: () => Boolean(options.numeigReadOnly), hint: options.numeigReadOnly ? "Derived: highest target band + 1" : undefined },
    { kind: "select", key: "polarization", label: "Polarization", options: [["te", "TE"], ["tm", "TM"]] },
  ],
});
const qpoint = (visible?: FieldSpec["visible"]): FieldSpec => ({ kind: "qpoint", key: "qpoint", label: "q (reduced Cartesian)", visible });
const band = (visible?: FieldSpec["visible"]): FieldSpec => ({ kind: "int", key: "band_one_based", label: "Target band (one-based)", min: 1, visible });
const bandRange = (visible?: FieldSpec["visible"]): FieldSpec => ({
  kind: "row", key: "band_range", label: "", visible,
  fields: [
    { kind: "int", key: "berry_first_band", label: "First band", min: 1 },
    { kind: "int", key: "berry_last_band", label: "Last band", min: 1 },
  ],
});

// ---- shared style groups ---------------------------------------------------
const frequencyUnit: StyleSpec = { key: "frequency_unit", label: "Frequency unit", kind: "select", options: [["Normalized", "Normalized (ωa/2πc)"], ["GHz", "GHz"], ["THz", "THz"]] };
const colourMap: StyleSpec[] = [
  { key: "cmap", label: "Colour map", kind: "select", options: ["RdBu_r", "coolwarm", "viridis", "plasma", "magma"].map((n) => [n, n]) },
  { key: "colorbar", label: "Colour bar", kind: "check" },
];
const colourRange: StyleSpec[] = [
  { key: "berry_vmin", label: "Colour min", kind: "optionalNumber" },
  { key: "berry_vmax", label: "Colour max", kind: "optionalNumber" },
];

export const OPERATIONS: OperationDef[] = [
  {
    id: "frequency_at_k",
    label: "Frequency at k",
    fields: [qpoint(), band(), solver()],
    style: [frequencyUnit],
    viewKeys: ["frequency_unit"],
    render: () => null,
    describe: (c) => `band ${c.band_one_based}, q=(${c.qpoint?.join(", ")})`,
  },
  {
    id: "band_structure",
    label: "Band Structure",
    fields: [
      solver(),
      { kind: "int", key: "samples_per_segment", label: "Samples per segment", min: 2 },
      { kind: "note", key: "path_note", label: "Path: automatic high-symmetry identity path of the effective lattice." },
    ],
    style: [
      frequencyUnit,
      { key: "band_line", label: "Band lines", kind: "check" },
      { key: "band_markers", label: "Band markers", kind: "check" },
      { key: "linewidth", label: "Line width", kind: "number", step: 0.1, min: 0.1 },
      { key: "marker_size", label: "Marker size", kind: "number", step: 0.5, min: 0.5 },
    ],
    viewKeys: ["frequency_unit"],
    render: (view, style) => renderBandLines(view, style),
    describe: (c) => `gmax=${c.gmax}, ${c.samples_per_segment}/seg`,
  },
  {
    id: "fields_energy",
    label: "Fields / Energy",
    fields: [qpoint(), bandRange(), solver(), { kind: "int", key: "grid_size", label: "Field grid size", min: 2 }],
    style: [
      { key: "field_quantity", label: "Field quantity", kind: "select", options: [["energy_density", "energy density"], ["E2", "|E|²"], ["H2", "|H|²"]] },
      { key: "component_index", label: "Band column", kind: "number", step: 1, min: 0 },
      ...colourMap, ...colourRange,
    ],
    viewKeys: ["field_quantity", "component_index"],
    render: (view, style) => renderMap(view, style, { columns: ["energy_density", "E2", "H2"], spatial: true }),
    describe: (c) => `bands ${c.composite_bands_one_based?.join(",")}, grid=${c.grid_size}`,
  },
  {
    id: "efs",
    label: "EFS",
    fields: [bandRange(), solver(), { kind: "int", key: "efs_grid_size", label: "EFS grid size", min: 2 }],
    style: [frequencyUnit, ...colourMap, ...colourRange],
    viewKeys: ["frequency_unit"],
    render: (view, style) => renderMap(view, style, { columns: [] }),
    describe: (c) => `bands ${c.composite_bands_one_based?.join(",")}, grid=${c.efs_grid_size}`,
  },
  {
    id: "berry",
    label: "Berry Curvature",
    fields: [
      { kind: "select", key: "berry_target_mode", label: "Berry target", options: [["single_band", "Single band (rank 1)"], ["composite_subspace", "Composite subspace"]] },
      { kind: "select", key: "sampling_mode", label: "Sampling", options: [["single_plaquette", "Single plaquette"], ["first_bz_grid", "First BZ grid"], ["explicit_centers", "Explicit centres"]] },
      qpoint((p) => p.sampling_mode === "single_plaquette"),
      band(isSingleBerry),
      bandRange((p) => !isSingleBerry(p)),
      solver({ numeigReadOnly: true }),
      { kind: "number", key: "berry_step", label: "Plaquette half-step", min: 0 },
      { kind: "int", key: "grid_size", label: "BZ grid size", min: 1, visible: (p) => p.sampling_mode === "first_bz_grid" },
      { kind: "centers", key: "centers", label: "Centres (one “qₓ, qᵧ” per line)", visible: (p) => p.sampling_mode === "explicit_centers" },
      { kind: "note", key: "berry_note", label: "Raw, unsymmetrised output. Rank-1 values are RANK1_WITHHELD when gap, link or branch qualification fails." },
    ],
    style: [
      { key: "berry_render_mode", label: "Rendering", kind: "select", options: [["sample_cells", "Sample-cell tiling (raw)"], ["linear_interpolation", "Linear interpolation (display only)"]] },
      { key: "show_sample_centers", label: "Sample centres", kind: "check" },
      ...colourMap, ...colourRange,
    ],
    render: (view, style) => renderMap(view, style, { columns: ["Berry curvature"], sampleCells: true, qualification: true }),
    describe: (c) => `${isSingleBerry(c) ? `band ${c.band_one_based}` : `bands ${c.berry_first_band}–${c.berry_last_band}`}, gmax=${c.gmax}`,
  },
  {
    id: "berry_curvature_dipole",
    label: "Berry Curvature Dipole",
    fields: [
      { kind: "berrySource", key: "berry_record_path", label: "Berry source result", hint: "Requires a QUALIFIED Berry result of this project." },
      { kind: "list", key: "frequency_window", label: "Frequency window (min, max)", placeholder: "geometric only" },
      { kind: "list", key: "frequency_samples", label: "Frequency samples" },
      { kind: "list", key: "response_weights", label: "Response weights" },
      { kind: "list", key: "occupation", label: "Occupation" },
      { kind: "note", key: "bcd_note", label: "Without a frequency window and weights the dipole is a geometric functional, not a physical response." },
    ],
    style: [...colourMap],
    render: (view, style) => renderMap(view, style, { columns: ["Berry curvature"] }),
    describe: (c) => (c.response_weights || c.occupation ? "physical weighting" : "geometric"),
  },
  {
    id: "valley_chern",
    label: "Valley Chern (half BZ)",
    fields: [
      { kind: "select", key: "valley", label: "Valley", options: [["K'", "K′  (−2/3, 0) — legacy/paper domain"], ["K", "K  (2/3, 0)"]] },
      bandRange(),
      solver({ numeigReadOnly: true }),
      {
        kind: "row", key: "truncation", label: "",
        fields: [
          { kind: "number", key: "delta_k", label: "δK (HBZ shrink)", min: 0 },
          { kind: "number", key: "delta_gamma", label: "δΓ (Γ hexagons)", min: 0 },
        ],
      },
      {
        kind: "row", key: "grid", label: "",
        fields: [
          { kind: "int", key: "grid_denominator", label: "Grid 1/N", min: 1 },
          { kind: "number", key: "gap_floor", label: "Gap floor", min: 0 },
        ],
      },
      { kind: "note", key: "valley_note", label: "Dai et al. truncated HBZ (FR0: δK=δΓ=0.10; FR0.4: δK=0.05, δΓ=0.13). Rank-1 per band; a band is reported only if every sample passes the gap/link/branch gates. Requires a verified C3 model." },
    ],
    style: [
      { key: "component_index", label: "Band column (0 = first band)", kind: "number", step: 1, min: 0 },
      ...colourMap, ...colourRange,
    ],
    viewKeys: ["component_index"],
    render: (view, style) => {
      const rendered = renderMap(view, style, { columns: ["Berry curvature"], sampleCells: true, qualification: true });
      for (const hole of view.valley?.exclusions ?? []) {
        const ring = [...hole, hole[0]];
        rendered.data.push({ type: "scatter", mode: "lines", x: ring.map((p) => p[0]), y: ring.map((p) => p[1]), line: { color: "#555", width: 1, dash: "dash" }, hoverinfo: "skip", showlegend: false } as Plotly.Data);
      }
      return rendered;
    },
    describe: (c) => `${String(c.valley ?? "K'")} bands ${c.berry_first_band}–${c.berry_last_band}, gmax=${c.gmax}`,
  },
];


export const OPERATION_BY_ID: Record<string, OperationDef> = Object.fromEntries(OPERATIONS.map((op) => [op.id, op]));

/** Style controls every result shares, shown after the operation-specific ones. */
export const COMMON_STYLE: StyleSpec[] = [
  { key: "title", label: "Title", kind: "text" },
  { key: "x_label", label: "X label", kind: "text" },
  { key: "y_label", label: "Y label", kind: "text" },
  { key: "x_limits", label: "X min / max", kind: "limits" },
  { key: "y_limits", label: "Y min / max", kind: "limits" },
  { key: "grid", label: "Grid", kind: "check" },
  { key: "legend", label: "Legend", kind: "check" },
];

export const EXPORT_STYLE: StyleSpec[] = [
  { key: "width_px", label: "Width px", kind: "number", step: 10, min: 100 },
  { key: "height_px", label: "Height px", kind: "number", step: 10, min: 100 },
  { key: "dpi", label: "DPI", kind: "number", step: 10, min: 30 },
];
