// Hand-written mirrors of the Studio project schema (legumephc.studio.project)
// and the FastAPI payloads in legumephc.server.app.

export type Operation =
  | "frequency_at_k"
  | "band_structure"
  | "fields_energy"
  | "efs"
  | "berry"
  | "berry_curvature_dipole";

export interface Motif {
  name?: string;
  kind: "circle" | "polygon";
  radius: number;
  sides: number;
  angle_degrees: number;
  center: [number, number];
  epsilon: number;
}

export interface Model {
  name?: string;
  lattice: "triangular" | "square" | "custom";
  lattice_constant: number;
  actual_lattice_constant_m: number | null;
  direct_basis: number[][];
  geometry: Record<string, unknown> & { name?: string; motifs?: Motif[]; epsilon_background: number };
  affine: { linear: number[][]; translation: number[] };
  deformation?: Record<string, unknown>;
  basis_policy: string;
}

export type CalculationParameters = Record<string, unknown> & {
  operation: Operation;
  qpoint: [number, number];
  band_one_based: number;
  composite_bands_one_based: number[];
  berry_target_mode: string;
  berry_first_band: number;
  berry_last_band: number;
  gmax: number;
  numeig: number;
  polarization: string;
  samples_per_segment: number;
  grid_size: number;
  efs_grid_size: number;
  berry_step: number;
  sampling_mode: string;
  centers: number[][];
  frequency_window: number[] | null;
  frequency_samples: number[] | null;
  response_weights: number[] | null;
  occupation: number[] | null;
  berry_record_path: string | null;
};

export interface Calculation {
  id: string;
  name: string;
  operation: Operation;
  parameters: CalculationParameters;
}

export interface RecordReference {
  path: string;
  files: Record<string, { size: number; sha256: string }>;
  identity: Record<string, unknown>;
  created_at?: string;
}

export type PlotStyle = Record<string, unknown>;

export interface ResultEntry {
  id: string;
  calculation_id: string;
  record_reference: RecordReference;
  model_snapshot: Model;
  calculation_snapshot: CalculationParameters;
  plot?: PlotStyle;
  display_state?: { pins: number[] };
}

export interface Project {
  schema: string;
  name: string;
  model: Model;
  calculations: Calculation[];
  results: ResultEntry[];
  selected_node: { kind: "geometry" | "calculation" | "result"; id: string };
  selected_result: string | null;
  plot: PlotStyle;
  ui_state?: Record<string, unknown>;
}

export interface ModelForm {
  lattice: string;
  lattice_summary: string;
  actual_a: string;
  length_unit: string;
  material_representation: string;
  background: string;
  basis_policy: string;
  deformation: string;
  factor: string;
  angle: string;
  basis: string[];
  affine: string[];
  tx: string;
  ty: string;
  geometry_scale: string;
}

export interface SiteRow {
  name: string;
  shape: string;
  radius: string;
  rotation: string;
  center_x: string;
  center_y: string;
  material: string;
  sides: string;
}

export interface Meta {
  operations: Record<Operation, string>;
  lattice_summary: Record<string, string>;
  preview_views: string[];
  frequency_units: string[];
  length_units: string[];
  shapes: string[];
  projects_dir: string;
  project_suffix: string;
}

export interface InspectionRow {
  x: number;
  y: number;
  values: Record<string, unknown>;
}

export interface RecordViewData {
  operation: Operation;
  summary: Record<string, unknown>;
  config: Record<string, unknown>;
  columns: string[];
  rows: InspectionRow[];
  arrays: Record<string, unknown>;
  frequency_factor?: number;
  sample_cells?: number[][][];
}

export interface PreviewData {
  view: string;
  x_grid?: number[][];
  y_grid?: number[][];
  epsilon?: number[][] | null;
  points?: number[][];
  direct_basis?: number[][];
  reciprocal_basis?: number[][];
  bz_vertices?: number[][];
  point_group?: string | null;
  motifs?: { kind: string; center: number[]; radius?: number; width?: number; height?: number; angle_degrees?: number; vertices?: number[][] }[];
}

export interface WorkerEvent {
  event: string;
  phase?: string;
  message?: string;
  error?: string;
  completed?: number;
  total?: number;
  elapsed_s?: number;
  state?: string;
  record_path?: string;
  result_entry?: Omit<ResultEntry, "id"> & { id: null; error?: string };
}
