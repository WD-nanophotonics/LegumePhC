import type {
  Calculation, CalculationParameters, Meta, Model, ModelForm, PlotStyle, PreviewData, Project, RecordViewData, SiteRow, WorkerEvent,
} from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number, public detail: Record<string, unknown> = {}) {
    super(message);
  }
}

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail: Record<string, unknown> = {};
    try {
      detail = await response.json();
    } catch {
      /* non-JSON error body */
    }
    const message = typeof detail.detail === "string" ? detail.detail : `${response.status} ${response.statusText}`;
    throw new ApiError(message, response.status, detail);
  }
  return (await response.json()) as T;
}

const post = <T>(url: string, body: unknown) => call<T>("POST", url, body);

export interface Opened {
  path: string | null;
  project: Project;
}

export const api = {
  meta: () => call<Meta>("GET", "/api/meta"),
  listProjects: () => call<{ path: string; name: string; modified: number }[]>("GET", "/api/projects"),
  newProject: (name = "Untitled") => post<Opened>(`/api/projects/new?name=${encodeURIComponent(name)}`, {}),
  openProject: (path: string) => call<Opened>("GET", `/api/projects/open?path=${encodeURIComponent(path)}`),
  saveProject: (path: string | null, project: Project) => post<Opened>("/api/projects/save", { path, project }),

  addCalculation: (project: Project, operation = "frequency_at_k") =>
    post<{ project: Project; calculation: Calculation }>("/api/calculations/add", { project, operation }),
  copyCalculation: (project: Project, calculation_id: string) =>
    post<{ project: Project; calculation: Calculation }>("/api/calculations/copy", { project, calculation_id }),
  renameCalculation: (project: Project, calculation_id: string, name: string) =>
    post<{ project: Project }>("/api/calculations/rename", { project, calculation_id, name }),
  deleteCalculation: (project: Project, calculation_id: string) =>
    post<{ project: Project }>("/api/calculations/delete", { project, calculation_id }),
  normalizeCalculation: (parameters: CalculationParameters) => post<CalculationParameters>("/api/calculations/normalize", parameters),

  modelForm: (model: Model, ui_state: Record<string, unknown>) => post<ModelForm>("/api/model/form", { model, ui_state }),
  applyModelForm: (model: Model, form: ModelForm, ui_state: Record<string, unknown>) =>
    post<{ model: Model; form: ModelForm }>("/api/model/apply", { model, form, ui_state }),
  siteRows: (model: Model, representation: string) =>
    post<{ rows: SiteRow[]; cell_center: number[] }>("/api/sites/rows", { model, representation }),
  addSite: (model: Model, rows: SiteRow[], representation: string) => post<{ rows: SiteRow[] }>("/api/sites/add", { model, rows, representation }),
  applySites: (model: Model, rows: SiteRow[], representation: string) =>
    post<{ model: Model; rows: SiteRow[] }>("/api/sites/apply", { model, rows, representation }),
  preview: (model: Model, size = 96) => post<Record<string, PreviewData>>("/api/preview", { model, size }),

  startRun: (path: string | null, project: Project) => post<{ run_id: string; path: string }>("/api/runs", { path, project }),
  cancelRun: (runId: string) => call<{ state: string }>("DELETE", `/api/runs/${runId}`),
  runEvents: (runId: string, onEvent: (event: WorkerEvent) => void): EventSource => {
    const source = new EventSource(`/api/runs/${runId}/events`);
    source.onmessage = (message) => onEvent(JSON.parse(message.data) as WorkerEvent);
    return source;
  },

  viewRecord: (path: string, record: string, style: PlotStyle) => post<RecordViewData>("/api/records/view", { path, record, style }),
  exportRecord: async (path: string, record: string, style: PlotStyle, fmt: string) => {
    const response = await fetch("/api/records/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, record, style, fmt }),
    });
    if (!response.ok) throw new ApiError((await response.json()).detail ?? "export failed", response.status);
    return response.blob();
  },

  listPresets: () => call<{ path: string; name: string }[]>("GET", "/api/presets"),
  savePreset: (name: string, model: Model, calculation: CalculationParameters, ui_state: Record<string, unknown>) =>
    post<{ path: string }>("/api/presets/save", { name, model, calculation, ui_state }),
  applyPreset: (project: Project, preset: string) =>
    post<{ project: Project }>(`/api/presets/apply?preset=${encodeURIComponent(preset)}`, { project }),
};
