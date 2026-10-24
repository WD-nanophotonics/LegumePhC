import type {
  Calculation, CalculationParameters, DocumentInfo, Meta, Model, ModelForm, Parameter, PlotStyle, PreviewData, Project, RecordViewData,
  ResultEntry, SiteRow, WorkerEvent,
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

/** An open single-file document: `docId` addresses its server-side workspace. */
export interface Opened {
  doc_id: string;
  path: string | null;
  project: Project;
  converted_from?: string | null;
}

export const api = {
  meta: () => call<Meta>("GET", "/api/meta"),
  listDocuments: () => call<DocumentInfo[]>("GET", "/api/projects"),
  newDocument: (name = "Untitled") => post<Opened>(`/api/documents/new?name=${encodeURIComponent(name)}`, {}),
  openDocument: (path: string) => post<Opened>("/api/documents/open", { path }),
  saveDocument: (docId: string, project: Project, path?: string | null) => post<Opened>(`/api/documents/${docId}/save`, { project, path: path ?? null }),
  closeDocument: (docId: string) => call<{ closed: boolean }>("DELETE", `/api/documents/${docId}`),
  attachSolution: (docId: string, project: Project, entry: Omit<ResultEntry, "id">) =>
    post<{ project: Project }>(`/api/documents/${docId}/attach`, { project, entry }),
  clearSolutions: (docId: string, project: Project, calculationId: string | null = null) =>
    post<{ project: Project }>(`/api/documents/${docId}/clear`, { project, calculation_id: calculationId }),

  addStudy: (project: Project, operation = "frequency_at_k") =>
    post<{ project: Project; calculation: Calculation }>("/api/calculations/add", { project, operation }),
  copyStudy: (project: Project, calculation_id: string) =>
    post<{ project: Project; calculation: Calculation }>("/api/calculations/copy", { project, calculation_id }),
  renameStudy: (project: Project, calculation_id: string, name: string) =>
    post<{ project: Project }>("/api/calculations/rename", { project, calculation_id, name }),
  deleteStudy: (project: Project, calculation_id: string) =>
    post<{ project: Project }>("/api/calculations/delete", { project, calculation_id }),
  normalizeCalculation: (parameters: CalculationParameters) => post<CalculationParameters>("/api/calculations/normalize", parameters),

  applyParameters: (project: Project) => post<{ project: Project; values: Record<string, number> }>("/api/parameters/apply", { project }),
  modelForm: (model: Model, ui_state: Record<string, unknown>) => post<ModelForm>("/api/model/form", { model, ui_state }),
  applyModelForm: (model: Model, form: ModelForm, ui_state: Record<string, unknown>, parameters: Parameter[]) =>
    post<{ model: Model; form: ModelForm }>("/api/model/apply", { model, form, ui_state, parameters }),
  siteRows: (model: Model, representation: string) =>
    post<{ rows: SiteRow[]; cell_center: number[] }>("/api/sites/rows", { model, representation }),
  addSite: (model: Model, rows: SiteRow[], representation: string, parameters: Parameter[]) =>
    post<{ rows: SiteRow[] }>("/api/sites/add", { model, rows, representation, parameters }),
  applySites: (model: Model, rows: SiteRow[], representation: string, parameters: Parameter[]) =>
    post<{ model: Model; rows: SiteRow[] }>("/api/sites/apply", { model, rows, representation, parameters }),
  preview: (model: Model, size = 96) => post<Record<string, PreviewData>>("/api/preview", { model, size }),

  startRun: (docId: string, project: Project) => post<{ run_id: string }>(`/api/documents/${docId}/runs`, { project }),
  cancelRun: (runId: string) => call<{ state: string }>("DELETE", `/api/runs/${runId}`),
  runEvents: (runId: string, onEvent: (event: WorkerEvent) => void): EventSource => {
    const source = new EventSource(`/api/runs/${runId}/events`);
    source.onmessage = (message) => onEvent(JSON.parse(message.data) as WorkerEvent);
    return source;
  },

  viewRecord: (docId: string, record: string, style: PlotStyle) => post<RecordViewData>(`/api/documents/${docId}/records/view`, { record, style }),
  exportRecord: async (docId: string, record: string, style: PlotStyle, fmt: string) => {
    const response = await fetch(`/api/documents/${docId}/records/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record, style, fmt }),
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
