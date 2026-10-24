import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type Opened } from "./api";
import type {
  CalculationParameters, Meta, ModelForm, PlotStyle, PreviewData, Project, RecordViewData, ResultEntry, SiteRow, WorkerEvent,
} from "./types";
import { ProjectTree } from "./components/ProjectTree";
import { GeometrySettings } from "./components/GeometrySettings";
import { CalculationSettings } from "./components/CalculationSettings";
import { PlotStyleSettings } from "./components/PlotStyleSettings";
import { GeometryPreview } from "./components/GeometryPreview";
import { ResultView } from "./components/ResultView";
import { BottomPanel } from "./components/BottomPanel";
import { useDialog } from "./components/Dialog";
import { OPERATION_BY_ID } from "./operations/definitions";

type SettingsTab = "geometry" | "calculation" | "plot";

function message(exc: unknown): string {
  return exc instanceof Error ? exc.message : String(exc);
}

function stamp(): string {
  return new Date().toLocaleTimeString();
}

export function App() {
  const dialogs = useDialog();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [path, setPath] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [modelForm, setModelForm] = useState<ModelForm | null>(null);
  const [rows, setRows] = useState<SiteRow[]>([]);
  const [siteError, setSiteError] = useState<{ index: number; field: string; message: string } | null>(null);
  const [previews, setPreviews] = useState<Record<string, PreviewData> | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [view, setView] = useState<RecordViewData | null>(null);
  const [viewMessage, setViewMessage] = useState<string | null>(null);
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [centerTab, setCenterTab] = useState<"geometry" | "result">("geometry");
  const [settingsTab, setSettingsTab] = useState<SettingsTab>("geometry");
  const [runId, setRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ label: string; fraction: number | null; running: boolean }>({ label: "Idle", fraction: null, running: false });
  const [log, setLog] = useState<string[]>([]);
  const [dialog, setDialog] = useState<null | "open" | "presets">(null);
  const [listing, setListing] = useState<{ path: string; name: string }[]>([]);
  const projectRef = useRef<Project | null>(null);
  const pathRef = useRef<string | null>(null);
  projectRef.current = project;
  pathRef.current = path;

  const addLog = useCallback((line: string) => setLog((old) => [...old.slice(-500), `[${stamp()}] ${line}`]), []);
  const ui = (p: Project | null) => (p?.ui_state ?? {}) as Record<string, unknown>;
  const representation = String(ui(project).material_representation ?? "n");

  const refreshPreview = useCallback(async (model: Project["model"]) => {
    try {
      setPreviews(await api.preview(model));
      setPreviewError(null);
    } catch (exc) {
      setPreviewError(message(exc));
    }
  }, []);

  const selectedResult = useMemo<ResultEntry | null>(() => {
    if (!project) return null;
    return project.results.find((r) => r.record_reference.path === project.selected_result || r.id === project.selected_result) ?? null;
  }, [project]);
  const style = useMemo<PlotStyle>(() => ({ ...(project?.plot ?? {}), ...(selectedResult?.plot ?? {}) }), [project?.plot, selectedResult?.plot]);
  const pins = selectedResult?.display_state?.pins ?? [];

  const loadView = useCallback(async (p: Project, filePath: string | null, entry: ResultEntry | null) => {
    setSelectedRow(null);
    if (!entry || !filePath) {
      setView(null);
      return;
    }
    try {
      const s = { ...p.plot, ...(entry.plot ?? {}) };
      setView(await api.viewRecord(filePath, entry.record_reference.path, s));
      setViewMessage(null);
    } catch (exc) {
      setView(null);
      setViewMessage(`Result unavailable: ${message(exc)}`);
    }
  }, []);

  const loadProject = useCallback(async (opened: Opened) => {
    let p = opened.project;
    const shown = p.results.find((r) => r.record_reference.path === p.selected_result);
    if (shown && p.selected_node.kind === "result" && p.selected_node.id !== shown.id) {
      p = { ...p, selected_node: { kind: "result", id: shown.id } };
    }
    setProject(p);
    setPath(opened.path);
    setDirty(opened.path === null);
    setSiteError(null);
    const state = ui(p);
    const [form, sites] = await Promise.all([api.modelForm(p.model, state), api.siteRows(p.model, String(state.material_representation ?? "n"))]);
    setModelForm(form);
    setRows(sites.rows);
    void refreshPreview(p.model);
    const entry = p.results.find((r) => r.record_reference.path === p.selected_result) ?? null;
    if (entry) {
      setCenterTab("result");
      void loadView(p, opened.path, entry);
    } else {
      setView(null);
      setCenterTab("geometry");
    }
    setStatus(opened.path ? `Opened ${opened.path}` : "New unsaved project");
  }, [loadView, refreshPreview]);

  useEffect(() => {
    (async () => {
      try {
        setMeta(await api.meta());
        const projects = await api.listProjects();
        await loadProject(projects.length ? await api.openProject(projects[0].path) : await api.newProject());
      } catch (exc) {
        setStatus(`Startup failed: ${message(exc)}`);
      }
    })();
  }, [loadProject]);

  const update = useCallback((next: Project, note?: string) => {
    setProject(next);
    setDirty(true);
    if (note) setStatus(note);
  }, []);

  const save = useCallback(async (asName?: string): Promise<boolean> => {
    const p = projectRef.current;
    if (!p) return false;
    let target = pathRef.current;
    if (!target || asName) {
      const name = asName ?? await dialogs.prompt("Save project as", p.name || "Untitled", { label: "Project name", confirm: "Save" });
      if (!name) return false;
      target = name;
      p.name = name.replace(/\.legumephc-studio\.json$/, "");
    }
    try {
      const saved = await api.saveProject(target, p);
      setProject(saved.project);
      setPath(saved.path);
      setDirty(false);
      setStatus(`Saved ${saved.path}`);
      return true;
    } catch (exc) {
      setStatus(`Save failed: ${message(exc)}`);
      return false;
    }
  }, [dialogs]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void save();
      }
    };
    const onUnload = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("beforeunload", onUnload);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("beforeunload", onUnload);
    };
  }, [save, dirty]);

  // ---------------------------------------------------------------- geometry
  const onForm = useCallback(async (form: ModelForm, commit: boolean) => {
    setModelForm(form);
    const p = projectRef.current;
    if (!commit || !p) return;
    try {
      const state = { ...ui(p), length_unit: form.length_unit, material_representation: form.material_representation };
      const applied = await api.applyModelForm(p.model, form, state);
      const next = { ...p, model: applied.model, ui_state: state };
      update(next, "Geometry changed — pending Run");
      setModelForm(applied.form);
      if (form.material_representation !== representation) setRows((await api.siteRows(applied.model, form.material_representation)).rows);
      void refreshPreview(applied.model);
    } catch (exc) {
      setStatus(`Geometry input: ${message(exc)}`);
    }
  }, [refreshPreview, representation, update]);

  const onRows = useCallback(async (next: SiteRow[], commit: boolean) => {
    setRows(next);
    const p = projectRef.current;
    if (!commit || !p) return;
    try {
      const applied = await api.applySites(p.model, next, representation);
      setSiteError(null);
      update({ ...p, model: applied.model }, "Sites updated");
      void refreshPreview(applied.model);
    } catch (exc) {
      const detail = exc instanceof ApiError ? exc.detail : {};
      setSiteError({ index: Number(detail.site_index ?? -1), field: String(detail.field ?? ""), message: message(exc) });
    }
  }, [refreshPreview, representation, update]);

  const onAddSite = useCallback(async () => {
    const p = projectRef.current;
    if (!p) return;
    try {
      const added = await api.addSite(p.model, rows, representation);
      await onRows(added.rows, true);
    } catch (exc) {
      setStatus(message(exc));
    }
  }, [onRows, representation, rows]);

  const onCenterSite = useCallback(async (index: number) => {
    const p = projectRef.current;
    if (!p) return;
    const center = (await api.siteRows(p.model, representation)).cell_center;
    await onRows(rows.map((row, i) => (i === index ? { ...row, center_x: String(center[0]), center_y: String(center[1]) } : row)), true);
  }, [onRows, representation, rows]);

  // ------------------------------------------------------------- calculation
  const selectedCalc = useMemo(() => {
    if (!project) return null;
    const id = project.selected_node.kind === "calculation" ? project.selected_node.id : null;
    return project.calculations.find((c) => c.id === id) ?? project.calculations[0];
  }, [project]);

  const onCalculation = useCallback(async (parameters: CalculationParameters) => {
    const p = projectRef.current;
    if (!p || !selectedCalc) return;
    try {
      const normalized = await api.normalizeCalculation(parameters);
      // A study still carrying its operation's default name follows the operation.
      const defaultName = (op: string) => OPERATION_BY_ID[op]?.label ?? op;
      const calculations = p.calculations.map((c) => (c.id !== selectedCalc.id ? c : {
        ...c,
        name: c.name === defaultName(c.operation) || c.name === "New calculation" ? defaultName(normalized.operation) : c.name,
        operation: normalized.operation,
        parameters: normalized,
      }));
      update({ ...p, calculations }, "Calculation changed");
    } catch (exc) {
      setStatus(`Calculation input: ${message(exc)}`);
    }
  }, [selectedCalc, update]);

  const calcAction = useCallback(async (action: "add" | "copy" | "rename" | "delete") => {
    const p = projectRef.current;
    if (!p || !selectedCalc) return;
    try {
      if (action === "add") update((await api.addCalculation(p)).project, "Calculation added");
      if (action === "copy") update((await api.copyCalculation(p, selectedCalc.id)).project, "Calculation copied");
      if (action === "rename") {
        const name = await dialogs.prompt("Rename study", selectedCalc.name, { label: "Name", confirm: "Rename" });
        if (name?.trim()) update((await api.renameCalculation(p, selectedCalc.id, name.trim())).project, "Calculation renamed");
      }
      if (action === "delete" && await dialogs.confirm(`Delete “${selectedCalc.name}”?`, { body: "Results already computed by it stay in the project.", confirm: "Delete", danger: true })) update((await api.deleteCalculation(p, selectedCalc.id)).project, "Calculation deleted");
      setSettingsTab("calculation");
    } catch (exc) {
      setStatus(message(exc));
    }
  }, [dialogs, selectedCalc, update]);

  const onSelectNode = useCallback((kind: "geometry" | "calculation" | "result", id: string) => {
    const p = projectRef.current;
    if (!p) return;
    const next: Project = { ...p, selected_node: { kind, id } };
    if (kind === "geometry") {
      setSettingsTab("geometry");
      setCenterTab("geometry");
    } else if (kind === "calculation") {
      setSettingsTab("calculation");
    } else {
      const entry = p.results.find((r) => r.id === id) ?? null;
      next.selected_result = entry?.record_reference.path ?? null;
      setSettingsTab("plot");
      setCenterTab("result");
      void loadView(next, pathRef.current, entry);
    }
    setProject(next);
  }, [loadView]);

  // -------------------------------------------------------------------- plot
  const onStyle = useCallback((patch: PlotStyle) => {
    const p = projectRef.current;
    if (!p) return;
    const plot = { ...p.plot, ...patch };
    const results = p.results.map((r) => (r.record_reference.path === p.selected_result ? { ...r, plot: { ...(r.plot ?? {}), ...patch } } : r));
    const next = { ...p, plot, results };
    update(next);
    const entry = results.find((r) => r.record_reference.path === p.selected_result) ?? null;
    const viewKeys = OPERATION_BY_ID[view?.operation ?? ""]?.viewKeys ?? [];
    if (Object.keys(patch).some((key) => viewKeys.includes(key))) void loadView(next, pathRef.current, entry);
  }, [loadView, update, view?.operation]);

  const onSelectRow = useCallback((index: number, pin: boolean) => {
    setSelectedRow(index);
    if (!pin) return;
    const p = projectRef.current;
    if (!p) return;
    const results = p.results.map((r) => {
      if (r.record_reference.path !== p.selected_result) return r;
      const current = r.display_state?.pins ?? [];
      const nextPins = current.includes(index) ? current.filter((i) => i !== index) : [...current, index];
      return { ...r, display_state: { ...(r.display_state ?? {}), pins: nextPins } };
    });
    update({ ...p, results });
  }, [update]);

  const onExport = useCallback(async (fmt: string) => {
    const p = projectRef.current;
    if (!p || !pathRef.current || !selectedResult) return;
    try {
      const blob = await api.exportRecord(pathRef.current, selectedResult.record_reference.path, style, fmt);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${selectedResult.record_reference.path.split("/").pop()}.${fmt}`;
      link.click();
      URL.revokeObjectURL(url);
      setStatus(`Exported ${fmt.toUpperCase()} (also stored in the record's figures/ folder)`);
    } catch (exc) {
      setStatus(`Export failed: ${message(exc)}`);
    }
  }, [selectedResult, style]);

  // --------------------------------------------------------------------- run
  const run = useCallback(async () => {
    const p = projectRef.current;
    if (!p || runId) return;
    let target = pathRef.current;
    if (!target) {
      const name = await dialogs.prompt("Save the project before running", p.name || "Untitled", { label: "Project name", confirm: "Save and run" });
      if (!name) return;
      target = name;
      p.name = name;
    }
    try {
      const started = await api.startRun(target, p);
      setPath(started.path);
      pathRef.current = started.path;
      setDirty(false);
      setRunId(started.run_id);
      setProgress({ label: "Starting worker…", fraction: null, running: true });
      addLog(`Run ${started.run_id} started (${selectedCalc?.operation})`);
      const source = api.runEvents(started.run_id, (event: WorkerEvent) => {
        const elapsed = event.elapsed_s != null ? ` · ${event.elapsed_s.toFixed(1)} s` : "";
        if (event.completed != null && event.total) {
          setProgress({ label: `${event.phase ?? event.event}: ${event.completed}/${event.total}${elapsed}`, fraction: event.completed / event.total, running: true });
        } else if (event.event !== "finished") {
          setProgress((old) => ({ ...old, label: `${event.message ?? event.error ?? event.event}${elapsed}` }));
        }
        if (["phase", "record_written", "failed", "cancelled"].includes(event.event)) addLog(event.message ?? event.error ?? event.event);
        if (event.event === "completed" && event.result_entry && !event.result_entry.error) {
          const current = projectRef.current!;
          const nextIndex = current.results.reduce((max, r) => Math.max(max, Number(r.id.split("-").pop()) || 0), 0) + 1;
          const entry: ResultEntry = { ...event.result_entry, id: `result-${nextIndex}`, plot: { ...current.plot } } as ResultEntry;
          const next: Project = {
            ...current,
            results: [...current.results, entry],
            selected_result: entry.record_reference.path,
            selected_node: { kind: "result", id: entry.id },
          };
          setProject(next);
          projectRef.current = next;
          void api.saveProject(pathRef.current, next).then((saved) => { setProject(saved.project); setDirty(false); });
          setCenterTab("result");
          setSettingsTab("plot");
          void loadView(next, pathRef.current, entry);
          addLog(`Record ${entry.record_reference.path}`);
        }
        if (event.event === "finished" || event.event === "cancelled") {
          source.close();
          setRunId(null);
          const ok = event.state === "succeeded";
          setProgress({ label: event.event === "cancelled" ? "Cancelled; exact worker terminated" : ok ? `Completed${elapsed}` : `Failed${elapsed}`, fraction: ok ? 1 : 0, running: false });
          setStatus(ok ? "Calculation completed" : "Calculation did not complete — see log");
        }
      });
      source.onerror = () => addLog("event stream interrupted");
    } catch (exc) {
      setStatus(`Run failed: ${message(exc)}`);
      addLog(`Run failed: ${message(exc)}`);
    }
  }, [addLog, dialogs, loadView, runId, selectedCalc?.operation]);

  const cancel = useCallback(async () => {
    if (runId) await api.cancelRun(runId);
  }, [runId]);

  // ---------------------------------------------------------------- dialogs
  const openDialog = useCallback(async (kind: "open" | "presets") => {
    setListing(kind === "open" ? await api.listProjects() : await api.listPresets());
    setDialog(kind);
  }, []);

  const pick = useCallback(async (item: string) => {
    setDialog(null);
    const p = projectRef.current;
    try {
      if (dialog === "open") {
        if (dirty && !await dialogs.confirm("Discard unsaved changes?", { confirm: "Discard", danger: true })) return;
        await loadProject(await api.openProject(item));
      } else if (p) {
        const applied = await api.applyPreset(p, item);
        await loadProject({ path: pathRef.current, project: applied.project });
        setDirty(true);
      }
    } catch (exc) {
      setStatus(message(exc));
    }
  }, [dialog, dialogs, dirty, loadProject]);

  const savePreset = useCallback(async () => {
    const p = projectRef.current;
    if (!p || !selectedCalc) return;
    const name = await dialogs.prompt("Save parameter preset", "Untitled preset", { label: "Preset name", confirm: "Save" });
    if (!name) return;
    try {
      const saved = await api.savePreset(name, p.model, selectedCalc.parameters, ui(p));
      setStatus(`Preset saved: ${saved.path}`);
    } catch (exc) {
      setStatus(message(exc));
    }
  }, [dialogs, selectedCalc]);

  if (!project) return <div className="loading">{status}</div>;
  const berrySources = project.results.filter((r) => r.calculation_snapshot?.operation === "berry");

  return (
    <div className="app">
      <header className="toolbar">
        <div className="brand">LegumePhC <span>Studio</span></div>
        <button onClick={async () => { if (!dirty || await dialogs.confirm("Discard unsaved changes?", { confirm: "Discard", danger: true })) await loadProject(await api.newProject()); }}>New</button>
        <button onClick={() => openDialog("open")}>Open…</button>
        <button onClick={() => save()}>Save</button>
        <button onClick={async () => { const name = await dialogs.prompt("Save project as", project.name, { label: "Project name", confirm: "Save" }); if (name) await save(name); }}>Save as…</button>
        <span className="sep" />
        <button onClick={() => openDialog("presets")}>Apply preset…</button>
        <button onClick={savePreset}>Save preset…</button>
        <span className="sep" />
        <button className="primary" disabled={!!runId} onClick={run}>▶ Run</button>
        <button disabled={!runId} onClick={cancel}>■ Cancel</button>
        <div className="title">{dirty ? "● " : ""}{project.name}<span className="muted">{path ? ` — ${path}` : " — unsaved"}</span></div>
      </header>

      <aside className="left">
        <ProjectTree meta={meta} project={project} onSelect={onSelectNode} />
      </aside>

      <main className="center">
        <div className="tabs">
          <button className={centerTab === "geometry" ? "active" : ""} onClick={() => setCenterTab("geometry")}>Geometry</button>
          <button className={centerTab === "result" ? "active" : ""} onClick={() => setCenterTab("result")}>Result</button>
        </div>
        <div className="center-body">
          {centerTab === "geometry"
            ? <GeometryPreview previews={previews} error={previewError} />
            : <ResultView view={view} style={style} selected={selectedRow} pins={pins} onSelect={onSelectRow} message={viewMessage} />}
        </div>
      </main>

      <aside className="right">
        <div className="tabs">
          <button className={settingsTab === "geometry" ? "active" : ""} onClick={() => setSettingsTab("geometry")}>Geometry</button>
          <button className={settingsTab === "calculation" ? "active" : ""} onClick={() => setSettingsTab("calculation")}>Calculation</button>
          <button className={settingsTab === "plot" ? "active" : ""} onClick={() => setSettingsTab("plot")}>Plot style</button>
        </div>
        <div className="right-body">
          {settingsTab === "geometry" && (
            <GeometrySettings meta={meta} form={modelForm} onForm={onForm} rows={rows} siteError={siteError} onRows={onRows}
              onAddSite={onAddSite} onCenterSite={onCenterSite} advanced={Boolean(ui(project).advanced_expanded)}
              onAdvanced={(open) => setProject({ ...project, ui_state: { ...ui(project), advanced_expanded: open } })} />
          )}
          {settingsTab === "calculation" && (
            <CalculationSettings name={selectedCalc?.name ?? ""} parameters={selectedCalc?.parameters ?? null} onChange={onCalculation} berrySources={berrySources}
              running={!!runId} onRun={run} onCancel={cancel} onAdd={() => calcAction("add")} onCopy={() => calcAction("copy")}
              onRename={() => calcAction("rename")} onDelete={() => calcAction("delete")} />
          )}
          {settingsTab === "plot" && (
            <PlotStyleSettings style={style} operation={view?.operation ?? selectedCalc?.operation ?? null} onChange={onStyle} onExport={onExport} canExport={!!view && !!path} />
          )}
        </div>
      </aside>

      <BottomPanel view={view} result={selectedResult} selected={selectedRow} pins={pins} onSelect={onSelectRow} progress={progress} log={log} />

      <footer className="status">{status}</footer>

      {dialog && (
        <div className="modal-backdrop" onClick={() => setDialog(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{dialog === "open" ? "Open project" : "Apply parameter preset"}</h3>
            {listing.length === 0 && <div className="empty">Nothing found{dialog === "open" && meta ? ` in ${meta.projects_dir}` : ""}.</div>}
            <ul>
              {listing.map((item) => <li key={item.path}><button onClick={() => pick(item.path)}>{item.name}<span className="muted"> {item.path}</span></button></li>)}
            </ul>
            <div className="button-row"><button onClick={() => setDialog(null)}>Close</button></div>
          </div>
        </div>
      )}
    </div>
  );
}
