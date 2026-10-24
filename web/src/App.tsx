import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type Opened } from "./api";
import type {
  CalculationParameters, DocumentInfo, Meta, ModelForm, NodeKind, Parameter, PlotGroup, PlotStyle, PreviewData, Project, RecordViewData, ResultEntry,
  SiteRow, WorkerEvent,
} from "./types";
import { ModelBuilder } from "./components/ModelBuilder";
import { ComponentSettings, type ComponentSection } from "./components/ComponentSettings";
import { ParametersSettings } from "./components/ParametersSettings";
import { CalculationSettings } from "./components/CalculationSettings";
import { PlotStyleSettings } from "./components/PlotStyleSettings";
import { GeometryPreview } from "./components/GeometryPreview";
import { ResultView } from "./components/ResultView";
import { BottomPanel } from "./components/BottomPanel";
import { DatasetSettings } from "./components/DatasetSettings";
import { useDialog } from "./components/Dialog";
import { OPERATION_BY_ID } from "./operations/definitions";

type FieldError = { index: number; field: string; message: string } | null;
const NODE_KINDS = new Set(["parameters", "component", "study", "dataset", "plot"]);
const DEFAULT_NODE = { kind: "component", id: "sites" } as const;

function message(exc: unknown): string {
  return exc instanceof Error ? exc.message : String(exc);
}

function fieldError(exc: unknown): FieldError {
  const detail = exc instanceof ApiError ? exc.detail : {};
  return { index: Number(detail.index ?? detail.site_index ?? -1), field: String(detail.field ?? ""), message: message(exc) };
}

function formatBytes(size: number): string {
  return size > 1 << 20 ? `${(size / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(size / 1024))} KB`;
}

export function App() {
  const dialogs = useDialog();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [docId, setDocId] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [path, setPath] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [status, setStatus] = useState("Starting…");
  const [modelForm, setModelForm] = useState<ModelForm | null>(null);
  const [rows, setRows] = useState<SiteRow[]>([]);
  const [siteError, setSiteError] = useState<FieldError>(null);
  const [paramError, setParamError] = useState<FieldError>(null);
  const [paramValues, setParamValues] = useState<Record<string, number>>({});
  const [previews, setPreviews] = useState<Record<string, PreviewData> | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [view, setView] = useState<RecordViewData | null>(null);
  const [viewMessage, setViewMessage] = useState<string | null>(null);
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [running, setRunning] = useState<{ runId: string; studyId: string } | null>(null);
  const [progress, setProgress] = useState<{ label: string; fraction: number | null; running: boolean }>({ label: "Idle", fraction: null, running: false });
  const [log, setLog] = useState<string[]>([]);
  const [dialog, setDialog] = useState<null | "open" | "presets">(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [presets, setPresets] = useState<{ path: string; name: string }[]>([]);
  const projectRef = useRef<Project | null>(null);
  const docRef = useRef<string | null>(null);
  projectRef.current = project;
  docRef.current = docId;

  const addLog = useCallback((line: string) => setLog((old) => [...old.slice(-500), `[${new Date().toLocaleTimeString()}] ${line}`]), []);
  const ui = (p: Project | null) => (p?.ui_state ?? {}) as Record<string, unknown>;
  const representation = String(ui(project).material_representation ?? "n");

  // ------------------------------------------------------------ selection
  const node = project && NODE_KINDS.has(project.selected_node.kind) ? project.selected_node : DEFAULT_NODE;
  const group: PlotGroup | null = project && node.kind === "plot" ? project.plot_groups.find((g) => g.id === node.id) ?? null : null;
  const studyId = node.kind === "study" || node.kind === "dataset" ? node.id : group?.calculation_id ?? null;
  const study = project?.calculations.find((c) => c.id === studyId) ?? null;
  const solution: ResultEntry | null = project && studyId ? project.results.find((r) => r.calculation_id === studyId) ?? null : null;
  const showsResult = node.kind === "plot" || node.kind === "dataset";
  const style = useMemo<PlotStyle>(() => ({ ...(project?.plot ?? {}), ...(group?.style ?? {}) }), [project?.plot, group?.style]);
  const pins = group?.pins ?? [];

  // ---------------------------------------------------------------- views
  const refreshPreview = useCallback(async (model: Project["model"]) => {
    try {
      setPreviews(await api.preview(model));
      setPreviewError(null);
    } catch (exc) {
      setPreviewError(message(exc));
    }
  }, []);

  const op = solution ? OPERATION_BY_ID[solution.calculation_snapshot.operation] : undefined;
  const viewKey = JSON.stringify((op?.viewKeys ?? []).map((key) => style[key]));
  const record = showsResult ? solution?.record_reference.path ?? null : null;
  useEffect(() => {
    setSelectedRow(null);
    if (!record || !docId) {
      setView(null);
      setViewMessage(showsResult && !solution ? "This study has no solution yet — select the study and press Compute." : null);
      return;
    }
    let cancelled = false;
    api.viewRecord(docId, record, style).then(
      (data) => { if (!cancelled) { setView(data); setViewMessage(null); } },
      (exc) => { if (!cancelled) { setView(null); setViewMessage(`Result unavailable: ${message(exc)}`); } },
    );
    return () => { cancelled = true; };
  }, [docId, record, viewKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // ------------------------------------------------------------- documents
  const loadDocument = useCallback(async (opened: Opened) => {
    if (docRef.current && docRef.current !== opened.doc_id) void api.closeDocument(docRef.current);
    let p = opened.project;
    if (!NODE_KINDS.has(p.selected_node.kind)) p = { ...p, selected_node: { ...DEFAULT_NODE } };
    setDocId(opened.doc_id);
    setProject(p);
    setPath(opened.path);
    setDirty(false);
    setSiteError(null);
    setParamError(null);
    const state = ui(p);
    const [form, sites, evaluated] = await Promise.all([
      api.modelForm(p.model, state),
      api.siteRows(p.model, String(state.material_representation ?? "n")),
      api.applyParameters(p).catch(() => ({ values: {} as Record<string, number> })),
    ]);
    setModelForm(form);
    setRows(sites.rows);
    setParamValues(evaluated.values);
    void refreshPreview(p.model);
    if (opened.converted_from) {
      addLog(`Converted ${opened.converted_from} → ${opened.path} (original left untouched)`);
      setStatus(`Converted legacy project into ${opened.path}`);
    } else {
      setStatus(opened.path ? `Opened ${opened.path}` : "New untitled document");
    }
  }, [addLog, refreshPreview]);

  useEffect(() => {
    (async () => {
      try {
        setMeta(await api.meta());
        await loadDocument(await api.newDocument());
      } catch (exc) {
        setStatus(`Startup failed: ${message(exc)}`);
      }
    })();
  }, [loadDocument]);

  const update = useCallback((next: Project, note?: string) => {
    setProject(next);
    projectRef.current = next;
    setDirty(true);
    if (note) setStatus(note);
  }, []);

  const save = useCallback(async (saveAs = false): Promise<boolean> => {
    const p = projectRef.current;
    const id = docRef.current;
    if (!p || !id) return false;
    let target: string | null = null;
    if (saveAs || !path) {
      const name = await dialogs.prompt(saveAs ? "Save document as" : "Save document", p.name || "Untitled", { label: `File name (${meta?.document_suffix ?? ".lphc"})`, confirm: "Save" });
      if (!name) return false;
      target = name;
      p.name = name.replace(/\.lphc$/i, "").split("/").pop() || p.name;
    }
    try {
      const saved = await api.saveDocument(id, p, target);
      setProject(saved.project);
      setPath(saved.path);
      setDirty(false);
      setStatus(`Saved ${saved.path}`);
      return true;
    } catch (exc) {
      setStatus(`Save failed: ${message(exc)}`);
      return false;
    }
  }, [dialogs, meta?.document_suffix, path]);

  const confirmDiscard = useCallback(async () => !dirty || dialogs.confirm("Discard unsaved changes?", { body: "Unsaved geometry edits and computed solutions will be lost.", confirm: "Discard", danger: true }), [dialogs, dirty]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void save(event.shiftKey);
      }
    };
    const onUnload = (event: BeforeUnloadEvent) => { if (dirty) event.preventDefault(); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("beforeunload", onUnload);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("beforeunload", onUnload);
    };
  }, [save, dirty]);

  // ------------------------------------------------- global definitions
  const onParameters = useCallback(async (parameters: Parameter[]) => {
    const p = projectRef.current;
    if (!p) return;
    try {
      const applied = await api.applyParameters({ ...p, parameters });
      setParamError(null);
      setParamValues(applied.values);
      update(applied.project, "Parameters applied — component re-evaluated");
      const state = ui(applied.project);
      setModelForm(await api.modelForm(applied.project.model, state));
      setRows((await api.siteRows(applied.project.model, String(state.material_representation ?? "n"))).rows);
      void refreshPreview(applied.project.model);
    } catch (exc) {
      setParamError(fieldError(exc));
      setProject({ ...p, parameters });
    }
  }, [refreshPreview, update]);

  // ------------------------------------------------------------ component
  const onForm = useCallback(async (form: ModelForm, commit: boolean) => {
    setModelForm(form);
    const p = projectRef.current;
    if (!commit || !p) return;
    try {
      const state = { ...ui(p), length_unit: form.length_unit, material_representation: form.material_representation };
      const applied = await api.applyModelForm(p.model, form, state, p.parameters);
      update({ ...p, model: applied.model, ui_state: state }, "Component changed — recompute studies to update results");
      setModelForm(applied.form);
      if (form.material_representation !== representation) setRows((await api.siteRows(applied.model, form.material_representation)).rows);
      void refreshPreview(applied.model);
    } catch (exc) {
      setStatus(`Input: ${message(exc)}`);
    }
  }, [refreshPreview, representation, update]);

  const onRows = useCallback(async (next: SiteRow[], commit: boolean) => {
    setRows(next);
    const p = projectRef.current;
    if (!commit || !p) return;
    try {
      const applied = await api.applySites(p.model, next, representation, p.parameters);
      setSiteError(null);
      update({ ...p, model: applied.model }, "Sites updated");
      void refreshPreview(applied.model);
    } catch (exc) {
      setSiteError(fieldError(exc));
    }
  }, [refreshPreview, representation, update]);

  const onAddSite = useCallback(async () => {
    const p = projectRef.current;
    if (!p) return;
    try {
      await onRows((await api.addSite(p.model, rows, representation, p.parameters)).rows, true);
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

  // --------------------------------------------------------------- studies
  const onCalculation = useCallback(async (parameters: CalculationParameters) => {
    const p = projectRef.current;
    if (!p || !study) return;
    try {
      const normalized = await api.normalizeCalculation(parameters);
      // A study still carrying its operation's default name follows the operation.
      const defaultName = (operation: string) => OPERATION_BY_ID[operation]?.label ?? operation;
      const calculations = p.calculations.map((c) => (c.id !== study.id ? c : {
        ...c,
        name: c.name === defaultName(c.operation) || c.name === "New calculation" ? defaultName(normalized.operation) : c.name,
        operation: normalized.operation,
        parameters: normalized,
      }));
      update({ ...p, calculations }, "Study changed");
    } catch (exc) {
      setStatus(`Study input: ${message(exc)}`);
    }
  }, [study, update]);

  const select = useCallback((kind: NodeKind, id: string) => {
    const p = projectRef.current;
    if (p) setProject({ ...p, selected_node: { kind, id } });
  }, []);

  const studyAction = useCallback(async (action: "add" | "copy" | "rename" | "delete" | "clear") => {
    const p = projectRef.current;
    const id = docRef.current;
    if (!p || !id) return;
    try {
      if (action === "add") update((await api.addStudy(p)).project, "Study added");
      if (!study) return;
      if (action === "copy") update((await api.copyStudy(p, study.id)).project, "Study duplicated");
      if (action === "rename") {
        const name = await dialogs.prompt("Rename study", study.name, { label: "Name", confirm: "Rename" });
        if (name) update((await api.renameStudy(p, study.id, name)).project, "Study renamed");
      }
      if (action === "delete" && await dialogs.confirm(`Delete study “${study.name}”?`, { body: "Its solution and plot groups are removed too.", confirm: "Delete", danger: true })) {
        update((await api.deleteStudy(p, study.id)).project, "Study deleted");
      }
      if (action === "clear") update((await api.clearSolutions(id, p, study.id)).project, "Solution cleared");
    } catch (exc) {
      setStatus(message(exc));
    }
  }, [dialogs, study, update]);

  const clearAll = useCallback(async () => {
    const p = projectRef.current;
    const id = docRef.current;
    if (!p || !id || !p.results.length) return;
    if (!await dialogs.confirm("Clear all solutions?", { body: "Studies, geometry and plot settings are kept; the file becomes small again.", confirm: "Clear", danger: true })) return;
    update((await api.clearSolutions(id, p)).project, "All solutions cleared");
  }, [dialogs, update]);

  const compute = useCallback(async () => {
    const p = projectRef.current;
    const id = docRef.current;
    const target = study ?? p?.calculations[0];
    if (!p || !id || !target || running) return;
    try {
      const run = { ...p, selected_node: { kind: "study", id: target.id } };
      const started = await api.startRun(id, run);
      setRunning({ runId: started.run_id, studyId: target.id });
      setProgress({ label: "Starting worker…", fraction: null, running: true });
      addLog(`Computing “${target.name}” (${OPERATION_BY_ID[target.operation]?.label ?? target.operation})`);
      const source = api.runEvents(started.run_id, (event: WorkerEvent) => {
        const elapsed = event.elapsed_s != null ? ` · ${event.elapsed_s.toFixed(1)} s` : "";
        if (event.completed != null && event.total) {
          setProgress({ label: `${event.phase ?? event.event}: ${event.completed}/${event.total}${elapsed}`, fraction: event.completed / event.total, running: true });
        } else if (event.event !== "finished") {
          setProgress((old) => ({ ...old, label: `${event.message ?? event.error ?? event.event}${elapsed}` }));
        }
        if (["phase", "record_written", "failed", "cancelled"].includes(event.event)) addLog(event.message ?? event.error ?? event.event);
        if (event.event === "completed" && event.result_entry && !event.result_entry.error) {
          const { id: _ignored, ...entry } = event.result_entry;
          void api.attachSolution(id, projectRef.current!, entry).then((attached) => {
            update(attached.project, `“${target.name}” solved — save to keep it in ${path ?? "the document"}`);
          });
        }
        if (event.event === "finished" || event.event === "cancelled") {
          source.close();
          setRunning(null);
          const ok = event.state === "succeeded";
          setProgress({ label: event.event === "cancelled" ? "Stopped; worker terminated" : ok ? `Completed${elapsed}` : `Failed${elapsed}`, fraction: ok ? 1 : 0, running: false });
          if (!ok) setStatus("Computation did not complete — see Messages");
        }
      });
      source.onerror = () => addLog("event stream interrupted");
    } catch (exc) {
      setStatus(`Compute failed: ${message(exc)}`);
      addLog(`Compute failed: ${message(exc)}`);
    }
  }, [addLog, path, running, study, update]);

  const stop = useCallback(async () => {
    if (running) await api.cancelRun(running.runId);
  }, [running]);

  // ---------------------------------------------------------- plot groups
  const updateGroup = useCallback((patch: Partial<PlotGroup>) => {
    const p = projectRef.current;
    if (!p || !group) return;
    update({ ...p, plot_groups: p.plot_groups.map((g) => (g.id === group.id ? { ...g, ...patch } : g)) });
  }, [group, update]);

  const onStyle = useCallback((patch: PlotStyle) => {
    if (group) updateGroup({ style: { ...group.style, ...patch } });
  }, [group, updateGroup]);

  const addPlotGroup = useCallback((calculationId: string) => {
    const p = projectRef.current;
    if (!p) return;
    const used = new Set(p.plot_groups.map((g) => g.id));
    let index = 1;
    while (used.has(`pg-${index}`)) index += 1;
    const owner = p.calculations.find((c) => c.id === calculationId);
    const count = p.plot_groups.filter((g) => g.calculation_id === calculationId).length + 1;
    const created: PlotGroup = { id: `pg-${index}`, name: `${owner?.name ?? "Study"} plot ${count}`, calculation_id: calculationId, style: {}, pins: [] };
    update({ ...p, plot_groups: [...p.plot_groups, created], selected_node: { kind: "plot", id: created.id } }, "Plot group added");
  }, [update]);

  const deleteGroup = useCallback(async () => {
    const p = projectRef.current;
    if (!p || !group || !await dialogs.confirm(`Delete plot group “${group.name}”?`, { confirm: "Delete", danger: true })) return;
    update({ ...p, plot_groups: p.plot_groups.filter((g) => g.id !== group.id), selected_node: { kind: "dataset", id: group.calculation_id } }, "Plot group deleted");
  }, [dialogs, group, update]);

  const onSelectRow = useCallback((index: number, pin: boolean) => {
    setSelectedRow(index);
    if (pin && group) updateGroup({ pins: pins.includes(index) ? pins.filter((i) => i !== index) : [...pins, index] });
  }, [group, pins, updateGroup]);

  const onExport = useCallback(async (fmt: string) => {
    if (!docId || !solution) return;
    try {
      const blob = await api.exportRecord(docId, solution.record_reference.path, style, fmt);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${(group?.name ?? "plot").replace(/[^\w.-]+/g, "_")}.${fmt}`;
      link.click();
      URL.revokeObjectURL(url);
      setStatus(`Exported ${fmt.toUpperCase()}`);
    } catch (exc) {
      setStatus(`Export failed: ${message(exc)}`);
    }
  }, [docId, group?.name, solution, style]);

  // --------------------------------------------------------------- dialogs
  const openDialog = useCallback(async (kind: "open" | "presets") => {
    if (kind === "open") setDocuments(await api.listDocuments());
    else setPresets(await api.listPresets());
    setDialog(kind);
  }, []);

  const openDocument = useCallback(async (item: string) => {
    setDialog(null);
    if (!await confirmDiscard()) return;
    try {
      setStatus(`Opening ${item}…`);
      await loadDocument(await api.openDocument(item));
    } catch (exc) {
      setStatus(`Open failed: ${message(exc)}`);
    }
  }, [confirmDiscard, loadDocument]);

  const applyPreset = useCallback(async (item: string) => {
    setDialog(null);
    const p = projectRef.current;
    if (!p) return;
    try {
      const applied = (await api.applyPreset(p, item)).project;
      update(applied, `Preset applied: ${item}`);
      const state = ui(applied);
      setModelForm(await api.modelForm(applied.model, state));
      setRows((await api.siteRows(applied.model, String(state.material_representation ?? "n"))).rows);
      void refreshPreview(applied.model);
    } catch (exc) {
      setStatus(message(exc));
    }
  }, [refreshPreview, update]);

  const savePreset = useCallback(async () => {
    const p = projectRef.current;
    const source = study ?? p?.calculations[0];
    if (!p || !source) return;
    const name = await dialogs.prompt("Save parameter preset", "Untitled preset", { label: "Preset name (model + selected study, no results)", confirm: "Save" });
    if (!name) return;
    try {
      setStatus(`Preset saved: ${(await api.savePreset(name, p.model, source.parameters, ui(p))).path}`);
    } catch (exc) {
      setStatus(message(exc));
    }
  }, [dialogs, study]);

  if (!project) return <div className="loading">{status}</div>;
  const berrySources = project.results.filter((r) => r.calculation_snapshot?.operation === "berry");
  const solvedStudies = project.calculations.filter((c) => project.results.some((r) => r.calculation_id === c.id));

  let settings: JSX.Element;
  if (node.kind === "parameters") {
    settings = <ParametersSettings parameters={project.parameters} values={paramValues} error={paramError} onCommit={onParameters} />;
  } else if (node.kind === "component") {
    settings = <ComponentSettings section={node.id as ComponentSection} meta={meta} form={modelForm} onForm={onForm} rows={rows} siteError={siteError}
      onRows={onRows} onAddSite={onAddSite} onCenterSite={onCenterSite} pointGroup={previews?.unit_cell?.point_group} />;
  } else if (node.kind === "study") {
    settings = <CalculationSettings name={study?.name ?? ""} parameters={study?.parameters ?? null} solution={solution} onChange={onCalculation}
      berrySources={berrySources} running={!!running} thisRunning={running?.studyId === study?.id} onRun={compute} onCancel={stop}
      onClear={() => studyAction("clear")} onCopy={() => studyAction("copy")} onRename={() => studyAction("rename")} onDelete={() => studyAction("delete")} />;
  } else if (node.kind === "dataset") {
    settings = <DatasetSettings study={study} solution={solution} view={view} onAddPlot={() => study && addPlotGroup(study.id)} />;
  } else if (group) {
    settings = <PlotStyleSettings group={group} studies={solvedStudies.length ? solvedStudies : project.calculations} style={style}
      operation={view?.operation ?? study?.operation ?? null} onChange={onStyle} onGroup={updateGroup} onDelete={deleteGroup} onExport={onExport} canExport={!!view} />;
  } else {
    settings = <div className="empty">Select a node in the Model Builder.</div>;
  }

  const graphicsTitle = showsResult ? `${group?.name ?? `${study?.name ?? ""} / Solution`}` : "Geometry";

  return (
    <div className="app">
      <header className="toolbar">
        <div className="brand">LegumePhC <span>Studio</span></div>
        <div className="group">
          <button onClick={async () => { if (await confirmDiscard()) await loadDocument(await api.newDocument()); }}>New</button>
          <button onClick={() => openDialog("open")}>Open…</button>
          <button onClick={() => save(false)} title="Ctrl+S">Save</button>
          <button onClick={() => save(true)} title="Ctrl+Shift+S">Save as…</button>
        </div>
        <div className="group">
          <button className="primary" disabled={!!running} onClick={compute} title="Compute the selected study">= Compute{study ? `: ${study.name}` : ""}</button>
          <button disabled={!running} onClick={stop}>■ Stop</button>
          <button disabled={!project.results.length} onClick={clearAll}>Clear all solutions</button>
        </div>
        <div className="group">
          <button onClick={() => openDialog("presets")}>Apply preset…</button>
          <button onClick={savePreset}>Save preset…</button>
        </div>
        <div className="title">{dirty ? "● " : ""}{project.name}<span className="muted">{path ? ` — ${path}` : " — not saved"}</span></div>
      </header>

      <aside className="pane builder">
        <div className="pane-title">Model Builder</div>
        <ModelBuilder project={project} running={running?.studyId ?? null} onSelect={select} onAddStudy={() => studyAction("add")} onAddPlotGroup={addPlotGroup} />
      </aside>

      <section className="pane settings">
        <div className="pane-title">Settings</div>
        <div className="pane-body">{settings}</div>
      </section>

      <main className="pane graphics">
        <div className="pane-title">Graphics <span className="muted">· {graphicsTitle}</span></div>
        <div className="graphics-body">
          {showsResult
            ? <ResultView view={view} style={style} selected={selectedRow} pins={pins} onSelect={onSelectRow} message={viewMessage} />
            : <GeometryPreview previews={previews} error={previewError} />}
        </div>
      </main>

      <BottomPanel view={showsResult ? view : null} result={solution} selected={selectedRow} pins={pins} onSelect={onSelectRow} progress={progress} log={log} />

      <footer className="status">{status}</footer>

      {dialog && (
        <div className="modal-backdrop" onMouseDown={() => setDialog(null)}>
          <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
            <h3>{dialog === "open" ? "Open document" : "Apply parameter preset"}</h3>
            {dialog === "open" && (
              <>
                {documents.length === 0 && <div className="empty">No documents in {meta?.projects_dir}.</div>}
                <ul className="doc-list">
                  {documents.map((item) => (
                    <li key={item.path}>
                      <button onClick={() => openDocument(item.path)}>
                        <span className="doc-name">{item.kind === "legacy" ? "⇪ " : "▣ "}{item.name}</span>
                        <span className="muted">
                          {item.kind === "legacy"
                            ? item.converted ? "legacy project · already converted, opens the .lphc" : "legacy project · converts to .lphc on open"
                            : `${item.saved_at ? new Date(item.saved_at).toLocaleString() : ""} · ${item.studies ?? "?"} studies · ${item.results ?? 0} solutions · ${formatBytes(item.size)}`}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
            {dialog === "presets" && (
              <>
                {presets.length === 0 && <div className="empty">No presets saved yet.</div>}
                <ul className="doc-list">{presets.map((item) => <li key={item.path}><button onClick={() => applyPreset(item.path)}>{item.name}</button></li>)}</ul>
              </>
            )}
            <div className="button-row end"><button onClick={() => setDialog(null)}>Close</button></div>
          </div>
        </div>
      )}
    </div>
  );
}
