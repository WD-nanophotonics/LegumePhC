// COMSOL-style Model Builder: Global Definitions → Component → Studies → Results.
import { useState, type ReactNode } from "react";
import type { NodeKind, Project } from "../types";
import { OPERATION_BY_ID } from "../operations/definitions";

export const COMPONENT_NODES: [string, string, string][] = [
  ["lattice", "Lattice", "⬡"],
  ["deformation", "Deformation (affine)", "⤢"],
  ["materials", "Materials", "◐"],
  ["sites", "Geometry · Sites", "◯"],
];

interface Props {
  project: Project;
  running: string | null;
  onSelect: (kind: NodeKind, id: string) => void;
  onAddStudy: () => void;
  onAddPlotGroup: (calculationId: string) => void;
}

function Branch({ label, icon, children, actions, defaultOpen = true }: { label: string; icon: string; children: ReactNode; actions?: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mb-branch">
      <div className="mb-branch-head" onClick={() => setOpen(!open)}>
        <span className="caret">{open ? "▾" : "▸"}</span><span className="icon">{icon}</span>{label}
        {actions && <span className="mb-actions" onClick={(e) => e.stopPropagation()}>{actions}</span>}
      </div>
      {open && <div className="mb-children">{children}</div>}
    </div>
  );
}

export function ModelBuilder({ project, running, onSelect, onAddStudy, onAddPlotGroup }: Props) {
  const selected = project.selected_node;
  const node = (kind: NodeKind, id: string, label: ReactNode, icon: string, title?: string) => (
    <div key={`${kind}:${id}`} className={`mb-node ${selected.kind === kind && selected.id === id ? "selected" : ""}`} title={title} onClick={() => onSelect(kind, id)}>
      <span className="icon">{icon}</span><span className="label">{label}</span>
    </div>
  );
  const solved = new Map(project.results.map((r) => [r.calculation_id, r]));
  return (
    <nav className="model-builder" aria-label="Model Builder">
      <div className="mb-root">▣ {project.name}</div>
      <Branch label="Global Definitions" icon="π">
        {node("parameters", "parameters", <>Parameters <span className="muted">({project.parameters.length})</span></>, "Σ")}
      </Branch>
      <Branch label="Component 1" icon="◇">
        {COMPONENT_NODES.map(([id, label, icon]) => node("component", id, id === "sites" ? `${label} (${(project.model.geometry.motifs ?? [1]).length})` : label, icon))}
      </Branch>
      <Branch label="Studies" icon="∿" actions={<button title="Add study" onClick={onAddStudy}>＋</button>}>
        {project.calculations.map((study) => {
          const solution = solved.get(study.id);
          const state = running === study.id ? "computing" : solution ? "solved" : "not computed";
          return node("study", study.id, <>
            {study.name}
            <span className={`state ${state.replace(" ", "-")}`}>{state === "computing" ? "⟳" : solution ? "✓" : "○"}</span>
          </>, "∿", `${OPERATION_BY_ID[study.operation]?.label ?? study.operation} · ${state}`);
        })}
      </Branch>
      <Branch label="Results" icon="▤">
        <Branch label="Datasets" icon="⛁">
          {project.calculations.filter((study) => solved.has(study.id)).map((study) =>
            node("dataset", study.id, `${study.name} / Solution`, "⛁", solved.get(study.id)!.record_reference.path))}
          {project.results.length === 0 && <div className="mb-empty">Compute a study to create a dataset</div>}
        </Branch>
        {project.plot_groups.map((group) => {
          const study = project.calculations.find((s) => s.id === group.calculation_id);
          return node("plot", group.id, <>{group.name}{!solved.has(group.calculation_id) && <span className="muted"> (no solution)</span>}</>, "📈", study ? `Data: ${study.name}` : undefined);
        })}
        {project.calculations.some((s) => solved.has(s.id)) && (
          <div className="mb-add">
            <select value="" onChange={(e) => e.target.value && onAddPlotGroup(e.target.value)}>
              <option value="">＋ Add plot group…</option>
              {project.calculations.filter((s) => solved.has(s.id)).map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </div>
        )}
      </Branch>
    </nav>
  );
}
