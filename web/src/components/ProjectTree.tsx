import type { Meta, Project, ResultEntry } from "../types";
import { OPERATION_BY_ID } from "../operations/definitions";

interface Props {
  meta: Meta | null;
  project: Project;
  onSelect: (kind: "geometry" | "calculation" | "result", id: string) => void;
}

export function resultLabel(result: ResultEntry): string {
  const c = result.calculation_snapshot ?? ({} as ResultEntry["calculation_snapshot"]);
  const name = result.record_reference.path.split("/").pop() ?? "result";
  const stamp = name.slice(0, 15).replace(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2}).*/, "$1-$2-$3 $4:$5:$6");
  const detail = OPERATION_BY_ID[c.operation]?.describe(c) ?? c.operation;
  return `${stamp} · ${detail}`;
}

export function ProjectTree({ meta, project, onSelect }: Props) {
  const selected = project.selected_node;
  const is = (kind: string, id: string) => (selected.kind === kind && selected.id === id ? "selected" : "");
  const groups = new Map<string, ResultEntry[]>();
  [...project.results].reverse().forEach((result) => {
    const op = result.calculation_snapshot?.operation ?? "result";
    groups.set(op, [...(groups.get(op) ?? []), result]);
  });
  return (
    <nav className="tree">
      <div className={`tree-item root ${is("geometry", "geometry")}`} onClick={() => onSelect("geometry", "geometry")}>◇ Model / Geometry</div>
      <div className="tree-group">Studies</div>
      {project.calculations.map((calc) => (
        <div key={calc.id} className={`tree-item ${is("calculation", calc.id)}`} onClick={() => onSelect("calculation", calc.id)}>
          ▸ {calc.name || meta?.operations[calc.operation] || calc.operation}
          <span className="muted"> · {meta?.operations[calc.operation] ?? calc.operation}</span>
        </div>
      ))}
      <div className="tree-group">Results ({project.results.length})</div>
      {[...groups.entries()].map(([op, results]) => (
        <div key={op}>
          <div className="tree-subgroup">{meta?.operations[op as keyof Meta["operations"]] ?? op}</div>
          {results.map((result) => (
            <div key={result.id} className={`tree-item leaf ${is("result", result.id)}`} title={result.record_reference.path} onClick={() => onSelect("result", result.id)}>
              {resultLabel(result)}
            </div>
          ))}
        </div>
      ))}
    </nav>
  );
}
