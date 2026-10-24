import type { CalculationParameters, Operation, ResultEntry } from "../types";
import { OPERATIONS, OPERATION_BY_ID } from "../operations/definitions";
import { FieldForm } from "./SpecForm";

interface Props {
  parameters: CalculationParameters | null;
  name: string;
  solution: ResultEntry | null;
  onChange: (parameters: CalculationParameters) => void;
  berrySources: ResultEntry[];
  running: boolean;
  thisRunning: boolean;
  onRun: () => void;
  onCancel: () => void;
  onClear: () => void;
  onCopy: () => void;
  onRename: () => void;
  onDelete: () => void;
}

/** Study settings: one study = one operation with one (replaceable) solution. */
export function CalculationSettings(props: Props) {
  const { parameters: p, onChange, berrySources, running, solution } = props;
  if (!p) return <div className="empty">Select a study in the Model Builder.</div>;
  const op = OPERATION_BY_ID[p.operation];
  return (
    <div className="form">
      <div className="calc-header">
        <strong>{props.name}</strong>
        <span className="actions">
          <button title="Duplicate study" onClick={props.onCopy}>⧉</button>
          <button title="Rename study" onClick={props.onRename}>✎</button>
          <button title="Delete study" onClick={props.onDelete}>🗑</button>
        </span>
      </div>
      <div className={`solution-status ${solution ? "solved" : ""}`}>
        {props.thisRunning ? "⟳ Computing…" : solution ? `✓ Solved ${solution.solved_at ? new Date(solution.solved_at).toLocaleString() : ""}` : "○ Not computed"}
        {solution && !props.thisRunning && <button className="link" onClick={props.onClear}>Clear solution</button>}
      </div>
      <label>Study type
        <select value={p.operation} onChange={(e) => onChange({ ...p, operation: e.target.value as Operation })}>
          {OPERATIONS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
      </label>
      {op ? <FieldForm fields={op.fields} parameters={p} onChange={(patch) => onChange({ ...p, ...patch })} context={{ berrySources }} />
        : <div className="error-box">This frontend has no form for operation “{p.operation}”.</div>}
      <div className="button-row">
        <button className="primary" disabled={running} onClick={props.onRun}>= Compute</button>
        <button disabled={!props.thisRunning} onClick={props.onCancel}>■ Stop</button>
      </div>
      {solution && <div className="hint">Computing again replaces this study's solution. Duplicate the study (⧉) to keep both.</div>}
    </div>
  );
}
