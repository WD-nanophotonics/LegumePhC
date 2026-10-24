import type { CalculationParameters, Operation, ResultEntry } from "../types";
import { OPERATIONS, OPERATION_BY_ID } from "../operations/definitions";
import { FieldForm } from "./SpecForm";

interface Props {
  parameters: CalculationParameters | null;
  name: string;
  onChange: (parameters: CalculationParameters) => void;
  berrySources: ResultEntry[];
  running: boolean;
  onRun: () => void;
  onCancel: () => void;
  onAdd: () => void;
  onCopy: () => void;
  onRename: () => void;
  onDelete: () => void;
}

export function CalculationSettings(props: Props) {
  const { parameters: p, onChange, berrySources, running } = props;
  if (!p) return <div className="empty">Select a calculation in the project tree.</div>;
  const op = OPERATION_BY_ID[p.operation];
  return (
    <div className="form">
      <div className="calc-header">
        <strong>{props.name}</strong>
        <span className="actions">
          <button title="Add calculation" onClick={props.onAdd}>＋</button>
          <button title="Duplicate calculation" onClick={props.onCopy}>⧉</button>
          <button title="Rename calculation" onClick={props.onRename}>✎</button>
          <button title="Delete calculation" onClick={props.onDelete}>🗑</button>
        </span>
      </div>
      <label>Calculation
        <select value={p.operation} onChange={(e) => onChange({ ...p, operation: e.target.value as Operation })}>
          {OPERATIONS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
      </label>
      {op ? <FieldForm fields={op.fields} parameters={p} onChange={(patch) => onChange({ ...p, ...patch })} context={{ berrySources }} />
        : <div className="error-box">This frontend has no form for operation “{p.operation}”.</div>}
      <div className="button-row">
        <button className="primary" disabled={running} onClick={props.onRun}>▶ Run</button>
        <button disabled={!running} onClick={props.onCancel}>■ Cancel</button>
      </div>
    </div>
  );
}
