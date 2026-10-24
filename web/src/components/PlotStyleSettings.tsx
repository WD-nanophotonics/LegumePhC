import { useEffect, useState } from "react";
import type { Calculation, Operation, PlotGroup, PlotStyle } from "../types";
import { COMMON_STYLE, EXPORT_STYLE, OPERATION_BY_ID } from "../operations/definitions";
import { StyleForm } from "./SpecForm";

interface Props {
  group: PlotGroup;
  studies: Calculation[];
  style: PlotStyle;
  operation: Operation | null;
  onChange: (patch: PlotStyle) => void;
  onGroup: (patch: Partial<PlotGroup>) => void;
  onDelete: () => void;
  onExport: (fmt: string) => void;
  canExport: boolean;
}

/** Plot group settings: data source, operation-specific styling, axes, export. */
export function PlotStyleSettings({ group, studies, style, operation, onChange, onGroup, onDelete, onExport, canExport }: Props) {
  const op = operation ? OPERATION_BY_ID[operation] : undefined;
  const [name, setName] = useState(group.name);
  useEffect(() => setName(group.name), [group.name]);
  return (
    <div className="form">
      <div className="calc-header">
        <input className="title-input" value={name} onChange={(e) => setName(e.target.value)} onBlur={() => name.trim() && onGroup({ name: name.trim() })} />
        <span className="actions"><button title="Delete plot group" onClick={onDelete}>🗑</button></span>
      </div>
      <label>Dataset (study solution)
        <select value={group.calculation_id} onChange={(e) => onGroup({ calculation_id: e.target.value, pins: [] })}>
          {studies.map((study) => <option key={study.id} value={study.id}>{study.name}</option>)}
        </select>
      </label>
      {op && op.style.length > 0 && (
        <>
          <div className="section-title">{op.label}</div>
          <StyleForm specs={op.style} style={style} onChange={onChange} />
        </>
      )}
      <div className="section-title">Axes</div>
      <StyleForm specs={COMMON_STYLE} style={style} onChange={onChange} />
      <div className="section-title">Export (Matplotlib, publication)</div>
      <div className="row"><StyleForm specs={EXPORT_STYLE} style={style} onChange={onChange} /></div>
      <div className="button-row small">
        {["png", "svg", "pdf"].map((fmt) => <button key={fmt} disabled={!canExport} onClick={() => onExport(fmt)}>Export {fmt.toUpperCase()}</button>)}
      </div>
    </div>
  );
}
