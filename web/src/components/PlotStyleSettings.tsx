import type { Operation, PlotStyle } from "../types";
import { COMMON_STYLE, EXPORT_STYLE, OPERATION_BY_ID } from "../operations/definitions";
import { StyleForm } from "./SpecForm";

interface Props {
  style: PlotStyle;
  operation: Operation | null;
  onChange: (patch: PlotStyle) => void;
  onExport: (fmt: string) => void;
  canExport: boolean;
}

export function PlotStyleSettings({ style, operation, onChange, onExport, canExport }: Props) {
  const op = operation ? OPERATION_BY_ID[operation] : undefined;
  return (
    <div className="form">
      {op && op.style.length > 0 && (
        <>
          <div className="section-title first">{op.label}</div>
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
