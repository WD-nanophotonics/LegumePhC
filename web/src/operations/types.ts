// Declarative contracts for Studio operations. To add a calculation:
//   1. backend: implement it in legumephc.studio.worker.execute_request and
//      list it in legumephc.studio.forms.OPERATIONS (and inspection.record_view);
//   2. frontend: add one OperationDef in ./definitions.ts.
// Forms, plot-style panels and result plots are all generated from these specs.
import type Plotly from "plotly.js";
import type { CalculationParameters, PlotStyle, RecordViewData, ResultEntry } from "../types";

export interface Rendered {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
}

interface FieldBase {
  /** Parameter key in CalculationParameters (or a pseudo-key for composite kinds). */
  key: string;
  label: string;
  hint?: string;
  /** Show only when this returns true; default always. */
  visible?: (p: CalculationParameters) => boolean;
  readOnly?: (p: CalculationParameters) => boolean;
}

export type FieldSpec =
  | (FieldBase & { kind: "number"; min?: number })
  | (FieldBase & { kind: "int"; min?: number })
  | (FieldBase & { kind: "select"; options: [string, string][] })
  | (FieldBase & { kind: "list"; placeholder?: string })
  | (FieldBase & { kind: "qpoint" })
  | (FieldBase & { kind: "centers" })
  | (FieldBase & { kind: "berrySource" })
  | (FieldBase & { kind: "row"; fields: FieldSpec[] })
  | (FieldBase & { kind: "note" });

export type StyleSpec =
  | { key: string; label: string; kind: "check" }
  | { key: string; label: string; kind: "select"; options: [string, string][] }
  | { key: string; label: string; kind: "number"; step?: number; min?: number }
  | { key: string; label: string; kind: "optionalNumber" }
  | { key: string; label: string; kind: "text" }
  | { key: string; label: string; kind: "limits" };

export interface OperationDef {
  id: string;
  label: string;
  /** Calculation form, top to bottom. */
  fields: FieldSpec[];
  /** Operation-specific plot-style controls (common ones are added automatically). */
  style: StyleSpec[];
  /** Style keys whose change needs the record view re-fetched from the server. */
  viewKeys?: string[];
  /** Draw the record, or return null to show `summaryCard` instead. */
  render: (view: RecordViewData, style: PlotStyle) => Rendered | null;
  /** Short label for a result in the project tree. */
  describe: (c: CalculationParameters) => string;
}

export interface FieldContext {
  berrySources: ResultEntry[];
}
