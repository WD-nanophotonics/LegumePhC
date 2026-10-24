// Settings for the Component nodes (Lattice, Deformation, Materials, Sites).
// All numeric fields accept expressions using Global Definitions parameters.
import type { ChangeEvent, KeyboardEvent } from "react";
import type { Meta, ModelForm, SiteRow } from "../types";

export type ComponentSection = "lattice" | "deformation" | "materials" | "sites";

interface Props {
  section: ComponentSection;
  meta: Meta | null;
  form: ModelForm | null;
  onForm: (form: ModelForm, commit: boolean) => void;
  rows: SiteRow[];
  siteError: { index: number; field: string; message: string } | null;
  onRows: (rows: SiteRow[], commit: boolean) => void;
  onAddSite: () => void;
  onCenterSite: (index: number) => void;
  pointGroup: string | null | undefined;
}

export function ComponentSettings({ section, meta, form, onForm, rows, siteError, onRows, onAddSite, onCenterSite, pointGroup }: Props) {
  if (!form) return <div className="empty">Loading model…</div>;
  const set = (patch: Partial<ModelForm>, commit = false) => onForm({ ...form, ...patch }, commit);
  const commit = () => onForm(form, true);
  const text = (key: keyof ModelForm) => ({
    value: String(form[key] ?? ""),
    onChange: (e: ChangeEvent<HTMLInputElement>) => set({ [key]: e.target.value } as Partial<ModelForm>),
    onBlur: commit,
    onKeyDown: (e: KeyboardEvent) => e.key === "Enter" && commit(),
  });
  const matrix = (key: "basis" | "affine", disabled: boolean) => (
    <div className="grid2">
      {form[key].map((value, i) => (
        <input key={i} value={value} disabled={disabled}
          onChange={(e) => set({ [key]: form[key].map((v, k) => (k === i ? e.target.value : v)) } as Partial<ModelForm>)} onBlur={commit} />
      ))}
    </div>
  );
  const representation = form.material_representation;

  if (section === "lattice") {
    return (
      <div className="form">
        <div className="section-title first">Lattice</div>
        <label>Bravais lattice
          <select value={form.lattice} onChange={(e) => set({ lattice: e.target.value }, true)}>
            <option value="triangular">triangular</option><option value="square">square</option><option value="custom">custom</option>
          </select>
        </label>
        <div className="hint">{meta?.lattice_summary[form.lattice] ?? form.lattice_summary}</div>
        {form.lattice === "custom" && (<><label>Direct basis (columns a₁, a₂)</label>{matrix("basis", false)}</>)}
        <label>Geometry scale (dimensionless a)<input {...text("geometry_scale")} /></label>
        <label>Physical lattice constant
          <span className="row">
            <input {...text("actual_a")} placeholder="unset (normalized units only)" />
            <select value={form.length_unit} onChange={(e) => set({ length_unit: e.target.value }, true)}>
              {(meta?.length_units ?? ["nm", "μm", "m"]).map((unit) => <option key={unit}>{unit}</option>)}
            </select>
          </span>
        </label>
        <div className="section-title">Plane-wave basis</div>
        <label>Basis policy
          <select value={form.basis_policy} onChange={(e) => set({ basis_policy: e.target.value }, true)}>
            <option value="auto">auto (verified C3/C4 orbit closure)</option>
            <option value="native">native (comparison control)</option>
          </select>
        </label>
        <div className="kv"><span>Verified point group</span><span>{pointGroup ?? "none — generic basis"}</span></div>
      </div>
    );
  }

  if (section === "deformation") {
    return (
      <div className="form">
        <div className="section-title first">Affine deformation r → F·r + t</div>
        <label>Deformation
          <select value={form.deformation} onChange={(e) => set({ deformation: e.target.value }, true)}>
            <option value="none">none</option><option value="uniaxial">uniaxial strain</option><option value="custom">custom matrix F</option>
          </select>
        </label>
        {form.deformation === "uniaxial" && (
          <div className="row">
            <label>Factor<input {...text("factor")} /></label>
            <label>Axis angle (°)<input {...text("angle")} /></label>
          </div>
        )}
        {form.deformation === "custom" && (<><label>Matrix F (row-major)</label>{matrix("affine", false)}</>)}
        <div className="row">
          <label>Translation tₓ<input {...text("tx")} /></label>
          <label>Translation tᵧ<input {...text("ty")} /></label>
        </div>
        <div className="hint">A deformed lattice is never assigned the C3/C4 symmetry of its reference family; the BZ is rebuilt by half-plane clipping.</div>
      </div>
    );
  }

  if (section === "materials") {
    return (
      <div className="form">
        <div className="section-title first">Materials</div>
        <label>Material representation
          <select value={representation} onChange={(e) => set({ material_representation: e.target.value }, true)}>
            <option value="n">n (refractive index)</option><option value="epsilon">ε (relative permittivity)</option>
          </select>
        </label>
        <label>Background {representation === "n" ? "n" : "ε"}<input {...text("background")} /></label>
        <div className="hint">Inclusion materials are set per site under Geometry · Sites. The display representation never changes the stored ε.</div>
      </div>
    );
  }

  const editRow = (index: number, key: keyof SiteRow, value: string, commitNow = false) => {
    onRows(rows.map((row, i) => (i === index ? { ...row, [key]: value } : row)), commitNow);
  };
  return (
    <div className="form">
      <div className="section-title first">
        Geometry · Sites
        <span className="actions"><button onClick={onAddSite}>＋ Add site</button></span>
      </div>
      <div className="sites-table-wrap">
        <table className="sites-table">
          <thead>
            <tr><th>Name</th><th>Shape</th><th>Sides</th><th>r/a</th><th>Rot°</th><th>x/a</th><th>y/a</th><th>{representation === "n" ? "n" : "ε"}</th><th /></tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const bad = (field: string) => (siteError && siteError.index === index && siteError.field === field ? "bad" : "");
              const cell = (key: keyof SiteRow, disabled = false) => (
                <td><input className={bad(key)} value={row[key]} disabled={disabled} onChange={(e) => editRow(index, key, e.target.value)}
                  onBlur={() => onRows(rows, true)} onKeyDown={(e) => e.key === "Enter" && onRows(rows, true)} /></td>
              );
              return (
                <tr key={index}>
                  {cell("name")}
                  <td>
                    <select className={bad("shape")} value={row.shape} onChange={(e) => editRow(index, "shape", e.target.value, true)}>
                      {(meta?.shapes ?? ["Circle", "Triangle", "Square", "Regular polygon"]).map((shape) => <option key={shape}>{shape}</option>)}
                    </select>
                  </td>
                  {cell("sides", row.shape !== "Regular polygon")}
                  {cell("radius")}
                  {cell("rotation", row.shape === "Circle")}
                  {cell("center_x")}
                  {cell("center_y")}
                  {cell("material")}
                  <td className="row-actions">
                    <button title="Move to unit-cell centre" onClick={() => onCenterSite(index)}>⌖</button>
                    <button title="Remove site" disabled={rows.length <= 1} onClick={() => onRows(rows.filter((_, i) => i !== index), true)}>✕</button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {siteError && <div className="error-box">{siteError.message}</div>}
      <div className="hint">Centres are physical Cartesian x/a, y/a. Fields accept expressions and parameters, e.g. <code>sqrt(3)/6</code> or <code>r0*1.1</code>.</div>
    </div>
  );
}
