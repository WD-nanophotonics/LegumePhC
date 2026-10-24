import type { Meta, ModelForm, SiteRow } from "../types";

interface Props {
  meta: Meta | null;
  form: ModelForm | null;
  onForm: (form: ModelForm, commit: boolean) => void;
  rows: SiteRow[];
  siteError: { index: number; field: string; message: string } | null;
  onRows: (rows: SiteRow[], commit: boolean) => void;
  onAddSite: () => void;
  onCenterSite: (index: number) => void;
  advanced: boolean;
  onAdvanced: (open: boolean) => void;
}

const FIELD_COLUMN: Record<string, keyof SiteRow> = {
  name: "name", shape: "shape", radius: "radius", rotation: "rotation", center_x: "center_x", center_y: "center_y", material: "material", sides: "sides",
};

export function GeometrySettings({ meta, form, onForm, rows, siteError, onRows, onAddSite, onCenterSite, advanced, onAdvanced }: Props) {
  if (!form) return <div className="empty">Loading model…</div>;
  const set = (patch: Partial<ModelForm>, commit = false) => onForm({ ...form, ...patch }, commit);
  const commit = () => onForm(form, true);
  const text = (key: keyof ModelForm) => ({
    value: String(form[key] ?? ""),
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => set({ [key]: e.target.value } as Partial<ModelForm>),
    onBlur: commit,
    onKeyDown: (e: React.KeyboardEvent) => e.key === "Enter" && commit(),
  });
  const representation = form.material_representation;
  const editRow = (index: number, key: keyof SiteRow, value: string, commitNow = false) => {
    const next = rows.map((row, i) => (i === index ? { ...row, [key]: value } : row));
    onRows(next, commitNow);
  };
  return (
    <div className="form">
      <label>Lattice
        <select value={form.lattice} onChange={(e) => set({ lattice: e.target.value }, true)}>
          <option value="triangular">triangular</option><option value="square">square</option><option value="custom">custom</option>
        </select>
      </label>
      <div className="hint">{meta?.lattice_summary[form.lattice] ?? form.lattice_summary}</div>
      <label>Actual lattice constant
        <span className="row">
          <input {...text("actual_a")} placeholder="unset" />
          <select value={form.length_unit} onChange={(e) => set({ length_unit: e.target.value }, true)}>
            {(meta?.length_units ?? ["nm", "μm", "m"]).map((unit) => <option key={unit}>{unit}</option>)}
          </select>
        </span>
      </label>
      <label>Material representation
        <select value={representation} onChange={(e) => set({ material_representation: e.target.value }, true)}>
          <option value="n">n (refractive index)</option><option value="epsilon">ε (permittivity)</option>
        </select>
      </label>
      <label>Background {representation === "n" ? "n" : "ε"}<input {...text("background")} /></label>
      <label>Basis policy
        <select value={form.basis_policy} onChange={(e) => set({ basis_policy: e.target.value }, true)}>
          <option value="auto">auto (verified C3/C4 orbit closure)</option>
          <option value="native">native (comparison control)</option>
        </select>
      </label>
      <label>Deformation
        <select value={form.deformation} onChange={(e) => set({ deformation: e.target.value }, true)}>
          <option value="none">none</option><option value="uniaxial">uniaxial</option><option value="custom">custom affine</option>
        </select>
      </label>
      {form.deformation === "uniaxial" && (
        <div className="row">
          <label>Factor<input {...text("factor")} /></label>
          <label>Angle (°)<input {...text("angle")} /></label>
        </div>
      )}

      <div className="section-title">
        Sites
        <span className="actions">
          <button onClick={onAddSite}>Add</button>
        </span>
      </div>
      <div className="sites-table-wrap">
        <table className="sites-table">
          <thead>
            <tr><th>Name</th><th>Shape</th><th>Sides</th><th>r/a</th><th>Rot°</th><th>x/a</th><th>y/a</th><th>{representation === "n" ? "n" : "ε"}</th><th /></tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const bad = (field: string) => siteError && siteError.index === index && FIELD_COLUMN[siteError.field] === field ? "bad" : "";
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
      <div className="hint">Centres are physical Cartesian x/a, y/a. Expressions like <code>sqrt(3)/6</code> are accepted.</div>

      <details open={advanced} onToggle={(e) => onAdvanced((e.target as HTMLDetailsElement).open)}>
        <summary>Advanced geometry</summary>
        <label>Custom direct basis (columns a₁, a₂)</label>
        <div className="grid2">
          {form.basis.map((value, i) => (
            <input key={i} value={value} disabled={form.lattice !== "custom"}
              onChange={(e) => set({ basis: form.basis.map((v, k) => (k === i ? e.target.value : v)) })} onBlur={commit} />
          ))}
        </div>
        <label>Affine matrix F</label>
        <div className="grid2">
          {form.affine.map((value, i) => (
            <input key={i} value={value} disabled={form.deformation !== "custom"}
              onChange={(e) => set({ affine: form.affine.map((v, k) => (k === i ? e.target.value : v)) })} onBlur={commit} />
          ))}
        </div>
        <div className="row">
          <label>Translation x<input {...text("tx")} /></label>
          <label>Translation y<input {...text("ty")} /></label>
        </div>
        <label>Geometry scale<input {...text("geometry_scale")} /></label>
        <div className="hint">A generic affine model is never assigned C3/C4 symmetry; its point group shows as "none".</div>
      </details>
    </div>
  );
}
