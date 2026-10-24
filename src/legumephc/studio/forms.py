"""Toolkit-independent Studio form <-> project-model conversion.

These functions hold the editing semantics that used to live inside the Qt
window (`_sync_model`, `populate`, and the Sites dialog) so that any client
(the web Studio today) edits a model exactly the same way.  Form values are
the raw strings a user typed; numbers go through :func:`safe_number`.

The typed expressions are stored next to the evaluated numbers
(``model["form_expressions"]`` and ``geometry["site_expressions"]``) so a
change to a Global Definitions parameter re-evaluates the whole component,
as in COMSOL.  The solver only ever sees the evaluated numbers.
"""

from __future__ import annotations

from copy import deepcopy
import keyword
from typing import Any, Mapping

import numpy as np

from ..motifs import triangular_motifs
from ..units import LENGTH_UNITS
from .geometry_editor import (
    GeometryExpressionError, canonical_shape, editor_value, epsilon_from_editor, model_name, safe_number, shape_choice,
    uniaxial_matrix,
)


Variables = Mapping[str, float] | None
RESERVED_NAMES = {"pi", "sqrt", "root"}
MODEL_FORM_KEYS = ("lattice", "actual_a", "length_unit", "material_representation", "background", "basis_policy",
                   "deformation", "factor", "angle", "basis", "affine", "tx", "ty", "geometry_scale")


class ParameterError(ValueError):
    def __init__(self, index: int, field: str, message: str):
        super().__init__(f"Parameter {index + 1} · {field}: {message}")
        self.index = index
        self.field = field


def evaluate_parameters(parameters: list[dict[str, Any]] | None) -> dict[str, float]:
    """Evaluate Global Definitions in order; later rows may use earlier ones."""

    values: dict[str, float] = {}
    for index, row in enumerate(parameters or []):
        name = str(row.get("name", "")).strip()
        if not name.isidentifier() or keyword.iskeyword(name) or name in RESERVED_NAMES:
            raise ParameterError(index, "name", f"'{name}' is not a valid parameter name")
        if name in values:
            raise ParameterError(index, "name", f"duplicate parameter '{name}'")
        try:
            values[name] = safe_number(row.get("expression", ""), values)
        except GeometryExpressionError as exc:
            raise ParameterError(index, "expression", str(exc)) from exc
    return values


TRIANGULAR_BASIS = [[0.5, 0.5], [float(np.sqrt(3) / 2), float(-np.sqrt(3) / 2)]]
SQUARE_BASIS = [[1.0, 0.0], [0.0, 1.0]]
LATTICE_SUMMARY = {
    "triangular": "a₁=(a/2, √3a/2), a₂=(a/2, −√3a/2)",
    "square": "a₁=(a, 0), a₂=(0, a)",
    "custom": "Custom dimensionless direct basis",
}


def number_text(value: Any) -> str:
    return format(float(value), ".12g")


def motifs(case: dict[str, Any]) -> list[dict[str, Any]]:
    geometry = case["geometry"]
    return deepcopy(geometry.get("motifs") or [{
        "name": geometry.get("name", "Site A"), "kind": geometry.get("kind", "circle"),
        "radius": geometry.get("radius", 0.2), "sides": geometry.get("sides", 0),
        "angle_degrees": geometry.get("angle_degrees", 0.0), "center": geometry.get("center", [0.5, 0.0]),
        "epsilon": geometry.get("epsilon_inclusion", 1.0),
    }])


def model_form(model: dict[str, Any], ui_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render a model into editable strings (the old Qt `populate`)."""

    state = ui_state or {}
    unit = state.get("length_unit", "nm")
    representation = state.get("material_representation", "n")
    stored = model.get("form_expressions")
    if stored and stored.get("length_unit") == unit and stored.get("material_representation") == representation:
        # Show what the user typed (it may reference parameters).
        return {**{key: deepcopy(stored[key]) for key in MODEL_FORM_KEYS if key in stored},
                "lattice_summary": LATTICE_SUMMARY.get(stored.get("lattice", "triangular"), "")}
    geometry = model["geometry"]
    deformation = model.get("deformation", {})
    basis = np.asarray(model.get("direct_basis", TRIANGULAR_BASIS), dtype=float).reshape(-1)
    affine = np.asarray(deformation.get("linear", model.get("affine", {}).get("linear", np.eye(2))), dtype=float).reshape(-1)
    translation = deformation.get("translation", model.get("affine", {}).get("translation", [0, 0]))
    actual = model.get("actual_lattice_constant_m")
    return {
        "lattice": model.get("lattice", "triangular"),
        "lattice_summary": LATTICE_SUMMARY.get(model.get("lattice", "triangular"), ""),
        "actual_a": "" if actual is None else number_text(float(actual) / LENGTH_UNITS[unit]),
        "length_unit": unit,
        "material_representation": representation,
        "background": number_text(editor_value(float(geometry.get("epsilon_background", 7.29)), representation)),
        "basis_policy": model.get("basis_policy", "auto"),
        "deformation": deformation.get("kind", "none"),
        "factor": number_text(deformation.get("factor", 1.0)),
        "angle": number_text(deformation.get("angle_degrees", 0.0)),
        "basis": [number_text(value) for value in basis],
        "affine": [number_text(value) for value in affine],
        "tx": number_text(translation[0]),
        "ty": number_text(translation[1]),
        "geometry_scale": number_text(model.get("lattice_constant", 1)),
    }


def apply_model_form(model: dict[str, Any], form: dict[str, Any], variables: Variables = None) -> dict[str, Any]:
    """Return an updated model from form strings (the old Qt `_sync_model`)."""

    def num(value: Any) -> float:
        return safe_number(value, variables)

    model = deepcopy(model)
    geometry = model["geometry"]
    lattice = str(form.get("lattice", model.get("lattice", "triangular")))
    if lattice not in LATTICE_SUMMARY:
        raise ValueError("lattice must be triangular, square, or custom")
    scale = num(form.get("geometry_scale", model.get("lattice_constant", 1)))
    if scale <= 0:
        raise ValueError("Geometry scale must be positive")
    if lattice == "triangular":
        direct = deepcopy(TRIANGULAR_BASIS)
    elif lattice == "square":
        direct = deepcopy(SQUARE_BASIS)
    else:
        direct = np.asarray([num(value) for value in form["basis"]], dtype=float).reshape(2, 2).tolist()
    kind = str(form.get("deformation", "none"))
    factor = num(form.get("factor", 1.0))
    angle = num(form.get("angle", 0.0))
    if kind == "none":
        linear = np.eye(2)
    elif kind == "uniaxial":
        linear = uniaxial_matrix(factor, angle)
    elif kind == "custom":
        linear = np.asarray([num(value) for value in form["affine"]], dtype=float).reshape(2, 2)
    else:
        raise ValueError("deformation must be none, uniaxial, or custom")
    if abs(float(np.linalg.det(linear))) < 1e-14:
        raise ValueError("Affine matrix must be non-singular")
    translation = [num(form.get("tx", 0)), num(form.get("ty", 0))]
    unit = str(form.get("length_unit", "nm"))
    actual_text = str(form.get("actual_a", "")).strip()
    actual_length = None if not actual_text else num(actual_text) * LENGTH_UNITS[unit]
    if actual_length is not None and actual_length <= 0:
        raise ValueError("Actual lattice constant must be positive")
    policy = str(form.get("basis_policy", "auto"))
    if policy not in {"auto", "native"}:
        raise ValueError("basis policy must be auto or native")
    model.update({"lattice": lattice, "lattice_constant": scale, "actual_lattice_constant_m": actual_length, "direct_basis": direct, "basis_policy": policy})
    geometry["epsilon_background"] = epsilon_from_editor(form.get("background", 2.7), str(form.get("material_representation", "n")), variables)
    model["deformation"] = {"kind": kind, "factor": factor, "angle_degrees": angle, "linear": linear.tolist(), "translation": translation}
    model["affine"] = {"linear": linear.tolist(), "translation": translation}
    model["name"] = model_name(lattice, motifs(model))
    geometry["name"] = model["name"]
    model["form_expressions"] = {key: deepcopy(form[key]) for key in MODEL_FORM_KEYS if key in form}
    return model


SITE_FIELDS = ("name", "shape", "radius", "rotation", "center_x", "center_y", "material", "sides")


def site_rows(model: dict[str, Any], representation: str) -> list[dict[str, str]]:
    """Render motifs as editable site-table rows (stored expressions first)."""

    stored = model["geometry"].get("site_expressions")
    if stored and stored.get("representation") == representation and len(stored.get("rows", [])) == len(motifs(model)):
        return deepcopy(stored["rows"])
    rows = []
    for motif in motifs(model):
        rows.append(_site_row(motif, representation))
    return rows


def _site_row(motif: dict[str, Any], representation: str) -> dict[str, str]:
    return {
        "name": str(motif.get("name", "Site")),
        "shape": shape_choice(motif.get("kind", "circle"), motif.get("sides")),
        "radius": number_text(motif.get("radius", 0.2)),
        "rotation": number_text(motif.get("angle_degrees", 0.0)),
        "center_x": number_text(motif.get("center", [0, 0])[0]),
        "center_y": number_text(motif.get("center", [0, 0])[1]),
        "material": number_text(editor_value(float(motif.get("epsilon", 1.0)), representation)),
        "sides": str(int(motif.get("sides") or 6)),
    }


def parse_site_row(row: dict[str, Any], index: int, representation: str, variables: Variables = None) -> dict[str, Any]:
    """Parse one table row; errors name the site and field for the client."""

    field = "shape"
    try:
        shape = str(row.get("shape", "Circle"))
        field = "sides"
        raw_sides = safe_number(row.get("sides"), variables) if shape == "Regular polygon" else {"Circle": 0, "Triangle": 3, "Square": 4}.get(shape, 0)
        kind, sides = canonical_shape(shape, raw_sides)
        values = {}
        for key in ("radius", "rotation", "center_x", "center_y", "material"):
            field = key
            values[key] = 0.0 if shape == "Circle" and key == "rotation" else safe_number(row.get(key, ""), variables)
        if values["radius"] <= 0:
            field = "radius"
            raise ValueError("must be positive")
        if values["material"] <= 0:
            field = "material"
            raise ValueError("must be positive")
        return {
            "name": str(row.get("name", "")).strip() or f"Site {index + 1}",
            "kind": kind, "radius": values["radius"], "sides": sides,
            "angle_degrees": 0.0 if kind == "circle" else values["rotation"],
            "center": [values["center_x"], values["center_y"]],
            "epsilon": epsilon_from_editor(values["material"], representation, variables),
        }
    except Exception as exc:
        raise SiteError(index, field, str(exc)) from exc


class SiteError(ValueError):
    def __init__(self, index: int, field: str, message: str):
        super().__init__(f"Site {index + 1} · {field}: {message}")
        self.index = index
        self.field = field


def apply_site_rows(model: dict[str, Any], rows: list[dict[str, Any]], representation: str, variables: Variables = None) -> dict[str, Any]:
    """Replace the model's motifs with parsed rows (the old Sites dialog accept)."""

    if not rows:
        raise ValueError("add at least one site")
    parsed = [parse_site_row(row, index, representation, variables) for index, row in enumerate(rows)]
    model = deepcopy(model)
    geometry = model["geometry"]
    first = parsed[0]
    geometry.update({
        "motifs": parsed, "kind": first["kind"], "radius": first["radius"], "sides": first["sides"],
        "angle_degrees": first["angle_degrees"], "center": first["center"], "epsilon_inclusion": first["epsilon"],
    })
    model["name"] = model_name(model.get("lattice", "triangular"), parsed)
    geometry["name"] = model["name"]
    geometry["site_expressions"] = {"representation": representation, "rows": [{key: str(row.get(key, "")) for key in SITE_FIELDS} for row in rows]}
    return model



def reevaluate_model(model: dict[str, Any], variables: Variables) -> dict[str, Any]:
    """Re-evaluate stored form and site expressions after a parameter change."""

    stored_form = model.get("form_expressions")
    if stored_form:
        model = apply_model_form(model, stored_form, variables)
    stored_sites = model["geometry"].get("site_expressions")
    if stored_sites:
        model = apply_site_rows(model, stored_sites["rows"], stored_sites["representation"], variables)
    return model


def add_site_row(model: dict[str, Any], rows: list[dict[str, Any]], representation: str, variables: Variables = None) -> list[dict[str, str]]:
    """Append a site; a second triangular site splits into a honeycomb pair."""

    rows = [dict(row) for row in rows]
    new = {"name": f"Site {len(rows) + 1}", "kind": "circle", "radius": 0.2, "angle_degrees": 0.0, "center": [0.5, 0.0], "epsilon": 1.0, "sides": 0}
    if model.get("lattice") == "triangular" and len(rows) == 1:
        try:
            first = parse_site_row(rows[0], 0, representation, variables)
            scale = float(model.get("lattice_constant", 1.0))
            if np.allclose(first["center"], np.asarray([0.5, 0.0]) * scale, rtol=0, atol=1e-12):
                locations = triangular_motifs((first["radius"], first["radius"]), (first["angle_degrees"], first["angle_degrees"]), kind=first["kind"], sides=first["sides"] or 3, epsilon=first["epsilon"], scale=scale)
                first.update({"name": "Site A", "center": locations[0]["center"]})
                rows[0] = _site_row(first, representation)
                new = {**first, "name": "Site B", "center": locations[1]["center"]}
        except ValueError:
            pass
    rows.append(_site_row(new, representation))
    return rows


OPERATIONS = {
    "frequency_at_k": "Frequency at k",
    "band_structure": "Band Structure",
    "berry": "Berry Curvature",
    "efs": "EFS",
    "fields_energy": "Fields / Energy",
    "berry_curvature_dipole": "Berry Curvature Dipole",
}


def normalize_calculation(parameters: dict[str, Any]) -> dict[str, Any]:
    """Derive band lists and Berry numeig (the old Qt `_sync_calculation`)."""

    value = deepcopy(parameters)
    operation = value.get("operation", "frequency_at_k")
    if operation not in OPERATIONS:
        raise ValueError("unsupported calculation operation")
    target_mode = value.get("berry_target_mode") or "single_band"
    band = int(value.get("band_one_based", 2))
    first = int(value.get("berry_first_band", band))
    last = int(value.get("berry_last_band", first))
    if operation == "berry" and target_mode == "single_band":
        bands = [band]
    else:
        if last < first:
            raise ValueError("last band must not precede first band")
        bands = list(range(first, last + 1))
    value["composite_bands_one_based"] = bands
    value["berry_target_mode"] = target_mode
    if operation == "berry":
        value["numeig"] = max(bands) + 1
    value["polarization"] = str(value.get("polarization", "te")).lower()
    return value


def cell_center(model: dict[str, Any]) -> list[float]:
    """The unit-cell centre used by the 'Center' site action."""

    scale = float(model.get("lattice_constant", 1.0))
    if model.get("lattice") == "triangular":
        return [0.5 * scale, 0.0]
    if model.get("lattice") == "square":
        return [0.5 * scale, 0.5 * scale]
    basis = np.asarray(model.get("direct_basis"), dtype=float)
    return ((basis[:, 0] + basis[:, 1]) / 2).tolist()
