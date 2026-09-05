from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib.image import imread
import numpy as np
import pytest

from legumephc.records import create_record
from legumephc.studio.plotting import export_figure, plot_record
from legumephc.studio.preview import preview_geometry
from legumephc.studio.profile import model_from_case, zero_based_band, zero_based_bands
from legumephc.studio.project import (
    apply_preset,
    load_preset,
    load_project,
    new_preset,
    new_project,
    record_available,
    record_reference,
    project_records_dir,
    save_preset,
    save_project,
    validate_project,
)
from legumephc.studio.worker import REQUEST_SCHEMA, WorkerProcess, build_worker_request, execute_request, validate_request


def test_project_preset_roundtrip_and_preset_has_no_result_references(tmp_path):
    project = new_project("demo")
    project["case"]["lattice"] = "custom"
    project["case"]["direct_basis"] = [[1.0, 0.2], [0.0, 0.9]]
    project_path = save_project(tmp_path / "demo.legumephc-studio.json", project)
    assert load_project(project_path) == project
    assert "frequencies" not in project_path.read_text(encoding="utf-8")
    preset = new_preset("demo parameters")
    preset_path = save_preset(tmp_path / "demo.legumephc-preset.json", preset)
    loaded = load_preset(preset_path)
    assert set(loaded) == {"schema", "name", "parameters"}
    assert "records" not in loaded["parameters"]
    applied = apply_preset(project, loaded)
    assert applied["records"] == project["records"]
    assert applied["selected_result"] == project["selected_result"]


def test_record_reference_rejects_escape_and_detects_missing_changed_and_valid(tmp_path):
    record = create_record(tmp_path / "results", identity={"model": "Model2D", "geometry": "x", "affine": {}, "basis": {}, "solver": "test", "operation": "band_structure"}, config={}, summary={"status": "succeeded"}, arrays={"frequencies": np.ones((2, 1))})
    reference = record_reference(record, tmp_path)
    assert record_available(tmp_path, reference, operation="band_structure") == (True, "available")
    (record / "summary.json").write_text("changed", encoding="utf-8")
    available, reason = record_available(tmp_path, reference)
    assert not available and ("hash" in reason or "size" in reason)
    missing = dict(reference)
    missing["path"] = "does-not-exist"
    assert not record_available(tmp_path, missing)[0]
    with pytest.raises(ValueError, match="escapes"):
        record_reference(tmp_path.parent, tmp_path)


def test_project_input_validation_and_band_mapping():
    project = new_project()
    project["calculation"]["band_one_based"] = 0
    with pytest.raises(ValueError, match="one-based"):
        validate_project(project)
    assert zero_based_band(1) == 0
    assert zero_based_bands([2, 3]) == (1, 2)
    with pytest.raises(ValueError, match="one-based"):
        zero_based_band(0)


def test_preview_is_solver_free_equal_aspect_and_affine_aware():
    project = new_project()
    project["case"]["affine"] = {"linear": [[1.0, 0.18], [0.0, 0.92]], "translation": [0.07, -0.03]}
    data = preview_geometry(project["case"], view="epsilon", size=32)
    model = model_from_case(project["case"])
    assert data["equal_aspect"] and data["shape"] == (32, 32)
    assert data["epsilon"].shape == (32, 32)
    assert np.allclose(data["direct_basis"], model.effective_lattice.direct_basis)
    assert np.isfinite(data["epsilon"]).all()
    sites = preview_geometry(project["case"], view="lattice_sites", size=32)
    assert sites["equal_aspect"] and sites["points"].shape == (9, 2)


def test_plot_fixture_and_exact_pixel_export(tmp_path):
    record = create_record(tmp_path / "results", identity={"model": "Model2D", "geometry": "x", "affine": {}, "basis": {}, "solver": "test", "operation": "band_structure"}, config={}, summary={"status": "succeeded", "operation": "band_structure"}, arrays={"frequencies": np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]), "qpoints": np.zeros((3, 2))})
    figure = plot_record(record, {"width_px": 400, "height_px": 300, "dpi": 100, "title": "Fixture"})
    output = export_figure(figure, tmp_path / "plot.png", width_px=400, height_px=300, dpi=100)
    image = imread(output)
    assert image.shape[1] == 400 and image.shape[0] == 300


def test_plot_berry_uses_multi_plaquette_centers_and_status(tmp_path):
    plaquettes = np.asarray([
        [[0.0, 0.0], [0.0, 0.1], [0.1, 0.1], [0.1, 0.0]],
        [[0.2, 0.0], [0.2, 0.1], [0.3, 0.1], [0.3, 0.0]],
    ])
    record = create_record(tmp_path / "results", identity={"model": "Model2D", "geometry": "x", "affine": {}, "basis": {}, "solver": "test", "operation": "berry"}, config={}, summary={"status": "succeeded", "operation": "berry", "qualification": {"overall_status": "UNQUALIFIED_CONVERGENCE_NOT_ASSESSED"}}, arrays={"qpoints": plaquettes.reshape(-1, 2), "plaquettes": plaquettes, "curvature": np.asarray([1.0, -2.0])})
    figure = plot_record(record, {"berry_coloring": True})
    axis = figure.axes[0]
    offsets = axis.collections[0].get_offsets()
    assert offsets.shape == (2, 2)
    assert "UNQUALIFIED_CONVERGENCE_NOT_ASSESSED" in figure.texts[0].get_text()


def test_plot_efs_reconstructs_grid_contour_and_labels_sparse(tmp_path):
    grid = np.asarray([[x, y] for y in (-0.5, 0.0, 0.5) for x in (-0.5, 0.0, 0.5)])
    mask = np.ones((3, 3), dtype=bool)
    frequencies = (grid[:, 0] ** 2 + grid[:, 1] ** 2).reshape(-1, 1)
    record = create_record(tmp_path / "results", identity={"model": "Model2D", "geometry": "x", "affine": {}, "basis": {}, "solver": "test", "operation": "efs"}, config={}, summary={"status": "succeeded", "operation": "efs", "grid_shape": [3, 3]}, arrays={"grid_qpoints": grid, "inside_bz_mask": mask, "qpoints": grid, "frequencies": frequencies})
    figure = plot_record(record)
    assert len(figure.axes) == 2
    assert figure.axes[0].collections

    sparse = create_record(tmp_path / "results", identity={"model": "Model2D", "geometry": "x", "affine": {}, "basis": {}, "solver": "test", "operation": "efs"}, config={}, summary={"status": "succeeded", "operation": "efs"}, arrays={"qpoints": grid[:3], "frequencies": frequencies[:3]})
    sparse_figure = plot_record(sparse)
    assert "sparse" in sparse_figure.axes[0].get_title().lower()


def test_fake_worker_success_failure_cancel_and_no_orphan(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")
    success = WorkerProcess(subprocess.Popen([sys.executable, "-c", "print('ok')"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), request_path)
    stdout, _ = success.communicate()
    assert stdout.strip() == "ok" and not request_path.exists()

    failure_path = tmp_path / "failure.json"
    failure_path.write_text("{}", encoding="utf-8")
    failure = WorkerProcess(subprocess.Popen([sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(3)"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), failure_path)
    failure_stdout, failure_stderr = failure.communicate()
    assert failure.process.returncode == 3 and "bad" in failure_stderr and failure_stdout == ""

    cancel_path = tmp_path / "cancel.json"
    cancel_path.write_text("{}", encoding="utf-8")
    cancel = WorkerProcess(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), cancel_path)
    cancel_pid = cancel.process.pid
    cancel.cancel()
    assert cancel.process.poll() is not None and not cancel_path.exists()
    time.sleep(0.05)
    assert cancel.process.poll() is not None


def test_record_reload_type_check(tmp_path):
    record = create_record(tmp_path / "results", identity={"model": "Model2D", "geometry": "x", "affine": {}, "basis": {}, "solver": "test", "operation": "frequency_at_k"}, config={}, summary={"status": "succeeded"}, arrays={"frequency": np.asarray([0.2])})
    reference = record_reference(record, tmp_path)
    assert record_available(tmp_path, reference, operation="frequency_at_k")[0]
    assert record_available(tmp_path, reference, operation="efs")[0]


def test_worker_requires_contained_records_and_bcd_consumes_selected_berry_record(tmp_path):
    project = new_project()
    project["calculation"]["operation"] = "berry_curvature_dipole"
    berry_dir = tmp_path / "data" / ".studio" / "records"
    source_identity = {"model": "Model2D", "geometry": "SourceGeometry", "affine": {"linear": [[1.0, 0.4], [0.0, 0.8]], "translation": [0.1, -0.2]}, "basis": {"policy": "source", "direct_basis": [[1.0, 0.0], [0.0, 1.0]]}, "solver": "test", "operation": "berry"}
    arrays = {"plaquettes": np.asarray([[[0.0, 0.0], [0.0, 0.1], [0.1, 0.1], [0.1, 0.0]], [[0.2, 0.0], [0.2, 0.1], [0.3, 0.1], [0.3, 0.0]]]), "curvature": np.asarray([1.0, -2.0])}
    berry = create_record(berry_dir, identity=source_identity, config={}, summary={"status": "succeeded", "qualification": {"overall_status": "UNQUALIFIED_CONVERGENCE_NOT_ASSESSED"}}, arrays=arrays)
    reference = record_reference(berry, tmp_path)
    request = {"schema": REQUEST_SCHEMA, "project_dir": str(tmp_path), "records_dir": str(berry_dir), "records": [reference], "selected_result": reference["path"], "case": project["case"], "calculation": project["calculation"]}
    with pytest.raises(ValueError, match="overall_status=UNQUALIFIED_CONVERGENCE_NOT_ASSESSED"):
        validate_request(request)

    qualified_summary = {"status": "succeeded", "qualification": {"overall_status": "QUALIFIED"}}
    qualified = create_record(berry_dir, identity=source_identity, config={}, summary=qualified_summary, arrays=arrays)
    qualified_reference = record_reference(qualified, tmp_path)
    request["records"] = [qualified_reference]
    request["selected_result"] = qualified_reference["path"]
    project["case"]["geometry"]["name"] = "DifferentCurrentGeometry"
    request["case"] = project["case"]
    result = execute_request(request)
    result_dir = Path(result["record_path"])
    with (result_dir / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    with (result_dir / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    assert summary["kind"] == "geometric" and summary["physical_response"] is False
    assert config["identity"]["geometry"] == "SourceGeometry"
    assert config["identity"]["affine"] == source_identity["affine"]
    assert config["identity"]["basis"] == source_identity["basis"]
    assert config["config"]["source_berry"]["path"] == qualified_reference["path"]
    assert config["config"]["source_berry"]["files"] == qualified_reference["files"]
    assert summary["source_berry"]["files"] == qualified_reference["files"]

    physical_request = {**request, "calculation": {**request["calculation"], "frequency_window": [0.0, 1.0], "frequency_samples": [0.2, 0.3], "occupation": [1.0, 1.0]}}
    physical = execute_request(physical_request)
    with (Path(physical["record_path"]) / "summary.json").open(encoding="utf-8") as stream:
        assert json.load(stream)["physical_response"] is True
    bad_length = {**request, "calculation": {**request["calculation"], "frequency_samples": [0.2]}}
    with pytest.raises(ValueError, match="one value per Berry plaquette center"):
        validate_request(bad_length)
    outside = {**request, "records_dir": str(tmp_path.parent / "outside")}
    with pytest.raises(ValueError, match="inside"):
        validate_request(outside)
    missing = {**request, "selected_result": None, "records": []}
    with pytest.raises(ValueError, match="selected compatible Berry"):
        validate_request(missing)


def test_saved_project_sidecar_and_old_result_can_be_reloaded_and_replotted(tmp_path):
    project = new_project("saved")
    project_path = save_project(tmp_path / "saved.legumephc-studio.json", project)
    sidecar = project_records_dir(project_path)
    record = create_record(sidecar, identity={"model": "Model2D", "geometry": "x", "affine": {}, "basis": {}, "solver": "test", "operation": "band_structure"}, config={}, summary={"status": "succeeded", "operation": "band_structure", "path_labels": ["Gamma", "X"]}, arrays={"frequencies": np.asarray([[0.1], [0.2]]), "qpoints": np.zeros((2, 2))})
    project["records"] = [record_reference(record, project_path.parent)]
    project["selected_result"] = project["records"][0]["path"]
    save_project(project_path, project)
    reopened = load_project(project_path)
    reopened["calculation"]["operation"] = "efs"
    assert record_available(project_path.parent, reopened["records"][0], operation="efs")[0]
    assert plot_record(project_path.parent / reopened["selected_result"]).axes[0].get_xticklabels()[0].get_text() == "Gamma"


def test_run_request_requires_saved_project_and_uses_project_sidecar(tmp_path):
    project = new_project()
    with pytest.raises(ValueError, match="save"):
        build_worker_request(project, tmp_path / "not-yet-saved.legumephc-studio.json")
    path = save_project(tmp_path / "saved.legumephc-studio.json", project)
    request = build_worker_request(project, path)
    assert Path(request["records_dir"]) == project_records_dir(path)
    assert Path(request["records_dir"]).is_relative_to(path.parent)


def test_hidden_root_tk_smoke_when_available():
    try:
        import tkinter as tk
    except ImportError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        from legumephc.studio.ui import StudioApp
        root = StudioApp()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    root.update_idletasks()
    assert "Studio" in root.title()
    root.destroy()
