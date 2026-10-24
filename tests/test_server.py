"""HTTP contract of the localhost web Studio backend (single-file documents)."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from legumephc.records import create_record
from legumephc.server import create_app
from legumephc.server.app import ROOT
from legumephc.studio.project import new_project, project_records_dir, record_reference, save_project


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(tmp_path / "projects", workspace_root=tmp_path / "ws")) as value:
        yield value


def _events(client: TestClient, run_id: str) -> list[dict]:
    events = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
    return events


def _fake_band_record(records: Path, project: dict) -> Path:
    return create_record(records, identity={"model": "Model2D", "geometry": "g", "affine": {}, "basis": "native", "solver": "legumephc.public-api", "operation": "band_structure"},
                         config={"case": project["model"], "calculation": {**project["calculations"][0]["parameters"], "operation": "band_structure"}},
                         summary={"status": "succeeded", "path_labels": ["Gamma", "K", "M", "Gamma"]},
                         arrays={"frequencies": np.array([[0.0, 0.4], [0.2, 0.5], [0.25, 0.45], [0.0, 0.4]]), "qpoints": np.zeros((4, 2)), "vectors": np.ones((4, 2), dtype=complex)})


def _entry(record: Path, workspace: Path, project: dict, calculation_id: str = "calc-1") -> dict:
    return {"calculation_id": calculation_id, "record_reference": record_reference(record, workspace), "model_snapshot": project["model"],
            "calculation_snapshot": {**project["calculations"][0]["parameters"], "operation": "band_structure"}}


def test_meta_and_new_document(client):
    meta = client.get("/api/meta").json()
    assert set(meta["operations"]) == {"frequency_at_k", "band_structure", "fields_energy", "efs", "berry", "berry_curvature_dipole", "valley_chern"}
    assert meta["document_suffix"] == ".lphc"
    created = client.post("/api/documents/new", params={"name": "A"}).json()
    assert created["path"] is None and created["project"]["name"] == "A"
    assert created["project"]["parameters"] == [] and created["project"]["plot_groups"] == []


def test_save_reopen_and_path_containment(client, tmp_path):
    doc = client.post("/api/documents/new").json()
    saved = client.post(f"/api/documents/{doc['doc_id']}/save", json={"path": "nested/P", "project": doc["project"]}).json()
    assert saved["path"] == "nested/P.lphc"
    file = tmp_path / "projects" / "nested" / "P.lphc"
    with zipfile.ZipFile(file) as archive:
        assert {"meta.json", "project.json"} <= set(archive.namelist())
    listing = client.get("/api/projects").json()
    assert listing[0]["path"] == "nested/P.lphc" and listing[0]["kind"] == "document"
    reopened = client.post("/api/documents/open", json={"path": "nested/P.lphc"}).json()
    assert reopened["project"]["name"] == doc["project"]["name"] and reopened["doc_id"] != doc["doc_id"]
    assert client.post(f"/api/documents/{doc['doc_id']}/save", json={"path": "../outside", "project": doc["project"]}).status_code == 403
    assert client.post(f"/api/documents/{doc['doc_id']}/save", json={"path": "Bad", "project": dict(doc["project"], calculations=[])}).status_code == 422
    assert client.delete(f"/api/documents/{doc['doc_id']}").json()["closed"] is True
    assert client.post(f"/api/documents/{doc['doc_id']}/save", json={"path": "X", "project": doc["project"]}).status_code == 404


def test_solution_replace_view_export_and_prune(client, tmp_path):
    doc = client.post("/api/documents/new", params={"name": "R"}).json()
    workspace = client.app.state.documents[doc["doc_id"]].workspace
    project = doc["project"]
    records = project_records_dir(workspace / "project.json")
    first = _fake_band_record(records, project)
    project = client.post(f"/api/documents/{doc['doc_id']}/attach", json={"project": project, "entry": _entry(first, workspace, project)}).json()["project"]
    assert len(project["results"]) == 1 and project["plot_groups"][0]["calculation_id"] == "calc-1"
    assert project["selected_node"] == {"kind": "plot", "id": project["plot_groups"][0]["id"]}
    second = _fake_band_record(records, project)
    project = client.post(f"/api/documents/{doc['doc_id']}/attach", json={"project": project, "entry": _entry(second, workspace, project)}).json()["project"]
    assert len(project["results"]) == 1 and project["results"][0]["record_reference"]["path"].endswith(second.name)
    assert len(project["plot_groups"]) == 1  # the plot group follows the study

    record = project["results"][0]["record_reference"]["path"]
    view = client.post(f"/api/documents/{doc['doc_id']}/records/view", json={"record": record, "style": {"frequency_unit": "THz"}}).json()
    assert view["operation"] == "band_structure" and len(view["rows"]) == 8
    assert "vectors" not in view["arrays"]  # raw complex eigenvectors never leave the server
    assert view["frequency_factor"] == pytest.approx(299792458.0 / 400e-9 / 1e12)
    exported = client.post(f"/api/documents/{doc['doc_id']}/records/export", json={"record": record, "style": dict(new_project()["plot"]), "fmt": "png"})
    assert exported.status_code == 200 and exported.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (second / "figures" / "figure.png").exists()  # records stay immutable
    assert client.post(f"/api/documents/{doc['doc_id']}/records/view", json={"record": "../../etc"}).status_code == 404

    client.post(f"/api/documents/{doc['doc_id']}/save", json={"path": "R", "project": project})
    assert not first.exists() and second.exists()  # replaced solution pruned on save
    with zipfile.ZipFile(tmp_path / "projects" / "R.lphc") as archive:
        members = archive.namelist()
    assert any(second.name in name for name in members) and not any(first.name in name for name in members)
    cleared = client.post(f"/api/documents/{doc['doc_id']}/clear", json={"project": project}).json()["project"]
    assert cleared["results"] == [] and cleared["selected_result"] is None


def test_legacy_project_is_converted_on_open(client, tmp_path):
    projects = tmp_path / "projects"
    legacy = projects / "Old.legumephc-studio.json"
    project = new_project("Old")
    records = project_records_dir(legacy)
    a, b = _fake_band_record(records, project), _fake_band_record(records, project)
    project["results"] = [{**_entry(record, projects, project), "id": f"result-{i}"} for i, record in enumerate((a, b), start=1)]
    save_project(legacy, project)
    before = legacy.read_bytes()
    listing = client.get("/api/projects").json()
    assert listing[0]["kind"] == "legacy" and listing[0]["converted"] is False
    opened = client.post("/api/documents/open", json={"path": "Old.legumephc-studio.json"}).json()
    assert opened["path"] == "Old.lphc" and opened["converted_from"] == "Old.legumephc-studio.json"
    converted = opened["project"]
    assert len(converted["results"]) == 2 and len({r["calculation_id"] for r in converted["results"]}) == 2  # one solution per study
    assert any("imported" in study["name"] for study in converted["calculations"])
    assert legacy.read_bytes() == before and a.exists() and b.exists()  # originals untouched
    again = client.post("/api/documents/open", json={"path": "Old.legumephc-studio.json"}).json()
    assert again["converted_from"] is None  # existing .lphc is reused, not overwritten


def test_parameters_reevaluate_model_and_report_errors(client):
    project = client.post("/api/documents/new").json()["project"]
    project["parameters"] = [{"name": "r0", "expression": "0.2", "description": ""}, {"name": "rr", "expression": "r0*1.5"}]
    rows = client.post("/api/sites/rows", json={"model": project["model"]}).json()["rows"]
    model = client.post("/api/sites/apply", json={"model": project["model"], "rows": [{**rows[0], "radius": "rr"}], "parameters": project["parameters"]}).json()["model"]
    assert model["geometry"]["motifs"][0]["radius"] == pytest.approx(0.3)
    project["model"] = model
    project["parameters"][0]["expression"] = "0.1"
    applied = client.post("/api/parameters/apply", json={"project": project}).json()
    assert applied["values"] == {"r0": 0.1, "rr": pytest.approx(0.15)}
    assert applied["project"]["model"]["geometry"]["motifs"][0]["radius"] == pytest.approx(0.15)
    assert client.post("/api/sites/rows", json={"model": applied["project"]["model"]}).json()["rows"][0]["radius"] == "rr"
    project["parameters"].append({"name": "pi", "expression": "3"})
    bad = client.post("/api/parameters/apply", json={"project": project})
    assert bad.status_code == 422 and bad.json()["index"] == 2 and bad.json()["field"] == "name"


def test_sites_errors_are_structured_and_preview_is_json(client):
    project = client.post("/api/documents/new").json()["project"]
    rows = client.post("/api/sites/rows", json={"model": project["model"]}).json()["rows"]
    bad = client.post("/api/sites/apply", json={"model": project["model"], "rows": [{**rows[0], "center_x": "nope"}]})
    assert bad.status_code == 422 and bad.json()["site_index"] == 0 and bad.json()["field"] == "center_x"
    preview = client.post("/api/preview", json={"model": project["model"], "size": 16}).json()
    assert set(preview) == {"unit_cell", "motif_array", "lattice_sites", "reciprocal_bz", "epsilon"}
    assert len(preview["epsilon"]["epsilon"]) == 16 and "bz_vertices" in preview["reciprocal_bz"]


def test_study_crud_round_trip(client):
    project = client.post("/api/documents/new").json()["project"]
    added = client.post("/api/calculations/add", json={"project": project, "operation": "berry"}).json()
    assert added["calculation"]["name"] == "Berry Curvature" and added["project"]["selected_node"] == {"kind": "study", "id": added["calculation"]["id"]}
    renamed = client.post("/api/calculations/rename", json={"project": added["project"], "calculation_id": added["calculation"]["id"], "name": "B"}).json()["project"]
    assert renamed["calculations"][-1]["name"] == "B"
    deleted = client.post("/api/calculations/delete", json={"project": renamed, "calculation_id": added["calculation"]["id"]}).json()["project"]
    assert len(deleted["calculations"]) == 1
    last = client.post("/api/calculations/delete", json={"project": deleted, "calculation_id": deleted["calculations"][0]["id"]})
    assert last.status_code == 422


def test_presets_save_list_apply(client):
    project = client.post("/api/documents/new").json()["project"]
    project["model"]["geometry"]["epsilon_background"] = 9.0
    client.post("/api/presets/save", json={"name": "P1", "model": project["model"], "calculation": project["calculations"][0]["parameters"]})
    assert [item["name"] for item in client.get("/api/presets").json()] == ["P1"]
    fresh = client.post("/api/documents/new").json()["project"]
    applied = client.post("/api/presets/apply", params={"preset": "P1.legumephc-preset.json"}, json={"project": fresh}).json()["project"]
    assert applied["model"]["geometry"]["epsilon_background"] == 9.0
    assert client.post("/api/presets/apply", params={"preset": "../x"}, json={"project": fresh}).status_code == 404


@pytest.mark.integration
def test_unsaved_document_computes_and_streams_events(client):
    doc = client.post("/api/documents/new").json()
    project = doc["project"]
    project["calculations"][0]["parameters"]["gmax"] = 1.5
    started = client.post(f"/api/documents/{doc['doc_id']}/runs", json={"project": project})
    assert started.status_code == 200  # no save needed before computing
    events = _events(client, started.json()["run_id"])
    names = [event["event"] for event in events]
    assert names[0] == "started" and names[-1] == "finished" and "completed" in names
    assert names.index("record_written") < names.index("completed")
    entry = next(event for event in events if event["event"] == "completed")["result_entry"]
    assert entry["record_reference"]["identity"]["operation"] == "frequency_at_k"
    attached = client.post(f"/api/documents/{doc['doc_id']}/attach", json={"project": project, "entry": entry}).json()["project"]
    saved = client.post(f"/api/documents/{doc['doc_id']}/save", json={"path": "Computed", "project": attached}).json()
    reopened = client.post("/api/documents/open", json={"path": saved["path"]}).json()
    assert reopened["project"]["results"][0]["record_reference"] == entry["record_reference"]


@pytest.mark.integration
def test_run_cancel_terminates_worker(client):
    doc = client.post("/api/documents/new").json()
    project = doc["project"]
    project["calculations"][0]["parameters"].update({"operation": "band_structure", "gmax": 6, "numeig": 6, "samples_per_segment": 200})
    project["calculations"][0]["operation"] = "band_structure"
    run_id = client.post(f"/api/documents/{doc['doc_id']}/runs", json={"project": project}).json()["run_id"]
    assert client.post(f"/api/documents/{doc['doc_id']}/runs", json={"project": project}).status_code == 409
    assert client.delete(f"/api/runs/{run_id}").json()["state"] == "cancelled"
    assert _events(client, run_id)[-1]["event"] == "cancelled"


def test_static_frontend_is_served_when_built(client):
    response = client.get("/")
    built = (Path(ROOT) / "src" / "legumephc" / "server" / "static" / "index.html").is_file()
    assert response.status_code == 200
    assert ("<div id=\"root\">" in response.text) if built else ("not built" in response.text)
