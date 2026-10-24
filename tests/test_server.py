"""HTTP contract of the localhost web Studio backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from legumephc.records import create_record
from legumephc.server import create_app
from legumephc.server.app import ROOT
from legumephc.studio.project import new_project, project_records_dir, record_reference


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(tmp_path / "projects")) as value:
        yield value


def _events(client: TestClient, run_id: str) -> list[dict]:
    events = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        for line in stream.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:]))
    return events


def test_meta_and_new_project(client):
    meta = client.get("/api/meta").json()
    assert set(meta["operations"]) == {"frequency_at_k", "band_structure", "fields_energy", "efs", "berry", "berry_curvature_dipole"}
    created = client.post("/api/projects/new", params={"name": "A"}).json()
    assert created["path"] is None and created["project"]["name"] == "A"


def test_save_open_list_and_path_containment(client, tmp_path):
    project = client.post("/api/projects/new").json()["project"]
    saved = client.post("/api/projects/save", json={"path": "nested/P", "project": project}).json()
    assert saved["path"] == "nested/P.legumephc-studio.json"
    assert (tmp_path / "projects" / "nested" / "P.legumephc-studio.json").is_file()
    assert [item["path"] for item in client.get("/api/projects").json()] == [saved["path"]]
    assert client.get("/api/projects/open", params={"path": saved["path"]}).json()["project"]["name"] == project["name"]
    escape = client.post("/api/projects/save", json={"path": "../outside", "project": project})
    assert escape.status_code == 403
    invalid = dict(project, calculations=[])
    assert client.post("/api/projects/save", json={"path": "Bad", "project": invalid}).status_code == 422


def test_sites_errors_are_structured_and_preview_is_json(client):
    project = client.post("/api/projects/new").json()["project"]
    rows = client.post("/api/sites/rows", json={"model": project["model"]}).json()["rows"]
    bad = client.post("/api/sites/apply", json={"model": project["model"], "rows": [{**rows[0], "center_x": "nope"}]})
    assert bad.status_code == 422 and bad.json()["site_index"] == 0 and bad.json()["field"] == "center_x"
    preview = client.post("/api/preview", json={"model": project["model"], "size": 16}).json()
    assert set(preview) == {"unit_cell", "motif_array", "lattice_sites", "reciprocal_bz", "epsilon"}
    assert len(preview["epsilon"]["epsilon"]) == 16 and "bz_vertices" in preview["reciprocal_bz"]


def test_calculation_crud_round_trip(client):
    project = client.post("/api/projects/new").json()["project"]
    added = client.post("/api/calculations/add", json={"project": project, "operation": "berry"}).json()
    assert added["calculation"]["operation"] == "berry" and added["project"]["selected_node"]["id"] == added["calculation"]["id"]
    renamed = client.post("/api/calculations/rename", json={"project": added["project"], "calculation_id": added["calculation"]["id"], "name": "B"}).json()["project"]
    assert renamed["calculations"][-1]["name"] == "B"
    deleted = client.post("/api/calculations/delete", json={"project": renamed, "calculation_id": added["calculation"]["id"]}).json()["project"]
    assert len(deleted["calculations"]) == 1
    last = client.post("/api/calculations/delete", json={"project": deleted, "calculation_id": deleted["calculations"][0]["id"]})
    assert last.status_code == 422


def test_record_view_and_export_use_project_records(client, tmp_path):
    project = new_project("R")
    path = tmp_path / "projects" / "R.legumephc-studio.json"
    saved = client.post("/api/projects/save", json={"path": "R", "project": dict(project)}).json()
    records = project_records_dir(path)
    import numpy as np
    record = create_record(records, identity={"model": "Model2D", "geometry": "g", "affine": {}, "basis": "native", "solver": "legumephc.public-api", "operation": "band_structure"},
                           config={"case": project["model"], "calculation": project["calculations"][0]["parameters"]},
                           summary={"status": "succeeded", "path_labels": ["Gamma", "K", "M", "Gamma"]},
                           arrays={"frequencies": np.array([[0.0, 0.4], [0.2, 0.5], [0.25, 0.45], [0.0, 0.4]]), "qpoints": np.zeros((4, 2)), "vectors": np.ones((4, 2), dtype=complex)})
    reference = record_reference(record, path.parent)
    stored = saved["project"]
    stored["results"].append({"id": "result-1", "calculation_id": "calc-1", "record_reference": reference, "model_snapshot": stored["model"], "calculation_snapshot": {**stored["calculations"][0]["parameters"], "operation": "band_structure"}})
    stored["selected_result"] = reference["path"]
    client.post("/api/projects/save", json={"path": saved["path"], "project": stored})
    view = client.post("/api/records/view", json={"path": saved["path"], "record": reference["path"], "style": {"frequency_unit": "THz"}}).json()
    assert view["operation"] == "band_structure" and len(view["rows"]) == 8
    assert "vectors" not in view["arrays"]  # raw complex eigenvectors never leave the server
    assert view["frequency_factor"] == pytest.approx(299792458.0 / 400e-9 / 1e12)
    exported = client.post("/api/records/export", json={"path": saved["path"], "record": reference["path"], "style": dict(project["plot"]), "fmt": "png"})
    assert exported.status_code == 200 and exported.content[:8] == b"\x89PNG\r\n\x1a\n"
    missing = client.post("/api/records/view", json={"path": saved["path"], "record": "data/other"})
    assert missing.status_code == 404


def test_presets_save_list_apply(client):
    project = client.post("/api/projects/new").json()["project"]
    project["model"]["geometry"]["epsilon_background"] = 9.0
    client.post("/api/presets/save", json={"name": "P1", "model": project["model"], "calculation": project["calculations"][0]["parameters"]})
    assert [item["name"] for item in client.get("/api/presets").json()] == ["P1"]
    fresh = client.post("/api/projects/new").json()["project"]
    applied = client.post("/api/presets/apply", params={"preset": "P1.legumephc-preset.json"}, json={"project": fresh}).json()["project"]
    assert applied["model"]["geometry"]["epsilon_background"] == 9.0
    assert client.post("/api/presets/apply", params={"preset": "../x"}, json={"project": fresh}).status_code == 404


@pytest.mark.integration
def test_frequency_run_streams_events_and_returns_result_entry(client):
    project = client.post("/api/projects/new").json()["project"]
    project["calculations"][0]["parameters"]["gmax"] = 1.5
    started = client.post("/api/runs", json={"path": "Run", "project": project})
    assert started.status_code == 200
    events = _events(client, started.json()["run_id"])
    names = [event["event"] for event in events]
    assert names[0] == "started" and names[-1] == "finished" and "completed" in names
    assert names.index("record_written") < names.index("completed")
    completed = next(event for event in events if event["event"] == "completed")
    entry = completed["result_entry"]
    assert entry["record_reference"]["identity"]["operation"] == "frequency_at_k"
    assert entry["calculation_snapshot"]["operation"] == "frequency_at_k"
    assert events[-1]["state"] == "succeeded"


@pytest.mark.integration
def test_run_cancel_terminates_worker(client):
    project = client.post("/api/projects/new").json()["project"]
    parameters = project["calculations"][0]["parameters"]
    parameters.update({"operation": "band_structure", "gmax": 6, "numeig": 6, "samples_per_segment": 200})
    project["calculations"][0]["operation"] = "band_structure"
    run_id = client.post("/api/runs", json={"path": "Cancel", "project": project}).json()["run_id"]
    assert client.post("/api/runs", json={"path": "Cancel", "project": project}).status_code == 409
    assert client.delete(f"/api/runs/{run_id}").json()["state"] == "cancelled"
    events = _events(client, run_id)
    assert events[-1]["event"] == "cancelled"
    assert client.get(f"/api/runs/{run_id}").json()["state"] == "cancelled"


def test_static_frontend_is_served_when_built(client):
    response = client.get("/")
    built = (Path(ROOT) / "src" / "legumephc" / "server" / "static" / "index.html").is_file()
    assert response.status_code == 200
    assert ("<div id=\"root\">" in response.text) if built else ("not built" in response.text)
