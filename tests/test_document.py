"""Single-file .lphc container safety."""

from __future__ import annotations

import json
import zipfile

import pytest

from legumephc.studio.document import DOCUMENT_FORMAT, DocumentError, new_workspace, open_document, save_document
from legumephc.studio.project import new_project


def _document(path, members):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_round_trip_and_atomic_overwrite(tmp_path):
    workspace = new_workspace(tmp_path / "ws")
    target = tmp_path / "A.lphc"
    save_document(workspace, new_project("A"), target)
    project = new_project("A")
    project["parameters"] = [{"name": "r0", "expression": "0.2"}]
    save_document(workspace, project, target)  # overwrite in place
    assert not list(tmp_path.glob(".A.lphc.*.tmp"))
    reopened = open_document(target, new_workspace(tmp_path / "ws"))
    assert reopened["parameters"] == [{"name": "r0", "expression": "0.2"}]


@pytest.mark.parametrize("member", ["../evil.json", "/abs.json", "data/.studio/records/../../x", "other/file.txt"])
def test_unsafe_or_foreign_members_are_rejected(tmp_path, member):
    path = tmp_path / "bad.lphc"
    _document(path, {"meta.json": json.dumps({"format": DOCUMENT_FORMAT}), "project.json": json.dumps(dict(new_project())), member: "x"})
    with pytest.raises(DocumentError):
        open_document(path, new_workspace(tmp_path / "ws"))
    assert not (tmp_path / "evil.json").exists()


def test_wrong_format_is_rejected(tmp_path):
    path = tmp_path / "other.lphc"
    _document(path, {"meta.json": json.dumps({"format": "something-else"}), "project.json": "{}"})
    with pytest.raises(DocumentError, match="not a LegumePhC document"):
        open_document(path, new_workspace(tmp_path / "ws"))
