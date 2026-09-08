import numpy as np
import pytest
from legumephc.motifs import triangular_motifs
from legumephc.geometry import SUBLATTICE_CENTERS
from legumephc.units import frequency_factor, reference_length
from legumephc.studio.project import new_project
from legumephc.studio.profile import model_from_case
from legumephc.studio.plotting import plot_record
from legumephc.records import create_record


def test_honeycomb_relative_positions_and_angles():
    motifs = triangular_motifs((.2, .18), (0, 60), kind="polygon", sides=3)
    centers = np.array([m["center"] for m in motifs])
    np.testing.assert_allclose(centers[1] - centers[0], SUBLATTICE_CENTERS[1] - SUBLATTICE_CENTERS[0])
    np.testing.assert_allclose(centers.mean(axis=0), [.5, 0])
    assert [m["angle_degrees"] for m in motifs] == [0, 60]
    assert len(triangular_motifs(.2)) == 1
    with pytest.raises(ValueError, match="Rotation"):
        triangular_motifs((.2, .18), (0,))


def test_physical_length_does_not_change_solver_model():
    case = new_project()["case"]
    before = model_from_case(case)
    case["actual_lattice_constant_m"] = 900e-9
    after = model_from_case(case)
    np.testing.assert_array_equal(before.effective_lattice.direct_basis, after.effective_lattice.direct_basis)
    assert before.geometry.radii == after.geometry.radii
    assert new_project()["ui_state"]["material_representation"] == "n"


def test_frequency_conversion_roundtrip_and_missing_length():
    length = reference_length(400, "nm")
    f = frequency_factor("THz", length)
    assert .3*f == pytest.approx(224.8443435)
    assert (.3*f)/f == pytest.approx(.3)
    assert reference_length(.4, "μm") == pytest.approx(length)
    assert frequency_factor("GHz", length) == pytest.approx(224844.3435 / .3)
    with pytest.raises(ValueError, match="not set"):
        frequency_factor("Hz", None)


def test_plot_uses_record_length(tmp_path):
    path = create_record(tmp_path, identity={"model":"Model2D", "geometry":"test", "affine":{}, "basis":{}, "solver":"test", "operation":"band_structure"}, config={"case":{"actual_lattice_constant_m":400e-9}}, summary={}, arrays={"frequencies":np.array([[.3],[.4]])})
    fig = plot_record(path, {"frequency_unit":"THz"})
    assert fig.axes[0].lines[0].get_ydata()[0] == pytest.approx(224.8443435)
    assert fig.axes[0].get_ylabel() == "Frequency (THz)"
    raw = plot_record(path)
    assert raw.axes[0].lines[0].get_ydata()[0] == pytest.approx(.3)


def test_result_groups_are_named_by_operation():
    from types import SimpleNamespace
    from legumephc.studio.ui import StudioApp
    app = SimpleNamespace(project={"results": [
        {"id": "result-1", "calculation_snapshot": {"operation": "band_structure"}, "record_reference": {"path": "a"}},
        {"id": "result-2", "calculation_snapshot": {"operation": "berry"}, "record_reference": {"path": "b"}},
        {"id": "result-3", "calculation_snapshot": {"operation": "band_structure"}, "record_reference": {"path": "c"}},
    ]})
    groups = StudioApp._grouped_results(app)
    assert list(groups) == ["band_structure", "berry"]
    assert [item["id"] for item in groups["band_structure"]] == ["result-1", "result-3"]
    assert StudioApp._operation_label("band_structure") == "Band Structure"


def test_studio_honeycomb_controls_and_band_smoke(tmp_path):
    import tkinter as tk
    import pytest
    from legumephc.studio.ui import StudioApp
    from legumephc.studio.worker import execute_request, REQUEST_SCHEMA
    try:
        app = StudioApp()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        app.update()
        from legumephc.studio.sites_editor import SitesDialog
        dialog = SitesDialog(app, motifs=app._case_from_controls()['geometry']['motifs'], representation='n', lattice='triangular', center=[.5,0])
        dialog.add_site()
        for row, radius, angle in zip(dialog.rows, ('.2','.18'), ('0','60')):
            row['vars']['shape'].set('Triangle')
            dialog.shape_changed(row)
            row['vars']['radius'].set(radius)
            row['vars']['angle'].set(angle)
        dialog.apply()
        assert dialog.result is not None
        app.motif_tree.delete(*app.motif_tree.get_children())
        for index,motif in enumerate(dialog.result):
            app._set_motif_row(f'motif-{index+1}',motif,insert=True)
        case = app._case_from_controls()
        assert len(case["geometry"]["motifs"]) == 2
        assert case["geometry"]["motifs"][1]["angle_degrees"] == 60
        app._vars["length_unit"].set("μm")
        app._length_unit_changed()
        assert float(app._vars["actual_length"].get()) == pytest.approx(.4)
        calculation = dict(app.project["calculation"], operation="band_structure", gmax=1, numeig=3, samples_per_segment=2)
        result = execute_request({"schema":REQUEST_SCHEMA,"project_dir":str(tmp_path),"case":case,"calculation":calculation,"records_dir":str(tmp_path)})
        fig = plot_record(result["record_path"], {"frequency_unit":"THz"})
        assert fig.axes[0].get_ylabel() == "Frequency (THz)"
    finally:
        app.destroy()


def test_site_editor_validation_cancel_and_custom_position():
    import tkinter as tk
    import pytest
    from copy import deepcopy
    from legumephc.studio.sites_editor import SitesDialog, number_text
    try:
        root=tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    source=triangular_motifs(.234567890123456)
    source[0]['center']=[.123456789012345,.2]
    before=deepcopy(source)
    try:
        d=SitesDialog(root,motifs=source,representation='n',lattice='triangular',center=[.5,0])
        d.add_site()
        assert d.rows[1]['vars']['x'].get()==''
        d.rows[0]['vars']['radius'].set('0.3,0.2')
        d.apply()
        assert d.result is None and 'Site 1, radius' in d.error.cget('text')
        d.destroy()
        assert source==before
        d=SitesDialog(root,motifs=source,representation='n',lattice='custom',center=[.5,.5])
        d.apply()
        assert d.result[0]['radius']==source[0]['radius']
        assert d.result[0]['center']==source[0]['center']
        assert number_text(399.99999999999994)=='400'
    finally:
        root.destroy()


def test_mixed_sites_affine_and_project_roundtrip(tmp_path):
    import tkinter as tk
    import pytest
    from legumephc.studio.sites_editor import SitesDialog
    from legumephc.studio.project import save_project, load_project
    from legumephc.studio.preview import preview_geometry
    try:
        root=tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    try:
        original=triangular_motifs((.15,.12))
        d=SitesDialog(root,motifs=original,representation='n',lattice='triangular',center=[.5,0])
        d.rows[1]['vars']['shape'].set('Triangle')
        d.shape_changed(d.rows[1])
        d.rows[1]['vars']['angle'].set('60')
        d.apply()
        assert [m['kind'] for m in d.result]==['circle','polygon']
        project=new_project()
        project['case']['geometry']['motifs']=d.result
        project['case']['deformation']={'kind':'custom','linear':[[1.2,.1],[0,.8]],'translation':[.1,0]}
        model=model_from_case(project['case'])
        expected=np.array([m['center'] for m in d.result]) @ np.array([[1.2,.1],[0,.8]]).T + [.1,0]
        np.testing.assert_allclose(model.effective_geometry.centers,expected)
        assert model.effective_geometry.ellipse_parameters[0] is not None
        assert model.effective_geometry.transformed_vertices[1].shape==(3,2)
        preview=preview_geometry(project['case'],view='motif',size=16)
        assert len(preview['motifs'])==2
        path=save_project(tmp_path/'sites.legumephc-studio.json',project)
        loaded=load_project(path)
        assert loaded['case']['geometry']['motifs']==d.result
    finally:
        root.destroy()
