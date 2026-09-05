import json
from pathlib import Path

from legumephc.reporting import closed_basis_metrics, load_summary, render_closed_basis_numeric_appendix, validate_closed_basis_report


ROOT = Path(__file__).resolve().parents[1]


def test_closed_basis_report_numbers_are_summary_derived():
    compact = json.loads((ROOT / "evidence" / "closed_basis_berry_20260905_summary.json").read_text(encoding="utf-8"))
    metrics = compact["cases"]
    source = ROOT / compact["source_record"]
    if source.exists():
        assert closed_basis_metrics(load_summary(source)) == metrics
    report = (ROOT / "evidence" / "closed_basis_berry_20260905.md").read_text(encoding="utf-8")
    appendix = render_closed_basis_numeric_appendix(metrics, source_record=compact["source_record"])
    assert appendix in report
    assert validate_closed_basis_report(report, metrics) == []
