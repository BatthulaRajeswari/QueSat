import numpy as np
import pytest

from src.classical_model import run_classical
from src.data_loader import FEATURES, generate_demo_data
from src.evaluation import comparison_table, compute_metrics, neutral_summary
from src.preprocessing import preprocess


@pytest.fixture(scope="module")
def prepared():
    return preprocess(generate_demo_data("Krishna", "Water-body Change"), FEATURES)


def test_metrics_known_values():
    m = compute_metrics([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_metrics_zero_division_safe():
    m = compute_metrics([0, 0, 1], [0, 0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_classical_svm_runs_and_is_better_than_chance(prepared):
    r = run_classical(prepared)
    assert r["status"] == "ok"
    assert len(r["y_pred"]) == len(prepared.y_test) == len(r["decision"])
    assert r["metrics"]["accuracy"] > 0.7
    cm = np.array(r["metrics"]["confusion_matrix"])
    assert cm.sum() == len(prepared.y_test)


def test_comparison_table_marks_unavailable(prepared):
    r = run_classical(prepared)
    t = comparison_table(r, {"status": "error"})
    assert t.loc[0, "Status"] == "ok" and t.loc[1, "Status"] == "unavailable"
    assert np.isnan(t.loc[1, "Accuracy"])


def test_summary_never_declares_winner(prepared):
    r = run_classical(prepared)
    q = {"status": "ok", "metrics": dict(r["metrics"], accuracy=0.99)}
    text = neutral_summary(r, q).lower()
    assert "outperform" not in text and "better" not in text and "winner" not in text
    assert "does not support a general" in text
    assert "unavailable" in neutral_summary(r, None)
