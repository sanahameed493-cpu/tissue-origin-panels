"""Step 4 tests: fast macro F1 matches scikit-learn; full report runs on synthetic results."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import experiment, stats
from tests.test_experiment import fake_data


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    t, p = rng.integers(0, 5, 300), rng.integers(0, 6, 300)   # prediction class 5 never true
    labels = np.array([0, 1, 2, 3])
    ours = stats.macro_f1(t, p, labels, 6)
    ref = f1_score(t, p, labels=labels, average="macro", zero_division=0)
    assert abs(ours - ref) < 1e-12


def test_slope_sign():
    assert stats.slope([10, 20, 40], [0.3, 0.2, 0.1]) < 0


def test_full_report(monkeypatch, tmp_path):
    fake_data(monkeypatch, tmp_path)
    monkeypatch.setattr(stats, "RESULTS", tmp_path / "results" / "primary")
    monkeypatch.setattr(experiment, "SIZES", [10, 20, 50, 100, "all"])
    monkeypatch.setattr(experiment, "PAIR_MAX", 100)
    experiment.main(["--seeds", "0", "1"])
    stats.main(n_boot=40)
    out = tmp_path / "results" / "primary" / "stats"
    h2 = pd.read_csv(out / "H2_prespecified.csv")
    assert len(h2) == 6 and set(h2["size"].astype(str)) == {"10", "20", "50"}
    h1 = pd.read_csv(out / "H1_prespecified.csv")
    assert h1.loc[0, "ci_low"] <= h1.loc[0, "slope_per_doubling"] <= h1.loc[0, "ci_high"]
    est = pd.read_csv(out / "estimates_with_ci.csv")
    assert (est["ext_ci_low"] <= est["external_macro_f1"] + 1e-9).all()
    assert (out / "figure_with_ci.png").exists() and (out / "report.txt").exists()
    assert "PRE-SPECIFIED TEST H1" in (out / "report.txt").read_text()
