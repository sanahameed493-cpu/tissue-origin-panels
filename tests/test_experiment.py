"""Step 3 tests on synthetic cohorts with a known class signal and a batch shift."""
import numpy as np
import pandas as pd

from src import experiment, prepare, summarize

CLASSES = ["A", "B", "C", "D"]


def make_cohort(n_per, rng, shift=0.0, n_genes=200):
    X, y = [], []
    for c, cls in enumerate(CLASSES):
        block = rng.normal(3, 1, (n_per, n_genes))
        block[:, c * 5:(c + 1) * 5] += 3          # 5 marker genes per class
        X.append(block + shift)
        y += [cls] * n_per
    X = np.clip(np.vstack(X), 0, None).astype(np.float32)
    n = len(y)
    return {"X": X, "labels": np.array(y), "samples": np.array([f"s{i}" for i in range(n)]),
            "genes": np.array([f"ENSG{i:011d}" for i in range(n_genes)]),
            "biopsy_site": np.array(["liver"] * n), "tumor_content": np.full(n, 0.5)}


def fake_data(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    data = {"tcga": make_cohort(60, rng), "cptac": make_cohort(20, rng, shift=0.8),
            "met500": make_cohort(12, rng, shift=1.5)}
    monkeypatch.setattr(prepare, "load", lambda name: data[name])
    monkeypatch.setattr(experiment, "RESULTS", tmp_path / "results" / "primary")
    monkeypatch.setattr(summarize, "RESULTS", tmp_path / "results" / "primary")
    return data


def test_pair_features():
    X = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]])
    f = experiment.PairFeatures().fit(X, np.array([0, 1, 2]))
    # pairs (0,1), (0,2), (1,2): is first gene higher than second?
    assert f.transform(X).tolist() == [[0, 0, 1], [1, 1, 0]]


def test_selection_finds_markers_and_filters():
    rng = np.random.default_rng(1)
    d = make_cohort(30, rng)
    d["X"][:, 199] = 0.0                       # unexpressed gene must be filtered
    genes = experiment.select_genes(d["X"], d["labels"], 20)
    assert set(range(20)) == set(genes)        # the 20 marker genes
    assert 199 not in experiment.select_genes(d["X"], d["labels"], "all")


def test_selection_never_sees_heldout_rows(monkeypatch, tmp_path):
    data = fake_data(monkeypatch, tmp_path)
    seen = []
    real = experiment.select_genes
    monkeypatch.setattr(experiment, "select_genes",
                        lambda X, y, k: (seen.append(len(y)), real(X, y, k))[1])
    experiment.run_one(data, 0, 10, "expression", 3, [0.1], tmp_path / "r.json")
    n_dev = int(round(len(data["tcga"]["labels"]) * 0.8))
    assert max(seen) == n_dev and len(seen) == 4   # 3 inner folds + 1 final fit


def test_quick_run_and_summary(monkeypatch, tmp_path):
    fake_data(monkeypatch, tmp_path)
    experiment.main(["--quick"])
    runs = sorted((tmp_path / "results" / "quick" / "runs").glob("*.json"))
    assert len(runs) == 4                       # 2 sizes x 2 feature types
    experiment.main(["--quick"])                # second call resumes: nothing to redo
    summarize.main(["--quick"])
    table = pd.read_csv(tmp_path / "results" / "quick" / "summary_runs.csv")
    assert table["tcga_macro_f1_all"].min() > 0.9
    assert (tmp_path / "results" / "quick" / "figure3_panel_size.png").exists()
    assert {"gap_cptac", "gap_met500", "cptac_coverage"} <= set(table.columns)
