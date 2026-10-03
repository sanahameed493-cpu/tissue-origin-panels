"""Step 5 tests on synthetic cohorts and fake raw files."""
import gzip

import numpy as np
import pandas as pd

from src import experiment, prepare, secondary
from tests.test_prepare import setup_fake

CLS = ["LUAD", "LUSC", "BRCA", "PRAD"]


def cohort(n_per, rng, shift=0.0, classes=CLS, n_genes=120):
    X, y = [], []
    for c, cls in enumerate(classes):
        b = rng.normal(3, 1, (n_per, n_genes))
        b[:, c * 4:(c + 1) * 4] += 3
        X.append(b + shift)
        y += [cls] * n_per
    n = len(y)
    return {"X": np.clip(np.vstack(X), 0, None).astype(np.float32), "labels": np.array(y),
            "samples": np.array([f"s{i}" for i in range(n)]),
            "genes": np.array([f"ENSG{i:011d}" for i in range(n_genes)]),
            "biopsy_site": np.array(["liver"] * n), "tumor_content": np.full(n, 0.5)}


def fake(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    data = {"tcga": cohort(40, rng), "cptac": cohort(15, rng, 0.5),
            "met500": cohort(12, rng, 1.0, ["BRCA", "PRAD"]),
            "met500_lung": cohort(12, rng, 1.0, ["LUNG"]),
            "tcga_metastatic": cohort(12, rng, 0.3, ["BRCA", "PRAD"])}
    data["met500_lung"]["X"][:, 0:8] += 1.5          # lung-like profile
    monkeypatch.setattr(prepare, "load", lambda n: data[n])
    monkeypatch.setattr(secondary, "RESULTS", tmp_path / "results" / "secondary")
    monkeypatch.setattr(secondary, "PRIMARY", tmp_path / "results" / "primary")
    for part in secondary.PARTS.values():
        part["seeds"] = [0]
        part["sizes"] = [8, 16]
        part["folds"] = 2
    return data


def test_metastatic_barcodes():
    s = ["TCGA-AA-0001-06A", "TCGA-AA-0001-06B", "TCGA-AA-0002-01A"]
    assert secondary.tcga_metastatic_patients(s) == ["TCGA-AA-0001-06A"]


def test_lr_weight_selection_returns_k_markers():
    rng = np.random.default_rng(3)
    d = cohort(30, rng)
    genes = secondary.select_genes_lr(d["X"], d["labels"], 16)
    assert len(genes) == 16 and len(set(genes) & set(range(16))) >= 12


def test_all_parts_and_summary(monkeypatch, tmp_path):
    fake(monkeypatch, tmp_path)
    for part in ("extra", "models", "selection"):
        secondary.run_part(part)
    secondary.run_part("models")                      # resumes: nothing to redo
    secondary.summary()
    out = tmp_path / "results" / "secondary" / "summary"
    report = (out / "report.txt").read_text()
    assert "ORGAN LEVEL" in report and "TCGA METASTATIC" in report
    assert {"RF", "SVM", "LR_WEIGHTS"} <= {w for w in report.split() if w.isupper()} | set()
    org = pd.read_csv(out / "organ_level_runs.csv")
    assert org["met500_lung_sensitivity"].min() >= 0
    assert (out / "figure_models_cptac.png").exists()


def test_prepare_extra(tmp_path, monkeypatch):
    setup_fake(tmp_path, monkeypatch)
    monkeypatch.setattr(secondary, "RAW", tmp_path / "raw")
    prepare.main()
    # add a metastatic sample to one fake TCGA file
    path = tmp_path / "raw" / "tcga" / "TCGA-SKCM.star_tpm.tsv.gz"
    df = pd.read_csv(path, sep="\t", index_col=0)
    df["TCGA-CM-0009-06A"] = df.iloc[:, 0]
    with gzip.open(path, "wt") as f:
        df.to_csv(f, sep="\t")
    secondary.prepare_extra()
    met = prepare.load("tcga_metastatic")
    assert list(met["labels"]) == ["SKCM"] and met["X"].shape[1] == 25
    assert prepare.MET500_CODES["BRCA"] == "BRCA"     # original rules restored


def test_extra_reproduces_primary_models(monkeypatch, tmp_path):
    data = fake(monkeypatch, tmp_path)
    for k, feature in ((8, "expression"), (8, "pairs")):
        prim = experiment.run_one(data, 0, k, feature, 3, experiment.C_GRID,
                                  tmp_path / f"p_{feature}.json")
        sec = secondary.run_secondary(data, 0, k, feature, "lr", "anova", 3,
                                      tmp_path / f"s_{feature}.json", ["cptac", "met500"])
        assert prim["C"] == sec["params"]["C"] and prim["genes"] == sec["genes"]
        for c in ("tcga_heldout", "cptac", "met500"):
            assert prim[c]["pred"] == sec[c]["pred"]
