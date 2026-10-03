"""Step 3: the primary experiment from the study plan.

For each random seed (5), each panel size, and each feature type
(expression values vs gene pairs):
  1. Split TCGA 80/20 (development / held-out), stratified by class.
  2. Within the development set, 5-fold cross-validation chooses the logistic
     regression strength C. Gene filtering and ANOVA selection are redone inside
     every fold, so no held-out or external sample influences any choice.
  3. Refit on the whole development set with the chosen C.
  4. Predict the TCGA held-out set, CPTAC and MET500 (models are never changed
     after seeing these).

Each run is saved as its own file in results/primary/runs/, so an interrupted
run continues where it stopped. Summaries are made by src.summarize.

Usage:
    python -m src.experiment --quick     # a few minutes: checks everything works
    python -m src.experiment             # the full primary analysis (several hours)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import prepare
from .common import ROOT

RESULTS = ROOT / "results" / "primary"
SIZES = [10, 20, 50, 100, 150, 250, 500, "all"]
PAIR_MAX = 150            # gene-pair features only up to 150 genes (study plan, Section 4)
C_GRID = [0.01, 0.1, 1.0]
SEEDS = [0, 1, 2, 3, 4]
MIN_MEDIAN = 1.0          # drop genes with median log2(TPM + 1) below 1 in the training fold


# ---------- gene selection (always fitted on training rows only) ----------

def select_genes(X, y, k):
    """Filter low-expressed genes, then rank by ANOVA F. Returns column indices."""
    keep = np.flatnonzero(np.median(X, axis=0) >= MIN_MEDIAN)
    if k == "all":
        return keep
    F, _ = f_classif(X[:, keep], y)
    F = np.nan_to_num(F, nan=0.0)
    return keep[np.argsort(-F, kind="stable")[:k]]


# ---------- the two feature types ----------

class ExpressionFeatures:
    """Selected genes, standardised with the training mean and SD."""

    def fit(self, X, genes):
        self.genes = genes
        self.mean = X[:, genes].mean(axis=0)
        self.sd = X[:, genes].std(axis=0) + 1e-6
        return self

    def transform(self, X):
        return (X[:, self.genes] - self.mean) / self.sd


class PairFeatures:
    """For every pair of selected genes: 1 if the first is higher than the second
    in that sample, else 0. Uses only the sample itself, so no scaling is needed."""

    def fit(self, X, genes):
        self.genes = genes
        self.i, self.j = np.triu_indices(len(genes), k=1)
        return self

    def transform(self, X):
        sub = X[:, self.genes]
        return (sub[:, self.i] > sub[:, self.j]).astype(np.float32)


FEATURES = {"expression": ExpressionFeatures, "pairs": PairFeatures}


def make_model(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


def fit_predict(Xtr, ytr, Xtests, k, feature, C):
    genes = select_genes(Xtr, ytr, k)
    feat = FEATURES[feature]().fit(Xtr, genes)
    model = make_model(C).fit(feat.transform(Xtr), ytr)
    outs = []
    for Xte in Xtests:
        proba = model.predict_proba(feat.transform(Xte))
        outs.append((model.classes_[proba.argmax(axis=1)], proba.max(axis=1)))
    return genes, outs


def choose_C(X, y, k, feature, seed, n_folds, c_grid):
    """Inner cross-validation on the development set only."""
    scores = {C: [] for C in c_grid}
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, va in folds.split(X, y):
        genes = select_genes(X[tr], y[tr], k)          # redone inside each fold
        feat = FEATURES[feature]().fit(X[tr], genes)
        Ftr, Fva = feat.transform(X[tr]), feat.transform(X[va])
        for C in c_grid:
            pred = make_model(C).fit(Ftr, y[tr]).predict(Fva)
            scores[C].append(f1_score(y[va], pred, average="macro"))
    means = {C: float(np.mean(v)) for C, v in scores.items()}
    return max(means, key=means.get), means


def run_one(data, seed, k, feature, n_folds, c_grid, out_path):
    tcga = data["tcga"]
    X, y = tcga["X"], tcga["labels"]
    dev, held = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                                 random_state=seed)
    t0 = time.time()
    best_C, cv_scores = choose_C(X[dev], y[dev], k, feature, seed, n_folds, c_grid)
    tests = [X[held], data["cptac"]["X"], data["met500"]["X"]]
    genes, outs = fit_predict(X[dev], y[dev], tests, k, feature, best_C)

    record = {
        "seed": seed, "size": k, "feature": feature, "C": best_C, "cv_macro_f1": cv_scores,
        "n_genes": int(len(genes)), "genes": data["tcga"]["genes"][genes].tolist(),
        "minutes": round((time.time() - t0) / 60, 2),
    }
    for name, truth, (pred, prob) in zip(
            ["tcga_heldout", "cptac", "met500"],
            [y[held], data["cptac"]["labels"], data["met500"]["labels"]], outs):
        record[name] = {"true": [str(t) for t in truth], "pred": [str(p) for p in pred],
                        "prob": np.round(prob, 4).tolist()}
    record["met500"]["biopsy_site"] = data["met500"]["biopsy_site"].tolist()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(out_path)
    return record


def plan_runs(seeds, sizes):
    runs = []
    for seed in seeds:
        for feature in FEATURES:
            for k in sizes:
                if feature == "pairs" and (k == "all" or k > PAIR_MAX):
                    continue
                runs.append((seed, k, feature))
    return runs


def subsample(d, n, seed):
    """Quick mode only: keep a stratified subset of a cohort to run fast."""
    if len(d["labels"]) <= n:
        return d
    idx, _ = train_test_split(np.arange(len(d["labels"])), train_size=n,
                              stratify=d["labels"], random_state=seed)
    return {k: (v[idx] if k in ("X", "labels", "samples", "biopsy_site", "tumor_content")
                else v) for k, v in d.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny check run: 1 seed, sizes 10 and 50, small subsets")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args(argv)

    print("Loading prepared data ...")
    data = {name: prepare.load(name) for name in ("tcga", "cptac", "met500")}

    if args.quick:
        out_dir = RESULTS.parent / "quick" / "runs"
        data = {"tcga": subsample(data["tcga"], 1500, 0),
                "cptac": subsample(data["cptac"], 400, 0),
                "met500": data["met500"]}
        runs, n_folds, c_grid = plan_runs([0], [10, 50]), 3, [0.1]
    else:
        out_dir = RESULTS / "runs"
        runs, n_folds, c_grid = plan_runs(args.seeds, SIZES), 5, C_GRID

    todo = [r for r in runs if not (out_dir / f"seed{r[0]}_{r[2]}_{r[1]}.json").exists()]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to go.")
    for n, (seed, k, feature) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] seed {seed}, {feature}, {k} genes ...", flush=True)
        rec = run_one(data, seed, k, feature, n_folds, c_grid,
                      out_dir / f"seed{seed}_{feature}_{k}.json")
        held = rec["tcga_heldout"]
        f1 = f1_score(held["true"], held["pred"], average="macro")
        print(f"   done in {rec['minutes']} min. C = {rec['C']}, "
              f"TCGA held-out macro F1 = {f1:.3f}", flush=True)
    print(f"\nAll runs finished. Results are in {out_dir}")
    print("Next: python -m src.summarize" + (" --quick" if args.quick else ""))


if __name__ == "__main__":
    main()
