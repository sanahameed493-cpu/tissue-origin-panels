"""Step 5: the secondary analyses listed in the study plan.

Parts (run them one at a time; each saves every run and resumes if stopped):

  prepare    Build two extra test sets: TCGA metastatic samples, and MET500 lung
             samples whose subtype is not given (organ-level analysis only).
  extra      The primary models (logistic regression, ANOVA genes, 5 seeds) rerun
             exactly as in step 3, now also predicting the two extra test sets.
             Used for the organ-level analysis and the TCGA metastases test.
  models     Random forest and SVM instead of logistic regression.
  selection  He et al.-style gene selection (logistic regression weights, equal
             genes per class) instead of ANOVA.
  summary    Tables and figures for all parts.

To keep run time reasonable on a laptop, 'models' and 'selection' use seeds 0-2 and
3-fold inner cross-validation (recorded in the study plan's changes log).

Usage:
    python -m src.secondary prepare
    python -m src.secondary extra
    python -m src.secondary models
    python -m src.secondary selection
    python -m src.secondary summary
    (add --quick to any part for a fast check on small subsets)
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from . import prepare  # noqa: E402
from .common import RAW, TCGA_PROJECTS  # noqa: E402
from .experiment import (C_GRID, FEATURES, MIN_MEDIAN, PAIR_MAX, SEEDS, SIZES,  # noqa: E402
                         select_genes, subsample)
from .experiment import RESULTS as PRIMARY  # noqa: E402
from .inventory import tcga_header  # noqa: E402
from .summarize import MIN_CLASS, eligible, size_order  # noqa: E402

RESULTS = PRIMARY.parent / "secondary"
SECONDARY_SEEDS = [0, 1, 2]
SECONDARY_SIZES = [10, 20, 50, 100, 150, "all"]
ORGAN = {"LUAD": "LUNG", "LUSC": "LUNG"}


# ---------- extra test sets ----------

def tcga_metastatic_patients(samples):
    """TCGA barcodes with sample type '06' (metastatic); one per patient."""
    keep = {}
    for s in samples:
        if len(s) >= 15 and s.startswith("TCGA-") and s[13:15] == "06":
            keep.setdefault(s[:12], s)
    return sorted(keep.values())


def prepare_extra():
    genes = pd.Index(prepare.load("tcga")["genes"])
    frames, labels = [], []
    for cls, projects in TCGA_PROJECTS.items():
        for proj in projects:
            path = RAW / "tcga" / f"{proj}.star_tpm.tsv.gz"
            header = tcga_header(path)
            keep = tcga_metastatic_patients(header[1:])
            if not keep:
                continue
            df = pd.read_csv(path, sep="\t", index_col=0, usecols=[header[0]] + keep)
            df.index = prepare.strip(df.index)
            df = df[~df.index.duplicated() & df.index.isin(genes)].astype(np.float32)
            frames.append(np.exp2(df) - 1.0)
            labels += [cls] * len(keep)
            print(f"  {proj}: {len(keep)} metastatic patients")
    if not frames:
        raise SystemExit("No TCGA metastatic samples found in the downloaded files.")
    tpm = pd.concat(frames, axis=1)
    prepare.save("tcga_metastatic", prepare.to_log2_rescaled(tpm, genes), tpm.columns,
                  labels, genes)

    old = prepare.MET500_CODES
    prepare.MET500_CODES = {"LUNG": "LUNG"}
    try:
        tpm, lab, meta = prepare.read_met500()
    finally:
        prepare.MET500_CODES = old
    prepare.save("met500_lung", prepare.to_log2_rescaled(tpm, genes), tpm.columns, lab, genes,
                 extra={"biopsy_site": meta["biopsy_tissue"].astype(str).to_numpy()})
    print("Extra test sets ready.")


# ---------- alternative gene selection (He et al. style) ----------

def select_genes_lr(X, y, k):
    """Rank genes by multinomial logistic regression weights and take an equal number
    of the most positive genes per class (He et al., 2023 used weights per class)."""
    keep = np.flatnonzero(np.median(X, axis=0) >= MIN_MEDIAN)
    if k == "all":
        return keep
    Z = X[:, keep]
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-6)
    model = LogisticRegression(C=0.1, max_iter=500).fit(Z, y)
    per_class = int(np.ceil(k / len(model.classes_)))
    chosen = []
    for row in model.coef_:
        for g in np.argsort(-row)[:per_class]:
            if g not in chosen:
                chosen.append(g)
    if len(chosen) < k:                       # fill with the largest weights overall
        for g in np.argsort(-np.abs(model.coef_).max(axis=0)):
            if g not in chosen:
                chosen.append(g)
            if len(chosen) == k:
                break
    order = np.argsort(-np.abs(model.coef_).max(axis=0)[chosen], kind="stable")
    return keep[np.array(chosen)[order][:k]]


SELECTORS = {"anova": select_genes, "lr_weights": select_genes_lr}


# ---------- models and their tuning grids ----------

def model_grid(name, n_folds):
    if name == "lr":
        return [{"C": c} for c in C_GRID]
    if name == "rf":
        return [{"max_features": m} for m in ("sqrt", "log2")]
    if name == "svm":
        return [{"C": c, "gamma_factor": g} for c in (1.0, 10.0) for g in (0.3, 1.0)]
    raise ValueError(name)


def build_model(name, params, seed, F):
    if name == "lr":
        return LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_features=params["max_features"],
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed)
    if name == "svm":
        gamma = params["gamma_factor"] / (F.shape[1] * max(float(F.var()), 1e-6))
        return SVC(C=params["C"], gamma=gamma, kernel="rbf", class_weight="balanced")
    raise ValueError(name)


def run_secondary(data, seed, k, feature, model_name, selector, n_folds, out_path, tests):
    X, y = data["tcga"]["X"], data["tcga"]["labels"]
    dev, held = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                                 random_state=seed)
    Xd, yd = X[dev], y[dev]
    select = SELECTORS[selector]
    grid = model_grid(model_name, n_folds)
    t0 = time.time()

    scores = [[] for _ in grid]
    for tr, va in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Xd, yd):
        genes = select(Xd[tr], yd[tr], k)
        feat = FEATURES[feature]().fit(Xd[tr], genes)
        Ftr, Fva = feat.transform(Xd[tr]), feat.transform(Xd[va])
        for i, params in enumerate(grid):
            pred = build_model(model_name, params, seed, Ftr).fit(Ftr, yd[tr]).predict(Fva)
            scores[i].append(f1_score(yd[va], pred, average="macro"))
    best = grid[int(np.argmax([np.mean(s) for s in scores]))]

    genes = select(Xd, yd, k)
    feat = FEATURES[feature]().fit(Xd, genes)
    Fd = feat.transform(Xd)
    model = build_model(model_name, best, seed, Fd).fit(Fd, yd)

    record = {"seed": seed, "size": k, "feature": feature, "model": model_name,
              "selector": selector, "params": best, "n_genes": int(len(genes)),
              "genes": data["tcga"]["genes"][genes].tolist(),
              "minutes": round((time.time() - t0) / 60, 2)}
    sets = {"tcga_heldout": (X[held], y[held])}
    sets.update({name: (data[name]["X"], data[name]["labels"]) for name in tests})
    for name, (Xt, yt) in sets.items():
        pred = model.predict(feat.transform(Xt))
        record[name] = {"true": [str(t) for t in yt], "pred": [str(p) for p in pred]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(out_path)
    return record


PARTS = {
    "extra": dict(models=["lr"], selector="anova", seeds=SEEDS, sizes=SIZES, folds=5,
                  tests=["cptac", "met500", "met500_lung", "tcga_metastatic"]),
    "models": dict(models=["rf", "svm"], selector="anova", seeds=SECONDARY_SEEDS,
                   sizes=SECONDARY_SIZES, folds=3, tests=["cptac", "met500"]),
    "selection": dict(models=["lr"], selector="lr_weights", seeds=SECONDARY_SEEDS,
                      sizes=[s for s in SECONDARY_SIZES if s != "all"], folds=3,
                      tests=["cptac", "met500"]),
}


def run_part(part, quick=False):
    cfg = PARTS[part]
    names = ["tcga"] + cfg["tests"]
    data = {n: prepare.load(n) for n in names}
    out_dir = RESULTS / ("quick_" + part if quick else part) / "runs"
    seeds, sizes, folds = cfg["seeds"], cfg["sizes"], cfg["folds"]
    if quick:
        data["tcga"] = subsample(data["tcga"], 1500, 0)
        data["cptac"] = subsample(data["cptac"], 300, 0)   # small sets are kept whole
        seeds, sizes, folds = [0], [10, 50], 2
    runs = [(s, k, f, m) for s in seeds for m in cfg["models"] for f in FEATURES for k in sizes
            if not (f == "pairs" and (k == "all" or k > PAIR_MAX))]
    todo = [r for r in runs if not (out_dir / f"{r[3]}_seed{r[0]}_{r[2]}_{r[1]}.json").exists()]
    print(f"[{part}] {len(runs)} runs planned, {len(runs) - len(todo)} done, {len(todo)} to go.")
    for n, (seed, k, feature, model_name) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {model_name}, seed {seed}, {feature}, {k} genes ...", flush=True)
        rec = run_secondary(data, seed, k, feature, model_name, cfg["selector"], folds,
                            out_dir / f"{model_name}_seed{seed}_{feature}_{k}.json", cfg["tests"])
        h = rec["tcga_heldout"]
        print(f"   done in {rec['minutes']} min, TCGA held-out macro F1 = "
              f"{f1_score(h['true'], h['pred'], average='macro'):.3f}", flush=True)
    print(f"[{part}] finished. Next part, or: python -m src.secondary summary")


# ---------- summary ----------

def macro(block, labels, mapping=None):
    t = np.array([mapping.get(x, x) if mapping else x for x in block["true"]])
    p = np.array([mapping.get(x, x) if mapping else x for x in block["pred"]])
    return f1_score(t, p, labels=labels, average="macro", zero_division=0)


def summarise_part(part_dir, cohorts):
    rows = []
    for path in sorted((part_dir / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        row = {"model": r.get("model", "lr"), "selector": r.get("selector", "anova"),
               "seed": r["seed"], "feature": r["feature"], "size": r["size"]}
        row["size"] = str(row["size"])
        for c in cohorts:
            labels = eligible(r[c]["true"])
            row[f"{c}_macro_f1"] = macro(r[c], labels)
            row[f"gap_{c}"] = macro(r["tcga_heldout"], labels) - row[f"{c}_macro_f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def organ_level(part_dir):
    """Merge LUAD and LUSC into LUNG (truth and predictions); MET500 adds its LUNG samples."""
    rows = []
    for path in sorted((part_dir / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        met = {"true": r["met500"]["true"] + r["met500_lung"]["true"],
               "pred": r["met500"]["pred"] + r["met500_lung"]["pred"]}
        row = {"seed": r["seed"], "feature": r["feature"], "size": str(r["size"])}
        for c, block in (("cptac", r["cptac"]), ("met500_with_lung", met)):
            truth = [ORGAN.get(x, x) for x in block["true"]]
            labels = eligible(truth)
            row[f"organ_{c}_macro_f1"] = macro(block, labels, ORGAN)
            row[f"organ_gap_{c}"] = macro(r["tcga_heldout"], labels, ORGAN) - row[f"organ_{c}_macro_f1"]
            row[f"organ_{c}_classes"] = len(labels)
        lung_t = np.array(r["met500_lung"]["true"])
        lung_p = np.array([ORGAN.get(x, x) for x in r["met500_lung"]["pred"]])
        row["met500_lung_sensitivity"] = float((lung_p == "LUNG").mean()) if len(lung_t) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def metastatic(part_dir):
    rows = []
    for path in sorted((part_dir / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        t, p = np.array(r["tcga_metastatic"]["true"]), np.array(r["tcga_metastatic"]["pred"])
        for cls in sorted(set(t)):
            m = t == cls
            rows.append({"seed": r["seed"], "feature": r["feature"], "size": str(r["size"]),
                         "class": cls, "n": int(m.sum()), "sensitivity": float((p[m] == cls).mean())})
    return pd.DataFrame(rows)


def mean_table(df, keys, cols):
    out = df.groupby(keys)[cols].mean().round(3)
    return out.reindex(sorted(out.index, key=lambda i: tuple(size_order(x) if x in
                       {str(s) for s in SIZES} else x for x in i)))


def plot_models(tables, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    styles = {"expression": "-", "pairs": "--"}
    colors = {"lr": "#3182bd", "rf": "#e6550d", "svm": "#31a354", "lr_weights": "#756bb1"}
    for name, df in tables.items():
        for feature, sub in df.groupby("feature"):
            sub = sub.groupby("size")["cptac_macro_f1"].mean()
            sizes = sorted(sub.index, key=size_order)
            ax.plot([str(s) for s in sizes], sub[sizes].values, ls=styles[feature], marker="o",
                    ms=4, color=colors[name], label=f"{name}, {feature}")
    ax.set_xlabel("Genes in panel")
    ax.set_ylabel("CPTAC macro F1 (mean over seeds)")
    ax.set_title("External accuracy across models and selection methods")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary(quick=False):
    prefix = "quick_" if quick else ""
    out = RESULTS / (prefix + "summary")
    out.mkdir(parents=True, exist_ok=True)
    lines = ["STEP 5 REPORT: secondary analyses (planned in the study plan)", "=" * 60]
    cols = ["cptac_macro_f1", "met500_macro_f1", "gap_cptac", "gap_met500"]
    tables = {}

    prim = PRIMARY / "runs"
    if prim.exists() and not quick:
        p = summarise_part(PRIMARY, ["cptac", "met500"])
        p = p.assign(model="lr", selector="anova")
        p = p[p["seed"].isin(SECONDARY_SEEDS)]
        tables["lr"] = p
    part_dir = RESULTS / (prefix + "models")
    if (part_dir / "runs").exists():
        df = summarise_part(part_dir, ["cptac", "met500"])
        for m, sub in df.groupby("model"):
            tables[m] = sub
    part_dir = RESULTS / (prefix + "selection")
    if (part_dir / "runs").exists():
        tables["lr_weights"] = summarise_part(part_dir, ["cptac", "met500"])
    for name, df in tables.items():
        df.to_csv(out / f"runs_{name}.csv", index=False)
        lines += [f"\n{name.upper()} (seeds 0-2), mean over seeds:",
                  mean_table(df, ["feature", "size"], cols).to_string()]
    if tables:
        plot_models(tables, out / "figure_models_cptac.png")

    part_dir = RESULTS / (prefix + "extra")
    if (part_dir / "runs").exists():
        org = organ_level(part_dir)
        org.to_csv(out / "organ_level_runs.csv", index=False)
        ocols = [c for c in org.columns if c.startswith(("organ_", "met500_lung")) and
                 not c.endswith("classes")]
        lines += ["\nORGAN LEVEL (LUAD + LUSC = LUNG; MET500 includes its LUNG samples):",
                  mean_table(org, ["feature", "size"], ocols).to_string()]
        met = metastatic(part_dir)
        met.to_csv(out / "tcga_metastatic_runs.csv", index=False)
        mt = met.groupby(["class", "n", "feature", "size"])["sensitivity"].mean().unstack(
            ["feature", "size"]).round(3)
        lines += ["\nTCGA METASTATIC SAMPLES: sensitivity per class (mean over seeds):",
                  mt.to_string()]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report)
    print(report)
    print(f"\nSaved in {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["prepare", "extra", "models", "selection", "summary"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    if a.part == "prepare":
        prepare_extra()
    elif a.part == "summary":
        summary(a.quick)
    else:
        run_part(a.part, a.quick)


if __name__ == "__main__":
    main()
