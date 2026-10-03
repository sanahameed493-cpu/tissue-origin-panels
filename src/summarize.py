"""Summarise the experiment runs: metrics per run, averages over seeds, and figures.

Metrics follow the study plan (Section 5):
- Macro F1 on each cohort, averaged over classes with at least 10 patients in that cohort.
- The gap = TCGA held-out macro F1 - external macro F1. For a fair comparison, the TCGA
  value is computed over the same classes as the external cohort.
- Accuracy, balanced accuracy, coverage (top probability >= 0.5) and accuracy among
  covered samples. Main results always include every sample.

Confidence intervals and the formal hypothesis tests come in step 4.

Usage:
    python -m src.summarize [--quick]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402

from .experiment import RESULTS  # noqa: E402

MIN_CLASS = 10
COHORTS = ["tcga_heldout", "cptac", "met500"]


def eligible(truth):
    counts = pd.Series(truth).value_counts()
    return sorted(counts[counts >= MIN_CLASS].index)


def cohort_metrics(block, labels):
    t, p = np.array(block["true"]), np.array(block["pred"])
    prob = np.array(block["prob"])
    mask = np.isin(t, labels)
    covered = prob >= 0.5
    return {
        "macro_f1": f1_score(t, p, labels=labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(t[mask], p[mask]),
        "balanced_accuracy": recall_score(t[mask], p[mask], labels=labels,
                                          average="macro", zero_division=0),
        "coverage": float(covered[mask].mean()),
        "accuracy_covered": accuracy_score(t[mask & covered], p[mask & covered])
        if (mask & covered).any() else np.nan,
    }


def summarise_runs(run_dir):
    rows, per_class = [], []
    for path in sorted(run_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        base = {k: rec[k] for k in ("seed", "feature", "size", "C", "n_genes")}
        row = dict(base)
        held = rec["tcga_heldout"]
        row["tcga_macro_f1_all"] = f1_score(held["true"], held["pred"], average="macro",
                                            zero_division=0)
        for cohort in ("cptac", "met500"):
            labels = eligible(rec[cohort]["true"])
            ext = cohort_metrics(rec[cohort], labels)
            internal = cohort_metrics(held, labels)
            row[f"{cohort}_macro_f1"] = ext["macro_f1"]
            row[f"tcga_matched_{cohort}_macro_f1"] = internal["macro_f1"]
            row[f"gap_{cohort}"] = internal["macro_f1"] - ext["macro_f1"]
            for m in ("accuracy", "balanced_accuracy", "coverage", "accuracy_covered"):
                row[f"{cohort}_{m}"] = ext[m]
        for cohort in COHORTS:
            t, p = np.array(rec[cohort]["true"]), np.array(rec[cohort]["pred"])
            for cls in sorted(set(t)):
                m = t == cls
                per_class.append({**base, "cohort": cohort, "class": cls, "n": int(m.sum()),
                                  "sensitivity": float((p[m] == cls).mean())})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_class)


def size_order(s):
    return 100000 if s == "all" else int(s)


def plot_curves(mean, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    xs_all = sorted(mean["size"].unique(), key=size_order)
    pos = {s: i for i, s in enumerate(xs_all)}
    colors = {"tcga_macro_f1_all": "#3182bd", "cptac_macro_f1": "#e6550d",
              "met500_macro_f1": "#31a354"}
    names = {"tcga_macro_f1_all": "TCGA held-out", "cptac_macro_f1": "CPTAC",
             "met500_macro_f1": "MET500"}
    for feature, style in (("expression", "-"), ("pairs", "--")):
        sub = mean[mean["feature"] == feature].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("size", key=lambda s: s.map(size_order))
        x = sub["size"].map(pos)
        for col, c in colors.items():
            axes[0].errorbar(x, sub[(col, "mean")], yerr=sub[(col, "std")], color=c,
                             ls=style, marker="o", ms=4, capsize=2,
                             label=f"{names[col]}, {feature}")
        for col, c in (("gap_cptac", "#e6550d"), ("gap_met500", "#31a354")):
            axes[1].errorbar(x, sub[(col, "mean")], yerr=sub[(col, "std")], color=c,
                             ls=style, marker="o", ms=4, capsize=2,
                             label=f"{col.split('_')[1].upper()}, {feature}")
    for ax in axes:
        ax.set_xticks(range(len(xs_all)))
        ax.set_xticklabels([str(s) for s in xs_all])
        ax.set_xlabel("Genes in panel")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Accuracy vs panel size (solid: expression, dashed: gene pairs)")
    axes[1].set_ylabel("Gap: TCGA minus external macro F1")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Internal-external gap vs panel size")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    base = RESULTS.parent / "quick" if args.quick else RESULTS
    runs, per_class = summarise_runs(base / "runs")
    if runs.empty:
        print("No runs found yet. Run python -m src.experiment first.")
        return
    runs["size"] = runs["size"].astype(str)
    per_class["size"] = per_class["size"].astype(str)
    runs.to_csv(base / "summary_runs.csv", index=False)
    per_class.to_csv(base / "per_class_sensitivity.csv", index=False)

    cols = ["tcga_macro_f1_all", "cptac_macro_f1", "met500_macro_f1",
            "gap_cptac", "gap_met500", "cptac_coverage", "met500_coverage"]
    mean = runs.groupby(["feature", "size"])[cols].agg(["mean", "std"]).reset_index()
    mean.to_csv(base / "summary_by_size.csv", index=False)
    plot_curves(mean, base / "figure3_panel_size.png")

    show = runs.groupby(["feature", "size"])[cols[:5]].mean().round(3)
    show = show.reindex(sorted(show.index, key=lambda i: (i[0], size_order(i[1]))))
    print("Average over seeds (macro F1 and gaps):")
    print(show.to_string())
    print(f"\nSaved tables and figure3_panel_size.png in {base}")


if __name__ == "__main__":
    main()
