"""Step 4: confidence intervals, the pre-specified hypothesis tests, and explanations.

Pre-specified (study plan, Section 5):
  - 95% CIs from 1,000 bootstrap resamples of the test samples. Each resample is
    applied to every seed and both feature types alike (paired), and results are
    averaged over the 5 seeds, as in the main summary.
  - H1: slope of the CPTAC gap against log2(panel size). Supported if negative with a
    95% CI excluding 0. ("all genes" uses the number of genes that passed the filter.)
  - H2: gap(expression) - gap(pairs) at 10, 20 and 50 genes. Supported if positive
    with a 95% CI excluding 0 at two or more of the three sizes.

Additional analyses (not pre-specified; labelled as such in every output):
  - A. External macro F1, pairs minus expression, at each size up to 150 genes.
  - B. H1 slope without the "all genes" point.
  - C. Per-class sensitivity, the most common errors, and MET500 accuracy by
       biopsy site and tumour content, to explain where models fail.

Usage:
    python -m src.stats
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import prepare  # noqa: E402
from .experiment import RESULTS  # noqa: E402
from .summarize import MIN_CLASS, size_order  # noqa: E402

N_BOOT = 1000
H2_SIZES = ["10", "20", "50"]
EXTERNAL = ["cptac", "met500"]


# ---------- fast macro F1 for bootstrapping ----------

def macro_f1(t, p, labels, K):
    """Macro F1 over `labels` (ints), from integer-coded truth and predictions."""
    cm = np.bincount(t * K + p, minlength=K * K).reshape(K, K)
    tp = cm[labels, labels]
    fp = cm[:, labels].sum(axis=0) - tp
    fn = cm[labels, :].sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def load_runs(run_dir):
    runs = []
    for path in sorted(run_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        rec["size"] = str(rec["size"])
        runs.append(rec)
    return runs


def encode(runs):
    """Integer-code every label seen anywhere, and find the eligible classes per cohort."""
    names = sorted({x for r in runs for c in ["tcga_heldout"] + EXTERNAL
                    for x in r[c]["true"] + r[c]["pred"]})
    code = {n: i for i, n in enumerate(names)}
    for r in runs:
        for c in ["tcga_heldout"] + EXTERNAL:
            r[c]["t"] = np.array([code[x] for x in r[c]["true"]])
            r[c]["p"] = np.array([code[x] for x in r[c]["pred"]])
    eligible = {}
    for c in EXTERNAL:
        counts = pd.Series(runs[0][c]["true"]).value_counts()
        eligible[c] = np.array([code[n] for n in sorted(counts[counts >= MIN_CLASS].index)])
    return names, code, eligible


def bootstrap(runs, eligible, K, n_boot, rng):
    """Seed-averaged external macro F1 and gap per (feature, size, cohort), for the
    observed data (b = 0) and n_boot paired resamples (b = 1..n_boot)."""
    seeds = sorted({r["seed"] for r in runs})
    n_held = {s: len(next(r for r in runs if r["seed"] == s)["tcga_heldout"]["t"])
              for s in seeds}
    n_ext = {c: len(runs[0][c]["t"]) for c in EXTERNAL}
    held_idx = {s: [np.arange(n_held[s])] + [rng.integers(0, n_held[s], n_held[s])
                                            for _ in range(n_boot)] for s in seeds}
    ext_idx = {c: [np.arange(n_ext[c])] + [rng.integers(0, n_ext[c], n_ext[c])
                                          for _ in range(n_boot)] for c in EXTERNAL}
    groups = {}
    for r in runs:
        groups.setdefault((r["feature"], r["size"]), []).append(r)
    out = {}
    for (feature, size), rs in groups.items():
        for c in EXTERNAL:
            ext = np.zeros((len(rs), n_boot + 1))
            gap = np.zeros((len(rs), n_boot + 1))
            for i, r in enumerate(rs):
                h, e = r["tcga_heldout"], r[c]
                for b in range(n_boot + 1):
                    hi, ei = held_idx[r["seed"]][b], ext_idx[c][b]
                    f_int = macro_f1(h["t"][hi], h["p"][hi], eligible[c], K)
                    f_ext = macro_f1(e["t"][ei], e["p"][ei], eligible[c], K)
                    ext[i, b], gap[i, b] = f_ext, f_int - f_ext
            out[(feature, size, c)] = {"ext": ext.mean(axis=0), "gap": gap.mean(axis=0),
                                       "n_genes": float(np.mean([r["n_genes"] for r in rs]))}
    return out


def ci(dist):
    return float(dist[0]), float(np.percentile(dist[1:], 2.5)), float(np.percentile(dist[1:], 97.5))


def slope(xs, ys):
    x = np.log2(np.asarray(xs, dtype=float))
    return float(np.polyfit(x, ys, 1)[0])


def h1_test(boot, cohort, feature, sizes):
    keys = [(feature, s, cohort) for s in sizes if (feature, s, cohort) in boot]
    xs = [boot[k]["n_genes"] if k[1] == "all" else float(k[1]) for k in keys]
    gaps = np.array([boot[k]["gap"] for k in keys])           # sizes x (1 + n_boot)
    dist = np.array([slope(xs, gaps[:, b]) for b in range(gaps.shape[1])])
    est, lo, hi = ci(dist)
    return {"cohort": cohort, "feature": feature, "sizes": ",".join(sizes),
            "slope_per_doubling": est, "ci_low": lo, "ci_high": hi,
            "supported": bool(hi < 0)}


def diff_table(boot, cohort, sizes, metric, sign):
    """sign=+1: expression - pairs; sign=-1: pairs - expression."""
    rows = []
    for s in sizes:
        e, p = boot.get(("expression", s, cohort)), boot.get(("pairs", s, cohort))
        if e is None or p is None:
            continue
        est, lo, hi = ci(sign * (e[metric] - p[metric]))
        rows.append({"cohort": cohort, "size": s, "estimate": est, "ci_low": lo,
                     "ci_high": hi, "ci_excludes_0_positive": bool(lo > 0)})
    return pd.DataFrame(rows)


# ---------- additional analysis C: where do models fail? ----------

def per_class(runs, cohort, picks):
    rows = []
    for feature, size in picks:
        rs = [r for r in runs if r["feature"] == feature and r["size"] == size]
        for cls in sorted(set(rs[0][cohort]["true"])):
            sens = [np.mean(np.array(r[cohort]["pred"])[np.array(r[cohort]["true"]) == cls] == cls)
                    for r in rs]
            n = int(np.sum(np.array(rs[0][cohort]["true"]) == cls))
            rows.append({"cohort": cohort, "model": f"{feature} {size}", "class": cls,
                         "n": n, "sensitivity": float(np.mean(sens))})
    return pd.DataFrame(rows).pivot_table(index=["class", "n"], columns="model",
                                          values="sensitivity").round(3)


def top_errors(runs, cohort, feature, size, n=8):
    rs = [r for r in runs if r["feature"] == feature and r["size"] == size]
    pairs = pd.Series([f"{t} -> {p}" for r in rs for t, p in zip(r[cohort]["true"], r[cohort]["pred"])
                       if t != p]).value_counts() / len(rs)
    return pairs.head(n).round(1)


def by_biopsy_and_purity(runs, picks):
    met = prepare.load("met500")
    tc = met["tumor_content"].astype(float)
    tc_group = pd.cut(tc, [0, 0.4, 0.6, 1.0], labels=["low (<=0.4)", "mid", "high (>0.6)"])
    rows = []
    for feature, size in picks:
        rs = [r for r in runs if r["feature"] == feature and r["size"] == size]
        correct = np.mean([np.array(r["met500"]["pred"]) == np.array(r["met500"]["true"])
                           for r in rs], axis=0)
        df = pd.DataFrame({"site": rs[0]["met500"]["biopsy_site"], "tc": tc_group,
                           "correct": correct})
        for col in ("site", "tc"):
            g = df.groupby(col, observed=True)["correct"].agg(["mean", "size"])
            for k, v in g.iterrows():
                rows.append({"model": f"{feature} {size}", "grouping": col, "group": k,
                             "n": int(v["size"]), "accuracy": round(float(v["mean"]), 3)})
    return pd.DataFrame(rows).pivot_table(index=["grouping", "group", "n"],
                                          columns="model", values="accuracy")


def plot_with_ci(boot, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sizes = sorted({k[1] for k in boot}, key=size_order)
    pos = {s: i for i, s in enumerate(sizes)}
    colors = {"cptac": "#e6550d", "met500": "#31a354"}
    for feature, style in (("expression", "-"), ("pairs", "--")):
        for c, col in colors.items():
            ks = [s for s in sizes if (feature, s, c) in boot]
            x = [pos[s] for s in ks]
            for ax, metric in zip(axes, ("ext", "gap")):
                vals = np.array([ci(boot[(feature, s, c)][metric]) for s in ks])
                ax.plot(x, vals[:, 0], ls=style, color=col, marker="o", ms=4,
                        label=f"{c.upper()}, {feature}")
                ax.fill_between(x, vals[:, 1], vals[:, 2], color=col, alpha=0.12)
    for ax in axes:
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)
        ax.set_xlabel("Genes in panel")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("External macro F1 (95% CI)")
    axes[0].set_title("External accuracy (solid: expression, dashed: gene pairs)")
    axes[1].set_ylabel("Gap: TCGA minus external macro F1 (95% CI)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Internal-external gap")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(n_boot=N_BOOT, seed=2026):
    base = RESULTS
    runs = load_runs(base / "runs")
    if not runs:
        print("No runs found. Run python -m src.experiment first.")
        return
    out = base / "stats"
    out.mkdir(parents=True, exist_ok=True)
    names, code, eligible = encode(runs)
    print(f"Bootstrapping {n_boot} resamples (this takes a few minutes) ...")
    boot = bootstrap(runs, eligible, len(names), n_boot, np.random.default_rng(seed))

    # Table of point estimates with CIs
    rows = []
    for (feature, size, c), v in sorted(boot.items(), key=lambda kv: (kv[0][2], kv[0][0],
                                                                      size_order(kv[0][1]))):
        e, g = ci(v["ext"]), ci(v["gap"])
        rows.append({"cohort": c, "feature": feature, "size": size,
                     "external_macro_f1": e[0], "ext_ci_low": e[1], "ext_ci_high": e[2],
                     "gap": g[0], "gap_ci_low": g[1], "gap_ci_high": g[2]})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(out / "estimates_with_ci.csv", index=False)

    all_sizes = sorted({k[1] for k in boot}, key=size_order)
    no_all = [s for s in all_sizes if s != "all"]
    h1 = pd.DataFrame([h1_test(boot, "cptac", "expression", all_sizes)])
    h1_extra = pd.DataFrame([h1_test(boot, c, f, sz) for c in EXTERNAL
                             for f, sz in (("expression", all_sizes), ("expression", no_all),
                                           ("pairs", [s for s in no_all if int(s) <= 150]))])
    h2 = pd.concat([diff_table(boot, c, H2_SIZES, "gap", +1) for c in EXTERNAL])
    extra_a = pd.concat([diff_table(boot, c, [s for s in no_all if int(s) <= 150], "ext", -1)
                         for c in EXTERNAL])
    h1.round(4).to_csv(out / "H1_prespecified.csv", index=False)
    h1_extra.round(4).to_csv(out / "H1_additional_variants.csv", index=False)
    h2.round(4).to_csv(out / "H2_prespecified.csv", index=False)
    extra_a.round(4).to_csv(out / "additional_A_external_f1_pairs_minus_expression.csv",
                            index=False)

    picks = [(f, s) for f, s in (("expression", "100"), ("pairs", "100"), ("expression", "all"))
             if any(r["feature"] == f and r["size"] == s for r in runs)]
    pc = {c: per_class(runs, c, picks) for c in EXTERNAL}
    for c, t in pc.items():
        t.to_csv(out / f"additional_C_per_class_{c}.csv")
    errors = {(c, f, s): top_errors(runs, c, f, s) for c in EXTERNAL for f, s in picks}
    met = by_biopsy_and_purity(runs, picks)
    met.to_csv(out / "additional_C_met500_by_biopsy_and_purity.csv")
    plot_with_ci(boot, out / "figure_with_ci.png")

    # ---------- printed report ----------
    lines = ["STEP 4 REPORT", "=" * 60, "",
             "PRE-SPECIFIED TEST H1 (CPTAC, expression): slope of gap per doubling of panel size"]
    r = h1.iloc[0]
    lines += [f"  slope = {r.slope_per_doubling:+.4f}  95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
              f"  H1 supported (negative, CI excludes 0)? {'YES' if r.supported else 'NO'}", "",
              "PRE-SPECIFIED TEST H2: gap(expression) - gap(pairs); positive = pairs smaller gap"]
    for c in EXTERNAL:
        sub = h2[h2.cohort == c]
        n_pos = int(sub.ci_excludes_0_positive.sum())
        for _, row in sub.iterrows():
            lines.append(f"  {c.upper():6s} {row['size']:>3s} genes: {row.estimate:+.3f}  "
                         f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]")
        tag = "(primary)" if c == "cptac" else "(supporting)"
        lines.append(f"  {c.upper()} {tag}: positive with CI excluding 0 at {n_pos} of 3 sizes "
                     f"-> H2 {'SUPPORTED' if n_pos >= 2 else 'NOT supported'}")
    lines += ["", "ADDITIONAL A (not pre-specified): external macro F1, pairs minus expression"]
    for _, row in extra_a.iterrows():
        lines.append(f"  {row.cohort.upper():6s} {row['size']:>3s} genes: {row.estimate:+.3f}  "
                     f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]")
    lines += ["", "ADDITIONAL B (not pre-specified): H1 slope variants"]
    for _, row in h1_extra.iterrows():
        lines.append(f"  {row.cohort.upper():6s} {row.feature:10s} sizes {row.sizes}: "
                     f"{row.slope_per_doubling:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}]")
    lines += ["", "ADDITIONAL C (not pre-specified): where models fail"]
    for c in EXTERNAL:
        lines += [f"\n  Per-class sensitivity, {c.upper()} (mean over seeds):", pc[c].to_string()]
    for (c, f, s), e in errors.items():
        lines += [f"\n  Most common errors, {c.upper()}, {f} {s} (per seed):", e.to_string()]
    lines += ["\n  MET500 accuracy by biopsy site and tumour content:", met.to_string()]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report)
    print(report)
    print(f"\nAll tables, figure_with_ci.png and report.txt saved in {out}")


if __name__ == "__main__":
    main()
