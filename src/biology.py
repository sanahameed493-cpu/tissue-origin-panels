"""Step 6: biological interpretation (study plan, Section 7).

1. Applies the plan's rule for the "smallest reliable panel": the smallest panel whose
   CPTAC macro F1 (expression, mean over seeds) is within 2 percentage points of the
   all-genes model. The 100-gene panel, which gave the best external accuracy, is also
   described (additional, not pre-specified).
2. Stability: how many of the 5 seeds selected each gene. Genes chosen in at least
   4 of 5 seeds form the stable core. (Expression and gene-pair models use the same
   ANOVA-selected genes, so there is one panel per size.)
3. Gene names, the cancer type each gene is highest in (TCGA), and whether it is on a
   short list of well-known tissue markers.
4. The most informative gene pairs for each cancer type (100 genes, seed 0).
5. Files ready for Enrichr or g:Profiler, including a background gene list.

Usage:
    python -m src.biology
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from . import prepare  # noqa: E402
from .common import CLASSES, PROCESSED, ROOT  # noqa: E402
from .experiment import RESULTS as PRIMARY, PairFeatures, select_genes  # noqa: E402

OUT = ROOT / "results" / "biology"
MIN_SEEDS = 4

# A short list of widely used tissue or lineage markers, only for flagging genes.
# Always check each gene yourself in GeneCards / the Human Protein Atlas.
KNOWN_MARKERS = {
    "KLK3": "prostate (PSA)", "KLK2": "prostate", "NKX3-1": "prostate", "TMPRSS2": "prostate",
    "ESR1": "breast / endometrium (oestrogen receptor)", "GATA3": "breast / urothelium",
    "TFF1": "breast / stomach", "SCGB2A2": "breast (mammaglobin)", "PGR": "breast / endometrium",
    "CDX2": "intestine", "CDX1": "intestine", "CDH17": "intestine", "VIL1": "intestine",
    "SATB2": "colorectum", "LGALS4": "intestine",
    "NKX2-1": "lung / thyroid (TTF-1)", "NAPSA": "lung / kidney", "SFTPB": "lung",
    "SFTPC": "lung", "SFTPA1": "lung", "SFTPA2": "lung",
    "TP63": "squamous epithelium", "KRT5": "squamous epithelium", "DSG3": "squamous epithelium",
    "SOX2": "squamous / neural", "PAX8": "kidney / Mullerian (ovary, endometrium)",
    "CA9": "clear cell kidney cancer", "WT1": "serous ovarian cancer / mesothelium",
    "MLANA": "melanocyte", "PMEL": "melanocyte", "TYR": "melanocyte", "SOX10": "melanocyte / glia",
    "GFAP": "astrocyte / glioma", "OLIG2": "glioma", "CLDN18": "stomach",
    "GKN1": "stomach", "GKN2": "stomach", "TFF2": "stomach", "MUC5AC": "stomach / pancreas",
    "PRSS1": "pancreas (acinar)", "CPA1": "pancreas (acinar)", "CELA3A": "pancreas (acinar)",
}


def load_run(seed, feature, size):
    return json.loads((PRIMARY / "runs" / f"seed{seed}_{feature}_{size}.json").read_text())


def smallest_reliable(summary):
    expr = summary[summary["feature"] == "expression"].set_index("size")["cptac_macro_f1"]
    ref = expr["all"]
    ok = [s for s in expr.index if s != "all" and expr[s] >= ref - 0.02]
    return min(ok, key=int), ref


def gene_names():
    info = pd.read_csv(PROCESSED / "gene_info.csv", index_col=0)
    info.index = [i.split(".")[0] for i in info.index]
    return info[~info.index.duplicated()]["gene_name"]


def stability(size, seeds):
    counts = {}
    for s in seeds:
        for g in load_run(s, "expression", size)["genes"]:
            counts[g] = counts.get(g, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


def class_profile(tcga, genes):
    """Mean z-scored expression of each gene in each cancer type (all TCGA samples)."""
    idx = [int(np.flatnonzero(tcga["genes"] == g)[0]) for g in genes]
    X = tcga["X"][:, idx].astype(np.float64)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    prof = pd.DataFrame(Z, columns=genes).groupby(tcga["labels"]).mean()
    order = [c for c in CLASSES if c in prof.index]
    return prof.reindex(order + [c for c in prof.index if c not in order])


def core_table(size, seeds, tcga, names):
    stab = stability(size, seeds)
    core = stab[stab >= MIN_SEEDS]
    prof = class_profile(tcga, list(core.index))
    rows = []
    for g in core.index:
        sym = names.get(g, g)
        col = prof[g]
        rows.append({"ensembl_id": g, "gene": sym, "seeds_selected": int(core[g]),
                     "highest_in": col.idxmax(), "z_in_that_class": round(float(col.max()), 2),
                     "known_marker_of": KNOWN_MARKERS.get(sym, "")})
    table = pd.DataFrame(rows).sort_values(["highest_in", "z_in_that_class"],
                                           ascending=[True, False])
    return table, prof, stab


def top_pairs(tcga, names, seed=0, size=100, per_class=5):
    """Refit the seed-0 gene-pair model (same split, genes and C as step 3) and list,
    for each cancer type, the pair rules with the largest positive weights."""
    run = load_run(seed, "pairs", size)
    X, y = tcga["X"], tcga["labels"]
    dev, _ = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    genes = select_genes(X[dev], y[dev], size)
    feat = PairFeatures().fit(X[dev], genes)
    model = LogisticRegression(C=run["C"], class_weight="balanced", max_iter=2000)
    model.fit(feat.transform(X[dev]), y[dev])
    sym = [names.get(g, g) for g in tcga["genes"][genes]]
    rows = []
    for c, cls in enumerate(model.classes_):
        for j in np.argsort(-model.coef_[c])[:per_class]:
            a, b = sym[feat.i[j]], sym[feat.j[j]]
            rows.append({"cancer": cls, "rule": f"{a} > {b}", "weight": round(float(model.coef_[c][j]), 3),
                         "first_gene_marker": KNOWN_MARKERS.get(a, ""),
                         "second_gene_marker": KNOWN_MARKERS.get(b, "")})
    return pd.DataFrame(rows)


def heatmap(prof, table, out, title):
    order = table.sort_values(["highest_in", "z_in_that_class"], ascending=[True, False])
    data = prof[order["ensembl_id"]].clip(-2.5, 2.5)
    fig, ax = plt.subplots(figsize=(max(6, 0.16 * len(order) + 2), 4.8))
    im = ax.imshow(data.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order["gene"], rotation=90, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Mean z-score (TCGA)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(PRIMARY / "summary_runs.csv")
    summary["size"] = summary["size"].astype(str)
    means = summary.groupby(["feature", "size"], as_index=False)["cptac_macro_f1"].mean()
    small, ref = smallest_reliable(means)
    seeds = sorted(summary["seed"].unique())
    tcga = prepare.load("tcga")
    names = gene_names()
    lines = ["STEP 6 REPORT: biological interpretation", "=" * 60,
             f"All-genes CPTAC macro F1 (expression): {ref:.3f}",
             f"Pre-specified 'smallest reliable panel' (within 0.02 of all genes): {small} genes",
             "Also described (additional): 100 genes, the size with the best external accuracy.",
             f"Stable core = genes selected in at least {MIN_SEEDS} of {len(seeds)} seeds.", ""]

    for size, tag in ((small, "prespecified"), ("100", "additional")):
        table, prof, stab = core_table(size, seeds, tcga, names)
        table.to_csv(OUT / f"core_genes_{size}_{tag}.csv", index=False)
        heatmap(prof, table, OUT / f"figure6_heatmap_{size}_genes.png",
                f"Stable core of the {size}-gene panel ({len(table)} genes)")
        (OUT / f"enrichr_{size}_genes.txt").write_text("\n".join(table["gene"]))
        n_mark = int((table["known_marker_of"] != "").sum())
        lines += [f"--- {size}-gene panel ({tag}) ---",
                  f"Genes ever selected: {len(stab)}; stable core: {len(table)}; "
                  f"on the known-marker list: {n_mark}",
                  "Stable-core genes per cancer type (highest mean expression):"]
        for cls, genes in table.groupby("highest_in")["gene"]:
            lines.append(f"  {cls:5s} ({len(genes)}): {', '.join(genes)}")
        lines.append("")

    pairs = top_pairs(tcga, names)
    pairs.to_csv(OUT / "top_gene_pairs_100.csv", index=False)
    lines += ["--- Most informative gene-pair rules (100 genes, seed 0) ---", pairs.to_string(index=False)]

    background = sorted({names.get(g, g) for g in tcga["genes"]})
    (OUT / "background_all_genes.txt").write_text("\n".join(background))
    lines += ["", "For pathway enrichment, paste enrichr_<size>_genes.txt into Enrichr or g:Profiler,",
              "and in g:Profiler use background_all_genes.txt as a custom background."]
    report = "\n".join(lines)
    (OUT / "report.txt").write_text(report)
    print(report)
    print(f"\nSaved in {OUT}")


if __name__ == "__main__":
    main()
