"""Step 1d: check the downloaded data and answer the study plan's checklist.

Reports, for each cohort: samples per class (Table 1), genes shared between
cohorts, the units of the expression values, CPTAC cases the label rules could
not place, possible patient overlap, and MET500 file contents.
Everything is also saved to data/processed/inventory_report.txt.

Usage:
    python -m src.inventory
"""
import gzip
from pathlib import Path

import pandas as pd

from . import met500
from .common import CLASSES, PROCESSED, RAW, TCGA_PROJECTS

lines = []


def say(text=""):
    print(text)
    lines.append(str(text))


def strip_version(ids):
    return {str(i).split(".")[0] for i in ids}


def tcga_header(path: Path):
    with gzip.open(path, "rt") as f:
        return f.readline().rstrip("\n").split("\t")


def tcga_primary_patients(samples):
    """TCGA barcodes: characters 14-15 give the sample type; '01' is primary tumour.
    Keep one sample per patient (the first 12 characters)."""
    keep = {}
    for s in samples:
        if len(s) >= 15 and s.startswith("TCGA-") and s[13:15] == "01":
            keep.setdefault(s[:12], s)
    return sorted(keep.values())


def tcga_section():
    say("=== TCGA (training cohort) ===")
    counts, genes, unit_note = {}, None, None
    for cls, projects in TCGA_PROJECTS.items():
        n = 0
        for proj in projects:
            path = RAW / "tcga" / f"{proj}.star_tpm.tsv.gz"
            if not path.exists():
                say(f"  missing {path.name} - run python -m src.download_tcga")
                continue
            n += len(tcga_primary_patients(tcga_header(path)[1:]))
            if genes is None:
                block = pd.read_csv(path, sep="\t", index_col=0, nrows=2000)
                genes = strip_version(pd.read_csv(path, sep="\t", usecols=[0]).iloc[:, 0])
                lo = float(block.min().min())
                unit_note = ("values look like log2(TPM + 0.001): minimum is "
                             f"{lo:.2f}" if lo < -1 else
                             f"values look like log2(TPM + 1): minimum is {lo:.2f}")
        counts[cls] = n
    if unit_note:
        say(f"  Units: {unit_note}")
    if genes:
        say(f"  Genes in TCGA files: {len(genes)}")
    return counts, genes


def cptac_section(tcga_genes):
    say("\n=== CPTAC (external test A) ===")
    meta_path = RAW / "cptac" / "cptac_metadata.csv"
    if not meta_path.exists():
        say("  no metadata yet - run python -m src.download_cptac --metadata-only")
        return {}
    meta = pd.read_csv(meta_path)
    labelled = meta[meta["label"].notna()].drop_duplicates("case_id")
    counts = labelled["label"].value_counts().to_dict()  # one per patient

    unplaced = meta[meta["label"].isna()]
    if len(unplaced):
        say(f"  {len(unplaced)} files not placed in a study class. Please check these "
            "are truly outside the 13 classes:")
        combos = (unplaced.fillna("-")
                  .groupby(["primary_site", "disease_type", "primary_diagnosis"])
                  .size().sort_values(ascending=False).head(30))
        say(combos.to_string())

    tcga_like = meta[meta["case_id"].astype(str).str.startswith("TCGA-")]
    say(f"\n  CPTAC cases with TCGA-style IDs (possible overlap with training): "
        f"{tcga_like['case_id'].nunique()}")

    gene_info = PROCESSED / "gene_info.csv"
    if gene_info.exists() and tcga_genes:
        info = pd.read_csv(gene_info, index_col=0)
        cptac_genes = strip_version(info.index)
        pc = strip_version(info.index[info["gene_type"] == "protein_coding"])
        say(f"  Genes shared by TCGA and CPTAC: {len(cptac_genes & tcga_genes)} "
            f"(protein-coding: {len(pc & tcga_genes)})")
    return counts


def met500_section():
    say("\n=== MET500 (external test B) ===")
    files = met500.find_files()
    if not files:
        say("  no files yet - see instructions at the top of src/met500.py")
        return
    for f in files:
        info = met500.describe(f)
        say(f"  {info['file']}: looks like {info['kind']}")
        for k, v in info.items():
            if k not in ("file", "kind"):
                say(f"    {k}: {v}")
    say("  Send this section to set the MET500 class rules.")


def main():
    say("Data inventory for the study plan\n")
    tcga_counts, tcga_genes = tcga_section()
    cptac_counts = cptac_section(tcga_genes)
    met500_section()

    table = pd.DataFrame({"TCGA": tcga_counts, "CPTAC": cptac_counts}).reindex(CLASSES)
    say("(counts are patients: one sample per patient)")
    table = table.fillna(0).astype(int)
    table.loc["Total"] = table.sum()
    say("\n=== Table 1: samples per class ===")
    say(table.to_string())
    small = [c for c in CLASSES if 0 < table.loc[c, "CPTAC"] < 10]
    if small:
        say(f"  CPTAC classes under 10 samples (shown, not averaged): {small}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "table1_counts.csv")
    (PROCESSED / "inventory_report.txt").write_text("\n".join(lines))
    print(f"\nSaved to {PROCESSED / 'inventory_report.txt'}")


if __name__ == "__main__":
    main()
