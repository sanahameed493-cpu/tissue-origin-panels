"""Step 2: build analysis-ready tables for TCGA, CPTAC and MET500.

For every cohort:
  1. Convert values to TPM (TCGA: log2(TPM+1) -> TPM; CPTAC: TPM; MET500: FPKM -> TPM).
  2. Keep protein-coding genes measured in all three cohorts (Ensembl IDs, version removed).
  3. Rescale each sample so its TPM over those genes sums to one million. This uses
     only the sample itself, so it is fine for single-sample prediction and causes no leakage.
  4. Store log2(TPM + 1).
  5. Keep one sample per patient and attach the class label.

Output: data/processed/<cohort>.npz, plus Table 1 with all three cohorts.

Usage:
    python -m src.prepare
"""
import numpy as np
import pandas as pd

from . import met500
from .common import CLASSES, PROCESSED, RAW, TCGA_PROJECTS
from .inventory import tcga_header, tcga_primary_patients

# MET500 'cohort' codes -> study classes. Codes not listed here are excluded
# and printed, so the list can be checked against the data.
MET500_CODES = {
    "BRCA": "BRCA", "PRAD": "PRAD", "SKCM": "SKCM",
    "COLO": "CRC", "COAD": "CRC", "READ": "CRC",
    "PAAD": "PAAD", "OV": "OV", "STAD": "STAD", "KIRC": "KIRC",
    "LUAD": "LUAD", "LUSC": "LUSC", "HNSC": "HNSC", "GBM": "GBM", "UCEC": "UCEC",
}


def strip(ids):
    return pd.Index([str(i).split(".")[0] for i in ids])


def to_log2_rescaled(tpm: pd.DataFrame, genes: pd.Index) -> np.ndarray:
    """genes x samples TPM -> samples x genes log2(TPM + 1), rescaled over `genes`."""
    sub = tpm.reindex(genes).fillna(0.0)
    sub = sub / sub.sum(axis=0).replace(0, np.nan) * 1e6
    return np.log2(sub.fillna(0.0).T.to_numpy(dtype=np.float32) + 1.0)


def save(name, X, samples, labels, genes, extra=None):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    payload = dict(X=X.astype(np.float32), samples=np.array(samples, dtype=str),
                   labels=np.array(labels, dtype=str), genes=np.array(genes, dtype=str))
    for k, v in (extra or {}).items():
        v = np.asarray(v)
        payload[k] = v.astype(str) if v.dtype == object else v
    np.savez_compressed(PROCESSED / f"{name}.npz", **payload)
    print(f"  saved {name}.npz: {X.shape[0]} samples x {X.shape[1]} genes")


def load(name):
    """Load a prepared cohort. Returns X (samples x genes), labels, samples, genes, extras."""
    d = np.load(PROCESSED / f"{name}.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


# ---------- reading each cohort as genes x samples TPM ----------

def coding_genes():
    info = pd.read_csv(PROCESSED / "gene_info.csv", index_col=0)
    return set(strip(info.index[info["gene_type"] == "protein_coding"]))


def read_tcga(coding):
    """Read each project, keeping only primary tumours (one per patient) and
    protein-coding genes, so memory stays manageable on a laptop."""
    frames, labels = [], []
    for cls, projects in TCGA_PROJECTS.items():
        for proj in projects:
            path = RAW / "tcga" / f"{proj}.star_tpm.tsv.gz"
            header = tcga_header(path)
            keep = tcga_primary_patients(header[1:])
            df = pd.read_csv(path, sep="\t", index_col=0, usecols=[header[0]] + keep)
            df.index = strip(df.index)
            df = df[~df.index.duplicated() & df.index.isin(coding)].astype(np.float32)
            frames.append(np.exp2(df) - 1.0)  # log2(TPM + 1) -> TPM
            labels += [cls] * len(keep)
            print(f"  {proj}: {len(keep)} patients")
    return pd.concat(frames, axis=1), labels


def read_cptac():
    tpm = pd.read_csv(PROCESSED / "cptac_tpm.tsv.gz", sep="\t", index_col=0)
    tpm.index = strip(tpm.index)
    tpm = tpm[~tpm.index.duplicated()]
    meta = pd.read_csv(RAW / "cptac" / "cptac_metadata.csv").drop_duplicates("sample_id")
    labels = meta.set_index("sample_id").loc[tpm.columns, "label"].tolist()
    return tpm, labels


def met500_files():
    files = met500.find_files()
    expr = next(f for f in files if ".mx" in f.name)
    meta = next(f for f in files if "meta" in f.name)
    return expr, meta


def met500_select(meta: pd.DataFrame) -> pd.DataFrame:
    """Label MET500 samples and keep one per patient (polyA preferred, since the
    data were adjusted to look like polyA). Returns the kept rows with a 'label'."""
    meta = meta.copy()
    meta["label"] = meta["cohort"].astype(str).str.upper().map(MET500_CODES)
    meta["is_poly"] = meta.index.astype(str).str.contains("-poly-")
    lab = meta[meta["label"].notna()]
    lab = lab.sort_values(["sample_source", "is_poly"], ascending=[True, False])
    return lab[~lab["sample_source"].duplicated()]


def read_met500():
    expr_path, meta_path = met500_files()
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)
    print("  MET500 cohort codes (samples):")
    codes = meta["cohort"].astype(str).str.upper().value_counts()
    for code, n in codes.items():
        print(f"    {code:6s} {n:4d}  -> {MET500_CODES.get(code, 'excluded')}")
    kept = met500_select(meta)
    fpkm = pd.read_csv(expr_path, sep="\t", index_col=0)
    fpkm = fpkm[[c for c in kept.index if c in fpkm.columns]]
    fpkm.index = strip(fpkm.index)
    fpkm = fpkm[~fpkm.index.duplicated()]
    tpm = fpkm / fpkm.sum(axis=0) * 1e6  # FPKM -> TPM, over all genes in the file
    kept = kept.loc[tpm.columns]
    return tpm, kept["label"].tolist(), kept


def main():
    coding = coding_genes()
    print("Reading TCGA ...")
    tcga, tcga_labels = read_tcga(coding)
    print("Reading CPTAC ...")
    cptac, cptac_labels = read_cptac()
    print("Reading MET500 ...")
    met, met_labels, met_meta = read_met500()

    genes = pd.Index(sorted(coding & set(tcga.index) & set(cptac.index) & set(met.index)))
    print(f"\nProtein-coding genes shared by all three cohorts: {len(genes)}")

    print("Saving ...")
    save("tcga", to_log2_rescaled(tcga, genes), tcga.columns, tcga_labels, genes)
    save("cptac", to_log2_rescaled(cptac, genes), cptac.columns, cptac_labels, genes)
    save("met500", to_log2_rescaled(met, genes), met.columns, met_labels, genes,
         extra={"biopsy_site": met_meta["biopsy_tissue"].astype(str).to_numpy(),
                "tumor_content": pd.to_numeric(met_meta["tc"], errors="coerce").to_numpy()})

    table = pd.DataFrame({
        "TCGA": pd.Series(tcga_labels).value_counts(),
        "CPTAC": pd.Series(cptac_labels).value_counts(),
        "MET500": pd.Series(met_labels).value_counts(),
    }).reindex(CLASSES).fillna(0).astype(int)
    table.loc["Total"] = table.sum()
    table.to_csv(PROCESSED / "table1_counts.csv")
    print("\n=== Table 1: patients per class (final) ===")
    print(table.to_string())
    print("\nStep 2 done. Classes with fewer than 10 MET500 patients are shown in "
          "results but left out of averaged metrics.")


if __name__ == "__main__":
    main()
