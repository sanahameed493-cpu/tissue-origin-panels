"""Step 1b: download CPTAC primary-tumour expression (GDC STAR pipeline) from the GDC API.

First run with --metadata-only (takes seconds) to see what's available.
The full download fetches one file per tumour (about 4 MB each); it can be
stopped and rerun, and files already downloaded are skipped.

Usage:
    python -m src.download_cptac --metadata-only
    python -m src.download_cptac
"""
import argparse

import certifi
import pandas as pd
import requests

from .common import GDC_API, PROCESSED, RAW, cptac_class, download

FIELDS = ",".join([
    "file_id", "file_name",
    "cases.submitter_id", "cases.project.project_id", "cases.primary_site",
    "cases.disease_type", "cases.diagnoses.primary_diagnosis",
    "cases.samples.submitter_id", "cases.samples.sample_type",
])
FILTERS = {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.program.name", "value": ["CPTAC"]}},
    {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
    {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
    {"op": "in", "content": {"field": "access", "value": ["open"]}},
    {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
]}


def flatten_hit(hit: dict) -> dict:
    case = (hit.get("cases") or [{}])[0]
    samples = case.get("samples") or [{}]
    diagnoses = case.get("diagnoses") or [{}]
    return {
        "file_id": hit["file_id"],
        "case_id": case.get("submitter_id"),
        "sample_id": samples[0].get("submitter_id"),
        "sample_type": samples[0].get("sample_type"),
        "project": (case.get("project") or {}).get("project_id"),
        "primary_site": case.get("primary_site"),
        "disease_type": case.get("disease_type"),
        "primary_diagnosis": diagnoses[0].get("primary_diagnosis"),
    }


def fetch_metadata() -> pd.DataFrame:
    r = requests.post(f"{GDC_API}/files", json={
        "filters": FILTERS, "fields": FIELDS, "format": "JSON", "size": 10000,
    }, timeout=120, verify=certifi.where())
    r.raise_for_status()
    rows = [flatten_hit(h) for h in r.json()["data"]["hits"]]
    meta = pd.DataFrame(rows)
    meta = meta[meta["sample_type"] == "Primary Tumor"].copy()
    meta["label"] = [cptac_class(s, d, dx) for s, d, dx in
                     zip(meta.primary_site, meta.disease_type, meta.primary_diagnosis)]
    return meta.reset_index(drop=True)


def one_sample_per_patient(meta: pd.DataFrame) -> pd.DataFrame:
    """Keep labelled files from one sample per patient (the first sample ID),
    matching how TCGA is handled. All files of that sample are kept and averaged."""
    lab = meta[meta["label"].notna()]
    first = lab.sort_values("sample_id").groupby("case_id")["sample_id"].first()
    return lab[lab["sample_id"].isin(first)].reset_index(drop=True)


def read_star_file(path) -> pd.DataFrame:
    """Read one GDC 'augmented STAR gene counts' file; return gene rows only."""
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df[df["gene_id"].str.startswith("ENSG")]
    return df.set_index("gene_id")


def build_matrix(meta: pd.DataFrame, raw_dir) -> None:
    tpm, gene_info = {}, None
    for i, row in enumerate(meta.itertuples(), 1):
        df = read_star_file(raw_dir / f"{row.file_id}.tsv")
        tpm[row.file_id] = df["tpm_unstranded"].astype("float32")
        if gene_info is None:
            gene_info = df[["gene_name", "gene_type"]]
        if i % 100 == 0:
            print(f"  read {i} of {len(meta)} files")
    mat = pd.DataFrame(tpm)
    mat.columns = meta.set_index("file_id").loc[mat.columns, "sample_id"].values
    mat = mat.T.groupby(level=0).mean().T  # average repeated files from one sample
    PROCESSED.mkdir(parents=True, exist_ok=True)
    mat.to_csv(PROCESSED / "cptac_tpm.tsv.gz", sep="\t")
    gene_info.to_csv(PROCESSED / "gene_info.csv")
    print(f"Saved CPTAC matrix: {mat.shape[0]} genes x {mat.shape[1]} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()

    print("Asking GDC which CPTAC files are available ...")
    meta = fetch_metadata()
    out = RAW / "cptac"
    out.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / "cptac_metadata.csv", index=False)
    print(f"{len(meta)} primary-tumour expression files found.")
    print(meta["label"].fillna("(not in study)").value_counts().to_string())

    if args.metadata_only:
        print(f"\nMetadata saved to {out / 'cptac_metadata.csv'}. "
              "Run again without --metadata-only to download.")
        return

    keep = one_sample_per_patient(meta)
    print(f"\nDownloading {len(keep)} files for the study classes ...")
    failed = []
    for i, row in enumerate(keep.itertuples(), 1):
        print(f"[{i}/{len(keep)}]", end=" ")
        try:
            download(f"{GDC_API}/data/{row.file_id}", out / f"{row.file_id}.tsv")
        except Exception as err:  # keep going; the file is retried on the next run
            print(f"  skipped {row.file_id} for now: {type(err).__name__}")
            failed.append(row.file_id)
    if failed:
        print(f"\n{len(failed)} files could not be downloaded this time. "
              "Run the same command again to fetch just those.")
        return
    print("All files downloaded. Combining them into one matrix ...")
    build_matrix(keep, out)


if __name__ == "__main__":
    main()
