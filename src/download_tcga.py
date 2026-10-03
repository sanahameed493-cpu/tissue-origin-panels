"""Step 1a: download TCGA expression (GDC STAR pipeline, TPM) from the UCSC Xena GDC hub.

One file per TCGA project, plus the gene annotation file. Expect a few GB in total;
files already downloaded are skipped, so you can stop and rerun safely.

Usage:
    python -m src.download_tcga
"""
from .common import RAW, TCGA_PROJECTS, XENA_GDC, download

PROBEMAP = "gencode.v36.annotation.gtf.gene.probemap"


def main():
    out = RAW / "tcga"
    print("Downloading gene annotation ...")
    download(f"{XENA_GDC}/{PROBEMAP}", out / PROBEMAP)
    projects = [p for ps in TCGA_PROJECTS.values() for p in ps]
    for i, proj in enumerate(projects, 1):
        print(f"[{i}/{len(projects)}] {proj}")
        download(f"{XENA_GDC}/{proj}.star_tpm.tsv.gz", out / f"{proj}.star_tpm.tsv.gz")
    print(f"\nDone. Files are in {out}")


if __name__ == "__main__":
    main()
