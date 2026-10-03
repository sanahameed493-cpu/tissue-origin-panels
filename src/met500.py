"""Step 1c: MET500 metastatic tumours, downloaded by hand from UCSC Xena.

Xena doesn't publish a fixed download address for MET500, so get the files
from the website:

  1. Open https://xenabrowser.net/datapages/?cohort=MET500%20(expression%20centric)
  2. Click the gene expression dataset, then its "download" link (a .gz file).
  3. Go back, click the phenotype / sample information dataset, download that too.
  4. Put both files in  data/raw/met500/  (create the folder if needed).

Then run:  python -m src.inventory
It will find the files, report their shape and columns, and show which column
holds the primary tissue, so the class rules can be fixed before experiments.
"""
from pathlib import Path

import pandas as pd

from .common import RAW

MET_DIR = RAW / "met500"
SITE_WORDS = ("primary", "tissue", "site", "cancer", "type", "origin")


def find_files():
    if not MET_DIR.exists():
        return []
    return sorted(p for p in MET_DIR.iterdir()
                  if p.is_file() and not p.name.startswith("."))


def describe(path: Path) -> dict:
    """Read a little of the file and decide whether it is expression or sample info."""
    head = pd.read_csv(path, sep="\t", nrows=50, index_col=0)
    n_cols = head.shape[1]
    numeric = head.select_dtypes("number").shape[1]
    kind = "expression" if n_cols > 50 and numeric == n_cols else "sample information"
    info = {"file": path.name, "kind": kind, "columns_seen": n_cols}
    if kind == "expression":
        first = head.index[:3].tolist()
        info["first_genes"] = first
        info["min_value"] = float(head.min().min())
    else:
        full = pd.read_csv(path, sep="\t", index_col=0)
        info["samples"] = len(full)
        info["column_names"] = list(full.columns)
        info["likely_site_columns"] = {
            c: full[c].value_counts().head(25).to_dict()
            for c in full.columns if any(w in c.lower() for w in SITE_WORDS)
        }
    return info
