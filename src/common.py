"""Settings and helpers shared by every step. These follow the study plan."""
from pathlib import Path

import time

import certifi
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# The 11 classes in the study plan, and the TCGA projects that make each one
TCGA_PROJECTS = {
    "BRCA": ["TCGA-BRCA"],
    "CRC": ["TCGA-COAD", "TCGA-READ"],   # colon and rectum merged
    "KIRC": ["TCGA-KIRC"],
    "LUAD": ["TCGA-LUAD"],
    "LUSC": ["TCGA-LUSC"],
    "UCEC": ["TCGA-UCEC"],
    "OV": ["TCGA-OV"],
    "GBM": ["TCGA-GBM"],
    "HNSC": ["TCGA-HNSC"],
    "PAAD": ["TCGA-PAAD"],
    "PRAD": ["TCGA-PRAD"],
    "STAD": ["TCGA-STAD"],   # added 2026-10-01 after CPTAC inventory, before any results
    "SKCM": ["TCGA-SKCM"],   # added 2026-10-01 after CPTAC inventory, before any results
}
CLASSES = list(TCGA_PROJECTS)

XENA_GDC = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
GDC_API = "https://api.gdc.cancer.gov"

HEAD_NECK_WORDS = ("lip", "oral", "mouth", "tongue", "gum", "palate", "larynx",
                   "pharynx", "tonsil", "head", "neck", "ill-defined")


def cptac_class(primary_site: str, disease_type: str, diagnosis: str):
    """Map a CPTAC case to one of the 11 classes, or None if it doesn't fit.

    Uses the GDC fields primary_site, disease_type and primary_diagnosis.
    Anything unmapped is listed by inventory.py so the rules can be checked.
    """
    site = (primary_site or "").lower()
    dx = f"{disease_type or ''} {diagnosis or ''}".lower()
    squamous = "squamous" in dx
    if "lung" in site or "bronchus" in site:
        if squamous:
            return "LUSC"
        return "LUAD" if "adenocarcinoma" in dx else None
    if "kidney" in site:
        # CPTAC's clear cell cohort is mostly labelled "Renal cell carcinoma, NOS";
        # papillary, chromophobe, oncocytoma and other kidney tumours are excluded
        return "KIRC" if ("clear cell" in dx or "renal cell carcinoma, nos" in dx) else None
    if "uter" in site or "endometri" in site or "corpus" in site:
        return "UCEC"
    if "brain" in site:
        # gliosarcoma is a glioblastoma variant; lower-grade gliomas are excluded
        return "GBM" if ("glioblastoma" in dx or "gliosarcoma" in dx) else None
    if "pancrea" in site:
        return "PAAD" if ("adenocarcinoma" in dx or "ductal" in dx) else None
    if "breast" in site:
        return None if squamous else "BRCA"
    if "colon" in site or "rect" in site:
        return "CRC"
    if any(w in site for w in ("ovary", "female genital", "peritoneum")):
        # high-grade serous cancers of the ovary, fallopian tube and peritoneum
        return "OV" if "serous" in dx else None
    if "stomach" in site:
        return "STAD" if "adenocarcinoma" in dx else None
    if "skin" in site:
        return "SKCM" if "melanoma" in dx else None
    if squamous and any(w in site for w in HEAD_NECK_WORDS):
        return "HNSC"
    return None


def download(url: str, dest: Path, method: str = "GET", tries: int = 6, **kwargs) -> Path:
    """Download a file with a progress line, skipping it if already present.

    If the connection drops, it waits and tries again (up to `tries` times),
    waiting a little longer each time. Uses certifi's certificate list, which
    avoids the SSL error seen in Project 1.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            _fetch(url, dest, tmp, method, **kwargs)
            return dest
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as err:
            if attempt == tries:
                raise
            wait = 10 * attempt
            print(f"\n  connection problem ({type(err).__name__}); "
                  f"retrying in {wait} s (attempt {attempt + 1} of {tries})")
            time.sleep(wait)


def _fetch(url, dest, tmp, method, **kwargs):
    with requests.request(method, url, stream=True, timeout=120,
                          verify=certifi.where(), **kwargs) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:.0f} of {total / 1e6:.0f} MB",
                          end="", flush=True)
    tmp.replace(dest)
    print(f"\r  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)" + " " * 20)
