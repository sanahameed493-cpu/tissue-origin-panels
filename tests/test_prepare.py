"""Step 2 test: fake TCGA, CPTAC and MET500 files in their real formats."""
import gzip

import numpy as np
import pandas as pd

from src import common, inventory, met500, prepare

N_GENES = 30
GENES = [f"ENSG{i:011d}" for i in range(N_GENES)]


def setup_fake(tmp_path, monkeypatch):
    raw, proc = tmp_path / "raw", tmp_path / "processed"
    for mod in (common, inventory, prepare):
        monkeypatch.setattr(mod, "RAW", raw, raising=False)
        monkeypatch.setattr(mod, "PROCESSED", proc, raising=False)
    monkeypatch.setattr(met500, "MET_DIR", raw / "met500")
    rng = np.random.default_rng(0)

    # Gene info: first 25 genes protein-coding, last 5 lncRNA
    proc.mkdir(parents=True)
    pd.DataFrame({"gene_name": [f"G{i}" for i in range(N_GENES)],
                  "gene_type": ["protein_coding"] * 25 + ["lncRNA"] * 5},
                 index=pd.Index([g + ".3" for g in GENES], name="gene_id")
                 ).to_csv(proc / "gene_info.csv")

    # TCGA: log2(TPM + 1), versioned IDs, 4 primary patients + 1 normal per project
    (raw / "tcga").mkdir(parents=True)
    for projects in common.TCGA_PROJECTS.values():
        for proj in projects:
            cols = [f"TCGA-{proj[-2:]}-{k:04d}-01A" for k in range(4)] + [f"TCGA-{proj[-2:]}-0000-11A"]
            tpm = rng.gamma(2, 50, (N_GENES, len(cols)))
            df = pd.DataFrame(np.log2(tpm + 1), columns=cols,
                              index=pd.Index([g + ".7" for g in GENES], name="Ensembl_ID"))
            with gzip.open(raw / "tcga" / f"{proj}.star_tpm.tsv.gz", "wt") as f:
                df.to_csv(f, sep="\t")

    # CPTAC: TPM matrix (genes x samples) + metadata
    (raw / "cptac").mkdir(parents=True)
    samples = ["S1", "S2", "S3"]
    pd.DataFrame(rng.gamma(2, 50, (N_GENES, 3)), columns=samples,
                 index=[g + ".3" for g in GENES]).to_csv(proc / "cptac_tpm.tsv.gz", sep="\t")
    pd.DataFrame({"sample_id": samples, "label": ["KIRC", "LUAD", "STAD"]}
                 ).to_csv(raw / "cptac" / "cptac_metadata.csv", index=False)

    # MET500: FPKM + metadata; patient P1 has capture and polyA samples; P3 is unmapped
    (raw / "met500").mkdir(parents=True)
    ids = ["P1-capt-A", "P1-poly-B", "P2-poly-C", "P3-poly-D"]
    meta = pd.DataFrame({"sample_type": "tumor", "sample_source": ["P1", "P1", "P2", "P3"],
                         "dataset": "mctp", "tissue": ["breast", "breast", "prostate", "lung"],
                         "cohort": ["BRCA", "BRCA", "PRAD", "LUNG"], "run.id": ids,
                         "idx": range(4), "test": False, "tc": [0.5, 0.5, 0.3, 0.8],
                         "biopsy_tissue": ["liver", "liver", "lymph_node", "lung"]},
                        index=pd.Index(ids, name="Sample_id"))
    meta.to_csv(raw / "met500" / "MET500_geneExpression_M.meta.plus sample metadata,.txt", sep="\t")
    fpkm = pd.DataFrame(rng.gamma(2, 5, (N_GENES + 2, 4)), columns=ids,
                        index=pd.Index([g + ".1" for g in GENES] + ["ENSG99999999998.1",
                                                                   "ENSG99999999999.1"], name="sample"))
    with gzip.open(raw / "met500" / "MET500_geneExpression_M.mx.txt.gz", "wt") as f:
        fpkm.to_csv(f, sep="\t")
    return proc


def test_met500_select_prefers_polya():
    meta = pd.DataFrame({"sample_source": ["P1", "P1", "P2"], "cohort": ["BRCA", "BRCA", "LUNG"]},
                        index=["P1-capt-A", "P1-poly-B", "P2-poly-C"])
    kept = prepare.met500_select(meta)
    assert list(kept.index) == ["P1-poly-B"]   # polyA kept, unmapped code dropped


def test_prepare_end_to_end(tmp_path, monkeypatch):
    proc = setup_fake(tmp_path, monkeypatch)
    prepare.main()
    tcga, cptac, met = (prepare.load(n) for n in ("tcga", "cptac", "met500"))

    # Only the 25 protein-coding genes, identical across cohorts
    assert tcga["X"].shape[1] == 25
    assert list(tcga["genes"]) == list(cptac["genes"]) == list(met["genes"])

    # TCGA: 4 primary patients per project, normals dropped, COAD + READ merged
    assert tcga["X"].shape[0] == 4 * 14
    assert (tcga["labels"] == "CRC").sum() == 8

    # Every sample rescaled so TPM over the shared genes sums to one million
    for d in (tcga, cptac, met):
        sums = (np.exp2(d["X"].astype(np.float64)) - 1).sum(axis=1)
        assert np.allclose(sums, 1e6, rtol=1e-3)

    # MET500: one sample per patient, polyA preferred, LUNG excluded, extras kept
    assert list(met["samples"]) == ["P1-poly-B", "P2-poly-C"]
    assert list(met["labels"]) == ["BRCA", "PRAD"]
    assert list(met["biopsy_site"]) == ["liver", "lymph_node"]

    table = pd.read_csv(proc / "table1_counts.csv", index_col=0)
    assert table.loc["BRCA", "MET500"] == 1 and table.loc["KIRC", "CPTAC"] == 1
