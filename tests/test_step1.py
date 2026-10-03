"""Step 1 tests on small fake files shaped like the real downloads. No internet needed."""
import gzip

import numpy as np
import pandas as pd

from src import common, download_cptac, inventory, met500
from src.common import cptac_class

GENES = [f"ENSG{i:011d}.{i % 9 + 1}" for i in range(60)]


def test_cptac_label_rules():
    assert cptac_class("Bronchus and lung", "Adenomas and Adenocarcinomas",
                       "Adenocarcinoma, NOS") == "LUAD"
    assert cptac_class("Bronchus and lung", "Squamous Cell Neoplasms",
                       "Squamous cell carcinoma, NOS") == "LUSC"
    assert cptac_class("Kidney", "Adenomas and Adenocarcinomas",
                       "Renal cell carcinoma, clear cell type") == "KIRC"
    assert cptac_class("Pancreas", "Ductal and Lobular Neoplasms",
                       "Infiltrating duct carcinoma, NOS") == "PAAD"
    assert cptac_class("Larynx", "Squamous Cell Neoplasms",
                       "Squamous cell carcinoma, NOS") == "HNSC"
    assert cptac_class("Brain", "Gliomas", "Glioblastoma") == "GBM"
    assert cptac_class("Ovary", "Cystic, Mucinous and Serous Neoplasms",
                       "Serous cystadenocarcinoma, NOS") == "OV"
    assert cptac_class("Breast", "Squamous Cell Neoplasms", "Squamous") is None
    assert cptac_class("Liver", "Adenomas and Adenocarcinomas", "HCC") is None
    assert cptac_class("Kidney", "Adenomas and Adenocarcinomas",
                       "Renal cell carcinoma, NOS") == "KIRC"
    assert cptac_class("Kidney", "Adenomas and Adenocarcinomas",
                       "Papillary renal cell carcinoma") is None
    assert cptac_class("Kidney", "Adenomas and Adenocarcinomas",
                       "Renal cell carcinoma, chromophobe type") is None
    assert cptac_class("Brain", "Gliomas", "Gliosarcoma") == "GBM"
    assert cptac_class("Brain", "Gliomas", "Oligodendroglioma, NOS") is None
    assert cptac_class("Other and unspecified female genital organs",
                       "Cystic, Mucinous and Serous Neoplasms",
                       "Serous adenocarcinoma, NOS") == "OV"
    assert cptac_class("Stomach", "Adenomas and Adenocarcinomas",
                       "Adenocarcinoma, NOS") == "STAD"
    assert cptac_class("Skin", "Nevi and Melanomas", "Malignant melanoma, NOS") == "SKCM"


def test_one_sample_per_patient():
    meta = pd.DataFrame({"file_id": ["a", "b", "c", "d"],
                         "case_id": ["P1", "P1", "P1", "P2"],
                         "sample_id": ["S2", "S1", "S1", "S3"],
                         "label": ["KIRC"] * 4})
    kept = download_cptac.one_sample_per_patient(meta)
    assert sorted(kept.file_id) == ["b", "c", "d"]   # both files of S1, none of S2


def test_tcga_primary_one_per_patient():
    samples = ["TCGA-AA-0001-01A", "TCGA-AA-0001-01B", "TCGA-AA-0001-11A",
               "TCGA-AA-0002-06A", "TCGA-AA-0003-01A"]
    assert inventory.tcga_primary_patients(samples) == ["TCGA-AA-0001-01A",
                                                        "TCGA-AA-0003-01A"]


def write_star_file(path, seed):
    rng = np.random.default_rng(seed)
    rows = ["# gene-model: GENCODE v36",
            "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\t"
            "stranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded"]
    for s in ["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]:
        rows.append(f"{s}\t\t\t5\t5\t5\t\t\t")
    for i, g in enumerate(GENES):
        gtype = "protein_coding" if i % 3 else "lncRNA"
        rows.append(f"{g}\tG{i}\t{gtype}\t10\t5\t5\t{rng.gamma(2, 3):.3f}\t1\t1")
    path.write_text("\n".join(rows) + "\n")


def test_read_star_file(tmp_path):
    write_star_file(tmp_path / "x.tsv", 0)
    df = download_cptac.read_star_file(tmp_path / "x.tsv")
    assert len(df) == len(GENES) and "tpm_unstranded" in df.columns


def test_full_inventory(tmp_path, monkeypatch):
    raw, proc = tmp_path / "raw", tmp_path / "processed"
    for mod in (inventory, download_cptac, common):
        monkeypatch.setattr(mod, "RAW", raw, raising=False)
        monkeypatch.setattr(mod, "PROCESSED", proc, raising=False)
    monkeypatch.setattr(met500, "MET_DIR", raw / "met500")

    # Fake TCGA matrices in Xena's log2(tpm + 0.001) style
    (raw / "tcga").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for cls, projects in common.TCGA_PROJECTS.items():
        for proj in projects:
            cols = [f"TCGA-{proj[-2:]}-{k:04d}-01A" for k in range(12)]
            cols += [f"TCGA-{proj[-2:]}-0000-11A"]  # a normal sample, dropped
            df = pd.DataFrame(np.log2(rng.gamma(2, 3, (len(GENES), len(cols))) + 0.001),
                              index=pd.Index(GENES, name="Ensembl_ID"), columns=cols)
            with gzip.open(raw / "tcga" / f"{proj}.star_tpm.tsv.gz", "wt") as f:
                df.to_csv(f, sep="\t")

    # Fake CPTAC metadata, files, and matrix
    (raw / "cptac").mkdir(parents=True)
    meta = pd.DataFrame({
        "file_id": [f"f{i}" for i in range(6)],
        "case_id": ["C3L-1", "C3L-2", "C3L-3", "C3L-4", "C3L-5", "C3L-5"],
        "sample_id": ["S1", "S2", "S3", "S4", "S5", "S5"],
        "sample_type": ["Primary Tumor"] * 6,
        "project": ["CPTAC-3"] * 6,
        "primary_site": ["Kidney", "Bronchus and lung", "Pancreas", "Liver", "Brain", "Brain"],
        "disease_type": ["Adenomas and Adenocarcinomas", "Squamous Cell Neoplasms",
                         "Ductal and Lobular Neoplasms", "Adenomas and Adenocarcinomas",
                         "Gliomas", "Gliomas"],
        "primary_diagnosis": ["Renal cell carcinoma, clear cell type", "Squamous cell carcinoma",
                              "Infiltrating duct carcinoma", "Hepatocellular carcinoma",
                              "Glioblastoma", "Glioblastoma"],
    })
    meta["label"] = [cptac_class(*r) for r in
                     meta[["primary_site", "disease_type", "primary_diagnosis"]].values]
    meta.to_csv(raw / "cptac" / "cptac_metadata.csv", index=False)
    keep = download_cptac.one_sample_per_patient(meta)
    for i, fid in enumerate(keep.file_id):
        write_star_file(raw / "cptac" / f"{fid}.tsv", i)
    download_cptac.build_matrix(keep.reset_index(drop=True), raw / "cptac")
    mat = pd.read_csv(proc / "cptac_tpm.tsv.gz", sep="\t", index_col=0)
    assert mat.shape == (len(GENES), 4)  # S5 had two files, averaged into one

    # Fake MET500 files
    (raw / "met500").mkdir(parents=True)
    expr = pd.DataFrame(rng.random((40, 60)), index=[g.split(".")[0] for g in GENES[:40]],
                        columns=[f"MO_{i}" for i in range(60)])
    expr.to_csv(raw / "met500" / "expression.tsv", sep="\t")
    pheno = pd.DataFrame({"tissue": ["prostate"] * 30 + ["breast"] * 30,
                          "biopsy_site": ["liver"] * 60}, index=expr.columns)
    pheno.to_csv(raw / "met500" / "phenotype.tsv", sep="\t")

    inventory.lines.clear()
    inventory.main()
    table = pd.read_csv(proc / "table1_counts.csv", index_col=0)
    assert table.loc["BRCA", "TCGA"] == 12
    assert table.loc["CRC", "TCGA"] == 24          # COAD + READ merged
    assert table.loc["KIRC", "CPTAC"] == 1 and table.loc["GBM", "CPTAC"] == 1
    assert "STAD" in table.index and "SKCM" in table.index
    report = (proc / "inventory_report.txt").read_text()
    assert "log2(TPM + 0.001)" in report
    assert "Hepatocellular" in report               # unplaced case listed for checking
    assert "Genes shared by TCGA and CPTAC: 60" in report
    assert "sample information" in report and "expression" in report
