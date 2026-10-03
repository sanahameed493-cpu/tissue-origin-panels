"""Step 6 test on synthetic results."""
import pandas as pd

from src import biology, experiment, summarize
from tests.test_experiment import fake_data


def test_biology_report(monkeypatch, tmp_path):
    data = fake_data(monkeypatch, tmp_path)
    monkeypatch.setattr(experiment, "SIZES", [10, 20, 50, 100, "all"])
    monkeypatch.setattr(experiment, "PAIR_MAX", 100)
    experiment.main(["--seeds", "0", "1", "2", "3", "4"])
    summarize.main([])
    proc = tmp_path / "processed"
    proc.mkdir()
    names = ["KLK3", "ESR1", "CDX2", "MLANA"] + [f"G{i}" for i in range(4, 200)]
    pd.DataFrame({"gene_name": names, "gene_type": "protein_coding"},
                 index=[g + ".1" for g in data["tcga"]["genes"]]).to_csv(proc / "gene_info.csv")
    monkeypatch.setattr(biology, "PROCESSED", proc)
    monkeypatch.setattr(biology, "OUT", tmp_path / "results" / "biology")
    monkeypatch.setattr(biology, "PRIMARY", tmp_path / "results" / "primary")
    biology.main()
    out = tmp_path / "results" / "biology"
    core = pd.read_csv(out / "core_genes_100_additional.csv")
    assert {"KLK3", "ESR1", "CDX2", "MLANA"} <= set(core["gene"])
    assert (core["seeds_selected"] >= 4).all()
    marked = core.set_index("gene")
    assert marked.loc["KLK3", "highest_in"] == "A"            # gene 0 is class A's marker
    assert marked.loc["KLK3", "known_marker_of"].startswith("prostate")
    pairs = pd.read_csv(out / "top_gene_pairs_100.csv")
    assert set(pairs["cancer"]) == {"A", "B", "C", "D"}
    report = (out / "report.txt").read_text()
    assert "smallest reliable panel" in report
    assert (out / "figure6_heatmap_100_genes.png").exists()
    assert len((out / "background_all_genes.txt").read_text().split()) == 200
