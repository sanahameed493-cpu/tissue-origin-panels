# Gene Panel Size and Cross-Cohort Robustness in Tissue-of-Origin Classification

Research code for the study plan: how much accuracy do small gene panels lose on independent cohorts, and do rank-based gene-pair rules reduce that loss?

Status: **Steps 1–6 (data, harmonisation, primary experiment, statistics, secondary analyses, biology).**

## Step 1: get and check the data

Set up once (same as Projects 1 and 2):

```
python -m venv .venv
.venv\Scripts\activate          (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest
```

Then run these in order:

| Command | What it does | Time and size |
|---|---|---|
| `python -m src.download_tcga` | TCGA expression for the 13 classes (14 projects) from UCSC Xena | A few GB; can be stopped and rerun |
| `python -m src.download_cptac --metadata-only` | Asks GDC which CPTAC tumours exist | Seconds |
| `python -m src.download_cptac` | Downloads CPTAC tumour files and combines them | About 1,500 files of ~4 MB (one sample per patient) |
| Manual: MET500 | Follow the instructions at the top of `src/met500.py` | Two files |
| `python -m src.inventory` | Counts samples per class, checks genes, units and overlap | Seconds |

The inventory answers the "Decisions to confirm" checklist in the study plan and writes:

- `data/processed/table1_counts.csv`: samples per class (Table 1 of the paper)
- `data/processed/inventory_report.txt`: the full report

## Step 2: harmonise the cohorts

```
python -m src.prepare
```

Converts all three cohorts to the same units (TPM, rescaled over shared protein-coding genes, then log2(TPM + 1)), keeps one sample per patient (polyA preferred in MET500), applies the class labels, and saves `data/processed/tcga.npz`, `cptac.npz` and `met500.npz`, plus the final Table 1.

## Step 3: the primary experiment

```
python -m src.experiment --quick      a few minutes: checks everything works
python -m src.summarize --quick
python -m src.experiment              full primary analysis (can take hours; resumes if stopped)
python -m src.summarize
```

Logistic regression with ANOVA-selected genes at 10, 20, 50, 100, 150, 250, 500 and all genes, for expression values and gene pairs (pairs up to 150 genes), over 5 random TCGA splits. Gene selection and tuning happen only inside the TCGA development set. Each run is saved to `results/primary/runs/`; the summary writes `summary_runs.csv`, `summary_by_size.csv`, `per_class_sensitivity.csv` and `figure3_panel_size.png`.

## Step 4: statistics

```
python -m src.stats
```

Bootstrap 95% confidence intervals (1,000 paired resamples, averaged over seeds), the two pre-specified hypothesis tests (H1: slope of the CPTAC gap against log2 panel size; H2: gap difference at 10, 20 and 50 genes), and clearly labelled additional analyses: external accuracy of pairs vs expression, H1 variants, per-class sensitivity, common errors, and MET500 accuracy by biopsy site and tumour content. Output: `results/primary/stats/`.

## Step 5: secondary analyses

```
python -m src.secondary prepare      extra test sets: TCGA metastases, MET500 lung
python -m src.secondary extra        primary models rerun, predicting the extra sets
python -m src.secondary models       random forest and SVM
python -m src.secondary selection    He et al.-style gene selection
python -m src.secondary summary      report and figure
```

Each part resumes if stopped; add `--quick` for a fast check. Output: `results/secondary/`.

## Step 6: biological interpretation

```
python -m src.biology
```

Stable core genes (selected in at least 4 of 5 seeds) for the pre-specified smallest reliable panel and the 100-gene panel, with gene names, the cancer type each gene is highest in, known tissue markers, a heatmap, the most informative gene-pair rules, and gene lists for Enrichr / g:Profiler. Output: `results/biology/`.

## Project structure

```
src/
  common.py          classes, TCGA projects, CPTAC label rules, download helper
  download_tcga.py   step 1a
  download_cptac.py  step 1b
  met500.py          step 1c (manual download helper)
  inventory.py       step 1d
  prepare.py         step 2
  experiment.py      step 3: runs the models
  summarize.py       step 3: tables and figure
  stats.py           step 4: confidence intervals and hypothesis tests
  secondary.py       step 5: secondary analyses
  biology.py         step 6: biological interpretation
tests/
  test_*.py          every step tested on small fake files
```

`data/` is excluded from Git; the data are public and can be downloaded again with these scripts.

## Data sources

- TCGA: GDC data via the UCSC Xena GDC hub
- CPTAC: open gene expression files from the NCI Genomic Data Commons
- MET500: Robinson et al. (2017), *Nature*, expression hosted on UCSC Xena

Code was developed with assistance from Claude (Anthropic).
