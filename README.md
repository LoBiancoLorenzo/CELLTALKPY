# CELLTALKPY — cell-cell communication inference from CellPhoneDB database


## Overview

`celltalkpy` is a Python script that infers cell-cell communication from a single-cell AnnData object using the CellPhoneDB database. For every pair of cell types (sender → receiver) it:

1. Scores each ligand-receptor pair as the average expression of the ligand in the sender and the receptor in the receiver.
2. Tests that score against a random shuffling of cell-type labels (a permutation test), to check it's not just noise.
3. Checks that the ligand is specifically enriched in the sender and the receptor specifically enriched in the receiver (not just expressed everywhere).
4. Requires the interaction to replicate across several patients/samples before calling it "confident-interaction".
5. Plots the resulting communication network.

It ships with a fake-data generator with known "fixed" signals, so you can try the whole pipeline before touching your real data.

**Requirements:** Python ≥ 3.9 with `numpy`, `scipy`, `pandas`, `anndata`, `matplotlib`.

```{bash, eval=FALSE}
pip install numpy scipy pandas anndata matplotlib
```

You also need the CellPhoneDB database tables (e.g. `cellphonedb-data-5.0.0` from [github.com/ventolab/cellphonedb-data](https://github.com/ventolab/cellphonedb-data)). Point `--cpdb-dir` at the folder that directly contains `gene_input.csv`, `protein_input.csv`, `complex_input.csv`, `interaction_input.csv` (for that repo, this is the `data/` sub-folder).

---

## Quick start with fake data

```{bash, eval=FALSE}
# 1) generate a fake AnnData with known signals
python celltalkpy.py fake-data --cpdb-dir cellphonedb-data-5.0.0/data --out fake_adata.h5ad

# 2) run the analysis
python celltalkpy.py run \
  --adata fake_adata.h5ad \
  --cpdb-dir cellphonedb-data-5.0.0/data \
  --celltype-col celltype \
  --sample-col Patient Tumor_type_recurrence \
  --condition-col Tumor_type_recurrence \
  --cascade-source "Tumor epithelial" \
  --outdir fake_out

# 3) sanity check: shuffle cell types, confident interactions should drop to ~0
python ccflow.py run \
  --adata fake_adata.h5ad --cpdb-dir cellphonedb-data-5.0.0/data \
  --celltype-col celltype --sample-col Patient Tumor_type_recurrence \
  --condition-col Tumor_type_recurrence --negative-control --no-plots --outdir fake_nc
```

The fake dataset has 9 cell types, 3 patients, and 3 tissues (`CRC`, `CRLM`, `PBMC`). It comes with 15 planted ligand-receptor signals, listed in `fake_adata_ground_truth.csv`, so you can check the tool finds what it's supposed to find.

---

## Using your own data

Before running, check that:

- **Expression is normalised and log-transformed** (e.g. `sc.pp.normalize_total` + `sc.pp.log1p`), not raw counts. The script warns you if it looks like raw counts.
- **Gene names** are HGNC symbols by default (`adata.var_names`), or Ensembl IDs (`--gene-id-type ensembl`).
- You know which `obs` column holds the **cell type** (`--celltype-col`).

### Samples and conditions 

- `--sample-col` defines a biological replicate. If a patient contributes several tissues, combine columns: `--sample-col Patient Tissue`, so each patient-tissue combination is tested separately.
- `--condition-col` groups samples for the final result (e.g. tumor type). Each sample must belong to exactly one condition.
- If you skip both, all cells get pooled into a single unit — useful only when you have no replicates.

```{bash, eval=FALSE}
python celltalkpy.py run \
  --adata your_data.h5ad \
  --cpdb-dir cellphonedb-data-5.0.0/data \
  --celltype-col cell_type \
  --sample-col Patient Tumor_type_recurrence \
  --condition-col Tumor_type_recurrence \
  --cascade-source "CD8 T" \
  --outdir results
```

---

## Commands

| Command | What it does |
|---|---|
| `run` | Full analysis: statistics, tables, plots. |
| `plot` | Redraw the plots of a previous run (e.g. with different thresholds), no recomputation. |
| `fake-data` | Generate a fake AnnData with planted signals, to try the pipeline. |

Run `python celltalkpy.py <command> --help` for the full list of options with defaults.

### Most useful options

| Option | Meaning |
|---|---|
| `--n-perm` | Number of label permutations (more = more precise p-values, slower). Default 1000. |
| `--expr-frac` | Minimum fraction of cells expressing a gene to consider it "expressed". Default 0.1. |
| `--min-cells` | Minimum cells per cell type per sample to include it. Default 10. |
| `--require-specific` / `--no-require-specific` | Whether ligand and receptor must each be specifically enriched in their own cell type (default: yes). |
| `--min-sample-frac` | Fraction of eligible samples in which an interaction must be significant to be "confident". Default 1.0 (all of them). |
| `--negative-control` | Shuffles cell types before running, as a sanity check (see below). |
| `--cascade-source` | Cell type to start the "who talks to whom" cascade plot from. |
| `--outdir` | Where results are written. |

---

## Output files

| File | Content |
|---|---|
| `confident_interactions.csv` | **Main result**: interactions that passed allt he filters. |
| `consensus_interactions.csv.gz` | All tested interactions, with a `confident` column. |
| `tested_interactions.csv.gz` | Every single test, per sample (scores, p-values). |
| `network_edges.csv` | Sender --> receiver edges (number of confident pairs, strength) per condition. |
| `network_node_metrics.csv` | Per cell type: how much it sends/receives. |
| `plots/` | Network, heatmap, dotplot, and cascade plots. |
| `run_parameters.json` | Every parameter used, due to reproducibility. |

---

## Plots

- **Network**: a circular graph, arrows from sender to receiver, width = number of confident ligand-receptor pairs.
- **Heatmap**: same information as a sender × receiver matrix with exact counts.
- **Dotplot**: which specific ligand-receptor pairs are behind the busiest connections.
- **Cascade**: starting from one cell type (`--cascade-source`), shows who it talks to, and who those talk to, hop by hop.

---

## The statistics, in plain terms

- **Permutation test.** For each sample, cell-type labels are shuffled hundreds/thousands of times, and the real score is compared against this random distribution. If the real score rarely happens by chance, the interaction is significant *for that sample*.
- **Specificity filter.** A high score can come from a ligand or receptor that is just expressed everywhere, not truly specific to a cell type. `--require-specific` checks each partner is individually enriched in its own cell type, not just "expressed enough".
- **Replication across patients.** An interaction found in only one patient could be noise from that patient (e.g. technical artifacts). The tool requires the same interaction to be independently significant in several patients before calling it "confident" — each patient is treated as an independent replicate, not pooled together.
- **Negative control.** Shuffling cell types across the whole dataset destroys all real biology; whatever still comes out "confident" is by definition a false positive. Running this on your own data tells you if your thresholds are strict enough — aim for close to 0 confident hits.
- Results are drawn **per condition** (e.g. per tumor type), not per single patient, because the goal is to find patterns that hold across a patient population

---

## Limitations

- Results are hypotheses based on RNA expression, not direct proof of protein-level signalling.
- The specificity filter (default on) is stricter than classic CellPhoneDB: some real (but weak) interactions may be dropped. Use `--no-require-specific` if you want the classic behaviour.
- run `--negative-control` on your own data before trusting a real run.

---

## References

- Troulé K. et al. *CellPhoneDB v5: inferring cell-cell communication from single-cell multiomics data.* arXiv:2311.04567.
- Efremova M., Vento-Tormo M., Teichmann S.A., Vento-Tormo R. *CellPhoneDB: inferring cell-cell communication from combined expression of multi-subunit ligand-receptor complexes.* Nature Protocols 15, 1484-1506 (2020).
- Phipson B., Smyth G.K. *Permutation p-values should never be zero.* Statistical Applications in Genetics and Molecular Biology 9, 39 (2010).
