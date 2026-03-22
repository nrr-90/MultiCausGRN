# MultiCausGRN

**MultiCausGRN: Causal Prior-Guided Graph Attention Model for Multi-Omics Gene Regulatory Network Inference**

MultiCausGRN is a graph attention-based framework for gene regulatory network (GRN) inference from paired single-cell gene expression and chromatin accessibility data. The model formulates GRN reconstruction as a supervised TF-target link prediction problem and integrates curated causal prior edges to bias message passing toward biologically plausible regulatory directions.

## Highlights

- Two-layer Graph Attention Network (GAT) for gene representation learning
- Multi-omics integration using scRNA-seq and scATAC-seq features
- Prior-guided causal regulatory modeling using curated TF-target interactions
- Supervised link prediction with hard negative sampling
- Evaluation using AUROC and AUPRC


## Data Sources

### 1) PBMC multiome dataset (10x Genomics)
Use the official 10x Genomics PBMC multiome dataset page:

- **10k Human PBMCs, Multiome v1.0, Chromium X**  
  https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0

This dataset provides paired scRNA-seq and scATAC-seq measurements suitable for multi-omics GRN reconstruction.

### 2) Benchmark datasets
If you use benchmark datasets from BEELINE, cite the benchmark paper and place processed splits under `data/processed/`.

### 3) Prior regulatory network
Curated TF-target prior interactions can be obtained from:

- **OmniPath**  
  https://omnipathdb.org/

## Expected Input Files

The implementation described in the manuscript expects the following processed files:

### Expression matrix
A cleaned gene-by-cell matrix, for example:

```text
data/processed/ExpressionData_clean.csv
```

- Rows: genes
- Columns: cells
- First column: gene names / gene IDs

### Positive regulatory edges
A CSV file containing known TF-target regulatory pairs, for example:

```text
data/processed/Positive_edges.csv
```

Example columns:

```csv
tf_idx,target_idx
1039,120
1039,454
1541,880
```

### Train / validation / test splits
Optional separate files or split generation during preprocessing.

### Prior network file
Processed causal prior edges from OmniPath mapped to genes in the expression matrix, for example:

```text
data/prior/omnipath_prior_edges.csv
```

## Minimal Dependencies

Typical dependencies for the current implementation include:

- python >= 3.9
- numpy
- pandas
- scikit-learn
- matplotlib
- torch
- torch-geometric


## Preprocessing Workflow

1. Load the paired scRNA-seq and scATAC-seq data.
2. Filter low-quality cells and low-expressed genes.
3. Select highly variable genes.
4. Harmonize gene identifiers across modalities.
5. Map OmniPath TF-target pairs to the filtered gene list.
6. Build positive regulatory edges.
7. Generate negative TF-target pairs with no overlap with positives.
8. Split edges into train / validation / test sets.


### Train the causal-prior model

```bash
python Model/code.py \
  --data_dir data/processed/ \
  --use_causal_prior true \
  --seed 1
```

###  Repeat over multiple seeds

```bash
for seed in 1 2 4 8 16
do
  python Model/Demo.py \ --data_dir data/processed/ --use_causal_prior true --seed $seed
done
```

## Suggested Outputs

Store outputs under `outputs/`:

- trained model checkpoints
- seed-wise AUROC / AUPRC logs
- predicted TF-target scores
- PCA embedding plots
- causal prior network plots
- robustness plots across random seeds


Recommended workflow:

1. Download the official 10x PBMC multiome dataset.
2. Preprocess the gene expression and accessibility data.
3. Filter OmniPath prior edges to genes present in the processed matrix.
4. Generate positive and negative TF-target pairs.
5. Train the baseline and causal-prior variants across multiple seeds.
6. Report mean ± standard deviation for AUROC and AUPRC.
7. Save the best-seed result separately.


The manuscript includes:

- performance comparison with and without causal prior
- benchmark vs PBMC performance overview
- seed-wise robustness plots
- PCA visualization of TF and target embeddings
- hub TFs in the causal prior network

## Contact

**Noor Alkhateeb**  
For questions related to the implementation, data processing, or manuscript, please open an issue in this repository.
202090300@uaeu.ac.ae
