# UPI Anomaly Detection — GNN-based Fraud Detection

This repository implements a graph-based anomaly detection pipeline for payment fraud using PyTorch Geometric. The code trains edge-level Graph Neural Networks (GAT and GraphSAGE) to predict fraudulent transactions and includes explainability via Captum (integrated gradients) and network-topological features.

---

**Key Features**

- **Graph construction:** Builds a customer graph from transaction pairs and computes node-level topological features (degree, PageRank).
- **Node & edge features:** Aggregated sender/receiver statistics, transaction-type one-hot encodings, scaled amounts.
- **Edge-level GNNs:** Trains `GAT` and `GraphSAGE` edge classifiers for fraud prediction.
- **Imbalanced training handling:** Uses `BCEWithLogitsLoss` with positive-class weighting.
- **Evaluation & visualization:** Confusion matrices, ROC and PR curves, threshold analysis, and comparative plots saved to `plots/`.
- **Explainability:** Uses `torch_geometric.explain.Explainer` with `CaptumExplainer` to produce feature importance visuals saved to `xai_explanations/`.

**Repository files**

- **[script.py](script.py)**: Main pipeline — data download, graph construction, feature engineering, train/val/test split, model definitions (GAT & GraphSAGE), training loop, evaluation, plotting, and XAI generation.

**Models & Algorithms**

- `EdgeClassifierGAT`: Graph Attention Network-based edge classifier.
- `EdgeClassifierGraphSAGE`: GraphSAGE-based edge classifier.
- `StandardScaler` for numeric feature normalization.
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, AUPRC; threshold optimization via F1 over PR thresholds.

**Dataset**

- The script downloads the dataset via `kagglehub.dataset_download("rupakroy/online-payments-fraud-detection-dataset")` and expects the CSV `PS_20174392719_1491204439457_log.csv` inside the downloaded folder.
- For demonstration the script samples `50,000` transactions (random_state=42). Adjust sampling in `script.py` if you need the full dataset.

**Dependencies**
Install the required Python packages (preferably in a virtualenv):

```bash
pip install pandas numpy matplotlib networkx scikit-learn torch torchvision torchaudio
pip install torch-geometric  # follow official install per OS/CUDA: https://pytorch-geometric.readthedocs.io/
pip install captum kagglehub
```

Notes:

- `torch-geometric` has platform-specific wheels; follow the official install instructions for the correct command matching your CUDA and PyTorch versions.
- If you don't want to use `kagglehub`, download the dataset manually and update the path in `script.py`.

**Quick Start**

1. Ensure Kaggle or `kagglehub` access is configured (or place the CSV in a known folder and update `script.py`).
2. Create a virtual environment and install dependencies (see above).
3. Run the pipeline:

```bash
python3 script.py
```

Outputs (saved by the script):

- Model weights: `best_model_GAT.pth`, `best_model_GraphSAGE.pth` (saved in repo root during training).
- Plots: saved under `plots/` — confusion matrices, ROC/PR comparison, threshold analysis.
- XAI artifacts: saved under `xai_explanations/`.

**Runtime notes & configuration**

- The script auto-detects CUDA and will move data + models to GPU if available. Training is configured for 100 epochs by default — adjust `range(1, 101)` in `script.py` if needed.
- The code subsamples `n=50000` rows for demo speed. Increase or remove sampling for full dataset runs (may require more memory/time).
- The script saves best models by validation ROC-AUC checkpoints every time a better validation AUC is found.

**Reproducibility & tuning**

- Random seed used for sampling: `random_state=42` in `script.py`.
- You can tune hidden dimensions, learning rate, and weight decay in the model/training setup inside `script.py`.


**Authors**

- Vivin Chandrra Paasam
- Topalle Siddha Sankalp
- Aniketh Gandhari
- Syeda Sayeeda Farhath
- Dhiren Rao B.
- Y. Vijayalata
