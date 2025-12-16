## Graph Finals – GNN Message Passing (Cora)

End-to-end graph node classification project on the Cora citation network using a custom message-passing GNN built with PyTorch Geometric. The workflow lives in notebooks and produces trained weights, evaluation figures, and a saved best model checkpoint.

### What’s inside
- Custom `MessagePassing` layer (`W_msg` for neighbors + `W_self` for self) feeding stacked residual `GNNLayer`s with BatchNorm, Dropout, and ReLU.
- 4-layer GNN backbone with an input projection head and log-softmax classifier.
- Early-stopped training with gradient clipping, cosine scheduler, and checkpointing to `models/best_model.pth`.
- Visuals: accuracy/loss curves, confusion matrix, class distribution, node embeddings, and subgraph views in `figures/`.

### Dataset
- Cora citation network (via `torch_geometric.datasets.Planetoid`).
- 2,708 nodes, 5,429 edges, 1,433 bag-of-words features, 7 classes.
- Standard train/val/test splits provided by the dataset loader.

### Results (current run)
- Test accuracy: ~0.74
- Macro F1: ~0.73 (see `notebooks/main_notebook.ipynb`, classification report cell)
- Confusion matrix saved to `figures/confussion_matrix.png`
- Accuracy & loss curves saved to `figures/accuracy.png` and `figures/training_loss.png`

### Repository layout
- `notebooks/main_notebook.ipynb` – full training, evaluation, and visualization flow.
- `notebooks/testing_notebook.ipynb` – quick experimentation.
- `models/best_model.pth` – checkpoint saved when validation accuracy improves.
- `figures/` – generated plots (metrics, embeddings, subgraphs).
- `requirements.txt` – minimal dependencies.

### Quickstart
Prereqs: Python 3.10+ and pip. Install PyTorch+CUDA per your platform from https://pytorch.org/get-started/locally/ before installing Torch Geometric wheels if you need GPU support.

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# Install core deps (PyTorch must match your platform/CUDA)
pip install -r requirements.txt
```

If PyTorch Geometric wheels fail on your platform, install with the official selector:
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html
```
(replace `${TORCH_VERSION}` / `${CUDA_TAG}` as directed by the PyG install guide).

### Run training & evaluation
1) Launch Jupyter and open the main notebook:
```bash
jupyter notebook notebooks/main_notebook.ipynb
```
2) Run cells top to bottom. The notebook will:
   - Download Cora, build data splits, and set seeds (42).
   - Initialize the custom GNN and optimizer/scheduler.
   - Train with early stopping and gradient clipping; the best model is saved to `models/best_model.pth`.
   - Log metrics and render plots saved under `figures/`.

### Using the saved checkpoint
Inside a Python session or notebook cell:
```python
import torch
from notebooks.main_notebook import GNNModel  # or redefine the same class locally
from torch_geometric.datasets import Planetoid

data = Planetoid(root="data/Cora", name="Cora")[0]
model = GNNModel(
    in_channels=data.num_features,
    hidden_channels=256,
    out_channels=7
)
model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
model.eval()
```

### Reproducibility notes
- Seeds set to 42 for PyTorch/NumPy; CuDNN deterministic flags set.
- Early stopping patience: 20; max epochs: 150; gradient clipping max-norm: 2.0.
- Hidden size 256, dropout 0.5/0.3 in deeper layers.

### Figures to check
- `figures/accuracy.png`, `figures/training_loss.png` – learning curves.
- `figures/confussion_matrix.png` – test confusion matrix.
- `figures/node_embedding.png` – t-SNE of learned embeddings.
- `figures/subgraph2hopneighborhood.png` – neighborhood visualization.
- `figures/class_distribution.png` – label balance.

<!-- ### Troubleshooting
- PyG install issues: re-run installation with the matching CUDA/CPU wheels from `https://data.pyg.org/whl/`.
- CUDA not found: ensure the PyTorch install matches your NVIDIA driver/CUDA; fall back to CPU by uninstalling the CUDA wheel and installing the CPU-only build.
- Notebook OOM: reduce `hidden_channels`, drop layers, or switch to CPU to avoid GPU memory fragmentation. -->