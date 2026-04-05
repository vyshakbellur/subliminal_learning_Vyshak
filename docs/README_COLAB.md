# Running Subliminal Learning on Google Colab

This document provides a "one-click" way to run the **Hierarchical VAE-MoE** model on a high-performance cloud GPU (like NVIDIA A100 or L4).

## 1. Setup

Open a new notebook in Google Colab and ensure you have **GPU Acceleration** enabled (`Runtime` > `Change runtime type` > `Hardware accelerator` > `A100` or `T4`).

## 2. One-Cell Execution Script

Copy and paste the following block into a Colab cell to clone your repository, install dependencies, and launch the 40-sample validation run:

```python
# 1. Clone the repository
!git clone https://github.com/YOUR_USERNAME/subliminal_learning_Vyshak.git
%cd subliminal_learning_Vyshak

# 2. Install requirements
!pip install -q torch pandas matplotlib scikit-learn tqdm

# 3. Running the High-Resolution Validation 
# This will download real MGnify samples and train the VAE-MoE model
!python run_real_mgnify.py \
    --arch hierarchical \
    --n-per-env 20 \
    --epochs 10 \
    --adapt-steps 100 \
    --kl-weight 0.05 \
    --device cuda
```

## 3. Visualizing Results

After the run finishes, you can generate the PCA plot directly in Colab:

```python
!python scripts/plot_pca.py
from IPython.display import Image
Image('outputs/real_mgnify_hierarchical/pca_plot.png')
```

## Tips for Success
- **KL Weight**: If clusters are too scattered, increase `--kl-weight` (e.g., to `0.1`). If the model doesn't learn any DNA features, decrease it.
- **Stride**: For high resolution, the `stride` in `run_real_mgnify.py` is set to `15` for hierarchical models, ensuring dense sequence overlap.
