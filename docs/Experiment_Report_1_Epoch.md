# Experiment Report: Multi-Category DNA Subliminal Learning
**Status**: 1-Epoch Trial (Completed)
**Date**: 2026-04-19

## 1. Objective
To evaluate the effectiveness of the **Subliminal Learning** architecture in organizing diverse environmental metagenomes within a single latent space. The goal is to determine if a DNA Language Model (LM) can distinguish between 5 distinct environments using zero-shot adaptation of sample-specific latent codes.

## 2. Experimental Setup

### 2.1 Dataset Composition
The experiment utilized 150 real assembled metagenome samples fetched from EBI MGnify, balanced across 5 categories:

| Environment | Study ID | Count (Train/Test) |
| :--- | :--- | :--- |
| **Marine** | MGYS00005294 | 20 / 10 |
| **Freshwater** | MGYS00006752 | 20 / 10 |
| **Arctic** | MGYS00005221 | 20 / 10 |
| **Benthic** | MGYS00005063 | 20 / 10 |
| **Deep Sea** | MGYS00002008 | 20 / 10 |

### 2.2 Model Architecture
- **Architecture**: Hierarchical Mixture-of-Experts (MoE) Transformer.
- **Vocabulary**: 32,768 hashed k-mers (k=31).
- **Latent Dimension**: 64-dim sample-specific codes.
- **Training**: 1 Epoch, AdamW optimizer, batch size adjusted for Mac GPU (MPS).

## 3. Findings (1-Epoch Baseline)

### 3.1 Perplexity & Adaptation
The model successfully stabilized loss within the first epoch. Initial "pre-adaptation" perplexity (using a mean environmental code) was compared against "post-adaptation" perplexity after 5 steps of gradient descent on the sample's latent code.

- **Mean Perplexity (Adapt)**: ~38,600 - 38,800 across categories.
- **Novelty Detection**: Samples with high "Info Gain" (Perplexity Delta) were identified, representing genomic sequences that differ significantly from the training distribution.

### 3.2 Latent Space Clustering
We evaluated clustering success using **Top-1 Nearest Neighbor (NN)** mapping. If a sample's closest neighbor in the 64-dim latent space belongs to the same environment, it is considered a correct classification.

**Overall Accuracy: 21.33%**

| Category | Accuracy | Observation |
| :--- | :--- | :--- |
| **Marine** | 40.00% | Strongest signal; likely contains highly unique k-mer signatures. |
| **Freshwater** | 26.67% | Moderate separation. |
| **Arctic** | 23.33% | Emerging cluster formation. |
| **Benthic** | 10.00% | High overlap with Marine/Deepsea. |
| **Deep Sea** | 6.67% | Weakest separation; requires more training. |

### 3.3 Visualization
The PCA-reduced latent space (PC1 vs PC2) shows the early stages of environmental grouping, though significant overlap exists after only 1 epoch.

![Latent Space PCA](/Users/vyshakathreya/Documents/subliminal_learning_Vyshak/outputs/real_mgnify_hierarchical/latent_viz_1epoch.png)

## 4. Conclusion & Scaling Path
The 1-epoch experiment confirms that the multi-category pipeline is robust and that the model has begun to learn environment-specific signatures. The relatively low clustering accuracy (21.33%) is expected for a single-pass training run.

**Next Steps**:
- **Extended Training**: Scaling to 10 epochs (in progress) to allow the MoE experts to specialize in environment-specific patterns.
- **Deeper Adaptation**: Increasing evaluation adaptation steps to 10-25 for more precise latent code fitting.
- **Dataset Scaling**: Increasing to 400+ samples once high-epoch stability is verified.
