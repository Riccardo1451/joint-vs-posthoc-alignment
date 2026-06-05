# Joint vs Post-hoc Cross-Modal Alignment

A study comparing two families of methods for learning a shared embedding space between two heterogeneous modalities: **joint alignment** (training both encoders together end-to-end) versus **post-hoc alignment** (training each encoder independently, then geometrically mapping one onto the other).

---

## The Problem

When two modalities describe the same underlying concepts — for example, a 1D waveform and a 2D image of the same handwritten digit — we want their representations to live in the same geometric space. This enables cross-modal retrieval: given a signal, find the matching image, and vice versa.

There are two natural strategies to achieve this:

1. **Train jointly** — use a contrastive objective (like CLIP) that directly pulls together representations of matched pairs during training.
2. **Align post-hoc** — train each encoder independently with a classification objective, then compute a geometric transformation that maps one embedding space onto the other.

The central question is: *which approach produces better-aligned, more generalizable representations?*

---

## Data

Two modalities, same semantic content (digits 0–9):

| Modality | Dataset | Format |
|---|---|---|
| Signal | [MNIST-1D](https://github.com/greydanus/mnist1d) | 1D time series, length 40 |
| Image | Scikit-learn Digits | 8×8 grayscale images, flattened to 64 |

Both datasets contain the same 10 digit classes. Paired samples (same digit class from both modalities) are used for training and alignment.

---

## Methods

### Joint Alignment

Both encoders are trained together by minimizing a contrastive loss over paired batches.

**InfoNCE (CLIP baseline)**  
The standard CLIP objective: for a batch of N pairs, each sample should be closer to its match than to all N−1 negatives in both directions.

**InfoNCE + DeepCORAL**  
Adds a second term that penalizes differences in the second-order statistics (covariance matrices) of the two embedding distributions:

$$\mathcal{L} = \mathcal{L}_{\text{InfoNCE}} + \lambda \cdot \frac{1}{4d^2} \| \Sigma_{\text{img}} - \Sigma_{\text{sig}} \|_F^2$$

This encourages the geometry of the two embedding spaces to match, not just the pairwise similarities.

**Label Flip**  
An ablation where the pairing between modalities is deliberately randomized during training, used to understand the contribution of semantic alignment.

### Post-hoc Alignment

Each encoder is first trained as an independent classifier. Their embedding spaces are then aligned using:

**Procrustes**  
Given a paired alignment set, find the orthogonal rotation matrix Q that minimizes the Frobenius distance between the two embedding clouds:

$$Q^* = \arg\min_{Q^\top Q = I} \| Y - X Q^\top \|_F$$

Solved in closed form via SVD: `M = YᵀX`, then `Q = UVᵀ`.

**Procrustes + CORAL**  
Extends Procrustes by also matching the covariance structure of the two spaces, borrowing from domain adaptation.

---

## Evaluation

| Metric | What it measures |
|---|---|
| **Recall@5** | Cross-modal retrieval: fraction of queries whose true match appears in the top 5 retrieved items |
| **CKA** | Centered Kernel Alignment — how similar two representation spaces are, across different seeds or architectures |
| **Modality gap** | Mean distance between the two embedding clouds and their covariance trace |
| **t-SNE** | Visual inspection of cluster structure before and after alignment |

All experiments are repeated over 3 seeds (42, 123, 999) to assess stability.

---

## Architecture

Encoders are available in two flavors, both projecting into a shared 32-dimensional space:

- **MLP** — two-layer fully connected network
- **CNN** — convolutional backbone adapted to each input format

For CLIP models, a separate linear projection head is applied after the encoder before normalization.

---

## Platonic Representation Hypothesis

A secondary experiment (`evaluate_architectures.py`) measures CKA between CNN and MLP encoders trained on the same task. If different architectures converge to similar representations, this supports the hypothesis that the learned geometry is driven by the task rather than the model choice.

---

## Project Structure

```
scripts/
├── train_unimodal.py           # Train independent classifiers (pre-step for post-hoc)
├── train_clip_infonce.py       # Joint training: InfoNCE only
├── train_clip_deepcoral.py     # Joint training: InfoNCE + DeepCORAL
├── train_clip_labelflip.py     # Joint training: with label permutation
├── align_procrustes.py         # Post-hoc: Procrustes alignment
├── align_procrustes_coral.py   # Post-hoc: Procrustes + CORAL
├── evaluate_clip_infonce.py    # Evaluation for CLIP InfoNCE
├── evaluate_clip_deepcoral.py  # Evaluation for CLIP DeepCORAL
├── evaluate_label_flip.py      # Evaluation for label flip ablation
└── evaluate_architectures.py   # CKA cross-architecture (Platonic hypothesis)

methods/
├── procrustes.py   # Procrustes alignment (paired and centroid-based)
├── coral.py        # CORAL domain adaptation
└── losses.py       # InfoNCE and DeepCORAL loss functions

models/
├── clip_model.py   # Dual-encoder CLIP architecture
├── encoders.py     # MLP and CNN encoders
└── unimodal.py     # Unimodal classifiers
```

---

## Running the Experiments

All scripts are run from the project root.

```bash
# 1. Train unimodal classifiers (required for post-hoc methods)
python scripts/train_unimodal.py

# 2a. Joint alignment
python scripts/train_clip_infonce.py
python scripts/train_clip_deepcoral.py

# 2b. Post-hoc alignment
python scripts/align_procrustes.py
python scripts/align_procrustes_coral.py

# 3. Evaluate
python scripts/evaluate_clip_infonce.py
python scripts/evaluate_clip_deepcoral.py
python scripts/evaluate_architectures.py
```

Figures are saved under `figures/` and model checkpoints under `checkpoints/`.

---

## Dependencies

```bash
pip install torch numpy scikit-learn mnist1d tqdm matplotlib
```
