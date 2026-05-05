# C3D-VAE: Learning Structure–Function Relationships in the Human Brain

A Conditional 3D Variational Autoencoder that generates subject-specific
structural MRI spatial maps corresponding to fMRI ICA functional networks,
trained without ground-truth structural component supervision.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Project Structure

```
cvae-main/
├── main.py                         Training entry point
├── requirements.txt                Python dependencies
├── .gitignore
│
├── model/
│   └── cvae.py                     C3DVAE architecture (encoder, ICA encoder, decoder)
│
├── trainer/
│   ├── loss.py                     Four-term training objective (ELBO variant)
│   ├── metric.py                   Evaluation metrics (RECON, PC, ISC)
│   └── trainer.py                  CVAETrainer with chunked forward pass, monitoring
│
├── data_loader/
│   ├── datasets.py                 UKBBData — lazy load, disk cache, 72/8/20 split
│   ├── data_loaders.py             UKBB DataLoader wrapper
│   ├── ukbb.csv                    Subject manifest (NOT committed — see below)
│   └── tensor_cache/               Auto-created on first run — preprocessed .pt files
│
├── explore/
│   └── eda.py                      Exploratory analysis (run before training)
│
├── evaluate/
│   └── evaluate.py                 Full test-set evaluation + all plots
│
├── ablation/
│   └── run_ablations.py            Four ablation studies (H1–H3)
│
├── utils/
│   └── model_utils.py              Shared: load_model, is_bad, table formatters
│
├── scripts/
│   ├── slurm_train.sh              SLURM job: training
│   ├── slurm_eval.sh               SLURM job: evaluation + ablations
│   └── slurm_eda.sh                SLURM job: exploratory analysis (CPU)
│
└── config/
    ├── runs/C3DVAE.json            Hyperparameters (epochs, lr, λ weights)
    ├── models/C3DVAE.json          Architecture (latent_dim, cond_dim)
    └── data/ukbb.json              Data (batch_size, num_subjects)
```

---

## Subject Manifest (ukbb.csv)

Create `data_loader/ukbb.csv` with columns:

| Column       | Type | Description |
|-------------|------|-------------|
| `subject_id` | int  | Unique subject identifier |
| `ica_path`   | str  | Path to NIfTI file, shape (x, y, z, 53) |
| `smri_path`  | str  | Path to T1w NIfTI, shape (121, 145, 121) |

---

## Recommended Workflow

### Step 0 — Exploratory Analysis (CPU, run first)

```bash
# Local
python explore/eda.py --num_subjects 100 --save_dir eda_results/

# SLURM
sbatch scripts/slurm_eda.sh
```

Produces intensity histograms, ICA sparsity charts, split coverage plots.
**Always run before training** to catch loading failures and normalisation issues.

### Step 1 — Training

```bash
# Local (single GPU)
python main.py \
    --config       config/runs/C3DVAE.json \
    --model_config config/models/C3DVAE.json \
    --data_config  config/data/ukbb.json

# SLURM
sbatch scripts/slurm_train.sh
```

Training saves to `saved/C3DVAE/`:
- `last.pth`, `model_best.pth` — checkpoints
- `train.csv`, `valid.csv` — per-epoch metrics (weighted, recon, pc_loss, orth, kl)
- `loss_curves.png` — updated every epoch; `loss_curves_ep{N}.png` at milestones

### Step 2 — Evaluation

```bash
python evaluate/evaluate.py \
    --checkpoint saved/C3DVAE/model_best.pth \
    --num_subjects 140 \
    --save_dir eval_results/

# SLURM
sbatch scripts/slurm_eval.sh
```

Produces:
- `metrics.csv`
- `rho_histogram.png`, `isc_per_component.png`, `latent_pca.png`
- `reconstruction_subj{i}.png` — sMRI / sum / residual
- `components/subj{i}/component_{k}.png` — ICA vs generated (axial + coronal)

### Step 3 — Ablation Studies

```bash
python ablation/run_ablations.py \
    --checkpoint saved/C3DVAE/model_best.pth \
    --ablation all \
    --save_dir ablation_results/
```

Produces `ablation_summary.csv`

---

## Architecture

```
sMRI Xi (1×64³)  → SMRIEncoder → μ_φ, log σ²_φ → zi ∈ ℝ^128
                                                     │
ICA c̃_ik (1×64³) → ICAEncoder (shared) → e_ik ∈ ℝ^64
                                                     │
              [zi ‖ e_ik] → ComponentDecoder (shared) → ŝ_ik (1×64³)
```

K=53 components decoded in chunked batches of 32 (subject × component) pairs.
~6.6M trainable parameters.

### Encoder (sMRI and ICA, shared structure)

```
ConvBlock_16 : 1  × 64³ → 16 × 32³   (stride-2, BN, LeakyReLU)
ConvBlock_32 : 16 × 32³ → 32 × 16³   (stride-2, BN, LeakyReLU)
ConvBlock_64 : 32 × 16³ → 64 × 8³    (stride-2, BN, LeakyReLU)
GAP          : 64 × 8³  → 64
Linear       : 64       → embedding_dim
```

The sMRI encoder adds `fc_mu` and `fc_logvar` heads (→ ℝ^128 each).
The ICA encoder projects to `cond_dim=64`.

### Decoder (shared across all K components)

```
[zi ‖ e_ik] → Linear(192, 32768) + LeakyReLU → reshape (64×8³)
            → UpBlock_32: trilinear×2 + Conv → 32×16³
            → UpBlock_16: trilinear×2 + Conv → 16×32³
            → trilinear×2 + Conv3d(16→1)     → 1×64³  (identity activation)
```

Final Conv3d is zero-initialized for stable early training.

### Reparameterization

During training: `z = μ + σ⊙ε`, `ε ~ N(0, I)` — stochastic path kept active
to maintain a meaningful KL term and prevent posterior collapse.

During evaluation (`generate()`): defaults to `stochastic=False` (uses `z = μ`)
for deterministic reconstruction. Pass `stochastic=True` for novel generation.

---

## Loss Function

```
L = λ1·L_recon  +  λ2·L_PC  +  λ3·L_orth  +  λ4_w·KL
```

| Term | Default λ | Description |
|------|-----------|-------------|
| `L_recon` | 1.0 | Masked MSE: mean squared error over in-brain voxels only |
| `L_PC`    | 0.1 | Spatial Pearson alignment: (1 − ρ_ik) |
| `L_orth`  | 0.5 | Gram-matrix soft orthogonality |
| `KL`      | 0.001 | VAE regulariser (cosine-annealed from 0) |

**Rationale for λ weights:** `L_PC` is intentionally down-weighted (`λ2=0.1`)
to prevent the model from collapsing to copying ICA maps directly. `L_orth` is
up-weighted (`λ3=0.5`) to actively enforce diverse structural decompositions.
The reconstruction loss is applied over in-brain voxels using a global brain
mask, preventing wasted capacity on background zeros.

---

## Evaluation Metrics

| Metric   | Direction | Reference threshold (Luo et al. 2020) |
|----------|-----------|----------------------------------------|
| RECON    | ↓         | ≤ 0.10                                 |
| PC       | ↑         | > 0.25 (significance)                  |
| PC_025   | ↑         | > 0.62                                 |
| ISC      | ↑         | > 0.50                                 |

---

## Data Pipeline

### Disk Cache

On first access, each subject's NIfTI files are loaded, resampled (1mm → 3mm
ICA grid via nilearn), padded to 64³, and saved as
`data_loader/tensor_cache/subject_<id>.pt`. All subsequent runs load the
preprocessed tensors directly, skipping the expensive resample step.

Set `CACHE_DIR = None` in `datasets.py` to disable caching (e.g. when storage
is limited). First run over 700 subjects will be slow regardless; every
subsequent run benefits from the cache.

### Preprocessing

1. **ICA maps** — loaded as (x, y, z, 53), transposed to (53, x, y, z),
   zero-padded to (53, 64, 64, 64). Voxels with |Z| < 0.2 suppressed.
2. **sMRI** — resampled to ICA grid (~3mm), zero-padded to (1, 64, 64, 64).
3. **Normalisation** — training-set mean sMRI subtracted at runtime (computed
   once, stored as `train_smri_mean`).
4. **Brain mask** — union of non-zero ICA voxels from the first training subject,
   used to restrict `L_recon` to in-brain voxels.

---

## Training Monitoring

The trainer emits warnings for:
- **NaN/Inf in loss** — halts immediately with per-term diagnostics
- **KL < 1e-4 for 5 consecutive epochs** — possible posterior collapse
- **Component std < 1e-5** — possible mode collapse (all 53 components identical)

`valid.csv` tracks all loss terms per epoch, enabling early detection of
training issues before full evaluation.

---

## Key Design Decisions

**Why masked MSE, not normalised MSE?**
The reconstruction loss is `mean((X - Σŝ_ik)²)` over in-brain voxels,
not divided by `‖X‖²`. This keeps the loss scale stable across subjects with
varying intensity ranges and is consistent with the evaluation metric.

**Why Pearson, not cosine similarity?**
Pearson = cosine on mean-centred vectors. ICA maps are Z-scored (≈ zero mean);
generated components carry intensity offsets from the sMRI background signal.
Cosine is sensitive to these offsets; Pearson is not.

**Why encode sMRI, not the 53 ICA maps?**
The sMRI contains the subject's full anatomical fingerprint as a dense signal.
The 53 ICA maps are already low-dimensional sparse statistical maps. Encoding
the sMRI into `z_i` gives a compact subject-specific anatomical prior; each ICA
map `e_ik` then acts as a functional query: *which part of this anatomy supports
this network?*

**Why shared decoder?**
Sharing weights across all 53 components enforces the biological universality
prior and reduces parameters significantly. Components differ only via their ICA
conditioning vector `e_ik`.

**Why chunked decoder?**
Decoding all B×K=212 components simultaneously would hold the full upsampling
path in VRAM at once. Chunking over 32 (subject × component) pairs at a time
bounds peak memory while preserving correct gradients.

**Ablation 4 — permuted component assignment**
The model forward pass uses the *correct* ICA conditioning. Only the
evaluation pairing (which generated component is scored against which ICA map)
is permuted. A drop in PC confirms the ICA conditioning drives spatial
specificity, not the latent code alone.