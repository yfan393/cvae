"""
data_loader/datasets.py
─────────────────────────────────────────────────────────────────
UK Biobank neuroimaging dataset for C3D-VAE.

Expected CSV columns (ukbb.csv):
  subject_id : int   unique subject ID
  ica_path   : str   NIfTI (x, y, z, 53)  — 53 ICA Z-score spatial maps
  smri_path  : str   NIfTI (121, 145, 121) — T1w structural MRI
  gm_path    : str   (optional) FSL FAST GM probability map, same voxel grid as sMRI

Each __getitem__ returns a 2-tuple:
  smri        (1,  64, 64, 64) float32  — sMRI resampled to ICA grid, zero-padded
  ica_stacked (53, 64, 64, 64) float32  — ICA maps, zero-padded, |Z|<0.2 suppressed

Training split additionally exposes:
  .global_mask      (1, 64, 64, 64) bool    union brain mask for visualisation
  .train_smri_mean  (1, 64, 64, 64) float   population sMRI mean for normalisation
"""

import torch
import numpy as np
import pandas as pd
import nibabel as nb
from pathlib import Path
from typing import Optional, Tuple
from torch.nn import functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from nilearn.image import resample_to_img

CSV_PATH = Path(__file__).parent / 'ukbb.csv'

# Preprocessed tensor cache directory.
# Set to None to disable caching (e.g. when storage is limited).
# When set, _load() saves (smri, ica_stacked) as subject_<id>.pt on first
# access and reads from disk on all subsequent runs — skipping the expensive
# NIfTI load + nilearn resample step entirely.
CACHE_DIR = Path(__file__).parent / 'tensor_cache'

# Set True to load all subjects at __init__ time (small datasets / ablations)
PRELOAD = False

# Noise suppression threshold for ICA Z-score maps (Du et al., 2020)
ICA_THRESHOLD = 0.2

# GM probability threshold for FSL FAST masks (VBM standard)
GM_PROB_THRESHOLD = 0.5


class UKBBData(Dataset):
    """
    UK Biobank dataset — 72/8/20 train/valid/test split, random seed 42.

    Parameters
    ----------
    split        : 'train' | 'valid' | 'test'
    num_subjects : cap on number of subjects per split  (None = all)
    target_size  : spatial side length after padding    (default 64)
    """

    _TEST_FRAC = 0.20
    _VALID_OF_TV = 0.10   # 10% of 80% = 8% overall

    def __init__(
        self,
        split: str,
        num_subjects: Optional[int] = None,
        target_size: int = 64
    ):
        super().__init__()
        self.split = split
        self.T = target_size   # short alias used in padding

        # ── 1. Read manifest and build splits ────────────────────────────────
        df = pd.read_csv(CSV_PATH, index_col=0)
        subjects = np.unique(df['subject_id'].values)

        tv, test = train_test_split(
            subjects,
            test_size=self._TEST_FRAC,
            shuffle=True,
            random_state=42
        )
        train, val = train_test_split(
            tv,
            test_size=self._VALID_OF_TV,
            shuffle=True,
            random_state=42
        )

        split_map = {'train': train, 'valid': val, 'test': test}
        if split not in split_map:
            raise ValueError(f"split must be 'train'/'valid'/'test', got '{split}'")

        self.df = (
            df.loc[df['subject_id'].isin(split_map[split])]
              .copy()
              .reset_index(drop=True)
        )

        if num_subjects is not None:
            self.df = self.df.iloc[:num_subjects].copy()

        self.subjects = self.df['subject_id'].tolist()
        self.has_gm = 'gm_path' in self.df.columns

        # ── 2. Lazy-load cache ────────────────────────────────────────────────
        # Each slot is None until first access, then holds:
        #   (smri, ica_stacked)
        #
        # DataLoader workers each own their own copy of this list after fork,
        # so cache entries written by worker N are visible only to worker N —
        # no inter-process locking needed.
        self._cache: list = [None] * len(self.df)

        if PRELOAD:
            print(f"[UKBBData:{split}] Pre-loading {len(self.df)} subjects …")
            for i in range(len(self.df)):
                self._cache[i] = self._load(i)

        # ── 3. Training-split extras ──────────────────────────────────────────
        if split == 'train':
            # Global brain mask: union of non-zero voxels across the first
            # subject's ICA stack. Shared across training as a visualisation aid.
            ica_ref = nb.load(self.df.iloc[0]['ica_path'])
            ica0_np = ica_ref.get_fdata(dtype=np.float32)   # (x,y,z,53)
            union_np = (np.abs(ica0_np).sum(axis=-1) > 0).astype(np.float32)

            self.global_mask = self._pad(
                torch.from_numpy(union_np).unsqueeze(0)      # (1,x,y,z)
            ).bool()                                        # (1,64,64,64)

            # Population mean sMRI. Iterating __getitem__ fills the cache at
            # the same time; each subject is loaded exactly once.
            print(
                f"[UKBBData:train] Computing mean sMRI over "
                f"{len(self.df)} subjects …"
            )

            acc = torch.zeros(1, self.T, self.T, self.T)

            for i in range(len(self.df)):
                smri, _ = self[i]
                acc.add_(smri)

            self.train_smri_mean = acc.div_(len(self.df))   # (1,64,64,64)

            # Sanity-check the mean
            mean_val = self.train_smri_mean.mean().item()
            assert not np.isnan(mean_val), \
                "[UKBBData] train_smri_mean contains NaN — check sMRI paths"
            assert mean_val != 0, \
                "[UKBBData] train_smri_mean is all zeros — check sMRI loading"

            print(f"[UKBBData:train] mean sMRI intensity: {mean_val:.4f}")

        else:
            self.global_mask = None
            self.train_smri_mean = None

    # ── Subject loading ───────────────────────────────────────────────────────

    def _load(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one subject, using a disk cache to skip the expensive NIfTI
        load + nilearn resample on all runs after the first.

        Cache layout:  CACHE_DIR/subject_<id>.pt  →  (smri, ica_stacked)

        First access: load from NIfTI → preprocess → save .pt
        Subsequent:   torch.load .pt directly  (no NIfTI, no resample)

        Returns
        -------
        smri        : (1,  64, 64, 64) float32
        ica_stacked : (53, 64, 64, 64) float32
        """
        row = self.df.iloc[idx]
        subject_id = row['subject_id']

        # ── Try cache first ───────────────────────────────────────────────────
        if CACHE_DIR is not None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = CACHE_DIR / f"subject_{subject_id}.pt"

            if cache_path.exists():
                smri_t, ica_stacked = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=True,
                )
                return smri_t, ica_stacked

        # ── Cache miss: load and preprocess from NIfTI ────────────────────────

        # ICA: raw shape (x, y, z, 53) → (53, x, y, z)
        ica_img = nb.load(row['ica_path'])
        ica_np = ica_img.get_fdata(dtype=np.float32)              # (x,y,z,53)
        ica_np = np.ascontiguousarray(np.moveaxis(ica_np, -1, 0)) # (53,x,y,z)

        ica_stacked = torch.from_numpy(ica_np).float()            # (53,x,y,z)
        ica_stacked = self._pad(ica_stacked)                      # (53,64,64,64)

        # Suppress noise: voxels with |Z| < 0.2 carry no meaningful signal.
        ica_stacked = ica_stacked * (ica_stacked.abs() > ICA_THRESHOLD)

        # sMRI: resample 1mm → ICA grid (~3mm), then pad
        smri_img = nb.load(row['smri_path'])
        smri_img = resample_to_img(
            smri_img,
            ica_img,
            force_resample=True,
            copy_header=True,
        )

        smri_np = np.ascontiguousarray(smri_img.get_fdata(dtype=np.float32))
        smri_t = self._pad(
            torch.from_numpy(smri_np).unsqueeze(0)
        )                                                          # (1,64,64,64)

        # ── Write cache ───────────────────────────────────────────────────────
        if CACHE_DIR is not None:
            torch.save((smri_t, ica_stacked), cache_path)

        return smri_t, ica_stacked

    # ── Vectorised padding ────────────────────────────────────────────────────

    def _pad(self, data: torch.Tensor) -> torch.Tensor:
        """
        Symmetric zero-pad any (C, D, H, W) tensor to (C, T, T, T).

        Works for C=1 and C=53 with a single F.pad call.
        If any axis >= T it is left unchanged.

        F.pad argument order:
          (left_w, right_w, left_h, right_h, left_d, right_d)
        """
        assert data.ndim == 4, f"Expected (C,D,H,W), got {tuple(data.shape)}"

        _, d, h, w = data.shape
        T = self.T

        def _sym(n: int):
            p = max(0, T - n)
            return p // 2, p - p // 2

        pd, ph, pw = _sym(d), _sym(h), _sym(w)

        return F.pad(
            data,
            (pw[0], pw[1], ph[0], ph[1], pd[0], pd[1]),
            mode='constant',
            value=0
        )

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lazy load with RAM cache.

        Returns
        -------
        smri : torch.Tensor
            Shape (1, 64, 64, 64).

        ica_stacked : torch.Tensor
            Shape (53, 64, 64, 64).
        """
        if self._cache[idx] is None:
            self._cache[idx] = self._load(idx)

        return self._cache[idx]