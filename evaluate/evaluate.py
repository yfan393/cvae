"""
evaluate/evaluate.py
─────────────────────────────────────────────────────────────────
Full test-set evaluation for stacked-ICA C3D-VAE.
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
from nilearn import plotting
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_utils import load_model, format_eval_table
from trainer.metric import compute_all_metrics, compute_isc
from data_loader.datasets import UKBBData


_AFFINE = np.array(
    [[-3, 0, 0, 93],
     [0, 3, 0, -112],
     [0, 0, 3, -88],
     [0, 0, 0, 1]],
    dtype=np.float32,
)


@torch.no_grad()
def run_inference(model, loader, device, mean, use_amp=True):
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_amp_ = use_amp and device.type == "cuda"

    smri_l, comp_l, ica_l, mu_l = [], [], [], []

    model.eval()

    for smri, ica_stacked in loader:
        B = smri.size(0)

        s = smri.to(device, non_blocking=True)
        c = ica_stacked.to(device, non_blocking=True)

        s = s - mean.expand(B, -1, -1, -1, -1)

        with autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp_,
        ):
            out = model(s, c)
            comps = out["components"]

        smri_l.append(s.cpu().float())
        comp_l.append(comps.cpu().float())
        ica_l.append(c.cpu().float())
        mu_l.append(out["mu"].cpu().float())

    return (
        torch.cat(smri_l),
        torch.cat(comp_l),
        torch.cat(ica_l),
        torch.cat(mu_l),
    )


def report_metrics(smri_all, comps_all, ica_all, mask, save_dir, compute_mi=True):
    N = smri_all.size(0)

    m = compute_all_metrics(
        smri=smri_all,
        components=comps_all,
        ica_stacked=ica_all,
        mask=mask,
        compute_mi_flag=compute_mi,
    )

    isc, isc_k = compute_isc(comps_all)
    m["ISC"] = isc
    m["ISC_k"] = isc_k.numpy()

    print(format_eval_table(m, N))

    row = {
        k: m[k]
        for k in ["RECON", "PC", "PC_025", "MI", "MI_02", "ISC"]
        if k in m
    }
    row["N"] = N

    pd.DataFrame([row]).to_csv(save_dir / "metrics.csv", index=False)
    print("  → metrics.csv")

    return m


def _nii(vol, mask):
    return nib.Nifti1Image(
        np.where(mask, vol, np.nan).astype(np.float32),
        _AFFINE,
    )


def plot_rho_histogram(rho, save_dir):
    rho_np = rho.numpy().ravel()
    frac = (rho_np > 0.25).mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        rho_np,
        bins=80,
        range=(-1, 1),
        color="steelblue",
        alpha=0.85,
        edgecolor="white",
        lw=0.3,
    )
    ax.axvline(
        0.25,
        color="red",
        lw=1.5,
        label=f"rho=0.25 threshold, PC_025={frac:.3f}",
    )
    ax.axvline(
        rho_np.mean(),
        color="orange",
        lw=1.5,
        label=f"Mean rho={rho_np.mean():.3f}",
    )

    ax.set_xlabel("Spatial Pearson rho_ik")
    ax.set_ylabel("Count")
    ax.set_title("Structure-function alignment")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_dir / "rho_histogram.png", dpi=150)
    plt.close(fig)
    print("  → rho_histogram.png")


def plot_isc_bar(isc_k, save_dir):
    K = len(isc_k)
    order = np.argsort(isc_k)

    colors = [
        "#2196F3" if v > 0.5 else ("#FF9800" if v > 0 else "#F44336")
        for v in isc_k
    ]

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.bar(range(K), isc_k, color=colors, alpha=0.85, edgecolor="none")

    for i in order[-5:]:
        ax.text(i, isc_k[i] + 0.01, str(i), ha="center", fontsize=7)

    for i in order[:5]:
        ax.text(i, isc_k[i] - 0.05, str(i), ha="center", fontsize=7)

    ax.axhline(0.5, color="green", linestyle="--", lw=1)
    ax.axhline(0.0, color="black", lw=0.5)

    ax.legend(
        handles=[
            mpatches.Patch(color="#2196F3", label="ISC > 0.5"),
            mpatches.Patch(color="#FF9800", label="0 < ISC <= 0.5"),
            mpatches.Patch(color="#F44336", label="ISC <= 0"),
            plt.Line2D([0], [0], color="green", linestyle="--", label="ISC=0.5"),
        ]
    )

    ax.set_xlabel("ICA component index k")
    ax.set_ylabel("ISC_k")
    ax.set_title("Inter-subject consistency per generated component")

    plt.tight_layout()
    fig.savefig(save_dir / "isc_per_component.png", dpi=150)
    plt.close(fig)
    print("  → isc_per_component.png")


def plot_latent_pca(mu_all, save_dir):
    mu_np = mu_all.numpy()

    pca = PCA(n_components=2)
    z2 = pca.fit_transform(mu_np)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        z2[:, 0],
        z2[:, 1],
        c=np.arange(len(mu_np)),
        cmap="viridis",
        s=15,
        alpha=0.7,
    )

    plt.colorbar(sc, ax=ax, label="Subject index")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% var)")
    ax.set_title(f"PCA of latent codes z_i, N={len(mu_np)}")

    plt.tight_layout()
    fig.savefig(save_dir / "latent_pca.png", dpi=150)
    plt.close(fig)
    print("  → latent_pca.png")


def plot_reconstruction(smri_v, sum_v, mask, save_dir, s_idx):
    res = smri_v - sum_v

    err = float(
        np.mean(res[mask] ** 2)
        / (np.mean(smri_v[mask] ** 2) + 1e-8)
    )

    vmin = smri_v[mask].min()
    vmax = smri_v[mask].max()

    fig, axes = plt.subplots(3, 1, figsize=(20, 15))

    for ax, vol, title, use_range in zip(
        axes,
        [smri_v, sum_v, res],
        [
            "Original sMRI",
            "Sum of generated components",
            f"Residual (RECON={err:.4f})",
        ],
        [True, True, False],
    ):
        plotting.plot_epi(
            _nii(vol, mask),
            axes=ax,
            title=title,
            display_mode="z",
            cut_coords=8,
            vmin=vmin if use_range else None,
            vmax=vmax if use_range else None,
            colorbar=True,
        )

    plt.suptitle(f"Reconstruction — subject {s_idx}", fontsize=13)
    plt.tight_layout()

    out = save_dir / f"reconstruction_subj{s_idx:03d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.name}")


def plot_all_components(comps, ica_stacked, mask, rho, save_dir, s_idx):
    comp_dir = save_dir / "components" / f"subj{s_idx:03d}"
    comp_dir.mkdir(parents=True, exist_ok=True)

    K = comps.size(1)
    rho_k = rho[s_idx].numpy()

    z_c = [-30, -10, 10, 30, 50]
    y_c = [-90, -55, -20, 15, 50]

    for k in range(K):
        iv = ica_stacked[s_idx, k].numpy()
        cv = comps[s_idx, k].numpy()

        vm_i = max(float(np.nanmax(np.abs(np.where(mask, iv, 0)))), 1e-6)
        vm_c = max(float(np.nanmax(np.abs(np.where(mask, cv, 0)))), 1e-6)

        fig, axes = plt.subplots(2, 2, figsize=(22, 10))

        plotting.plot_stat_map(
            _nii(iv, mask),
            axes=axes[0, 0],
            title=f"ICA {k} — axial",
            cut_coords=z_c,
            display_mode="z",
            vmax=vm_i,
            colorbar=True,
            symmetric_cbar=True,
        )

        plotting.plot_stat_map(
            _nii(cv, mask),
            axes=axes[0, 1],
            title=f"Generated {k} — axial (rho={rho_k[k]:.3f})",
            cut_coords=z_c,
            display_mode="z",
            vmax=vm_c,
            colorbar=True,
            symmetric_cbar=True,
        )

        plotting.plot_stat_map(
            _nii(iv, mask),
            axes=axes[1, 0],
            title=f"ICA {k} — coronal",
            cut_coords=y_c,
            display_mode="y",
            vmax=vm_i,
            colorbar=True,
            symmetric_cbar=True,
        )

        plotting.plot_stat_map(
            _nii(cv, mask),
            axes=axes[1, 1],
            title=f"Generated {k} — coronal",
            cut_coords=y_c,
            display_mode="y",
            vmax=vm_c,
            colorbar=True,
            symmetric_cbar=True,
        )

        plt.tight_layout()
        fig.savefig(
            comp_dir / f"component_{k:02d}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"  → {K} component plots for subject {s_idx}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate stacked-ICA C3D-VAE")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_dir", default="eval_results")
    parser.add_argument("--num_subjects", type=int, default=140)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--vis_subjects", type=int, default=3)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_mi", action="store_true",
                        help="Skip mutual information computation (faster)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = UKBBData(split="test", num_subjects=args.num_subjects)
    train_ds = UKBBData(split="train", num_subjects=50)

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    mean = train_ds.train_smri_mean.to(device)

    if train_ds.global_mask is not None:
        mask = train_ds.global_mask.unsqueeze(0).float()
    else:
        mask = None

    model = load_model(args.checkpoint, device)
    print(model)

    print("\nRunning inference ...")
    smri_all, comps_all, ica_all, mu_all = run_inference(
        model=model,
        loader=loader,
        device=device,
        mean=mean,
        use_amp=not args.no_amp,
    )

    print(
        f"  smri {tuple(smri_all.shape)}, "
        f"components {tuple(comps_all.shape)}, "
        f"ica_stacked {tuple(ica_all.shape)}"
    )

    if mask is not None:
        mask_all = mask.expand(smri_all.size(0), -1, -1, -1, -1)
    else:
        mask_all = torch.ones_like(smri_all)

    m = report_metrics(
        smri_all=smri_all,
        comps_all=comps_all,
        ica_all=ica_all,
        mask=mask_all,
        save_dir=save_dir,
        compute_mi=not args.no_mi,
    )

    print("\nGenerating plots ...")
    plot_rho_histogram(m["rho"], save_dir)

    if "ISC_k" in m:
        plot_isc_bar(m["ISC_k"], save_dir)

    plot_latent_pca(mu_all, save_dir)

    mask_np = mask_all[0, 0].numpy().astype(bool)
    n_vis = min(args.vis_subjects, smri_all.size(0))

    for s in range(n_vis):
        plot_reconstruction(
            smri_all[s, 0].numpy(),
            comps_all[s].sum(dim=0).numpy(),
            mask_np,
            save_dir,
            s,
        )

        plot_all_components(
            comps_all,
            ica_all,
            mask_np,
            m["rho"],
            save_dir,
            s,
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()