"""
ablation/run_ablations.py
─────────────────────────────────────────────────────────────────
Ablation studies for stacked-ICA C3D-VAE.

Updated for dataloader returning:
  smri        : (B, 1,  64, 64, 64)
  ica_stacked : (B, 53, 64, 64, 64)
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_utils import load_model, format_ablation_table
from trainer.metric import compute_all_metrics, compute_isc
from data_loader.datasets import UKBBData


def make_loader(num_subjects: int, batch_size: int) -> DataLoader:
    ds = UKBBData(split="valid", num_subjects=num_subjects)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )


@torch.no_grad()
def run_forward(
    model,
    loader,
    device,
    mean,
    ica_mode: str = "normal",
) -> tuple:
    smri_l, comp_l, ica_l = [], [], []

    model.eval()

    for smri, ica_stacked in loader:
        B = smri.size(0)

        s = smri.to(device, non_blocking=True)
        c = ica_stacked.to(device, non_blocking=True)

        s = s - mean.expand(B, -1, -1, -1, -1)

        if ica_mode == "zero":
            c_in = torch.zeros_like(c)

        elif ica_mode == "perm_within":
            idx = torch.randperm(c.size(1), device=device)
            c_in = c[:, idx, ...]

        elif ica_mode == "perm_across" and B > 1:
            idx = torch.randperm(B, device=device)
            c_in = c[idx]

        elif ica_mode == "perm_eval":
            # Ablation 4: the model forward pass uses the *correct* ICA maps
            # (c_in = c).  The permutation is applied only to the evaluation
            # pairing below, after inference.  This branch is intentionally
            # the same as "normal" for the forward pass — the scrambling
            # happens at metric-computation time so that any drop in PC/MI
            # reveals whether components are specifically tied to their ICA maps.
            c_in = c

        else:
            # "normal" mode: correct ICA conditioning, correct eval pairing.
            c_in = c

        out = model(s, c_in)
        comps = out["components"]

        if ica_mode == "perm_eval":
            # Scramble the component→ICA assignment used for metric computation.
            # Generated component k is evaluated against ICA map idx[k],
            # breaking the correct pairing; any drop in PC/MI is attributable
            # to the ICA conditioning signal driving spatial specificity.
            idx = torch.randperm(c.size(1), device=device)
            c_eval = c[:, idx, ...]
        else:
            c_eval = c

        smri_l.append(s.cpu())
        comp_l.append(comps.cpu())
        ica_l.append(c_eval.cpu())

    return torch.cat(smri_l), torch.cat(comp_l), torch.cat(ica_l)


def evaluate(smri_a, comp_a, ica_a, mask) -> dict:
    N = smri_a.size(0)

    if mask is not None:
        mask_e = mask.detach().cpu().expand(N, -1, -1, -1, -1)
    else:
        mask_e = None

    m = compute_all_metrics(
        smri=smri_a,
        components=comp_a,
        ica_stacked=ica_a,
        mask=mask_e,
        compute_mi_flag=True,
    )

    isc, isc_k = compute_isc(comp_a)
    m["ISC"] = isc
    m["ISC_k"] = isc_k.numpy()

    return m


def _print(label: str, m: dict):
    print(f"\n{'─' * 60}\n  {label}\n{'─' * 60}")

    for k in ["RECON", "PC", "PC_025", "MI", "MI_02", "ISC"]:
        if k in m:
            print(f"  {k:<10}: {m[k]:.5f}")


def _save(label: str, m: dict, save_dir: Path):
    row = {
        k: m[k]
        for k in ["RECON", "PC", "PC_025", "MI", "MI_02", "ISC"]
        if k in m
    }
    row["ablation"] = label

    pd.DataFrame([row]).to_csv(
        save_dir / f"{label.replace(' ', '_')}.csv",
        index=False,
    )


def plot_isc_bar(isc_k: np.ndarray, save_dir: Path, tag: str = "full"):
    K = len(isc_k)

    colors = [
        "#2196F3" if v > 0.5 else ("#FF9800" if v > 0 else "#F44336")
        for v in isc_k
    ]

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.bar(range(K), isc_k, color=colors, alpha=0.85, edgecolor="none")

    order = np.argsort(isc_k)

    for i in order[-5:]:
        ax.text(i, isc_k[i] + 0.01, str(i), ha="center", fontsize=7)

    for i in order[:5]:
        ax.text(i, isc_k[i] - 0.05, str(i), ha="center", fontsize=7)

    ax.axhline(0.5, color="green", linestyle="--", lw=1, label="ISC=0.5")
    ax.set_xlabel("ICA component k")
    ax.set_ylabel("ISC_k")
    ax.set_title(f"ISC per component ({tag})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_dir / f"isc_per_component_{tag}.png", dpi=150)
    plt.close(fig)


def ablation1(model, loader, device, mean, mask, save_dir):
    print("\n=== Ablation 1: No ICA conditioning ===")

    smri_f, comp_f, ica_f = run_forward(
        model=model,
        loader=loader,
        device=device,
        mean=mean,
        ica_mode="normal",
    )

    smri_z, comp_z, ica_z = run_forward(
        model=model,
        loader=loader,
        device=device,
        mean=mean,
        ica_mode="zero",
    )

    m_f = evaluate(smri_f, comp_f, ica_f, mask)
    m_z = evaluate(smri_z, comp_z, ica_z, mask)

    _print("Full C3D-VAE", m_f)
    _print("No ICA conditioning", m_z)

    _save("Abl1_full", m_f, save_dir)
    _save("Abl1_no_cond", m_z, save_dir)

    keys = ["RECON", "PC", "PC_025", "MI", "MI_02", "ISC"]

    deltas = [
        -(m_f.get(k, 0) - m_z.get(k, 0)) if k == "RECON"
        else m_f.get(k, 0) - m_z.get(k, 0)
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        keys,
        deltas,
        color=["#4CAF50" if d > 0 else "#F44336" for d in deltas],
        alpha=0.85,
        edgecolor="none",
    )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Ablation 1: Full minus no conditioning")
    ax.set_ylabel("Metric change")

    plt.tight_layout()
    fig.savefig(save_dir / "abl1_delta.png", dpi=150)
    plt.close(fig)


def ablation2(lam2_ckpts, loader, device, mean, mask, save_dir):
    print("\n=== Ablation 2: lambda2 sweep ===")

    if not lam2_ckpts:
        print("  No lambda2 checkpoints. Generating schematic only.")

        lambdas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(lambdas, [1 / (1 + l) for l in lambdas], "b-o")
        ax1.set(xlabel="lambda2", title="Expected RECON")

        ax2.plot(lambdas, [l / (1 + l) for l in lambdas], "g-o")
        ax2.set(xlabel="lambda2", title="Expected PC")

        plt.suptitle("Ablation 2 schematic")
        plt.tight_layout()

        fig.savefig(save_dir / "abl2_schematic.png", dpi=150)
        plt.close(fig)

        return

    results = []

    for lam, ckpt_path in lam2_ckpts:
        print(f"  lambda2={lam} from {ckpt_path}")

        m_model = load_model(ckpt_path, device)

        s, c, i = run_forward(
            model=m_model,
            loader=loader,
            device=device,
            mean=mean,
        )

        m = evaluate(s, c, i, mask)
        m["lambda2"] = lam

        results.append(m)
        _save(f"Abl2_lam{lam}", m, save_dir)

    df = pd.DataFrame(
        [
            {k: r[k] for k in ["lambda2", "RECON", "PC", "MI"]}
            for r in results
        ]
    ).sort_values("lambda2")

    df.to_csv(save_dir / "abl2_lambda2_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col, color, title in zip(
        axes,
        ["RECON", "PC", "MI"],
        ["tomato", "steelblue", "seagreen"],
        ["RECON", "PC", "MI"],
    ):
        ax.plot(df["lambda2"], df[col], "o-", color=color)
        ax.set(xlabel="lambda2", title=title)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Ablation 2: reconstruction-alignment trade-off")
    plt.tight_layout()

    fig.savefig(save_dir / "abl2_lambda2_sweep.png", dpi=150)
    plt.close(fig)


def ablation3(model, loader, device, mean, mask, save_dir):
    print("\n=== Ablation 3: Permuted ICA conditioning ===")

    sf, cf, if_ = run_forward(model, loader, device, mean, "normal")
    sw, cw, iw = run_forward(model, loader, device, mean, "perm_within")
    sx, cx, ix = run_forward(model, loader, device, mean, "perm_across")

    m_f = evaluate(sf, cf, if_, mask)
    m_w = evaluate(sw, cw, iw, mask)
    m_x = evaluate(sx, cx, ix, mask)

    _print("Full", m_f)
    _print("Permuted within-subject", m_w)
    _print("Permuted across-subject", m_x)

    _save("Abl3_full", m_f, save_dir)
    _save("Abl3_perm_within", m_w, save_dir)
    _save("Abl3_perm_across", m_x, save_dir)

    keys = ["PC", "PC_025", "MI", "MI_02", "ISC"]
    x = np.arange(len(keys))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x - w, [m_f.get(k, 0) for k in keys], w, label="Full")
    ax.bar(x, [m_w.get(k, 0) for k in keys], w, label="Perm within")
    ax.bar(x + w, [m_x.get(k, 0) for k in keys], w, label="Perm across")

    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_title("Ablation 3: ICA permutation")
    ax.set_ylabel("Metric value")
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_dir / "abl3_permuted_ica.png", dpi=150)
    plt.close(fig)


def ablation4(model, loader, device, mean, mask, save_dir):
    print("\n=== Ablation 4: Permuted component assignment ===")

    sf, cf, if_ = run_forward(model, loader, device, mean, "normal")
    sp, cp, ip = run_forward(model, loader, device, mean, "perm_eval")

    m_f = evaluate(sf, cf, if_, mask)
    m_p = evaluate(sp, cp, ip, mask)

    _print("Full", m_f)
    _print("Permuted component-ICA pair", m_p)

    _save("Abl4_full", m_f, save_dir)
    _save("Abl4_comp_perm", m_p, save_dir)

    keys = ["RECON", "PC", "PC_025", "MI", "MI_02"]

    deltas = [
        -(m_f.get(k, 0) - m_p.get(k, 0)) if k == "RECON"
        else m_f.get(k, 0) - m_p.get(k, 0)
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        keys,
        deltas,
        color=["#4CAF50" if d > 0 else "#F44336" for d in deltas],
        alpha=0.85,
        edgecolor="none",
    )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Ablation 4: Full minus permuted assignment")
    ax.set_ylabel("Metric change")

    plt.tight_layout()
    fig.savefig(save_dir / "abl4_comp_perm.png", dpi=150)
    plt.close(fig)


def build_summary(save_dir: Path):
    dfs = [pd.read_csv(p) for p in sorted(save_dir.glob("Abl*.csv"))]

    if not dfs:
        return

    summary = pd.concat(dfs, ignore_index=True)
    cols = ["ablation", "RECON", "PC", "PC_025", "MI", "MI_02", "ISC"]
    summary = summary[[c for c in cols if c in summary.columns]]

    summary.to_csv(save_dir / "ablation_summary.csv", index=False)

    print(format_ablation_table(summary))
    print("  → ablation_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="C3D-VAE ablation studies")

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_dir", default="ablation_results")
    parser.add_argument("--num_subjects", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--ablation",
        default="all",
        choices=["all", "1", "2", "3", "4"],
    )
    parser.add_argument(
        "--lambda2_checkpoints",
        nargs="*",
        default=[],
        metavar="LAM:PATH",
    )

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = make_loader(args.num_subjects, args.batch_size)

    train_ds = UKBBData(split="train", num_subjects=args.num_subjects)
    mean = train_ds.train_smri_mean.to(device)

    if train_ds.global_mask is not None:
        mask = train_ds.global_mask.unsqueeze(0)
    else:
        mask = None

    model = None

    if args.checkpoint:
        model = load_model(args.checkpoint, device)
        print(f"Model loaded from {args.checkpoint}")

    lam2_ckpts = []

    for pair in args.lambda2_checkpoints:
        lam_str, ckpt_path = pair.split(":")
        lam2_ckpts.append((float(lam_str), ckpt_path))

    if args.ablation in ("all", "1"):
        assert model is not None, "Ablation 1 requires --checkpoint"
        ablation1(model, loader, device, mean, mask, save_dir)

    if args.ablation in ("all", "2"):
        ablation2(lam2_ckpts, loader, device, mean, mask, save_dir)

    if args.ablation in ("all", "3"):
        assert model is not None, "Ablation 3 requires --checkpoint"
        ablation3(model, loader, device, mean, mask, save_dir)

    if args.ablation in ("all", "4"):
        assert model is not None, "Ablation 4 requires --checkpoint"
        ablation4(model, loader, device, mean, mask, save_dir)

    if model is not None:
        sf, cf, if_ = run_forward(model, loader, device, mean)
        m_full = evaluate(sf, cf, if_, mask)

        if "ISC_k" in m_full:
            plot_isc_bar(m_full["ISC_k"], save_dir, tag="full")

    build_summary(save_dir)


if __name__ == "__main__":
    main()