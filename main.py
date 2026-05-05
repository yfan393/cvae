"""
main.py
─────────────────────────────────────────────────────────────────
Training entry point for stacked-ICA C3D-VAE.
"""

import argparse
import json
import traceback

import torch
import numpy as np

from model.cvae import C3DVAE
from trainer.trainer import CVAETrainer
from data_loader.data_loaders import UKBB


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train stacked-ICA C3D-VAE")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--data_config", required=True)
    args = parser.parse_args()

    run_cfg = _load(args.config)
    model_cfg = _load(args.model_config)
    data_cfg = _load(args.data_config)

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED = run_cfg.get("seed", 42)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    print(f"Seed: {SEED}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dc = data_cfg["data"]["args"]

    num_sub = dc.get("num_subjects", 700)
    batch_size = dc.get("batch_size", 4)
    num_workers = dc.get("num_workers", 4)

    train_ukbb = UKBB(
        split="train",
        num_subjects=num_sub,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    valid_ukbb = UKBB(
        split="valid",
        batch_size=batch_size,
        num_workers=num_workers,
    )

    assert hasattr(train_ukbb.loader, "dataset"), \
        "DataLoader missing .dataset"

    assert hasattr(train_ukbb.loader.dataset, "global_mask"), \
        "Dataset missing .global_mask — ensure split='train'"

    assert hasattr(train_ukbb.loader.dataset, "train_smri_mean"), \
        "Dataset missing .train_smri_mean — ensure split='train'"

    # ── Model ────────────────────────────────────────────────────────────────
    mc = model_cfg["arch"]["args"]

    model = C3DVAE(
        num_components=mc.get("num_components", 53),
        latent_dim=mc.get("latent_dim", 128),
        cond_dim=mc.get("cond_dim", 64),
        smri_embedding_dim=mc.get("smri_embedding_dim", 64),
        dropout=mc.get("dropout", 0.1),
    ).to(device)

    print(model)

    # ── Optimizer ────────────────────────────────────────────────────────────
    opt_args = run_cfg.get("optimizer", {}).get("args", {})

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_args.get("lr", 3e-4),
        weight_decay=opt_args.get("weight_decay", 1e-5),
        amsgrad=opt_args.get("amsgrad", True),
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    lw = run_cfg.get("loss_weights", {})
    tr_cfg = run_cfg.get("trainer", {})

    trainer = CVAETrainer(
        model=model,
        optimizer=optimizer,
        config=run_cfg,
        device=device,
        train_loader=train_ukbb.loader,
        valid_loader=valid_ukbb.loader,
        num_epochs=tr_cfg.get("epochs", 50),
        lambda1=lw.get("lambda1", 1.0),
        lambda2=lw.get("lambda2", 0.5),
        lambda3=lw.get("lambda3", 0.1),
        lambda4=lw.get("lambda4", 0.001),
        warmup_epochs=lw.get("warmup_epochs", 20),
        vis_every=tr_cfg.get("vis_every", 10),
        log_every=tr_cfg.get("log_every", 10),
        save_dir=tr_cfg.get("save_dir", "saved/C3DVAE"),
        patience=tr_cfg.get("patience", 30),
    )

    try:
        trainer.train()
    except Exception:
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()