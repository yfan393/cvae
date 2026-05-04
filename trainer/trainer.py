"""
trainer/trainer.py
─────────────────────────────────────────────────────────────────
CVAETrainer — single-GPU float32 training loop for stacked-ICA C3D-VAE.

Updated for dataloader returning:
  smri        : (B, 1,  64, 64, 64)
  ica_stacked : (B, 53, 64, 64, 64)
"""

import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
import matplotlib
import matplotlib.cm as _mcm
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nibabel.nifti1 import Nifti1Image
from nilearn import plotting
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .loss import CVAELoss
from utils.model_utils import is_bad


CHUNK_SIZE = 32


def _forward_chunked(
    model,
    smri: torch.Tensor,
    ica_stacked: torch.Tensor,
) -> dict:
    """
    Chunked forward pass.

    smri:
      (B, 1, 64, 64, 64)

    ica_stacked:
      (B, K, 64, 64, 64)
    """
    B, K, D, H, W = ica_stacked.shape

    mu, logvar = model.smri_encoder(smri)
    z = model.reparameterise(mu, logvar)

    ica_flat = ica_stacked.reshape(B * K, 1, D, H, W)
    e_flat = model.ica_encoder(ica_flat)

    z_flat = (
        z.unsqueeze(1)
         .expand(B, K, model.latent_dim)
         .reshape(B * K, model.latent_dim)
    )

    chunks = []
    for start in range(0, B * K, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, B * K)
        chunk = model.decoder(z_flat[start:end], e_flat[start:end])
        chunks.append(chunk)

    comp_flat = torch.cat(chunks, dim=0)
    components = comp_flat.reshape(B, K, D, H, W)

    return {
        "components": components,
        "mu": mu,
        "logvar": logvar,
        "ica_embed": e_flat.reshape(B, K, model.cond_dim),
    }


class CVAETrainer:
    _NAMES = ["weighted", "recon", "pc_loss", "orth", "kl"]

    def __init__(
        self,
        model,
        optimizer,
        config,
        device: torch.device,
        train_loader,
        valid_loader,
        num_epochs: int = 200,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        lambda3: float = 0.1,
        lambda4: float = 0.001,
        warmup_epochs: int = 20,
        vis_every: int = 10,
        log_every: int = 10,
        save_dir: str = "saved/C3DVAE",
        patience: int = 30,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.vis_every = vis_every
        self.log_every = log_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience

        self.criterion = CVAELoss(
            lambda1,
            lambda2,
            lambda3,
            lambda4,
            warmup_epochs,
        )

        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=10,
            factor=0.5,
        )

        idx = range(1, num_epochs + 1)

        self.train_df = pd.DataFrame(
            np.zeros((num_epochs, len(self._NAMES))),
            index=idx,
            columns=[f"tr_{n}" for n in self._NAMES],
        )

        self.valid_df = pd.DataFrame(
            np.zeros((num_epochs, len(self._NAMES))),
            index=idx,
            columns=[f"va_{n}" for n in self._NAMES],
        )

        self.best_va_weighted = float("inf")
        self._no_improve_count = 0
        self._kl_low_streak = 0

        self.affine = np.array(
            [
                [-3, 0, 0, 93],
                [0, 3, 0, -112],
                [0, 0, 3, -88],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.global_mask: Optional[torch.Tensor] = None
        self.train_smri_mean: Optional[torch.Tensor] = None

        print(
            f"[Trainer] device={device}  "
            f"chunk_size={CHUNK_SIZE}  "
            f"float32 eager mode"
        )

    def train(self):
        ds = self.train_loader.dataset

        self.global_mask = ds.global_mask.to(self.device)
        self.train_smri_mean = ds.train_smri_mean.to(self.device)

        self._health_check()

        for epoch in range(1, self.num_epochs + 1):
            tr = self._train_epoch(epoch)
            va = self._valid_epoch(epoch)

            for k, v in tr.items():
                self.train_df.loc[epoch, f"tr_{k}"] = float(v)

            for k, v in va.items():
                self.valid_df.loc[epoch, f"va_{k}"] = float(v)

            self.train_df.to_csv(self.save_dir / "train.csv")
            self.valid_df.to_csv(self.save_dir / "valid.csv")

            self.scheduler.step(va["recon"])

            if va["kl"] < 1e-4:
                self._kl_low_streak += 1
                if self._kl_low_streak >= 5:
                    print(
                        "  WARNING: KL < 1e-4 for 5 consecutive epochs. "
                        "Possible posterior collapse."
                    )
            else:
                self._kl_low_streak = 0

            improved = va["weighted"] < self.best_va_weighted

            if improved:
                self.best_va_weighted = va["weighted"]
                self._no_improve_count = 0
                self._save_checkpoint(epoch, best=True)
            else:
                self._no_improve_count += 1

            self._save_checkpoint(epoch, best=False)

            self._plot_loss_curves(epoch)

            if epoch % self.vis_every == 0:
                src = self.save_dir / "loss_curves.png"
                if src.exists():
                    shutil.copy(
                        src,
                        self.save_dir / f"loss_curves_ep{epoch:03d}.png",
                    )

            print(
                f"Ep {epoch:>3d} | "
                f"tr_wtd={tr['weighted']:.4f}  "
                f"va_wtd={va['weighted']:.4f}  "
                f"va_recon={va['recon']:.4f}  "
                f"va_pc={va['pc_loss']:.4f}  "
                f"va_kl={va['kl']:.5f}  "
                f"{'BEST' if improved else f'no-imp {self._no_improve_count}'}"
            )

            if self._no_improve_count >= self.patience:
                print(f"\n[Trainer] Early stopping at epoch {epoch}.")
                break

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()

        totals = {n: 0.0 for n in self._NAMES}
        n = 0
        t0 = time.time()

        assert self.train_smri_mean is not None

        for idx, (smri, ica_stacked) in enumerate(self.train_loader):
            B = smri.size(0)

            s = smri.to(self.device)
            c = ica_stacked.to(self.device)

            s = s - self.train_smri_mean.expand(B, -1, -1, -1, -1)

            out = _forward_chunked(self.model, s, c)
            comps = out["components"]

            total, recon, pc, orth, kl_val, _ = self.criterion(
                smri=s,
                components=comps,
                ica_stacked=c,
                mu=out["mu"],
                logvar=out["logvar"],
                mask=None,
                epoch=epoch,
            )

            if is_bad(total):
                print(f"\n  NaN/Inf at epoch {epoch} batch {idx}")
                print(
                    f"  recon={recon.item():.4f}  "
                    f"pc={pc.item():.4f}  "
                    f"orth={orth.item():.4f}  "
                    f"kl={kl_val.item():.6f}"
                )
                raise RuntimeError("NaN/Inf in loss.")

            self.optimizer.zero_grad(set_to_none=True)
            total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=5.0,
            )

            self.optimizer.step()

            totals["weighted"] += total.detach().item()
            totals["recon"] += recon.item()
            totals["pc_loss"] += pc.item()
            totals["orth"] += orth.item()
            totals["kl"] += kl_val.item()
            n += 1

            if idx % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [Ep {epoch} {idx:>4d}] lr={lr:.2e}  "
                    f"loss={total.detach().item():.4f}  "
                    f"recon={recon.item():.4f}  "
                    f"pc={pc.item():.4f}  "
                    f"orth={orth.item():.4f}  "
                    f"kl={kl_val.item():.6f}"
                )

            if idx % 1000 == 0 and idx > 0:
                torch.save(
                    self.model.state_dict(),
                    self.save_dir / "recent.pth",
                )

        print(f"  Training time: {(time.time() - t0) / 60:.2f} min")

        return {k: v / n for k, v in totals.items()}

    def _valid_epoch(self, epoch: int) -> dict:
        self.model.eval()

        totals = {n: 0.0 for n in self._NAMES}
        n = 0
        t0 = time.time()
        last_comps = None

        assert self.train_smri_mean is not None

        with torch.no_grad():
            for smri, ica_stacked in self.valid_loader:
                B = smri.size(0)

                s = smri.to(self.device)
                c = ica_stacked.to(self.device)

                s = s - self.train_smri_mean.expand(B, -1, -1, -1, -1)

                out = _forward_chunked(self.model, s, c)
                comps = out["components"]

                total, recon, pc, orth, kl_val, rho = self.criterion(
                    smri=s,
                    components=comps,
                    ica_stacked=c,
                    mu=out["mu"],
                    logvar=out["logvar"],
                    mask=None,
                    epoch=epoch,
                )

                totals["weighted"] += total.item()
                totals["recon"] += recon.item()
                totals["pc_loss"] += pc.item()
                totals["orth"] += orth.item()
                totals["kl"] += kl_val.item()
                n += 1

                last_comps = comps.detach()

                # Optional lightweight visualization:
                # if not vis_done and epoch % self.vis_every == 0:
                #     self._visualise_light(s, c, comps, rho, epoch)
                #     vis_done = True

        if last_comps is not None:
            comp_std = last_comps.std(dim=1).mean().item()

            if comp_std < 1e-5:
                print(
                    f"  WARNING: component diversity very low "
                    f"(std={comp_std:.2e}) — possible mode collapse."
                )

        print(f"  Validation time: {(time.time() - t0) / 60:.2f} min")

        return {k: v / n for k, v in totals.items()}

    def _health_check(self):
        print("[Trainer] Pre-training health check ...")

        assert self.global_mask is not None, "global_mask is None"
        assert self.train_smri_mean is not None, "train_smri_mean is None"

        assert not is_bad(self.train_smri_mean), \
            "train_smri_mean contains NaN/Inf"

        assert self.global_mask.any(), \
            "global_mask is all False — check ICA paths"

        try:
            smri, ica_stacked = next(iter(self.train_loader))
        except Exception as exc:
            raise RuntimeError(f"[HealthCheck] DataLoader failed: {exc}")

        assert smri.shape[1:] == (1, 64, 64, 64), \
            f"Unexpected smri shape: {tuple(smri.shape)}"

        assert ica_stacked.ndim == 5, \
            f"Unexpected ica_stacked shape: {tuple(ica_stacked.shape)}"

        assert ica_stacked.shape[1:] == (53, 64, 64, 64), \
            f"Unexpected ica_stacked shape: {tuple(ica_stacked.shape)}"

        self.model.eval()

        with torch.no_grad():
            s = smri[:1].to(self.device)
            c = ica_stacked[:1].to(self.device)
            s = s - self.train_smri_mean.expand(1, -1, -1, -1, -1)

            out = _forward_chunked(self.model, s, c)

        assert out["components"].shape == (1, 53, 64, 64, 64), \
            f"Unexpected output shape: {tuple(out['components'].shape)}"

        assert not is_bad(out["components"]), \
            "Model output contains NaN/Inf after init."

        self.model.train()

        mask_coverage = self.global_mask.float().mean().item()

        print(
            f"[Trainer] Health check passed  "
            f"global_mask_coverage={mask_coverage:.3f}  "
            f"mean_smri={self.train_smri_mean.mean().item():.4f}"
        )

    def _visualise_light(
        self,
        smri,
        ica_stacked,
        components,
        rho,
        epoch,
    ):
        vis_dir = self.save_dir / f"epoch_{epoch:03d}"
        vis_dir.mkdir(exist_ok=True)

        if self.global_mask is not None:
            mask_np = self.global_mask[0].detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.ones_like(
                smri[0, 0].detach().cpu().numpy(),
                dtype=bool,
            )

        self._plot_reconstruction(
            smri[0, 0].detach().cpu().numpy(),
            components[0].sum(dim=0).detach().cpu().numpy(),
            mask_np,
            vis_dir / "smri_reconstruction.png",
        )

        self._plot_rho_heatmap(
            rho.detach().cpu().numpy(),
            vis_dir / "rho_heatmap.png",
        )

        self._plot_activation_bar(
            components[0].abs().amax(dim=(1, 2, 3)).detach().cpu().numpy(),
            epoch,
            vis_dir / "component_activation.png",
        )

        plt.close("all")

    def _plot_reconstruction(self, smri_v, sum_v, mask, path):
        def _m(v):
            return np.where(mask, v, np.nan)

        res = smri_v - sum_v

        err = float(
            np.mean(res[mask] ** 2)
            / (np.mean(smri_v[mask] ** 2) + 1e-8)
        )

        vmin = float(smri_v[mask].min())
        vmax = float(smri_v[mask].max())

        fig, axes = plt.subplots(3, 1, figsize=(20, 15))

        for ax, vol, title, use_range in zip(
            axes,
            [_m(smri_v), _m(sum_v), _m(res)],
            [
                "Original sMRI",
                "Sum of 53 components",
                f"Residual  (RECON={err:.4f})",
            ],
            [True, True, False],
        ):
            plotting.plot_epi(
                Nifti1Image(vol.astype(np.float32), self.affine),
                axes=ax,
                title=title,
                display_mode="z",
                cut_coords=8,
                vmin=vmin if use_range else None,
                vmax=vmax if use_range else None,
                colorbar=True,
            )

        plt.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_rho_heatmap(self, rho_np, path):
        B, K = rho_np.shape

        fig, ax = plt.subplots(
            figsize=(max(10, K // 4), max(3, B)),
        )

        im = ax.imshow(
            rho_np,
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
            aspect="auto",
        )

        plt.colorbar(im, ax=ax, label="Pearson rho")

        ax.set_xlabel("ICA component k")
        ax.set_ylabel("Subject in batch")
        ax.set_title(f"rho_ik mean={rho_np.mean():.3f}")

        plt.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

    def _plot_activation_bar(self, comp_max, epoch, path):
        K = len(comp_max)

        fig, ax = plt.subplots(figsize=(16, 3))

        ax.bar(
            range(K),
            comp_max[np.argsort(comp_max)[::-1]],
            color=_mcm.viridis(np.linspace(0, 1, K)),
            alpha=0.85,
        )

        ax.set_xlabel("Component, sorted by activation")
        ax.set_ylabel("Max |activation|")
        ax.set_title(f"Epoch {epoch}: uniform=collapse | near-zero=dead")

        plt.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)

    def _plot_loss_curves(self, epoch: int):
        ep = np.arange(1, epoch + 1)

        fig, axes = plt.subplots(
            1,
            len(self._NAMES),
            figsize=(4 * len(self._NAMES), 4),
        )

        for ax, name in zip(axes, self._NAMES):
            ax.plot(
                ep,
                self.train_df.loc[ep, f"tr_{name}"],
                color="steelblue",
                label="Train",
            )

            ax.plot(
                ep,
                self.valid_df.loc[ep, f"va_{name}"],
                color="tomato",
                linestyle="--",
                label="Valid",
            )

            ax.set_title(name.upper())
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.suptitle(f"Training progress — epoch {epoch}", fontsize=11)
        plt.tight_layout()

        fig.savefig(self.save_dir / "loss_curves.png", dpi=100)
        plt.close(fig)

    def _save_checkpoint(self, epoch: int, best: bool = False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_va_weighted,
            "config": dict(self.config) if isinstance(self.config, dict) else {},
        }

        torch.save(state, self.save_dir / "last.pth")

        if best:
            torch.save(state, self.save_dir / "model_best.pth")
            print(
                f"  Best model saved "
                f"(epoch {epoch}, "
                f"va_weighted={self.best_va_weighted:.5f})"
            )