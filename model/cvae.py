"""
model/cvae.py
─────────────────────────────────────────────────────────────────
Conditional 3D Variational Autoencoder (C3D-VAE)

Data input:
  smri        : (B, 1,  64, 64, 64)
  ica_stacked : (B, 53, 64, 64, 64)

sMRI encoder  (SimpleVolumeEncoder):
  ConvBlock_16 : 1  × 64³ → 16 × 32³   (stride-2)
  ConvBlock_32 : 16 × 32³ → 32 × 16³   (stride-2)
  ConvBlock_64 : 32 × 16³ → 64 × 8³    (stride-2)
  GAP          : 64 × 8³  → 64
  Linear       : 64       → embedding_dim
  → mu, logvar → z_i ∈ R^{latent_dim}

ICA encoder  (StackedICAEncoder — grouped convolutions):
  All K=53 maps encoded in ONE GPU call using groups=K.
  GroupedConvBlock_16K : K×64³ → 16K×32³   (groups=K)
  GroupedConvBlock_32K : 16K×32³→ 32K×16³  (groups=K)
  GroupedConvBlock_64K : 32K×16³→ 64K×8³   (groups=K)
  GAP per group        : 64K×8³ → 64K
  Linear (shared)      : 64    → cond_dim  (applied to each of K groups)
  → e_ik ∈ R^{K × cond_dim}

  Chunked fallback: if B*K > ICA_CHUNK_SIZE, the grouped encoder is
  called in mini-batches of ICA_CHUNK_SIZE // K subjects at a time,
  bounding peak memory regardless of batch size.

Decoder  (StackComponentDecoder, shared across all k):
  [z_i || e_ik] → Linear+LeakyReLU → 64×8³ → 32×16³ → 16×32³ → 1×64³ (identity)

Output:
  components : (B, 53, 64, 64, 64)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_weights(module: nn.Module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="leaky_relu",
            a=0.2,
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.BatchNorm3d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────────
# Basic blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock3d(nn.Module):
    """
    Conv3d → BatchNorm3d → LeakyReLU → Dropout3d

    With stride=2, spatial size halves:
      64³ → 32³ → 16³ → 8³
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3d(nn.Module):
    """
    Trilinear upsample ×2 → ConvBlock3d

    Used by the symmetric decoder:
      8³ → 16³ → 32³ → 64³
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="trilinear",
                align_corners=False,
            ),
            ConvBlock3d(
                in_ch,
                out_ch,
                stride=1,
                dropout=dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# Shared-style encoder
# ─────────────────────────────────────────────────────────────────────────────

class SimpleVolumeEncoder(nn.Module):
    """
    Encoder used for both:
      1. sMRI volume      : (B, 1, 64, 64, 64)
      2. each ICA volume  : (B*K, 1, 64, 64, 64)

    Mathematical structure:

      g1 = ConvBlock_16(x)   ∈ R^{16 × 32 × 32 × 32}
      g2 = ConvBlock_32(g1)  ∈ R^{32 × 16 × 16 × 16}
      g3 = ConvBlock_64(g2)  ∈ R^{64 × 8 × 8 × 8}
      g4 = GAP(g3)           ∈ R^{64}
      e  = W_e g4 + b_e      ∈ R^{embedding_dim}
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.enc = nn.Sequential(
            ConvBlock3d(1, 16, stride=2, dropout=dropout),
            ConvBlock3d(16, 32, stride=2, dropout=dropout),
            ConvBlock3d(32, 64, stride=2, dropout=dropout),
        )

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, 1, 64, 64, 64)

        returns:
          embedding : (N, embedding_dim)
        """
        h = self.enc(x)              # (N, 64, 8, 8, 8)
        h = self.gap(h).flatten(1)   # (N, 64)
        h = self.proj(h)             # (N, embedding_dim)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Grouped ICA stack encoder
# ─────────────────────────────────────────────────────────────────────────────

class StackedICAEncoder(nn.Module):
    """
    Encodes all K ICA maps in a single grouped-convolution forward pass.

    Key idea: treat the K=53 ICA maps as K independent "groups" in a
    grouped Conv3d.  This replaces 53 sequential single-channel encoder
    calls with one kernel launch, giving a ~K× reduction in GPU overhead.

    Input  : (B, K, D, H, W)  — the full stacked ICA tensor
    Output : (B, K, cond_dim) — one embedding vector per component

    Grouped convolution mechanics
    ─────────────────────────────
    A standard Conv3d(in, out, ...) treats the input as one group.
    With groups=K:
      in_channels  = K * 1   (each group owns 1 input channel)
      out_channels = K * C   (each group produces C output channels)
    The K groups never communicate — exactly equivalent to running K
    separate Conv3d(1, C, ...) calls but in a single kernel.

    The GAP + Linear projection are applied per-group by reshaping:
      (B, K*64, 8, 8, 8) → GAP → (B, K*64) → view (B*K, 64) → Linear → (B*K, cond_dim)

    Memory note
    ───────────
    Peak activation during the grouped conv is (B, K*64, 8³) which for
    B=4, K=53 is 4×3392×512 float32 ≈ 27 MB — well within budget.
    If a very large batch is used, the chunked wrapper in trainer.py
    splits the B dimension before calling this encoder.
    """

    def __init__(
        self,
        num_components: int = 53,
        cond_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        K = num_components
        self.K = K
        self.cond_dim = cond_dim

        # Grouped convolutions: groups=K, each group maps 1→C channels.
        # in_channels = K*1, out_channels = K*C so that channel dim stays
        # interpretable as (K groups) × (C feature maps per group).
        self.enc = nn.Sequential(
            # Layer 1: K×1 → K×16   64³ → 32³
            nn.Conv3d(K * 1,  K * 16, kernel_size=3, stride=2,
                      padding=1, groups=K, bias=False),
            nn.BatchNorm3d(K * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),

            # Layer 2: K×16 → K×32  32³ → 16³
            nn.Conv3d(K * 16, K * 32, kernel_size=3, stride=2,
                      padding=1, groups=K, bias=False),
            nn.BatchNorm3d(K * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),

            # Layer 3: K×32 → K×64  16³ → 8³
            nn.Conv3d(K * 32, K * 64, kernel_size=3, stride=2,
                      padding=1, groups=K, bias=False),
            nn.BatchNorm3d(K * 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),
        )

        # Global average pool collapses spatial dims: (B, K*64, 8³) → (B, K*64, 1, 1, 1)
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Shared linear projection applied identically to each group's 64-d vector.
        # Implemented as a single Linear(64, cond_dim) applied after reshaping to
        # (B*K, 64) — equivalent to K separate Linear layers with tied weights.
        self.proj = nn.Linear(64, cond_dim)

    def forward(self, ica_stacked: torch.Tensor) -> torch.Tensor:
        """
        ica_stacked : (B, K, D, H, W)

        returns:
          embeddings : (B, K, cond_dim)
        """
        B, K, D, H, W = ica_stacked.shape

        assert K == self.K, \
            f"StackedICAEncoder expects K={self.K} components, got {K}"

        # Input to grouped conv must have shape (B, K*1, D, H, W).
        # ica_stacked is already (B, K, D, H, W) — same thing.
        h = self.enc(ica_stacked)           # (B, K*64, 8, 8, 8)

        h = self.gap(h)                     # (B, K*64, 1, 1, 1)
        h = h.flatten(1)                    # (B, K*64)

        # Reshape so each group's 64 features are a separate row.
        h = h.view(B * K, 64)              # (B*K, 64)
        h = self.proj(h)                    # (B*K, cond_dim)

        return h.view(B, K, self.cond_dim)  # (B, K, cond_dim)


# ─────────────────────────────────────────────────────────────────────────────
# sMRI VAE encoder
# ─────────────────────────────────────────────────────────────────────────────

class SMRIVAEEncoder(nn.Module):
    """
    sMRI encoder with the same convolutional structure as ICA encoder.

    smri → SimpleVolumeEncoder → h_i
    h_i  → mu, logvar
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        latent_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = SimpleVolumeEncoder(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.fc_mu = nn.Linear(embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(embedding_dim, latent_dim)

    def forward(self, smri: torch.Tensor):
        """
        smri : (B, 1, 64, 64, 64)

        returns:
          mu     : (B, latent_dim)
          logvar : (B, latent_dim)
        """
        h = self.encoder(smri)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ─────────────────────────────────────────────────────────────────────────────
# Symmetric stack decoder
# ─────────────────────────────────────────────────────────────────────────────

class StackComponentDecoder(nn.Module):
    """
    Symmetric decoder for stacked component generation.

    Input:
      z_i    : global sMRI latent
      e_ik   : ICA component embedding

    Decode:
      [z_i || e_ik]
          → Linear + LeakyReLU  (projection to seed volume)
          → reshape: 64 × 8 × 8 × 8       (f^(1))
          → UpBlock_32: 32 × 16 × 16 × 16 (f^(2))
          → UpBlock_16: 16 × 32 × 32 × 32 (f^(3))
          → Upsample + Conv3d: 1 × 64 × 64 × 64 (f^(4), identity activation)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seed_ch = 64
        self.seed_size = 8
        seed_dim = self.seed_ch * self.seed_size ** 3

        self.proj = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, seed_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up1 = UpBlock3d(64, 32, dropout=dropout)
        self.up2 = UpBlock3d(32, 16, dropout=dropout)

        self.up3 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="trilinear",
                align_corners=False,
            ),
            nn.Conv3d(
                16,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """
        z : (B*K, latent_dim)
        e : (B*K, cond_dim)

        returns:
          component : (B*K, 1, 64, 64, 64)
        """
        h = torch.cat([z, e], dim=1)
        h = self.proj(h)

        h = h.view(
            h.size(0),
            self.seed_ch,
            self.seed_size,
            self.seed_size,
            self.seed_size,
        )                               # (B*K, 64, 8, 8, 8)

        h = self.up1(h)                 # (B*K, 32, 16, 16, 16)
        h = self.up2(h)                 # (B*K, 16, 32, 32, 32)
        h = self.up3(h)                 # (B*K, 1, 64, 64, 64)

        return h


# ─────────────────────────────────────────────────────────────────────────────
# Full C3D-VAE
# ─────────────────────────────────────────────────────────────────────────────

class C3DVAE(nn.Module):
    """
    Conditional 3D VAE with stacked ICA input.

    Forward:
      smri        : (B, 1,  64, 64, 64)
      ica_stacked : (B, 53, 64, 64, 64)

    Steps:
      1. Encode sMRI → mu, logvar → z_i  (B, latent_dim)
      2. Encode ALL 53 ICA maps in one grouped-conv call → e_ik  (B, K, cond_dim)
      3. Expand z_i to (B*K, latent_dim), flatten e_ik to (B*K, cond_dim)
      4. Decode all components in chunked batches → (B, K, D, H, W)
    """

    def __init__(
        self,
        num_components: int = 53,
        latent_dim: int = 128,
        cond_dim: int = 64,
        smri_embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.K = num_components
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.smri_encoder = SMRIVAEEncoder(
            embedding_dim=smri_embedding_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # StackedICAEncoder encodes all K maps in one grouped-conv call
        # instead of K sequential single-map calls. Same learned function,
        # much lower GPU kernel overhead.
        self.ica_encoder = StackedICAEncoder(
            num_components=num_components,
            cond_dim=cond_dim,
            dropout=dropout,
        )

        self.decoder = StackComponentDecoder(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            dropout=dropout,
        )

        self.apply(_init_weights)

        # Start decoder close to zero for stable early training.
        nn.init.zeros_(self.decoder.up3[1].weight)
        nn.init.zeros_(self.decoder.up3[1].bias)

    def reparameterise(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """
        z = mu + sigma * eps   (stochastic=True, default)
        z = mu                 (stochastic=False, for deterministic eval)

        Training always uses the stochastic path to keep the KL term
        meaningful and to avoid posterior collapse.  At test time the
        default is also stochastic; pass stochastic=False explicitly
        when you want a single deterministic reconstruction.
        """
        if stochastic:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps

        return mu

    def forward(
        self,
        smri: torch.Tensor,
        ica_stacked: torch.Tensor,
    ) -> dict:
        """
        smri:
          (B, 1, 64, 64, 64)

        ica_stacked:
          (B, K, 64, 64, 64)

        returns:
          components : (B, K, 64, 64, 64)
          mu         : (B, latent_dim)
          logvar     : (B, latent_dim)
          ica_embed  : (B, K, cond_dim)
        """
        assert smri.ndim == 5, \
            f"Expected smri shape (B,1,D,H,W), got {tuple(smri.shape)}"

        assert ica_stacked.ndim == 5, \
            f"Expected ica_stacked shape (B,K,D,H,W), got {tuple(ica_stacked.shape)}"

        B, K, D, H, W = ica_stacked.shape

        assert K == self.K, \
            f"Expected {self.K} ICA components, got {K}"

        assert smri.shape[0] == B, \
            f"Batch mismatch: smri batch={smri.shape[0]}, ica batch={B}"

        assert smri.shape[1] == 1, \
            f"Expected smri channel dimension 1, got {smri.shape[1]}"

        # 1. Encode sMRI and sample latent z.
        mu, logvar = self.smri_encoder(smri)
        z = self.reparameterise(mu, logvar, stochastic=self.training)  # (B, latent_dim)

        # 2. Encode all K ICA maps in one grouped-conv call.
        #    Output is already (B, K, cond_dim) — no reshape needed.
        ica_embed = self.ica_encoder(ica_stacked)           # (B, K, cond_dim)
        e_flat = ica_embed.reshape(B * K, self.cond_dim)   # (B*K, cond_dim)

        # 3. Expand z so each ICA component gets the same subject-level z.
        z_flat = (
            z.unsqueeze(1)
             .expand(B, K, self.latent_dim)
             .reshape(B * K, self.latent_dim)
        )                                                   # (B*K, latent_dim)

        # 4. Decode all components together.
        comp_flat = self.decoder(z_flat, e_flat)            # (B*K, 1, D, H, W)

        components = comp_flat.reshape(B, K, D, H, W)      # (B, K, D, H, W)

        return {
            "components": components,
            "mu": mu,
            "logvar": logvar,
            "ica_embed": ica_embed,
        }

    @torch.no_grad()
    def generate(
        self,
        smri: torch.Tensor,
        ica_stacked: torch.Tensor,
        stochastic: bool = False,
    ) -> torch.Tensor:
        """
        Inference convenience wrapper.

        stochastic=False (default): use z = mu for a single deterministic
          reconstruction — useful for evaluation and visualisation.
        stochastic=True: sample z ~ q(z|X) for probabilistic generation
          of novel structural components.

        returns:
          components : (B, K, 64, 64, 64)
        """
        mu, logvar = self.smri_encoder(smri)
        z = self.reparameterise(mu, logvar, stochastic=stochastic)

        B, K, D, H, W = ica_stacked.shape
        ica_embed = self.ica_encoder(ica_stacked)           # (B, K, cond_dim)
        e_flat = ica_embed.reshape(B * K, self.cond_dim)

        z_flat = (
            z.unsqueeze(1)
             .expand(B, K, self.latent_dim)
             .reshape(B * K, self.latent_dim)
        )

        comp_flat = self.decoder(z_flat, e_flat)
        return comp_flat.reshape(B, K, D, H, W)

    def __str__(self) -> str:
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + f"\nTrainable parameters: {params:,}"