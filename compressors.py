"""
TurboQuant KV cache v2: Asymmetric attention.

Instead of decompressing KV vectors and feeding them to standard attention,
we compute attention scores DIRECTLY from compressed representations using
the TurboQuant asymmetric inner product estimator.

Key insight from the paper:
  <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, sign(S@r_k)>

This is unbiased with variance O(1/d), even though k_mse itself has high
per-vector error. The estimator works because QJL corrects the bias in the
inner product space, not in the vector space.

For values, we use MSE-only decompression since the weighted sum in
softmax(scores) @ V averages out per-vector errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .lloyd_max import LloydMaxCodebook
from .turboquant import generate_qjl_matrix, generate_rotation_matrix, resolve_torch_dtype


class TurboQuantCompressorV2:
    """
    Compressor that stores compressed representations AND supports
    direct inner product computation without full decompression.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu", dtype: torch.dtype | str = torch.float32):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device
        self.dtype = resolve_torch_dtype(dtype)

        # Rotation matrix
        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device, dtype=self.dtype)

        # Lloyd-Max codebook
        self.codebook = LloydMaxCodebook(head_dim, self.mse_bits)
        self.centroids = self.codebook.centroids.to(device)
        self.boundaries = self.codebook.boundaries.to(device)

        # QJL matrix
        self.S = generate_qjl_matrix(head_dim, m=head_dim, seed=seed + 10000, device=device, dtype=self.dtype)

        # Precompute Pi^T for fast dequant
        self.PiT = self.Pi.T.contiguous()

    def _solve_codebook(self, d: int, bits: int) -> torch.Tensor:
        from scipy import integrate
        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)

        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))

        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_centroids = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_centroids.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_centroids

        return torch.tensor(centroids, dtype=torch.float32)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        Compress states: (batch, heads, seq, head_dim) -> compressed dict.
        Stores everything needed for asymmetric inner product computation.
        """
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).to(device=self.Pi.device, dtype=self.Pi.dtype)

        # Store original norms
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)  # (N, 1)
        flat_norm = flat / (vec_norms + 1e-8)

        # Rotate and quantize
        rotated = flat_norm @ self.PiT
        indices = self.codebook.quantize(rotated).to(torch.uint8)

        # MSE reconstruction in original space (for inner product term 1)
        reconstructed_rotated = self.centroids.to(device=indices.device, dtype=self.Pi.dtype)[indices.long()]
        k_mse = (reconstructed_rotated @ self.Pi) * vec_norms  # (N, D) - back in original scale

        # Residual in original space
        residual = flat - k_mse
        residual_norm = torch.norm(residual, dim=-1)  # (N,)

        # QJL signs of residual
        projected = residual @ self.S.T
        signs = torch.where(projected >= 0, torch.ones_like(projected), -torch.ones_like(projected))

        return {
            "k_mse": k_mse.to(self.dtype).reshape(B, H, S, D),
            "qjl_signs": signs.to(self.dtype).reshape(B, H, S, D),
            "residual_norm": residual_norm.to(self.dtype).reshape(B, H, S),
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def asymmetric_attention_scores(self, queries: torch.Tensor, compressed: dict) -> torch.Tensor:
        """
        Compute attention scores <Q, K> directly from compressed K.

        Uses the asymmetric estimator:
            <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(pi/2)/m * <S@q, signs_k>

        Args:
            queries: (batch, heads, seq_q, head_dim)
            compressed: dict from compress()

        Returns:
            scores: (batch, heads, seq_q, seq_k)
        """
        target_dtype = compressed["k_mse"].dtype
        queries_input = queries.to(device=self.S.device, dtype=target_dtype)
        k_mse = compressed["k_mse"].to(device=queries_input.device, dtype=target_dtype)
        signs = compressed["qjl_signs"].to(device=queries_input.device, dtype=target_dtype)
        r_norm = compressed["residual_norm"].to(device=queries_input.device, dtype=target_dtype)

        q_projected = torch.matmul(queries_input, self.S.to(device=queries_input.device, dtype=target_dtype).T)
        correction_scale = math.sqrt(math.pi / 2) / self.S.shape[0]
        weighted_signs = signs * (correction_scale * r_norm).unsqueeze(-1)
        query_search = torch.cat([queries_input, q_projected], dim=-1)
        key_search = torch.cat([k_mse, weighted_signs], dim=-1)
        return torch.matmul(query_search, key_search.transpose(-2, -1))


class TurboQuantCompressorMSE:
    """Simpler MSE-only compressor for values (no QJL needed)."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu", dtype: torch.dtype | str = torch.float32):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.dtype = resolve_torch_dtype(dtype)

        self.Pi = generate_rotation_matrix(head_dim, seed=seed, device=device, dtype=self.dtype)
        self.codebook = LloydMaxCodebook(head_dim, bits)
        self.centroids = self.codebook.centroids.to(device)

    def _solve_codebook(self, d, bits):
        from scipy import integrate
        n_levels = 2 ** bits
        sigma = 1.0 / math.sqrt(d)
        def pdf(x):
            return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))
        lo, hi = -3.5 * sigma, 3.5 * sigma
        centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]
        for _ in range(200):
            boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
            edges = [lo * 3] + boundaries + [hi * 3]
            new_c = []
            for i in range(n_levels):
                a, b = edges[i], edges[i + 1]
                num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
                den, _ = integrate.quad(pdf, a, b)
                new_c.append(num / den if den > 1e-15 else centroids[i])
            if max(abs(new_c[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
                break
            centroids = new_c
        return torch.tensor(centroids, dtype=torch.float32)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        B, H, S, D = states.shape
        flat = states.reshape(-1, D).to(device=self.Pi.device, dtype=self.Pi.dtype)
        vec_norms = torch.norm(flat, dim=-1, keepdim=True)
        flat_norm = flat / (vec_norms + 1e-8)
        rotated = flat_norm @ self.Pi.T
        indices = self.codebook.quantize(rotated).to(torch.uint8)
        return {
            "indices": indices,
            "vec_norms": vec_norms.squeeze(-1).to(self.dtype),
            "shape": (B, H, S, D),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        B, H, S, D = compressed["shape"]
        indices = compressed["indices"].long()
        reconstructed = self.centroids.to(device=indices.device, dtype=self.Pi.dtype)[indices] @ self.Pi
        vec_norms = compressed["vec_norms"].to(device=indices.device, dtype=self.Pi.dtype).unsqueeze(-1)
        return (reconstructed * vec_norms).reshape(B, H, S, D)


