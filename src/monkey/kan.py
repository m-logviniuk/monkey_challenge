"""Kolmogorov-Arnold layer for the density-regressor decoder head.

``KANLinear`` replaces an ``nn.Linear`` with a learnable B-spline activation
per (input, output) channel pair, following the EfficientKAN construction
(Blealtan, 2024) adapted here to a 2D convolution-head context: it is applied
per pixel to the channel vector. Each edge is a univariate spline that can be
read directly, which is what makes the head interpretable. A SiLU residual
base path is kept for training stability.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """Cross-channel KAN layer with one B-spline per (in, out) edge.

    Parameter count is ``out * in * (grid_size + spline_order)`` for the
    splines plus ``out * in`` for the base path and ``out * in`` for the
    per-edge spline scaler. The grid is a fixed buffer, so it is not counted
    as a parameter.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 4,
                 spline_order: int = 3, grid_range: tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h
             + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.scale_noise = 0.1
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features,
                            self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                self._curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order], noise,
                )
            )
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline bases for ``x`` of shape ``[N, in_features]``.

        Returns ``[N, in_features, grid_size + spline_order]``.
        """
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :-(k + 1)]
            left_den = grid[:, k:-1] - grid[:, :-(k + 1)]
            right_num = grid[:, k + 1:] - x
            right_den = grid[:, k + 1:] - grid[:, 1:(-k)]
            bases = (
                left_num / left_den.clamp(min=1e-8) * bases[:, :, :-1]
                + right_num / right_den.clamp(min=1e-8) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Least-squares fit of spline coefficients to target values ``y``."""
        A = self._b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute base + spline output for ``x`` of shape ``[N, in_features]``."""
        base_out = F.linear(F.silu(x), self.base_weight)
        bases = self._b_splines(x)
        flat_basis = bases.reshape(x.size(0), -1)
        flat_weight = (
            self.spline_weight * self.spline_scaler.unsqueeze(-1)
        ).reshape(self.out_features, -1)
        spline_out = F.linear(flat_basis, flat_weight)
        return base_out + spline_out

    @torch.no_grad()
    def edge_functions(self, n_points: int = 200,
                       grid_range: tuple[float, float] = (-1.0, 1.0)):
        """Sample every per-edge spline for interpretability plots.

        Returns ``(xs, curves)`` where ``xs`` has shape ``[n_points]`` and
        ``curves`` has shape ``[out_features, in_features, n_points]``: the
        spline response of each (input, output) edge over the grid range,
        excluding the base SiLU path.
        """
        xs = torch.linspace(grid_range[0], grid_range[1], n_points,
                            device=self.spline_weight.device)
        probe = xs.unsqueeze(1).expand(n_points, self.in_features)
        bases = self._b_splines(probe)
        weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        curves = torch.einsum("nik,oik->oin", bases, weight)
        return xs, curves
