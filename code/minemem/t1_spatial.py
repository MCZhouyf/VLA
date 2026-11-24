"""Differentiable heading-indexed rolling spatial memory for T1."""

import math

import numpy as np
import torch
from torch import nn


SIDE = 32
CELL_BLOCKS = 2
HEADINGS = 8
QUERIES = 32


def retained_cells(poses, side=SIDE):
    """Return observations whose cells survive every subsequent grid shift."""
    cells = np.floor(np.asarray(poses)[:, :2] / CELL_BLOCKS).astype(np.int64)
    # The composition of two clamps is another clamp. Compose successive
    # prefixes so NumPy can evaluate whole ranges of grid shifts at once.
    left = cells - side + 1
    right = cells.copy()
    stride = 1
    while stride < len(cells):
        lo, hi = left[stride:], right[stride:]
        next_left = np.clip(left[:-stride], lo, hi)
        next_right = np.clip(right[:-stride], lo, hi)
        left[stride:], right[stride:] = next_left, next_right
        stride *= 2
    origins = np.clip(cells[0] - side // 2, left, right)
    lower = np.maximum.accumulate(origins[::-1], axis=0)[::-1]
    upper = np.minimum.accumulate(origins[::-1] + side - 1, axis=0)[::-1]
    keep = np.all((cells >= lower) & (cells <= upper), axis=1)[:-1]
    return cells[:-1], origins[-1], keep


def write_weights(gate, inverse, slot_count):
    """Apply the same-cell writes in temporal order without a dense pair matrix."""
    order = torch.argsort(inverse, stable=True)
    grouped = inverse[order]
    log_gate = torch.log1p(-gate[order]).double()
    inclusive = log_gate.cumsum(0)
    end = torch.zeros(slot_count, device=gate.device, dtype=torch.long).scatter_reduce(
        0, grouped, torch.arange(len(order), device=gate.device), reduce="amax")
    suffix = (inclusive[end[grouped]] - inclusive).to(gate.dtype)
    weights = gate[order] * torch.exp(suffix)
    return torch.empty_like(weights).scatter(0, order, weights)


class SpatialMemory(nn.Module):
    def __init__(self, vision_dim, llm_dim, width=512, grid_side=SIDE, position_value_scale=0.0):
        super().__init__()
        if grid_side < 2:
            raise ValueError("A spatial grid requires at least two cells per side")
        self.grid_side = grid_side
        self.position_value_scale = position_value_scale
        self.write = nn.Sequential(nn.Linear(vision_dim, width), nn.LayerNorm(width))
        self.gate = nn.Linear(width, 1)
        self.position = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.randn(1, QUERIES, width) / math.sqrt(width))
        self.attention = nn.MultiheadAttention(width, 8, batch_first=True)
        self.read = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, llm_dim))

    def forward(self, visual, poses):
        if len(poses) != len(visual) + 1:
            raise ValueError("Memory input must contain all prior visual features and the current pose")
        device, dtype = self.queries.device, self.queries.dtype
        if not len(visual):
            return self.read(self.queries)
        poses = np.asarray(poses, dtype=np.float32)
        side = self.grid_side
        cells, origin, keep = retained_cells(poses, side)
        if not keep.any():
            return self.read(self.queries)

        cells = cells[keep] - origin
        headings = np.floor((poses[:-1, 2][keep] % 360) * HEADINGS / 360).astype(np.int64)
        keys = (cells[:, 0] * side + cells[:, 1]) * HEADINGS + headings
        keys = torch.as_tensor(keys, dtype=torch.long, device=device)
        slots, inverse = torch.unique(keys, return_inverse=True)
        keep_index = torch.as_tensor(np.flatnonzero(keep), dtype=torch.long, device=device)
        values = self.write(visual.to(device=device, dtype=dtype)[keep_index])
        gate = torch.sigmoid(self.gate(values).float()).flatten().clamp(0.01, 0.99)

        weight = write_weights(gate, inverse, len(slots))
        mass = torch.zeros(len(slots), device=device).index_add(0, inverse, weight)
        cells_value = torch.zeros((len(slots), values.shape[-1]), device=device).index_add(
            0, inverse, values.float() * weight[:, None]
        )
        cells_value = (cells_value / mass.clamp_min(1e-6)[:, None]).to(dtype)

        last = torch.zeros(len(slots), device=device, dtype=torch.long).scatter_reduce(
            0, inverse, keep_index, reduce="amax"
        )
        x = ((slots // HEADINGS) // side + int(origin[0]) + 0.5) * CELL_BLOCKS - poses[-1, 0]
        z = ((slots // HEADINGS) % side + int(origin[1]) + 0.5) * CELL_BLOCKS - poses[-1, 1]
        yaw = math.radians(float(poses[-1, 2]))
        relative_yaw = (slots % HEADINGS + 0.5) * (2 * math.pi / HEADINGS) - yaw
        # Keep the trained coordinate scale when testing a larger storage window.
        position = torch.stack((
            (-x * math.cos(yaw) - z * math.sin(yaw)) / (SIDE * CELL_BLOCKS),
            (-x * math.sin(yaw) + z * math.cos(yaw)) / (SIDE * CELL_BLOCKS),
            relative_yaw.sin(), relative_yaw.cos(),
            torch.log1p((len(visual) - last).float()) / 8,
        ), dim=-1).to(dtype)
        position_embedding = self.position(position)
        keys = cells_value + position_embedding
        values = cells_value
        if self.position_value_scale:
            values = values + self.position_value_scale * position_embedding
        response, _ = self.attention(self.queries, keys[None], values[None], need_weights=False)
        return self.read(self.queries + response)
