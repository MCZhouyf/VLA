"""Differentiable heading-indexed rolling spatial memory for T1."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SIDE = 32
CELL_BLOCKS = 2
HEADINGS = 8
QUERIES = 32


def pool_vision_grid(tokens, shape, merge, side):
    time, height, width = (int(value) for value in shape)
    if time != 1 or height % merge or width % merge:
        raise ValueError(f"Unexpected single-frame patch shape: {shape}")
    spatial = tokens.reshape(height//merge, width//merge, -1).float().permute(2, 0, 1)
    return F.adaptive_avg_pool2d(spatial, (side, side)).flatten(1).transpose(0, 1).contiguous()


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


def project_onto_segments(point, start, end):
    segments = end - start
    square = (segments**2).sum(-1)
    fraction = np.clip(((point-start)*segments).sum(-1) / np.maximum(square, 1e-8), 0, 1)
    return ((point - start - fraction[:, None]*segments)**2).sum(-1)


def path_corners(points, tolerance=0.65):
    """Find turns of the observed path, using only past horizontal positions."""
    def simplify(first, last):
        if last <= first + 1:
            return [first, last]
        delta = points[last] - points[first]
        inner = points[first+1:last] - points[first]
        fraction = np.clip(inner @ delta / max(float(delta @ delta), 1e-8), 0, 1)
        distance = ((inner - fraction[:, None]*delta)**2).sum(-1)
        turn = int(distance.argmax())
        if distance[turn] <= tolerance**2:
            return [first, last]
        split = first + turn + 1
        return simplify(first, split)[:-1] + simplify(split, last)

    return np.asarray(simplify(0, len(points)-1), dtype=np.int64)


def spatial_read_plan(history, current, grid_side=64, room_anchors=None):
    """Locate local cells for goal, return-leg, endpoint and route-history queries."""
    history = np.asarray(history, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if len(history) < 2 or history.shape[1:] != (3,) or current.shape != (3,):
        raise ValueError("The spatial reader requires past x/z/yaw and the current pose")
    cells, origin, retained = retained_cells(np.vstack((history, current)), grid_side)
    indices = np.flatnonzero(retained)
    if not len(indices):
        raise ValueError("The rolling grid retained no observations")
    relative_cells = cells[indices] - origin
    headings = np.floor((history[indices, 2] % 360) * HEADINGS / 360).astype(np.int64)
    keys = (relative_cells[:, 0]*grid_side + relative_cells[:, 1])*HEADINGS + headings
    _, first = np.unique(keys, return_index=True)
    _, reversed_last = np.unique(keys[::-1], return_index=True)
    retained_indices = indices[np.unique(np.concatenate((first, len(keys)-1-reversed_last)))]

    points = history[indices, :2]
    corners = path_corners(points)
    segments = points[corners]
    next_leg = int(project_onto_segments(current[:2], segments[:-1], segments[1:]).argmin())
    if next_leg and np.linalg.norm(current[:2]-segments[next_leg]) < 1.0:
        next_leg -= 1
    travelled = np.r_[0.0, np.linalg.norm(np.diff(points, axis=0), axis=-1).cumsum()]
    route_queries = indices[np.minimum(np.searchsorted(travelled, np.linspace(0, travelled[-1], QUERIES-3)),
                                      len(indices)-1)]
    anchors = np.r_[indices[0], indices[corners[next_leg]], indices[-1], route_queries]
    if room_anchors is not None:
        room_anchors = np.asarray(room_anchors, dtype=np.int64)
        if room_anchors.shape != (2,) or not np.isin(room_anchors, indices).all():
            raise ValueError("Both candidate room observations must be retained in the grid")
        anchors[:2] = room_anchors
        retained_indices = np.union1d(retained_indices, room_anchors)
    # The nearby stored observations supply the readout. Queries have no direct output residual.
    offsets = np.abs(anchors[:, None] - retained_indices[None, :])
    order = np.argsort(offsets, axis=1, kind="stable")[:, :2]
    if order.shape[1] == 1:
        order = np.repeat(order, 2, axis=1)
    selection = retained_indices[order]
    bias = -0.5 * np.take_along_axis(offsets, order, axis=1).astype(np.float32)

    selected = history[selection]
    x, z = (selected[..., :2] - current[:2]).transpose(2, 0, 1)
    yaw = math.radians(float(current[2]))
    right = -x * math.cos(yaw) - z * math.sin(yaw)
    forward = -x * math.sin(yaw) + z * math.cos(yaw)
    heading = np.radians(selected[..., 2] - current[2])
    distance = np.hypot(right, forward)
    unit = np.maximum(distance, 0.1)
    age = len(history) - selection
    geometry = np.stack((right/64, forward/64, right/unit, forward/unit, np.sin(heading),
                         np.cos(heading), np.log1p(age)/8, selection/len(history), np.log1p(distance)/5), axis=-1)
    return selection, geometry.astype(np.float32), bias


class GeometricSpatialMemory(nn.Module):
    """Store patch-level visuals and egocentric geometry together in each read value."""

    def __init__(self, visual_dim, llm_dim, embedding_norm, patch_side=2, patch_width=16, grid_side=64):
        super().__init__()
        if grid_side not in (32, 64, 128) or patch_side not in (2, 4) or embedding_norm <= 0:
            raise ValueError("Invalid spatial grid, image grid or LLM embedding norm")
        self.grid_side = grid_side
        self.patch_side = patch_side
        self.width = patch_side**2*patch_width + 32
        self.patch = nn.Linear(visual_dim, patch_width, bias=False)
        self.patch_norm = nn.LayerNorm(patch_width, elementwise_affine=False)
        self.position = nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, 23))
        self.joint = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, self.width))
        self.score = nn.Linear(self.width, 1, bias=False)
        self.norm = nn.LayerNorm(self.width)
        self.to_llm = nn.Linear(self.width, llm_dim, bias=False)
        self.register_buffer("embedding_norm", torch.tensor(float(embedding_norm)))

    def encode_visual(self, visual):
        # All patch vectors are kept in raster order. The visual input has train-only standardization.
        return self.patch_norm(self.patch(visual)).flatten(-2)

    def read_projected(self, appearance, geometry, bias):
        position = torch.cat((geometry, self.position(geometry)), dim=-1)
        value = torch.cat((appearance, position), dim=-1)
        value = value + 0.25*self.joint(value)
        weights = (self.score(value).squeeze(-1)+bias).softmax(-1)
        return self.norm((value * weights.unsqueeze(-1)).sum(-2))

    def read(self, visual, geometry, bias):
        return self.read_projected(self.encode_visual(visual), geometry, bias)

    def as_prefix(self, readout):
        projected = self.to_llm(readout)
        return F.normalize(projected, dim=-1) * self.embedding_norm

    def forward(self, visual, geometry, bias):
        return self.as_prefix(self.read(visual, geometry, bias))
