"""Demo-only gray-cell traversal, with original map pixels preserved."""
from __future__ import annotations
import math
import numpy as np
from avatar_demo.core import FloorMap, Planner
from avatar_demo.manual import segment_cells


def navigation_map(source: FloorMap, allow_unknown: bool = True) -> FloorMap:
    grid = source.occupancy.copy()
    if allow_unknown:
        grid[grid == -1] = 0
    return FloorMap(grid, source.resolution, source.origin.copy(), source.source)


def preferred_start(source: FloorMap, planner: Planner):
    # Prefer an originally green cell, not an exterior gray patch.
    free = (source.occupancy == 0) & planner.walkable
    if not free.any():
        return planner.auto_start()
    counts = np.bincount(planner.components[free], minlength=planner.component_count + 1)
    counts[0] = 0
    largest = int(np.argmax(counts))
    choices = free & (planner.components == largest)
    cell = np.unravel_index(np.argmax(np.where(choices, planner.clearance, -1)), source.shape)
    return source.cell_to_world(cell)


def route_unknown_cells(source: FloorMap, xy):
    cells = set()
    for a, b in zip(xy[:-1], xy[1:]):
        cells.update(segment_cells(source, a, b))
    if len(xy) == 1:
        cells.add(source.world_to_cell(xy[0]))
    return sum(source.contains(c) and source.occupancy[c] == -1 for c in cells)


def target_orientation(xy, target_xyz, eye_height):
    d = np.asarray(target_xyz, float) - [*xy, float(eye_height)]
    yaw = math.atan2(d[1], d[0])
    pitch = float(np.clip(math.degrees(math.atan2(d[2], max(float(np.linalg.norm(d[:2])), 1e-6))), -80, 80))
    return yaw, pitch


def same_pose(a, b, tolerance=0.002):
    if a is None or b is None or len(a) != 5 or len(b) != 5:
        return False
    d = np.asarray(a, float) - np.asarray(b, float)
    d[2] = (d[2] + math.pi) % (2 * math.pi) - math.pi
    return bool(np.isfinite(d).all() and np.max(np.abs(d)) <= tolerance)
