"""CPU-only manual input and grid collision checks; never edits the occupancy map."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import FloorMap, angle_wrap


class HeldControls:
    """Track independent keyboard/mouse sources without relying on key repeat."""
    ACTIONS = {"forward", "backward", "left", "right", "look_up", "look_down"}

    def __init__(self):
        self.sources: dict[str, str] = {}

    def press(self, source: str, action: str):
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown manual action: {action}")
        self.sources[source] = action

    def release(self, source: str):
        self.sources.pop(source, None)

    def clear(self):
        self.sources.clear()

    def axes(self):
        active = set(self.sources.values())
        return (int("forward" in active) - int("backward" in active),
                int("left" in active) - int("right" in active),
                int("look_up" in active) - int("look_down" in active))


def segment_cells(floor_map: FloorMap, a, b):
    """All cells touched by a line, including either side of grid-edge/corner contact.

    Use exact grid crossing events, not sparse point sampling. This stops a short
    diagonal from cutting through an occupied corner or skipping a thin wall.
    """
    p = floor_map.world_to_grid_continuous(a)
    q = floor_map.world_to_grid_continuous(b)
    if not np.isfinite([*p, *q]).all():
        raise ValueError("Motion coordinates must be finite.")
    delta = q - p
    times = [0.0, 1.0]
    for axis in range(2):
        if abs(delta[axis]) > 1e-12:
            lo, hi = sorted([float(p[axis]), float(q[axis])])
            for edge in range(math.floor(lo) + 1, math.ceil(hi)):
                t = (edge - p[axis]) / delta[axis]
                if 0.0 < t < 1.0:
                    times.append(float(t))
    times = sorted(set(times))
    probes = times + [(a + b) / 2 for a, b in zip(times[:-1], times[1:])]
    result = []
    seen = set()
    for t in sorted(probes):
        col, row = p + t * delta
        def indices(value):
            integer = round(float(value))
            if abs(float(value) - integer) < 1e-9:
                return (integer - 1, integer)
            return (math.floor(float(value)),)
        for r in indices(row):
            for c in indices(col):
                if (r, c) not in seen:
                    seen.add((r, c))
                    result.append((r, c))
    return result


def point_allowed(floor_map: FloorMap, walkable, xy, strict=False):
    cell = floor_map.world_to_cell(xy)
    return (floor_map.contains(cell) and
            bool(walkable[cell] if strict else floor_map.occupancy[cell] == 0))


def segment_allowed(floor_map: FloorMap, walkable, a, b, strict=False):
    for cell in segment_cells(floor_map, a, b):
        if not floor_map.contains(cell):
            return False, "map boundary"
        value = int(floor_map.occupancy[cell])
        if value != 0:
            return False, "unknown cell" if value == -1 else "occupied cell"
        if strict and not walkable[cell]:
            return False, "body-clearance margin"
    return True, ""


@dataclass
class ManualResult:
    xy: np.ndarray
    yaw: float
    blocked: bool
    reason: str


def manual_step(xy, yaw, forward, turn, dt, speed, turn_rate,
                floor_map: FloorMap, walkable, collision=True, strict=False):
    """Forward/reverse and yaw, no strafing. All rates use seconds and radians.

    ON: either point-centre known-free checking or the existing strict footprint
    mask. OFF: inspection-only unrestricted planar motion, even beyond map bounds.
    Turns remain possible while translation is blocked. This is not 3-D collision.
    """
    xy = np.asarray(xy, dtype=float).copy()
    vals = [*xy, yaw, forward, turn, dt, speed, turn_rate]
    if xy.shape != (2,) or not np.isfinite(vals).all() or speed <= 0 or turn_rate <= 0:
        raise ValueError("Invalid manual-motion arguments.")
    dt = min(max(float(dt), 0.0), 0.1)
    v = float(np.clip(forward, -1, 1)) * speed
    omega = float(np.clip(turn, -1, 1)) * turn_rate
    final_yaw = angle_wrap(yaw + omega * dt)
    if abs(v) < 1e-12 or dt <= 0:
        return ManualResult(xy, final_yaw, False, "")
    if collision and not point_allowed(floor_map, walkable, xy, strict):
        return ManualResult(xy, final_yaw, True,
                            "current position is not allowed; turn checks OFF to return or use Return to free")
    # Bound curve approximation and travelled distance per segment.
    steps = max(1, math.ceil(abs(v) * dt / (floor_map.resolution * .25)),
                math.ceil(abs(omega) * dt / math.radians(3)))
    h = dt / steps
    angle = yaw
    for _ in range(steps):
        next_angle = angle + omega * h
        if abs(omega) < 1e-10:
            proposed = xy + v * h * np.array([math.cos(angle), math.sin(angle)])
        else:
            proposed = xy + (v / omega) * np.array([
                math.sin(next_angle) - math.sin(angle),
                math.cos(angle) - math.cos(next_angle)])
        if collision:
            allowed, why = segment_allowed(floor_map, walkable, xy, proposed, strict)
            if not allowed:
                return ManualResult(xy, final_yaw, True, why)
        xy = proposed
        angle = next_angle
    return ManualResult(xy, final_yaw, False, "")


def tilted_pitch(pitch_deg, direction, dt, rate_deg=25.0):
    if not np.isfinite([pitch_deg, direction, dt, rate_deg]).all() or rate_deg <= 0:
        raise ValueError("Invalid head-tilt arguments.")
    delta = np.clip(direction, -1, 1) * rate_deg * min(max(float(dt), 0), .1)
    return float(np.clip(pitch_deg + delta, -80.0, 80.0))
