"""Map-based camera spawn clearance, in the map's provisional metre units.

The source occupancy map is never mutated. These distances concern the 2-D
occupancy boundary, NOT a measured physical wall or height-dependent 3-D surface.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import distance_transform_edt


def required_clearance(cfg):
    """Keep the existing --clearance/config name, now enforced for ALL stations."""
    value = float(cfg.get('auto_clearance_m', .25))
    if not math.isfinite(value) or value < 0:
        raise ValueError('Camera spawn clearance must be finite and nonnegative.')
    return value


def clearance_layers(fmap):
    """Conservative cell-center to blocked-cell-square distance lower bounds.

    EDT measures distance to cell centers. Subtracting half a cell diagonal
    ensures a passing center is at least the requested distance from the entire
    blocked square (including diagonal corners). Pad outside as blocked.

    wall_m: occupied cells and map exterior are blockers; unknown is NOT assumed
            to be a wall. Only explicitly approved pins/lines may use gray cells.
    known_m: occupied, unknown and map exterior block automatic free sampling.
    """
    occ = np.asarray(fmap.occupancy)
    half_diagonal = float(fmap.resolution) / math.sqrt(2.)

    def distances(eligible):
        dist = distance_transform_edt(np.pad(eligible, 1, constant_values=False),
                                      sampling=float(fmap.resolution))[1:-1, 1:-1]
        return np.maximum(0., dist - half_diagonal)

    return {'wall_m': distances(occ != 100),
            'known_m': distances(occ == 0),
            'cell_edge_allowance_m': half_diagonal}


def pin_clearance(fmap, pin, layers, required):
    """Validate a selected center; return snap and exclusion details, without moving it."""
    xy = np.array([pin['x'], pin['y']], dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError('Pinned coordinates must be finite.')
    cell = fmap.world_to_cell(xy)
    if not fmap.contains(cell):
        raise ValueError('Pinned camera is outside the map.')
    state = int(fmap.occupancy[cell])
    if state == 100:
        raise ValueError('Occupied cells cannot be pinned in this capture tool.')
    if state == -1 and not pin.get('allow_unknown', False):
        raise ValueError('Unknown pin requires explicit capture-only approval.')
    d = float(layers['wall_m'][cell])
    snapped = fmap.cell_to_world(cell)
    return cell, {'x': float(snapped[0]), 'y': float(snapped[1]),
                  'row': int(cell[0]), 'col': int(cell[1]),
                  'wall_clearance_lower_bound_m': d,
                  'required_clearance_m': required,
                  'map_state': state,
                  'accepted': bool(d + 1e-12 >= required)}
