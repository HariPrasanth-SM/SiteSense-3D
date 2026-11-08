"""CPU-only map, planning, motion and camera-coordinate code. No ROS or CUDA imports."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt, label


class PlanningError(ValueError):
    """The requested action has no valid solution in the supplied map."""


def angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class FloorMap:
    # Rows increase toward local map +Y, unlike image pixel rows.
    occupancy: np.ndarray
    resolution: float
    origin: np.ndarray
    source: str = ""

    @classmethod
    def load(cls, filename: str | Path) -> "FloorMap":
        path = Path(filename).expanduser().resolve()
        with path.open() as stream:
            cfg = yaml.safe_load(stream)
        if cfg.get("mode", "trinary") != "trinary":
            raise ValueError("This prototype supports ROS trinary YAML/PGM maps only.")
        res = float(cfg["resolution"])
        origin = np.asarray(cfg["origin"], dtype=float)
        if not np.isfinite(res) or res <= 0 or origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Invalid map resolution or origin.")
        image_file = Path(str(cfg["image"])).expanduser()
        if not image_file.is_absolute():
            image_file = path.parent / image_file
        with Image.open(image_file) as image:
            # Existing Valmet map is grayscale PGM. For RGB, average the color channels.
            values = np.asarray(image.convert("RGB"), dtype=float).mean(axis=2) / 255.0
        occ_probability = values if int(cfg.get("negate", 0)) else 1.0 - values
        free_t, occupied_t = float(cfg["free_thresh"]), float(cfg["occupied_thresh"])
        if not 0 <= free_t < occupied_t <= 1:
            raise ValueError("Expected 0 <= free_thresh < occupied_thresh <= 1.")
        grid = np.full(values.shape, -1, dtype=np.int8)
        grid[occ_probability < free_t] = 0
        grid[occ_probability > occupied_t] = 100
        return cls(np.flipud(grid).copy(), res, origin, str(path))

    @property
    def shape(self):
        return self.occupancy.shape

    @property
    def rotation(self):
        c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
        return np.array([[c, -s], [s, c]])

    def world_to_grid_continuous(self, xy):
        """Return continuous (column, row), using the lower-left cell boundary as zero."""
        return (np.asarray(xy, dtype=float) - self.origin[:2]) @ self.rotation / self.resolution

    def world_to_cell(self, xy):
        col, row = np.floor(self.world_to_grid_continuous(xy)).astype(int)
        return int(row), int(col)

    def cell_to_world(self, cell):
        row, col = cell
        local = np.array([col + 0.5, row + 0.5]) * self.resolution
        return self.origin[:2] + self.rotation @ local

    def cells_to_world(self, cells):
        cells = np.asarray(cells)
        local = (cells[:, [1, 0]] + 0.5) * self.resolution
        return local @ self.rotation.T + self.origin[:2]

    def contains(self, cell):
        r, c = cell
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]


@dataclass
class Plan:
    xy: np.ndarray
    target: np.ndarray
    endpoint: np.ndarray
    mode: str
    note: str

    @property
    def length(self):
        if len(self.xy) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())


class Planner:
    """A* on known free cells; footprint erosion and no diagonal corner-cutting.

    This is a static, approximate 2-D planner, not a 3-D collision checker.
    Existing inflation in the input map is not removed.
    """
    MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
             (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

    def __init__(self, floor_map: FloorMap, radius: float = 0.25, margin: float = 0.03):
        if radius < 0 or margin < 0:
            raise ValueError("Radius and margin must be nonnegative.")
        self.map = floor_map
        self.radius, self.margin = radius, margin
        known_free = floor_map.occupancy == 0
        # Treat the map exterior as blocked, even for an otherwise all-free map.
        distances = distance_transform_edt(np.pad(known_free, 1, constant_values=False))[1:-1, 1:-1]
        self.clearance = distances * floor_map.resolution
        # An extra half cell diagonal is conservative for full cell traversal,
        # not just the center point. Unknown and occupied cells both block motion.
        required = radius + margin + floor_map.resolution / math.sqrt(2.0)
        self.walkable = known_free & (self.clearance > required)
        self.components, self.component_count = label(self.walkable)  # 4-connectivity
        self.free_cells = np.argwhere(self.walkable)

    def auto_start(self):
        if not len(self.free_cells):
            raise PlanningError(
                "No footprint-safe free cells. Inspect/correct the map. "
                "Existing obstacle inflation may already be excessive; do not mark unknown free automatically."
            )
        counts = np.bincount(self.components.ravel())
        counts[0] = 0
        largest = int(np.argmax(counts))
        score = np.where(self.components == largest, self.clearance, -1)
        cell = np.unravel_index(int(np.argmax(score)), score.shape)
        return self.map.cell_to_world(cell)

    def set_start(self, xy):
        cell = self.map.world_to_cell(xy)
        if not self.map.contains(cell) or not self.walkable[cell]:
            raise PlanningError("Start must be inside the footprint-safe free area. No automatic teleport through obstacles.")
        return self.map.cell_to_world(cell)

    def _neighbors(self, cell):
        r, c = cell
        for dr, dc, cost in self.MOVES:
            nb = (r + dr, c + dc)
            if not self.map.contains(nb) or not self.walkable[nb]:
                continue
            if dr and dc and not (self.walkable[r + dr, c] and self.walkable[r, c + dc]):
                continue
            yield nb, cost

    def astar(self, start, goal):
        """Return cells, including start and goal. All returned transitions are valid."""
        if not self.map.contains(start) or not self.map.contains(goal):
            raise PlanningError("Start or approach point lies outside the map.")
        if not self.walkable[start] or not self.walkable[goal]:
            raise PlanningError("Start or approach point violates map clearance.")
        if self.components[start] != self.components[goal]:
            raise PlanningError("No connected free path between start and approach point.")
        def h(cell):
            dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)
        scores = {start: 0.0}
        parents = {}
        queue = [(h(start), 0.0, start)]
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost > scores.get(cell, math.inf) + 1e-10:
                continue
            if cell == goal:
                path = [cell]
                while cell != start:
                    cell = parents[cell]
                    path.append(cell)
                return path[::-1]
            for nb, step in self._neighbors(cell):
                g = cost + step
                if g < scores.get(nb, math.inf):
                    scores[nb], parents[nb] = g, cell
                    heapq.heappush(queue, (g + h(nb), g, nb))
        raise PlanningError("A* could not find a path.")

    @staticmethod
    def _compress_collinear(cells):
        """Only remove exactly collinear intermediate cells; no unsafe curve smoothing."""
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        last_dir = np.asarray(cells[1]) - cells[0]
        for i in range(1, len(cells) - 1):
            direction = np.asarray(cells[i + 1]) - cells[i]
            if not np.array_equal(direction, last_dir):
                out.append(cells[i])
            last_dir = direction
        out.append(cells[-1])
        return out

    def plan(self, start_xy, target_xy, mode="object", standoff=0.8, max_approach=2.5):
        start_xy, target_xy = np.asarray(start_xy, float), np.asarray(target_xy, float)
        if not np.isfinite(start_xy).all() or not np.isfinite(target_xy).all():
            raise PlanningError("Coordinates must be finite.")
        start = self.map.world_to_cell(start_xy)
        target_cell = self.map.world_to_cell(target_xy)
        if not self.map.contains(start) or not self.walkable[start]:
            raise PlanningError("The current avatar position is not in footprint-safe free space.")
        if not self.map.contains(target_cell):
            raise PlanningError("The target is outside the map. Select a mapped location.")
        if mode == "point":
            goal = target_cell
            if not self.walkable[goal]:
                raise PlanningError("An exact destination must be walkable. Use 'Inspect object' for an occupied target.")
            note = "Destination snapped to its free cell center. Final heading follows the last segment."
        elif mode == "object":
            if not 0 < standoff <= max_approach:
                raise PlanningError("Expected 0 < standoff <= maximum approach distance.")
            # Choose only approach cells in the avatar's connected component.
            candidates = np.argwhere(self.components == self.components[start])
            world = self.map.cells_to_world(candidates)
            distance = np.linalg.norm(world - target_xy, axis=1)
            allowed = (distance >= standoff) & (distance <= max_approach)
            if not allowed.any():
                raise PlanningError(
                    "No reachable approach cell within the requested distance band. "
                    "Try another target/start, increase maximum approach distance, or review the map."
                )
            candidates, world, distance = candidates[allowed], world[allowed], distance[allowed]
            # Target stand-off is a point-to-center distance, not asset-surface clearance.
            # Slight preference for a closer start avoids arbitrary tie choices.
            score = distance - standoff + 0.05 * np.linalg.norm(world - start_xy, axis=1)
            goal = tuple(int(v) for v in candidates[int(np.argmin(score))])
            note = "Reachable nearby stop; final yaw faces target. Visibility and full 3-D clearance are NOT verified."
        else:
            raise PlanningError("Unknown target mode.")
        cells = self._compress_collinear(self.astar(start, goal))
        xy = self.map.cells_to_world(cells)
        # On a mid-route replan, first move to the center of the current free cell.
        if np.linalg.norm(start_xy - xy[0]) > 1e-8:
            xy = np.vstack([start_xy, xy])
        return Plan(xy, target_xy.copy(), self.map.cell_to_world(goal), mode, note)


class Walker:
    """Deterministic stop-turn-walk kinematics; not wheel/leg physics."""
    def __init__(self, xy, yaw=0.0, speed=0.35, turn_rate=math.radians(75)):
        self.xy = np.asarray(xy, dtype=float).copy()
        self.yaw, self.speed, self.turn_rate = float(yaw), speed, turn_rate
        self.plan: Plan | None = None
        self.index = 0
        self.running = False
        self.state = "idle"

    def assign(self, plan: Plan):
        self.plan = plan
        self.index = 1 if len(plan.xy) > 1 else len(plan.xy)
        self.running = False
        self.state = "planned"

    def start(self):
        if self.plan is None:
            raise PlanningError("Plan a route first.")
        self.running, self.state = True, "moving"

    def pause(self):
        self.running = False
        if self.state not in ("idle", "arrived"):
            self.state = "paused"

    def _turn(self, desired, dt):
        error = angle_wrap(desired - self.yaw)
        limit = self.turn_rate * dt
        self.yaw = angle_wrap(self.yaw + float(np.clip(error, -limit, limit)))
        return abs(error) <= limit + 1e-8

    def update(self, dt):
        if not self.running or self.plan is None:
            return
        dt = min(max(float(dt), 0.0), 0.1)  # Never leap through the map after a GUI stall.
        if self.index < len(self.plan.xy):
            destination = self.plan.xy[self.index]
            delta = destination - self.xy
            distance = float(np.linalg.norm(delta))
            if distance < 1e-7:
                self.index += 1
                return
            desired = math.atan2(delta[1], delta[0])
            if abs(angle_wrap(desired - self.yaw)) > 1e-3:
                self.state = "turning"
                self._turn(desired, dt)
                return
            self.state = "walking"
            self.xy += delta / distance * min(self.speed * dt, distance)
            if distance <= self.speed * dt + 1e-8:
                self.xy = destination.copy()
                self.index += 1
            return
        if self.plan.mode == "object":
            delta = self.plan.target - self.xy
            if np.linalg.norm(delta) > 1e-8:
                desired = math.atan2(delta[1], delta[0])
                self.state = "facing target"
                if not self._turn(desired, dt):
                    return
        self.state, self.running = "arrived", False


def camera_matrices(x, y, yaw, height, pitch_deg=0.0, scale=1.43, floor_y=1.1,
                    width=480, height_px=360, fov_y_deg=70.0):
    """Map frame: x=scale*gs.x, y=scale*gs.z, z=scale*(floor_y-gs.y).

    Camera: OpenCV +X right, +Y down, +Z forward. Positive pitch looks upward.
    Scale changes positions/depth, not rotation.
    """
    if scale <= 0 or not 1 < fov_y_deg < 179 or abs(pitch_deg) >= 89:
        raise ValueError("Invalid scale/FOV/pitch.")
    pitch = math.radians(pitch_deg)
    eye = np.array([x / scale, floor_y - height / scale, y / scale], dtype=np.float32)
    forward = np.array([math.cos(yaw) * math.cos(pitch), -math.sin(pitch),
                        math.sin(yaw) * math.cos(pitch)], dtype=np.float32)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    actual_up = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.column_stack([right, -actual_up, forward])
    c2w[:3, 3] = eye
    f = height_px / (2 * math.tan(math.radians(fov_y_deg) / 2))
    K = np.array([[f, 0, width / 2], [0, f, height_px / 2], [0, 0, 1]], dtype=np.float32)
    return K, c2w, np.linalg.inv(c2w).astype(np.float32)
