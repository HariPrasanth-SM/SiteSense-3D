import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from avatar_demo.core import FloorMap, Planner, PlanningError, Walker, angle_wrap, camera_matrices


def blank_map(rows=60, cols=80, resolution=0.1):
    grid = np.zeros((rows, cols), dtype=np.int8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 100
    return FloorMap(grid, resolution, np.array([-2., -1., 0.]))


class CoreTests(unittest.TestCase):
    def test_map_image_flip_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            # Top image row = upper grid row.
            image = np.array([[254, 205], [0, 254]], dtype=np.uint8)
            Image.fromarray(image).save(path / "map.pgm")
            cfg = dict(image="map.pgm", resolution=.2, origin=[-1., -2., 0.], negate=0,
                       free_thresh=.196, occupied_thresh=.65, mode="trinary")
            (path / "map.yaml").write_text(yaml.safe_dump(cfg))
            m = FloorMap.load(path / "map.yaml")
            np.testing.assert_array_equal(m.occupancy, [[100, 0], [0, -1]])
            np.testing.assert_allclose(m.cell_to_world((0, 0)), [-.9, -1.9])

    def test_coordinate_roundtrip_rotated_origin(self):
        m = blank_map()
        m.origin[2] = .37
        for cell in [(0, 0), (12, 43), (59, 79)]:
            self.assertEqual(m.world_to_cell(m.cell_to_world(cell)), cell)
        cells = np.array([[4, 7], [18, 31]])
        np.testing.assert_allclose(m.cells_to_world(cells), [m.cell_to_world(c) for c in cells])

    def test_unknown_is_blocked(self):
        m = blank_map()
        m.occupancy[10:20, 10:20] = -1
        p = Planner(m, radius=.15, margin=.02)
        self.assertFalse(p.walkable[15, 15])
        self.assertTrue(np.all(~p.walkable[m.occupancy != 0]))

    def test_no_diagonal_corner_cut(self):
        grid = np.array([[0, 100], [100, 0]], dtype=np.int8)
        m = FloorMap(grid, 1., np.zeros(3))
        p = Planner(m, radius=0, margin=0)
        with self.assertRaises(PlanningError):
            p.astar((0, 0), (1, 1))

    def test_obstacle_detour(self):
        m = blank_map()
        m.occupancy[1:43, 39:42] = 100
        p = Planner(m, radius=.1, margin=0)
        path = p.astar((15, 15), (15, 60))
        self.assertGreater(max(c[0] for c in path), 42)
        for a, b in zip(path, path[1:]):
            self.assertTrue(p.walkable[b])
            self.assertIn(b, [v[0] for v in p._neighbors(a)])

    def test_occupied_target_reachable_approach(self):
        m = blank_map()
        m.occupancy[25:35, 35:45] = 100
        p = Planner(m, radius=.2, margin=.02)
        start, target = m.cell_to_world((20, 15)), m.cell_to_world((30, 40))
        plan = p.plan(start, target, standoff=.8, max_approach=2)
        self.assertTrue(p.walkable[m.world_to_cell(plan.endpoint)])
        self.assertGreaterEqual(np.linalg.norm(plan.endpoint - target), .8)
        self.assertLessEqual(np.linalg.norm(plan.endpoint - target), 2)
        np.testing.assert_array_equal(plan.target, target)

    def test_approach_stays_in_start_component(self):
        m = blank_map()
        m.occupancy[:, 40] = 100
        p = Planner(m, radius=.1, margin=0)
        start, target = m.cell_to_world((25, 20)), m.cell_to_world((25, 42))
        plan = p.plan(start, target, standoff=.8, max_approach=2.5)
        self.assertLess(m.world_to_cell(plan.endpoint)[1], 40)

    def test_unreachable_exact_destination(self):
        m = blank_map()
        m.occupancy[:, 40] = 100
        p = Planner(m, radius=.1, margin=0)
        with self.assertRaises(PlanningError):
            p.plan(m.cell_to_world((20, 20)), m.cell_to_world((20, 60)), mode="point")

    def test_nonfree_start_not_silently_snapped(self):
        m = blank_map()
        p = Planner(m, radius=.25)
        with self.assertRaises(PlanningError):
            p.set_start(m.cell_to_world((0, 0)))

    def test_walker_stops_and_faces_target(self):
        m = blank_map()
        m.occupancy[25:35, 35:45] = 100
        p = Planner(m, radius=.2, margin=.02)
        start, target = m.cell_to_world((20, 15)), m.cell_to_world((30, 40))
        plan = p.plan(start, target)
        w = Walker(start, yaw=-2, speed=.7)
        w.assign(plan)
        w.start()
        for _ in range(12000):
            w.update(.04)
            self.assertTrue(p.walkable[m.world_to_cell(w.xy)])
            if w.state == "arrived":
                break
        self.assertEqual(w.state, "arrived")
        self.assertFalse(w.running)
        np.testing.assert_allclose(w.xy, plan.endpoint, atol=1e-8)
        delta = target - w.xy
        self.assertLess(abs(angle_wrap(w.yaw - math.atan2(delta[1], delta[0]))), 1e-8)
        before = w.xy.copy()
        w.update(10)
        np.testing.assert_array_equal(w.xy, before)

    def test_midsegment_replan_connector(self):
        m = blank_map()
        p = Planner(m, radius=.1, margin=0)
        start = m.cell_to_world((20, 15)) + [.02, .01]
        plan = p.plan(start, m.cell_to_world((40, 50)), mode="point")
        np.testing.assert_allclose(plan.xy[0], start)
        self.assertEqual(m.world_to_cell(plan.xy[1]), m.world_to_cell(start))

    def test_camera_transform_heading_height(self):
        s, fy = 1.43, 1.1
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
        t = np.array([0, 0, s*fy])
        for yaw in [0, .8, math.pi/2, -math.pi]:
            K, c2w, w2c = camera_matrices(2.3, -1.2, yaw, 1.7)
            np.testing.assert_allclose(s * (R @ c2w[:3, 3]) + t, [2.3, -1.2, 1.7], atol=1e-6)
            np.testing.assert_allclose(R @ c2w[:3, 2], [math.cos(yaw), math.sin(yaw), 0], atol=1e-6)
            np.testing.assert_allclose(c2w @ w2c, np.eye(4), atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.det(c2w[:3, :3])), 1., places=5)

    def test_camera_pitch(self):
        _, c2w, _ = camera_matrices(0, 0, 0, 1.7, pitch_deg=-20)
        # Downward camera looks toward positive scene Y.
        self.assertGreater(c2w[1, 2], 0)

    def test_auto_start_and_empty_map(self):
        m = blank_map()
        p = Planner(m)
        self.assertTrue(p.walkable[m.world_to_cell(p.auto_start())])
        m.occupancy[:] = -1
        with self.assertRaises(PlanningError):
            Planner(m).auto_start()


if __name__ == "__main__":
    unittest.main()
