import math
import unittest
import numpy as np

from avatar_demo.core import FloorMap, Planner, angle_wrap, camera_matrices
from avatar_demo.manual import HeldControls, manual_step, point_allowed, segment_allowed, segment_cells, tilted_pitch


class ManualTests(unittest.TestCase):
    def setUp(self):
        grid = np.zeros((60, 80), np.int8)
        grid[[0, -1], :] = 100
        grid[:, [0, -1]] = 100
        grid[:, 40] = 100
        grid[30:35, 12:18] = -1
        self.m = FloorMap(grid, .1, np.zeros(3))
        self.p = Planner(self.m, .25, .03)

    def step(self, xy, yaw=0, f=1, t=0, collision=True, strict=False, speed=.25, dt=.1):
        return manual_step(xy, yaw, f, t, dt, speed, math.radians(60),
                           self.m, self.p.walkable, collision, strict)

    def test_forward_backward_relative_to_heading(self):
        start = np.array([2.5, 2.5])
        for yaw in (0, .5, math.pi/2, -math.pi):
            f = self.step(start, yaw=yaw)
            b = self.step(start, yaw=yaw, f=-1)
            np.testing.assert_allclose(f.xy-start, .025*np.array([math.cos(yaw), math.sin(yaw)]), atol=1e-9)
            np.testing.assert_allclose(f.xy-start, -(b.xy-start), atol=1e-9)

    def test_left_right_rotate_without_strafing(self):
        for turn in (-1, 1):
            r = self.step([2.5, 2.5], f=0, t=turn)
            np.testing.assert_array_equal(r.xy, [2.5, 2.5])
            self.assertAlmostEqual(r.yaw, math.radians(6)*turn)

    def test_curve_matches_unicycle(self):
        r = self.step([2.5, 2.5], t=1)
        v, omega, dt = .25, math.radians(60), .1
        expected = [2.5+v/omega*math.sin(omega*dt), 2.5+v/omega*(1-math.cos(omega*dt))]
        np.testing.assert_allclose(r.xy, expected, atol=1e-9)

    def test_wall_blocks_forward(self):
        pos = np.array([3.85, 2.55])
        for _ in range(20):
            r = self.step(pos)
            pos = r.xy
        self.assertTrue(r.blocked)
        self.assertLess(pos[0], 4.0)
        self.assertEqual(r.reason, 'occupied cell')

    def test_backward_collision(self):
        r = self.step([4.105, 2.55], f=-1)
        self.assertTrue(r.blocked)
        self.assertGreaterEqual(r.xy[0], 4.1)

    def test_unknown_blocks(self):
        r = self.step([1.195, 3.15])
        self.assertTrue(r.blocked)
        self.assertEqual(r.reason, 'unknown cell')

    def test_noclip_crosses_wall(self):
        r = self.step([3.99, 2.55], collision=False)
        self.assertGreater(r.xy[0], 4)
        self.assertFalse(r.blocked)

    def test_noclip_outside_map(self):
        r = self.step([7.99, 3.55], collision=False)
        self.assertGreater(r.xy[0], 8)
        self.assertFalse(r.blocked)

    def test_checks_on_inside_obstacle_freezes_translation_not_turn(self):
        r = self.step([4.05, 2.55], t=1)
        np.testing.assert_array_equal(r.xy, [4.05, 2.55])
        self.assertTrue(r.blocked)
        self.assertGreater(r.yaw, 0)

    def test_relaxed_manual_allows_amber(self):
        xy = self.m.cell_to_world((20, 38))
        self.assertFalse(self.p.walkable[20, 38])
        self.assertTrue(point_allowed(self.m, self.p.walkable, xy, strict=False))
        self.assertFalse(point_allowed(self.m, self.p.walkable, xy, strict=True))
        self.assertFalse(self.step(xy, strict=False).blocked)
        self.assertTrue(self.step(xy, strict=True).blocked)

    def test_teleport_not_inferred_on_guard_enable(self):
        xy = np.array([-1., -1.])
        r = self.step(xy, collision=True)
        np.testing.assert_array_equal(r.xy, xy)
        self.assertTrue(r.blocked)

    def test_large_dt_clamped(self):
        a = self.step([2.55, 2.55], dt=10)
        b = self.step([2.55, 2.55], dt=.1)
        np.testing.assert_allclose(a.xy,b.xy)

    def test_exact_segment_no_thin_wall_tunneling(self):
        allowed, why = segment_allowed(self.m, self.p.walkable, [2.55, 2.55], [5.55, 2.55])
        self.assertFalse(allowed)
        self.assertEqual(why, 'occupied cell')

    def test_exact_segment_no_diagonal_corner_cut(self):
        m = FloorMap(np.array([[0,100],[100,0]],np.int8),1,np.zeros(3))
        p = Planner(m,0,0)
        cells = set(segment_cells(m,[.5,.5],[1.5,1.5]))
        self.assertIn((0,1),cells)
        self.assertIn((1,0),cells)
        self.assertFalse(segment_allowed(m,p.walkable,[.5,.5],[1.5,1.5])[0])

    def test_segment_boundary_grazing_conservative(self):
        allowed, why = segment_allowed(self.m,self.p.walkable,[4.,1.5],[4.,2.5])
        self.assertFalse(allowed)

    def test_rotated_map(self):
        self.m.origin=np.array([-2.,3.,.4])
        a=self.m.cell_to_world((20,20));b=self.m.cell_to_world((20,60))
        self.assertFalse(segment_allowed(self.m,self.p.walkable,a,b)[0])

    def test_input_map_never_changed(self):
        before=self.m.occupancy.copy()
        self.step([3.99,2.5],collision=False)
        self.step([3.99,2.5],collision=True)
        np.testing.assert_array_equal(before,self.m.occupancy)

    def test_input_sources_release_independently(self):
        c=HeldControls()
        c.press('key:Up','forward');c.press('mouse:forward','forward')
        c.release('key:Up')
        self.assertEqual(c.axes(),(1,0,0))
        c.release('mouse:forward')
        self.assertEqual(c.axes(),(0,0,0))

    def test_opposing_keys_cancel(self):
        c=HeldControls()
        c.press('1','forward');c.press('2','backward');c.press('3','left');c.press('4','right')
        c.press('5','look_up');c.press('6','look_down')
        self.assertEqual(c.axes(),(0,0,0))
        c.clear();self.assertEqual(c.sources,{})

    def test_pitch_rate_and_limits(self):
        self.assertEqual(tilted_pitch(0,1,.1),2.5)
        self.assertEqual(tilted_pitch(0,-1,.1),-2.5)
        self.assertEqual(tilted_pitch(79,1,.1),80)
        self.assertEqual(tilted_pitch(-79,-1,.1),-80)

    def test_head_tilt_camera_height_unchanged(self):
        R=np.array([[1,0,0],[0,0,1],[0,-1,0]])
        for pitch in (-80,0,80):
            _,c2w,_=camera_matrices(1,2,.3,1.7,pitch_deg=pitch)
            pos=1.43*(R@c2w[:3,3])+[0,0,1.43*1.1]
            np.testing.assert_allclose(pos,[1,2,1.7],atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.det(c2w[:3,:3])),1,places=5)

    def test_invalid_motion_rejected(self):
        with self.assertRaises(ValueError):
            self.step([np.nan,0])
        with self.assertRaises(ValueError):
            self.step([0,0],speed=-1)


if __name__=='__main__':
    unittest.main()
