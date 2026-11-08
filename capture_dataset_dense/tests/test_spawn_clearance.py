"""Artificial-grid regression tests; no proprietary scene, GPU, or rendering needed."""
import argparse
import copy
import math
import os
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest

import numpy as np

from avatar_demo.core import FloorMap
from capture_dataset.common import load_plan, file_sha256, read_json, world_to_pixel
from capture_dataset_dense.clearance import clearance_layers, pin_clearance, required_clearance
from capture_dataset_dense.__main__ import planning_args, config_from_args
from capture_dataset_dense.planning import choose_stations, create_dense_plan, save_dense_plan, HEADER
from capture_dataset_dense.picker import DensePicker
from test_dense import fixture, args_for


def cfg(**kwargs):
    return config_from_args(args_for(**kwargs))


def selection():
    return dict(pins=[], corridors=[], excluded_capture_disks=[])


def pin(fmap, cell, allow=False):
    x, y = fmap.cell_to_world(cell)
    return dict(x=float(x), y=float(y), allow_unknown=allow)


class SpawnClearanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fmap = fixture(self.root)
    def tearDown(self):
        self.tmp.cleanup()

    def test_default_and_alias(self):
        p = argparse.ArgumentParser(); planning_args(p)
        self.assertEqual(config_from_args(p.parse_args([]))['auto_clearance_m'], .25)
        self.assertEqual(config_from_args(p.parse_args(['--wall-clearance', '.30']))['auto_clearance_m'], .30)
        self.assertEqual(config_from_args(p.parse_args(['--clearance', '.20']))['auto_clearance_m'], .20)

    def test_cell_square_edge_not_center(self):
        grid = np.zeros((60, 60), dtype=np.int8); grid[:, 10] = 100
        fmap = FloorMap(grid, .05, np.zeros(3))
        layers = clearance_layers(fmap)
        # From column 15 to wall column 10 is 0.25 center-to-center but <0.25 to its edge.
        _, a = pin_clearance(fmap, pin(fmap, (30, 15)), layers, .25)
        _, b = pin_clearance(fmap, pin(fmap, (30, 16)), layers, .25)
        self.assertFalse(a['accepted']); self.assertTrue(b['accepted'])
        self.assertLessEqual(layers['wall_m'][30, 16], (16-10-.5)*.05)

    def test_diagonal_corner_is_buffered(self):
        grid = np.zeros((60,60), np.int8); grid[20,20] = 100
        fmap = FloorMap(grid, .05, np.zeros(3)); layers = clearance_layers(fmap)
        actual = math.sqrt(2)*3.5*.05
        self.assertAlmostEqual(layers['wall_m'][24,24], actual)
        self.assertLess(layers['wall_m'][24,24], .25)

    def test_map_exterior_is_buffered_even_without_walls(self):
        fmap = FloorMap(np.zeros((30,30), np.int8), .1, np.zeros(3))
        bounds = clearance_layers(fmap)
        self.assertLess(bounds['wall_m'][0,15], .25)
        self.assertGreater(bounds['wall_m'][15,15], .25)

    def test_all_automatic_stations_pass_known_boundary(self):
        for margin in (.20, .25, .30):
            ss = choose_stations(self.fmap, selection(), cfg(clearance=margin))
            self.assertTrue(all(s['wall_clearance_lower_bound_m'] >= margin for s in ss))
            self.assertTrue(all(s['known_free_clearance_lower_bound_m'] >= margin for s in ss))
            self.assertTrue(all(s['map_state'] == 0 for s in ss))

    def test_near_wall_pin_skipped_not_shifted(self):
        sel = selection(); bad = pin(self.fmap, (2,10)); good = pin(self.fmap,(10,10))
        sel['pins'] = [bad,good]; before = copy.deepcopy(sel); audit = {}
        ss = choose_stations(self.fmap, sel, cfg(), audit)
        self.assertEqual(sel, before)
        self.assertEqual(audit['rejected_count'], 1)
        self.assertEqual(audit['rejected_explicit_samples'][0]['row'], 2)
        kept = [s for s in ss if s['selection'] == 'manual_pin']
        self.assertEqual(len(kept), 1)
        self.assertEqual([kept[0]['x'],kept[0]['y']], [good['x'],good['y']])
        self.assertEqual(len(ss), 200)

    def test_corridor_samples_near_wall_skipped(self):
        sel = selection(); sel['corridors'] = [dict(
            points=[self.fmap.cell_to_world(c).tolist() for c in [(53,3),(53,73)]],
            allow_unknown=True)]
        audit={}; ss=choose_stations(self.fmap,sel,cfg(),audit)
        self.assertGreater(audit['rejected_count'], 0)
        corridor=[s for s in ss if s.get('corridor_id')]
        self.assertGreater(len(corridor), 0)
        self.assertTrue(all(s['wall_clearance_lower_bound_m'] >= .25 for s in corridor))
        self.assertTrue(all(s['unknown_capture_only'] for s in corridor))

    def test_approved_gray_does_not_bypass_wall_clearance(self):
        self.fmap.occupancy[2,10] = -1
        sel=selection();sel['pins']=[pin(self.fmap,(2,10),True)]
        audit={};ss=choose_stations(self.fmap,sel,cfg(),audit)
        self.assertEqual(audit['rejected_count'],1)
        self.assertFalse(any((s['row'],s['col'])==(2,10) for s in ss))

    def test_gray_interior_kept_without_assuming_gray_wall(self):
        sel=selection();sel['pins']=[pin(self.fmap,(53,10),True)]
        ss=choose_stations(self.fmap,sel,cfg())
        self.assertTrue(ss[0]['unknown_capture_only'])
        self.assertEqual(ss[0]['known_free_clearance_lower_bound_m'],0)
        self.assertGreaterEqual(ss[0]['wall_clearance_lower_bound_m'],.25)

    def test_gray_permission_still_required(self):
        sel=selection();sel['pins']=[pin(self.fmap,(53,10),False)]
        with self.assertRaisesRegex(ValueError,'explicit'):choose_stations(self.fmap,sel,cfg())

    def test_occupied_and_outside_not_skipped_silently(self):
        for p in (pin(self.fmap,(0,10)),dict(x=999,y=999,allow_unknown=True)):
            sel=selection();sel['pins']=[p]
            with self.assertRaises(ValueError):choose_stations(self.fmap,sel,cfg())

    def test_no_automatic_relaxation_when_nothing_fits(self):
        narrow=FloorMap(np.zeros((3,30),np.int8),.1,np.zeros(3))
        with self.assertRaisesRegex(ValueError,'No camera positions satisfy'):
            choose_stations(narrow,selection(),cfg())

    def test_frame_budget_stays_exact_with_fewer_stations(self):
        c=cfg(spacing=100)
        ss=choose_stations(self.fmap,selection(),c)
        p=create_dense_plan(self.root/'map.yaml',self.root/'fake.ply',ss,selection(),c)
        self.assertEqual(len(ss),1)
        self.assertEqual(len(p['frames']),2000)
        self.assertTrue(all(f['spawn_required_clearance_m']==.25 for f in p['frames']))

    def test_report_and_csv_saved_source_unchanged(self):
        sel=selection();sel['pins']=[pin(self.fmap,(2,10))]
        c=cfg();before=file_sha256(self.root/'map.pgm')
        ss=choose_stations(self.fmap,sel,c)
        p=create_dense_plan(self.root/'map.yaml',self.root/'fake.ply',ss,sel,c)
        out=self.root/'out';save_dense_plan(out,p,sel)
        audit=read_json(out/'spawn_clearance_report.json')
        self.assertEqual(audit['rejected_count'],1)
        self.assertEqual(load_plan(out)['plan_hash'],p['plan_hash'])
        self.assertIn('wall_clearance_lower_bound_m',(out/'camera_positions.csv').read_text().splitlines()[0])
        self.assertIn('spawn_required_clearance_m',(out/'frame_poses.csv').read_text().splitlines()[0])
        self.assertEqual(before,file_sha256(self.root/'map.pgm'))
        # Store the ORIGINAL requested pin so it can be reviewed in a later selection.
        self.assertEqual(read_json(out/'selection.json')['pins'],sel['pins'])

    def test_invalid_settings(self):
        for value in (-.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):required_clearance(dict(auto_clearance_m=value))

    def test_rotation_origin_does_not_change_grid_clearance(self):
        ss=choose_stations(self.fmap,selection(),cfg(positions=10))
        rotated=FloorMap(self.fmap.occupancy.copy(),self.fmap.resolution,np.array([10.,20.,.6]))
        rr=choose_stations(rotated,selection(),cfg(positions=10))
        # Tied farthest-point samples can break differently after rotation due to
        # floating-point roundoff; the geometric clearance must still be identical.
        np.testing.assert_allclose(clearance_layers(self.fmap)['wall_m'],
                                   clearance_layers(rotated)['wall_m'])
        self.assertTrue(all(s['wall_clearance_lower_bound_m'] >= .25 for s in rr))
        for s in rr:
            self.assertEqual(rotated.world_to_cell([s['x'],s['y']]),(s['row'],s['col']))


@unittest.skipUnless(os.environ.get('DISPLAY'),'requires X display')
class SpawnClearanceUITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.fmap=fixture(self.path);self.root=tk.Tk()
        self.args=args_for(map=str(self.path/'map.yaml'),scene=str(self.path/'fake.ply'),out=str(self.path/'out'))
        self.cfg=config_from_args(self.args)
        self.ui=DensePicker(self.root,self.args,self.cfg,selection());self.root.update()
    def tearDown(self):
        try:self.root.destroy()
        except tk.TclError:pass
        self.tmp.cleanup()
    def event(self,cell):
        x,y=world_to_pixel(self.fmap,self.fmap.cell_to_world(cell),self.ui.size)
        return SimpleNamespace(x=x,y=y+HEADER,state=0)
    def test_pin_rejected_at_click_time(self):
        self.ui.click(self.event((2,10)))
        self.assertEqual(len(self.ui.selection['pins']),0)
        self.assertIn('Too close',self.ui.status.get())
    def test_hover_exposes_clearance(self):
        self.ui.hover(self.event((10,10)))
        self.assertIn('wall buffer',self.ui.hover_text.get())
        self.assertIn('0.25',self.ui.hover_text.get())
    def test_loaded_near_pin_is_reported(self):
        self.ui.selection['pins']=[pin(self.fmap,(2,10))];self.ui.refresh()
        self.assertEqual(self.ui.audit['rejected_count'],1)
        self.assertIn('1 selected samples skipped',self.ui.status.get())
    def test_corridor_draw_filters_samples(self):
        self.ui.mode.set('corridor');self.ui.allow_unknown.set(True)
        self.ui.click(self.event((53,3)));self.ui.click(self.event((53,73)))
        self.ui.finish_corridor()
        self.assertEqual(len(self.ui.selection['corridors']),1)
        self.assertGreater(self.ui.audit['rejected_count'],0)
        self.assertTrue(all(s['wall_clearance_lower_bound_m']>=.25 for s in self.ui.stations))

if __name__=='__main__':unittest.main()
