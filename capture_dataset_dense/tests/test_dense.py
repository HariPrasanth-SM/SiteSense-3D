import argparse
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from capture_dataset.common import (FloorMap, camera_record, file_sha256, frame_complete,
                                    load_plan, map_identity, save_frame, write_json)
from capture_dataset.capture import report
from capture_dataset_dense.__main__ import config_from_args, planning_args
from capture_dataset_dense.planning import *
from capture_dataset_dense.reports import add_dense_reports


def args_for(**changes):
    p=argparse.ArgumentParser();planning_args(p)
    a=p.parse_args([])
    for k,v in changes.items():setattr(a,k,v)
    return a


def fixture(root):
    grid=np.zeros((64,80),np.int8)
    grid[:2]=grid[-2:]=100;grid[:,:2]=grid[:,-2:]=100
    grid[51:56,3:77]=-1  # narrow gray corridor
    grid[20:39,27:38]=100
    image=np.full(grid.shape,205,np.uint8);image[grid==0]=254;image[grid==100]=0
    Image.fromarray(np.flipud(image)).save(root/'map.pgm')
    (root/'map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm',resolution=.15,origin=[-6.,-4.8,0.],
                                   mode='trinary',negate=0,occupied_thresh=.65,free_thresh=.196)))
    (root/'fake.ply').write_bytes(b'SYNTHETIC CPU TEST ONLY - NOT A REAL SCENE')
    return FloorMap.load(root/'map.yaml')


class DenseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.map=fixture(self.root)
        self.cfg=config_from_args(args_for(width=64,height=48))
        self.selection=dict(pins=[],corridors=[],excluded_capture_disks=[])
    def tearDown(self):self.tmp.cleanup()
    def corridor(self, allow=True):
        return dict(points=[self.map.cell_to_world(c).tolist() for c in [(53,6),(53,38),(53,73)]],allow_unknown=allow)
    def plan(self,selection=None):
        selection=selection or self.selection
        ss=choose_stations(self.map,selection,self.cfg)
        return create_dense_plan(self.root/'map.yaml',self.root/'fake.ply',ss,selection,self.cfg)

    def test_default_budget(self):
        p=self.plan();self.assertEqual(len(p['stations']),200);self.assertEqual(len(p['frames']),2000)
        self.assertEqual(len({v['frame_id'] for v in p['frames']}),2000)
    def test_total_per_station(self):
        p=self.plan();_,counts=np.unique([v['station_id'] for v in p['frames']],return_counts=True)
        self.assertTrue((counts==10).all())
    def test_uneven_budget_stays_exact(self):
        self.cfg['requested_positions']=160;p=self.plan()
        _,counts=np.unique([v['station_id'] for v in p['frames']],return_counts=True)
        self.assertEqual(set(counts),{12,13});self.assertEqual(sum(counts),2000)
    def test_repeatable_frames(self):
        self.assertEqual(self.plan()['frames'],self.plan()['frames'])
    def test_new_seed_changes_views(self):
        p=self.plan();self.cfg['seed']=9;q=self.plan()
        self.assertNotEqual(p['frames'],q['frames']);self.assertEqual(p['stations'],q['stations'])
    def test_height_bounds_and_strata(self):
        vv=self.plan()['frames']
        for start in range(0,len(vv),10):
            vs=vv[start:start+10]
            self.assertEqual(set(v['height_stratum'] for v in vs),set(range(10)))
            self.assertTrue(all(.5<v['camera_height_m']<2.5 for v in vs))
            self.assertLess(min(v['camera_height_m'] for v in vs),.7)
            self.assertGreater(max(v['camera_height_m'] for v in vs),2.3)
    def test_pitch_avoids_ceiling_targets(self):
        for v in self.plan()['frames']:
            lo,hi=pitch_bounds(v['camera_height_m'])
            self.assertTrue(lo<=v['pitch_deg']<=hi)
            if v['camera_height_m']<.9:self.assertLessEqual(v['pitch_deg'],0)
            if v['camera_height_m']>=2:self.assertLessEqual(v['pitch_deg'],-10)
    def test_yaw_stratification(self):
        vv=self.plan()['frames']
        for i in range(0,len(vv),10):
            group=vv[i:i+10]
            self.assertEqual(set(v['yaw_sector'] for v in group),set(range(10)))
            self.assertTrue(all(0<=v['yaw_deg']<360 for v in group))
            for v in group:
                sector=int(((v['yaw_deg']-v['yaw_phase_deg'])%360)//36)
                self.assertEqual(sector,v['yaw_sector'])
    def test_corridor_has_many_unknown_cameras(self):
        self.selection['corridors']=[self.corridor()]
        p=self.plan();cs=[s for s in p['stations'] if s['corridor_id']]
        self.assertGreater(len(cs),35);self.assertTrue(all(s['unknown_capture_only'] for s in cs))
        auto=[s for s in p['stations'] if s['selection']=='auto_known_free']
        self.assertTrue(all(s['map_state']==0 for s in auto))
    def test_no_implicit_gray_permission(self):
        self.selection['corridors']=[self.corridor(False)]
        with self.assertRaisesRegex(ValueError,'GRAY'):self.plan()
    def test_no_occupied_corridor(self):
        a=self.map.cell_to_world((30,15));b=self.map.cell_to_world((30,45))
        with self.assertRaisesRegex(ValueError,'OCCUPIED'):validate_segment(self.map,a,b,True)
    def test_no_outside_corridor(self):
        with self.assertRaises(ValueError):validate_segment(self.map,[-100,0],[0,0],True)
    def test_corridor_spacing(self):
        ps=corridor_pins(self.map,[self.corridor()],.25)
        xy=np.array([[p['x'],p['y']] for p in ps])
        self.assertLessEqual(np.linalg.norm(np.diff(xy,axis=0),axis=1).max(),.25+self.map.resolution)
    def test_pins_corridor_deduplicate(self):
        cor=self.corridor();xy=cor['points'][0]
        self.selection['corridors']=[cor]
        self.selection['pins']=[dict(x=xy[0],y=xy[1],allow_unknown=True)]
        p=self.plan();self.assertEqual(p['stations'][0]['selection'],'manual_pin')
        self.assertTrue(p['stations'][0]['corridor_id'])
        self.assertEqual(len({(s['row'],s['col']) for s in p['stations']}),200)
    def test_source_map_unchanged(self):
        before=self.map.occupancy.copy();h=file_sha256(self.root/'map.pgm')
        self.selection['corridors']=[self.corridor()];self.plan()
        np.testing.assert_array_equal(before,self.map.occupancy)
        self.assertEqual(h,file_sha256(self.root/'map.pgm'))
    def test_pilot_has_48_and_gray_extremes(self):
        self.selection['corridors']=[self.corridor()]
        p=self.plan();vv=[v for v in p['frames'] if v['pilot']]
        self.assertEqual(len(vv),48)
        self.assertGreater(sum(v['unknown_capture_only'] for v in vv),0)
        for sid in p['pilot_station_ids']:
            h=[v['camera_height_m'] for v in vv if v['station_id']==sid]
            self.assertLess(min(h),.7);self.assertGreater(max(h),2.3)
    def test_camera_transforms_follow_height(self):
        p=self.plan()
        for v in p['frames'][::29]:
            c=camera_record(p,v);cm=np.array(c['camera_to_map']);cg=np.array(c['camera_to_world'])
            np.testing.assert_allclose(cm[:3,3],[v['x'],v['y'],v['camera_height_m']],atol=2e-6)
            np.testing.assert_allclose(np.array(c['world_to_camera'])@cg,np.eye(4),atol=3e-6)
            self.assertAlmostEqual(np.linalg.det(cm[:3,:3]),1,places=5)
    def test_save_is_base_compatible(self):
        p=self.plan();out=self.root/'dataset';save_dense_plan(out,p,self.selection)
        self.assertEqual(load_plan(out)['plan_hash'],p['plan_hash'])
        with (out/'frame_poses.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),2000)
        self.assertTrue((out/'camera_positions.png').exists())
        summary=read_json(out/'sampling_summary.json')
        self.assertEqual(summary['counts_in_height_bins'],[400]*5)
    def test_plan_refuses_overwrite(self):
        p=self.plan();out=self.root/'dataset';save_dense_plan(out,p,self.selection)
        with self.assertRaises(FileExistsError):save_dense_plan(out,p,self.selection)
    def test_legacy_selection_reusable(self):
        xy=self.map.cell_to_world((53,8));pin=dict(x=float(xy[0]),y=float(xy[1]),allow_unknown=True)
        path=self.root/'legacy.json'
        write_json(path,dict(map_identity=map_identity(self.root/'map.yaml'),pins=[pin],excluded_capture_disks=[]))
        s=load_selection(path,self.root/'map.yaml');self.assertEqual(s['corridors'],[])
        p=self.plan(s);self.assertTrue(p['stations'][0]['unknown_capture_only'])
    def test_corridor_selection_roundtrip(self):
        self.selection['corridors']=[self.corridor()];p=self.plan();out=self.root/'data'
        save_dense_plan(out,p,self.selection)
        self.assertEqual(load_selection(out/'selection.json',self.root/'map.yaml'),self.selection)
    def test_station_limit_no_fakes(self):
        self.cfg['auto_spacing_m']=100
        p=self.plan();self.assertEqual(len(p['stations']),1)
        self.assertEqual(len(p['frames']),2000)
    def test_corridor_exceed_budget(self):
        self.cfg['requested_positions']=5;self.selection['corridors']=[self.corridor()]
        with self.assertRaisesRegex(ValueError,'exceeding budget'):self.plan()
    def test_corridor_exclusion_conflict(self):
        cor=self.corridor();self.selection['corridors']=[cor]
        x,y=cor['points'][0];self.selection['excluded_capture_disks']=[dict(x=x,y=y,radius=.4)]
        with self.assertRaisesRegex(ValueError,'exclusion'):self.plan()
    def test_bad_cfg(self):
        for kw in [dict(frames=0),dict(positions=0),dict(height_min=3),dict(seed=-1),dict(seed=2**33),
                   dict(spacing=0),dict(height_max=float('nan')),dict(fov=180),dict(pilot_views=0)]:
            with self.assertRaises(ValueError):config_from_args(args_for(**kw))
    def test_saved_frame_and_dense_report(self):
        p=self.plan();out=self.root/'dataset';save_dense_plan(out,p,self.selection)
        v=next(v for v in p['frames'] if v['pilot']);folder=out/'frames'/v['frame_id']
        save_frame(folder,p,v,np.full((48,64,3),100,np.uint8),np.ones((48,64),np.float32),np.ones((48,64),np.float32),.02)
        self.assertTrue(frame_complete(folder,p['plan_hash']))
        report(out);add_dense_reports(out)
        self.assertTrue((out/'qa'/'dense_rgb_001.jpg').exists())
        self.assertTrue((out/'qa'/'dense.html').exists())
        with (out/'quality_dense.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(float(rows[0]['camera_height_m']),v['camera_height_m'])
    def test_source_modules_unmodified_by_plan(self):
        import capture_dataset.common as common
        import capture_dataset.capture as capture
        a=file_sha256(common.__file__);b=file_sha256(capture.__file__)
        self.plan()
        self.assertEqual(file_sha256(common.__file__),a);self.assertEqual(file_sha256(capture.__file__),b)


if __name__=='__main__':unittest.main()
