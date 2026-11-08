import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from capture_dataset.common import *
from capture_dataset.capture import report
from capture_dataset.__main__ import config_from_args


def make_args(**changes):
    values = dict(width=64, height=48, camera_height=1.7, scale=1.43, floor_y=1.1,
                  fov=70., yaw_step=60, pitches=[0., -20.], sh_degree=3,
                  alpha_min=.2, positions=40, clearance=.10, spacing=.50)
    values.update(changes)
    return argparse.Namespace(**values)


def create_map(root):
    grid = np.zeros((40, 50), dtype=np.int8)
    grid[:2] = grid[-2:] = 100
    grid[:, :2] = grid[:, -2:] = 100
    grid[29:34, 3:47] = -1
    grid[15:22, 17:22] = 100
    im = np.full(grid.shape, 205, np.uint8)
    im[grid == 0] = 254; im[grid == 100] = 0
    Image.fromarray(np.flipud(im)).save(root / "map.pgm")
    cfg = dict(image="map.pgm", resolution=.2, origin=[-5.,-4.,0.], mode="trinary",
               negate=0, occupied_thresh=.65, free_thresh=.196)
    (root / "map.yaml").write_text(yaml.safe_dump(cfg))
    (root / "dummy.ply").write_bytes(b"artificial scene placeholder, NOT used for rendering")
    return FloorMap.load(root / "map.yaml")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.map = create_map(self.root)
        self.cfg = config_from_args(make_args())

    def tearDown(self):
        self.tmp.cleanup()

    def pin(self, cell, allow=False):
        x,y = self.map.cell_to_world(cell)
        return dict(x=float(x), y=float(y), allow_unknown=allow)

    def plan(self, pins=None):
        pins = pins or []
        stations = select_stations(self.map, pins, [], 40, .10, .50)
        return create_plan(self.root/"map.yaml", self.root/"dummy.ply", stations,pins,[],self.cfg)

    def test_map_roundtrip(self):
        self.assertEqual(self.map.occupancy[30,10], -1)
        self.assertEqual(self.map.occupancy[0,0], 100)
        self.assertEqual(self.map.occupancy[5,5], 0)

    def test_auto_only_known_free(self):
        st = select_stations(self.map, [], [], 40)
        self.assertEqual(len(st),40)
        self.assertTrue(all(s['map_state']==0 for s in st))
        self.assertEqual(len({(s['row'],s['col']) for s in st}),40)

    def test_auto_spacing(self):
        st=select_stations(self.map,[],[],40,min_spacing=.6)
        xy=np.array([[s['x'],s['y']] for s in st]); d=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
        np.fill_diagonal(d,np.inf)
        self.assertGreaterEqual(d.min(),.6-1e-8)

    def test_deterministic_sampling(self):
        self.assertEqual(select_stations(self.map,[],[],40), select_stations(self.map,[],[],40))

    def test_unknown_without_approval_rejected(self):
        with self.assertRaises(ValueError): select_stations(self.map,[self.pin((30,10))],[],40)

    def test_unknown_with_approval_retained(self):
        before=self.map.occupancy.copy()
        st=select_stations(self.map,[self.pin((30,10),True)],[],40)
        self.assertTrue(st[0]['unknown_capture_only'])
        np.testing.assert_array_equal(self.map.occupancy,before)

    def test_occupied_pin_rejected(self):
        with self.assertRaises(ValueError): select_stations(self.map,[self.pin((0,0),True)],[],40)

    def test_duplicate_pin_rejected(self):
        p=self.pin((5,5))
        with self.assertRaises(ValueError): select_stations(self.map,[p,p],[],40)

    def test_exclusion(self):
        st=select_stations(self.map,[],[dict(x=0,y=0,radius=1.2)],40)
        self.assertTrue(all(math.hypot(p['x'],p['y'])>1.2 for p in st))

    def test_exclusion_pin_conflict(self):
        p=self.pin((5,5))
        with self.assertRaises(ValueError): select_stations(self.map,[p],[dict(x=p['x'],y=p['y'],radius=.4)],40)

    def test_not_enough_space_no_fake_positions(self):
        st=select_stations(self.map,[],[],40,min_spacing=100.)
        self.assertEqual(len(st),1)

    def test_count_and_pilot(self):
        pins=[self.pin((31,c),True) for c in (5,15,25,35,44)]
        p=self.plan(pins)
        self.assertEqual(len(p['frames']),480)
        self.assertEqual(sum(v['pilot'] for v in p['frames']),36)
        pilot_ids=p['pilot_station_ids']
        self.assertEqual(sum(s['selection']=='manual_pin' for s in p['stations'] if s['station_id'] in pilot_ids),3)
        self.assertEqual(len({v['frame_id'] for v in p['frames']}),480)

    def test_save_plan_no_source_mutation(self):
        old=file_sha256(self.root/'map.pgm')
        p=self.plan(); save_plan(self.root/'dataset',p)
        self.assertEqual(file_sha256(self.root/'map.pgm'),old)
        self.assertEqual(load_plan(self.root/'dataset')['plan_hash'],p['plan_hash'])

    def test_immutable_plan(self):
        p=self.plan(); save_plan(self.root/'dataset',p)
        with self.assertRaises(FileExistsError): save_plan(self.root/'dataset',p)

    def test_tampering_detected(self):
        p=self.plan(); save_plan(self.root/'dataset',p)
        p['frames'][0]['x']+=1; write_json(self.root/'dataset'/'plan.json',p)
        with self.assertRaises(ValueError): load_plan(self.root/'dataset')

    def test_camera_rigid_and_scale(self):
        p=self.plan()
        for view in p['frames'][::43]:
            c=camera_record(p,view)
            cg=np.array(c['camera_to_world']); gc=np.array(c['world_to_camera']); cm=np.array(c['camera_to_map'])
            np.testing.assert_allclose(gc@cg,np.eye(4),atol=2e-6)
            self.assertAlmostEqual(np.linalg.det(cg[:3,:3]),1.,places=5)
            self.assertAlmostEqual(np.linalg.det(cm[:3,:3]),1.,places=5)
            np.testing.assert_allclose(cm[:3,3],[view['x'],view['y'],1.7],atol=2e-6)

    def test_backprojection_both_frames(self):
        p=self.plan(); c=camera_record(p,p['frames'][11])
        K=np.array(c['K']); cg=np.array(c['camera_to_world']); cm=np.array(c['camera_to_map'])
        uv=np.array([12.5,21.5,1.]); depth=2.
        xc=np.linalg.inv(K)@uv*depth
        xg=cg[:3,:3]@xc+cg[:3,3]
        xm1=1.43*ROTATION@xg+np.array([0,0,1.43*1.1])
        xm2=cm[:3,:3]@(xc*1.43)+cm[:3,3]
        np.testing.assert_allclose(xm1,xm2,atol=2e-6)

    def test_pixel_world_roundtrip(self):
        for cell in [(3,4),(30,20),(5,45)]:
            xy=self.map.cell_to_world(cell)
            pixel=world_to_pixel(self.map,xy,(1000,800))
            np.testing.assert_allclose(pixel_to_world(self.map,pixel,(1000,800)),xy,atol=1e-10)

    def frame(self):
        p=self.plan(); save_plan(self.root/'dataset',p)
        v=p['frames'][0]; folder=self.root/'dataset'/'frames'/v['frame_id']
        rgb=np.full((48,64,3),127,np.uint8); d=np.full((48,64),2.,np.float32); a=np.ones((48,64),np.float32)
        d[0,0]=np.nan; a[1,1]=.1
        save_frame(folder,p,v,rgb,d,a,.02)
        return p,v,folder

    def test_save_raw_depth_and_mask(self):
        p,v,f=self.frame()
        d=np.load(f/'depth_scene.npy'); a=np.load(f/'alpha.npy'); m=np.array(Image.open(f/'valid.png'))
        self.assertTrue(np.isnan(d[0,0])); self.assertEqual(d[1,1],2.)
        self.assertEqual(m[1,1],0); self.assertEqual(m[2,2],255)
        self.assertTrue(frame_complete(f,p['plan_hash']))

    def test_missing_frame_detected(self):
        p,v,f=self.frame(); (f/'alpha.npy').unlink()
        self.assertFalse(frame_complete(f,p['plan_hash']))

    def test_corruption_same_length_detected(self):
        p,v,f=self.frame(); file=f/'alpha.npy'; blob=bytearray(file.read_bytes()); blob[-1]^=1; file.write_bytes(blob)
        self.assertFalse(frame_complete(f,p['plan_hash']))

    def test_quality_all_invalid(self):
        rgb=np.zeros((3,4,3),np.uint8); d=np.full((3,4),np.nan); a=np.full((3,4),np.nan)
        q,m=quality_report(rgb,d,a,.2)
        json.dumps(q,allow_nan=False)
        self.assertEqual(q['valid_fraction'],0); self.assertIsNone(q['alpha_mean'])

    def test_report(self):
        p,v,f=self.frame(); report(self.root/'dataset')
        s=read_json(self.root/'dataset'/'summary.json')
        self.assertEqual(s['complete'],1)
        self.assertTrue((self.root/'dataset'/'qa'/'contact_rgb_001.jpg').exists())
        self.assertTrue((self.root/'dataset'/'qa'/'contact_depth_001.jpg').exists())
        lines=(self.root/'dataset'/'index.jsonl').read_text().splitlines()
        self.assertEqual(len(lines),480)

    def test_invalid_configs(self):
        for changes in [dict(scale=0),dict(yaw_step=77),dict(pitches=[90]),dict(pitches=[0,0]),dict(alpha_min=2),dict(width=0)]:
            with self.assertRaises(ValueError): config_from_args(make_args(**changes))

if __name__=='__main__': unittest.main()
