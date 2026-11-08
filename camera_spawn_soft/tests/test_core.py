import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

from camera_spawn_soft.core import (FloorMap, empty_region, polygon_mask, prepare_layers,
    nudge_camera, generate, save_results, import_selection, check_identity)


def write_map(root, grid, res=.1, yaw=0.):
    img=np.full(grid.shape,205,np.uint8);img[grid==0]=254;img[grid==100]=0
    Image.fromarray(np.flipud(img)).save(root/'map.pgm')
    cfg={'image':'map.pgm','mode':'trinary','resolution':res,'origin':[-1.,-2.,yaw],
         'free_thresh':.196,'occupied_thresh':.65,'negate':0}
    (root/'map.yaml').write_text(yaml.safe_dump(cfg))
    return FloorMap.load(root/'map.yaml')


def region_all(m):
    h,w=m.shape
    # Large enough to include edge cell centers, with origin rotation support.
    cc=np.array([[0.,0.],[w,0.],[w,h],[0.,h]])*m.resolution
    return dict(empty_region(),polygon=(cc@m.rotation.T+m.origin[:2]).tolist())


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_map_flip(self):
        g=np.zeros((5,7),np.int8);g[0,0]=100;g[3,2]=-1
        m=write_map(self.p,g)
        np.testing.assert_array_equal(g,m.grid)

    def test_rotation_roundtrip(self):
        m=write_map(self.p,np.zeros((7,9),np.int8),yaw=.6)
        for cell in [(0,0),(6,8),(3,5)]:self.assertEqual(m.world_to_cell(m.cells_to_world(cell)),cell)
        self.assertTrue(polygon_mask(m,region_all(m)['polygon']).all())

    def test_dark_blocked_gray_allowed(self):
        g=np.zeros((10,10),np.int8);g[3,4]=100;g[4,3]=-1
        m=write_map(self.p,g);ls=prepare_layers(m,region_all(m))
        self.assertFalse(ls['allowed'][3,4]);self.assertTrue(ls['allowed'][4,3])

    def test_polygon_excludes_exterior(self):
        m=write_map(self.p,np.full((20,20),-1,np.int8))
        reg=region_all(m)
        reg['polygon']=m.cells_to_world(np.array([[3,3],[3,15],[15,15],[15,3]])).tolist()
        ls=prepare_layers(m,reg)
        self.assertFalse(ls['allowed'][0,0]);self.assertTrue(ls['allowed'][10,10])

    def test_small_corridor_kept(self):
        g=np.full((9,25),100,np.int8);g[4,2:23]=-1
        m=write_map(self.p,g);reg=region_all(m)
        s,r,l=generate(m,reg,count=15,preferred=.20,max_shift=.30)
        self.assertEqual(len(s),15)
        self.assertEqual(r['below_preferred_clearance'],15)
        self.assertTrue(all(p['row']==4 for p in s))
        self.assertTrue(all(p['unknown_capture_only'] for p in s))

    def test_move_away_from_wall(self):
        g=np.zeros((20,20),np.int8);g[:,0:4]=100
        m=write_map(self.p,g);ls=prepare_layers(m,region_all(m))
        cell,d=nudge_camera((10,4),ls['allowed'],ls['clearance_m'],.1,.18,.30)
        self.assertGreater(cell[1],4);self.assertGreaterEqual(ls['clearance_m'][cell],.18)
        self.assertLessEqual(d,.30+1e-9)

    def test_valid_clearance_stays(self):
        m=write_map(self.p,np.zeros((20,20),np.int8));ls=prepare_layers(m,region_all(m))
        cell,d=nudge_camera((10,10),ls['allowed'],ls['clearance_m'],.1,.18,.30)
        self.assertEqual(cell,(10,10));self.assertEqual(d,0)

    def test_not_across_wall(self):
        g=np.full((20,20),100,np.int8);g[2:18,2]=-1;g[2:18,4:18]=0
        m=write_map(self.p,g);ls=prepare_layers(m,region_all(m))
        cell,d=nudge_camera((10,2),ls['allowed'],ls['clearance_m'],.1,.18,.80)
        self.assertEqual(cell[1],2)

    def test_no_corner_cut(self):
        a=np.zeros((6,6),bool);a[1,1]=True;a[2,2]=True
        cl=np.zeros((6,6));cl[1,1]=.01;cl[2,2]=1.
        cell,_=nudge_camera((1,1),a,cl,.1,.18,.5)
        self.assertEqual(cell,(1,1))

    def test_zero_shift(self):
        m=write_map(self.p,np.zeros((20,20),np.int8));l=prepare_layers(m,region_all(m))
        cell,d=nudge_camera((0,0),l['allowed'],l['clearance_m'],.1,.18,0)
        self.assertEqual(cell,(0,0));self.assertEqual(d,0)

    def test_counts_100_200_and_unique(self):
        m=write_map(self.p,np.zeros((40,50),np.int8));reg=region_all(m)
        for n in [100,200]:
            ss,rr,_=generate(m,reg,count=n)
            self.assertEqual(len(ss),n);self.assertEqual(len({(s['row'],s['col']) for s in ss}),n)

    def test_reproducible_seed(self):
        m=write_map(self.p,np.zeros((20,20),np.int8));reg=region_all(m)
        s,_,_=generate(m,reg,count=30,seed=7)
        q,_,_=generate(m,reg,count=30,seed=7)
        self.assertEqual(s,q)

    def test_source_map_unchanged(self):
        g=np.zeros((20,20),np.int8);g[4,4]=100
        m=write_map(self.p,g);before=hashlib.sha256((self.p/'map.pgm').read_bytes()).hexdigest()
        ss,rr,ll=generate(m,region_all(m),count=10)
        save_results(self.p/'out',m,region_all(m),ss,rr,ll)
        self.assertEqual(hashlib.sha256((self.p/'map.pgm').read_bytes()).hexdigest(),before)
        np.testing.assert_array_equal(m.grid,g)

    def test_saved_selection_compatibility(self):
        m=write_map(self.p,np.zeros((20,20),np.int8));reg=region_all(m)
        ss,rr,ll=generate(m,reg,count=15)
        save_results(self.p/'out',m,reg,ss,rr,ll)
        sel=import_selection(self.p/'out/selection.json',m)
        self.assertEqual(len(sel['pins']),15)
        self.assertEqual(sel['corridors'],[])
        obj=json.loads((self.p/'out/camera_positions.json').read_text())
        self.assertEqual(len(obj['stations']),15)
        with self.assertRaises(FileExistsError):save_results(self.p/'out',m,reg,ss,rr,ll)

    def test_gray_corridor_priority(self):
        g=np.full((30,40),100,np.int8)
        g[3:20,3:37]=0;g[23,3:37]=-1
        m=write_map(self.p,g);reg=region_all(m)
        reg['corridors']=[{'points':m.cells_to_world(np.array([[23,4],[23,34]])).tolist(),'allow_unknown':True}]
        s,r,l=generate(m,reg,count=80,corridor_spacing=.3)
        corridor=[p for p in s if p['selection']=='corridor']
        self.assertGreaterEqual(len(corridor),10)
        self.assertTrue(all(p['row']==23 for p in corridor))
        self.assertTrue(all(p['below_preferred_clearance'] for p in corridor))

    def test_bad_pin_skipped_not_moved_across_wall(self):
        g=np.zeros((20,20),np.int8);g[5,5]=100
        m=write_map(self.p,g);reg=region_all(m)
        xy=m.cells_to_world([5,5]);reg['pins']=[{'x':float(xy[0]),'y':float(xy[1])}]
        s,r,l=generate(m,reg,count=10)
        self.assertEqual(len(s),10)
        self.assertEqual(r['skipped_explicit_or_duplicate_seeds'][0]['reason'],'occupied')

    def test_capacity_report(self):
        g=np.full((10,10),100,np.int8);g[3:5,3:5]=0
        m=write_map(self.p,g);s,r,l=generate(m,region_all(m),count=10)
        self.assertEqual(len(s),4);self.assertEqual(r['requested_positions'],10)

    def test_explicit_budget_error(self):
        m=write_map(self.p,np.zeros((20,20),np.int8));reg=region_all(m)
        reg['corridors']=[{'points':m.cells_to_world(np.array([[10,2],[10,17]])).tolist()}]
        with self.assertRaises(ValueError):generate(m,reg,count=3,corridor_spacing=.1)

    def test_exclusion(self):
        m=write_map(self.p,np.zeros((20,20),np.int8));reg=region_all(m)
        xy=m.cells_to_world([10,10]);reg['excluded_capture_disks']=[{'x':xy[0],'y':xy[1],'radius':.3}]
        ss,_,_=generate(m,reg,count=100)
        for s in ss:self.assertGreater(np.linalg.norm(np.array([s['x'],s['y']])-xy),.3)

    def test_invalid_settings(self):
        m=write_map(self.p,np.zeros((20,20),np.int8));reg=region_all(m)
        for kw in [{'preferred':-1},{'max_shift':-1},{'count':0},{'gray_fraction':2},{'seed':-1}]:
            with self.assertRaises(ValueError):generate(m,reg,**kw)

    def test_height_transform(self):
        m=write_map(self.p,np.zeros((20,20),np.int8));ss,_,_=generate(m,region_all(m),count=10)
        for s in ss:
            gx,gy,gz=s['preview_gaussian_xyz']
            self.assertAlmostEqual(1.43*gx,s['x']);self.assertAlmostEqual(1.43*gz,s['y'])
            self.assertAlmostEqual(1.43*(1.1-gy),1.7)

    def test_wrong_map_identity_rejected(self):
        m=write_map(self.p,np.zeros((10,10),np.int8))
        with self.assertRaises(ValueError):check_identity({'map_identity':{'yaml_sha256':'wrong'}},m)


if __name__=='__main__':unittest.main()
