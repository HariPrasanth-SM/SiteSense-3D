"""Focused update checks; no models, CUDA, or real Valmet geometry."""
import math,unittest
import numpy as np
from PIL import Image
from unittest.mock import patch
from service_demo.navigation import navigation_map,preferred_start,route_unknown_cells,target_orientation,same_pose
from service_demo.arrival import focused_image,mask_points,ArrivalVerifier
from avatar_demo.core import FloorMap,Planner,PlanningError,camera_matrices
from avatar_demo.manual import manual_step

class UpdateChecks(unittest.TestCase):
    def map(self):
        g=np.zeros((16,22),np.int8);g[[0,-1]]=100;g[:,[0,-1]]=100
        g[:,10]=100;g[6:10,10]=-1
        return FloorMap(g,.1,np.zeros(3))
    def test_gray_connects_without_source_change(self):
        src=self.map();old=src.occupancy.copy();nav=navigation_map(src);pl=Planner(nav,0,0)
        path=pl.plan(src.cell_to_world((7,4)),src.cell_to_world((7,16)),mode='point')
        self.assertGreater(route_unknown_cells(src,path.xy),0)
        np.testing.assert_array_equal(src.occupancy,old)
    def test_strict_optional(self):
        src=self.map();pl=Planner(navigation_map(src,False),0,0)
        with self.assertRaises(PlanningError):pl.plan(src.cell_to_world((7,4)),src.cell_to_world((7,16)),mode='point')
    def test_dark_stays_blocked(self):
        nav=navigation_map(self.map());pl=Planner(nav,0,0)
        self.assertFalse(pl.walkable[2,10]);self.assertTrue(pl.walkable[7,10])
    def test_exterior_not_passable(self):
        pl=Planner(navigation_map(self.map()),0,0)
        with self.assertRaises(PlanningError):pl.set_start([-2,-2])
    def test_start_stays_original_free(self):
        src=self.map();pl=Planner(navigation_map(src),0,0)
        self.assertEqual(src.occupancy[src.world_to_cell(preferred_start(src,pl))],0)
    def test_manual_gray_and_dark(self):
        src=self.map();nav=navigation_map(src);pl=Planner(nav,0,0)
        good=manual_step(src.cell_to_world((7,9)),0,1,0,.1,1.,1.,nav,pl.walkable,True,False)
        bad=manual_step(src.cell_to_world((2,9)),0,1,0,.1,1.,1.,nav,pl.walkable,True,False)
        self.assertFalse(good.blocked);self.assertTrue(bad.blocked)
    def test_target_head_pitch(self):
        yaw,pitch=target_orientation([1,1],[2,2,.7],1.7)
        self.assertAlmostEqual(yaw,math.pi/4);self.assertLess(pitch,0)
        self.assertAlmostEqual(pitch,math.degrees(math.atan2(-1,math.sqrt(2))))
    def test_stale_pose_and_wrap(self):
        self.assertTrue(same_pose([0,0,math.pi,1.7,0],[0,0,-math.pi,1.7,0]))
        self.assertFalse(same_pose([0,0,0,1.7,0],[.2,0,0,1.7,0]))
    def test_target_crop_is_tight(self):
        rgb=np.full((100,140,3),80,np.uint8);mask=np.zeros((100,140),bool);mask[30:50,50:70]=True
        out=focused_image(rgb,mask,tint=True)
        self.assertEqual(out.shape,(32,32,3));self.assertGreater(out[10,10,1],80)
        self.assertEqual(int(rgb[35,55,1]),80)
    def test_backprojection_uses_scale(self):
        K,T,_=camera_matrices(0,0,0,1.7,0,width=32,height_px=32)
        f={'depth_m':np.full((32,32),2.),'alpha':np.ones((32,32)),
           'camera':{'K':K,'camera_to_world':T,'scale':1.43,'floor_y':1.1}}
        m=np.zeros((32,32),bool);m[15:17,15:17]=True
        points=mask_points(f,m)
        np.testing.assert_allclose(points.mean(0),[2,0,1.7],atol=1e-6)
    def test_highlight_requires_reference(self):
        v=ArrivalVerifier({})
        result=v.verify({'rgb':np.zeros((4,4,3),np.uint8)},
                        {'references':np.zeros((1,4))},None)
        self.assertEqual(result['state'],'unavailable')
    def test_highlight_requires_geometry(self):
        v=ArrivalVerifier({})
        result=v.verify({'rgb':np.zeros((4,4,3),np.uint8)},
                        {'references':np.ones((1,4)),'support_xyz':[]},None)
        self.assertEqual(result['state'],'unavailable')

    def mock_verifier(self, two_masks=False, mismatch=False):
        from types import SimpleNamespace
        h=w=32;rgb=np.full((h,w,3),100,np.uint8)
        masks=np.zeros((2 if two_masks else 1,h,w),bool)
        masks[0,6:20,2:10]=True
        if two_masks:masks[1,6:20,22:30]=True
        K,T,_=camera_matrices(0,0,0,1.7,0,width=w,height_px=h)
        frame={'rgb':rgb,'depth_m':np.full((h,w),2.),'alpha':np.ones((h,w)),
               'camera':{'K':K,'camera_to_world':T,'scale':1.43,'floor_y':1.1}}
        class Tensor:
            def cpu(self):return self
            def numpy(self):return masks.astype(float)
        class Model:
            def __call__(self,*a,**kw):return [SimpleNamespace(masks=SimpleNamespace(data=Tensor()))]
        class Encoder:
            def infer(self,batch):return np.ones((len(batch),256,4),np.float32),None
        v=ArrivalVerifier({});v.model=Model()
        refs=np.array([[1,-1,1,-1] if mismatch else [1,1,1,1]],np.float32)
        job={'references':refs,'support_xyz':np.concatenate([mask_points(frame,m) for m in masks]),
             'feature_run':{'recipe':{'crop_size':256,'crop_padding_fraction':.08}}}
        return v.verify(frame,job,Encoder())
    def test_accepted_candidate_has_green_overlay(self):
        r=self.mock_verifier();self.assertEqual(r['state'],'matched')
        self.assertTrue((r['overlay'][r['mask'],1]>100).all())
        self.assertTrue(r['crop'].shape[0]<32)
    def test_weak_appearance_abstains(self):
        self.assertEqual(self.mock_verifier(mismatch=True)['state'],'not_found')
    def test_disjoint_similar_masks_abstain(self):
        self.assertEqual(self.mock_verifier(two_masks=True)['state'],'ambiguous')

if __name__=='__main__':unittest.main()
