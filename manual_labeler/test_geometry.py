import unittest
from pathlib import Path
import tempfile
import numpy as np
from avatar_demo.core import camera_matrices
from manual_labeler.geometry import *
from manual_labeler.storage import AnnotationStore


def frame():
    w,h=160,120;s=1.43;f=1.1
    K,C,W=camera_matrices(.5,-1,.4,1.7,-8,s,f,w,h,70)
    d=np.full((h,w),4/s,np.float32);d[25:95,50:115]=2/s
    rgb=np.zeros((h,w,3),np.uint8);rgb[:]=[170,160,150];rgb[25:95,50:115]=[180,40,30]
    cam={'K':K.tolist(),'camera_to_world':C.tolist(),'world_to_camera':W.tolist(),
         'scale':s,'floor_y':f,'width':w,'height':h,'map_pose':{'x':.5,'y':-1,'yaw_rad':.4,'camera_height_m':1.7,'pitch_deg':-8}}
    return {'id':'test','rgb':rgb,'depth_scene':d,'depth_m':d*s,'alpha':np.ones_like(d),'camera':cam}

class Tests(unittest.TestCase):
    def test_roundtrip(self):
        p=np.array([[1,2,3],[-2,5,.4]])
        self.assertTrue(np.allclose(map_to_scene(scene_to_map(p,1.43,1.1),1.43,1.1),p))
    def test_reprojection(self):
        f=frame();uv=np.array([[70,30],[110,90]])
        points=backproject(f['depth_scene'],uv,f['camera']);p,ok=project(points,f['camera'])
        self.assertTrue(ok.all());self.assertTrue(np.allclose(p,uv+.5,atol=1e-4))
    def test_metric_rigid_camera(self):
        f=frame();M=camera_map(f['camera']);self.assertTrue(np.allclose(M[:3,:3].T@M[:3,:3],np.eye(3),atol=1e-6))
        uv=np.array([[55,40]])
        q=np.array([55.5,40.5,1.])@np.linalg.inv(f['camera']['K']).T * f['depth_m'][40,55]
        self.assertTrue(np.allclose(q@M[:3,:3].T+M[:3,3],backproject(f['depth_scene'],uv,f['camera'])[0]))
    def test_seed(self):
        f=frame();shape={'kind':'rectangle','xyxy':[20,10,140,110]}
        d=build_draft(f,shape,{'seed_enabled':True,'seed':[80,60],'depth_tolerance':.2,'connected':True,'sor':False})
        self.assertEqual(d['stats']['retained_points'],65*70)
        self.assertTrue(np.all(d['uv'][:,0]>=50));self.assertTrue(np.all(d['uv'][:,0]<115))
    def test_invalid_seed(self):
        with self.assertRaises(ValueError):
            build_draft(frame(),{'kind':'rectangle','xyxy':[50,25,115,95]},{'seed_enabled':True,'seed':[10,10]})
    def test_polygon_and_aim(self):
        f=frame();d=build_draft(f,{'kind':'polygon','points':[[51,26],[113,26],[113,93],[51,93]]},{'aim':[80,60],'sor':False})
        self.assertEqual(d['aim_source'],'human_clicked_depth_pixel')
        self.assertTrue(np.allclose(d['target'],backproject(f['depth_scene'],[[80,60]],f['camera'])[0]))
    def test_box(self):
        box=checked_box({'center':[1,2,3],'extent':[2,4,1],'yaw_deg':30})
        self.assertTrue(inside_box(box_corners(box),box).all())
        self.assertFalse(inside_box(np.array([[100,2,3]]),box)[0])
    def test_save_append(self):
        f=frame();d=build_draft(f,{'kind':'rectangle','xyxy':[50,25,115,95]},{'sor':False})
        d['id']='d1'
        with tempfile.TemporaryDirectory() as tmp:
            identity={'scale':1.43,'floor_y':1.1,'scene_sha256':'a','map_yaml_sha256':'b','map_image_sha256':'c'}
            store=AnnotationStore(tmp,identity);a=store.save(f,d,'Red tank','tank','front only')
            self.assertEqual(a['object_id'],'manual_0001');self.assertTrue((Path(tmp)/a['cloud_path']).exists())
            b=store.save(f,d,'Red tank','tank','second view',a['object_id'])
            self.assertEqual(len(b['observations']),2);self.assertEqual(b['revision'],2)
            self.assertTrue((Path(tmp)/'objects/manual_0001/revisions/rev0001/cloud_map.ply').exists())
            again=AnnotationStore(tmp,identity);self.assertEqual(len(again.objects),1)
            with self.assertRaises(ValueError):AnnotationStore(tmp,{**identity,'scale':2})
if __name__=='__main__':unittest.main()
