from __future__ import annotations
from dataclasses import replace,asdict
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
import numpy as np
from PIL import Image
from segment_dataset.quality import (Policy,frame_metrics,depth_support,image_features,
        mask_metrics,decide_mask,surface_summary,validate_camera)
from segment_dataset.storage import (digest,sha,write_json,load_plan,read_source,
        pack_masks,unpack_masks,mark_complete,archive_valid,read_json)
from segment_dataset.pipeline import setup,audit,choose_pilot,segment,reselect,bind_run
from segment_dataset.report import build_report


def camera(h=48,w=64):
    # level camera +Z forward, standing at map height 1.7
    R=np.array([[0,0,1],[-1,0,0],[0,-1,0]],float)
    T=np.eye(4); T[:3,:3]=R;T[:3,3]=[0,0,1.7]
    return {'width':w,'height':h,'K':[[50,0,w/2],[0,50,h/2],[0,0,1]],
        'camera_to_map':T.tolist(),'depth_type':'expected_camera_z',
        'pixel_sampling':{'u':'column + 0.5','v':'row + 0.5'},'depth_scale_scene_to_metres':1.43}


def arrays(h=48,w=64):
    rgb=np.random.default_rng(1).integers(10,245,(h,w,3),dtype=np.uint8)
    d=np.ones((h,w),np.float32)*3
    return rgb,d,np.ones((h,w),np.float32)


def make_dataset(root,count=8):
    root=Path(root);root.mkdir()
    plan={'schema':'valmet.calibrated_views.v1','config':{'scale':1.43},'frames':[]}
    for i in range(count):
        plan['frames'].append({'frame_id':f's{i:03d}_v000','station_id':f's{i:03d}',
            'x':i*.5,'y':0.,'yaw_deg':30*i,'pitch_deg':-10.,'camera_height_m':.5+(i%5)*.45,
            'pilot':i%2==0,'unknown_capture_only':i%3==0})
    plan['plan_hash']=digest(plan);write_json(root/'plan.json',plan)
    for i,v in enumerate(plan['frames']):
        folder=root/'frames'/v['frame_id'];folder.mkdir(parents=True)
        rgb,d,a=arrays()
        if i==count-1: d[:]=.05 # clearly too near, source still valid
        Image.fromarray(rgb).save(folder/'rgb.png')
        np.save(folder/'depth_m.npy',d);np.save(folder/'depth_scene.npy',d/1.43);np.save(folder/'alpha.npy',a)
        Image.fromarray(np.ones_like(d,dtype=np.uint8)*255).save(folder/'valid.png')
        c={**camera(),'frame_id':v['frame_id'],'plan_hash':plan['plan_hash']}
        write_json(folder/'camera.json',c)
        write_json(folder/'qa.json',{'flags':['many_surfaces_within_0.2m_check_camera'] if i==count-1 else []})
        files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.iterdir()}
        write_json(folder/'complete.json',{'frame_id':v['frame_id'],'plan_hash':plan['plan_hash'],'files':files})
    return plan


class QualityTests(unittest.TestCase):
    def setUp(self): self.p=Policy(); self.rgb,self.d,self.a=arrays()
    def test_good_frame(self): self.assertEqual(frame_metrics(self.rgb,self.d,self.a,self.p)['status'],'keep')
    def test_near_reject(self):
        self.d[:]=.05; r=frame_metrics(self.rgb,self.d,self.a,self.p)
        self.assertEqual(r['status'],'exclude');self.assertIn('view_mostly_inside_or_very_close_to_surface',r['reasons'])
    def test_all_invalid(self):
        self.d[:]=np.nan; self.a[:]=0
        self.assertEqual(frame_metrics(self.rgb,self.d,self.a,self.p)['status'],'exclude')
    def test_low_alpha(self):
        self.a[:]=.1;self.assertEqual(frame_metrics(self.rgb,self.d,self.a,self.p)['usable_fraction'],0)
    def test_partial_near_keeps_other_masks(self):
        self.d[:,:20]=.05
        r=frame_metrics(self.rgb,self.d,self.a,self.p); self.assertEqual(r['status'],'review')
        m=np.zeros_like(self.d,bool);m[5:43,30:60]=True
        f,v,e=depth_support(self.d,self.a,self.p)
        met=mask_metrics(m,.9,[30,5,60,43],image_features(self.rgb),self.d,f,v,e,camera(),self.p)
        dec=decide_mask(met,r,self.p);self.assertEqual(dec['status'],'keep')
    def test_blank(self):
        self.rgb[:]=120;self.assertEqual(frame_metrics(self.rgb,self.d,self.a,self.p)['status'],'exclude')
    def test_smooth_low_detail_review_not_hard_reject(self):
        self.rgb[:]=np.linspace(30,180,64,dtype=np.uint8)[None,:,None]
        self.assertEqual(frame_metrics(self.rgb,self.d,self.a,self.p)['status'],'review')
    def test_shape_error(self):
        with self.assertRaises(ValueError): frame_metrics(self.rgb[:10],self.d,self.a,self.p)
    def test_depth_jump(self):
        self.d[:,32:]=8; _,v,e=depth_support(self.d,self.a,self.p)
        self.assertTrue(e[:,31:33].all());self.assertFalse(e[:,:30].any())
    def test_missing_depth_not_infinite_jump(self):
        self.d[:,32:]=np.nan;_,v,e=depth_support(self.d,self.a,self.p);self.assertFalse(e.any())
    def test_planar_summary_center_and_height(self):
        m=np.ones_like(self.d,bool);r=surface_summary(m,self.d,camera(),self.p)
        np.testing.assert_allclose(r['median_map_xyz_m'],[3,0,1.7],atol=.04)
    def test_pixel_centers(self):
        p=replace(self.p,min_geometry_pixels=1,max_geometry_samples=16)
        m=np.zeros_like(self.d,bool);m[24,32]=True
        r=surface_summary(m,self.d,camera(),p)
        np.testing.assert_allclose(r['median_map_xyz_m'],[3,-.03,1.67],atol=1e-6)
    def test_no_scale_in_rotation(self):
        c=camera();T=np.array(c['camera_to_map']);T[:3,:3]*=1.43;c['camera_to_map']=T.tolist()
        with self.assertRaises(ValueError): validate_camera(c,self.d.shape)
    def test_wrong_camera_resolution(self):
        with self.assertRaises(ValueError): validate_camera(camera(),(49,64))
    def test_wrong_depth_convention(self):
        c=camera();c['depth_type']='ray_length'
        with self.assertRaises(ValueError): validate_camera(c,self.d.shape)
    def test_wrong_pixel_convention(self):
        c=camera();c['pixel_sampling']={'u':'column','v':'row'}
        with self.assertRaises(ValueError): validate_camera(c,self.d.shape)
    def test_valid_camera(self): validate_camera(camera(),self.d.shape)
    def test_invalid_policy(self):
        for p in [replace(self.p,alpha_min=2),replace(self.p,min_depth_m=50),replace(self.p,min_mask_area=1.2)]:
            with self.assertRaises(ValueError): p.validate()
    def test_tiny_mask_saved_but_excluded(self):
        mask=np.zeros_like(self.d,bool);mask[3,3]=1
        f,v,e=depth_support(self.d,self.a,self.p)
        m=mask_metrics(mask,.9,[3,3,4,4],image_features(self.rgb),self.d,f,v,e,camera(),self.p)
        self.assertEqual(decide_mask(m,{'status':'keep'},self.p)['status'],'exclude')
    def test_empty_mask(self):
        mask=np.zeros_like(self.d,bool);f,v,e=depth_support(self.d,self.a,self.p)
        m=mask_metrics(mask,.7,[0,0,0,0],image_features(self.rgb),self.d,f,v,e,camera(),self.p)
        self.assertEqual(decide_mask(m,{'status':'keep'},self.p)['status'],'exclude')
    def test_thin_mask_kept_raw_even_if_no_geometry(self):
        mask=np.zeros_like(self.d,bool);mask[:,20]=1
        f,v,e=depth_support(self.d,self.a,self.p)
        m=mask_metrics(mask,.9,[20,0,21,48],image_features(self.rgb),self.d,f,v,e,camera(),self.p)
        self.assertEqual(m['area_px'],48);self.assertFalse(decide_mask(m,{'status':'keep'},self.p)['eligible_geometry'])


class StorageTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_overlapping_masks_roundtrip(self):
        m=np.zeros((3,11,13),bool);m[0,1:8,2:9]=True;m[1,4:10,5:12]=True;m[2]=m[0]
        p=self.root/'m.npz';pack_masks(p,m,np.ones(3),np.zeros((3,4)))
        x,_,_=unpack_masks(p);np.testing.assert_array_equal(x,m);self.assertTrue((x[0]&x[1]).any())
    def test_zero_proposals_roundtrip(self):
        m=np.empty((0,11,13),bool);p=self.root/'m.npz';pack_masks(p,m,np.empty(0),np.empty((0,4)))
        self.assertEqual(unpack_masks(p)[0].shape,(0,11,13))
    def test_bad_score_count(self):
        with self.assertRaises(ValueError): pack_masks(self.root/'m.npz',np.zeros((2,3,4)),np.ones(1),np.zeros((2,4)))
    def test_source_checksum(self):
        p=make_dataset(self.root/'data',2);v=p['frames'][0];f=self.root/'data'/'frames'/v['frame_id']
        read_source(self.root/'data',v,p)
        (f/'alpha.npy').write_bytes(b'broken')
        with self.assertRaises(ValueError):read_source(self.root/'data',v,p)
    def test_source_preservation(self):
        p=make_dataset(self.root/'data',2); before={str(f):sha(f) for f in (self.root/'data').rglob('*') if f.is_file()}
        ds,out,plan,policy,ph=setup(self.root/'data',self.root/'out')
        with redirect_stdout(io.StringIO()):audit(ds,out,plan,policy,ph)
        after={str(f):sha(f) for f in ds.rglob('*') if f.is_file()};self.assertEqual(before,after)
    def test_tampered_plan(self):
        p=make_dataset(self.root/'data',2);p['frames'][0]['x']=999;write_json(self.root/'data'/'plan.json',p)
        with self.assertRaises(ValueError):load_plan(self.root/'data')
    def test_nested_output_rejected(self):
        make_dataset(self.root/'data',2)
        with self.assertRaises(ValueError):setup(self.root/'data',self.root/'data'/'out')
    def test_atomic_completion_and_corruption(self):
        d=self.root/'frame';d.mkdir();pack_masks(d/'masks.npz',np.zeros((1,3,4)),np.ones(1),np.zeros((1,4)))
        write_json(d/'inference.json',{'n':1});mark_complete(d,'run','source')
        self.assertTrue(archive_valid(d,'run','source'));self.assertFalse(archive_valid(d,'other','source'))
        (d/'masks.npz').write_bytes(b'corrupted');self.assertFalse(archive_valid(d,'run','source'))
    def test_wrong_frame_id_rejected(self):
        p=make_dataset(self.root/'data',2);v=p['frames'][0];cam=read_json(self.root/'data'/'frames'/v['frame_id']/'camera.json')
        with self.assertRaises(ValueError):validate_camera(cam,(48,64),'other',p['plan_hash'])


class FakePredictor:
    calls=0
    def __init__(self,args):
        self.binding={'fake_test_predictor':True,'conf':args.conf}
    def __call__(self,rgb):
        type(self).calls+=1
        h,w=rgb.shape[:2];m=np.zeros((3,h,w),bool);m[0,4:h-4,4:w//2+8]=1;m[1,10:h-2,w//2-8:w-4]=1;m[2,1,1]=1
        return m,np.array([.9,.8,.3]),np.array([[4,4,w//2+8,h-4],[w//2-8,10,w-4,h-2],[1,1,2,2]])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);make_dataset(self.root/'data',8)
        self.ds,self.out,self.plan,self.p,self.ph=setup(self.root/'data',self.root/'out')
        with redirect_stdout(io.StringIO()):self.records=audit(self.ds,self.out,self.plan,self.p,self.ph)
        self.args=SimpleNamespace(pilot=True,pilot_frames=5,segment_excluded=False,limit=None,max_det=300,conf=.25)
        FakePredictor.calls=0
    def tearDown(self): self.tmp.cleanup()
    def do_segment(self):
        with redirect_stdout(io.StringIO()):return segment(self.ds,self.out,self.plan,self.records,self.p,self.ph,self.args,FakePredictor)
    def test_audit_counts(self):
        self.assertEqual(len(self.records),8);self.assertEqual(sum(r['status']=='exclude' for r in self.records),1)
    def test_pilot_includes_flagged_and_diverse(self):
        rec=choose_pilot(self.records,5);self.assertEqual(len(rec),5);self.assertEqual(len({r['frame_id'] for r in rec}),5)
        self.assertTrue(any(r['status']=='exclude' for r in rec));self.assertGreater(len({r['camera_height_m'] for r in rec}),2)
    def test_pilot_resume_and_full(self):
        self.do_segment();n=FakePredictor.calls;self.assertGreater(n,0)
        self.do_segment();self.assertEqual(FakePredictor.calls,n)
        self.args.pilot=False;self.do_segment();self.assertEqual(FakePredictor.calls,7)
    def test_limit_new_only(self):
        self.args.pilot=False;self.args.limit=2;self.do_segment();self.assertEqual(FakePredictor.calls,2)
        self.do_segment();self.assertEqual(FakePredictor.calls,4)
    def test_severe_skipped_but_override_archives(self):
        self.args.pilot=False;self.do_segment();self.assertEqual(FakePredictor.calls,7)
        self.args.segment_excluded=True;self.do_segment();self.assertEqual(FakePredictor.calls,8)
    def test_all_proposals_preserved(self):
        self.do_segment();frames=list((self.out/'frames').glob('*/masks.npz'));self.assertTrue(frames)
        self.assertTrue(all(len(unpack_masks(f)[0])==3 for f in frames))
    def test_reselect_no_model_no_raw_changes(self):
        self.do_segment();before={str(f):sha(f) for f in (self.out/'frames').rglob('*') if f.is_file()}
        p=replace(self.p,min_mask_area=10000);ph=digest(asdict(p))
        with redirect_stdout(io.StringIO()):
            rec=audit(self.ds,self.out,self.plan,p,ph);reselect(self.ds,self.out,self.plan,rec,p,ph)
            summary=build_report(self.ds,self.out,self.plan,rec,ph)
        after={str(f):sha(f) for f in (self.out/'frames').rglob('*') if f.is_file()}
        self.assertEqual(before,after);self.assertEqual(summary['counts'].get('mask_keep',0),0)
    def test_changed_inference_refused(self):
        self.do_segment();self.args.conf=.5
        with self.assertRaises(ValueError):self.do_segment()
    def test_report_and_readable_manifests(self):
        self.do_segment()
        with redirect_stdout(io.StringIO()): s=build_report(self.ds,self.out,self.plan,self.records,self.ph)
        self.assertGreater(s['counts']['all_returned_masks'],0)
        self.assertTrue((self.out/'qa'/'contact_masks_001.jpg').is_file())
        self.assertTrue((self.out/'qa'/'index.html').is_file())
        rows=[json.loads(x) for x in (self.out/'observations.jsonl').read_text().splitlines()]
        self.assertEqual(len({r['observation_id'] for r in rows}),len(rows))
        self.assertTrue(all(len(r['metrics']['bbox_xyxy'])==4 for r in rows))
    def test_corrupt_capture_excluded_not_crash(self):
        f=self.ds/'frames'/self.plan['frames'][0]['frame_id']/'rgb.png';f.write_bytes(b'not_image')
        with redirect_stdout(io.StringIO()):records=audit(self.ds,self.out,self.plan,self.p,self.ph)
        self.assertFalse(records[0]['source_ok']);self.assertEqual(records[0]['status'],'exclude')
    def test_mask_export(self):
        from segment_dataset.__main__ import main
        self.do_segment();f=next((self.out/'frames').glob('*/masks.npz'));dest=self.root/'export'
        with redirect_stdout(io.StringIO()):main(['export-mask','--out',str(self.out),'--frame',f.parent.name,'--mask','0','--dest',str(dest)])
        self.assertTrue((dest/'mask.png').exists());self.assertTrue((dest/'crop_rgba.png').exists())

if __name__=='__main__': unittest.main()
