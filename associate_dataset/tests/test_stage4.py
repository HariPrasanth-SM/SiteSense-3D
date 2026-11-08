from __future__ import annotations
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import io
import copy
import tempfile
import unittest
import numpy as np
from associate_dataset.common import read_json, write_json, sha, norm_rows, disjoint, export_ply
from associate_dataset.catalog import Inputs, Catalog, prepare
from associate_dataset.geometry import (MatchPolicy, Surface, GeometryStore, backproject, project,
                                        reprojection, neighborhood_overlap, decide_pair, check_pair)
from associate_dataset.pipeline import run_queries
from associate_dataset.report import make_report


def cam(h=60, w=80, trans=(0,0,0), rotation=None):
    T = np.eye(4); T[:3,3]=trans
    if rotation is not None:T[:3,:3]=rotation
    return {'height':h,'width':w,'K':[[80,0,w/2],[0,80,h/2],[0,0,1]],'camera_to_map':T.tolist()}


def surf(mask, camera=None, depth=2., index=0):
    camera=cam(*mask.shape) if camera is None else camera
    v,u=np.nonzero(mask);uv=np.column_stack([u,v]).astype(np.int32)
    xyz=backproject(uv,np.full(len(uv),depth),camera).astype(np.float32)
    return Surface(index,'f'+str(index),mask,xyz,np.full((len(uv),3),150,np.uint8),uv,camera,
                   np.array([1,0],np.float32),np.array([1,0],np.float32),np.array([1,0],np.float32),len(uv))


def target_frame(shape,z=2.):
    return {'depth':np.full(shape,z,np.float32),'valid':np.ones(shape,bool)}


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.m=np.zeros((60,80),bool);self.m[15:45,20:60]=True
        self.a=surf(self.m);self.p=MatchPolicy(mask_dilation_px=0)
    def test_pixel_center_roundtrip(self):
        uv,z,ok=project(self.a.xyz,self.a.camera)
        np.testing.assert_allclose(uv,self.a.uv+.5,atol=1e-5)
        self.assertTrue(ok.all());np.testing.assert_allclose(z,2)
    def test_camera_rotation_translation_roundtrip(self):
        R=np.array([[0,0,1],[-1,0,0],[0,-1,0]],float)
        c=cam(trans=(1,2,1.7),rotation=R);a=surf(self.m,c)
        uv,z,ok=project(a.xyz,c);np.testing.assert_allclose(uv,a.uv+.5,atol=1e-4)
        self.assertTrue(ok.all())
    def test_backproject_not_double_scale(self):
        c=cam();p=backproject([[40,30]],[2.86],c)
        self.assertAlmostEqual(p[0,2],2.86)
    def test_same_surface_full_support(self):
        r,d=reprojection(self.a,self.a,target_frame(self.m.shape),self.p)
        self.assertEqual(r['support_fraction'],1);self.assertEqual(r['consistent_inside_mask'],1200)
    def test_occlusion_not_support_or_conflict(self):
        r,d=reprojection(self.a,self.a,target_frame(self.m.shape,1),self.p)
        self.assertEqual(r['occluded_pixels'],1200);self.assertEqual(r['front_conflict_pixels'],0)
        self.assertEqual(r['testable_visible_pixels'],0);self.assertEqual(r['consistent_inside_mask'],0)
    def test_front_conflict(self):
        r,d=reprojection(self.a,self.a,target_frame(self.m.shape,3),self.p)
        self.assertEqual(r['front_conflict_pixels'],1200);self.assertEqual(r['consistent_inside_mask'],0)
    def test_matching_depth_outside_mask(self):
        other=np.zeros_like(self.m);other[1:10,1:10]=True;b=surf(other)
        r,_=reprojection(self.a,b,target_frame(self.m.shape),self.p)
        self.assertEqual(r['consistent_outside_mask'],1200);self.assertEqual(r['support_fraction'],0)
    def test_out_of_view_is_not_support(self):
        b=surf(self.m,cam(trans=(100,0,0)))
        r,_=reprojection(self.a,b,target_frame(self.m.shape),self.p)
        self.assertEqual(r['usable_unique_pixels'],0)
    def test_behind_camera_is_not_support(self):
        b=surf(self.m,cam(trans=(0,0,5)))
        r,_=reprojection(self.a,b,target_frame(self.m.shape),self.p)
        self.assertEqual(r['usable_unique_pixels'],0)
    def test_invalid_depth_not_counted(self):
        f=target_frame(self.m.shape);f['valid'][:]=False
        r,_=reprojection(self.a,self.a,f,self.p);self.assertEqual(r['usable_unique_pixels'],0)
    def test_pixel_deduplication(self):
        a=copy.deepcopy(self.a);a.xyz=np.repeat(a.xyz,4,axis=0)
        r,_=reprojection(a,self.a,target_frame(self.m.shape),self.p)
        self.assertEqual(r['consistent_inside_mask'],1200)
    def test_same_surface_nn(self):
        r=neighborhood_overlap(self.a.xyz,self.a.xyz,.01)
        self.assertEqual(r['a_near_b_fraction'],1);self.assertEqual(r['b_near_a_fraction'],1)
    def test_distant_surface_nn(self):
        r=neighborhood_overlap(self.a.xyz,self.a.xyz+10,.12)
        self.assertEqual(r['a_near_b_fraction'],0)
    def test_one_sided_partial_geometry(self):
        small=self.a.xyz[:100];r=neighborhood_overlap(self.a.xyz,small,.001)
        self.assertEqual(r['b_near_a_fraction'],1);self.assertLess(r['a_near_b_fraction'],.10)
    def test_real_translated_camera_same_surface_supported(self):
        # 0.2 m camera translation at z=2, fx=80 shifts object by exactly 8 pixels.
        mb=np.zeros_like(self.m);mb[15:45,12:52]=True
        b=surf(mb,cam(trans=(.2,0,0)),index=1)
        ab,_=reprojection(self.a,b,target_frame(self.m.shape),self.p)
        ba,_=reprojection(b,self.a,target_frame(self.m.shape),self.p)
        nn=neighborhood_overlap(self.a.xyz,b.xyz,self.p.surface_tol_m)
        self.assertEqual(decide_pair(.94,.90,nn,ab,ba,self.p)[0],'supported')
    def test_similar_appearance_distinct_spatial_instances_not_supported(self):
        b=surf(self.m,cam(trans=(3,0,0)),index=1)
        ab,_=reprojection(self.a,b,target_frame(self.m.shape),self.p)
        ba,_=reprojection(b,self.a,target_frame(self.m.shape),self.p)
        nn=neighborhood_overlap(self.a.xyz,b.xyz,self.p.surface_tol_m)
        self.assertNotEqual(decide_pair(.99,.99,nn,ab,ba,self.p)[0],'supported')
    def test_nested_region_remains_partial(self):
        tiny=np.zeros_like(self.m);tiny[23:37,32:48]=True
        b=surf(tiny,index=1)
        ab,_=reprojection(self.a,b,target_frame(self.m.shape),self.p)
        ba,_=reprojection(b,self.a,target_frame(self.m.shape),self.p)
        nn=neighborhood_overlap(self.a.xyz,b.xyz,self.p.surface_tol_m)
        self.assertEqual(decide_pair(.95,.9,nn,ab,ba,self.p)[0],'partial_overlap')
    def test_nonfinite_projection(self):
        uv,z,ok=project([[np.nan,0,0],[0,0,-1]],cam());self.assertFalse(ok.any())


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.p=MatchPolicy();self.nn={'a_near_b_fraction':.8,'b_near_a_fraction':.8}
        self.r={'consistent_inside_mask':100,'support_fraction':.9,'testable_fraction':.8,
                'front_conflict_fraction':0.,'front_conflict_pixels':0,'testable_visible_pixels':110}
    def test_supported(self):self.assertEqual(decide_pair(.9,.8,self.nn,self.r,self.r,self.p)[0],'supported')
    def test_low_appearance_no_supported(self):self.assertNotEqual(decide_pair(.5,.8,self.nn,self.r,self.r,self.p)[0],'supported')
    def test_context_conflict(self):self.assertNotEqual(decide_pair(.9,.1,self.nn,self.r,self.r,self.p)[0],'supported')
    def test_context_optional_not_invented(self):self.assertEqual(decide_pair(.9,None,self.nn,self.r,self.r,self.p)[0],'supported')
    def test_low_samples_inconclusive(self):
        r=dict(self.r,consistent_inside_mask=2,testable_visible_pixels=2,testable_fraction=.01)
        self.assertEqual(decide_pair(.9,.9,self.nn,r,r,self.p)[0],'inconclusive')
    def test_partial_one_sided(self):
        r=dict(self.r,consistent_inside_mask=10,support_fraction=.1)
        self.assertEqual(decide_pair(.9,.9,self.nn,self.r,r,self.p)[0],'partial_overlap')
    def test_depth_conflict_rejected(self):
        r=dict(self.r,front_conflict_fraction=.8,front_conflict_pixels=100,consistent_inside_mask=0)
        self.assertEqual(decide_pair(.9,.9,self.nn,r,r,self.p)[0],'rejected')
    def test_policy_invalid(self):
        for p in (replace(self.p,samples=0),replace(self.p,depth_abs_m=-1),replace(self.p,appearance_min=1.5),replace(self.p,mask_dilation_px=.5)):
            with self.assertRaises(ValueError):p.validate()
    def test_float16_renormalized(self):
        v=norm_rows(np.array([.23,.33],np.float16));self.assertAlmostEqual(np.linalg.norm(v),1,places=6)
    def test_zero_descriptor_refused(self):
        with self.assertRaises(ValueError):norm_rows([[0,0]])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        from segment_dataset.tests.test_stage2 import make_dataset,FakePredictor
        from segment_dataset.pipeline import setup,audit as audit2,segment
        from segment_dataset.report import build_report
        from feature_dataset.tests.test_features import FakeEncoder,args
        from feature_dataset.source import Source
        from feature_dataset.pipeline import extract
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        make_dataset(self.root/'capture',5)
        ds,seg,plan,pol,ph=setup(self.root/'capture',self.root/'seg')
        a=SimpleNamespace(pilot=False,pilot_frames=5,segment_excluded=False,limit=None,max_det=300,conf=.25)
        with redirect_stdout(io.StringIO()):
            rec=audit2(ds,seg,plan,pol,ph);segment(ds,seg,plan,rec,pol,ph,a,FakePredictor)
            build_report(ds,seg,plan,rec,ph)
            feat=disjoint(self.root/'features',ds,seg)
            extract(Source(ds,seg),feat,args(),FakeEncoder)
        self.ds=ds;self.seg=seg;self.feat=feat;self.out=disjoint(self.root/'assoc',ds,seg,feat)
        self.inputs=Inputs(ds,seg,feat)
    def tearDown(self):self.tmp.cleanup()
    def build(self):
        with redirect_stdout(io.StringIO()):prepare(self.inputs,self.out)
        return Catalog(self.out)
    def test_index_count_and_selection(self):
        c=self.build();self.assertEqual(len(c.rows),8)
        self.assertTrue(all(r['mask_status']=='keep' and r['frame_status']=='keep' for r in c.rows))
        self.assertTrue(all(r['mask_index']<2 for r in c.rows))
    def test_duplicate_prepare_no_changes(self):
        self.build();before=sha(self.out/'crop_vectors.npy')
        with redirect_stdout(io.StringIO()):prepare(self.inputs,self.out)
        self.assertEqual(sha(self.out/'crop_vectors.npy'),before)
    def test_source_files_unchanged(self):
        before={str(p):sha(p) for d in (self.ds,self.seg,self.feat) for p in d.rglob('*') if p.is_file()}
        c=self.build()
        with redirect_stdout(io.StringIO()):run_queries(c,[0],MatchPolicy(),self.out/'runs'/'test',3,3)
        after={str(p):sha(p) for d in (self.ds,self.seg,self.feat) for p in d.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
    def test_candidates_other_stations(self):
        c=self.build();cs=c.candidates(0,4,4)
        self.assertTrue(cs)
        self.assertTrue(all(c.rows[i]['station_id']!=c.rows[0]['station_id'] for i,s,r in cs))
    def test_scores_finite(self):
        c=self.build();s=c.cosine(0,chunk=3);self.assertTrue(np.isfinite(s).all());self.assertAlmostEqual(s[0],1,places=5)
    def test_surface_shape(self):
        c=self.build();store=GeometryStore(c,MatchPolicy(samples=64));s=store.surface(0)
        self.assertEqual(s.xyz.shape,(64,3));self.assertEqual(s.rgb.shape,(64,3))
        np.testing.assert_allclose(s.xyz[:,0],3,atol=1e-5)
    def test_resume_provenance_and_no_merge(self):
        c=self.build();out=self.out/'runs'/'test';p=MatchPolicy()
        with redirect_stdout(io.StringIO()):
            r=run_queries(c,[0,2],p,out,3,3);r2=run_queries(c,[0,2],p,out,3,3)
        self.assertEqual(len(r2),2)
        for q in r:
            self.assertIsNone(q['hypothesis']['object_id']);self.assertIsNone(q['hypothesis']['navigation_goal'])
            self.assertTrue(q['hypothesis']['overlap_with_other_hypotheses_allowed'])
            self.assertTrue(all(not e['same_instance_verified'] for e in q['edges']))
    def test_same_station_rejected(self):
        c=self.build();store=GeometryStore(c,MatchPolicy())
        with self.assertRaises(ValueError):check_pair(c,store,0,1,.9,['test'])
    def test_depth_corruption_detected(self):
        c=self.build();fid=c.rows[0]['frame_id'];p=self.ds/'frames'/fid/'depth_m.npy'
        a=np.load(p);a[:]=9;np.save(p,a)
        with self.assertRaises(ValueError):GeometryStore(c,MatchPolicy()).surface(0)
    def test_mask_corruption_detected(self):
        c=self.build();fid=c.rows[0]['frame_id'];(self.seg/'frames'/fid/'masks.npz').write_bytes(b'bad')
        with self.assertRaises(ValueError):GeometryStore(c,MatchPolicy()).surface(0)
    def test_feature_corruption_detected(self):
        c=self.build();fid=c.rows[0]['frame_id'];(self.feat/'frames'/fid/'features.npz').write_bytes(b'bad')
        with self.assertRaises(ValueError):GeometryStore(c,MatchPolicy()).surface(0)
    def test_policy_drift_refused(self):
        self.build();p=read_json(self.seg/'quality_policy.json');p['alpha_min']=.3;write_json(self.seg/'quality_policy.json',p)
        with self.assertRaises(ValueError):Catalog(self.out)
    def test_feature_incomplete_refused(self):
        s=read_json(self.feat/'summary.json');s['counts']['extracted_frames']=1;write_json(self.feat/'summary.json',s)
        with self.assertRaises(ValueError):Inputs(self.ds,self.seg,self.feat)
    def test_catalog_corruption_refused(self):
        self.build();(self.out/'crop_vectors.npy').write_bytes(b'bad')
        with self.assertRaises(ValueError):Catalog(self.out)
    def test_different_policy_run_name_refused(self):
        c=self.build();out=self.out/'runs'/'test'
        with redirect_stdout(io.StringIO()):run_queries(c,[0],MatchPolicy(),out,2,2)
        with self.assertRaises(ValueError):run_queries(c,[0],MatchPolicy(appearance_min=.9),out,2,2)
    def test_report_complete(self):
        c=self.build();out=self.out/'runs'/'test'
        with redirect_stdout(io.StringIO()):
            run_queries(c,[0],MatchPolicy(),out,3,3);p=make_report(c,out)
        self.assertTrue(p.exists());self.assertTrue((out/'qa/associations_001.jpg').exists())
        self.assertTrue(list((out/'queries').glob('*/supported_surface_preview_map.ply')))
    def test_preparation_detects_count_mismatch(self):
        self.inputs.summary['counts']['geometry_candidates']=7
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):prepare(self.inputs,self.out)
    def test_seed_station_diversity(self):
        c=self.build();q=c.seed_indices(4);self.assertEqual(len({c.rows[i]['station_id'] for i in q}),len(q))
    def test_output_disjoint(self):
        with self.assertRaises(ValueError):disjoint(self.ds/'output',self.ds)
    def test_ply_header(self):
        p=self.out/'x.ply';export_ply(p,[[1,2,3]],[[10,20,30]])
        b=p.read_bytes();self.assertIn(b'element vertex 1\n',b);self.assertEqual(len(b.split(b'end_header\n')[1]),15)


if __name__=='__main__':unittest.main()
