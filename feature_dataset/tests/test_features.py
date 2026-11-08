from __future__ import annotations
from contextlib import redirect_stdout
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import copy
import io
import tempfile
import unittest
import numpy as np
from feature_dataset.preprocess import (make_layout,rgb_tensor,mask_crop,patch_weights,full_frame,
                                      weighted_pool,normalize_rows,MEAN,STD)
from feature_dataset.source import Source,audit,pilot
from feature_dataset.pipeline import infer_frame,extract,build_index,eligible
from feature_dataset.common import (read_json,write_json,write_npz,sha,digest,complete,is_complete,
                                    disjoint_output,json_lines)
from feature_dataset.report import make_report


class FakeEncoder:
    """Deterministic algebraic token stub. NOT DINO, NOT a learned retrieval test."""
    calls=0
    def __init__(self,*args):
        self.dim=8
        self.identity={'backbone':'TEST_STUB_NOT_DINO','dimension':8,'weights':'NO_REAL_WEIGHTS'}
    def infer(self,batch):
        type(self).calls+=1
        b,c,h,w=batch.shape
        p=batch.reshape(b,c,h//16,16,w//16,16).mean(axis=(3,5)).transpose(0,2,3,1).reshape(b,-1,3)
        tokens=np.concatenate([p,p*p,np.ones((*p.shape[:2],1),np.float32),p.mean(axis=-1,keepdims=True)],axis=-1)
        return tokens,tokens.mean(axis=1)


def args(**kw):
    a=SimpleNamespace(repo='unused',weights='unused',arch='dinov3_vitb16',device='cpu',precision='fp32',
        crop_size=64,frame_side=64,crop_pad=.08,batch_size=2,no_dense=False,pilot=False,pilot_frames=4,limit=None)
    for k,v in kw.items():setattr(a,k,v)
    return a


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.random.default_rng(1).integers(0,255,(48,64,3),dtype=np.uint8)
        self.mask=np.zeros((48,64),bool);self.mask[5:28,10:46]=True
    def test_full_native_layout(self):
        t,s=full_frame(self.rgb,640);self.assertEqual(t.shape,(3,48,64));self.assertEqual(s['scale_xy'],[1,1])
    def test_nonsquare_full_no_crop(self):
        s=make_layout(51,67,64,False);self.assertEqual(s['resized_hw'],[49,64]);self.assertEqual(s['canvas_hw'],[64,64])
    def test_crop_square_input(self):
        t,w,s=mask_crop(self.rgb,self.mask,64);self.assertEqual(t.shape,(3,64,64));self.assertEqual(w.shape,(16,))
    def test_aspect_preserving(self):
        t,w,s=mask_crop(self.rgb,self.mask,256)
        rh,rw=s['resized_hw'];h,ww=s['source_hw'];self.assertLess(abs(rh/rw-h/ww),.01)
    def test_affine_bbox_origin(self):
        _,_,s=mask_crop(self.rgb,self.mask,64);x0,y0,x1,y1=s['crop_xyxy'];A=np.array(s['original_to_encoder_pixel_edges'])
        np.testing.assert_allclose(A@[x0,y0,1],[s['pad_left'],s['pad_top'],1])
    def test_background_neutral(self):
        t,_,s=mask_crop(self.rgb,self.mask,64);self.assertLess(np.max(np.abs(t[:,0,0])),.03)
    def test_originals_unmodified(self):
        a=self.rgb.copy();b=self.mask.copy();mask_crop(self.rgb,self.mask);np.testing.assert_array_equal(a,self.rgb);np.testing.assert_array_equal(b,self.mask)
    def test_empty_mask(self): self.assertIsNone(mask_crop(self.rgb,np.zeros((48,64),bool)))
    def test_single_pixel_not_lost_crop(self):
        m=np.zeros((48,64),bool);m[1,1]=1;t,w,s=mask_crop(self.rgb,m,256);self.assertGreater(w.sum(),0)
    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):mask_crop(self.rgb,self.mask[:40])
    def test_padding_bounds(self):
        m=np.ones((48,64),bool);_,_,s=mask_crop(self.rgb,m,256)
        self.assertEqual(s['crop_xyxy'],[0,0,64,48])
    def test_invalid_side(self):
        with self.assertRaises(ValueError):make_layout(20,20,250,True)
    def test_weight_full_patch(self):
        s=make_layout(32,32,32,False);w=patch_weights(np.ones((32,32),bool),s);np.testing.assert_array_equal(w,1)
    def test_weight_fraction_preserved(self):
        m=np.zeros((32,32),bool);m[:8,:16]=1
        w=patch_weights(m,make_layout(32,32,32,False));np.testing.assert_allclose(w,[.5,0,0,0])
    def test_overlapping_masks_independent(self):
        a=patch_weights(self.mask,make_layout(48,64,64,False));b=patch_weights(self.mask.copy(),make_layout(48,64,64,False));np.testing.assert_array_equal(a,b)
    def test_pool_normalization(self):
        a,ok,m,e=weighted_pool(np.eye(2,dtype=np.float32),[1,1]);np.testing.assert_allclose(a,[[2**-.5,2**-.5]],atol=1e-6);self.assertTrue(ok[0]);self.assertEqual(e[0],2)
    def test_pool_no_background_weight(self):
        a,_,_,_=weighted_pool(np.eye(2,dtype=np.float32),[1,0]);np.testing.assert_array_equal(a,[[1,0]])
    def test_empty_pool_invalid(self):
        a,ok,m,e=weighted_pool(np.eye(2,dtype=np.float32),[0,0]);self.assertFalse(ok[0]);self.assertTrue((a==0).all())
    def test_bad_patch_count(self):
        with self.assertRaises(ValueError):weighted_pool(np.eye(2),[1,2,3])
    def test_nonfinite_normalization(self):
        a,ok=normalize_rows([[np.nan,1],[0,0],[3,4]]);self.assertEqual(ok.tolist(),[False,False,True]);np.testing.assert_allclose(a[2],[.6,.8])
    def test_inference_every_mask(self):
        m=np.stack([self.mask,self.mask,np.zeros_like(self.mask)])
        arrays,dense,meta=infer_frame(FakeEncoder(),self.rgb,m,args())
        self.assertEqual(arrays['crop_mask_pool'].shape,(3,8));np.testing.assert_array_equal(arrays['crop_mask_pool'][0],arrays['crop_mask_pool'][1]);self.assertFalse(arrays['crop_valid'][2])
        self.assertEqual(meta['empty_masks'],[2]);self.assertEqual(dense['patch_tokens'].shape,(3,4,8))
    def test_zero_masks(self):
        arrays,dense,meta=infer_frame(FakeEncoder(),self.rgb,np.zeros((0,48,64),bool),args());self.assertEqual(arrays['crop_mask_pool'].shape,(0,8))
    def test_batch_size_same_result(self):
        m=np.stack([self.mask]*5)
        a,_,_=infer_frame(FakeEncoder(),self.rgb,m,args(batch_size=1));b,_,_=infer_frame(FakeEncoder(),self.rgb,m,args(batch_size=3))
        for k in a:np.testing.assert_array_equal(a[k],b[k])


class PolicyTests(unittest.TestCase):
    def row(self,status='keep',geometry=True):return {'decision':{'status':status,'eligible_features':status=='keep','eligible_geometry':geometry}}
    def test_keep_accepted(self):self.assertEqual(eligible(self.row(),'keep',True),(True,True))
    def test_review_mask_not_accepted(self):self.assertEqual(eligible(self.row('review'),'keep',True),(False,False))
    def test_review_frame_not_accepted(self):self.assertEqual(eligible(self.row(),'review',True),(False,False))
    def test_invalid_descriptor(self):self.assertEqual(eligible(self.row(),'keep',False),(False,False))
    def test_visual_only_kept(self):self.assertEqual(eligible(self.row(geometry=False),'keep',True),(True,False))
    def test_exclude_mask_not_accepted(self):self.assertEqual(eligible(self.row('exclude'),'keep',True),(False,False))
    def test_pilot_unique(self):
        r=[dict(frame_id=f'f{i}',station_id=f's{i//3}',segmented=True,mask_count=1,frame_status='keep' if i%7 else 'review') for i in range(90)]
        p=pilot(r,32);self.assertEqual(len(p),32);self.assertEqual(len({x['frame_id'] for x in p}),32);self.assertTrue(any(x['frame_status']=='review' for x in p))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        from segment_dataset.tests.test_stage2 import make_dataset,FakePredictor
        from segment_dataset.pipeline import setup,audit as audit2,segment
        from segment_dataset.report import build_report
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        make_dataset(self.root/'capture',5)
        self.ds,self.seg,self.plan,self.pol,self.ph=setup(self.root/'capture',self.root/'seg')
        a=SimpleNamespace(pilot=False,pilot_frames=5,segment_excluded=False,limit=None,max_det=300,conf=.25)
        with redirect_stdout(io.StringIO()):
            self.records=audit2(self.ds,self.seg,self.plan,self.pol,self.ph)
            segment(self.ds,self.seg,self.plan,self.records,self.pol,self.ph,a,FakePredictor)
            build_report(self.ds,self.seg,self.plan,self.records,self.ph)
        self.source=Source(self.ds,self.seg);self.out=disjoint_output(self.root/'features',self.ds,self.seg)
        self.args=args();FakeEncoder.calls=0
    def tearDown(self):self.temp.cleanup()
    def runit(self):
        with redirect_stdout(io.StringIO()):return extract(self.source,self.out,self.args,FakeEncoder)
    def test_all_masks_and_strict_index(self):
        s=self.runit();self.assertEqual(s['counts']['archived_masks'],12)
        rows=list(json_lines(self.out/'observations.jsonl'));self.assertEqual(len(rows),12)
        self.assertTrue(all(r['semantic_label'] is None and r['object_id'] is None for r in rows))
        self.assertEqual(len({r['observation_id'] for r in rows}),12)
        self.assertTrue(all(r['mask_status']=='keep' and r['frame_status']=='keep' for r in json_lines(self.out/'association_candidates.jsonl')))
        self.assertTrue(any(r['mask_status']=='exclude' and r['feature_valid'] for r in rows))
    def test_resume_no_inference(self):
        self.runit();n=FakeEncoder.calls;self.runit();self.assertEqual(FakeEncoder.calls,n)
    def test_new_limit(self):
        self.args.limit=1;s=self.runit();self.assertEqual(s['counts']['extracted_frames'],1)
        s=self.runit();self.assertEqual(s['counts']['extracted_frames'],2)
    def test_corrupt_feature_regenerated(self):
        self.runit();f=next((self.out/'frames').glob('*/features.npz'));f.write_bytes(b'bad');s=self.runit();self.assertEqual(s['counts']['extracted_frames'],4)
    def test_source_mutation_refused(self):
        f=self.ds/'frames'/self.plan['frames'][0]['frame_id']/'rgb.png';f.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.runit()
    def test_originals_not_changed(self):
        before={str(p):sha(p) for folder in (self.ds,self.seg) for p in folder.rglob('*') if p.is_file()}
        self.runit()
        after={str(p):sha(p) for folder in (self.ds,self.seg) for p in folder.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
    def test_changed_crop_refused(self):
        self.runit();self.args.crop_size=80
        with self.assertRaises(ValueError):self.runit()
    def test_batch_size_resume_permitted(self):
        self.runit();self.args.batch_size=1;n=FakeEncoder.calls;self.runit();self.assertEqual(n,FakeEncoder.calls)
    def test_no_dense_option(self):
        self.args.no_dense=True;self.runit();self.assertFalse(list((self.out/'frames').glob('*/dense.npz')))
    def test_report_different_stations_and_no_merges(self):
        self.runit()
        with redirect_stdout(io.StringIO()):results=make_report(self.out,queries=3,topk=2)
        self.assertTrue((self.out/'qa/index.html').exists());self.assertTrue((self.out/'qa/neighbors_001.jpg').exists())
        for r in results:
            self.assertFalse(r['same_instance_verified'])
            for n in r['neighbors']:self.assertNotEqual(n['observation']['station_id'],r['query']['station_id'])
    def test_input_mask_archive_tamper(self):
        f=next((self.seg/'frames').glob('*/masks.npz'));f.write_bytes(b'bad')
        with self.assertRaises(ValueError):self.runit()
    def test_stale_policy_rejected(self):
        p=read_json(self.seg/'quality_policy.json');p['min_mask_area']=5000;write_json(self.seg/'quality_policy.json',p)
        with self.assertRaises(ValueError):Source(self.ds,self.seg)
    def test_source_row_order_rejected(self):
        f=self.seg/'analysis'/self.ph/self.plan['frames'][0]['frame_id']/'observations.jsonl'
        lines=f.read_text().splitlines();f.write_text('\n'.join(lines[::-1])+'\n')
        with self.assertRaises(ValueError):self.runit()
    def test_nested_output_rejected(self):
        with self.assertRaises(ValueError):disjoint_output(self.ds/'features',self.ds,self.seg)
    def test_audit_without_model(self):
        with redirect_stdout(io.StringIO()):r=audit(self.source,self.out)
        self.assertEqual(len(r),5);self.assertEqual(FakeEncoder.calls,0)
    def test_policy_reindex_without_inference(self):
        from segment_dataset.pipeline import audit as audit2,reselect
        from segment_dataset.report import build_report
        self.runit();n=FakeEncoder.calls
        p=replace(self.pol,min_mask_area=10000);ph=digest(asdict(p));write_json(self.seg/'quality_policy.json',asdict(p))
        with redirect_stdout(io.StringIO()):
            records=audit2(self.ds,self.seg,self.plan,p,ph);reselect(self.ds,self.seg,self.plan,records,p,ph)
            build_report(self.ds,self.seg,self.plan,records,ph)
            src=Source(self.ds,self.seg);s=build_index(src,self.out)
        self.assertEqual(FakeEncoder.calls,n);self.assertEqual(s['counts']['association_candidates'],0)
        self.assertEqual(s['counts']['archived_masks'],12)


if __name__=='__main__':unittest.main()
