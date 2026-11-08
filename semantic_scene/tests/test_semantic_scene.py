from __future__ import annotations
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import io,json,tempfile,unittest,threading,urllib.request,urllib.error
import numpy as np
from PIL import Image
from semantic_scene.common import read_json,write_json,write_npz,file_records,sha,color_for
from semantic_scene.fusion import fuse_samples,bounding_box,select_members,distinct_positions
from semantic_scene.build import crop_pair,build
from semantic_scene.semantics import decide_label,validate_categories
from semantic_scene.viewer import SceneState,make_handler
from http.server import ThreadingHTTPServer

class FusionTests(unittest.TestCase):
    def test_same_voxel_same_observation_not_multiview(self):
        a=fuse_samples([[.001,0,0],[.002,0,0]],[[1,2,3],[3,4,5]],[0,0],[0,0],.02)
        self.assertEqual(len(a['xyz']),1);self.assertEqual(a['station_count'][0],1)
    def test_multi_station_support(self):
        a=fuse_samples([[.001,0,0],[.002,0,0]],[[1,2,3],[3,4,5]],[0,1],[0,1],.02)
        self.assertEqual(a['station_count'][0],2);self.assertEqual(a['membership_observation_indices'].tolist(),[0,1])
    def test_two_observations_one_station_not_two_views(self):
        a=fuse_samples([[0,0,0],[0,0,0]],[[0,0,0]]*2,[0,1],[0,0]);self.assertEqual(a['station_count'][0],1)
    def test_negative_coordinates(self):
        a=fuse_samples([[-.001,0,0],[.001,0,0]],[[0,0,0]]*2,[0,1],[0,1]);self.assertEqual(len(a['xyz']),2)
    def test_equal_observation_not_density_weight(self):
        a=fuse_samples([[0,0,0]]*10+[[.01,0,0]],[[0,0,0]]*10+[[200,100,0]],[0]*10+[1],[0]*10+[1])
        self.assertAlmostEqual(a['xyz'][0,0],.005);self.assertEqual(a['rgb'][0].tolist(),[100,50,0])
    def test_sample_to_voxel_complete(self):
        a=fuse_samples([[0,0,0],[1,0,0],[0,0,0]],[[0,0,0]]*3,[0,1,2],[0,1,2]);self.assertEqual(len(a['sample_to_voxel']),3)
    def test_overlap_not_suppressed_between_groups(self):
        x=[[0,0,0],[1,1,1]];a=fuse_samples(x,[[0,0,0]]*2,[0,1],[0,1]);b=fuse_samples(x,[[0,0,0]]*2,[0,1],[0,1]);np.testing.assert_array_equal(a['xyz'],b['xyz'])
    def test_no_voxel_mutation(self):
        p=np.array([[1.,2.,3.]]);copy=p.copy();fuse_samples(p,[[0,0,0]],[0],[0]);np.testing.assert_array_equal(p,copy)
    def test_invalid_points_refused(self):
        with self.assertRaises(ValueError):fuse_samples([[np.nan,0,0]],[[0,0,0]],[0],[0])
    def test_invalid_voxel_refused(self):
        with self.assertRaises(ValueError):fuse_samples([[0,0,0]],[[0,0,0]],[0],[0],0)
    def test_bounds_enclose_all_samples(self):
        p=np.array([[-3,1,4],[10,2,5],[1,3,6]]);b=bounding_box(p);self.assertEqual(b['min'],[-3,1,4]);self.assertEqual(b['max'],[10,3,6]);self.assertFalse(b['complete_object_boundary'])
    def test_stable_colors(self):self.assertEqual(color_for('a'),color_for('a'));self.assertNotEqual(color_for('a'),color_for('b'))

class SemanticTests(unittest.TestCase):
    def setUp(self):self.cat=[{'label':'pump','kind':'equipment','description':'a pump'},{'label':'wall','kind':'structure','description':'a wall'},{'label':'unknown','kind':'unknown','description':'a fragment'}]
    def test_supported_suggestion_not_confirmed(self):
        r=decide_label([[.5,.1,.1],[.6,.2,.1],[.5,.15,.05]],self.cat)
        self.assertEqual(r['suggested_label'],'pump');self.assertIsNone(r['semantic_label']);self.assertFalse(r['score_is_probability'])
    def test_margin_abstains(self):self.assertIsNone(decide_label([[.3,.29,.1]]*3,self.cat)['suggested_label'])
    def test_low_score_abstains(self):self.assertIsNone(decide_label([[.2,.1,.05]]*3,self.cat)['suggested_label'])
    def test_unknown_abstains(self):self.assertIsNone(decide_label([[.1,.1,.6]]*3,self.cat)['suggested_label'])
    def test_disagreeing_views_abstain(self):self.assertIsNone(decide_label([[.5,.1,.1],[.1,.5,.1],[.1,.1,.5]],self.cat)['suggested_label'])
    def test_one_view_abstains(self):self.assertIsNone(decide_label([[.5,.1,.1]],self.cat)['suggested_label'])
    def test_nan_scores_rejected(self):
        with self.assertRaises(ValueError):decide_label([[np.nan,0,0]],self.cat)
    def test_categories_require_unique(self):
        with self.assertRaises(ValueError):validate_categories([self.cat[0],self.cat[0]])
    def test_crop_unannotated_aspect(self):
        rgb=np.full((50,100,3),255,np.uint8);mask=np.zeros((50,100),bool);mask[10:40,20:80]=True
        a,b,box=crop_pair(rgb,mask);self.assertEqual(a.size,(384,384));self.assertNotEqual(a.tobytes(),b.tobytes());self.assertTrue((rgb==255).all())
    def test_empty_mask_refused(self):
        with self.assertRaises(ValueError):crop_pair(np.zeros((10,10,3),np.uint8),np.zeros((10,10),bool))


def minimal_scene(root):
    root=Path(root);rid='region_test';folder=root/'regions'/rid;(folder/'views').mkdir(parents=True)
    xyz=np.array([[0,0,0],[1,1,1],[1,0,1]],np.float32);rgb=np.array([[255,0,0]]*3,np.uint8)
    write_npz(folder/'surface.npz',xyz=xyz,rgb=rgb,station_count=np.array([1,2,3]))
    T=np.eye(4).tolist();view={'observation_id':'obs','frame_id':'f','mask_index':0,'station_id':'s','masked_crop':'views/v00_masked.png','context_crop':'views/v00_context.png','overview':'views/v00_overview.jpg','camera_to_map':T,'K':np.eye(3).tolist()}
    for name in ('v00_masked.png','v00_context.png','v00_overview.jpg'):Image.new('RGB',(20,20),(100,130,150)).save(folder/'views'/name)
    r={'region_id':rid,'member_count':3,'distinct_camera_positions_10cm':3,'point_count':3,'multiview_voxel_count':2,'bbox':bounding_box(xyz),'views':[view],'color':color_for(rid),'semantic_label':None}
    write_json(folder/'region.json',r);(folder/'surface.ply').write_bytes(b'ply\n');(folder/'multiview_voxels.ply').write_bytes(b'ply\n')
    files=['surface.npz','region.json',*(str(p.relative_to(folder)) for p in (folder/'views').iterdir())]
    write_json(folder/'complete.json',{'scene_id':'test_scene','files':file_records(folder,files)})
    write_json(root/'scene.json',{'schema':'valmet.semantic_scene.v1','scene_id':'test_scene','scale_m_per_scene_unit':1.43,'regions':[r]})
    return root

class ViewerTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=minimal_scene(self.tmp.name);self.s=SceneState(self.root)
    def tearDown(self):self.tmp.cleanup()
    def payload(self,**kw):
        p={'token':self.s.token,'region_id':'region_test','semantic_label':'tank','kind':'equipment','decision':'accepted','note':'checked','expected_revision':0};p.update(kw);return p
    def test_manifest_unknown_initially(self):self.assertIsNone(self.s.manifest()['regions'][0]['review']['semantic_label'])
    def test_binary_size(self):self.assertEqual(len(self.s.cloud_bytes('region_test')),3*24)
    def test_consensus_subset(self):self.assertEqual(len(self.s.cloud_bytes('region_test','multi')),2*24)
    def test_environment_optional(self):self.assertEqual(self.s.manifest()['environment_points'],0)
    def test_save_review_and_preserve_geometry(self):
        before=sha(self.root/'regions/region_test/surface.npz');self.s.review(self.payload());self.assertEqual(self.s.manifest()['regions'][0]['review']['semantic_label'],'tank');self.assertEqual(before,sha(self.root/'regions/region_test/surface.npz'))
    def test_label_may_stay_unknown(self):self.s.review(self.payload(semantic_label=''));self.assertIsNone(self.s.manifest()['regions'][0]['review']['semantic_label'])
    def test_wrong_token_refused(self):
        with self.assertRaises(PermissionError):self.s.review(self.payload(token='bad'))
    def test_stale_edit_refused(self):
        self.s.review(self.payload())
        with self.assertRaises(RuntimeError):self.s.review(self.payload())
    def test_unknown_id_refused(self):
        with self.assertRaises(ValueError):self.s.review(self.payload(region_id='other'))
    def test_environment_capped_not_scaled_again(self):
        p=self.root/'e.npz';np.savez(p,xyz=np.array([[3,4,5],[9,7,6]],np.float32),rgb=np.zeros((2,3),np.uint8));s=SceneState(self.root,p,max_env_points=1);self.assertEqual(len(s.env['xyz']),1);self.assertIn(s.env['xyz'][0].tolist(),[[3,4,5],[9,7,6]])
    def test_corrupt_cloud_refused(self):
        (self.root/'regions/region_test/surface.npz').write_bytes(b'bad')
        with self.assertRaises(ValueError):self.s.cloud_bytes('region_test')
    def test_http_pages_and_review(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.s));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base+'/') as r:self.assertIn(b'Semantic 3D',r.read())
            with urllib.request.urlopen(base+'/api/scene') as r:self.assertEqual(json.load(r)['scene_id'],'test_scene')
            req=urllib.request.Request(base+'/api/review',data=json.dumps(self.payload()).encode(),headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req) as r:self.assertEqual(json.load(r)['revision'],1)
        finally:server.shutdown();server.server_close();thread.join()
    def test_http_traversal_refused(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(self.s));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/media/region_test/../../scene.json')
        finally:server.shutdown();server.server_close();thread.join()

class BuildIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from associate_dataset.tests.test_stage4 import IntegrationTests
        from associate_dataset.pipeline import run_queries
        from associate_dataset.geometry import MatchPolicy
        cls.fixture=IntegrationTests();cls.fixture.setUp();c=cls.fixture.build();cls.cat=c
        # Original fixture uses identical cameras; create distinct-view evidence without altering source files:
        # This test uses the existing verifier, but minimum pose count is patched only for synthetic identical-camera fixture.
        with redirect_stdout(io.StringIO()):run_queries(c,[0,2],MatchPolicy(),cls.fixture.out/'runs'/'pilot_v1',6,6)
    @classmethod
    def tearDownClass(cls):cls.fixture.tearDown()
    def test_build_real_upstream_contract_and_resume(self):
        from unittest.mock import patch
        out=self.fixture.root/'scene';a=SimpleNamespace(association=self.fixture.out,run='pilot_v1',out=out,samples=64,max_views=4,min_views=2,voxel=.05)
        before={str(p):sha(p) for d in (self.fixture.ds,self.fixture.seg,self.fixture.feat) for p in d.rglob('*') if p.is_file()}
        with patch('semantic_scene.build.distinct_positions',return_value=4),redirect_stdout(io.StringIO()):s=build(a);s2=build(a)
        self.assertEqual(s['region_count'],s2['region_count']);self.assertGreater(s['region_count'],0)
        for r in s['regions']:
            rr=read_json(out/'regions'/r['region_id']/'region.json');self.assertIsNone(rr['semantic_label']);self.assertIsNone(rr['navigation_goal']);self.assertTrue(all(v['mask_status']=='keep' for v in rr['views']))
        after={str(p):sha(p) for d in (self.fixture.ds,self.fixture.seg,self.fixture.feat) for p in d.rglob('*') if p.is_file()};self.assertEqual(before,after)
    def test_real_distinct_gate_rejects_coincident_fixture(self):
        self.assertEqual(distinct_positions([0,2,4],self.cat),1)
    def test_member_selection_excludes_partial_and_rejected(self):
        rec=read_json(next((self.fixture.out/'runs/pilot_v1/queries').glob('*/result.json')))
        for e in rec['edges']:e['status']='partial_overlap'
        self.assertEqual(select_members(rec,self.cat),[rec['query_index']])

if __name__=='__main__':unittest.main()
