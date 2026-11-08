"""Focused annotation-join/motion tests; no GPU, model inference or real scene."""
import copy,json,math,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from unittest.mock import patch
from service_demo.manual_assets import join_annotations,normal_tag,import_labels,ManualAssets,safe_media_path
from service_demo.workbook import asset_database,load
from service_demo.engine import AppState
from service_demo.common import atomic_json,read_json
from service_demo.navigation import target_orientation,navigation_map
from avatar_demo.core import FloorMap,Planner,Walker

ANNOT=Path(__file__).parents[1]/'data/manual_annotations_supplied.json'

class ManualChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ann=read_json(ANNOT);cls.db=asset_database(load());cls.cfg={'scale':1.43,'floor_y':1.1}
    def joined(self,ann=None):return join_annotations(self.db,self.cfg,self.ann if ann is None else ann)
    def test_all_17_exact_matches(self):
        db,b,cat,rep=self.joined()
        self.assertEqual(len(db['assets']),17);self.assertEqual(rep['hidden_unlabelled_workbook_assets'],13)
        self.assertEqual(rep['corrections'],[])
        self.assertEqual(rep['categories'],{'Control Valve':5,'Electric Motor':4,'Pressure Transmitter':2,'Process Pump':3,'Vibration Sensor':3})
    def test_geometry_not_recalibrated(self):
        db,b,cat,r=self.joined()
        for o in self.ann['objects']:
            row=next(v for v in cat['assets'].values() if v['object_id']==o['object_id']);bind=b['assets'][row['asset_id']]
            self.assertEqual(bind['map_xyz'],o['target_map_xyz']);self.assertEqual(bind['bbox_map'],o['bbox_map'])
            self.assertIsNone(bind['region_id'])
    def test_workbook_values_exact_and_old_anchors_unused(self):
        db,b,c,r=self.joined();aid='ASSET-014'
        self.assertEqual(db['assets'][aid]['state'],self.db['assets'][aid]['state'])
        self.assertEqual(db['assets'][aid]['telemetry'],self.db['assets'][aid]['telemetry'])
        self.assertEqual(db['assets'][aid]['anchor_original'],self.db['assets'][aid]['anchor_original'])
        self.assertEqual(b['assets'][aid]['map_xyz'],self.ann['objects'][0]['target_map_xyz'])
    def test_inactive_and_nonsequential_tags(self):
        a=copy.deepcopy(self.ann);a['objects'][1]['active']=False
        db,b,c,r=self.joined(a);self.assertEqual(len(db['assets']),16);self.assertNotIn('ASSET-016',db['assets'])
        self.assertIn('ASSET-023',db['assets'])
    def test_spelling_changes_recorded_no_guess_identity(self):
        a=copy.deepcopy(self.ann);a['objects'][0]['label']=' p – 2 ';a['objects'][0]['category']='Proces pump'
        db,b,c,r=self.joined(a);self.assertEqual(c['assets']['ASSET-014']['label'],'P-002');self.assertEqual(len(r['corrections']),2)
        a['objects'][0]['label']='P-099'
        with self.assertRaisesRegex(ValueError,'no exact'):self.joined(a)
    def test_mismatch_refused(self):
        a=copy.deepcopy(self.ann);a['source']['scale']=2
        with self.assertRaisesRegex(ValueError,'scale'):self.joined(a)
        a=copy.deepcopy(self.ann);a['objects'][0]['target_map_xyz'][0]+=1
        with self.assertRaisesRegex(ValueError,'inconsistency'):self.joined(a)
    def test_duplicates_not_merged(self):
        a=copy.deepcopy(self.ann);a['objects'].append(copy.deepcopy(a['objects'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.joined(a)
    def test_nearby_targets_preserved(self):
        db,b,c,r=self.joined();self.assertIn('ASSET-014',c['assets']);self.assertIn('ASSET-016',c['assets'])
        self.assertTrue(any(set(w.get('tags',[]))=={'P-002','P-004'} for w in r['warnings']))
    def test_path_traversal_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):safe_media_path(d,'../wrong.png')
            with self.assertRaises(ValueError):safe_media_path(d,'/etc/passwd')
    def minimal_state(self):
        s=AppState.__new__(AppState)
        m=FloorMap(np.zeros((30,30),np.int8),.1,np.zeros(3));s.walker=Walker([.55,.55],0,speed=1,turn_rate=10)
        s.walker.assign(Planner(m,0,0).plan([.55,.55],[2.,1.8],standoff=.5,max_approach=1.))
        s.pitch=-48.;s.manual=[0,0,0];s.arrival_key='stale';s.arrival_pending=False;s.arrival_pose=None;s.arrival_bytes={};return s
    def test_walk_resets_pitch_before_motion(self):
        s=self.minimal_state();pos=s.walker.xy.copy();s._start_walk()
        self.assertEqual(s.pitch,0);np.testing.assert_array_equal(s.walker.xy,pos);self.assertIsNone(s.arrival_key)
    def test_travel_level_and_arrival_aim(self):
        s=self.minimal_state();s._start_walk()
        for i in range(1000):
            s.pitch=27.;s._advance_walker(.05)
            self.assertEqual(s.pitch,0)
            if s.walker.state=='arrived':break
        self.assertEqual(s.walker.state,'arrived')
        _,s.pitch=target_orientation(s.walker.xy,[2,1.8,.5],1.7)
        self.assertLess(s.pitch,0)
        s.walker.assign(Planner(FloorMap(np.zeros((30,30),np.int8),.1,np.zeros(3)),0,0).plan(s.walker.xy,[.5,.5],standoff=.3,max_approach=1))
        s._start_walk();self.assertEqual(s.pitch,0)
    def test_manual_mask_crop(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'objects'/'x'/'observations'/'v';folder.mkdir(parents=True)
            rgb=np.full((120,160,3),120,np.uint8);mask=np.zeros((120,160),np.uint8);mask[30:80,40:90]=255
            Image.fromarray(rgb).save(folder/'rgb.png');Image.fromarray(mask).save(folder/'mask_depth_supported.png');Image.fromarray(rgb).save(folder/'object_crop.png')
            cat={'schema':'sitesense.manual_asset_catalog.v1','source':{},'assets':{'A':{'reference_crop':'objects/x/observations/v/object_crop.png','observations':[{'path':'objects/x/observations/v/observation.json'}]}}}
            atomic_json(root/'manual_assets.json',cat);reader=ManualAssets(root,{'manual_assets':'manual_assets.json','manual_labels_root':str(root)})
            data,typ=reader.media('A','focus');import io
            im=Image.open(io.BytesIO(data));self.assertEqual(im.size,(62,62));self.assertEqual(typ,'image/png')
    def test_import_preserves_base_new_bundle_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);base=p/'base';base.mkdir();labels=p/'labels';labels.mkdir();mp=p/'map.yaml'
            Image.fromarray(np.full((170,210),254,np.uint8)).save(p/'map.pgm')
            mp.write_text('image: map.pgm\nmode: trinary\nresolution: 0.0715\norigin: [-7.865, -6.435, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
            ply=p/'scene.ply';ply.write_text('fixture only')
            cfg=dict(self.cfg,map=str(mp),ply=str(ply));atomic_json(base/'config.json',cfg);atomic_json(base/'workbook.json',self.db)
            original=(base/'workbook.json').read_bytes();a=copy.deepcopy(self.ann)
            for k in ('map_yaml_sha256','map_image_sha256','scene_sha256'):a['source'].pop(k,None)
            atomic_json(labels/'annotations.json',a)
            args=SimpleNamespace(base=str(base),out=str(p/'new'),annotations=str(labels/'annotations.json'),labels_root=str(labels))
            import_labels(args);self.assertEqual(original,(base/'workbook.json').read_bytes())
            self.assertEqual(len(read_json(p/'new'/'workbook.json')['assets']),17)
            self.assertEqual(read_json(p/'new'/'config.json')['scale'],1.43)
            with self.assertRaises(ValueError):import_labels(args)

if __name__=='__main__':unittest.main()
