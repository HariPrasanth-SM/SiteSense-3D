"""Focused CPU checks. No GPU or private reconstruction required."""
import json,math,unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from avatar_demo.core import FloorMap,Planner,Walker,PlanningError,camera_matrices
from avatar_demo.manual import manual_step
from service_demo.workbook import load,asset_database,brief,excel_date
from service_demo.search import SearchIndex
from service_demo.common import atomic_json,read_json,number

class DemoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.db=asset_database(load())
    def test_workbook_counts(self):
        self.assertEqual(len(self.db['assets']),30)
        self.assertEqual(self.db['counts']['Telemetry_Sample_3d'],4320)
        self.assertEqual(self.db['counts']['Alarms_Events'],1000)
        self.assertEqual(self.db['counts']['Maintenance_History'],300)
    def test_placeholder_anchors(self):
        self.assertTrue(all('placeholder' in a['anchor_original']['mapping_status'] for a in self.db['assets'].values()))
    def test_brief_sources_and_recommendations(self):
        a=self.db['assets']['ASSET-003'];b=brief(a)
        self.assertEqual(b['next_check'],a['state']['recommended_next_check'])
        self.assertTrue(all(e['source'] for e in b['evidence']))
        self.assertIn('not a new model diagnosis',b['method'])
    def test_dates(self):self.assertEqual(excel_date(46303),'2026-10-08T00:00:00')
    def test_document_index_is_not_document_body(self):
        self.assertEqual(self.db['counts']['Knowledge_Docs'],8);self.assertEqual(self.db['counts']['RAG_Corpus_Index'],38)
        self.assertIn('not included',self.db['notice'])
    def planner(self):
        g=np.zeros((30,30),np.int8);g[:,0]=100;g[:,-1]=100;g[0]=100;g[-1]=100;g[8:20,14:17]=100;g[24,4]=-1
        return Planner(FloorMap(g,.1,np.array([-1.5,-1.5,0.])),0,0)
    def test_target_is_not_endpoint(self):
        p=self.planner();start=p.map.cell_to_world((14,6));target=p.map.cell_to_world((14,15));path=p.plan(start,target,standoff=.35,max_approach=.8)
        self.assertGreater(np.linalg.norm(path.endpoint-target),.34)
        for xy in path.xy:self.assertTrue(p.walkable[p.map.world_to_cell(xy)])
    def test_manual_unknown_blocked(self):
        p=self.planner();xy=p.map.cell_to_world((24,3));r=manual_step(xy,0,1,0,.1,1,1,p.map,p.walkable,True,False)
        self.assertTrue(r.blocked)
    def test_manual_noclip_explicit(self):
        p=self.planner();xy=p.map.cell_to_world((24,3));r=manual_step(xy,0,1,0,.1,1,1,p.map,p.walkable,False,False)
        self.assertFalse(r.blocked)
    def test_arrival_faces_target(self):
        p=self.planner();start=p.map.cell_to_world((14,6));target=p.map.cell_to_world((14,15));w=Walker(start,speed=.8,turn_rate=2);w.assign(p.plan(start,target,standoff=.35,max_approach=.8));w.start()
        for _ in range(2000):w.update(.05)
        self.assertEqual(w.state,'arrived');v=target-w.xy;self.assertAlmostEqual(w.yaw,math.atan2(v[1],v[0]))
    def test_camera_inverse(self):
        K,c2w,w2c=camera_matrices(1,2,.3,1.7,-10)
        p=c2w[:3,3];self.assertTrue(np.allclose([1.43*p[0],1.43*p[2],1.43*(1.1-p[1])],[1,2,1.7],atol=1e-5))
        self.assertTrue(np.allclose(c2w@w2c,np.eye(4),atol=1e-5))
    def index(self):
        x=SearchIndex.__new__(SearchIndex);rows=[]
        for i in range(4):rows.append({'index':i,'frame_id':f's{i}_v0','station_id':f's{i}','mask_index':0,'observation_id':str(i),'center':[i*.1,0,1],'lo':[i*.1,0,1],'hi':[i*.1+.1,.1,1.1]})
        x.catalog=SimpleNamespace(rows=rows,vectors=np.eye(4,dtype=np.float16),centers=np.array([r['center'] for r in rows]));x.by_obs={};return x
    def test_no_match_threshold(self):
        r=self.index().rank(np.array([1,1,1,1]),threshold=.7);self.assertTrue(r['no_match']);self.assertAlmostEqual(r['best_score'],.5)
    def test_match_not_probability(self):
        r=self.index().rank(np.array([1,0,0,0]),threshold=.7);self.assertEqual(len(r['matches']),1);self.assertIn('not calibrated confidence',r['score_note'])
    def test_nonfinite_control_refused(self):
        with self.assertRaises(ValueError):number(float('nan'),0,1,'speed')
    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'a.json';atomic_json(f,{'x':np.array([1,2])});self.assertEqual(read_json(f),{'x':[1,2]})

if __name__=='__main__':unittest.main()
