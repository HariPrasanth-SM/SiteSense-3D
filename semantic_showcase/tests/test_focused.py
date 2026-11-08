"""Focused CPU checks only; no assertion of real Valmet/GPU rendering quality."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from semantic_showcase.timeline import create_motion
from semantic_showcase.common import rz,scene_rotation,ROT,map_to_scene
from semantic_showcase.render import quat_left

class Focused(unittest.TestCase):
    def setUp(self):
        self.p={'regions':[{'region_id':f'r{i}','center':[i%4,i//4,.5],
             'display_label':f'R{i}', 'display_bounds_min':[i%4-.2,i//4-.2,.0],
             'display_bounds_max':[i%4+.2,i//4+.2,1.]} for i in range(20)],
             'fov_y_deg':55,'width':1280,'height':720}
        self.c=np.eye(4);self.c[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]];self.c[:3,3]=[2,-20,5]
        self.delta=np.zeros((20,3));self.delta[:,0]=np.arange(20)*2
        self.kw={'fps':24,'tail_frames':168,'formation':1.4,'pullback':1.4,'rotation_deg':35.,
                 'gap':.55,'forward_distance':None}
        self.ids=[f'r{i}' for i in range(16)]
        self.m,self.l=create_motion(self.p,self.c,self.delta,self.ids,**self.kw)
    def test_initial_continuity(self):
        np.testing.assert_allclose(self.m['camera_to_map'][0],self.c)
        np.testing.assert_allclose(self.m['center_map'][0],np.array([r['center'] for r in self.p['regions']])+self.delta)
        np.testing.assert_allclose(self.m['opacity_weight'][0],1)
        np.testing.assert_allclose(self.m['yaw_rad'][0],0)
    def test_order_and_end(self):
        self.assertEqual(self.l['first_four_ids'],['r0','r1','r2','r3'])
        self.assertEqual([(r['row'],r['column']) for r in self.l['positions'][:4]],[(0,0),(1,0),(0,1),(1,1)])
        np.testing.assert_allclose(self.m['center_map'][-1,:16],[r['center_map'] for r in self.l['positions']])
        np.testing.assert_allclose(self.m['opacity_weight'][-1,:16],1)
        np.testing.assert_allclose(self.m['opacity_weight'][-1,16:],0)
    def test_rigid_and_pivot(self):
        for k,j in enumerate(self.l['selected_indices']):
            c=np.r_[self.p['regions'][j]['center'],1]
            M=self.m['selected_transform_map'][-1,k]
            np.testing.assert_allclose((M@c)[:3],self.m['center_map'][-1,j])
            np.testing.assert_allclose(M[:3,:3].T@M[:3,:3],np.eye(3),atol=1e-9)
    def test_final_framing(self):
        # Rotational envelope of every selected region is within final safe camera frame.
        G=self.c[:3,:3];T=self.m['camera_to_map'][-1];tan=np.tan(np.deg2rad(55)/2)
        from semantic_showcase.common import corners
        for r in self.l['positions']:
            pts=corners(-np.array(r['rotation_envelope_half']),r['rotation_envelope_half'])@G.T+r['center_map']
            pc=(pts-T[:3,3])@T[:3,:3]
            self.assertTrue((pc[:,2]>0).all())
            self.assertLess((np.abs(pc[:,0]/pc[:,2])).max(),tan*1280/720)
            self.assertLess((np.abs(pc[:,1]/pc[:,2])).max(),tan)
    def test_camera_pan(self):
        i=round(1.4*24);j=round((167/24-1.4)*24)
        self.assertGreater((self.m['camera_to_map'][j,:3,3]-self.m['camera_to_map'][i,:3,3])@self.c[:3,0],0)
    def test_rotation_covariance(self):
        import torch
        rng=np.random.default_rng(5)
        Q=Rotation.random(9,random_state=rng).as_quat();qs=np.column_stack([Q[:,3],Q[:,:3]])
        R=scene_rotation(.43);q=Rotation.from_matrix(R).as_quat();q=np.r_[q[3],q[:3]]
        result=quat_left(torch.tensor(q),torch.tensor(qs),torch).numpy()
        newmat=Rotation.from_quat(np.column_stack([result[:,1:],result[:,0]])).as_matrix()
        oldmat=Rotation.from_quat(Q).as_matrix()
        np.testing.assert_allclose(newmat,R[None]@oldmat,atol=1e-10)
    def test_appearance_direction_equivalence(self):
        rng=np.random.default_rng(6);orig=rng.normal(size=(40,3));old=np.array([1,2,3.]);new=np.array([4,5,6.]);eye=np.array([-4,2,8.])
        R=scene_rotation(.35);scale=1.3
        world=(orig-old)@R.T*scale+new
        local_eye=old+R.T@(eye-new)/scale
        # These vectors differ only by the same positive scalar; normalized SH directions coincide.
        np.testing.assert_allclose((world-eye)@R/scale,orig-local_eye,atol=1e-12)
    def test_gaussian_similarity_conversion(self):
        # World-up map rotation maps correctly into original Y-down Gaussian axes.
        p=np.array([1,2,3.]);origin=np.array([-2,1,.5]);new=np.array([4,-1,2.]);R=rz(.35)
        result_map=(p-origin)@R.T+new
        ps=map_to_scene(p,1.43,1.1);os=map_to_scene(origin,1.43,1.1);ns=map_to_scene(new,1.43,1.1)
        result_scene=(ps-os)@scene_rotation(.35).T+ns
        np.testing.assert_allclose(result_scene,map_to_scene(result_map,1.43,1.1))

if __name__=='__main__':unittest.main()
