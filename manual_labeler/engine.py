from __future__ import annotations
from collections import OrderedDict
import copy
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
import uuid
import numpy as np
from PIL import Image
from avatar_demo.core import FloorMap, Planner
from avatar_demo.manual import segment_cells
from .geometry import (build_draft, camera_map, checked_box, draft_summary, box_corners,
                       project, scene_to_map, transform_record)
from .storage import AnnotationStore, image_data, depth_preview, overlay, atomic_json


def finite_number(x, low, high, label):
    v=float(x)
    if not math.isfinite(v) or not low <= v <= high:raise ValueError(f'{label} must be between {low} and {high}.')
    return v


class LabelEngine:
    def __init__(self,config,identity):
        self.cfg=config;self.lock=threading.RLock();self.map=FloorMap.load(config['map'])
        self.store=AnnotationStore(config['out'],identity)
        self.frames=OrderedDict();self.latest=None;self.frozen=None;self.draft=None;self.last_saved=None
        self.controls=[];self.heartbeat=0;self.status={'status':'starting','message':'Starting renderer'}
        self.message='Explore, stop, then Freeze to draw. No workbook mappings are used.'
        self.speed=config['speed'];self.turn_rate=math.radians(60);self.look_rate=25
        self.collision=True
        try:start=Planner(self.map,0,0).auto_start()
        except ValueError:
            valid=np.argwhere(self.map.occupancy!=100)
            if not len(valid):raise ValueError('Map contains no non-occupied starting cells.')
            start=self.map.cell_to_world(valid[len(valid)//2])
        self.pose=[float(start[0]),float(start[1]),math.radians(config['yaw']),config['camera_height'],0.]
        if config.get('start') is not None:
            p=np.asarray(config['start'],float);cell=self.map.world_to_cell(p)
            if not self.map.contains(cell) or self.map.occupancy[cell]==100:
                raise ValueError('--start must be in a non-occupied map cell.')
            self.pose[:2]=p.tolist()
        self.sequence=0;self.stop_local=threading.Event();self.process=None
        ctx=mp.get_context('spawn');self.shared=ctx.Array('d',[0,*self.pose]);self.q=ctx.Queue(maxsize=2)
        self.events=ctx.Queue(maxsize=16);self.stop_worker=ctx.Event()
        self.worker_id=uuid.uuid4().hex[:8]
        occ=self.map.occupancy
        colors=np.empty((*occ.shape,3),np.uint8);colors[:]=[112,119,130]
        colors[occ==100]=[35,42,54];colors[occ==0]=[203,222,189]
        self.map_png=image_data(np.flipud(colors))
        if config.get('snapshot_prefix'):
            self._load_snapshot(config['snapshot_prefix'])
            self.status={'status':'snapshot','message':'Saved-frame labeling: no live CUDA renderer.'}
        elif config.get('no_render'):
            self.status={'status':'disabled','message':'Controls only; no RGB/depth. Relaunch without --no-render to label.'}
        else:
            from .worker import render_worker
            self.process=ctx.Process(target=render_worker,args=(config,self.shared,self.q,self.events,self.stop_worker),daemon=True)
            self.process.start()
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()

    def _load_snapshot(self,prefix):
        prefix=str(Path(prefix).expanduser().resolve())
        rgb=np.asarray(Image.open(prefix+'_rgb.png').convert('RGB'))
        dep=np.load(prefix+'_depth.npy',allow_pickle=False).astype(np.float32)
        alpha=np.load(prefix+'_alpha.npy',allow_pickle=False).astype(np.float32)
        cam=json.loads(Path(prefix+'_camera.json').read_text())
        if dep.shape!=rgb.shape[:2] or alpha.shape!=dep.shape:raise ValueError('Snapshot dimensions disagree.')
        s=float(cam.get('scale',cam.get('scene_to_map_scale',self.cfg['scale'])))
        f=float(cam.get('floor_y',cam.get('floor_y_scene',self.cfg['floor_y'])))
        if abs(s-self.cfg['scale'])>1e-6 or abs(f-self.cfg['floor_y'])>1e-6:
            raise ValueError('Snapshot/map transforms disagree; do not silently rescale.')
        C=np.asarray(cam['camera_to_world'],float);W=np.linalg.inv(C)
        camera={**cam,'scale':s,'floor_y':f,'world_to_camera':W.tolist(),'width':rgb.shape[1],'height':rgb.shape[0]}
        CM=camera_map(camera);fw=CM[:3,2]
        camera.setdefault('map_pose',{'x':float(CM[0,3]),'y':float(CM[1,3]),'camera_height_m':float(CM[2,3]),
                                      'yaw_rad':math.atan2(fw[1],fw[0]),'pitch_deg':math.degrees(math.asin(np.clip(fw[2],-1,1)))})
        camera.update(camera_to_map=CM.tolist(),map_to_camera=np.linalg.inv(CM).tolist(),
                      pixel_sampling='column+0.5,row+0.5',scene_to_map=transform_record(s,f).tolist())
        p=camera['map_pose'];self.pose=[p['x'],p['y'],p['yaw_rad'],p['camera_height_m'],p['pitch_deg']]
        self._receive({'id':'snapshot_'+uuid.uuid4().hex[:10],'sequence':0,'rgb':rgb,'depth_scene':dep,
                       'depth_m':dep*s,'alpha':alpha,'camera':camera,'render_seconds':0})

    def _receive(self,frame):
        frame['id']=self.worker_id+'_'+frame['id']
        frame['rgb_png']=image_data(frame['rgb']);frame['depth_png']=image_data(depth_preview(frame))
        self.frames[frame['id']]=frame;self.latest=frame
        while len(self.frames)>8:
            key=next(iter(self.frames))
            if self.frozen is not None and self.frozen['id']==key:
                self.frames.move_to_end(key);continue
            self.frames.pop(key)

    def _loop(self):
        last=time.monotonic()
        while not self.stop_local.wait(.025):
            with self.lock:
                now=time.monotonic();dt=min(now-last,.1);last=now
                while True:
                    try:self.status=self.events.get_nowait()
                    except queue.Empty:break
                while True:
                    try:self._receive(self.q.get_nowait())
                    except queue.Empty:break
                old=self.pose[:]
                held=set(self.controls) if now-self.heartbeat<.45 and self.frozen is None and not self.cfg.get('snapshot_prefix') else set()
                v=(('forward' in held)-('backward' in held))*self.speed
                w=(('left' in held)-('right' in held))*self.turn_rate
                x,y,yaw,h,pitch=self.pose
                nyaw=(yaw+w*dt+math.pi)%(2*math.pi)-math.pi
                candidate=np.array([x,y])+v*dt*np.array([math.cos(yaw+w*dt/2),math.sin(yaw+w*dt/2)])
                allowed=True
                if v and self.collision:
                    allowed=all(self.map.contains(c) and self.map.occupancy[c]!=100 for c in segment_cells(self.map,[x,y],candidate))
                if allowed:x,y=candidate
                elif v:self.message='Movement blocked by occupied cells/map boundary. Inspection toggle can disable only this 2D check.'
                pitch=float(np.clip(pitch+25*dt*(('look_up' in held)-('look_down' in held)),-80,80))
                self.pose=[float(x),float(y),float(nyaw),h,pitch]
                if not np.allclose(self.pose,old,atol=1e-10,rtol=0):self._push_pose()

    def _push_pose(self):
        self.sequence+=1
        with self.shared.get_lock():self.shared[:]=[self.sequence,*self.pose]

    def state(self):
        with self.lock:
            f=self.frozen or self.latest
            frame=None
            if f is not None:
                frame={'id':f['id'],'width':f['rgb'].shape[1],'height':f['rgb'].shape[0],
                       'pose':f['camera']['map_pose'],'render_seconds':f['render_seconds'],
                       'captured_at':f['camera'].get('captured_wall_time')}
            items=[{k:o[k] for k in ('object_id','label','category','note','active','target_map_xyz','bbox_map','point_count','revision','reference_crop')}
                   | {'view_count':len(o['observations'])} for o in self.store.objects.values()]
            projections=[]
            if f:
                for o in self.store.objects.values():
                    if not o['active']:continue
                    points,valid=project(box_corners(o['bbox_map']),f['camera'])
                    projections.append({'id':o['object_id'],'label':o['label'],'uv':[p.tolist() if k else None for p,k in zip(points,valid)]})
            return {'pose':self.pose[:],'speed':self.speed,'turn_rate_deg':math.degrees(self.turn_rate),
                    'collision':self.collision,'frozen':self.frozen is not None,'frame':frame,
                    'renderer':self.status,'message':self.message,'objects':items,'box_projections':projections,
                    'map':{'width':self.map.shape[1],'height':self.map.shape[0],'resolution':self.map.resolution,'origin':self.map.origin.tolist()},
                    'transform':{'scale':self.cfg['scale'],'floor_y':self.cfg['floor_y'],'frame':'map'},
                    'output':str(self.store.root),'draft_id':self.draft['id'] if self.draft else None,
                    'static':bool(self.cfg.get('snapshot_prefix'))}

    def frame_bytes(self,ident,kind):
        with self.lock:
            f=self.frames.get(ident)
            if not f and self.frozen is not None and self.frozen['id']==ident:f=self.frozen
            if not f:raise ValueError('Frame expired. Refresh the live image and freeze again.')
            return f['depth_png'] if kind=='depth' else f['rgb_png']

    def command(self,cmd):
        with self.lock:
            action=cmd.get('action')
            if action=='drive':
                keys=cmd.get('keys',[])
                if not isinstance(keys,list) or any(k not in ('forward','backward','left','right','look_up','look_down') for k in keys):raise ValueError('Invalid drive controls.')
                self.controls=keys;self.heartbeat=time.monotonic()
            elif action=='stop':self.controls=[];self.heartbeat=0
            elif action=='settings':
                self.controls=[]
                self.speed=finite_number(cmd.get('speed',self.speed),.02,2,'Speed')
                self.collision=bool(cmd.get('collision',self.collision))
                if self.frozen is not None and any(k in cmd for k in ('height','pitch','yaw')):
                    raise ValueError('Resume live view before changing camera height or heading.')
                h=finite_number(cmd.get('height',self.pose[3]),.1,4,'Eye height')
                p=finite_number(cmd.get('pitch',self.pose[4]),-80,80,'Pitch')
                yaw=math.radians(finite_number(cmd.get('yaw',math.degrees(self.pose[2])),-36000,36000,'Heading'))
                self.pose[2:]=[yaw,h,p];self._push_pose()
            elif action=='start':
                if self.frozen is not None:raise ValueError('Resume live view before repositioning.')
                x=finite_number(cmd['x'],-10000,10000,'X');y=finite_number(cmd['y'],-10000,10000,'Y')
                c=self.map.world_to_cell([x,y])
                if self.collision and (not self.map.contains(c) or self.map.occupancy[c]==100):raise ValueError('Position is occupied or outside the map. Disable map checks only for inspection.')
                self.controls=[];self.pose[:2]=[x,y];self._push_pose();self.message='Explicit camera reposition; not a walked path.'
            elif action=='freeze':
                f=self.frames.get(str(cmd.get('frame_id','')))
                if f is None:raise ValueError('Displayed frame expired; wait for a fresh image and retry Freeze.')
                self.controls=[];self.heartbeat=0;self.frozen=f;self.draft=None;self.last_saved=None
                self.message='FROZEN: outline the object on this exact saved camera view.'
            elif action=='resume':
                self.frozen=None;self.draft=None;self.last_saved=None;self.controls=[]
                self._push_pose();self.message='Live exploration. Stop and Freeze before labeling.'
            elif action=='build':
                if self.frozen is None:raise ValueError('Freeze an image first.')
                if cmd.get('frame_id')!=self.frozen['id']:raise ValueError('Frozen frame changed; redraw on the current frame.')
                draft=build_draft(self.frozen,cmd['shape'],cmd.get('settings',{}))
                draft['id']=uuid.uuid4().hex;self.draft=draft;self.last_saved=None
                return draft_summary(draft,self.frozen['camera'])
            elif action=='box':
                self._check_draft(cmd)
                self.draft['box']=checked_box(cmd['box']);self.draft['box_manual']=True
                self.last_saved=None
                return draft_summary(self.draft,self.frozen['camera'])
            elif action=='save':
                self._check_draft(cmd)
                signature=(self.draft['id'],str(cmd.get('append_to','')),str(cmd.get('label','')),str(cmd.get('category','')),str(cmd.get('note','')))
                if self.last_saved and self.last_saved[0]==signature:return self.last_saved[1]
                rec=self.store.save(self.frozen,self.draft,cmd.get('label',''),cmd.get('category',''),cmd.get('note',''),cmd.get('append_to') or None)
                self.last_saved=(signature,rec);self.message=f"Saved {rec['object_id']} — {rec['label']}. Sources and coordinate frames preserved."
                return rec
            elif action=='rename':
                rec=self.store.rename(cmd['object_id'],cmd.get('label',''),cmd.get('category',''),cmd.get('note',''),cmd.get('active',True))
                return rec
            else:raise ValueError('Unknown action.')
            return {'ok':True}

    def _check_draft(self,cmd):
        if self.frozen is None or self.draft is None or cmd.get('draft_id')!=self.draft['id']:
            raise ValueError('The draft changed. Build the selection again before saving.')

    def draft_image(self):
        with self.lock:
            if self.frozen is None or self.draft is None:raise ValueError('No draft.')
            return image_data(np.asarray(overlay(self.frozen,self.draft)))

    def close(self):
        self.stop_local.set();self.stop_worker.set();self.controls=[]
        if hasattr(self,'thread'):self.thread.join(timeout=2)
        if self.process:
            self.process.join(timeout=3)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
