from __future__ import annotations
import base64, csv, io, json, math, multiprocessing as mp, queue, secrets, threading, time, traceback, uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from avatar_demo.core import FloorMap,Planner,Walker,PlanningError,angle_wrap
from avatar_demo.manual import manual_step
from .common import read_json,atomic_json,jsonable,image_bytes,number
from .regions import Regions
from .manual_assets import ManualAssets, safe_media_path
from .workbook import brief
from .gpu import worker
from .navigation import navigation_map, preferred_start, route_unknown_cells, target_orientation, same_pose

class AppState:
    def __init__(self,args):
        self.root=Path(args.out).resolve();self.cfg=read_json(self.root/'config.json')
        self.db=read_json(self.root/'workbook.json');self.bindings=read_json(self.root/'asset_bindings.json')
        self.manual_assets=ManualAssets(self.root,self.cfg) if self.cfg.get('manual_assets') else None
        if self.manual_assets and (set(self.db['assets']) != set(self.manual_assets.items) or set(self.bindings['assets']) != set(self.manual_assets.items)):
            raise ValueError('Manual catalog, workbook subset and bindings differ. Re-import into a NEW bundle; no synthetic fallback used.')
        self.regions=Regions(self.cfg['regions'],self.cfg.get('enhancement'))
        self.map=FloorMap.load(self.cfg['map'])
        self.allow_unknown=not getattr(args,'block_unknown',False)
        self.nav_map=navigation_map(self.map,self.allow_unknown)
        self.planner=Planner(self.nav_map,args.radius,args.margin)
        self.title=getattr(args,'title','SITESENSE 3D').strip()
        self.walker=Walker(preferred_start(self.map,self.planner),math.pi/2,.25,math.radians(60))
        self.arrival_enabled=bool(getattr(args,'arrival_highlight',False))
        self.arrival={'state':'idle','message':'Arrival verification is optional; no target selected.'}
        self.arrival_key=None;self.arrival_bytes={};self.arrival_pose=None;self.arrival_pending=False
        self.unknown_count=0
        self.height=1.7;self.pitch=0.;self.mode='assets';self.collision=True;self.manual=[0.,0.,0.];self.manual_until=0
        self.lock=threading.RLock();self.stop_event=threading.Event();self.token=secrets.token_urlsafe(24)
        self.selected=None;self.plan_id=None;self.notice='Select an asset, upload a component image, or explore manually.'
        self.nearby=[];self.trail=[];self.frame=None;self.target_frame=None;self.frame_id=0;self.frame_bytes={};self.target_bytes={}
        self.renderer_status={'state':'starting','message':'Starting renderer'};self.query_status={'state':'loading_index','message':'Loading keep-only search catalog'}
        self.search_index=None;self.search_error=None;self.searches={};self.last_query=None
        self.session=self.root/'sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');self.session.mkdir(parents=True)
        self.events_file=(self.session/'events.jsonl').open('a');self.trajectory=(self.session/'trajectory.csv').open('w',newline='')
        self.writer=csv.writer(self.trajectory);self.writer.writerow(['time','x','y','yaw_rad','camera_height_m','pitch_deg','state','mode','manual_checks'])
        self.last_log=0.;self.last_pose=None
        self.gpu_config={**self.cfg,'width':args.width,'height':args.height,'render_hz':args.render_hz,
                         'no_render':args.no_render,'search_device':args.search_device,
                         'arrival_highlight':self.arrival_enabled,'fastsam':getattr(args,'fastsam','FastSAM-s.pt'),
                         'arrival_min_similarity':getattr(args,'arrival_min_similarity',.72),
                         'arrival_distance':getattr(args,'arrival_distance',.30)}
        if args.weights:self.gpu_config['weights']=str(Path(args.weights).resolve())
        if args.repo:self.gpu_config['repo']=str(Path(args.repo).resolve())
        ctx=mp.get_context('spawn');self.shared=ctx.Array('d',[0,*self.walker.xy,self.walker.yaw,self.height,self.pitch])
        self.frames=ctx.Queue(maxsize=1);self.gpu_events=ctx.Queue(maxsize=8);self.jobs=ctx.Queue(maxsize=4);self.gpu_stop=ctx.Event()
        self.process=ctx.Process(target=worker,args=(self.gpu_config,self.shared,self.frames,self.gpu_events,self.jobs,self.gpu_stop),daemon=True)
        self.process.start()
        if not args.no_search:threading.Thread(target=self._load_index,daemon=True).start()
        else:self.query_status={'state':'disabled','message':'Image search disabled by --no-search'}
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
        atomic_json(self.session/'config.json',{**self.cfg,'runtime':self.gpu_config,'radius':args.radius,'margin':args.margin,'project_title':self.title,'allow_unknown_for_demo':self.allow_unknown})
        self.log('started',{'source_map':self.cfg['map'],'workbook':self.db['source']})
    def log(self,event,data):
        with self.lock:
            self.events_file.write(json.dumps(jsonable({'at':time.time(),'event':event,'data':data}),allow_nan=False)+'\n');self.events_file.flush()
    def _load_index(self):
        try:
            from .search import SearchIndex
            index=SearchIndex(self.cfg['association'],self.regions)
            with self.lock:self.search_index=index;self.query_status={'state':'ready','message':f'{len(index.catalog.rows):,} keep-only geometric observations'}
        except Exception as e:
            print('Search unavailable:',e,flush=True)
            with self.lock:self.search_error=str(e);self.query_status={'state':'error','message':str(e)}
    def _encode_frames(self,f):
        d=f['depth_m'];valid=np.isfinite(d)&(d>0)
        vis=np.zeros(d.shape,np.uint8);vis[valid]=np.rint(np.clip(1-d[valid]/8,0,1)*255).astype(np.uint8)
        return {'rgb':image_bytes(f['rgb'],quality=88),'depth':image_bytes(np.repeat(vis[:,:,None],3,axis=2),quality=88)}
    def _rank_result(self,event):
        qid=event['id']
        try:
            result=self.search_index.rank(event['vector'],self.searches[qid].get('threshold',self.cfg.get('min_similarity',.7)),self.cfg.get('search_count',200))
            with self.lock:
                if qid not in self.searches:return
                self.searches[qid].update(state='complete',preprocessing=event.get('preprocessing'),effective_patches=event.get('effective_patches'),**result)
                self.searches[qid]['message']='No reliable location found. Crop the component or upload another image.' if result['no_match'] else f"{len(result['matches'])} candidates. Choose a match to plan; no automatic navigation."
                atomic_json(self.session/'queries'/qid/'results.json',self.searches[qid])
            self.log('search_complete',{'query':qid,'best_score':result['best_score'],'candidates':len(result['matches'])})
        except Exception as e:
            with self.lock:self.searches[qid].update(state='error',message=str(e))
    def _loop(self):
        prev=time.monotonic()
        while not self.stop_event.wait(.03):
            now=time.monotonic();dt=min(now-prev,.1);prev=now
            if not self.process.is_alive() and self.process.exitcode is not None:
                with self.lock:
                    self.renderer_status={'state':'error','message':f'GPU worker exited ({self.process.exitcode}). Restart this server; map controls remain usable.'}
                    for query in self.searches.values():
                        if query['state']=='encoding':query.update(state='error',message='GPU worker exited; inspect terminal and restart server.')
            with self.lock:
                if now>self.manual_until:self.manual=[0.,0.,0.]
                f,t,l=self.manual
                if f or t:
                    result=manual_step(self.walker.xy,self.walker.yaw,f,t,dt,self.walker.speed,self.walker.turn_rate,self.nav_map,self.planner.walkable,self.collision,False)
                    self.walker.xy=result.xy;self.walker.yaw=result.yaw;self.walker.state='manual blocked' if result.blocked else 'manual'
                    if result.blocked:self.notice='Manual motion blocked: '+result.reason
                else:
                    before=self.walker.state;self._advance_walker(dt)
                    if self.walker.state=='arrived' and before!='arrived':
                        if self.selected:
                            self.walker.yaw,self.pitch=target_orientation(self.walker.xy,self.selected['xyz'],self.height)
                        self.arrival_pending=self.arrival_enabled
                        self.notice='Arrived. Heading and head tilt now face the target; visibility is not guaranteed.'
                        self.log('arrived',{'selected':self.selected,'position':self.walker.xy.tolist()})
                if l and not self.walker.running:
                    self.pitch=float(np.clip(self.pitch+l*25*dt,-80,80))
                    self._clear_arrival('Head angle changed; previous highlight cleared.')
                if self.arrival_pending and self.walker.state=='arrived':
                    self.arrival_pending=False
                    self._queue_arrival()
                pose=[*self.walker.xy,self.walker.yaw,self.height,self.pitch]
                if self.last_pose is None or not np.allclose(pose,self.last_pose,atol=.001,rtol=0):
                    with self.shared.get_lock():self.shared[0]+=1;self.shared[1:]=pose
                    self.last_pose=list(pose)
                if now-self.last_log>.2:
                    self.writer.writerow([time.time(),*self.walker.xy,self.walker.yaw,self.height,self.pitch,self.walker.state,self.mode,self.collision]);self.trajectory.flush();self.last_log=now
                    if not self.trail or np.linalg.norm(np.asarray(self.trail[-1][:2])-self.walker.xy)>.05:
                        self.trail.append([*self.walker.xy,not self.collision]);self.trail=self.trail[-1500:]
            try:
                while True:
                    frame=self.frames.get_nowait();encoded=self._encode_frames(frame)
                    with self.lock:self.frame=frame;self.frame_bytes=encoded;self.frame_id+=1;self.renderer_status={'state':'ready','message':f"Render {frame['render_seconds']:.3f}s · synthetic RGB/depth"}
            except queue.Empty:pass
            try:
                while True:
                    e=self.gpu_events.get_nowait();kind=e['kind']
                    with self.lock:
                        if kind=='render_status':self.renderer_status={'state':e['state'],'message':e['message']}
                        elif kind=='target_frame' and e['id']==self.plan_id:
                            self.target_frame=e['frame'];self.target_bytes=self._encode_frames(e['frame'])
                        elif kind=='arrival_status' and e['id']==self.arrival_key:
                            self.arrival.update(state='checking',message=e['message'])
                        elif kind=='arrival_result' and e['id']==self.arrival_key:
                            current=[*self.walker.xy,self.walker.yaw,self.height,self.pitch]
                            if self.walker.state=='arrived' and same_pose(current,e['frame']['pose']):
                                self._accept_arrival(e)
                        elif kind=='query_status' and e['id'] in self.searches:self.searches[e['id']]['message']=e['message']
                        elif kind=='query_encoded':threading.Thread(target=self._rank_result,args=(e,),daemon=True).start()
                        elif kind=='job_error':
                            if e['job_kind']=='query' and e['id'] in self.searches:self.searches[e['id']].update(state='error',message=e['message'])
                            elif e['job_kind']=='arrival' and e['id']==self.arrival_key:
                                self.arrival={'state':'error','message':e['message'],'id':self.arrival_key}
                                self.arrival_bytes={}
                            elif e['job_kind']!='arrival':self.notice=e['message']
            except queue.Empty:pass
    def _advance_walker(self, dt):
        # Every travelling frame is horizontal, including turns and resumed walks.
        # The existing arrival branch then aims at the target's annotated elevation.
        if self.walker.running:
            self.pitch = 0.0
        self.walker.update(dt)

    def _start_walk(self):
        if self.walker.plan is None:
            raise PlanningError('Plan a route first.')
        self._clear_arrival()
        self.manual = [0., 0., 0.]
        self.pitch = 0.0
        self.walker.start()
        self.notice = 'Navigating with a level view (pitch 0°). The camera will face the target at arrival.'

    def _clear_arrival(self,message='Target changed; previous highlight cleared.'):
        self.arrival_key=None;self.arrival_pose=None;self.arrival_pending=False;self.arrival_bytes={}
        self.arrival={'state':'idle','message':message}

    def _arrival_reference(self):
        selected=self.selected or {};rid=selected.get('region_id')
        if self.manual_assets and selected.get('kind') == 'asset':
            obj = self.manual_assets.items[selected['asset_id']]
            crop = safe_media_path(self.manual_assets.root, obj.get('reference_crop'))
            cloud = safe_media_path(self.manual_assets.root, obj.get('cloud_npz'))
            if crop is None or not crop.is_file() or cloud is None or not cloud.is_file():
                raise ValueError('Manual reference crop/cloud missing. Check the imported labels-root; no old synthetic reference substituted.')
            with np.load(cloud, allow_pickle=False) as d:
                support = np.asarray(d['xyz'], np.float32)
            support = support[np.isfinite(support).all(1)]
            if len(support)>4096:support=support[np.linspace(0,len(support)-1,4096,dtype=int)]
            binding=read_json(Path(self.cfg['association'])/'source.json')
            run=read_json(Path(binding['features'])/'run.json')
            return {'reference_image':str(crop),'support_xyz':support,'feature_run':run,
                    'reference_manual_object':obj['object_id']}
        if rid and rid in self.regions.items:
            root=self.regions.root/'regions'/rid
            with np.load(root/'descriptors.npz',allow_pickle=False) as d:
                refs=np.asarray(d['per_observation_crop'],np.float32)
            source=(self.regions.enh/'regions'/rid/'surface_clean.npz') if self.regions.enh else root/'surface.npz'
            if not source.exists():source=root/'surface.npz'
            with np.load(source,allow_pickle=False) as d:support=np.asarray(d['xyz'],np.float32)
            if len(support)>4096:support=support[np.linspace(0,len(support)-1,4096,dtype=int)]
            binding=read_json(Path(self.cfg['association'])/'source.json')
            run=read_json(Path(binding['features'])/'run.json')
            return {'references':refs,'support_xyz':support,'feature_run':run,'reference_region':rid}
        if selected.get('kind')=='image' and self.search_index is not None:
            from semantic_scene_plus.geometry import source_mask
            from associate_dataset.geometry import backproject
            row=self.search_index.catalog.rows[selected['match_index']]
            folder=self.search_index.dataset/'frames'/row['frame_id']
            mask=source_mask(self.search_index.seg/'frames'/row['frame_id']/'masks.npz',row['mask_index'])
            d=np.load(folder/'depth_m.npy');a=np.load(folder/'alpha.npy');cam=read_json(folder/'camera.json')
            yy,xx=np.where(mask & np.isfinite(d)&(d>0)&(a>=.5))
            if len(xx)>4096:
                ids=np.linspace(0,len(xx)-1,4096,dtype=int);xx,yy=xx[ids],yy[ids]
            support=backproject(np.column_stack([xx,yy]),d[yy,xx],cam)
            return {'references':np.asarray(self.search_index.catalog.vectors[row['index']],np.float32)[None],
                    'support_xyz':support,'feature_run':self.search_index.feature_run,
                    'reference_observation':row['observation_id']}
        raise ValueError('No linked region or geometric image reference for this target. No automatic highlight.')

    def _queue_arrival(self):
        if not self.arrival_enabled:
            self.arrival={'state':'disabled','message':'Start with --arrival-highlight to enable FastSAM + DINOv3.'};return
        if self.gpu_config['no_render']:
            self.arrival={'state':'disabled','message':'Camera rendering is disabled.'};return
        try:
            reference=self._arrival_reference()
            pose=[*self.walker.xy,self.walker.yaw,self.height,self.pitch]
            key=uuid.uuid4().hex[:16]
            self.jobs.put_nowait({'kind':'arrival','id':key,'pose':pose,**reference})
            self.arrival_key=key;self.arrival_pose=pose;self.arrival_bytes={}
            self.arrival={'state':'queued','message':'Checking the final rendered frame with FastSAM + DINOv3.','id':key}
        except Exception as exc:
            self.arrival={'state':'unavailable','message':str(exc)}

    def _accept_arrival(self,event):
        result=event['result'];frame=event['frame']
        small={k:v for k,v in result.items() if k not in ('overlay','crop','mask')}
        self.arrival={**small,'id':event['id'],'pose':frame['pose']}
        self.arrival_pose=frame['pose'];self.arrival_bytes={}
        self.frame=frame;self.frame_bytes=self._encode_frames(frame);self.frame_id+=1
        if result.get('state')=='matched':
            self.arrival_bytes={'rgb':image_bytes(result['overlay'],quality=90),'crop':image_bytes(result['crop'],'PNG')}
        dest=self.session/'arrivals'/event['id'];dest.mkdir(parents=True,exist_ok=True)
        # Separate raw camera and visualization products: never overwrite raw RGB/depth.
        from avatar_demo.render_worker import save_frame
        save_frame(frame,dest/'view')
        if result.get('state')=='matched':
            Image.fromarray(result['overlay']).save(dest/'overlay.png')
            Image.fromarray(result['crop']).save(dest/'target_crop.png')
            Image.fromarray(result['mask'].astype(np.uint8)*255).save(dest/'mask.png')
        atomic_json(dest/'verification.json',{'selected':self.selected,'verification':small,
                    'pose':frame['pose'],'plan_id':self.plan_id,'note':'Heuristic correspondence, not verified equipment identity.'})
        self.renderer_status={'state':'ready','message':'Arrival view rendered; raw images retained.'}
        self.log('arrival_check',small)

    def camera_image(self,kind):
        with self.lock:
            current=[*self.walker.xy,self.walker.yaw,self.height,self.pitch]
            active=self.walker.state=='arrived' and same_pose(current,self.arrival_pose)
            if kind=='highlight':return self.arrival_bytes.get('rgb') if active else None
            if kind=='target_crop':return self.arrival_bytes.get('crop') if active else None
            return self.frame_bytes.get(kind)

    def bootstrap(self):
        return {'token':self.token,'title':self.title,'team':'SpaceBots','allow_unknown':self.allow_unknown,'arrival_enabled':self.arrival_enabled,'map':{'width':self.map.shape[1],'height':self.map.shape[0],'resolution':self.map.resolution,'origin':self.map.origin.tolist()},
                'assets':[self.asset_summary(aid) for aid in self.db['assets']], 'regions':[] if self.manual_assets else self.regions.public(),
                'manual_annotation_mode':bool(self.manual_assets),
                'workbook_notice':self.db['notice'],'source':self.db['source'],'session':str(self.session),
                'scale':self.cfg['scale'],'floor_y':self.cfg['floor_y'],'min_similarity':self.cfg['min_similarity']}
    def asset_summary(self,aid):
        a=self.db['assets'][aid];s=a['state'];b=self.bindings['assets'][aid]
        return {'id':aid,'tag':a['master']['tag'],'class':a['master']['asset_class'],'area':a['master']['area_name'],
                'health':s.get('health_score'),'condition':s.get('condition_state'),'urgency':s.get('urgency'),
                'issue':s.get('issue_summary'),'mapping':b,'quality_flags':len(a['quality']),
                'manual_annotation':self.manual_assets.info(aid) if self.manual_assets else None}
    def asset_detail(self,aid):
        a=self.db['assets'][aid]
        return {**a,'binding':self.bindings['assets'][aid],'brief':brief(a),
                'manual_annotation':self.manual_assets.info(aid) if self.manual_assets else None}
    def snapshot(self):
        with self.lock:
            near=[]
            for aid,b in self.bindings['assets'].items():
                if not b.get('map_xyz'):continue
                dist=float(np.linalg.norm(np.asarray(b['map_xyz'][:2])-self.walker.xy))
                if dist<=self.cfg.get('nearby_radius',2):near.append({**self.asset_summary(aid),'distance':dist})
            near.sort(key=lambda r:r['distance'])
            frame_info=None
            if self.frame:
                frame_info={'pose':self.frame['pose'],'age_seconds':time.time()-self.frame['time'],'render_seconds':self.frame['render_seconds'],'id':self.frame_id}
            q=self.searches.get(self.last_query)
            return jsonable({'pose':[*self.walker.xy,self.walker.yaw],'height':self.height,'pitch':self.pitch,'speed':self.walker.speed,
                'turn_rate':math.degrees(self.walker.turn_rate),'state':self.walker.state,'running':self.walker.running,'mode':self.mode,
                'collision':self.collision,'allow_unknown':self.allow_unknown,'manual_annotation_mode':bool(self.manual_assets),'arrival_enabled':self.arrival_enabled,'arrival':self.arrival,
                'arrival_active':self.walker.state=='arrived' and same_pose([*self.walker.xy,self.walker.yaw,self.height,self.pitch],self.arrival_pose),
                'selected':self.selected,'plan':None if not self.walker.plan else {'xy':self.walker.plan.xy.tolist(),
                'endpoint':self.walker.plan.endpoint.tolist(),'target':self.walker.plan.target.tolist(),'length':self.walker.plan.length,'note':self.walker.plan.note,'unknown_cells':self.unknown_count},
                'notice':self.notice,'frame':frame_info,'target_preview':bool(self.target_bytes),'plan_id':self.plan_id,'target_time':self.target_frame.get('time') if self.target_frame else None,
                'renderer':self.renderer_status,'search':self.query_status,'query':{'id':self.last_query,'state':q['state'],'message':q.get('message','')} if q else None,
                'nearby':near[:6],'trail':self.trail[-600:]})
    def resolve_target(self,body):
        if body.get('asset_id'):
            aid=body['asset_id'];b=self.bindings['assets'][aid]
            if not b.get('map_xyz'):raise ValueError('This asset is unmapped. Use Bind to region or Place demo pin first.')
            target={'kind':'asset','asset_id':aid,'region_id':b.get('region_id'),'xyz':list(b['map_xyz']),'title':self.db['assets'][aid]['master']['tag'],'mapping_status':b['status']}
            if self.manual_assets:
                info=self.manual_assets.info(aid)
                target.update(manual_object_id=info['object_id'],bbox_map=info['bbox_map'],
                              crop_url=info['crop_url'],focus_url=info['focus_url'],
                              crop_available=info['crop_available'],reference_camera_pose=info['reference_camera_pose'])
            return target
        if body.get('query_id'):
            q=self.searches[body['query_id']];idx=int(body['match_index'])
            match=next((m for m in q.get('matches',[]) if m['index']==idx),None)
            if match is None:raise ValueError('Candidate is not in this query’s accepted result list.')
            return {'kind':'image','query_id':body['query_id'],'match_index':idx,'region_id':match.get('region_id'),'xyz':match['map_xyz'],
                    'title':match['frame_id']+f" · m{match['mask_index']:04d}",'score':match['score'],'match':match}
        xy=body.get('xy')
        if xy is None or len(xy)!=2:raise ValueError('Provide a target or valid asset/search result.')
        return {'kind':'point','xyz':[number(xy[0],-1000,1000,'X'),number(xy[1],-1000,1000,'Y'),1.0],'title':'Manual target','region_id':None}
    def command(self,b):
        action=b.get('action')
        with self.lock:
            if action=='mode':
                if b['mode'] not in ('assets','image','manual'):raise ValueError('Invalid mode')
                self.walker.pause();self.manual=[0,0,0];self.mode=b['mode'];return self.snapshot()
            if action in ('select','plan'):
                self.manual=[0,0,0];self.walker.pause();self._clear_arrival();self.selected=self.resolve_target(b);self.target_frame=None;self.target_bytes={}
                if action=='plan':
                    mode='point' if b.get('point_goal') else 'object'
                    p=self.planner.plan(self.walker.xy,self.selected['xyz'][:2],mode,number(b.get('standoff',.8),.1,4,'Stand-off'),number(b.get('max_approach',3.5),.2,10,'Max approach'))
                    self.unknown_count=route_unknown_cells(self.map,p.xy)
                    self.walker.assign(p);self.plan_id=uuid.uuid4().hex[:12];self.notice=f'Planned {p.length:.2f} m. Review the path, then Start.'+(' Includes unverified gray cells.' if self.unknown_count else '')
                    atomic_json(self.session/f'plan_{self.plan_id}.json',{'selected':self.selected,'xy':p.xy,'target':p.target,'endpoint':p.endpoint,'length':p.length,'note':p.note})
                    self.log('plan',{'id':self.plan_id,'target':self.selected})
                else:self.walker.plan=None;self.plan_id=None;self.walker.state='idle';self.notice='Target selected. Plan route to its nearby approach point.'
            elif action=='walk':self._start_walk()
            elif action=='pause':self.manual=[0,0,0];self.walker.pause();self.notice='Paused.'
            elif action=='controls':
                axes=[number(b.get(k,0),-1,1,k) for k in ('forward','turn','look')]
                if (axes[0] or axes[1]) and self.mode!='manual':raise ValueError('Select Manual mode for movement controls.')
                if axes[0] or axes[1]:
                    self._clear_arrival('Manual motion: old target highlight cleared.');self.walker.pause();self.walker.plan=None;self.plan_id=None;self.walker.state='manual';self.notice='Manual exploration; automatic route cleared.'
                self.manual=axes;self.manual_until=time.monotonic()+.45
            elif action=='settings':
                self._clear_arrival('Camera controls changed; previous highlight cleared.')
                self.walker.speed=number(b.get('speed',self.walker.speed),.05,1.,'Speed')
                self.height=number(b.get('height',self.height),.3,2.7,'Eye height')
                requested_pitch=number(b.get('pitch',self.pitch),-80,80,'Pitch')
                self.pitch=0.0 if self.walker.running else requested_pitch
                self.walker.turn_rate=math.radians(number(b.get('turn_rate',math.degrees(self.walker.turn_rate)),10,150,'Turn rate'))
                if 'collision' in b:
                    self.collision=bool(b['collision']);self.manual=[0,0,0];self.walker.pause()
                    self.notice='Manual map checks enabled.' if self.collision else 'FREE-ROAM INSPECTION: may pass through walls. This is not a valid route.'
            elif action in ('start','reset'):
                self._clear_arrival()
                xy=preferred_start(self.map,self.planner) if action=='reset' else self.planner.set_start(b['xy'])
                self.walker=Walker(xy,self.walker.yaw,self.walker.speed,self.walker.turn_rate);self.manual=[0,0,0];self.plan_id=None
                self.notice='Avatar reset to a valid map cell; reset is not travel.';self.log('reset',{'xy':xy})
            elif action=='highlight':
                if self.walker.state!='arrived':raise ValueError('Arrive at a planned target before checking its appearance.')
                if self.selected:self.walker.yaw,self.pitch=target_orientation(self.walker.xy,self.selected['xyz'],self.height)
                self._queue_arrival()
            elif action=='face_target':
                if not self.selected:raise ValueError('Select a target first.')
                self.walker.pause();self.manual=[0,0,0];self._clear_arrival()
                self.walker.yaw,self.pitch=target_orientation(self.walker.xy,self.selected['xyz'],self.height)
                self.notice='View aimed at the selected target.'
            elif action=='target_preview':
                if not self.walker.plan:raise ValueError('Plan a route before rendering its approach view.')
                if self.gpu_config['no_render']:raise ValueError('Rendering is disabled.')
                p=self.walker.plan;d=p.target-p.endpoint;yaw=math.atan2(d[1],d[0]);z=self.selected['xyz'][2]
                yaw,pitch=target_orientation(p.endpoint,self.selected['xyz'],self.height)
                self.target_frame=None;self.target_bytes={}
                self.jobs.put_nowait({'kind':'target','id':self.plan_id,'pose':[*p.endpoint,yaw,self.height,pitch]})
                self.notice='Rendering target approach view without moving the avatar.'
            elif action=='snapshot':
                if self.frame is None:raise ValueError('No rendered frame yet.')
                from avatar_demo.render_worker import save_frame
                frame={**self.frame,'camera':self.frame['camera']};prefix=self.session/'snapshots'/datetime.now().strftime('%H%M%S_%f')
                save_frame(frame,prefix);self.notice='Saved RGB/depth/camera: '+str(prefix)
            elif action=='bind':
                if self.manual_assets:raise ValueError('Manual annotation positions are read-only in this dashboard. Edit labels in the labeler and import a new bundle instead.')
                aid=b['asset_id']
                if aid not in self.db['assets']:raise ValueError('Unknown asset')
                rid=b.get('region_id')
                if rid:
                    r=self.regions.items[rid];xyz=r['bbox']['center']
                else:
                    xy=b['xy'];cell=self.map.world_to_cell(xy)
                    if not self.map.contains(cell):raise ValueError('Pin must be inside the map.')
                    xyz=[number(xy[0],-1000,1000,'X'),number(xy[1],-1000,1000,'Y'),1.0]
                binding=self.bindings['assets'][aid];binding.update(map_xyz=list(xyz),region_id=rid,status='user_demo_binding',
                    note='User-selected demo mapping; original synthetic anchor retained. This does not verify physical asset identity.')
                self.bindings['revision']+=1;atomic_json(self.root/'asset_bindings.json',self.bindings)
                self.log('asset_binding',binding);self.notice=f'Demo binding saved for {aid}.'
            else:raise ValueError('Unknown action')
            return self.snapshot()
    def upload(self,body):
        if self.search_index is None:raise ValueError('Image index not ready: '+self.query_status['message'])
        data=body.get('image','');data=data.split(',',1)[-1]
        raw=base64.b64decode(data,validate=True)
        if len(raw)>12*1024**2:raise ValueError('Use an image below 12 MiB.')
        with Image.open(io.BytesIO(raw)) as im:
            if im.width*im.height>20_000_000:raise ValueError('Image exceeds 20 megapixels; crop or reduce it first.')
            im=ImageOps.exif_transpose(im).convert('RGBA');im.thumbnail((2048,2048),Image.Resampling.LANCZOS)
            if min(im.size)<16:raise ValueError('The query crop is too small (minimum 16 pixels).')
        qid=uuid.uuid4().hex[:16];folder=self.session/'queries'/qid;folder.mkdir(parents=True);im.save(folder/'query.png')
        with self.lock:
            self.searches[qid]={'threshold':number(body.get('min_similarity',self.cfg.get('min_similarity',.7)),.4,.98,'Minimum cosine'),'id':qid,'state':'encoding','message':'Encoding the uploaded component…','matches':[],'file':str(folder/'query.png')};self.last_query=qid
        try:self.jobs.put_nowait({'kind':'query','id':qid,'file':str(folder/'query.png'),'feature_run':self.search_index.feature_run})
        except queue.Full:
            with self.lock:self.searches[qid].update(state='error',message='GPU job queue busy; submit again after the current request.')
        self.log('query_uploaded',{'id':qid,'size':im.size});return {'id':qid}
    def query_page(self,qid,offset=0):
        with self.lock:
            q=self.searches[qid];matches=q.get('matches',[]);offset=max(0,int(offset))
            return {k:v for k,v in {**q,'matches':matches[offset:offset+5],'offset':offset,'total':len(matches),
                                  'more':offset+5<len(matches)}.items() if k!='file'}
    def map_png(self):
        a=np.empty((*self.map.shape,3),np.uint8);a[:]=[71,85,105];a[self.map.occupancy==100]=[24,34,49];a[self.map.occupancy==0]=[130,174,154]
        return image_bytes(np.flipud(a),'PNG')
    def close(self):
        self.stop_event.set();self.gpu_stop.set();self.thread.join(timeout=2)
        self.process.join(timeout=3)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
        self.trajectory.close();self.events_file.close()
