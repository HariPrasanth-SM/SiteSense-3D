"""Loopback-only, dependency-free HTTP + browser WebGL viewer. No remote uploads."""
from __future__ import annotations
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,unquote,parse_qs
from datetime import datetime,timezone
import json
import secrets
import threading
import webbrowser
import numpy as np
from .common import (read_json,write_json,load_scene,load_review,load_suggestions,checked,lock)


class SceneState:
    def __init__(self,out,environment=None,max_env_points=200000,max_region_points=15000):
        self.out,self.scene=load_scene(out);self.token=secrets.token_urlsafe(32);self.mutex=threading.RLock()
        self.regions={r['region_id']:r for r in self.scene['regions']};self.buffer_cache={}
        self.max_region_points=max_region_points
        if max_region_points<1 or max_env_points<1:raise ValueError('Point caps must be positive')
        self.env={'xyz':np.zeros((0,3),np.float32),'rgb':np.zeros((0,3),np.uint8)}
        if environment:
            with np.load(Path(environment).expanduser(),allow_pickle=False) as a:
                xyz=np.asarray(a['xyz'],np.float32);rgb=np.asarray(a['rgb'])
            if xyz.ndim!=2 or xyz.shape[1]!=3 or rgb.shape!=xyz.shape:raise ValueError('Environment must contain xyz and rgb arrays with shape (N,3)')
            valid=np.isfinite(xyz).all(axis=1);xyz=xyz[valid];rgb=rgb[valid]
            if np.issubdtype(rgb.dtype,np.floating) and len(rgb) and rgb.max()<=1:rgb=rgb*255
            rgb=np.clip(rgb,0,255).astype(np.uint8)
            if len(xyz)>max_env_points:
                idx=np.sort(np.random.default_rng(42).choice(len(xyz),max_env_points,replace=False));xyz=xyz[idx];rgb=rgb[idx]
            self.env={'xyz':xyz,'rgb':rgb}
        self.environment_path=str(environment) if environment else None

    def manifest(self):
        with self.mutex:
            review=load_review(self.out,self.scene);suggest=load_suggestions(self.out,self.scene)
            regs=[]
            for rid,r in self.regions.items():
                folder=self.out/'regions'/rid
                record=read_json(folder/'region.json')
                item=dict(r);item['review']=review['regions'].get(rid,{'revision':0,'semantic_label':None,'kind':'unknown','decision':'unreviewed','note':''})
                item['suggestion']=suggest.get('regions',{}).get(rid)
                item['views']=[{**v,'masked_url':f'/media/{rid}/{v["masked_crop"]}',
                                'context_url':f'/media/{rid}/{v["context_crop"]}',
                                'overview_url':f'/media/{rid}/{v["overview"]}'} for v in record['views']]
                regs.append(item)
            points=self.env['xyz']
            if len(points):lo,hi=np.percentile(points,[1,99],axis=0)
            else:
                lo=np.min([r['bbox']['min'] for r in regs],axis=0);hi=np.max([r['bbox']['max'] for r in regs],axis=0)
            return {'scene_id':self.scene['scene_id'],'token':self.token,'regions':regs,
                    'environment_points':len(points),'bounds':[lo.tolist(),hi.tolist()],
                    'scale':self.scene['scale_m_per_scene_unit'],'suggestions_available':bool(suggest.get('regions'))}

    def cloud_bytes(self,rid,mode='all'):
        if mode not in ('all','multi'):raise ValueError('Invalid cloud mode')
        key=(rid,mode)
        with self.mutex:
            if key in self.buffer_cache:return self.buffer_cache[key]
            if rid=='environment':xyz=self.env['xyz'];rgb=self.env['rgb']
            else:
                if rid not in self.regions:raise KeyError('Unknown region')
                folder=self.out/'regions'/rid
                stamp=read_json(folder/'complete.json');checked(folder/'surface.npz',stamp['files']['surface.npz'])
                with np.load(folder/'surface.npz',allow_pickle=False) as a:
                    m=np.ones(len(a['xyz']),bool) if mode=='all' else a['station_count']>=2
                    xyz=a['xyz'][m];rgb=a['rgb'][m]
                if len(xyz)>self.max_region_points:
                    idx=np.linspace(0,len(xyz)-1,self.max_region_points).astype(int);xyz=xyz[idx];rgb=rgb[idx]
            a=np.empty((len(xyz),6),dtype='<f4');a[:,:3]=xyz;a[:,3:]=np.asarray(rgb,dtype=np.float32)/255
            result=a.tobytes();self.buffer_cache[key]=result;return result

    def review(self,body):
        if body.get('token')!=self.token:raise PermissionError('Invalid edit token')
        rid=body.get('region_id')
        if rid not in self.regions:raise ValueError('Unknown region ID')
        label=body.get('semantic_label');note=body.get('note','');decision=body.get('decision','unreviewed');kind=body.get('kind','unknown')
        if label is not None and (not isinstance(label,str) or len(label)>100):raise ValueError('Label must be a short string or null')
        label=(label.strip() or None) if label is not None else None
        if not isinstance(note,str) or len(note)>2000:raise ValueError('Note too long')
        if decision not in ('unreviewed','accepted','rejected'):raise ValueError('Invalid review decision')
        if kind not in ('unknown','equipment','part','structure','other'):raise ValueError('Invalid type')
        with self.mutex,lock(self.out):
            data=load_review(self.out,self.scene);previous=data['regions'].get(rid,{'revision':0})
            if body.get('expected_revision',0)!=previous['revision']:raise RuntimeError('This region was edited elsewhere. Reload before saving.')
            entry={'revision':previous['revision']+1,'semantic_label':label,'kind':kind,'decision':decision,'note':note,
                   'label_source':'human_review' if label else None,'updated_utc':datetime.now(timezone.utc).isoformat(),
                   'geometry_is_still_tentative':True}
            data['regions'][rid]=entry;write_json(self.out/'review.json',data)
            with (self.out/'review_history.jsonl').open('a') as f:
                f.write(json.dumps({'scene_id':self.scene['scene_id'],'region_id':rid,'entry':entry},allow_nan=False)+'\n')
            return entry


def make_handler(state):
    web=Path(__file__).with_name('web')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def send(self,data,content_type,status=200,download=None):
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; connect-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
            if download:self.send_header('Content-Disposition',f'attachment; filename="{download}"')
            self.end_headers();self.wfile.write(data)
        def json(self,obj,status=200):self.send(json.dumps(obj,allow_nan=False).encode(),'application/json',status)
        def allowed_host(self):
            return self.headers.get('Host','') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def do_GET(self):
            if not self.allowed_host():return self.json({'error':'Localhost Host header required'},403)
            try:
                parsed=urlparse(self.path);path=unquote(parsed.path)
                if path=='/':return self.send((web/'index.html').read_bytes(),'text/html; charset=utf-8')
                if path=='/app.js':return self.send((web/'app.js').read_bytes(),'text/javascript; charset=utf-8')
                if path=='/api/scene':return self.json(state.manifest())
                if path.startswith('/api/cloud/') and path.endswith('.bin'):
                    rid=path[len('/api/cloud/'):-4];mode=parse_qs(parsed.query).get('mode',['all'])[0]
                    return self.send(state.cloud_bytes(rid,mode),'application/octet-stream')
                if path.startswith('/media/'):
                    parts=path.split('/');rid=parts[2];rel='/'.join(parts[3:])
                    if rid not in state.regions:raise KeyError('Unknown region')
                    base=(state.out/'regions'/rid/'views').resolve();p=(state.out/'regions'/rid/rel).resolve()
                    if p.parent!=base or p.suffix.lower() not in ('.jpg','.png'):raise PermissionError('Media path rejected')
                    return self.send(p.read_bytes(),'image/png' if p.suffix=='.png' else 'image/jpeg')
                if path.startswith('/download/'):
                    parts=path.split('/');rid=parts[2];name=parts[3]
                    if len(parts)!=4 or rid not in state.regions or name not in ('surface.ply','multiview_voxels.ply','region.json'):
                        raise PermissionError('Download path rejected')
                    return self.send((state.out/'regions'/rid/name).read_bytes(),'application/octet-stream',download=f'{rid}_{name}')
                if path=='/favicon.ico':return self.send(b'','image/x-icon',204)
                return self.json({'error':'Not found'},404)
            except PermissionError as e:return self.json({'error':str(e)},403)
            except (KeyError,OSError,ValueError,IndexError) as e:return self.json({'error':str(e)},400)
        def do_POST(self):
            if not self.allowed_host():return self.json({'error':'Localhost Host header required'},403)
            try:
                if self.path!='/api/review':return self.json({'error':'Not found'},404)
                origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc!=self.headers.get('Host'):raise PermissionError('Cross-origin editing refused')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=16384:raise ValueError('Invalid body length')
                body=json.loads(self.rfile.read(n));return self.json(state.review(body))
            except PermissionError as e:return self.json({'error':str(e)},403)
            except RuntimeError as e:return self.json({'error':str(e)},409)
            except (KeyError,OSError,ValueError) as e:return self.json({'error':str(e)},400)
    return Handler


def serve(args):
    state=SceneState(args.scene,args.environment,args.max_env_points,args.max_region_points)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(state))
    url=f'http://127.0.0.1:{server.server_port}'
    print('Local 3D review:',url,flush=True)
    print('No ROS, GPU model, cloud upload, CDN or public share link. Keep this terminal running.',flush=True)
    if args.open_browser:webbrowser.open(url)
    try:server.serve_forever()
    finally:server.server_close()
