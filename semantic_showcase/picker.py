"""Local-only inspection and ordered selection. Never edits geometry or labels."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import secrets
import time
import webbrowser
from urllib.parse import urlparse,parse_qs
import numpy as np
from .common import *


def serve(args):
    base,p=base_plan(args.base);destination=args.out.expanduser().resolve()
    if destination==base/'plan.json':raise ValueError('Selection output cannot replace the base plan.')
    table={r['region_id']:r for r in p['regions']};token=secrets.token_urlsafe(24)
    cloud_cache={}
    def manifest():
        saved=[]
        if destination.exists():
            _,saved=selection_ids(destination,p)
        rows=[]
        for r in p['regions']:
            rows.append({'id':r['region_id'],'display_id':r['display_id'],'label':r['display_label'],
                'gaussians':r['gaussians'],'point_count':r['point_count'],'extent':r['extent'],
                'structural':r.get('structural',False),'views':[{'frame_id':v.get('frame_id'),
                    'mask_index':v.get('mask_index'),'station_id':v.get('station_id'),
                    'kinds':[k for k in ['masked_crop','context_crop','overlay','overview'] if v.get(k)]} for v in r.get('evidence',[])]})
        return {'regions':rows,'selected_ids':saved,'base_plan_id':p['plan_id'],'base':str(base),
                'destination':str(destination),'token':token,'base_duration':p['frame_count']/p['fps']}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*values):pass
        def send(self,data,ctype='application/json',status=200):
            if isinstance(data,(dict,list)):data=json.dumps(data,allow_nan=False).encode()
            elif isinstance(data,str):data=data.encode()
            self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            try:
                u=urlparse(self.path);q=parse_qs(u.query)
                if u.path=='/':return self.send((Path(__file__).parent/'static/picker.html').read_bytes(),'text/html; charset=utf-8')
                if u.path=='/api/catalog':return self.send(manifest())
                if u.path in ['/api/image','/api/cloud']:
                    rid=q.get('id',[''])[0]
                    if rid not in table:raise ValueError('Unknown region ID')
                    r=table[rid]
                    if u.path=='/api/image':
                        v=int(q.get('view',['0'])[0]);kind=q.get('kind',['masked_crop'])[0]
                        if kind not in ['masked_crop','context_crop','overlay','overview']:raise ValueError('Unknown evidence kind')
                        if not 0<=v<len(r.get('evidence',[])):raise ValueError('Missing evidence view')
                        name=r['evidence'][v].get(kind)
                        if not name:raise ValueError('Missing evidence image')
                        path=checked_evidence(base,r,name)
                        return self.send(path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                    if rid not in cloud_cache:
                        path=checked_asset(base,r,'display_points_map.npz');xyz,rgb=read_cloud(path)
                        if len(xyz)>8000:
                            ix=np.linspace(0,len(xyz)-1,8000,dtype=int);xyz=xyz[ix];rgb=rgb[ix]
                        cloud_cache[rid]={'xyz':xyz.tolist(),'rgb':rgb.tolist(),'center':r['center']}
                        if len(cloud_cache)>6:cloud_cache.pop(next(iter(cloud_cache)))
                    return self.send(cloud_cache[rid])
                return self.send({'error':'Not found'},status=404)
            except (ValueError,KeyError,FileNotFoundError,IndexError) as e:self.send({'error':str(e)},status=400)
            except BrokenPipeError:pass
        def do_POST(self):
            try:
                if urlparse(self.path).path!='/api/save':return self.send({'error':'Not found'},status=404)
                if self.headers.get('X-Showcase-Token')!=token:return self.send({'error':'Invalid session token'},status=403)
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<100000:raise ValueError('Invalid request size')
                ids=json.loads(self.rfile.read(length)).get('region_ids')
                if not isinstance(ids,list) or not 1<=len(ids)<=40 or len(ids)!=len(set(ids)) or any(x not in table for x in ids):
                    raise ValueError('Choose 1–40 unique valid IDs; 10–20 recommended.')
                record={'schema':'valmet.showcase_selection.v1','base':str(base),'base_plan_id':p['plan_id'],
                        'region_ids':ids,'order':'column-major pairs: top-left,bottom-left,top-second,bottom-second',
                        'saved_at_unix':time.time(),'labels_snapshot':{r:table[r]['display_label'] for r in ids}}
                if destination.exists():
                    backup=destination.with_name(destination.stem+f'.backup_{time.time_ns()}'+destination.suffix)
                    backup.write_bytes(destination.read_bytes())
                write_json(destination,record)
                print(f'Saved {len(ids)} selected IDs to {destination}',flush=True)
                return self.send({'saved':str(destination),'count':len(ids)})
            except (ValueError,TypeError,KeyError,OSError) as e:self.send({'error':str(e)},status=400)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Picker: http://127.0.0.1:{args.port}\nOutput: {destination}\nNo geometry is edited. Keep this terminal open.',flush=True)
    if args.open_browser:webbrowser.open(f'http://127.0.0.1:{args.port}')
    try:server.serve_forever()
    finally:server.server_close()
