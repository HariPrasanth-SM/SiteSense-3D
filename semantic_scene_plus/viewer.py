from __future__ import annotations
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse,unquote
import json
import webbrowser
import numpy as np
from semantic_scene.viewer import SceneState,make_handler
from semantic_scene.common import read_json,sha,checked


class EnhancedState(SceneState):
    def __init__(self,args):
        self.extra=args.enhancement.expanduser().resolve();self.extra_config=read_json(self.extra/'config.json')
        super().__init__(self.extra_config['source_scene'],args.environment,args.max_env_points,args.max_region_points)
        if self.scene['scene_id']!=self.extra_config['source_scene_id'] or sha(self.out/'scene.json')!=self.extra_config['source_scene_sha256']:
            raise ValueError('Source scene changed; regenerate enhancement into a new folder')
        self.extras={}
        for rid in self.regions:
            p=self.extra/'regions'/rid/'complete.json'
            if p.exists():
                stamp=read_json(p)
                if stamp['enhancement_id']!=self.extra_config['enhancement_id']:raise ValueError('Mixed enhancement')
                self.extras[rid]=read_json(p.parent/'enhancement.json')
        self.splat_renderer=None
        if args.live_splats:
            from .splats import SplatRenderer
            self.splat_renderer=SplatRenderer(self.extra)

    def manifest(self):
        m=super().manifest();m['live_splats']=self.splat_renderer is not None
        m['enhanced_regions']=len(self.extras)
        for r in m['regions']:
            rid=r['region_id'];extra=self.extras.get(rid)
            r['raw_bbox']=r['bbox'];r['clean_bbox']=r['bbox'];r['gaussian_bbox']=None;r['enhancement']=extra
            if extra:
                r['clean_bbox']=extra['clean_bbox'];r['bbox']=extra['clean_bbox'];r['gaussian_bbox']=extra['gaussian_bbox']
            for j,v in enumerate(r['views']):
                v['original_url']=f'/source/{rid}/{j}.png'
                if extra:
                    v['overview_url']=f'/extra/{rid}/views/v{j:02d}_overlay.jpg'
                render=self.extra/'regions'/rid/'renders'/f'v{j:02d}.png'
                v['splat_url']=f'/extra/{rid}/renders/v{j:02d}.png' if render.exists() else None
        return m

    def cloud_bytes(self,rid,mode='all'):
        if rid=='environment' or mode in ('all','multi'):return super().cloud_bytes(rid,mode)
        if mode not in ('clean','removed','gaussians'):raise ValueError('Unknown geometry mode')
        if rid not in self.regions:raise KeyError('Unknown region')
        if rid not in self.extras:
            return super().cloud_bytes(rid,'all') if mode=='clean' else b''
        key=(rid,mode)
        with self.mutex:
            if key in self.buffer_cache:return self.buffer_cache[key]
            name={'clean':'surface_clean.npz','removed':'surface_removed.npz','gaussians':'gaussian_centers_map.npz'}[mode]
            base=self.extra/'regions'/rid;p=base/name
            if not p.exists():return b''
            stamp=read_json(base/'complete.json');checked(p,stamp['files'][name])
            with np.load(p,allow_pickle=False) as a:xyz=a['xyz'];rgb=a['rgb']
            if len(xyz)>self.max_region_points:
                ids=np.linspace(0,len(xyz)-1,self.max_region_points).astype(int);xyz=xyz[ids];rgb=rgb[ids]
            data=np.empty((len(xyz),6),'<f4');data[:,:3]=xyz;data[:,3:]=rgb.astype(np.float32)/255
            result=data.tobytes();self.buffer_cache[key]=result;return result


def enhanced_handler(state):
    Parent=make_handler(state);web=Path(__file__).parent/'web'
    class Handler(Parent):
        def do_GET(self):
            if not self.allowed_host():return self.json({'error':'Localhost required'},403)
            try:
                path=unquote(urlparse(self.path).path)
                if path=='/':return self.send((web/'index.html').read_bytes(),'text/html; charset=utf-8')
                if path=='/app.js':return self.send((web/'app.js').read_bytes(),'text/javascript; charset=utf-8')
                if path.startswith('/source/'):
                    parts=path.split('/');rid=parts[2];i=int(Path(parts[3]).stem)
                    if len(parts)!=4 or rid not in state.regions:raise ValueError('Invalid source image')
                    r=read_json(state.out/'regions'/rid/'region.json')
                    if not 0<=i<len(r['views']):raise ValueError('View index out of range')
                    # Only the source RGB bound to this known observation, not a user-supplied path.
                    return self.send((Path(r['views'][i]['source_frame'])/'rgb.png').read_bytes(),'image/png')
                if path.startswith('/extra/'):
                    parts=path.split('/');rid=parts[2];rel='/'.join(parts[3:])
                    if rid not in state.extras:raise ValueError('No enhancement for region')
                    base=(state.extra/'regions'/rid).resolve();p=(base/rel).resolve()
                    if base not in p.parents:raise PermissionError('Path rejected')
                    if len(parts)==5 and parts[3] in ('views','renders') and p.suffix in ('.png','.jpg'):
                        return self.send(p.read_bytes(),'image/png' if p.suffix=='.png' else 'image/jpeg')
                    allowed=('surface_clean.ply','surface_removed.ply','surface_clean.npz','gaussians_original.ply',
                             'gaussian_centers_map.npz','gaussian_indices.npy','enhancement.json')
                    if len(parts)==4 and rel in allowed:
                        return self.send(p.read_bytes(),'application/octet-stream',download=f'{rid}_{rel}')
                    raise PermissionError('Extra path rejected')
                return super().do_GET()
            except PermissionError as exc:return self.json({'error':str(exc)},403)
            except (ValueError,KeyError,OSError,IndexError) as exc:return self.json({'error':str(exc)},400)
        def do_POST(self):
            if self.path!='/api/render-splats':return super().do_POST()
            if not self.allowed_host():return self.json({'error':'Localhost required'},403)
            try:
                origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc!=self.headers.get('Host'):raise PermissionError('Cross-origin render refused')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=16384:raise ValueError('Invalid request length')
                body=json.loads(self.rfile.read(n))
                if body.get('token')!=state.token:raise PermissionError('Invalid token')
                rid=body['region_id']
                if rid not in state.regions:raise ValueError('Unknown region')
                if not state.splat_renderer:raise ValueError('Start viewer with --live-splats for GPU rendering')
                return self.json(state.splat_renderer.browser(rid,body))
            except PermissionError as exc:return self.json({'error':str(exc)},403)
            except Exception as exc:
                # Keep CPU browser available even after CUDA load/rasterization errors.
                print('SPLAT REQUEST ERROR:',type(exc).__name__,str(exc),flush=True)
                return self.json({'error':f'{type(exc).__name__}: {exc}'},400)
    return Handler


def serve(args):
    state=EnhancedState(args)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),enhanced_handler(state))
    url=f'http://127.0.0.1:{server.server_port}'
    print('Local enhanced region viewer:',url,flush=True)
    print('True splat rendering: '+('on-demand GPU button enabled' if args.live_splats else 'disabled; use exported/pre-rendered splats'),flush=True)
    if args.open_browser:webbrowser.open(url)
    try:server.serve_forever()
    finally:server.server_close()
