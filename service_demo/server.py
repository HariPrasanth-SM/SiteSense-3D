from __future__ import annotations
import json, mimetypes, webbrowser
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,parse_qs,unquote
from .engine import AppState
from .common import jsonable

def serve(args):
    state=AppState(args);web=Path(__file__).parent/'web'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def send(self,data,typ='application/json',status=200):
            self.send_response(status);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; connect-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def js(self,obj,status=200):self.send(json.dumps(jsonable(obj),allow_nan=False).encode(),'application/json',status)
        def host(self):
            if self.headers.get('Host') not in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}'):raise PermissionError('Loopback Host required')
        def do_GET(self):
            try:
                self.host();u=urlparse(self.path);path=unquote(u.path);q=parse_qs(u.query)
                if path in ('/','/app.js','/style.css'):
                    p=web/('index.html' if path=='/' else path[1:]);return self.send(p.read_bytes(),mimetypes.guess_type(str(p))[0] or 'text/plain')
                if path=='/api/bootstrap':return self.js(state.bootstrap())
                if path=='/api/state':return self.js(state.snapshot())
                if path=='/map.png':return self.send(state.map_png(),'image/png')
                if path.startswith('/api/asset/'):
                    return self.js(state.asset_detail(path.split('/')[-1]))
                if path.startswith('/manual-asset/'):
                    if state.manual_assets is None:raise ValueError('No manual annotations loaded.')
                    parts=path.split('/')
                    if len(parts)!=4:raise ValueError('Invalid manual media request.')
                    data,typ=state.manual_assets.media(parts[2],parts[3]);return self.send(data,typ)
                if path.startswith('/api/region/'):
                    rid=path.split('/')[-1];return self.js(state.regions.detail(rid))
                if path.startswith('/cloud/'):
                    return self.js(state.regions.cloud(path.split('/')[-1]))
                if path.startswith('/region/'):
                    parts=path.split('/');rid=parts[2];i=int(parts[3]);kind=parts[4]
                    if len(parts)!=5 or i<0:raise ValueError('Invalid media request')
                    data,typ=state.regions.media(rid,i,kind);return self.send(data,typ)
                if path.startswith('/match/'):
                    if state.search_index is None:raise ValueError('Search index not ready')
                    parts=path.split('/');idx=int(parts[2]);kind=parts[3]
                    if len(parts)!=4 or not 0<=idx<len(state.search_index.catalog.rows):raise ValueError('Invalid match')
                    return self.send(state.search_index.media(idx,kind),'image/jpeg')
                if path.startswith('/api/query/'):
                    return self.js(state.query_page(path.split('/')[-1],q.get('offset',[0])[0]))
                if path in ('/camera/highlight.jpg','/camera/target_crop.png'):
                    kind='highlight' if path.endswith('highlight.jpg') else 'target_crop'
                    data=state.camera_image(kind)
                    return self.send(data or b'','image/jpeg' if kind=='highlight' else 'image/png',200 if data else 204)
                if path in ('/camera/rgb.jpg','/camera/depth.jpg','/target/rgb.jpg','/target/depth.jpg'):
                    parts=path.split('/');bank=state.frame_bytes if parts[1]=='camera' else state.target_bytes
                    data=bank.get(parts[2].split('.')[0])
                    if not data:return self.send(b'','image/jpeg',204)
                    return self.send(data,'image/jpeg')
                if path=='/api/visit-brief':
                    with state.lock:
                        s=state.selected;aid=s.get('asset_id') if s else None
                        result={'selected':s,'navigation':state.snapshot(),'asset':state.asset_detail(aid) if aid else None,
                                'warning':'Manual scene positions with synthetic workbook data joined by user tag; not verified physical asset identity. Draft only; no work order or equipment action occurs.' if state.manual_assets else 'Synthetic records and demo mapping. Draft only; no work order or equipment action occurs.'}
                    return self.js(result)
                if path=='/favicon.ico':return self.send(b'','image/x-icon',204)
                return self.js({'error':'Not found'},404)
            except PermissionError as e:return self.js({'error':str(e)},403)
            except Exception as e:return self.js({'error':f'{type(e).__name__}: {e}'},400)
        def do_POST(self):
            try:
                self.host();origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc!=self.headers.get('Host'):raise PermissionError('Cross-origin request refused')
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=18*1024**2:raise ValueError('Invalid request size')
                b=json.loads(self.rfile.read(n))
                if b.get('token')!=state.token:raise PermissionError('Invalid local session token')
                if self.path=='/api/action':return self.js(state.command(b))
                if self.path=='/api/search':return self.js(state.upload(b))
                return self.js({'error':'Not found'},404)
            except PermissionError as e:return self.js({'error':str(e)},403)
            except Exception as e:return self.js({'error':f'{type(e).__name__}: {e}'},400)
    try:server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    except Exception:state.close();raise
    server.daemon_threads=True
    url=f'http://127.0.0.1:{server.server_port}'
    print('\n'+state.title+':',url,'\nSession:',state.session,'\nUse this one local server. No ROS, Gazebo, cloud service or API key needed.',flush=True)
    if args.open_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();state.close()
