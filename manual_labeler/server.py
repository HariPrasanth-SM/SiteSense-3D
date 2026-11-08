from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import secrets
from urllib.parse import urlsplit, parse_qs
import zipfile
import io
from .storage import clean_json


def make_server(engine,port):
    root=Path(__file__).parent/'web'; token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def guard(self,write=False):
            host=self.headers.get('Host','')
            if host not in (f'127.0.0.1:{port}',f'localhost:{port}'):
                self.send_error(403,'Localhost requests only');return False
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
                self.send_error(403,'Cross-origin request refused');return False
            if write and not secrets.compare_digest(self.headers.get('X-Label-Token',''),token):
                self.send_error(403,'Session token required');return False
            return True
        def send(self,status,data,content_type='application/json; charset=utf-8',filename=None):
            if not isinstance(data,bytes):data=json.dumps(clean_json(data),allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            if not self.guard():return
            path=urlsplit(self.path);q=parse_qs(path.query)
            try:
                if path.path=='/':return self.send(200,(root/'index.html').read_bytes(),'text/html; charset=utf-8')
                if path.path in ('/app.js','/style.css'):
                    f=root/path.path[1:];return self.send(200,f.read_bytes(),'text/javascript' if f.suffix=='.js' else 'text/css')
                if path.path=='/api/init':return self.send(200,{'token':token,**engine.state()})
                if path.path=='/api/state':return self.send(200,engine.state())
                if path.path=='/api/map.png':return self.send(200,engine.map_png,'image/png')
                if path.path=='/api/frame':return self.send(200,engine.frame_bytes(q.get('id',[''])[0],q.get('kind',['rgb'])[0]),'image/png')
                if path.path=='/api/draft.png':return self.send(200,engine.draft_image(),'image/png')
                if path.path=='/api/annotations':
                    return self.send(200,(engine.store.root/'annotations.json').read_bytes(),'application/json','annotations.json')
                if path.path=='/api/output':
                    rel=q.get('path',[''])[0];p=(engine.store.root/rel).resolve()
                    if not p.is_relative_to(engine.store.root) or p.suffix.lower() not in ('.png','.jpg','.json','.csv','.ply','.npz') or not p.is_file():
                        raise ValueError('Output not available.')
                    return self.send(200,p.read_bytes(),mimetypes.guess_type(str(p))[0] or 'application/octet-stream')
                self.send_error(404)
            except (ValueError,KeyError,FileNotFoundError) as exc:self.send(400,{'error':str(exc)})
            except Exception as exc:
                import traceback;traceback.print_exc();self.send(500,{'error':f'{type(exc).__name__}: {exc}'})
        def do_POST(self):
            if not self.guard(True):return
            if urlsplit(self.path).path!='/api/command':return self.send_error(404)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1024*1024:raise ValueError('Invalid request size.')
                cmd=json.loads(self.rfile.read(size));result=engine.command(cmd);self.send(200,result)
            except (ValueError,KeyError,TypeError) as exc:self.send(400,{'error':str(exc)})
            except Exception as exc:
                import traceback;traceback.print_exc();self.send(500,{'error':f'{type(exc).__name__}: {exc}'})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server
