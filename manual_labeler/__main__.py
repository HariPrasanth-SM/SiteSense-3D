from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import sys
import threading
import webbrowser
import zipfile
import yaml
from .storage import sha, atomic_json


def parser():
    p=argparse.ArgumentParser(description='Local joystick navigation and manually labeled 3D object boxes. No asset workbook mapping.')
    p.add_argument('command',nargs='?',choices=['serve','export'],default='serve')
    p.add_argument('--bundle',default='outputs/service_demo/main_v1',help='Read source paths/scale from existing demo config; never modifies it.')
    p.add_argument('--map',default=None);p.add_argument('--scene',default=None)
    p.add_argument('--out',default='outputs/manual_labels/scene_v1')
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=480)
    p.add_argument('--render-hz',type=float,default=2);p.add_argument('--fov',type=float,default=70)
    p.add_argument('--camera-height',type=float,default=1.7);p.add_argument('--speed',type=float,default=.25)
    p.add_argument('--start',type=float,nargs=2,metavar=('X','Y'));p.add_argument('--yaw',type=float,default=90)
    p.add_argument('--sh-degree',type=int,default=3,choices=[0,1,2,3])
    p.add_argument('--port',type=int,default=8772);p.add_argument('--open-browser',action='store_true')
    p.add_argument('--no-render',action='store_true',help='Map/control smoke test; no images or labeling.')
    p.add_argument('--snapshot-prefix',help='Label an existing *_rgb.png, *_depth.npy, *_alpha.npy, *_camera.json set without CUDA.')
    p.add_argument('--zip',dest='zip_path',default=None,help='Output ZIP for export command.')
    return p


def main():
    args=parser().parse_args();out=Path(args.out).expanduser().resolve()
    if args.command=='export':
        if not (out/'annotations.json').is_file():raise ValueError('No annotations.json in --out.')
        dest=Path(args.zip_path or str(out)+'_export.zip').expanduser().resolve()
        if dest.is_relative_to(out):raise ValueError('ZIP must be outside the annotations directory.')
        lockpath=out/'.labeler.lock';lockpath.touch(exist_ok=True)
        with lockpath.open('r+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Stop the labeling server before exporting the complete bundle.')
            if dest.exists():raise ValueError('Export already exists. Choose a new --zip path.')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                for path in sorted(out.rglob('*')):
                    if path.is_file() and not path.name.startswith('.') and not path.is_symlink():
                        z.write(path,Path(out.name)/path.relative_to(out))
        print('Export saved:',dest);return
    if not 160<=args.width<=1920 or not 120<=args.height<=1080:raise ValueError('Use width 160–1920 and height 120–1080.')
    if not .1<=args.render_hz<=15:raise ValueError('Render rate must be 0.1–15 Hz.')
    if not .1<=args.camera_height<=4 or not .02<=args.speed<=2 or not 20<=args.fov<=120:raise ValueError('Invalid camera height, speed or FOV.')
    if not 1024<=args.port<=65535:raise ValueError('Use port 1024–65535.')
    cfgpath=Path(args.bundle).expanduser()/'config.json'
    base=json.loads(cfgpath.read_text()) if cfgpath.is_file() else {}
    mapfile=Path(args.map or base.get('map','outputs/ros/valmet_map.yaml')).expanduser().resolve()
    scene=Path(args.scene or base.get('ply','data/Industrial Boiler Room.ply')).expanduser().resolve()
    scale=float(base.get('scale',1.43));floor_y=float(base.get('floor_y',1.1))
    if not mapfile.is_file():raise ValueError(f'Map not found: {mapfile}')
    if not scene.is_file() and not args.snapshot_prefix and not args.no_render:raise ValueError(f'Original Gaussian scene not found: {scene}')
    cfg=yaml.safe_load(mapfile.read_text());img=Path(cfg['image']).expanduser()
    if not img.is_absolute():img=mapfile.parent/img
    config={'map':str(mapfile),'scene':str(scene),'out':str(out),'scale':scale,'floor_y':floor_y,
            'width':args.width,'height':args.height,'render_hz':args.render_hz,'fov':args.fov,
            'camera_height':args.camera_height,'speed':args.speed,'start':args.start,'yaw':args.yaw,
            'sh_degree':args.sh_degree,'no_render':args.no_render,'snapshot_prefix':args.snapshot_prefix}
    identity={'schema':'sitesense.manual_label_source.v1','scene':str(scene),'scene_sha256':sha(scene) if scene.is_file() else None,
              'map_yaml':str(mapfile),'map_image':str(img.resolve()),'map_yaml_sha256':sha(mapfile),'map_image_sha256':sha(img),
              'scale':scale,'floor_y':floor_y,'coordinate_frame':'map','units':'provisional_metres',
              'transform':'x_map=s*x_GS; y_map=s*z_GS; z_map=s*(floor_y-y_GS)',
              'original_demo_config_read_only':str(cfgpath.resolve()) if cfgpath.is_file() else None}
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.labeler.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('Another labeler is writing this output. Stop it first.')
    from .engine import LabelEngine
    from .server import make_server
    engine=None;server=None
    try:
        engine=LabelEngine(config,identity);server=make_server(engine,args.port)
        print(f'Existing transform retained: scale={scale}, floor_y={floor_y}. No Excel anchors or axis/scale changes.',flush=True)
        print(f'Labels: {out}\nLocal viewer: http://127.0.0.1:{args.port}\nStop with Ctrl+C. No source files are modified.',flush=True)
        if args.open_browser:threading.Timer(.5,lambda:webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:print('\nStopping manual labeler…',flush=True)
    finally:
        if server:server.server_close()
        if engine:engine.close()
        lock.close()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr);sys.exit(1)
