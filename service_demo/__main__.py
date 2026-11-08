import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='SITESENSE 3D — local spatial service dashboard')
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare',help='Read workbook, prepare local bindings; no GPU')
    a.add_argument('--out',default='outputs/service_demo/main_v1')
    a.add_argument('--workbook',help='Original XLSX. Omit to use the exact attached workbook snapshot bundled here.')
    a.add_argument('--map',default='outputs/ros/valmet_map.yaml')
    a.add_argument('--ply',default='data/Industrial Boiler Room.ply')
    a.add_argument('--regions',default='outputs/semantic_scene/valmet_regions150_v1')
    a.add_argument('--enhancement',default='outputs/semantic_enhance/valmet_regions150_v1')
    a.add_argument('--association',default='outputs/association/valmet_soft200_geometry_v1')
    a.add_argument('--repo',default='vendor/dinov3')
    a.add_argument('--weights',default='models/dinov3_vitb16_pretrain_lvd1689m.pth')
    a.add_argument('--demo-bindings',action='store_true',help='Explicitly assign synthetic workbook assets to scanned region hypotheses for this demo.')
    imp=sub.add_parser('import-labels',help='Join manual annotated tags to workbook records in a NEW bundle; preserve coordinates.')
    imp.add_argument('--base',default='outputs/service_demo/main_v1')
    imp.add_argument('--out',default='outputs/service_demo/manual_v1')
    imp.add_argument('--annotations',required=True)
    imp.add_argument('--labels-root',help='Folder containing the labeler objects/ directory; defaults to annotations parent.')
    b=sub.add_parser('serve',help='Start browser UI, navigation, optional GPU camera/search')
    b.add_argument('--out',default='outputs/service_demo/main_v1');b.add_argument('--port',type=int,default=8770)
    b.add_argument('--open-browser',action='store_true');b.add_argument('--no-render',action='store_true');b.add_argument('--no-search',action='store_true')
    b.add_argument('--width',type=int,default=480);b.add_argument('--height',type=int,default=360);b.add_argument('--render-hz',type=float,default=2)
    b.add_argument('--radius',type=float,default=0);b.add_argument('--margin',type=float,default=0)
    b.add_argument('--title',default='SITESENSE 3D',help='Dashboard project title (max 80 characters).')
    b.add_argument('--block-unknown',action='store_true',help='Optional conservative mode: block gray unknown cells. Default permits them for this demo only.')
    b.add_argument('--arrival-highlight',action='store_true',help='Optional one-shot FastSAM + DINOv3 arrival check; existing local checkpoints only.')
    b.add_argument('--fastsam',default='FastSAM-s.pt',help='Local FastSAM checkpoint; also searches models/ and weights/.')
    b.add_argument('--arrival-min-similarity',type=float,default=.72)
    b.add_argument('--arrival-distance',type=float,default=.30,help='Heuristic distance to stored target surface, provisional metres.')
    b.add_argument('--weights');b.add_argument('--repo');b.add_argument('--search-device',default='cuda:0',choices=['cuda:0','cpu'])
    c=sub.add_parser('check');c.add_argument('--out',default='outputs/service_demo/main_v1')
    args=p.parse_args()
    if args.cmd=='prepare':
        from .prepare import prepare
        prepare(args)
    elif args.cmd=='import-labels':
        from .manual_assets import import_labels
        import_labels(args)
    elif args.cmd=='serve':
        if not args.title.strip() or len(args.title)>80: p.error('Title must contain 1..80 characters.')
        if not .4<=args.arrival_min_similarity<=1 or not .02<=args.arrival_distance<=1: p.error('Invalid arrival thresholds.')
        if not 128<=args.width<=1920 or not 96<=args.height<=1080 or not .1<=args.render_hz<=10: p.error('Use width128..1920 height96..1080 fps0.1..10')
        from .server import serve
        serve(args)
    else:
        from .common import read_json
        from avatar_demo.core import FloorMap,Planner
        root=Path(args.out);cfg=read_json(root/'config.json');db=read_json(root/'workbook.json');b=read_json(root/'asset_bindings.json')
        m=FloorMap.load(cfg['map']);pl=Planner(m,0,0)
        from .manual_assets import ManualAssets
        manual=ManualAssets(root,cfg) if cfg.get('manual_assets') else None
        print(json.dumps({'assets':len(db['assets']),'mapped':sum(bool(a.get('map_xyz')) for a in b['assets'].values()),
                          'map_shape':m.shape,'start':pl.auto_start().tolist(),'source_counts':db['counts'],
                          'gaussian_ply_exists':Path(cfg['ply']).exists(),'catalog_exists':(Path(cfg['association'])/'catalog_complete.json').exists(),
                          'manual_objects':len(manual.items) if manual else 0,
                          'manual_crops_found':sum(manual.info(a)['crop_available'] for a in manual.items) if manual else None,
                          'travel_pitch_deg':0.0,'warning':cfg['warning']},indent=2))
if __name__=='__main__':main()
