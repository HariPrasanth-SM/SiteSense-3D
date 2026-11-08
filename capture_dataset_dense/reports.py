"""Extra height-aware review pages; uses the unchanged base index and raw frame files."""
from __future__ import annotations
import csv
import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from capture_dataset.common import load_plan, read_json


def add_dense_reports(out):
    out = Path(out)
    plan = load_plan(out)
    # Base report has already checked completion hashes and creates this index.
    records = [__import__('json').loads(x) for x in (out/'index.jsonl').read_text().splitlines() if x.strip()]
    records = [r for r in records if r['complete']]
    # Put pilot views first so a full run's first pages remain useful for diagnostics.
    records.sort(key=lambda v: (not v['pilot'], v['station_id'], v.get('view_index', 0)))
    qa_dir = out/'qa'; qa_dir.mkdir(exist_ok=True)
    rows = []
    for v in records:
        q = read_json(out/'frames'/v['frame_id']/'qa.json')
        rows.append({k:v.get(k) for k in ['frame_id','station_id','camera_height_m','yaw_deg','pitch_deg',
                                        'x','y','pilot','unknown_capture_only','corridor_id']} |
                    {'valid_fraction':q['valid_fraction'], 'flags':';'.join(q['flags'])})
    fields = ['frame_id','station_id','camera_height_m','yaw_deg','pitch_deg','x','y','pilot',
              'unknown_capture_only','corridor_id','valid_fraction','flags']
    with (out/'quality_dense.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    pages=[]
    for first in range(0,len(records),36):
        group=records[first:first+36]; columns=6; tw,th,lh=200,150,50
        for mode in ('rgb','depth'):
            sheet=Image.new('RGB',(columns*tw,math.ceil(len(group)/columns)*(th+lh)),(28,34,42))
            draw=ImageDraw.Draw(sheet)
            for j,v in enumerate(group):
                folder=out/'frames'/v['frame_id']
                if mode=='rgb':
                    with Image.open(folder/'rgb.png') as im: thumb=im.convert('RGB')
                else:
                    d=np.load(folder/'depth_m.npy',allow_pickle=False)
                    with Image.open(folder/'valid.png') as im: valid=np.array(im)>0
                    arr=np.rint(np.nan_to_num(np.where(valid,1-np.clip(d/8,0,1),0))*255).astype(np.uint8)
                    thumb=Image.fromarray(arr).convert('RGB')
                thumb.thumbnail((tw,th))
                x=(j%columns)*tw;y=(j//columns)*(th+lh)
                sheet.paste(thumb,(x+(tw-thumb.width)//2,y+(th-thumb.height)//2))
                draw.text((x+3,y+th+2),v['frame_id']+(' PILOT' if v['pilot'] else ''),fill='white')
                draw.text((x+3,y+th+17),f"h {v['camera_height_m']:.2f}m  yaw {v['yaw_deg']:.0f}  p {v['pitch_deg']:.0f}",fill='white')
                draw.text((x+3,y+th+32),('GRAY CAPTURE ONLY' if v['unknown_capture_only'] else 'known-free camera XY'),fill='white')
            name=f'dense_{mode}_{first//36+1:03d}.jpg'
            sheet.save(qa_dir/name,quality=90);pages.append(name)
    links='\n'.join(f'<p>{n}</p><a href="{n}"><img loading="lazy" src="{n}" style="max-width:100%"></a>' for n in pages)
    (qa_dir/'dense.html').write_text(
        "<!doctype html><meta charset='utf-8'><title>Dense capture review</title>"
        "<h1>Multi-height capture review</h1>"
        f"<p>{len(records)}/{len(plan['frames'])} frames complete. Pilot frames first. "
        "h = provisional camera height; p = central pitch (negative down). "
        "Depth display 0-8 provisional m. Inspect high/low cameras for intersections.</p>"+links)
    print(f"Height-aware review: {qa_dir/'dense.html'}",flush=True)
