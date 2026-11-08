"""Small cross-station appearance-only QA. Never asserts instance associations."""
from __future__ import annotations
from collections import defaultdict
from itertools import groupby
from pathlib import Path
import html
import json
import random
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .common import read_json,write_json,json_lines
from .preprocess import mask_crop,MEAN,STD,normalize_rows


def thumbnail(row,size=170):
    try:
        with Image.open(row['source_rgb']) as im: rgb=np.array(im.convert('RGB'))
        with np.load(row['source_mask_archive'],allow_pickle=False) as a:
            n,h,w=map(int,a['shape']); i=int(row['mask_index'])
            mask=np.unpackbits(a['bits'][i],count=h*w,bitorder='little').reshape(h,w).astype(bool)
        result=mask_crop(rgb,mask,256)
        if result is None: raise ValueError('Empty mask')
        rgb=np.clip((result[0].transpose(1,2,0)*STD+MEAN)*255,0,255).astype(np.uint8)
        return Image.fromarray(rgb).resize((size,size),Image.Resampling.LANCZOS)
    except (OSError,ValueError,KeyError,IndexError) as exc:
        im=Image.new('RGB',(size,size),(65,65,65));ImageDraw.Draw(im).text((5,5),f'Preview error\n{type(exc).__name__}',fill='white');return im


def make_report(out,queries=12,topk=3):
    out=Path(out);qa=out/'qa';qa.mkdir(exist_ok=True)
    summary=read_json(out/'summary.json');run=read_json(out/'run.json')
    rng=random.Random(42);chosen=[];seen=0
    # Reservoir of kept, non-tiny descriptors. Do not select review examples as seeds.
    for row in json_lines(out/'association_candidates.jsonl'):
        if row['crop_effective_patches']<2:continue
        seen+=1
        if len(chosen)<queries:chosen.append(row)
        else:
            i=rng.randrange(seen)
            if i<queries:chosen[i]=row
    vectors=[]
    for row in chosen:
        with np.load(out/row['feature_file'],allow_pickle=False) as a:
            vectors.append(a['crop_mask_pool'][row['feature_row']])
    if vectors: q,_=normalize_rows(np.stack(vectors))
    else: q=np.empty((0,run['model']['dimension']),np.float32)
    best=[{} for _ in chosen]
    if chosen:
        for fid,items in groupby(json_lines(out/'association_candidates.jsonl'),lambda r:r['frame_id']):
            rows=list(items)
            with np.load(out/rows[0]['feature_file'],allow_pickle=False) as a:
                feats,ok=normalize_rows(a['crop_mask_pool'][[r['feature_row'] for r in rows]])
            scores=q@feats.T
            for k,query in enumerate(chosen):
                if rows[0]['station_id']==query['station_id']:continue
                valid_ids=np.flatnonzero(ok)
                if not len(valid_ids): continue
                j=int(valid_ids[np.argmax(scores[k,valid_ids])]);score=float(scores[k,j])
                station=rows[j]['station_id']
                if station not in best[k] or score>best[k][station]['cosine']:
                    best[k][station]={'cosine':score,'observation':rows[j]}
                # retain only the strongest distinct stations; no NxN similarity matrix.
                if len(best[k])>topk:
                    loser=min(best[k],key=lambda s:best[k][s]['cosine']);best[k].pop(loser)
    results=[]
    for row,b in zip(chosen,best):
        results.append({'query':row,'neighbors':sorted(b.values(),key=lambda r:-r['cosine']),
                        'same_instance_verified':False,'matching_rule':'appearance only; same station excluded'})
    write_json(qa/'neighbors.json',results)
    write_json(qa/'summary.json',summary)
    sheets=[]
    try: font=ImageFont.truetype('DejaVuSans.ttf',11)
    except OSError: font=ImageFont.load_default()
    tilew,tileh=230,230
    for start in range(0,len(results),4):
        chunk=results[start:start+4]
        sheet=Image.new('RGB',((topk+1)*tilew,len(chunk)*tileh),(23,28,35));draw=ImageDraw.Draw(sheet)
        for ri,entry in enumerate(chunk):
            cells=[{'observation':entry['query'],'cosine':None}]+entry['neighbors']
            for ci,cell in enumerate(cells):
                row=cell['observation'];x=ci*tilew;y=ri*tileh
                sheet.paste(thumbnail(row),(x+30,y+3))
                score='QUERY — keep only' if ci==0 else f"Similarity {cell['cosine']:.3f} — NOT identity"
                text=f"{score}\n{row['frame_id']}  m{row['mask_index']:04d}\nStation {row['station_id']} | {row['mask_status']}/{row['frame_status']}"
                draw.multiline_text((x+6,y+177),text,font=font,fill=(235,235,235),spacing=3)
        name=f'neighbors_{start//4+1:03d}.jpg';sheet.save(qa/name,quality=92);sheets.append(name)
    counts='\n'.join(f'{k}: {v}' for k,v in summary['counts'].items())
    imgs='\n'.join(f'<a href="{n}"><img src="{n}" style="max-width:100%"></a>' for n in sheets)
    if not sheets: imgs='<p>No valid strict-keep queries available yet. Complete more frames before judging retrieval.</p>'
    page=f'''<!doctype html><meta charset="utf-8"><title>Valmet DINOv3 feature QA</title>
<style>body{{max-width:1200px;margin:30px auto;background:#111827;color:#eee;font:16px system-ui}}a{{color:#93c5fd}}pre{{padding:15px;background:#1f2937}}</style>
<h1>DINOv3 observation archive</h1><p>Appearance-only diagnostic. Neighbors are not verified object identities.
No labels, merges, point ownership, or navigation goals have been assigned.</p>
<p>Only frame=keep AND mask=keep observations are used below. Query station is excluded from neighbors;
results from different stations may still depict different but similar machines.</p>
<pre>{html.escape(counts)}</pre><p>Model: {html.escape(run['model']['backbone'])}; descriptor: crop_mask_pool.</p>
{imgs}<p>Inspect original images and geometry before any association. The query pool is limited to completed feature frames.</p>'''
    (qa/'index.html').write_text(page)
    print('Feature review:',qa/'index.html',flush=True)
    return results
