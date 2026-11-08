"""Local static reports: original crops and reprojection evidence, not model overlays as input."""
from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import html
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import read_json, write_json

BG=(25,31,41); FG=(225,230,237)
STATUS={'supported': (104,211,151), 'partial_overlap':(242,191,85),
        'inconclusive':(144,171,206), 'rejected':(242,126,126)}
FONT=ImageFont.load_default()


class Pictures:
    def __init__(self, catalog):
        self.cat=catalog;self.cache=OrderedDict()
    def frame(self, fid):
        if fid in self.cache:
            self.cache.move_to_end(fid);return self.cache[fid]
        ds=Path(self.cat.source['dataset']);seg=Path(self.cat.source['segmentation'])
        with Image.open(ds/'frames'/fid/'rgb.png') as im: rgb=np.array(im.convert('RGB'))
        with np.load(seg/'frames'/fid/'masks.npz',allow_pickle=False) as a:
            bits=a['bits'].copy();shape=tuple(map(int,a['shape']))
        self.cache[fid]=(rgb,bits,shape)
        while len(self.cache)>5:self.cache.popitem(last=False)
        return self.cache[fid]
    def pixels(self, i):
        r=self.cat.rows[i];rgb,bits,(n,h,w)=self.frame(r['frame_id'])
        mask=np.unpackbits(bits[r['mask_index']],count=h*w,bitorder='little').reshape(h,w).astype(bool)
        return rgb,mask
    def crop(self, i, size=(240,170)):
        rgb,m=self.pixels(i);v,u=np.nonzero(m)
        if len(u)==0:return Image.new('RGB',size,(115,111,101))
        lo=[max(0,u.min()-5),max(0,v.min()-5)];hi=[min(m.shape[1],u.max()+6),min(m.shape[0],v.max()+6)]
        obj=rgb.copy();obj[~m]=[124,116,102]
        im=Image.fromarray(obj[lo[1]:hi[1],lo[0]:hi[0]])
        factor=min(size[0]/im.width,size[1]/im.height)
        im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.LANCZOS)
        panel=Image.new('RGB',size,(40,44,49));panel.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2))
        return panel
    def original(self, i):
        rgb,m=self.pixels(i)
        from scipy.ndimage import binary_erosion
        edge=m & ~binary_erosion(m)
        overlay=rgb.copy();overlay[edge]=[255,203,78]
        return Image.fromarray(overlay)


def short_row(r):
    return f'{r["frame_id"]} m{r["mask_index"]:04d}'


def card(pic, row, title, lines, color=FG):
    panel=Image.new('RGB',(250,230),BG);panel.paste(pic,(5,3));d=ImageDraw.Draw(panel)
    d.text((5,177),title[:39],fill=color,font=FONT)
    d.text((5,191),short_row(row),fill=FG,font=FONT)
    for j,line in enumerate(lines[:2]):d.text((5,204+j*12),line[:42],fill=FG,font=FONT)
    return panel


def make_report(catalog, out):
    out=Path(out);qa=out/'qa';qa.mkdir(parents=True,exist_ok=True)
    records=[read_json(p) for p in sorted((out/'queries').glob('*/result.json'))]
    if not records:raise ValueError('No completed queries to report.')
    pics=Pictures(catalog);strips=[];links=[]
    for record in records:
        name=record['query_name'];folder=out/'queries'/name;sub=qa/name;sub.mkdir(exist_ok=True)
        q=record['query_index'];qr=catalog.rows[q]
        querypic=pics.crop(q);querypic.save(sub/'query_crop.jpg',quality=90)
        orig=pics.original(q);orig.thumbnail((720,540));orig.save(sub/'query_original.jpg',quality=88)
        rank={'supported':0,'partial_overlap':1,'inconclusive':2,'rejected':3}
        ordered=sorted(record['edges'],key=lambda e:(rank[e['status']],-e['crop_cosine']))
        show=[];stations=set()
        for e in ordered:
            s=catalog.rows[e['candidate_index']]['station_id']
            if s not in stations: show.append(e);stations.add(s)
            if len(show)==3:break
        strip=Image.new('RGB',(1000,230),BG)
        strip.paste(card(querypic,qr,'QUERY | keep + geometry',['Seed region; no semantic identity']), (0,0))
        for k,e in enumerate(show,1):
            r=catalog.rows[e['candidate_index']]
            p1=e.get('query_to_candidate',{}).get('support_fraction');p2=e.get('candidate_to_query',{}).get('support_fraction')
            evidence=f'reproj {p1:.2f}/{p2:.2f}' if p1 is not None else 'spatially separated (gate)'
            strip.paste(card(pics.crop(e['candidate_index']),r,
                             f'{e["status"].upper()} | DINO {e["crop_cosine"]:.3f}',
                             [evidence,'Heuristic evidence, NOT identity'],STATUS[e['status']]),(k*250,0))
        strips.append(strip)
        rows=[]
        debug=np.load(folder/'projection_checks.npz',allow_pickle=False) if (folder/'projection_checks.npz').exists() else None
        try:
            for j,e in enumerate(ordered):
                c=e['candidate_index'];r=catalog.rows[c];image=f'candidate_{j:03d}.jpg'
                pics.crop(c).save(sub/image,quality=87)
                projection_link=''
                key=e.get('projection_debug_key')
                # At most six useful diagnostic projections plus rejected candidates for comparison.
                if debug is not None and key and j<8:
                    pair=Image.new('RGB',(960,400),BG)
                    for offset, target_idx, suffix, title in [(0,c,'ab','QUERY surface projected INTO candidate'),
                                                            (480,q,'ba','CANDIDATE surface projected INTO query')]:
                        canvas=pics.original(target_idx);d=ImageDraw.Draw(canvas)
                        uv=debug[f'{key}_{suffix}_uv'];code=debug[f'{key}_{suffix}_code']
                        colors={1:(30,240,100),2:(100,170,255),3:(255,30,120),4:(255,205,40)}
                        for (u,v),ct in zip(uv[::max(1,len(uv)//500)],code[::max(1,len(uv)//500)]):
                            if ct:d.ellipse((u-1,v-1,u+1,v+1),fill=colors[int(ct)])
                        canvas.thumbnail((474,350),Image.Resampling.LANCZOS);pair.paste(canvas,(offset,25))
                        ImageDraw.Draw(pair).text((offset+4,5),title,fill=FG,font=FONT)
                    ImageDraw.Draw(pair).text((5,378),'Green=support  Blue=occluded  Pink=depth conflict  Yellow=outside target mask',fill=FG,font=FONT)
                    fn=f'projection_{j:03d}.jpg';pair.save(sub/fn,quality=91)
                    projection_link=f'<a href="{fn}">Bidirectional projection overlay</a>'
                metrics={k:e.get(k) for k in ('crop_cosine','context_cosine','cls_cosine','partial_bounds_gap_m',
                                             'camera_baseline_m','view_angle_deg','surface','query_to_candidate','candidate_to_query') if k in e}
                reasons=html.escape(', '.join(e['reasons']))
                rows.append(f'<tr><td><img src="{image}" width="240"><br>{html.escape(short_row(r))}<br>{html.escape(r["station_id"])}</td>'
                            f'<td><b>{e["status"]}</b><br>{reasons}<br>{projection_link}'
                            f'<details><summary>All numerical evidence</summary><pre>{html.escape(json.dumps(metrics,indent=2))}</pre></details></td></tr>')
        finally:
            if debug is not None:debug.close()
        text=f'''<!doctype html><meta charset="utf-8"><title>{name}</title>
<style>body{{font:15px system-ui;background:#18202b;color:#edf0f4;margin:24px}}a{{color:#95d5ff}}td{{vertical-align:top;padding:12px;border-bottom:1px solid #556}}pre{{white-space:pre-wrap}}img{{max-width:100%}}</style>
<a href="../index.html">All queries</a><h1>{html.escape(name)}</h1>
<p>Frame AND mask keep; geometry eligible. Scores are heuristics, not probabilities of identity.</p>
<img src="query_original.jpg"><img src="query_crop.jpg"><p>{html.escape(str(record['counts']))}</p>
<p>Projected blue samples are occluded: neither support nor contradiction. PLYs below are partial expected-depth samples, not original Gaussians, complete objects or navigation goals.</p>
<p><a href="../../queries/{name}/seed_surface_map.ply">Seed PLY (map metres)</a> | <a href="../../queries/{name}/supported_surface_preview_map.ply">Diagnostic supported-region PLY</a> | <a href="../../queries/{name}/hypothesis.json">Tentative hypothesis and provenance</a> | <a href="../../queries/{name}/result.json">All edges and source IDs</a></p>
<table>{''.join(rows)}</table>'''
        (sub/'index.html').write_text(text)
        links.append(f'<li><a href="{name}/index.html">{name}</a> — {html.escape(str(record["counts"]))}</li>')
    sheets=[]
    for start in range(0,len(strips),4):
        selected=strips[start:start+4];image=Image.new('RGB',(1000,230*len(selected)),BG)
        for i,strip in enumerate(selected):image.paste(strip,(0,i*230))
        fname=f'associations_{start//4+1:03d}.jpg';image.save(qa/fname,quality=92);sheets.append(fname)
    summary=read_json(out/'summary.json') if (out/'summary.json').exists() else {}
    (qa/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Valmet association pilot</title>
<style>body{{font:16px system-ui;background:#18202b;color:#edf0f4;max-width:1100px;margin:24px auto}}a{{color:#95d5ff}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Appearance + geometry association pilot</h1>
<p>This is NOT a scene segmentation or confirmed asset inventory. Each query forms an independent seed region. Review and excluded masks never participate. No text labels or navigation goals are generated.</p>
<pre>{html.escape(json.dumps(summary,indent=2))}</pre>
{''.join(f'<p><a href="{n}"><img src="{n}"></a></p>' for n in sheets)}<h2>Open each query for all candidates and projection checks</h2><ul>{''.join(links)}</ul>''')
    print('Association review:',qa/'index.html',flush=True)
    return qa/'index.html'
