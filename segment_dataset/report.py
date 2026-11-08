"""Local, static review pages and machine-readable selection manifests."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import math
import os
from PIL import Image,ImageDraw
from .storage import read_json,write_json,write_jsonl,archive_valid


def esc(x): return html.escape(str(x))
def uri(path): return Path(path).resolve().as_uri()

def dump_csv(path,rows,fields):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow({k:(';'.join(v) if isinstance(v,list) else v) for k,v in r.items() if k in fields})

STYLE='''<style>body{font-family:system-ui,sans-serif;margin:24px;background:#121821;color:#e4e8ee}a{color:#9ed2ff}table{border-collapse:collapse;width:100%}td,th{border:1px solid #485468;padding:7px;text-align:left;font-size:13px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}.card{background:#202b3b;padding:12px;border-radius:8px}.card img{width:100%;max-width:640px}.tag{font-weight:bold}.keep{color:#75e5a3}.review{color:#ffd06d}.exclude{color:#ff8e96}code{white-space:pre-wrap}button{padding:6px}p{line-height:1.5}</style>'''


def build_report(dataset,out,plan,records,ph):
    dataset,out=Path(dataset),Path(out); qa=out/'qa'; qa.mkdir(exist_ok=True)
    all_rows=[]; by_frame=defaultdict(list); frame_rows=[]; counts=Counter(); run=read_json(out/'run.json') if (out/'run.json').exists() else None
    for rec in records:
        fid=rec['frame_id']; dest=out/'analysis'/ph/fid
        summary=read_json(dest/'summary.json') if (dest/'summary.json').exists() else {}
        # Do not report stale analysis for changed/corrupted capture inputs.
        if summary and (summary.get('source_complete_sha256')!=rec.get('source_complete_sha256') or not rec.get('source_ok') or run is None or not archive_valid(out/'frames'/fid,run['run_id'],rec.get('source_complete_sha256'))): summary={}
        if summary:
            rows=[json.loads(l) for l in (dest/'observations.jsonl').read_text().splitlines() if l]
            all_rows.extend(rows); by_frame[fid]=rows; counts['segmented_frames']+=1
            for row in rows: counts['mask_'+row['decision']['status']]+=1
            counts['geometry_eligible']+=summary.get('geometry_eligible',0)
        row={**rec,'segmented':bool(summary),'mask_count':summary.get('mask_count',0),
             'mask_keep':summary.get('mask_keep',0),'mask_review':summary.get('mask_review',0),
             'mask_exclude':summary.get('mask_exclude',0),'geometry_eligible':summary.get('geometry_eligible',0),
             'proposal_limit_reached':summary.get('proposal_limit_reached',False)}
        frame_rows.append(row); counts['frame_'+rec['status']]+=1
    write_jsonl(out/'observations.jsonl',all_rows)
    write_jsonl(out/'usable_observations.jsonl',[r for r in all_rows if r['decision']['eligible_features']])
    write_jsonl(out/'geometry_observations.jsonl',[r for r in all_rows if r['decision']['eligible_geometry']])
    write_jsonl(out/'review_observations.jsonl',[r for r in all_rows if r['decision']['status']=='review'])
    fields=['frame_id','station_id','camera_height_m','yaw_deg','pitch_deg','status','segmented','mask_count',
            'mask_keep','mask_review','mask_exclude','geometry_eligible','usable_fraction','near_fraction_of_covered',
            'laplacian_variance','gray_std','gradient_mean','reasons','warnings','capture_flags','proposal_limit_reached','error']
    dump_csv(out/'frame_quality.csv',frame_rows,fields)
    dump_csv(out/'review_frames.csv',[r for r in frame_rows if r['status']!='keep' or r.get('capture_flags') or (r['segmented'] and r['mask_keep']==0)],fields)
    flat=[]
    for r in all_rows:
        m=r['metrics'];d=r['decision']
        flat.append({'frame_id':r['frame_id'],'mask_id':r['mask_id'],'status':d['status'],
                     **{k:m.get(k) for k in ['score','area_px','area_fraction','usable_depth_fraction','geometry_fraction','laplacian_variance']},
                     'eligible_features':d['eligible_features'],'eligible_geometry':d['eligible_geometry'],
                     'reasons':d['reasons'],'warnings':d['warnings'],'hints':d['hints']})
    dump_csv(out/'mask_quality.csv',flat,['frame_id','mask_id','status','score','area_px','area_fraction',
             'usable_depth_fraction','geometry_fraction','laplacian_variance','eligible_features','eligible_geometry','reasons','warnings','hints'])
    counts['planned_frames']=len(records); counts['all_returned_masks']=len(all_rows)
    counts['raw_archive_bytes']=sum(f.stat().st_size for f in (out/'frames').glob('*/masks.npz'))
    counts['not_yet_segmented_eligible_frames']=sum(not r['segmented'] and r['status']!='exclude' for r in frame_rows)
    summary={'schema':'valmet.stage2.summary.v1','plan_hash':plan['plan_hash'],'policy_hash':ph,
             'run_id':run['run_id'] if run else None,'counts':dict(counts),
             'warnings':['Heuristics, not semantic labels, confidence calibration or proof of valid geometry.',
                         'All original source files and returned masks remain unchanged; selection lists are reversible.',
                         'Review masks are excluded from default usable/geometry lists until reviewed/reselected.',
                         'Sparse expected-depth summaries describe visible surface support, not complete objects.']}
    write_json(out/'summary.json',summary)
    # Summary pages: audit concerns first, then segmented examples, then unprocessed.
    ordered=sorted(frame_rows,key=lambda r:(0 if r['status']=='exclude' else 1 if r['status']=='review' or r.get('capture_flags') else 2 if r['segmented'] else 3,r['frame_id']))
    pages=[]
    for start in range(0,len(ordered),48):
        group=ordered[start:start+48]; cards=[]
        for r in group:
            fid=r['frame_id']; dest=out/'analysis'/ph/fid
            source=dataset/'frames'/fid/'rgb.png'
            shown=dest/'overlay.jpg' if r['segmented'] else source
            url=uri(shown) if shown.exists() else ''
            detail=qa/f'{fid}.html'
            rows=by_frame[fid] if r['segmented'] else []
            table=[]
            for x in rows:
                m,d=x['metrics'],x['decision']
                table.append(f'<tr><td>{esc(x["mask_id"])}</td><td class="{d["status"]}">{d["status"]}</td>'
                    f'<td>{m["area_px"]}</td><td>{m["score"]:.3f}</td><td>{m.get("usable_depth_fraction",0):.2f}</td>'
                    f'<td>{d["eligible_features"]}/{d["eligible_geometry"]}</td><td>{esc(d["reasons"]+d["warnings"]+d["hints"])}</td></tr>')
            intro=(f'<p>Frame: {esc(r["status"])} | capture flags: {esc(r.get("capture_flags",[]))}<br>'
                   f'Reasons: {esc(r.get("reasons",[])+r.get("warnings",[]))}<br>'
                   'Green outlines=keep; amber=review; red=exclude. IDs refer to archived masks; overlapping masks are retained. '
                   'Up to 100 masks are drawn in the overview; all masks remain in the archive/table.</p>')
            images=f'<a href="{uri(source)}"><img style="max-width:48%" src="{uri(source)}"></a>' if source.exists() else ''
            if r['segmented']:
                images+=f'<a href="{uri(dest/"overlay.jpg")}"><img style="max-width:48%" src="{uri(dest/"overlay.jpg")}"></a>'
                images+=f'<p><a href="{uri(dest/"depth_support.png")}">Depth support bitmap</a> | <a href="{uri(dest/"observations.jsonl")}">Per-mask metrics and geometry</a></p>'
            detail.write_text('<!doctype html><meta charset="utf-8">'+STYLE+f'<h1>{fid}</h1><a href="index.html">Summary</a>'+intro+images+
                   '<p>Mask bbox coordinates use the original RGB/depth pixels. Depth support is a heuristic, not verified surface geometry.</p>'
                   '<table><tr><th>Mask</th><th>Status</th><th>Pixels</th><th>Model score</th><th>Usable depth</th><th>Features/geometry</th><th>Reasons</th></tr>'+''.join(table)+'</table>')
            cards.append(f'<div class="card"><a href="{fid}.html"><img loading="lazy" src="{url}"></a><b>{fid}</b> '
                 f'<span class="{r["status"]}">{r["status"]}</span><p>height={r.get("camera_height_m",0):.2f} m '
                 f'pitch={r.get("pitch_deg",0):.0f} | masks={r["mask_count"]}, keep={r["mask_keep"]}, review={r["mask_review"]}<br>'
                 f'{esc(r.get("reasons",[])+r.get("warnings",[]))}</p></div>')
        page=f'frames_{start//48+1:03d}.html'; pages.append(page)
        (qa/page).write_text('<!doctype html><meta charset="utf-8">'+STYLE+f'<p><a href="index.html">Summary</a></p><div class="cards">'+''.join(cards)+'</div>')
    (qa/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+'<h1>Valmet Stage 2 — masks and quality review</h1>'
        '<p>Local files only. Source renders and model-returned masks are never deleted. '
        'Review/excluded observations are omitted from the default downstream manifests.</p>'
        f'<pre>{esc(json.dumps(summary["counts"],indent=2))}</pre>'
        '<p><a href="contact_masks_001.jpg">First mask contact sheet</a> | <a href="../frame_quality.csv">Frame-quality CSV</a> | <a href="../mask_quality.csv">Mask-quality CSV</a></p>'
        '<p>Click a frame to inspect mask IDs and exact reasons. Low texture can look blurry; expected depth cannot prove valid geometry.</p>'+
        ''.join(f'<p><a href="{p}">Review page {i+1}</a></p>' for i,p in enumerate(pages)))
    # A manageable contact sheet for sharing: priority flags and accepted pilot examples.
    segmented=[r for r in ordered if r['segmented']]
    group=segmented[:24] if segmented else ordered[:24]
    if group:
        tw,th,lh=240,180,40; sheet=Image.new('RGB',(4*tw,math.ceil(len(group)/4)*(th+lh)),(25,31,42)); draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(group):
            fid=r['frame_id']; path=out/'analysis'/ph/fid/'overlay.jpg' if r['segmented'] else dataset/'frames'/fid/'rgb.png'
            try:
                with Image.open(path) as im:
                    im=im.convert('RGB'); im.thumbnail((tw,th)); x=(i%4)*tw; y=(i//4)*(th+lh)
                    sheet.paste(im,(x+(tw-im.width)//2,y+(th-im.height)//2))
                    draw.text((x+3,y+th+1),f'{fid} | {r["status"]}',fill='white')
                    draw.text((x+3,y+th+16),f'masks={r["mask_count"]} keep={r["mask_keep"]} review={r["mask_review"]}',fill='white')
            except OSError: pass
        sheet.save(qa/'contact_masks_001.jpg',quality=92)
    print('Summary:',dict(counts),flush=True)
    print('Open:',qa/'index.html',flush=True)
    return summary
