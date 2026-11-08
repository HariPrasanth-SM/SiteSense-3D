"""Read-only XLSX values, no Excel engine/install needed. Source rows are retained.

Only this builder pack's tabular worksheets are consumed. Cached formula values
are read; formulas are never recalculated or used to modify the workbook.
"""
from __future__ import annotations
import datetime as dt, json, posixpath, re, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from .common import sha
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
DATES={'installation_date','last_service_date','last_updated','timestamp','opened_at','date'}
SHEETS=['START_HERE','Asset_Master','Scene_Anchors','Current_State','Telemetry_Sample_3d',
        'Alarms_Events','Maintenance_History','Historical_Service_Cases','Knowledge_Docs',
        'RAG_Corpus_Index','Data_Quality','API_Contract']

def excel_date(x, date1904=False):
    if not isinstance(x,(int,float)): return x
    return ((dt.datetime(1904,1,1) if date1904 else dt.datetime(1899,12,30))+dt.timedelta(days=x)).isoformat(timespec='seconds')

def col_index(ref):
    n=0
    for c in re.match(r'[A-Z]+',ref).group(): n=n*26+ord(c)-64
    return n-1

def read_xlsx(path):
    path=Path(path).resolve(); result={}
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist())>300*1024**2: raise ValueError('Workbook exceeds local reader limit.')
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            ss=ET.fromstring(z.read('xl/sharedStrings.xml'))
            shared=[''.join(e.itertext()) if False else ''.join(t.text or '' for t in e.findall('.//s:t',NS)) for e in ss]
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        prop=wb.find('s:workbookPr',NS);epoch=prop is not None and prop.get('date1904') in ('1','true')
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets={r.get('Id'):r.get('Target') for r in rels}
        for sheet in wb.findall('s:sheets/s:sheet',NS):
            name=sheet.get('name')
            if name not in SHEETS: continue
            target=targets[sheet.get('{'+REL+'}id')]
            target=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
            root=ET.fromstring(z.read(target)); rows=[]
            for row in root.findall('s:sheetData/s:row',NS):
                vals={}
                for c in row.findall('s:c',NS):
                    v=c.find('s:v',NS);t=c.get('t');value=None
                    if t=='inlineStr':value=''.join(e.text or '' for e in c.findall('.//s:t',NS))
                    elif v is not None:
                        text=v.text or ''
                        if t=='s':value=shared[int(text)]
                        elif t=='b':value=text=='1'
                        elif t in ('str','e'):value=text
                        else:
                            try:value=float(text);value=int(value) if value.is_integer() else value
                            except ValueError:value=text
                    if value is not None:vals[col_index(c.get('r'))]=value
                if vals: rows.append((int(row.get('r')),vals))
            if name=='START_HERE':
                result[name]=[{'row':r,'values':[v.get(i) for i in range(max(v)+1)]} for r,v in rows];continue
            if not rows:result[name]=[];continue
            headers=rows[0][1]; records=[]
            for rownum,vals in rows[1:]:
                rec={str(h):excel_date(vals.get(c),epoch) if h in DATES else vals.get(c) for c,h in headers.items() if h}
                if all(v is None for v in rec.values()):continue
                rec['_source']=f'{name}!A{rownum}';rec['_row']=rownum;records.append(rec)
            result[name]=records
    required=('Asset_Master','Current_State','Scene_Anchors','Knowledge_Docs')
    if not all(result.get(k) for k in required): raise ValueError('Workbook does not contain the expected Valmet sheets.')
    return {'schema':'valmet.workbook_values.v1','source_filename':path.name,'source_sha256':sha(path),'sheets':result}

def load(path=None):
    if path:return read_xlsx(path)
    return json.loads((Path(__file__).parent/'data/workbook_snapshot.json').read_text())

def asset_database(workbook):
    s=workbook['sheets']; rows={};docs={r['doc_id']:r for r in s.get('Knowledge_Docs',[])}
    def per(sheet,aid):return [dict(r) for r in s.get(sheet,[]) if r.get('asset_id')==aid]
    states={r['asset_id']:r for r in s['Current_State']};anchors={r['asset_id']:r for r in s['Scene_Anchors']}
    for master in s['Asset_Master']:
        aid=master['asset_id'];state=states.get(aid,{})
        telemetry=sorted(per('Telemetry_Sample_3d',aid),key=lambda r:str(r.get('timestamp','')))
        events=sorted(per('Alarms_Events',aid),key=lambda r:str(r.get('timestamp','')),reverse=True)
        history=sorted(per('Maintenance_History',aid),key=lambda r:str(r.get('opened_at','')),reverse=True)
        cases=[r for r in s.get('Historical_Service_Cases',[]) if r.get('asset_class')==master['asset_class'] and r.get('historical_issue_code')==state.get('issue_code')]
        latest=telemetry[-1] if telemetry else None
        metric_names={'CONTROL_VALVE':['position_setpoint_pct','position_actual_pct','stiction_index','pneumatic_supply_bar'],
                      'PROCESS_PUMP':['vibration_mm_s','bearing_temp_c','process_pressure_bar','process_flow_pct'],
                      'ELECTRIC_MOTOR':['vibration_mm_s','bearing_temp_c','motor_current_a','speed_rpm'],
                      'PRESSURE_TRANSMITTER':['process_pressure_bar','sensor_bias','signal_quality_pct'],
                      'VIBRATION_SENSOR':['vibration_mm_s','sensor_bias','signal_quality_pct']}.get(master['asset_class'],[])
        docs_used=[docs[k] for k in (state.get('knowledge_doc_id'),'DOC-FIELD-SAFETY','DOC-RAG-GUIDE','DOC-3D-MAPPING') if k in docs]
        rows[aid]={'id':aid,'master':master,'state':state,'anchor_original':anchors.get(aid),
                   'latest':latest,'telemetry':telemetry,'metric_names':metric_names,'events':events,
                   'history':history,'cases':cases,'quality':per('Data_Quality',aid),'documents':docs_used,
                   'rag_index':per('RAG_Corpus_Index',aid)}
    return {'assets':rows,'source':{'filename':workbook['source_filename'],'sha256':workbook['source_sha256']},
            'counts':{k:len(v) for k,v in s.items()},'start_here':s.get('START_HERE',[]),
            'notice':'All workbook records are synthetic. Scene_Anchors are placeholders requiring explicit remapping. The workbook provides a 3-day telemetry sample; indexed external RAG documents and full telemetry CSV are not included.'}

def brief(a):
    st=a['state'];latest=a['latest'];doc=next((d for d in a['documents'] if d['doc_id']==st.get('knowledge_doc_id')),None)
    evidence=[{'source':st.get('_source'),'id':'Current_State/'+a['id'],'text':f"Workbook state: {st.get('condition_state')}; health {st.get('health_score')}/100. {st.get('issue_summary','')}"}]
    if latest:
        metrics=', '.join(f'{k}: {latest[k]}' for k in a['metric_names'] if latest.get(k) is not None)
        evidence.append({'source':latest['_source'],'id':'Telemetry_Sample_3d/'+a['id'],'text':f"Last supplied sample {latest.get('timestamp')}: {metrics}. This is a snapshot, not live equipment telemetry."})
    event=next((e for e in a['events'] if e.get('event_code')==st.get('issue_code')),None)
    if event:evidence.append({'source':event['_source'],'id':event.get('event_id'),'text':f"{event.get('timestamp')}: {event.get('description')} [{event.get('status')}]"})
    if doc:evidence.append({'source':doc['_source'],'id':doc['doc_id'],'text':doc['content']})
    return {'title':f"Service brief · {a['master']['tag']}", 'method':'Deterministic evidence assembly from supplied workbook; not a new model diagnosis.',
            'observation':st.get('issue_summary'), 'recorded_issue_code':st.get('issue_code'),
            'next_check':st.get('recommended_next_check'),'urgency':st.get('urgency'),
            'recorded_model_confidence':st.get('model_confidence'),
            'confidence_note':'Value supplied by the synthetic workbook; not our localization or diagnosis confidence.',
            'hypothesis':a['cases'][0]['historical_interpretation'] if a['cases'] else None,
            'hypothesis_note':'Class/issue-matched historical synthetic case; not proof of this asset’s current failure.',
            'evidence':evidence,'quality_flags':len(a['quality']),
            'safety':next((d['content'] for d in a['documents'] if d['doc_id']=='DOC-FIELD-SAFETY'),'Follow approved site procedures and qualified personnel.')}
