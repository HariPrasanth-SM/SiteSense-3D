'use strict';
const $=id=>document.getElementById(id);
let token='',state=null,displayed=null,frameImage=null,depthImage=null,mapImage=new Image();
let draft=null,overlayImage=null,shape=null,seed=null,aim=null,dragStart=null,polygon=[];
let imageVersion=0,loadingId=null,freezeBusy=false,selectedObject='',objectsKey='',lastMessage='',cameraControlsDirty=false;
const held=new Set();const drives=new Map();
let camera3={yaw:.65,pitch:.35,zoom:1},orbitDrag=null;
const BOX_EDGES=[];for(let a=0;a<8;a++)for(let b=a+1;b<8;b++){let n=a^b;if(n===1||n===2||n===4)BOX_EDGES.push([a,b]);}
function notice(t,error=false){$('message').textContent=t;$('message').classList.toggle('error',error);lastMessage=t;}
async function call(cmd){
 const res=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-Label-Token':token},body:JSON.stringify(cmd)});
 const body=await res.json();if(!res.ok)throw Error(body.error||res.statusText);return body;
}
function safely(fn){return async(...args)=>{try{await fn(...args);}catch(e){notice(e.message,true);}};}
function val(id){return Number($(id).value);}
function focusedInput(){return ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName);}
function fit(w,h,W,H){const s=Math.min(W/w,H/h);return {x:(W-w*s)/2,y:(H-h*s)/2,w:w*s,h:h*s,s};}
function resize(c){const r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;let W=Math.round(r.width*dpr),H=Math.round(r.height*dpr);if(c.width!==W||c.height!==H){c.width=W;c.height=H;}return {W,H};}
function eventPixel(e,c,img){if(!img)return null;const r=c.getBoundingClientRect(),x=(e.clientX-r.left)*c.width/r.width,y=(e.clientY-r.top)*c.height/r.height;const f=fit(img.width,img.height,c.width,c.height);if(x<f.x||y<f.y||x>f.x+f.w||y>f.y+f.h)return null;return [Math.max(0,Math.min(img.width-.001,(x-f.x)/f.s)),Math.max(0,Math.min(img.height-.001,(y-f.y)/f.s))];}
function line(ctx,a,b){ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
function invalidate(){draft=null;overlayImage=null;$('save').disabled=true;$('stats').textContent='Selection changed. Build / refresh 3D selection.';drawRGB();drawCloud();}
function rawShape(){return shape?.kind==='polygon'?{kind:'polygon',points:polygon}:shape;}
function clearShape(){shape=null;polygon=[];seed=null;aim=null;dragStart=null;invalidate();}
function drawRGB(){
 const c=$('rgb'),{W,H}=resize(c),ctx=c.getContext('2d');ctx.fillStyle='#242831';ctx.fillRect(0,0,W,H);
 if(!frameImage)return;const f=fit(frameImage.width,frameImage.height,W,H);ctx.drawImage(overlayImage||frameImage,f.x,f.y,f.w,f.h);
 const screen=p=>[f.x+p[0]*f.s,f.y+p[1]*f.s];
 if($('showBoxes').checked && state?.frame?.id===displayed){
  ctx.strokeStyle='#b69dff';ctx.fillStyle='#eee3ff';ctx.lineWidth=1.2;ctx.font='11px system-ui';
  for(const box of state.box_projections||[]){for(const [a,b] of BOX_EDGES){if(box.uv[a]&&box.uv[b])line(ctx,screen(box.uv[a]),screen(box.uv[b]));}const p=box.uv.find(v=>v);if(p)ctx.fillText(box.label,...screen(p));}
 }
 ctx.lineWidth=2;ctx.strokeStyle='#f5c55b';ctx.fillStyle='#f5c55b22';
 if(shape?.kind==='rectangle'){const [x1,y1,x2,y2]=shape.xyxy;ctx.strokeRect(f.x+x1*f.s,f.y+y1*f.s,(x2-x1)*f.s,(y2-y1)*f.s);}
 if(shape?.kind==='polygon'&&polygon.length){ctx.beginPath();polygon.forEach((p,i)=>i?ctx.lineTo(...screen(p)):ctx.moveTo(...screen(p)));if(polygon.length>=3)ctx.closePath();ctx.stroke();polygon.forEach(p=>{ctx.beginPath();ctx.arc(...screen(p),3,0,2*Math.PI);ctx.fillStyle='#f5c55b';ctx.fill();});}
 for(const [p,color,label] of [[seed,'#20d9ea','depth'],[aim,'#ef6283','aim']])if(p){const [x,y]=screen(p);ctx.strokeStyle=color;ctx.lineWidth=2;line(ctx,[x-6,y],[x+6,y]);line(ctx,[x,y-6],[x,y+6]);ctx.fillStyle=color;ctx.font='11px system-ui';ctx.fillText(label,x+8,y-5);}
}
function drawDepth(){const c=$('depth'),{W,H}=resize(c),ctx=c.getContext('2d');ctx.fillStyle='#252a34';ctx.fillRect(0,0,W,H);if(depthImage){const f=fit(depthImage.width,depthImage.height,W,H);ctx.drawImage(depthImage,f.x,f.y,f.w,f.h);}}
function mapUV(xy){const m=state.map,a=m.origin[2],dx=xy[0]-m.origin[0],dy=xy[1]-m.origin[1];return [(Math.cos(a)*dx+Math.sin(a)*dy)/m.resolution,m.height-(-Math.sin(a)*dx+Math.cos(a)*dy)/m.resolution];}
function fromMapUV(p){const m=state.map,a=m.origin[2],x=p[0]*m.resolution,y=(m.height-p[1])*m.resolution;return [m.origin[0]+Math.cos(a)*x-Math.sin(a)*y,m.origin[1]+Math.sin(a)*x+Math.cos(a)*y];}
function drawMap(){
 const c=$('map'),{W,H}=resize(c),ctx=c.getContext('2d');ctx.fillStyle='#252a34';ctx.fillRect(0,0,W,H);if(!state||!mapImage.complete)return;
 const m=state.map,f=fit(m.width,m.height,W,H);ctx.imageSmoothingEnabled=false;ctx.drawImage(mapImage,f.x,f.y,f.w,f.h);
 const sp=xy=>{const p=mapUV(xy);return [f.x+p[0]*f.s,f.y+p[1]*f.s]};
 for(const o of state.objects){if(!o.active)continue;const [x,y]=sp(o.target_map_xyz);ctx.fillStyle=o.object_id===selectedObject?'#ffbf63':'#b7a1ff';ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fill();ctx.font='10px system-ui';ctx.fillText(o.object_id.replace('manual_',''),x+5,y-5);}
 const [x,y]=sp(state.pose),a=state.pose[2]-m.origin[2];ctx.fillStyle=state.collision?'#58abff':'#ff7798';ctx.beginPath();ctx.arc(x,y,5,0,2*Math.PI);ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;line(ctx,[x,y],[x+14*Math.cos(a),y-14*Math.sin(a)]);
 if(state.frame){const p=state.frame.pose,[cx,cy]=sp([p.x,p.y]);ctx.strokeStyle='#36e2e1';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(cx,cy,8,0,2*Math.PI);ctx.stroke();}
}
function drawCloud(){
 const c=$('cloud'),{W,H}=resize(c),ctx=c.getContext('2d');ctx.fillStyle='#252a34';ctx.fillRect(0,0,W,H);
 if(!draft){ctx.fillStyle='#c8c1d7';ctx.font='12px system-ui';ctx.fillText('Build a selection to inspect its 3D box.',15,H/2);return;}
 const center=draft.box.center,span=Math.max(...draft.box.extent,.1),a=camera3.yaw,b=camera3.pitch;
 const convert=(p)=>{const X=p[0]-center[0],Y=p[1]-center[1],Z=p[2]-center[2];const x=Math.cos(a)*X-Math.sin(a)*Y,y=Math.sin(a)*X+Math.cos(a)*Y;
 return [W/2+x*Math.min(W,H)*.65*camera3.zoom/span,H/2-(Math.cos(b)*Z-Math.sin(b)*y)*Math.min(W,H)*.65*camera3.zoom/span,Math.sin(b)*Z+Math.cos(b)*y];};
 const pts=draft.points.map((p,i)=>({p:convert(p),i})).sort((a,b)=>a.p[2]-b.p[2]);
 for(const q of pts){const color=draft.colors[q.i];ctx.fillStyle=`rgb(${color[0]},${color[1]},${color[2]})`;ctx.fillRect(q.p[0],q.p[1],2,2);}
 const corners=draft.corners_map.map(convert);ctx.strokeStyle='#f5c55b';ctx.lineWidth=1.5;for(const [a,b] of draft.edges)line(ctx,corners[a].slice(0,2),corners[b].slice(0,2));
 const aimP=convert(draft.target);ctx.fillStyle='#f06b90';ctx.beginPath();ctx.arc(aimP[0],aimP[1],4,0,2*Math.PI);ctx.fill();
}
function updateControls(){
 $('waiting').hidden=!!frameImage;$('freeze').disabled=!frameImage||state?.frozen||freezeBusy;$('resume').disabled=!state?.frozen;
 $('build').disabled=!state?.frozen;$('viewTitle').textContent=state?.frozen?'Frozen annotation frame':'Live Gaussian RGB';
 $('drawHint').textContent=state?.frozen?'Draw an outline; Build previews the selected surface.':'Freeze an image to draw.';
 $('warning').hidden=state?.collision!==false;
 for(const b of document.querySelectorAll('[data-drive]'))b.disabled=!!state?.frozen||!!state?.static;
}
function loadFrame(frame){
 if(!frame||frame.id===displayed||frame.id===loadingId)return;
 loadingId=frame.id;const version=++imageVersion;const image=new Image();
 image.onload=()=>{if(version!==imageVersion)return;frameImage=image;displayed=frame.id;loadingId=null;
  const d=new Image();d.onload=()=>{if(displayed===frame.id){depthImage=d;drawDepth();}};d.src=`/api/frame?id=${encodeURIComponent(frame.id)}&kind=depth`;
  $('waiting').hidden=true;drawRGB();updateControls();};
 image.onerror=()=>{if(version===imageVersion)loadingId=null;};image.src=`/api/frame?id=${encodeURIComponent(frame.id)}&kind=rgb`;
}
function listObjects(){const key=JSON.stringify(state.objects);if(key===objectsKey)return;objectsKey=key;const old=$('append').value;$('append').innerHTML='<option value="">New object</option>';$('objects').replaceChildren();$('count').textContent=state.objects.length;
 for(const o of state.objects){const opt=document.createElement('option');opt.value=o.object_id;opt.textContent=`${o.object_id} · ${o.label}`;$('append').append(opt);
  const row=document.createElement('div');row.className='obj'+(o.object_id===selectedObject?' selected':'');const im=document.createElement('img');im.src='/api/output?path='+encodeURIComponent(o.reference_crop);row.append(im);
  const text=document.createElement('div'),b=document.createElement('b'),s=document.createElement('small');b.textContent=o.label;s.textContent=`${o.object_id} · ${o.view_count} view(s) · ${o.point_count.toLocaleString()} points`;text.append(b,s);row.append(text);
  row.onclick=()=>selectObject(o.object_id);$('objects').append(row);
 }if([...$('append').options].some(o=>o.value===old))$('append').value=old;
}
function selectObject(id){selectedObject=id;const o=state.objects.find(v=>v.object_id===id);$('append').value=id;if(o){$('label').value=o.label;$('category').value=o.category;$('note').value=o.note;notice(`Adding views to ${id}. Navigate to a new viewpoint, Freeze, outline, Build, then Save.`);}objectsKey='';listObjects();drawMap();}
async function refresh(){try{const r=await fetch('/api/state');if(!r.ok)throw Error(r.statusText);state=await r.json();$('worker').textContent=state.renderer.message;
 const p=state.pose;if(!cameraControlsDirty){$('eye').value=Number(p[3].toFixed(3));$('pitch').value=Number(p[4].toFixed(1));$('speed').value=Number(state.speed.toFixed(3));}$('pose').textContent=`X ${p[0].toFixed(2)} · Y ${p[1].toFixed(2)} · heading ${(p[2]*180/Math.PI).toFixed(1)}°`;
 if(state.frame){const f=state.frame;$('frameInfo').textContent=`${state.frozen?'Frozen':'Rendered'} X ${f.pose.x.toFixed(2)}, Y ${f.pose.y.toFixed(2)} · h ${f.pose.camera_height_m.toFixed(2)} m`;}
 if(!freezeBusy)loadFrame(state.frame);updateControls();listObjects();drawMap();if(state.renderer.status==='error')notice(state.renderer.message,true);
 }catch(e){notice('Connection: '+e.message,true);}}
async function stop(){held.clear();drives.clear();document.querySelectorAll('[data-drive]').forEach(b=>b.classList.remove('held'));await call({action:'stop'});}
$('freeze').onclick=safely(async()=>{if(!displayed)return;freezeBusy=true;imageVersion++;loadingId=null;await stop();try{await call({action:'freeze',frame_id:displayed});clearShape();notice('Image frozen. Draw a rectangle or polygon on the object.');}finally{freezeBusy=false;}await refresh();});
$('resume').onclick=safely(async()=>{await stop();await call({action:'resume'});clearShape();notice('Live view resumed.');await refresh();});
$('stop').onclick=safely(stop);
$('applyCamera').onclick=safely(async()=>{await stop();await call({action:'settings',height:val('eye'),pitch:val('pitch'),speed:val('speed'),collision:$('collision').checked});cameraControlsDirty=false;await refresh();});
$('collision').onchange=safely(async()=>{await stop();await call({action:'settings',collision:$('collision').checked});await refresh();});
$('level').onclick=safely(async()=>{$('pitch').value=0;await call({action:'settings',pitch:0});cameraControlsDirty=false;await refresh();});
$('floor').onclick=safely(async()=>{$('pitch').value=-35;await call({action:'settings',pitch:-35});cameraControlsDirty=false;await refresh();});
for(const b of document.querySelectorAll('[data-drive]')){
 b.onpointerdown=e=>{e.preventDefault();if(b.disabled)return;b.setPointerCapture(e.pointerId);drives.set(e.pointerId,b.dataset.drive);b.classList.add('held');sendDrive();};
 const release=e=>{drives.delete(e.pointerId);b.classList.remove('held');sendDrive();};b.onpointerup=release;b.onpointercancel=release;b.onlostpointercapture=release;
}
function sendDrive(){if(!token)return;call({action:'drive',keys:[...new Set([...held,...drives.values()])]}).catch(e=>notice(e.message,true));}
setInterval(()=>{if(held.size||drives.size)sendDrive();},120);
const keyAction={ArrowUp:'forward',ArrowDown:'backward',ArrowLeft:'left',ArrowRight:'right',w:'look_up',s:'look_down'};
window.onkeydown=e=>{if(focusedInput())return;const key=e.key.length===1?e.key.toLowerCase():e.key;if(keyAction[key]){e.preventDefault();if(!state?.frozen&&!state?.static){held.add(keyAction[key]);sendDrive();}}else if(key==='Escape'||key===' '){e.preventDefault();safely(stop)();}else if(key==='h')$('level').click();else if(key==='f'&&!e.repeat){state?.frozen?$('resume').click():$('freeze').click();}};
window.onkeyup=e=>{const key=e.key.length===1?e.key.toLowerCase():e.key;if(keyAction[key]){held.delete(keyAction[key]);sendDrive();}};
window.addEventListener('blur',()=>safely(stop)());document.addEventListener('visibilitychange',()=>{if(document.hidden)safely(stop)();});
$('rgb').onpointerdown=e=>{if(!state?.frozen||!frameImage)return;const p=eventPixel(e,$('rgb'),frameImage);if(!p)return;e.preventDefault();$('rgb').setPointerCapture(e.pointerId);
 const tool=$('tool').value;if(tool==='rectangle'){dragStart=p;polygon=[];shape={kind:'rectangle',xyxy:[...p,...p]};}
 else if(tool==='polygon'){if(shape?.kind!=='polygon'){polygon=[];}polygon.push(p);shape={kind:'polygon',points:polygon};}
 else if(tool==='seed'){seed=p;$('useSeed').checked=true;}
 else if(tool==='aim'){aim=p;}
 invalidate();};
$('rgb').onpointermove=e=>{if(!dragStart)return;const p=eventPixel(e,$('rgb'),frameImage);if(p){shape.xyxy=[...dragStart,...p];drawRGB();}};
$('rgb').onpointerup=()=>{dragStart=null;};$('rgb').onpointercancel=()=>{dragStart=null;};
$('clear').onclick=clearShape;$('undo').onclick=()=>{if(polygon.length){polygon.pop();invalidate();}};
$('showBoxes').onchange=drawRGB;
for(const name of ['near','far','tolerance','useSeed','connected','sor'])$(name).onchange=invalidate;
async function showDraft(d){draft=d;for(const [ids,vals] of [[['cx','cy','cz'],d.box.center],[['sx','sy','sz'],d.box.extent]])ids.forEach((id,i)=>$(id).value=vals[i].toFixed(4));$('boxYaw').value=d.box.yaw_deg.toFixed(1);
 $('stats').textContent=`${d.stats.retained_points.toLocaleString()} surface samples · ${d.stats.depth_supported_pixels.toLocaleString()} supported pixels · ${d.stats.sor_removed} removed. ${d.warnings.join(' ')}`;
 $('boxInfo').textContent=`Aim: ${d.target.map(v=>v.toFixed(3)).join(', ')}. ${(100*d.points_inside_box_fraction).toFixed(1)}% of samples inside this box. Hidden extent unknown.`;
 $('save').disabled=false;const img=new Image(),id=d.id;img.onload=()=>{if(draft?.id===id){overlayImage=img;drawRGB();}};img.src='/api/draft.png?t='+Date.now();drawCloud();}
$('build').onclick=safely(async()=>{if(!shape)throw Error('Draw a rectangle or a polygon first.');await stop();$('build').disabled=true;notice('Back-projecting selected depth and checking sparse outliers…');try{
 const d=await call({action:'build',frame_id:displayed,shape:rawShape(),settings:{near:val('near'),far:val('far'),alpha_min:.5,seed_enabled:$('useSeed').checked,seed,connected:$('connected').checked,depth_tolerance:val('tolerance'),sor:$('sor').checked,std_ratio:2,aim}});await showDraft(d);notice('3D preview ready. Check the gold box and red aim point before saving.');}finally{$('build').disabled=false;}});
$('applyBox').onclick=safely(async()=>{if(!draft)throw Error('Build a selection first.');const d=await call({action:'box',draft_id:draft.id,box:{center:['cx','cy','cz'].map(val),extent:['sx','sy','sz'].map(val),yaw_deg:val('boxYaw')}});await showDraft(d);notice('Human-adjusted 3D box ready. Surface samples are unchanged.');});
$('save').onclick=safely(async()=>{if(!draft)throw Error('Build a selection first.');$('save').disabled=true;let rec;try{rec=await call({action:'save',draft_id:draft.id,label:$('label').value,category:$('category').value,note:$('note').value,append_to:$('append').value});}catch(e){$('save').disabled=false;throw e;}
 notice(`Saved ${rec.object_id}: ${rec.label}. ${rec.observations.length} view(s).`);$('saveHint').textContent=`Saved ${rec.object_id}. Resume, move to another viewpoint, then add a view to this ID if useful.`;selectedObject=rec.object_id;await refresh();$('append').value=rec.object_id;});
$('append').onchange=()=>{selectedObject=$('append').value;if(selectedObject)selectObject(selectedObject);else{$('label').value='';$('category').value='';$('note').value='';objectsKey='';listObjects();}};
$('rename').onclick=safely(async()=>{const id=$('append').value;if(!id)throw Error('Select a saved object first.');await call({action:'rename',object_id:id,label:$('label').value,category:$('category').value,note:$('note').value,active:true});await refresh();notice('Saved label/note updated without changing its geometry.');});
$('map').onclick=safely(async e=>{if(!$('teleport').checked)return;if(state.frozen)throw Error('Resume live view before repositioning.');const p=eventPixel(e,$('map'),mapImage);if(!p)return;const [x,y]=fromMapUV(p);await stop();await call({action:'start',x,y});notice('Camera repositioned explicitly; this was not a walked route.');await refresh();});
$('cloud').onpointerdown=e=>{orbitDrag=[e.clientX,e.clientY];$('cloud').setPointerCapture(e.pointerId);};
$('cloud').onpointermove=e=>{if(!orbitDrag)return;camera3.yaw+=(e.clientX-orbitDrag[0])*.012;camera3.pitch=Math.max(-1.5,Math.min(1.5,camera3.pitch+(e.clientY-orbitDrag[1])*.012));orbitDrag=[e.clientX,e.clientY];drawCloud();};
$('cloud').onpointerup=()=>orbitDrag=null;$('cloud').onpointercancel=()=>orbitDrag=null;
$('cloud').onwheel=e=>{e.preventDefault();camera3.zoom=Math.max(.1,Math.min(8,camera3.zoom*Math.exp(-e.deltaY*.001)));drawCloud();};
$('reset3d').onclick=()=>{camera3={yaw:.65,pitch:.35,zoom:1};drawCloud();};
new ResizeObserver(()=>{drawRGB();drawDepth();drawMap();drawCloud();}).observe(document.querySelector('main'));
(async()=>{try{const r=await fetch('/api/init');state=await r.json();token=state.token;mapImage.onload=drawMap;mapImage.src='/api/map.png';$('eye').value=Number(state.pose[3].toFixed(3));$('pitch').value=Number(state.pose[4].toFixed(1));$('speed').value=state.speed;$('collision').checked=state.collision;await refresh();setInterval(refresh,250);}catch(e){notice(e.message,true);}})();

for(const id of ['eye','pitch','speed'])$(id).addEventListener('input',()=>{cameraControlsDirty=true;});
