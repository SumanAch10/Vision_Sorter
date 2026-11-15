'use strict';
const $ = id => document.getElementById(id);
const state = {blob:null, original:null, stream:null, downloadUrl:null, result:null};
const canvas = $('canvas');
const context = canvas.getContext('2d');
function headers(extra={}) {
  const key = $('api-key').value.trim();
  return {...extra, ...(key ? {Authorization:`Bearer ${key}`} : {})};
}
function showError(error) { $('error').textContent = error.message || String(error); }
function reset() {
  for (const id of ['confidence','latency','center','point']) $(id).textContent='—';
  $('object').textContent='Ready to inspect'; $('status').textContent='Awaiting inspection';
  $('reasons').replaceChildren(); $('candidates').replaceChildren();
  $('simulation').textContent='No command issued.'; $('prediction-note').textContent='';
  $('segmentation').textContent='No semantic mask yet.'; $('json').textContent='{}';
  $('download').hidden=true; state.result=null;
}
async function load(file) {
  if (!file) return;
  state.blob=null; $('run').disabled=true; reset();
  if (file.size>10485760) throw Error('Image exceeds 10 MiB.');
  const bitmap=await createImageBitmap(file);
  if (bitmap.width*bitmap.height>16000000 || Math.min(bitmap.width,bitmap.height)<64) {
    bitmap.close(); throw Error('Use an image at least 64px per side and at most 16 megapixels.');
  }
  if (state.original) state.original.close();
  state.original=bitmap; state.blob=file;
  canvas.width=bitmap.width; canvas.height=bitmap.height;
  context.drawImage(bitmap,0,0); canvas.hidden=false; $('empty').hidden=true; $('run').disabled=false;
}
function renderMask() {
  if (!state.original) return;
  context.drawImage(state.original,0,0);
  const d=state.result;
  if (!d || !$('show-mask').checked) return;
  const overlay=context.createImageData(canvas.width,canvas.height);
  for (const [start,length] of d.mask.runs) {
    for (let i=start;i<start+length;i++) {
      overlay.data[i*4]=150; overlay.data[i*4+1]=245; overlay.data[i*4+2]=100; overlay.data[i*4+3]=105;
    }
  }
  const off=document.createElement('canvas'); off.width=canvas.width; off.height=canvas.height;
  off.getContext('2d').putImageData(overlay,0,0); context.drawImage(off,0,0);
  if (d.center_px) {
    const {x,y}=d.center_px;
    context.strokeStyle='#fff'; context.lineWidth=Math.max(2,canvas.width/300);
    context.beginPath(); context.arc(x,y,Math.max(7,canvas.width/65),0,Math.PI*2); context.stroke();
  }
}
function populateList(id,items) {
  $(id).replaceChildren();
  for (const item of items) { const li=document.createElement('li'); li.textContent=item; $(id).append(li); }
}
async function connect() {
  $('error').textContent='';
  try {
    const readyResponse=await fetch('/ready'); const ready=await readyResponse.json();
    if (!readyResponse.ok) throw Error(ready.message||'Station is not ready.');
    $('health').textContent=ready.demo?'● Synthetic demo online':'● Inspection station online';
    $('version').textContent=ready.model_version;
    const response=await fetch('/v1/model',{headers:headers()}); const model=await response.json();
    if (!response.ok) throw Error(model.message||'Enter the API key and click Connect.');
    if (model.demo) {
      $('model-note').textContent='SYNTHETIC DEMO · Trained on colored shapes. Real-photo labels and pixel scores are unreliable.';
    } else {
      const facts=[];
      if (model.test?.accuracy!==undefined) facts.push(`Material classification: ${(model.test.accuracy*100).toFixed(1)}% accuracy on ${model.test.sample_count} held-out TrashNet photos.`);
      if (model.segmentation) {
        const test=model.segmentation.test;
        facts.push(`Binary litter segmentation: ${test.foreground_iou_macro.toFixed(2)} foreground IoU / ${test.foreground_dice_macro.toFixed(2)} Dice on ${test.sample_count} held-out TACO photos. Separate datasets; these are not a joint task score.`);
        $('mask-note').textContent='Green = predicted litter. White circle = predicted mask centroid. Material classes are image-level.';
      } else facts.push('No real semantic segmentation model is loaded; masks and pick coordinates are unavailable.');
      $('model-note').textContent=facts.join(' ');
    }
  } catch(error) { $('model-note').textContent=error.message; showError(error); }
}
$('connect').onclick=connect;
$('api-key').onkeydown=event=>{if(event.key==='Enter')connect();};
$('show-mask').onchange=renderMask;
$('file').onchange=async event=>{try{$('error').textContent='';await load(event.target.files[0]);}catch(error){showError(error);}};
$('camera').onclick=async()=>{
  try {
    if (state.stream) {
      const video=$('video'); if(!video.videoWidth)throw Error('Camera is not ready.');
      const frame=document.createElement('canvas'); frame.width=video.videoWidth; frame.height=video.videoHeight;
      frame.getContext('2d').drawImage(video,0,0);
      const capture=await new Promise(resolve=>frame.toBlob(resolve,'image/png'));
      state.stream.getTracks().forEach(track=>track.stop()); state.stream=null;
      video.hidden=true; $('camera').textContent='Use camera'; await load(capture);
    } else {
      state.stream=await navigator.mediaDevices.getUserMedia({video:true});
      $('video').srcObject=state.stream; $('video').hidden=false;
      $('run').disabled=true;canvas.hidden=true;$('empty').hidden=true;$('camera').textContent='Capture frame';
    }
  } catch(error) {showError(error);}
};
$('run').onclick=async()=>{
  if(!state.blob)return;
  for(const id of ['run','file','camera'])$(id).disabled=true;
  $('error').textContent='';$('status').textContent='Inspecting…';
  try {
    const response=await fetch('/v1/infer',{method:'POST',headers:headers({'Content-Type':state.blob.type}),body:state.blob});
    const d=await response.json();
    if(!response.ok)throw Error(`${d.message||d.error} (request ${d.request_id||'unknown'})`);
    state.result=d;
    $('object').textContent=(d.classification_status==='uncertain'?'Uncertain · ':'')+d.object_class.replaceAll('_',' ');
    $('status').textContent=d.status==='ready'?'Ready for simulation':'Review required';
    $('confidence-label').textContent=d.confidence_kind==='validation_temperature_scaled_class_probability'?'Calibrated class score':(d.demo?'Synthetic pixel score':'Pixel score');
    $('confidence').textContent=(d.confidence*100).toFixed(1)+'%';$('latency').textContent=d.latency_ms+' ms';
    const fmt=point=>point?`${point.x}, ${point.y}`:'—';$('center').textContent=fmt(d.center_px);$('point').textContent=fmt(d.pick_point_mm);
    $('segmentation').textContent=d.segmentation?`Binary semantic mask: background / litter · ${(d.segmentation.foreground_area_fraction*100).toFixed(1)}% foreground · ${d.segmentation.component_count} significant region(s). Pixel score ${(d.segmentation.score*100).toFixed(1)}% is uncalibrated.`:'No real litter semantic mask in this response.';
    $('prediction-note').textContent=d.warnings.join(' ');
    populateList('candidates',d.candidates?.map(item=>`${item.object_class}: ${(item.confidence*100).toFixed(1)}%`)||['Unavailable for synthetic demo']);
    populateList('reasons',d.reasons.length?d.reasons.map(reason=>reason.replaceAll('_',' ')):['All simulation gates passed']);
    $('simulation').textContent=d.simulation.executed?d.simulation.steps.map(step=>step.action).join(' → '):d.simulation.reason;
    $('json').textContent=JSON.stringify(d,null,2);renderMask();
    if(state.downloadUrl)URL.revokeObjectURL(state.downloadUrl);
    state.downloadUrl=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));
    $('download').href=state.downloadUrl;$('download').hidden=false;
  }catch(error){showError(error);$('status').textContent='Request failed';}
  finally{for(const id of ['run','file','camera'])$(id).disabled=false;}
};
window.addEventListener('pagehide',()=>{state.stream?.getTracks().forEach(track=>track.stop());});
connect();
