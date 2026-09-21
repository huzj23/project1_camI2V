function slotColor(id){const hue=((Number(id)||0)*137.507764)%360;return `hsl(${hue.toFixed(1)} 72% 58%)`;}
function sparseCosine(aIds,aProb,bIds,bProb){
  let dot=0,an=0,bn=0;const b=new Map(bIds.map((id,k)=>[id,bProb[k]]));
  aIds.forEach((id,k)=>{const p=aProb[k]||0;an+=p*p;dot+=p*(b.get(id)||0);});bProb.forEach(p=>bn+=p*p);
  return dot/Math.max(Math.sqrt(an*bn),1e-6);
}
const state={runs:[],windows:[],query:null,images:{},queryRequest:0,overlayRequest:0,completedCount:-1};
const $=id=>document.getElementById(id);
function error(message){$("error").textContent=String(message);$("error").classList.remove("hidden");}
function clearError(){$("error").classList.add("hidden");}
async function json(url,options){const r=await fetch(url,options);if(!r.ok)throw Error(await r.text());return r.json();}
function c(){const selectedRun=$("run").value,windowId=$("window").value,windowInfo=state.windows.find(w=>w.window_id===windowId);return {runId:selectedRun==="all17"?(windowInfo?.run_id||""):selectedRun,windowId,frameId:Number($("frameId").value),tokenId:Number($("tokenId").value),mode:$("mode").value,method:$("hypothesis").value,topL:Number($("topL").value)};}
function frameURL(frame){const x=c();return `/api/frame/${encodeURIComponent(x.runId)}/${encodeURIComponent(x.windowId)}/${frame}`;}
function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(Error("图片载入失败："+url));image.src=url+`?v=${Date.now()}`;});}
function rows(){const q=state.query;if(!q)return[];return [...q.candidates].sort((a,b)=>$("scoreKind").value==="raw"?a.raw_rank-b.raw_rank:a.verified_rank-b.verified_rank).slice(0,c().topL);}
function heatColor(value){const stops=[[0,[10,22,39]],[.25,[42,80,130]],[.5,[37,180,164]],[.75,[241,196,15]],[1,[231,76,60]]];let i=1;while(i<stops.length-1&&value>stops[i][0])i++;const [a,ca]=stops[i-1],[b,cb]=stops[i];const t=Math.max(0,Math.min(1,(value-a)/(b-a)));return "rgb("+ca.map((x,k)=>Math.round(x+(cb[k]-x)*t)).join(",")+")";}
function pane(kind){
  const q=state.query,image=state.images[kind];if(!q||!image)return;
  const canvas=$(kind+"Canvas"),ctx=canvas.getContext("2d"),{width,height,stride}=q.grid;
  canvas.width=image.width;canvas.height=image.height;ctx.drawImage(image,0,0);
  if(kind==="current"){
    if($("showGroup").checked){
      const selected=q.token_slot_probability,norm=selected?Math.hypot(...selected):0;ctx.save();ctx.fillStyle=slotColor(q.selected_motion_slot);
      for(let y=0;y<height;y++)for(let x=0;x<width;x++){
        let similarity=0;
        if(q.frame_motion.slot_ids){
          const chosen=q.token_slot_sparse||[],aIds=chosen.map(v=>v.slot_id),aProb=chosen.map(v=>v.probability);
          similarity=sparseCosine(aIds,aProb,q.frame_motion.slot_ids[y][x],q.frame_motion.slot_probabilities[y][x]);
        }else{
          const p=q.frame_motion.slot_probability[y][x],pn=Math.hypot(...p);similarity=p.reduce((sum,v,k)=>sum+v*selected[k],0)/Math.max(norm*pn,1e-6);
        }
        const strength=Math.max(0,Math.min(1,(similarity-.72)/.28))*Math.max(0,Math.min(1,(q.frame_motion.confidence[y][x]-.18)/.32));
        if(strength>0){ctx.globalAlpha=.12+.42*strength;ctx.fillRect(x*stride,y*stride,stride,stride);}
      }ctx.restore();
    }
    const [x0,y0,x1,y1]=q.token.nominal_pixel_box;ctx.fillStyle="rgba(0,255,164,.22)";ctx.fillRect(x0,y0,x1-x0,y1-y0);ctx.strokeStyle="#00ffa4";ctx.lineWidth=3;ctx.strokeRect(x0+1,y0+1,x1-x0-2,y1-y0-2);
    const support=Math.max(...q.token.descriptor_support_pixels),cx=(x0+x1)/2,cy=(y0+y1)/2;ctx.save();ctx.setLineDash([6,4]);ctx.strokeStyle="white";ctx.strokeRect(cx-support/2,cy-support/2,support,support);ctx.restore();return;
  }
  const endpoint=kind==="first"?"first":"last",scoreKind=$("scoreKind").value;
  if($("showHeat").checked){const map=(scoreKind==="raw"?q.raw_heatmap:q.verified_heatmap)[endpoint];ctx.globalAlpha=scoreKind==="raw"?.42:.58;for(let y=0;y<height;y++)for(let x=0;x<width;x++){ctx.fillStyle=heatColor(map[y][x]);ctx.fillRect(x*stride,y*stride,stride,stride);}ctx.globalAlpha=1;}
  rows().filter(row=>row.endpoint===endpoint).forEach(row=>{
    const [x,y]=row.reference_xy,v=scoreKind==="raw"?Math.max(0,Math.min(1,(row.raw_score-.45)/.55)):row.verified_weight;
    ctx.strokeStyle=endpoint==="first"?"#39a8ff":"#ff9638";ctx.lineWidth=1.5+5*v;ctx.strokeRect(x*stride+2,y*stride+2,stride-4,stride-4);
    ctx.fillStyle="#07101d";ctx.fillRect(x*stride+1,y*stride+1,18,14);ctx.fillStyle="white";ctx.font="bold 10px system-ui";ctx.fillText(scoreKind==="raw"?String(row.raw_rank):`V${row.verified_rank}`,x*stride+3,y*stride+11);
    if(row.sparse_allowed){ctx.fillStyle="#58f0be";ctx.fillRect(x*stride+stride-5,y*stride+2,3,stride-4);}
  });
  if(kind==="last"&&q.mode==="F"){ctx.fillStyle="rgba(5,10,18,.64)";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#d5dfeb";ctx.font="bold 22px system-ui";ctx.fillText("F 模式：尾帧仅展示，不参与候选",28,44);}
}
function crop(image,cx,cy,title,meta,allowed=false){
  const div=document.createElement("div");div.className="crop"+(allowed?" allowed":"");
  const canvas=document.createElement("canvas");canvas.width=canvas.height=102;const ctx=canvas.getContext("2d");ctx.imageSmoothingEnabled=false;ctx.drawImage(image,cx-32,cy-32,64,64,0,0,102,102);
  const b=document.createElement("b"),span=document.createElement("span");b.textContent=title;span.textContent=meta;div.append(canvas,b,span);return div;
}
function details(){
  const q=state.query,s=q.grid.stride,crops=$("crops"),body=$("candidateRows"),slots=$("slots");crops.replaceChildren();body.replaceChildren();slots.replaceChildren();
  const [qx,qy]=q.token.xy;crops.append(crop(state.images.current,(qx+.5)*s,(qy+.5)*s,"QUERY · 当前帧",`token ${q.token.id}`));
  rows().forEach(row=>{
    const [x,y]=row.reference_xy,endpoint=row.endpoint==="first"?"首":"尾";crops.append(crop(state.images[row.endpoint],(x+.5)*s,(y+.5)*s,`R${row.raw_rank} / V${row.verified_rank} · ${endpoint}`,`w=${row.verified_weight.toFixed(3)} · s${row.best_motion_slot}`,row.sparse_allowed));
    const tr=document.createElement("tr");const values=[row.raw_rank,row.verified_rank,endpoint,`${row.reference_token_id} / (${x},${y})`,row.raw_score.toFixed(5),row.verified_weight.toFixed(5),row.sparse_allowed?"YES":"—",`s${row.best_motion_slot}`];
    values.forEach(v=>{const td=document.createElement("td");td.textContent=v;tr.append(td);});body.append(tr);
  });
  q.motion_slots.forEach(slot=>{const div=document.createElement("div");div.className="slot-card";const dot=document.createElement("i");dot.style.background=slotColor(slot.slot_id);const text=document.createElement("div");text.innerHTML=`<strong>slot ${slot.slot_id} · p=${slot.token_probability.toFixed(3)}</strong><span>scale-rate ${slot.scale_rate.toFixed(3)}</span><span>v=(${slot.vx.toFixed(2)}, ${slot.vy.toFixed(2)})</span>${slot.mass===undefined?"":`<span>票质量 ${slot.mass.toFixed(2)} · 年龄 ${slot.age} 帧</span>`}`;div.append(dot,text);slots.append(div);});
  $("confidence").textContent=q.confidence.toFixed(3);$("reject").textContent=q.reject_probability.toFixed(3);$("slot").textContent=q.selected_motion_slot>=0?`slot ${q.selected_motion_slot}`:"未归属";
  $("activeSlots").textContent=`${q.active_slot_count}/${q.slot_capacity}`;
  $("capacityOptions").textContent=`本次推理上限 ${q.slot_capacity}；可选上限 ${q.slot_capacity_options.join(" / ")}。更换上限需要生成独立运行，网页不会伪造即时结果。`;
  const lifecycle=$("lifecycle");lifecycle.replaceChildren();
  const labels={birth:"新增",termination:"终止",split:"分裂",merge:"合并",disappearance:"暂时消失",reappearance:"重新出现"};
  if(!q.slot_lifecycle.length)lifecycle.textContent="本帧没有槽生命周期事件。";
  q.slot_lifecycle.forEach(event=>{const item=document.createElement("div");item.className=`lifecycle-event ${event.type}`;item.textContent=`${labels[event.type]||event.type} · ${JSON.stringify(event).replace(/\"/g,"")}`;lifecycle.append(item);});
  const notice=$("queryStatus");
  if(q.selected_reference_endpoint===2){notice.textContent=`P5 置信度 ${q.confidence.toFixed(3)} 低于 0.18：算法拒绝全部候选，实际允许参考 Token 为 0。首尾帧框仍按计算所得候选排名显示。`;notice.classList.remove("hidden");}
  else{notice.classList.add("hidden");}
  $("endpoint").textContent=q.selected_reference_endpoint===0?"首帧":q.selected_reference_endpoint===1?"尾帧":"拒绝";$("texture").textContent=q.token.texture_confidence.toFixed(3);
  $("sparseCount").textContent=`${rows().filter(row=>row.sparse_allowed).length}/${rows().length}`;$("queryMeta").textContent=`token ${q.token.id} · (${q.token.xy.join(",")})`;$("lastStatus").classList.toggle("disabled",q.mode==="F");
}
function slotOptions(){
  const selector=$("slotSelection"),previous=selector.value;selector.replaceChildren(new Option("全部（无白框）","all"));
  state.query.motion_slots.forEach(slot=>selector.add(new Option(`slot ${slot.slot_id}`,String(slot.slot_id))));
  selector.value=[...selector.options].some(option=>option.value===previous)?previous:"all";
}
async function overlay(){
  const q=state.query,x=c();if(!q)return;const request=++state.overlayRequest,url=`/api/overlay/${encodeURIComponent(x.runId)}/${encodeURIComponent(x.windowId)}/${x.mode}/${x.method}/${$("overlayKind").value}/${x.frameId}`;
  try{
    const image=await loadImage(url);if(request!==state.overlayRequest)return;
    const canvas=$("overlayCanvas"),ctx=canvas.getContext("2d");canvas.width=image.width;canvas.height=image.height;ctx.drawImage(image,0,0);
    const chosen=$("slotSelection").value;if(chosen==="all"){$("slotSelectionMeta").textContent="选择一个槽可用白边界追踪该槽成员。";return;}
    const slot=Number(chosen),{width,height,stride}=q.grid,prob=q.frame_motion.slot_probability,rejected=q.frame_motion.rejected||Array.from({length:height},(_,y)=>Array.from({length:width},(_,x)=>q.hypothesis==="P5"&&q.frame_motion.confidence[y][x]<.18));
    const member=Array.from({length:height},(_,y)=>Array.from({length:width},(_,x)=>{
      if(rejected[y][x])return false;
      if(q.frame_motion.slot_ids)return q.frame_motion.slot_ids[y][x][0]===slot&&q.frame_motion.slot_probabilities[y][x][0]>0;
      return prob[y][x].indexOf(Math.max(...prob[y][x]))===slot;
    }));
    let count=0;ctx.save();ctx.fillStyle="rgba(255,255,255,.13)";ctx.strokeStyle="#fff";ctx.lineWidth=2.5;ctx.shadowColor="#06111e";ctx.shadowBlur=4;ctx.beginPath();
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(member[y][x]){
      count++;const px=x*stride,py=y*stride;ctx.fillRect(px,py,stride,stride);
      if(y===0||!member[y-1][x]){ctx.moveTo(px,py);ctx.lineTo(px+stride,py);}
      if(x===width-1||!member[y][x+1]){ctx.moveTo(px+stride,py);ctx.lineTo(px+stride,py+stride);}
      if(y===height-1||!member[y+1][x]){ctx.moveTo(px+stride,py+stride);ctx.lineTo(px,py+stride);}
      if(x===0||!member[y][x-1]){ctx.moveTo(px,py+stride);ctx.lineTo(px,py);}
    }ctx.stroke();ctx.restore();$("slotSelectionMeta").textContent=`slot ${slot}：${count} / ${width*height} 个 Token；白边界是硬归属，不代表真实物体分割。`;
  }catch(e){if(request===state.overlayRequest)error(e);}
}
async function refresh(){
  const x=c();if(!x.runId||!x.windowId)return;const request=++state.queryRequest;
  try{
    clearError();const params=new URLSearchParams({run_id:x.runId,window_id:x.windowId,frame_id:x.frameId,token_id:x.tokenId,mode:x.mode,hypothesis:x.method,top_l:x.topL});
    const query=await json("/api/query?"+params),windowInfo=state.windows.find(w=>w.window_id===x.windowId);
    const [first,current,last]=await Promise.all([loadImage(frameURL(0)),loadImage(frameURL(x.frameId)),loadImage(frameURL(windowInfo.frame_count-1))]);
    if(request!==state.queryRequest)return;state.query=query;state.images={first,current,last};
    ["first","current","last"].forEach(pane);details();slotOptions();overlay();
  }catch(e){if(request===state.queryRequest)error(e);}
}
async function loadRuns(){
  try{state.runs=await json("/api/runs");$("run").replaceChildren(new Option("全部短窗（冻结 + 扩展，持续增加）","all17"),...state.runs.map(run=>new Option(`${run.run_id} · ${run.execution_mode}`,run.run_id)));const dynamic=state.runs.find(run=>run.run_id==="demo0_v11_dynamic32");$("run").value=dynamic?dynamic.run_id:"demo0_v11_d0real";await loadWindows();}catch(e){error(e);}
}
async function loadWindows(preserveSelection=false){
  try{
    const previous=preserveSelection===true?$("window").value:null;
    const runId=$("run").value;if(!runId)return;
    if(runId==="all17"){
      const groups=await Promise.all(state.runs.map(run=>json(`/api/runs/${encodeURIComponent(run.run_id)}/windows`).then(windows=>windows.filter(w=>w.status==="complete_d0_real").map(w=>({...w,run_id:run.run_id})))));
      state.windows=groups.flat();
    }else state.windows=(await json(`/api/runs/${encodeURIComponent(runId)}/windows`)).filter(w=>w.status==="complete_d0_real").map(w=>({...w,run_id:runId}));
    const order=w=>{const match=w.window_id.match(/_(\d{4})_(\d{4})$/);return (w.window_id.startsWith("simple")?0:10000)+(match?Number(match[1]):0);};
    state.windows.sort((a,b)=>order(a)-order(b));
    $("window").replaceChildren(...state.windows.map(w=>new Option(`${w.display_name}${runId==="all17"&&w.run_id==="demo0_v11_d0real"?" · 冻结":""}`,w.window_id)));
    if(!state.windows.length){error("此运行仍在推理中，暂无已完成窗口。先看冻结三窗口，稍后刷新页面。");return;}
    if(previous&&state.windows.some(w=>w.window_id===previous)){$("window").value=previous;return;}
    windowChanged();
  }catch(e){error(e);}
}
function windowChanged(){
  const w=state.windows.find(v=>v.window_id===$("window").value);if(!w)return;
  if(!w.reference_modes.includes($("mode").value))$("mode").value=w.reference_modes[0];
  if(!w.methods.includes($("hypothesis").value))$("hypothesis").value=w.methods[0];
  $("frameId").max=w.frame_count-1;$("frameId").value=Math.floor((w.frame_count-1)/2);$("frameLabel").textContent=`${$("frameId").value}/${w.frame_count-1}`;
  $("tokenId").max=w.token_grid.count-1;$("tokenId").value=Math.floor(w.token_grid.count/2);refresh();
}
$("currentCanvas").addEventListener("click",event=>{
  if(!state.query)return;const rect=event.currentTarget.getBoundingClientRect(),x=(event.clientX-rect.left)*event.currentTarget.width/rect.width,y=(event.clientY-rect.top)*event.currentTarget.height/rect.height;
  const {width,height,stride}=state.query.grid,tx=Math.max(0,Math.min(width-1,Math.floor(x/stride))),ty=Math.max(0,Math.min(height-1,Math.floor(y/stride)));
  $("tokenId").value=ty*width+tx;refresh();
});
$("frameId").addEventListener("input",()=>{$("frameLabel").textContent=`${$("frameId").value}/${$("frameId").max}`;refresh();});
$("run").addEventListener("change",loadWindows);$("window").addEventListener("change",windowChanged);
["mode","hypothesis","topL","tokenId"].forEach(id=>$(id).addEventListener("change",refresh));
["showHeat","scoreKind"].forEach(id=>$(id).addEventListener("change",()=>{if(!state.query)return;pane("first");pane("last");details();}));
$("showGroup").addEventListener("change",()=>pane("current"));
$("overlayKind").addEventListener("change",overlay);$("slotSelection").addEventListener("change",overlay);
$("saveAnnotation").addEventListener("click",async()=>{
  const x=c(),ranks=$("correctRanks").value.split(",").map(v=>Number(v.trim())).filter(v=>Number.isInteger(v)&&v>0);
  const body={run_id:x.runId,window_id:x.windowId,frame_id:x.frameId,token_id:x.tokenId,mode:x.mode,hypothesis:x.method,first_visible:$("firstVisible").checked,last_visible:$("lastVisible").checked,correct_candidate_ranks:ranks,should_reject:$("shouldReject").checked,temporal_slot_consistent:$("temporalConsistent").checked,category:$("category").value,note:$("note").value};
  try{const result=await json("/api/annotations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});$("saveStatus").textContent=`已保存；当前共 ${result.query_count} 条核验记录。`;}catch(e){error(e);}
});
async function pollCoverage(){
  try{
    const windows=await json("/api/runs/demo0_v11_fullcoverage_d0real/windows");
    const complete=windows.filter(w=>w.status==="complete_d0_real").length,previous=state.completedCount;
    state.completedCount=complete;
    $("coverageStatus").textContent=`扩展短窗口已完成 ${complete}/14；连同冻结三段，可查看 ${complete+3}/17 段。新段完成后自动进入“全部短窗”列表。`;
    if(previous>=0&&complete>previous&&["all17","demo0_v11_fullcoverage_d0real"].includes($("run").value))await loadWindows(true);
  }catch(e){$("coverageStatus").textContent="扩展窗口进度暂不可读；冻结三段仍可查看。";}
}
loadRuns();pollCoverage();setInterval(pollCoverage,30000);
