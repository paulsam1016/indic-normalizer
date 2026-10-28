"""The verifier's single HTML page (inline CSS + JS, no network)."""

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Verifier</title>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e2e2e5;--hover:#f3f3f6;--accent:#2f6fde;
--accbg:#e8f0fd;--ok:#1f7a4d;--okbg:#e5f4ec;--warn:#8a5a00;--warnbg:#fdf1d8;--bad:#b42318;--badbg:#fde7e5;--code:#f0f0f2}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1b1b1f;--fg:#e8e8ea;--muted:#9a9aa2;--line:#2c2c31;
--hover:#232328;--accent:#7aa7ff;--accbg:#1c2a44;--ok:#5fd39a;--okbg:#12301f;--warn:#f0c060;--warnbg:#33280f;--bad:#ff8a80;--badbg:#3a1714;--code:#232328}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI","Noto Sans Malayalam","Nirmala UI",sans-serif;display:flex;flex-direction:column}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:16px;margin:0 10px 0 0}
input,select,button{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button:hover{background:var(--hover)}button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button:disabled{opacity:.5;cursor:default}
.path{min-width:260px;flex:1}
main{flex:1;display:flex;min-height:0}
aside{width:320px;border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0;background:var(--panel)}
aside .tools{padding:8px;display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
#files{overflow:auto;flex:1}
.f{padding:7px 10px;border-bottom:1px solid var(--line);cursor:pointer}.f:hover{background:var(--hover)}
.f.sel{background:var(--accbg)}.f .n{font-family:ui-monospace,Consolas,monospace;font-size:12px;word-break:break-all}
.f .m{font-size:12px;color:var(--muted)}
.pill{display:inline-block;padding:0 7px;border-radius:10px;font-size:12px}
.verify{background:var(--warnbg);color:var(--warn)}.reverify{background:var(--accbg);color:var(--accent)}.failed{background:var(--badbg);color:var(--bad)}.done,.verbalized{background:var(--okbg);color:var(--ok)}
section#view{flex:1;overflow:auto;padding:14px 18px 80px}
.muted{color:var(--muted)}.warn{color:#c2410c;font-size:13px}.help{font-size:13px;color:var(--muted)}
.card{border:1px solid var(--line);border-radius:10px;background:var(--panel);margin:12px 0;padding:10px 12px}
.card.cur{border-color:var(--accent);box-shadow:0 0 0 2px var(--accbg)}
.sh{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--muted)}
.orig{margin:6px 0 2px}.prev{margin:2px 0 8px;font-size:15px}
mark{background:var(--warnbg);color:inherit;border-radius:3px;padding:0 2px}
.num{border-top:1px dashed var(--line);padding:8px 0 4px}
.num h4{margin:0 0 4px;font-size:14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.cands{display:flex;flex-direction:column;gap:3px}
.cands label{display:flex;gap:8px;align-items:center;padding:3px 6px;border-radius:6px;cursor:pointer}
.cands label:hover{background:var(--hover)}.cands label.on{background:var(--accbg)}
.k{font-family:ui-monospace,Consolas,monospace;font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--muted)}
.d{font-size:12px;color:var(--muted);min-width:64px;text-align:right;font-variant-numeric:tabular-nums}
.star{color:var(--accent);font-size:12px}
.custom{flex:1;min-width:200px}
footer{position:sticky;bottom:0;display:flex;gap:10px;align-items:center;padding:10px 14px;border-top:1px solid var(--line);background:var(--panel)}
#toast{position:fixed;right:16px;bottom:70px;padding:8px 12px;border-radius:8px;background:var(--fg);color:var(--bg);display:none;max-width:60ch}
dialog{border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--fg);width:min(720px,94vw);padding:14px}
dialog::backdrop{background:rgba(0,0,0,.35)}
#bdirs{max-height:50vh;overflow:auto;border:1px solid var(--line);border-radius:8px;margin:8px 0}
#bdirs div{padding:5px 10px;cursor:pointer;border-bottom:1px solid var(--line)}#bdirs div:hover{background:var(--hover)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:10px 0 0}
.tab{border:1px solid var(--line);border-bottom:none;border-radius:8px 8px 0 0;background:var(--bg);padding:6px 14px;margin-bottom:-1px}
.tab.on{background:var(--panel);font-weight:600;border-color:var(--line);border-bottom:1px solid var(--panel)}
.stepbar,.savebar{display:flex;gap:10px;align-items:center;margin:12px 0}
#pane-check .card{display:none}#pane-check .card.cur{display:block}
#pane-check .num{display:none}#pane-check .num.numcur{display:block}
table.rv{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px}
table.rv th,table.rv td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
table.rv th{color:var(--muted);font-weight:600}
.later{color:var(--warn)}.custompick{color:var(--accent)}
#autoplay.on{background:var(--accbg);border-color:var(--accent);color:var(--accent)}
.tabs .sep{flex:1}.tab.sents{border-style:dashed}
textarea.sedit{width:100%;box-sizing:border-box;min-height:3.2em;font:inherit;font-size:15px;color:var(--fg);background:var(--bg);border:1px solid var(--accent);border-radius:6px;padding:6px 8px}
.stext{font-size:15px;margin:6px 0 2px}.was{font-size:12px;color:var(--muted);margin-top:2px}
details{margin:6px 0}summary{cursor:pointer;color:var(--accent)}
dl{display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;margin:8px 0;font-size:13px}dt{font-weight:600}dd{margin:0}
</style></head><body>
<header>
 <h1>Verifier</h1>
 <label title="A verbalize output folder: OUT_DIR/&lt;data dir&gt; (contains verify/, failed/, verify.tsv, failed.tsv)">Run folder
  <input id="run" class="path" placeholder="…\verbalized\eval_frozen"></label><button data-b="run">Browse…</button>
 <label title="Folder with the &lt;stem&gt;.wav files">Audio folder <input id="aud" class="path" placeholder="…\audio\eval_frozen"></label><button data-b="aud">Browse…</button>
 <label title="Optional: the folder verbalize read (&lt;stem&gt;.json inputs, e.g. …\data\eval_frozen). Needed to change already-decided numbers of runs made before input/ copies were kept.">Input folder <input id="dat" class="path" placeholder="(optional) …\data\eval_frozen"></label><button data-b="dat">Browse…</button>
 <button class="primary" id="load">Load</button>
 <select id="recent" title="Recently used folders"><option value="">Recent…</option></select>
 <button id="gen" title="Candidate generator: type any number and see its readings (G)">🔢 Candidates</button>
 <span id="sum" class="muted"></span>
</header>
<main>
 <aside>
  <div class="tools">
   <select id="kind"><option value="">all</option><option value="verify">verify</option><option value="reverify">reverify</option><option value="failed">failed</option><option value="verbalized" title="finished files decided by the model only (no number verified by hand)">verbalized (auto)</option><option value="verified" title="finished files with numbers verified by hand">verified</option></select>
   <input id="q" placeholder="search file" style="flex:1;min-width:100px">
  </div>
  <div id="files"><p class="help" style="padding:10px">Load a run folder to see the files to check.</p></div>
 </aside>
 <section id="view">
  <h2 style="margin-top:0">How it works</h2>
  <p>Files land in <code>verify/</code> when a number's reading was uncertain, and in <code>failed/</code> when
  verbalize could not process them. In a verify file every confident number is already verbalized; only the
  uncertain ones are still digits, and only those are shown here — with the sentence audio, the candidate readings
  (the model's pick ★ pre-selected; Δ = how much less likely than the best, in nats), a box to type your own reading,
  and <i>later</i> to leave it open. <b>Save</b> applies your readings. When no number is left open the file is
  written to <code>verbalized/</code>, its rows move from <code>verify.tsv</code> to <code>verbalized.tsv</code>
  (status <code>verified</code>) and the copy is removed; otherwise it stays in <code>verify/</code> with the rest.</p>
  <p><b>Spelling</b>: ★ is the pick after the spelling rule — if the audio's own spelling (🔊) beat the others by less
  than <code>style_margin</code> the default style (contracted) is pre-selected. <b>Suffix</b> (orange): for numbers like
  27-ന് / 55000-യുടെ the written suffix may not be what was said — candidates marked <code>+nosuffix</code> are the bare
  number; if a word was left out (അമ്പത്തയ്യായിരം <i>രൂപ</i>യുടെ) type the whole reading, it replaces the full token.</p>
  <details open><summary>Keyboard</summary><dl>
   <dt>Space</dt><dd>play / stop the current sentence</dd><dt>R</dt><dd>replay from the start</dd>
   <dt>A</dt><dd>autoplay on / off: each sentence plays when its number comes up</dd>
   <dt>1 – 9</dt><dd>choose that candidate for the current number</dd><dt>0</dt><dd>type a custom reading</dd>
   <dt>L</dt><dd>leave this number for later (stays open)</dd>
   <dt>G</dt><dd>candidate generator (🔢): readings of any number you type; <i>Use</i> sets it for the current number</dd>
   <dt>Enter / → </dt><dd>next number (1 · Check tab); after the last one the 2 · Review tab opens</dd>
   <dt>← / ↓ ↑ / J K</dt><dd>previous / next number</dd><dt>Ctrl + Enter</dt><dd>go to Review; in Review: save the file and open the next one</dd>
   <dt>[ / ]</dt><dd>slower / faster playback</dd></dl></details>
 </section>
</main>
<footer id="foot" style="display:none">
 <span id="status" class="muted"></span>
 <span style="flex:1"></span>
 <button id="autoplay" title="play each sentence when its number comes up (A)">▶ Autoplay: off</button>
 <label title="Seconds of audio played before / after the sentence">pad <input id="pad" type="number" value="0.3" step="0.1" min="0" style="width:60px"> s</label>
 <label>speed <select id="rate"><option>0.75</option><option selected>1</option><option>1.25</option><option>1.5</option></select></label>
 <audio id="au" preload="none"></audio>
</footer>
<dialog id="browse"><b id="btitle">Choose a folder</b>
 <div style="display:flex;gap:6px;margin-top:8px"><button id="bup">⬆ Up</button><input id="bpath" style="flex:1"><button id="bgo">Go</button></div>
 <div id="binfo" class="help"></div><div id="bdirs"></div>
 <div style="display:flex;gap:8px;justify-content:flex-end"><button id="bcancel">Cancel</button><button class="primary" id="bok">Use this folder</button></div>
</dialog>
<dialog id="gdlg"><b>Candidate generator</b>
 <p class="help" style="margin:4px 0 8px">Type a number or a short phrase (55000, 250യിൽ, 4:30-ന്, 12.5%, <i>55000 രൂപയുടെ</i>; Hindi: <i>428 रुपये</i>, or pick Hindi).
 Every reading is listed; <b>Use</b> puts the whole phrase into the current number's custom box (it replaces the full token).</p>
 <div style="display:flex;gap:6px"><input id="gq" style="flex:1" placeholder="e.g. 55000 രൂപയുടെ">
  <select id="gstyle" title="default spelling style"><option value="">run style</option><option>contracted</option><option>glide</option></select>
  <select id="glang" title="language of the readings"><option value="">auto</option><option value="ml">Malayalam</option><option value="hi">Hindi</option></select>
  <button class="primary" id="ggo">Generate</button></div>
 <div id="gout" style="max-height:60vh;overflow:auto;margin-top:8px"></div>
 <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:8px"><span id="gtarget" class="help" style="flex:1"></span><button id="gclose">Close</button></div>
</dialog>
<div id="toast"></div>
<script>
(function(){
var $=function(id){return document.getElementById(id);};
function esc(s){return s==null?'':String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function api(path,body){return fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})
 .then(function(r){return r.json().then(function(j){if(!r.ok||j.error)throw new Error(j.error||r.statusText);return j;});});}
function toast(msg,ms){var t=$('toast');t.textContent=msg;t.style.display='block';clearTimeout(t._h);t._h=setTimeout(function(){t.style.display='none';},ms||3500);}
var S={files:[],cur:null,doc:null,choice:{},focus:0,flat:[]};

// ---------------------------------------------------------------- folders
function state(){return api('/api/state').then(function(s){
 $('run').value=s.run_dir||'';$('aud').value=s.audio_dir||'';$('dat').value=s.data_dir||'';
 $('recent').innerHTML='<option value="">Recent…</option>'+(s.recent||[]).map(function(r,i){return '<option value="'+i+'">'+esc(r.run_dir)+'</option>';}).join('');
 S.recent=s.recent||[];
 if(s.loaded){var m=s.summary||{};$('sum').textContent='verbalized '+(m.verbalized==null?'?':m.verbalized)+' · verify '+(m.verify==null?'?':m.verify)+' · reverify '+(m.reverify||0)+' · failed '+(m.failed==null?'?':m.failed)+(m.verified?' · verified by hand '+m.verified:'')+(s.audio_ok?'':' · ⚠ audio folder not set');return loadFiles().then(function(){return s;});}
 return s;});}
function picks(){var o={};if(!S.doc)return o;S.doc.sentences.forEach(function(s){s.numbers.forEach(function(n,j){o[n.id]=S.choice[s.idx][j];});});return o;}
$('load').onclick=function(){api('/api/open',{run_dir:$('run').value,audio_dir:$('aud').value,data_dir:$('dat').value}).then(function(){S.cur=null;return state();}).catch(function(e){toast(e.message,6000);});};
$('recent').onchange=function(){var r=S.recent[+this.value];if(r){$('run').value=r.run_dir;$('aud').value=r.audio_dir||'';$('dat').value=r.data_dir||'';$('load').onclick();}this.value='';};
var B={target:null};
function browse(path){api('/api/browse?path='+encodeURIComponent(path||'')).then(function(b){B.path=b.path;B.parent=b.parent;$('bpath').value=b.path||'';
 $('binfo').textContent=(b.is_run?'✓ this is a verbalize run folder':'')+(b.n_wav?(b.is_run?' · ':'')+b.n_wav+' .wav files here':'');
 $('bdirs').innerHTML=b.dirs.map(function(d){return '<div data-d="'+esc(d)+'">📁 '+esc(d)+'</div>';}).join('')||'<p class="help" style="padding:8px">no subfolders</p>';
 $('bdirs').querySelectorAll('div[data-d]').forEach(function(el){el.onclick=function(){var d=el.dataset.d;browse(b.path?(b.path.replace(/[\\\/]$/,'')+(b.path.indexOf('\\')>=0||/^[A-Z]:/.test(b.path)?'\\':'/')+d):d);};});
}).catch(function(e){toast(e.message);});}
document.querySelectorAll('button[data-b]').forEach(function(b){b.onclick=function(){B.target=b.dataset.b;$('btitle').textContent=B.target==='run'?'Choose the run folder (OUT_DIR\\<data dir>)':(B.target==='dat'?'Choose the input folder (the data dir verbalize read)':'Choose the audio folder');browse($(B.target).value);$('browse').showModal();};});
$('bup').onclick=function(){if(B.parent!==null&&B.parent!==undefined)browse(B.parent);};
$('bgo').onclick=function(){browse($('bpath').value);};
$('bcancel').onclick=function(){$('browse').close();};
var G={target:null,timer:null};
function openGen(){var x=S.cur&&S.flat?S.flat[S.focus]:null;G.target=x||null;
 if(S.tab==='review'&&S.editing&&$('dcustom')){G.target={decided:S.editing};$('gtarget').textContent='Use → decided number ('+S.editing+')';$('gdlg').showModal();$('gq').focus();if($('gq').value)gen();return;}
 if(S.tab==='review')x=G.target=null;
 $('gtarget').textContent=x?('Use → number '+numOf(x).core+' (sentence '+x[0]+')'):'Open a file to use a reading; Copy works anyway.';
 if(x&&!$('gq').value)$('gq').value=numOf(x).core;$('gdlg').showModal();$('gq').focus();$('gq').select();if($('gq').value)gen();}
function gen(){var q=$('gq').value.trim();if(!q){$('gout').innerHTML='';return;}
 api('/api/candidates?q='+encodeURIComponent(q)+'&style='+encodeURIComponent($('gstyle').value)+'&lang='+encodeURIComponent($('glang').value)).then(function(r){
  var h='';r.groups.forEach(function(g,gi){h+='<h4 style="margin:10px 0 4px"><mark>'+esc(g.token)+'</mark> <span class="muted">'+esc(g.class)+' · '+esc(r.lang)+' · '+g.candidates.length+' readings</span>'+
   (g.suffix_issue?' <span class="warn">suffix: '+esc(g.suffix_issue)+'</span>':'')+'</h4><div class="cands">';
   g.candidates.forEach(function(c,i){h+='<label style="cursor:default"><span class="k">'+(i+1)+'</span> <span>'+esc(c.text)+'</span> <span class="muted" style="font-size:12px">'+esc(c.form)+(c.scored?'':' · beyond max_candidates ('+r.max_candidates+'), not scored by verbalize')+'</span>'+
    (c.phrase!==c.text?'<span class="muted" style="font-size:12px"> → '+esc(c.phrase)+'</span>':'')+'<span style="flex:1"></span>'+
    '<button data-copy="'+esc(c.phrase)+'">Copy</button>'+(G.target?' <button data-use="'+esc(c.phrase)+'">Use</button>':'')+'</label>';});h+='</div>';});
  $('gout').innerHTML=h;
  $('gout').querySelectorAll('button[data-copy]').forEach(function(b){b.onclick=function(){var t=b.dataset.copy;
   (navigator.clipboard?navigator.clipboard.writeText(t):Promise.reject()).then(function(){toast('Copied: '+t);},function(){$('gq').value=t;toast('Clipboard blocked — the text is in the box');});};});
  $('gout').querySelectorAll('button[data-use]').forEach(function(b){b.onclick=function(){useGen(b.dataset.use);};});
 }).catch(function(e){$('gout').innerHTML='<p class="help" style="color:var(--bad)">'+esc(e.message||e)+'</p>';});}
function useGen(t){var x=G.target;if(!x)return;
 if(x.decided){var dc=$('dcustom');if(dc){dc.value=t;$('rlist').querySelector('input[name=de][value="__custom"]').checked=true;}$('gdlg').close();return;}var box=$('n'+x[0]+'_'+x[1]);if(!box){toast('That number is no longer open');return;}
 var c=$('c'+x[0]+'_'+x[1]);c.value=t;box.querySelector('input[value="__custom"]').checked=true;setChoice(x[0],x[1],t);
 var hit=numOf(x).candidates.findIndex(function(k){return k.text===t;});
 if(hit>=0){box.querySelectorAll('input[type=radio]')[hit].checked=true;c.value='';setChoice(x[0],x[1],t);}
 $('gdlg').close();toast('Reading set for '+numOf(x).core);}
$('gen').onclick=openGen;$('ggo').onclick=gen;$('gclose').onclick=function(){$('gdlg').close();};
$('gq').onkeydown=function(e){if(e.key==='Enter'){e.preventDefault();gen();}};
$('gq').oninput=function(){clearTimeout(G.timer);G.timer=setTimeout(gen,300);};
$('gstyle').onchange=gen;$('glang').onchange=gen;
$('bok').onclick=function(){$(B.target).value=$('bpath').value;$('browse').close();};

// ---------------------------------------------------------------- file list
function loadFiles(){return api('/api/files').then(function(r){S.files=r.files;renderFiles();});}
function fileKind(f){return f.kind==='verbalized'&&f.n_verified?'verified':f.kind;}   // finished: auto vs by hand
function renderFiles(){var k=$('kind').value,q=$('q').value.toLowerCase();
 var fs=S.files.filter(function(f){return fileKind(f)===k||(!k&&f.kind!=='verbalized')?(!q||f.file.toLowerCase().indexOf(q)>=0):false;});
 $('files').innerHTML=fs.length?fs.map(function(f){return '<div class="f'+(S.cur&&S.cur.file===f.file&&S.cur.kind===f.kind?' sel':'')+'" data-f="'+esc(f.file)+'" data-k="'+f.kind+'">'+
  '<div><span class="pill '+f.kind+'">'+fileKind(f)+'</span> <span class="n">'+esc(f.file)+'</span></div><div class="m">'+
  (f.kind==='verbalized'?(f.n_numbers||0)+' numbers'+(f.n_verified?' · '+f.n_verified+' verified by hand':' · all auto'):f.kind!=='failed'?(f.n_pending||0)+' open of '+(f.n_numbers||0)+' numbers':esc((f.stage||'')+': '+(f.error||'')))+
  (f.has_audio?'':' · no audio')+(f.has_copy===false?' · copy missing':'')+'</div></div>';}).join(''):'<p class="help" style="padding:10px">'+(k==='verbalized'?'No finished files decided by the model only.':k==='verified'?'No finished files verified by hand yet.':'Nothing left to check here 🎉 (finished files: pick <b>verbalized</b> or <b>verified</b> above)')+'</p>';
 $('files').querySelectorAll('.f').forEach(function(el){el.onclick=function(){openFile(el.dataset.f,el.dataset.k);};});}
$('kind').onchange=renderFiles;$('q').oninput=renderFiles;

// ---------------------------------------------------------------- one file
function openFile(name,kind){api('/api/file?file='+encodeURIComponent(name)+'&kind='+kind).then(function(d){
 S.cur={file:name,kind:kind};S.doc=d;S.choice={};S.flat=[];S.focus=0;
 d.sentences.forEach(function(s){S.choice[s.idx]=s.numbers.map(function(n){return n.model_pick;});s.numbers.forEach(function(n,j){S.flat.push([s.idx,j]);});});
 var au=$('au');au.pause();au.src=d.has_audio?'/api/audio?file='+encodeURIComponent(name):'';
 S.tab='check';S.pv={};S.edits={};S.editing=null;S.sedit=null;S.lastAuto=null;renderFile();renderFiles();$('foot').style.display='flex';preview();
 S.focus=0;setTab(S.flat.length?'check':'review');
}).catch(function(e){toast(e.message,6000);});}
function numOf(x){var s=S.doc.sentences.find(function(s){return s.idx===x[0];});return s.numbers[x[1]];}
function hl(text,cores){var t=esc(text);cores.forEach(function(c){var e=esc(c);t=t.split(e).join('<mark>'+e+'</mark>');});return t;}
function renderFile(){var d=S.doc,h='';
 h+='<h2 style="margin:0 0 4px">'+esc(d.file)+' <span class="pill '+d.kind+'">'+d.kind+'</span></h2>';
 if(d.kind==='verbalized')h+='<p class="help">Finished file: every number is decided. Change any of them under <b>Already decided</b> (Review tab); Save rewrites it in verbalized/.</p>';
 else if(d.kind==='failed')h+='<p class="help">Failed at <b>'+esc((d.info||{}).stage)+'</b>: '+esc((d.info||{}).error)+'. No model scores: candidates are in default order; the first is pre-selected.</p>';
 else h+='<p class="help">'+esc((d.info||{}).reasons||'')+(d.info&&d.info.total?' · '+d.info.decided+' of '+d.info.total+' numbers already decided (verbalized in this file)':'')+'</p>';
 if(!d.has_audio)h+='<p class="help">⚠ no audio for this file in the audio folder — you can still choose readings.</p>';
 if(d.error){h+='<p class="help" style="color:var(--bad)">'+esc(d.error)+'</p>';$('view').innerHTML=h;$('save').disabled=true;return;}
 h+='<p class="help">'+S.flat.length+' open number(s) in '+d.sentences.length+' of '+d.n_sentences+' sentences. The text below already has the decided numbers verbalized.</p>';
 h+='<div class="tabs"><button class="tab" id="tabC" title="one number at a time">1 · Check <span id="tcount" class="muted"></span></button>'+
    '<button class="tab" id="tabR" title="all choices, then save">2 · Review &amp; save</button><span class="sep"></span>'+
    '<button class="tab sents" id="tabS" title="the whole transcript: fix the text of any sentence (saved at once)">✎ Sentences</button></div>';
 h+='<div id="pane-check"><div class="stepbar"><button id="prev" title="← / K">◀ Prev</button><span id="step" class="muted"></span>'+
    '<span style="flex:1"></span><button class="primary" id="next" title="Enter / →">Next ▶</button></div>';
 d.sentences.forEach(function(s){h+='<div class="card" id="s'+s.idx+'"><div class="sh"><button data-play="'+s.idx+'" title="Space"'+(d.has_audio?'':' disabled')+'>▶ '+fmt(s.start)+' – '+fmt(s.end)+'</button>'+
  '<span>sentence '+s.idx+(s.speaker?' · speaker '+esc(s.speaker):'')+'</span></div>'+
  '<div class="orig">'+hl(s.text,s.numbers.map(function(n){return n.core;}))+'</div><div class="prev" id="p'+s.idx+'"></div>';
  s.numbers.forEach(function(n,j){var id=s.idx+'_'+j,ch=S.choice[s.idx][j],inC=n.candidates.some(function(c){return c.text===ch;});
   h+='<div class="num" id="n'+id+'"><h4><mark>'+esc(n.core)+'</mark> <span class="muted">'+esc(n.class)+'</span>'+(n.scored?'':'<span class="muted">(no model scores)</span>')+
    (n.review_margin?'<span class="muted" title="score gap to the best different reading (nats); below 1 = uncertain">review margin '+esc(n.review_margin)+'</span>':'')+(n.style_gap!==''&&n.style_gap!=null?' <span class="muted" title="spelling gap: best other spelling minus the audio\'s spelling (nats). Below style_margin -> default style (★); above -> the audio\'s spelling">spelling '+(n.style_fallback?'default':'audio')+' (gap '+esc(n.style_gap)+')</span>':'')+(n.suffix_issue?' <span class="warn" title="the written suffix may not be what was said - pick a bare reading or type the real one (e.g. അമ്പത്തയ്യായിരം രൂപയുടെ)">suffix: '+esc(n.suffix_issue)+'</span>':'')+(n.previous_pick?' <span class="warn" title="'+esc(n.reverify_note)+'">re-check: was '+esc(n.previous_pick)+'</span>':'')+(n.text_issue?' <span class="warn" title="currency sign apart: the sign stays next to your reading; non-Malayalam file: check the reading\'s language / script">'+esc(n.text_issue)+'</span>':'')+'</h4><div class="cands">';
   n.candidates.forEach(function(c,i){h+='<label'+(c.text===ch?' class="on"':'')+'><input type="radio" name="r'+id+'" value="'+esc(c.text)+'"'+(c.text===ch?' checked':'')+'> <span class="k">'+(i<9?i+1:'')+'</span> '+
    '<span>'+esc(c.text)+'</span>'+(c.text===n.model_pick&&d.kind!=='failed'?' <span class="star" title="pick (after the spelling rule)">★</span>':'')+(n.audio_pick&&n.audio_pick!==n.model_pick&&c.text===n.audio_pick?' <span class="star" title="the audio\'s own best (its spelling was too close to call, so the default style is pre-selected)">🔊</span>':'')+' <span class="muted" style="font-size:12px">'+esc(c.form)+'</span><span style="flex:1"></span><span class="d" title="nats less likely than the best candidate">'+(c.delta==null?'':(c.delta===0?'best':'Δ '+c.delta))+'</span></label>';});
   var isLater=ch===null,isCustom=!inC&&!isLater;
   h+='<label'+(isCustom?' class="on"':'')+'><input type="radio" name="r'+id+'" value="__custom"'+(isCustom?' checked':'')+'> <span class="k">0</span> <input class="custom" id="c'+id+'" placeholder="type the reading you hear" value="'+(isCustom?esc(ch):'')+'"></label>'+
    '<label'+(isLater?' class="on"':'')+'><input type="radio" name="r'+id+'" value="__later"'+(isLater?' checked':'')+'> <span class="k">L</span> <span class="muted">later — leave this number open</span></label></div></div>';});
  h+='</div>';});
 h+='</div><div id="pane-review" style="display:none">'+
    '<div class="savebar"><button class="primary" id="save" title="Ctrl+Enter">Save</button><span id="rstatus" class="muted"></span>'+
    '<span style="flex:1"></span><button id="back">◀ Back to checking</button></div><div id="rlist"></div></div>'+
    '<div id="pane-sent" style="display:none"><p class="help">Every sentence of the file. <b>Edit</b> fixes its text (wrong words, typos); it is saved at once, '+
    'separately from the numbers, and logged in sentence_edits.tsv. Numbers are written in words; the sentence\'s words get new timings spread over its span '+
    '(approximate). A sentence with open numbers can be edited after they are decided.</p><div id="slist"></div></div>';
 $('view').innerHTML=h;$('save').disabled=false;$('save').onclick=save;
 $('tabC').onclick=function(){setTab('check');};$('tabR').onclick=function(){setTab('review');};$('tabS').onclick=function(){setTab('sent');};
 $('prev').onclick=function(){go(S.focus-1);};$('next').onclick=next;$('back').onclick=function(){setTab('check');};
 $('view').querySelectorAll('button[data-play]').forEach(function(b){b.onclick=function(){play(+b.dataset.play);};});
 $('view').querySelectorAll('input[type=radio]').forEach(function(r){r.onchange=function(){var m=r.name.slice(1).split('_'),si=+m[0],j=+m[1];
  if(r.value==='__custom'){var c=$('c'+si+'_'+j);setChoice(si,j,c.value);c.focus();}else if(r.value==='__later')setChoice(si,j,null);else setChoice(si,j,r.value);};});
 $('view').querySelectorAll('input.custom').forEach(function(c){c.oninput=function(){var m=c.id.slice(1).split('_');var rr=c.closest('label').querySelector('input[type=radio]');rr.checked=true;setChoice(+m[0],+m[1],c.value);};
  c.onfocus=function(){var m=c.id.slice(1).split('_');var i=S.flat.findIndex(function(x){return x[0]===+m[0]&&x[1]===+m[1];});if(i>=0){S.focus=i;markFocus(false);}};});
}
function fmt(t){return t==null?'?':(+t).toFixed(2)+'s';}
function setChoice(si,j,v){S.choice[si][j]=v;var box=$('n'+si+'_'+j);box.querySelectorAll('label').forEach(function(l){var r=l.querySelector('input[type=radio]');l.classList.toggle('on',r.checked);});clearTimeout(S.pt);S.pt=setTimeout(preview,150);}
function preview(){if(!S.cur||S.doc.error)return;api('/api/preview',{file:S.cur.file,kind:S.cur.kind,picks:picks(),edits:S.edits}).then(function(r){
 if(r.invalid){$('status').textContent='⚠ '+r.invalid;$('save').disabled=true;return;}
 S.pv=r.preview;Object.keys(r.preview).forEach(function(k){var el=$('p'+k);if(el)el.innerHTML='→ '+esc(r.preview[k]);});
 var open=0;Object.keys(S.choice).forEach(function(k){S.choice[k].forEach(function(v){if(!String(v==null?'':v).trim())open++;});});
 var dec=S.flat.length-open,fin=S.cur.kind==='verbalized';$('status').textContent=fin?'finished file':dec+' of '+S.flat.length+' open number(s) decided'+(open?' · '+open+' left for later: Save keeps the file in verify/':' · Save moves the file to verbalized/');
 var ned=Object.keys(S.edits||{}).length;
 if(ned)$('status').textContent+=' · '+ned+' decided number(s) changed';
 if($('save')){$('save').disabled=!dec&&!ned;$('save').textContent=fin?'Save changes':open?'Save (keep '+open+' open)':'Save → verbalized';}
 if($('tcount'))$('tcount').textContent=dec+'/'+S.flat.length;if(S.tab==='review')renderReview();}).catch(function(){});}
function setTab(t){if(!$('pane-check'))return;S.tab=t;var c=t==='check',r=t==='review',s=t==='sent';
 $('pane-check').style.display=c?'':'none';$('pane-review').style.display=r?'':'none';$('pane-sent').style.display=s?'':'none';
 $('tabC').classList.toggle('on',c);$('tabR').classList.toggle('on',r);$('tabS').classList.toggle('on',s);
 if(c)markFocus(true);else if(r){renderReview();$('view').scrollTop=0;}else{renderSentences();$('view').scrollTop=0;}}

// ---------------------------------------------------------------- sentences (text edits)
function renderSentences(){var d=S.doc,L=d.all_sentences||[],h='';
 if(!d.can_edit_sentences)h='<p class="help">'+(d.kind==='failed'?'A failed file is the unchanged input: decide its numbers first (Save moves it to verify/), then edit sentences.':'This file has no sentence list, so its text can\'t be edited here.')+'</p>';
 L.forEach(function(s){var ed=S.sedit===s.idx;
  h+='<div class="card" id="ss'+s.idx+'"><div class="sh"><button data-splay="'+s.idx+'"'+(d.has_audio&&s.start!=null?'':' disabled')+'>▶ '+fmt(s.start)+' – '+fmt(s.end)+'</button>'+
   '<span>sentence '+s.idx+(s.speaker?' · speaker '+esc(s.speaker):'')+'</span>'+(s.edited?' <span class="pill done" title="edited by hand">edited</span>':'')+'<span style="flex:1"></span>'+
   (ed?'':(s.locked?'<span class="muted" title="decide its numbers first">open numbers</span>':'<button data-sedit="'+s.idx+'">Edit</button>'))+'</div>';
  if(ed)h+='<textarea class="sedit" id="stx">'+esc(s.text)+'</textarea><div class="savebar"><button class="primary" id="ssave" title="Ctrl+Enter">Save sentence</button><button id="scancel">Cancel</button>'+
     (d.has_audio&&s.start!=null?'<button id="splay2" title="Space outside the box">▶ Play</button>':'')+'<span class="help">numbers in words · punctuation is removed like the rest of the text</span></div>';
  else h+='<div class="stext">'+(s.locked?hl(s.text,(s.text.match(/[0-9\u0d66-\u0d6f][^\s]*/g)||[])):esc(s.text))+'</div>';
  if(s.edited&&s.original)h+='<div class="was">was: '+esc(s.original)+'</div>';
  h+='</div>';});
 $('slist').innerHTML=h;
 $('slist').querySelectorAll('button[data-splay]').forEach(function(b){b.onclick=function(){playS(+b.dataset.splay);};});
 $('slist').querySelectorAll('button[data-sedit]').forEach(function(b){b.onclick=function(){S.sedit=+b.dataset.sedit;renderSentences();var t=$('stx');if(t){t.focus();t.setSelectionRange(t.value.length,t.value.length);}autoPlay(S.sedit,true);};});
 if($('stx')){$('scancel').onclick=function(){S.sedit=null;renderSentences();};$('ssave').onclick=saveSentence;
  if($('splay2'))$('splay2').onclick=function(){playS(S.sedit);};
  var el=$('ss'+S.sedit);if(el)el.scrollIntoView({block:'center'});}}
function playS(si){var s=(S.doc.all_sentences||[]).find(function(x){return x.idx===si;});if(!s)return;
 var au=$('au');if(!au.src||!S.doc.has_audio){toast('no audio for this file');return;}var pad=parseFloat($('pad').value)||0;
 var go=function(){au.currentTime=Math.max(0,(+s.start||0)-pad);stopAt=(+s.end||0)+pad;au.playbackRate=+$('rate').value;safePlay(au);};
 if(au.readyState<1){au.preload='metadata';au.addEventListener('loadedmetadata',go,{once:true});au.load();}else go();}
function saveSentence(){var si=S.sedit,t=$('stx');if(si==null||!t||S.ssaving)return;var v=t.value.split(/\s+/).join(' ').trim();
 if(!v){toast('the sentence is empty');return;}if(/[0-9\u0d66-\u0d6f]/.test(v)){toast('write numbers in words (no digits)');return;}
 S.ssaving=true;$('ssave').disabled=true;
 api('/api/sentence',{file:S.cur.file,kind:S.cur.kind,rev:S.doc.rev,sentence:si,text:v}).then(function(r){
  S.doc.rev=r.rev;var s=S.doc.all_sentences.find(function(x){return x.idx===si;});
  if(!s.edited)s.original=r.before;s.text=r.text;s.edited=true;
  (S.doc.decided||[]).forEach(function(x){if(+x.sentence===si)x.sentence_edited=true;});
  S.sedit=null;renderSentences();toast('✓ sentence '+si+' saved'+(r.text!==v?' (normalized: '+r.text+')':''));
 }).catch(function(e){toast(e.message,7000);if(/changed since it was opened/.test(e.message))openFile(S.cur.file,S.cur.kind);})
 .then(function(){S.ssaving=false;if($('ssave'))$('ssave').disabled=false;});}
function go(i){if(!S.flat.length)return;S.focus=Math.max(0,Math.min(S.flat.length-1,i));if(S.tab!=='check')setTab('check');else markFocus(true);}
function next(){if(S.focus<S.flat.length-1)go(S.focus+1);else setTab('review');}
function renderReview(){var h='<table class="rv"><tr><th>#</th><th>Number</th><th>Reading</th><th>Sentence (as it will be written)</th><th></th></tr>';
 S.flat.forEach(function(x,i){var n=numOf(x),ch=S.choice[x[0]][x[1]],later=!String(ch==null?'':ch).trim();
  var inC=n.candidates.some(function(c){return c.text===ch;});
  var tag=later?'<span class="later">later — stays open</span>':esc(ch)+(ch===n.model_pick?' <span class="star" title="model pick">★</span>':(inC?'':' <span class="custompick">(typed)</span>'));
  h+='<tr><td>'+(i+1)+'</td><td><mark>'+esc(n.core)+'</mark><div class="muted" style="font-size:12px">sentence '+x[0]+(n.suffix_issue?' · <span class="warn">suffix</span>':'')+(n.text_issue?' · <span class="warn" title="'+esc(n.text_issue)+'">'+(/currency/.test(n.text_issue)?'₹ apart':'script')+'</span>':'')+'</div></td><td>'+tag+'</td>'+
   '<td>'+esc((S.pv||{})[x[0]]||'')+'</td><td><button data-edit="'+i+'">Edit</button></td></tr>';});
 h+='</table>';
 var D=S.doc.decided||[];
 if(D.length){h+='<h3 style="margin:18px 0 0;font-size:15px">Already decided ('+D.length+')</h3><p class="help" style="margin:2px 0 0">'+
   (S.doc.can_edit_decided?'Numbers verbalized automatically (auto) or by you earlier (verified). Change one if it is wrong; the file is rebuilt from the original input on Save.':
   '⚠ The original input file for this file was not found, so these can only be viewed. Set <b>Input folder</b> (top bar) to the data folder verbalize read (e.g. …\\data\\eval_frozen) and press Load.')+'</p>'+
   '<table class="rv"><tr><th>Number</th><th>Reading</th><th>Original sentence</th><th></th></tr>';
  D.forEach(function(r){var ed=S.edits[r.id],cur=ed!=null?ed:r.final_pick;
   h+='<tr><td><mark>'+esc(r.core)+'</mark><div class="muted" style="font-size:12px">sentence '+esc(r.sentence)+'</div></td><td>'+esc(cur)+
    (ed!=null?' <span class="custompick">(changed — was '+esc(r.final_pick)+')</span>':' <span class="pill '+(r.status==='verified'?'done':'verify')+'">'+esc(r.status)+'</span>')+'</td>'+
    '<td>'+esc(r.sentence_text)+'</td><td style="white-space:nowrap">'+(S.doc.has_audio&&r.start!=null&&r.start!==''?'<button data-dplay="'+esc(r.sentence)+'" title="play this sentence">▶</button> ':'')+(S.doc.can_edit_decided&&!r.sentence_edited?'<button data-dedit="'+esc(r.id)+'">Edit</button>':(r.sentence_edited?'<span class="muted" title="change it in the Sentences section">sentence edited</span>':''))+
    (ed!=null?' <button data-undo="'+esc(r.id)+'">Undo</button>':'')+'</td></tr>';
   if(S.editing===r.id){var opts=r.candidates.slice();if(opts.indexOf(r.final_pick)<0)opts.unshift(r.final_pick);
    h+='<tr><td></td><td colspan="3"><div class="cands">'+opts.map(function(c,i){return '<label'+(c===cur?' class="on"':'')+'><input type="radio" name="de" value="'+esc(c)+'"'+(c===cur?' checked':'')+'> '+esc(c)+
      (c===r.model_pick?' <span class="star">★</span>':'')+'</label>';}).join('')+
     '<label><input type="radio" name="de" value="__custom"'+(opts.indexOf(cur)<0?' checked':'')+'> <input class="custom" id="dcustom" placeholder="type the reading you hear" value="'+(opts.indexOf(cur)<0?esc(cur):'')+'"></label></div>'+
     '<div class="savebar">'+(S.doc.has_audio&&r.start!=null&&r.start!==''?'<button id="dplay" title="Space">▶ Play</button>':'')+'<button class="primary" id="dapply">Apply</button><button id="dcancel">Cancel</button>'+
     '<button id="dgen" title="candidate generator">🔢 Candidates…</button></div></td></tr>';}});
  h+='</table>';}
 $('rlist').innerHTML=h;
 $('rlist').querySelectorAll('button[data-edit]').forEach(function(b){b.onclick=function(){go(+b.dataset.edit);};});
 $('rlist').querySelectorAll('button[data-dplay]').forEach(function(b){b.onclick=function(){play(+b.dataset.dplay);};});
 $('rlist').querySelectorAll('button[data-dedit]').forEach(function(b){b.onclick=function(){S.editing=S.editing===b.dataset.dedit?null:b.dataset.dedit;renderReview();};});
 $('rlist').querySelectorAll('button[data-undo]').forEach(function(b){b.onclick=function(){delete S.edits[b.dataset.undo];S.editing=null;renderReview();preview();};});
 if($('dapply')){var dc=$('dcustom');dc.oninput=function(){$('rlist').querySelector('input[name=de][value="__custom"]').checked=true;};
  dc.onkeydown=function(e){if(e.key==='Enter'){e.preventDefault();$('dapply').click();}};
  $('dcancel').onclick=function(){S.editing=null;renderReview();};
  if($('dplay'))$('dplay').onclick=function(){var r=D.find(function(x){return x.id===S.editing;});if(r)play(+r.sentence);};
  $('dgen').onclick=function(){var r=D.find(function(x){return x.id===S.editing;});$('gq').value=r?r.core:'';openGen();};
  $('dapply').onclick=function(){var r=D.find(function(x){return x.id===S.editing;}),sel=$('rlist').querySelector('input[name=de]:checked');
   var v=!sel?'':(sel.value==='__custom'?dc.value:sel.value);v=v.split(/\s+/).join(' ').trim();
   if(!v){toast('choose or type a reading');return;}if(/[0-9\u0d66-\u0d6f]/.test(v)){toast('write the reading in words (no digits)');return;}
   if(v===r.final_pick)delete S.edits[r.id];else S.edits[r.id]=v;S.editing=null;renderReview();preview();};}
 $('rstatus').textContent=$('status').textContent;}
function markFocus(scroll){document.querySelectorAll('.card.cur').forEach(function(c){c.classList.remove('cur');});var x=S.flat[S.focus];if(!x)return;
 var c=$('s'+x[0]);c.classList.add('cur');document.querySelectorAll('.num').forEach(function(n){n.style.outline='';n.classList.remove('numcur');});var n=$('n'+x[0]+'_'+x[1]);n.style.outline='2px solid var(--accent)';n.style.outlineOffset='3px';n.classList.add('numcur');
 if($('step'))$('step').textContent='Number '+(S.focus+1)+' of '+S.flat.length+' · sentence '+x[0];
 if($('prev'))$('prev').disabled=S.focus===0;if($('next'))$('next').textContent=S.focus<S.flat.length-1?'Next ▶':'Review ▶';
 if(scroll)n.scrollIntoView({block:'center',behavior:'smooth'});
 autoPlay(x[0]);}

// ---------------------------------------------------------------- audio
function safePlay(au){var p=au.play();if(p&&p.catch)p.catch(function(){});}   // a blocked / interrupted play is not an error
function setAutoplay(on){S.autoplay=!!on;try{localStorage.setItem('verifier.autoplay',on?'1':'0');}catch(e){}
 var b=$('autoplay');if(b){b.textContent='▶ Autoplay: '+(on?'on':'off');b.classList.toggle('on',!!on);b.setAttribute('aria-pressed',on?'true':'false');}
 if(on){S.lastAuto=null;if(S.tab==='check'&&S.flat[S.focus])autoPlay(S.flat[S.focus][0]);else if(S.tab==='sent'&&S.sedit!=null)autoPlay(S.sedit,true);}
 else{var a=$('au');if(a&&!a.paused)a.pause();}}
// play a sentence once when it comes up (not again for the next number of the same sentence)
function autoPlay(si,sent){if(!S.autoplay||!S.cur||!S.doc||!S.doc.has_audio)return;var key=S.cur.file+'|'+(sent?'s':'n')+si;
 if(S.lastAuto===key)return;S.lastAuto=key;if(sent)playS(si);else play(si);}
$('autoplay').onclick=function(){setAutoplay(!S.autoplay);this.blur();};
try{S.autoplay=localStorage.getItem('verifier.autoplay')==='1';}catch(e){S.autoplay=false;}
setAutoplay(S.autoplay);
var stopAt=null;
function play(si){var au=$('au');if(!au.src||!S.doc.has_audio){toast('no audio for this file');return;}
 var s=S.doc.sentences.find(function(s){return s.idx===si;})||(S.doc.decided||[]).filter(function(r){return +r.sentence===si;}).map(function(r){return {start:r.start,end:r.end};})[0];
 if(!s||s.start==null){toast('no time for this sentence');return;}var pad=parseFloat($('pad').value)||0;
 var go=function(){au.currentTime=Math.max(0,(+s.start||0)-pad);stopAt=(+s.end||0)+pad;au.playbackRate=+$('rate').value;safePlay(au);};
 if(au.readyState<1){au.preload='metadata';au.addEventListener('loadedmetadata',go,{once:true});au.load();}else go();}
$('au').addEventListener('timeupdate',function(){if(stopAt!=null&&this.currentTime>=stopAt){this.pause();stopAt=null;}});
$('rate').onchange=function(){$('au').playbackRate=+this.value;};

// ---------------------------------------------------------------- save
function save(){if(!S.cur||!$('save')||$('save').disabled||S.saving)return;var done=S.cur;
 S.saving=true;$('save').disabled=true;                 // one save at a time (no double Ctrl+Enter)
 var old=S.files.slice(),i=old.findIndex(function(f){return f.file===done.file&&f.kind===done.kind;});
 api('/api/save',{file:done.file,kind:done.kind,rev:S.doc.rev,picks:picks(),edits:S.edits}).then(function(r){
  if(done.kind==='verbalized'){toast('✓ '+r.file+': '+r.decided+' change(s) saved in verbalized/'+(r.typed?' · '+r.typed+' typed reading(s) logged':''));
   openFile(done.file,'verbalized');return state();}
  toast((r.done?'✓ '+r.file+' → verbalized':'✓ '+r.file+': '+r.decided+' saved, '+r.open+' still open (in '+done.kind.replace('failed','verify')+'/)')+(r.typed?' · '+r.typed+' typed reading(s) saved to typed_readings.tsv':''));
  // the next file comes from the list as it was (only the saved file changed), so it opens right
  // away while the list and counts refresh alongside, instead of one request after the other
  var k=$('kind').value,after=old.slice(i+1).concat(old.slice(0,Math.max(i,0)));
  var next=after.find(function(f){return (k?fileKind(f)===k:f.kind!=='verbalized')&&f.file!==done.file;});
  if(next){openFile(next.file,next.kind);return state();}
  return state().then(function(){var again=!r.done&&S.files.find(function(f){return f.file===done.file;});   // only this file is left: reopen it
   if(again)openFile(again.file,again.kind);else{S.cur=null;$('view').innerHTML='<h2>All done 🎉</h2><p>Nothing left in this list.</p>';$('foot').style.display='none';}});
 }).catch(function(e){toast(e.message,7000);if(/changed since it was opened/.test(e.message))openFile(done.file,done.kind);else $('save').disabled=false;})
 .then(function(){S.saving=false;});}
// (the Save button lives in the Review tab; bound in renderFile)

// ---------------------------------------------------------------- keyboard
document.addEventListener('keydown',function(e){if($('browse').open||$('gdlg').open)return;
 var typing=e.target.tagName==='INPUT'&&e.target.type!=='radio'||e.target.tagName==='SELECT'||e.target.tagName==='TEXTAREA';
 if(!typing&&(e.key==='g'||e.key==='G')&&!e.ctrlKey&&!e.metaKey){e.preventDefault();openGen();return;}
 if(!typing&&(e.key==='a'||e.key==='A')&&!e.ctrlKey&&!e.metaKey&&!e.altKey){e.preventDefault();setAutoplay(!S.autoplay);return;}
 if(!S.cur)return;
 if(S.tab==='sent'){if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)&&$('stx')){e.preventDefault();saveSentence();}
  else if(e.key==='Escape'&&$('stx')){S.sedit=null;renderSentences();}
  else if(!typing&&e.key===' '&&S.sedit!=null){e.preventDefault();var a=$('au');if(!a.paused)a.pause();else playS(S.sedit);}return;}
 if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();if(S.tab==='check')setTab('review');else save();return;}
 if(typing){if(e.key==='Escape')e.target.blur();
  else if(e.key==='Enter'&&e.target.classList.contains('custom')&&S.tab==='check'){e.preventDefault();e.target.blur();next();}return;}
 if(S.tab!=='check'){if(e.key===' '){var a=$('au');e.preventDefault();if(!a.paused)a.pause();else if($('dplay'))$('dplay').click();}return;}
 var x=S.flat[S.focus];
 if(e.key==='Enter'&&e.target.tagName==='BUTTON')return;      // the focused button handles it
 if(e.key==='Enter'||e.key==='ArrowRight'){e.preventDefault();next();return;}
 if(e.key==='ArrowLeft'){e.preventDefault();go(S.focus-1);return;}
 if(e.key===' '){e.preventDefault();var au=$('au');if(!au.paused){au.pause();}else if(x)play(x[0]);}
 else if(e.key==='r'||e.key==='R'){if(x)play(x[0]);}
 else if(e.key==='ArrowDown'||e.key==='j'){e.preventDefault();go(S.focus+1);}
 else if(e.key==='ArrowUp'||e.key==='k'){e.preventDefault();go(S.focus-1);}
 else if(e.key==='['||e.key===']'){var r=$('rate'),i=r.selectedIndex+(e.key===']'?1:-1);r.selectedIndex=Math.max(0,Math.min(r.options.length-1,i));r.onchange();}
 else if(x&&(e.key==='l'||e.key==='L')){var bx=$('n'+x[0]+'_'+x[1]);bx.querySelector('input[value="__later"]').checked=true;setChoice(x[0],x[1],null);}
 else if(x&&/^[0-9]$/.test(e.key)){e.preventDefault();var n=numOf(x),box=$('n'+x[0]+'_'+x[1]);
  if(e.key==='0'){var c=$('c'+x[0]+'_'+x[1]);box.querySelector('input[value="__custom"]').checked=true;setChoice(x[0],x[1],c.value);c.focus();}
  else{var i=+e.key-1;if(i<n.candidates.length){var r=box.querySelectorAll('input[type=radio]')[i];r.checked=true;setChoice(x[0],x[1],n.candidates[i].text);}}}
});
state();
})();
</script></body></html>
"""
