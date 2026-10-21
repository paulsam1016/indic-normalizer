"""Self-contained interactive HTML reports (inline CSS/JS, no network): every table is sortable
(click a header) and filterable (box above it); ``numbers`` adds a per-number explorer with
method / reference / outcome / class / form / margin filters, search, paging and CSV export.

Report content is written as a small Markdown subset (headings, paragraphs, ``- `` lists, pipe
tables, `code`, **bold**, *italic*) and converted here, so report builders stay plain text."""

from __future__ import annotations

import html
import json
import re

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e2e2e5;--head:#f1f1f3;--hover:#f6f6f8;
--accent:#2f6fde;--ok:#1f7a4d;--okbg:#e5f4ec;--warn:#8a5a00;--warnbg:#fdf1d8;--bad:#b42318;--badbg:#fde7e5;--code:#f0f0f2}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#e8e8ea;--muted:#9a9aa2;--line:#2c2c31;--head:#1d1d21;
--hover:#1b1b1f;--accent:#7aa7ff;--ok:#5fd39a;--okbg:#12301f;--warn:#f0c060;--warnbg:#33280f;--bad:#ff8a80;--badbg:#3a1714;--code:#232328}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI","Noto Sans Malayalam","Nirmala UI",sans-serif}
main{max-width:1400px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 12px}h2{font-size:17px;margin:32px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
p,li{color:var(--fg)}p{margin:6px 0}.muted{color:var(--muted)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.5px}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0 16px;font-size:13px}nav.toc a{color:var(--accent);text-decoration:none}
.tbl{margin:8px 0 12px}.tbar{display:flex;gap:8px;align-items:center;margin-bottom:6px;flex-wrap:wrap}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;max-height:75vh;overflow-y:auto}
input,select,button{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button:hover{background:var(--hover)}
table{border-collapse:collapse;width:auto;min-width:min(100%,560px);font-size:13px}
#ex-table table{width:100%}
.scroll{width:fit-content;max-width:100%}#ex-table.scroll{width:auto}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--head);cursor:pointer;white-space:nowrap;user-select:none;z-index:1}
th[data-dir=asc]::after{content:" ▲";font-size:10px}th[data-dir=desc]::after{content:" ▼";font-size:10px}
tbody tr:hover{background:var(--hover)}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.wrap{min-width:260px;max-width:520px}
.o-success{color:var(--ok);background:var(--okbg)}.o-verify{color:var(--warn);background:var(--warnbg)}
.o-failed{color:var(--bad);background:var(--badbg)}.o-unchecked{color:var(--muted)}
.pill{display:inline-block;padding:0 7px;border-radius:10px;font-size:12px}
.stats{display:flex;flex-wrap:wrap;gap:6px 14px;margin:6px 0;font-size:13px}
.controls{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;margin:8px 0}
.controls label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:2px}
.pager{display:flex;gap:8px;align-items:center;margin:8px 0}
details.gloss{margin:6px 0 8px;font-size:13px}details.gloss summary{cursor:pointer;color:var(--accent)}
dl{display:grid;grid-template-columns:max-content 1fr;gap:3px 14px;margin:8px 0}dt{font-weight:600}dd{margin:0;color:var(--fg)}
p.cols{font-size:13px;color:var(--muted);margin:2px 0 6px}
"""

JS = r"""
(function(){
function val(td){var v=td.getAttribute('data-v');if(v!==null){var n=parseFloat(v);return isNaN(n)?v.toLowerCase():n;}
 var t=td.textContent.trim();var m=t.match(/^[-+]?\d[\d,]*(\.\d+)?/);if(m){return parseFloat(m[0].replace(/,/g,''));}return t.toLowerCase();}
function cmp(a,b){if(typeof a==='number'&&typeof b==='number')return a-b;if(typeof a==='number')return -1;if(typeof b==='number')return 1;return a<b?-1:a>b?1:0;}
document.querySelectorAll('table.sortable').forEach(function(tb){
 tb.querySelectorAll('thead th').forEach(function(th,i){th.addEventListener('click',function(){
  var dir=th.dataset.dir==='asc'?'desc':'asc';tb.querySelectorAll('thead th').forEach(function(h){delete h.dataset.dir;});th.dataset.dir=dir;
  var body=tb.tBodies[0];var rows=Array.prototype.slice.call(body.rows);
  rows.sort(function(a,b){var c=cmp(val(a.cells[i]),val(b.cells[i]));return dir==='asc'?c:-c;});rows.forEach(function(r){body.appendChild(r);});});});});
document.querySelectorAll('.tbl').forEach(function(w){var f=w.querySelector('.tfilter');if(!f)return;var tb=w.querySelector('table');var cnt=w.querySelector('.tcount');
 f.addEventListener('input',function(){var terms=f.value.toLowerCase().split(/\s+/).filter(Boolean);var n=0;
  Array.prototype.forEach.call(tb.tBodies[0].rows,function(r){var t=r.textContent.toLowerCase();var ok=terms.every(function(x){return t.indexOf(x)>=0;});r.style.display=ok?'':'none';if(ok)n++;});
  cnt.textContent=terms.length?(n+' of '+tb.tBodies[0].rows.length+' rows'):'';});});

var el=document.getElementById('numbers-data');if(!el)return;
var D=JSON.parse(el.textContent);var rows=D.rows.map(function(a){var o={};D.columns.forEach(function(c,i){o[c]=a[i];});return o;});
function $(id){return document.getElementById(id);}
function esc(s){return s==null?'':String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function fill(id,opts,sel){var s=$(id);s.innerHTML=opts.map(function(o){return '<option'+(o===sel?' selected':'')+'>'+esc(o)+'</option>';}).join('');}
function uniq(k){var m={};rows.forEach(function(r){if(r[k]!=null)m[r[k]]=1;});return Object.keys(m).sort();}
fill('ex-method',D.methods,D.default_method);fill('ex-ref',D.refs.length?D.refs:['(none)'],D.refs[0]||'(none)');
var KEYS=D.refs.length?['success','verify','failed','unchecked']:['accepted','verify'];fill('ex-outcome',['(all)'].concat(KEYS),'(all)');fill('ex-class',['(all)'].concat(uniq('class')),'(all)');
function hide(id){var e=$(id);if(e&&e.parentNode)e.parentNode.style.display='none';}
if(!D.refs.length)hide('ex-ref');if(D.methods.length<2){hide('ex-all');hide('ex-dis');}
var st={sort:null,dir:1,page:0};
function outcome(r,m,ref){if(!ref)return r[m+':review']===true?'verify':'accepted';if(r['hyp:'+ref]==null)return '';if(r[m+':review']===true)return 'verify';
 if(r[m+':match:'+ref]===true)return 'success';if(r['oracle:'+ref]===true)return 'failed';return 'unchecked';}
function fmt(x){return typeof x==='number'?(Math.round(x*1000)/1000):x;}
function cols(m,ref,all){var c=[
 {k:'file',l:'File',t:'Transcript / audio file name (stem). Hover a cell for the full name.'},
 {k:'sentence',l:'Sent',n:1,t:'Index of the sentence in the file (0 = first entry of "sentences").'},
 {k:'sent_start',l:'Start s',n:1,t:'Sentence start time in the audio, seconds. The whole sentence (plus pad_s) is what the model scored.'},
 {k:'sent_end',l:'End s',n:1,t:'Sentence end time in the audio, seconds.'},
 {k:'core',l:'Number',t:'The number as written in the train-normalized text (1,75,000 -> 175000; 23-ന്, 4:00-5:00 keep their punctuation).'},
 {k:'class',l:'Class',t:'Number type: integer, num_suffix (number + Malayalam suffix), time, ordinal, percent, range, decimal, date, year, phone, ...'},
 {k:'n_cands',l:'Cands',n:1,t:'How many candidate readings were scored for this number (at most max_candidates).'},
 {k:'default',l:'Default',t:'The reading you would get without audio: native Malayalam, contracted style (first candidate).'},
 {k:m,l:'Pick ('+m+')',pick:1,t:'The reading the method chose: the candidate with the lowest negative log-likelihood (most likely given the audio). Coloured by outcome.'},
 {k:m+':form',l:'Form',t:'How the pick reads: contracted / glide / spaced (Malayalam compound styles), english, ordinal, ordinal_th, half, hh_mm, oclock, drop_thousand, digit_by_digit, ...'},
 {k:m+':margin',l:'Margin',n:1,t:'Score of the runner-up minus score of the pick, in nats. Larger = more confident. Empty when there was only one candidate.'},
 {k:m+':review_margin',l:'Review margin',n:1,t:'Score of the best DIFFERENT reading minus the pick (spelling variants of the same reading, e.g. contracted vs spaced, do not count). Below margin_threshold -> needs review. Empty = no different reading exists.'},
 {k:'_outcome',l:'Outcome',t:ref?'success = not flagged and the pick is in the reference; verify = flagged needs_review; failed = not flagged, a candidate is in the reference but not the pick; unchecked = no candidate appears in the reference.':'accepted = not flagged (written as is); verify = flagged needs_review (review margin below the threshold) -> worth a listen.'}];
 if(ref)c.push({k:'hyp:'+ref,l:'Hyp ('+ref+')',w:1,t:'Reference hypothesis text that overlaps the sentence (+- hyp_pad_s), eval-normalized.'});
 if(all){D.methods.forEach(function(x){if(x!==m)c.push({k:x,l:x,t:'Pick of method '+x+' for comparison.'});});}
 if(D.columns.indexOf(m+':spelling')>=0){c.push({k:m+':style_gap',l:'Spelling gap',n:1,t:'Score of the best other spelling (contracted / glide / spaced) of the chosen reading minus the audio\'s spelling, nats. Empty = the reading has no spelling variants.'});
  c.push({k:m+':spelling',l:'Spelling',t:'audio = the audio\'s spelling was kept (gap >= style_margin); default = too close to call, the default style (contracted) was used.'});}
 if(D.columns.indexOf(m+':text_issue')>=0)c.push({k:m+':text_issue',l:'Text issue',t:'A problem around the number that its reading cannot fix: a currency sign written as a separate word (₹ 500, Rs. 500, 500 ₹) - the number is read alone and the sign stays; or a file with many non-Malayalam letters (English, Hindi ...), where the language of the reading is less certain. Always sent to verify.'});
 if(D.columns.indexOf(m+':suffix_gap')>=0)c.push({k:m+':suffix_gap',l:'Suffix gap',n:1,t:'num_suffix only: score of the best reading WITH the written suffix minus the best bare number, nats. Positive = the audio prefers the number without the suffix.'});
 if(D.columns.indexOf(m+':suffix_issue')>=0)c.push({k:m+':suffix_issue',l:'Suffix issue',t:'num_suffix only: why the written suffix looks wrong for what was said - the audio prefers the number without the suffix (27-ന് spoken ഇരുപത്തേഴ്), or a vowel suffix after -ം (55000-യുടെ: probably a dropped noun, അമ്പത്തയ്യായിരം രൂപയുടെ). Always sent to verify; type the real reading in the verifier.'});
 if(D.columns.indexOf('file_status')>=0)c.push({k:'file_status',l:'File status',t:'Where this number\'s file went: verbalized (all numbers confident, written verbalized), verify (some number needs review: input copied to verify/), failed.'});
 if(D.columns.indexOf('text_verbalized')>=0)c.push({k:'text_verbalized',l:'Verbalized sentence',w:1,t:'The sentence exactly as written to the output file: train-normalized, every number replaced by its pick.'});
 c.push({k:'sentence_text',l:'Sentence',w:1,t:'Original sentence text from the input JSON.'});
 c.push({k:'candidates',l:'Candidates',w:1,t:'Every reading that was scored, separated by | (first = default).'});return c;}
function current(){var m=$('ex-method').value,ref=D.refs.length?$('ex-ref').value:'',oc=$('ex-outcome').value,cl=$('ex-class').value,
 fm=$('ex-form').value,lg=$('ex-lang').value,q=$('ex-q').value.toLowerCase().split(/\s+/).filter(Boolean),
 mx=parseFloat($('ex-maxm').value),mn=parseFloat($('ex-minm').value),dis=$('ex-dis').checked,chg=$('ex-chg').checked;
 var out=rows.filter(function(r){r._outcome=outcome(r,m,ref);
  if(oc!=='(all)'&&r._outcome!==oc)return false;if(cl!=='(all)'&&r['class']!==cl)return false;
  if(fm!=='(all)'&&r[m+':form']!==fm)return false;if(lg!=='(all)'&&r[m+':lang']!==lg)return false;
  var mg=r[m+':margin'];if(!isNaN(mx)&&(mg==null||mg>mx))return false;if(!isNaN(mn)&&(mg==null||mg<mn))return false;
  if(chg&&r[m]===r['default'])return false;
  if(dis){var s={};D.methods.forEach(function(x){s[r[x]]=1;});if(Object.keys(s).length<2)return false;}
  if(q.length){var t=[r.file,r.core,r.candidates,r.sentence_text,r['hyp:'+ref],r[m]].join(' ').toLowerCase();
   if(!q.every(function(x){return t.indexOf(x)>=0;}))return false;}
  return true;});
 if(st.sort){var k=st.sort,d=st.dir;out.sort(function(a,b){var x=a[k],y=b[k];if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;
  return (typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y)))*d;});}
 return {m:m,ref:ref,rows:out};}
function formsFor(m){var f=['(all)'].concat((function(){var s={};rows.forEach(function(r){if(r[m+':form'])s[r[m+':form']]=1;});return Object.keys(s).sort();})());
 var cur=$('ex-form').value;fill('ex-form',f,f.indexOf(cur)>=0?cur:'(all)');}
function render(){var c=current(),per=parseInt($('ex-per').value,10),C=cols(c.m,c.ref,$('ex-all').checked);
 var pages=Math.max(1,Math.ceil(c.rows.length/per));if(st.page>=pages)st.page=pages-1;var sl=c.rows.slice(st.page*per,(st.page+1)*per);
 var tally={};KEYS.forEach(function(k){tally[k]=0;});c.rows.forEach(function(r){if(tally[r._outcome]!=null)tally[r._outcome]++;});
 var n=c.rows.length||1;$('ex-stats').innerHTML='<b>'+c.rows.length+'</b> of '+rows.length+' numbers'+KEYS.map(function(k){
  return ' · <span class="pill o-'+k+'">'+k+' '+tally[k]+' ('+(100*tally[k]/n).toFixed(1)+'%)</span>';}).join('');
 var h='<table><thead><tr>'+C.map(function(x){return '<th title="'+esc(x.t||'')+'" data-k="'+esc(x.k)+'"'+(st.sort===x.k?' data-dir="'+(st.dir>0?'asc':'desc')+'"':'')+'>'+esc(x.l)+'</th>';}).join('')+'</tr></thead><tbody>';
 $('ex-gloss').innerHTML=C.map(function(x){return '<dt>'+esc(x.l)+'</dt><dd>'+esc(x.t||'')+'</dd>';}).join('');
 sl.forEach(function(r){h+='<tr>'+C.map(function(x){var v=r[x.k];var cls=x.n?'num':(x.w?'wrap':'');
  if(x.k==='_outcome'&&v)return '<td><span class="pill o-'+v+'">'+v+'</span></td>';
  if(x.pick&&r._outcome!=='unchecked'&&r._outcome!=='accepted')cls+=' o-'+r._outcome;
  if(x.k==='file')return '<td title="'+esc(v)+'">'+esc(String(v).slice(0,8))+'…</td>';
  return '<td class="'+cls+'">'+esc(fmt(v))+'</td>';}).join('')+'</tr>';});
 $('ex-table').innerHTML=h+'</tbody></table>';$('ex-page').textContent='page '+(st.page+1)+' / '+pages;
 $('ex-table').querySelectorAll('th').forEach(function(th){th.addEventListener('click',function(){var k=th.dataset.k;
  if(st.sort===k)st.dir=-st.dir;else{st.sort=k;st.dir=1;}render();});});}
function csv(){var c=current(),C=cols(c.m,c.ref,$('ex-all').checked);var q=function(v){v=v==null?'':String(v);return /[",\n\t]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;};
 var s=C.map(function(x){return q(x.l);}).join(',')+'\n'+c.rows.map(function(r){return C.map(function(x){return q(r[x.k]);}).join(',');}).join('\n');
 var a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+s],{type:'text/csv;charset=utf-8'}));a.download='numbers_filtered.csv';a.click();}
formsFor($('ex-method').value);fill('ex-lang',['(all)','ml','en'],'(all)');
['ex-method'].forEach(function(id){$(id).addEventListener('change',function(){formsFor($(id).value);st.page=0;render();});});
['ex-ref','ex-outcome','ex-class','ex-form','ex-lang','ex-per','ex-all','ex-dis','ex-chg'].forEach(function(id){$(id).addEventListener('change',function(){st.page=0;render();});});
['ex-q','ex-maxm','ex-minm'].forEach(function(id){$(id).addEventListener('input',function(){st.page=0;render();});});
$('ex-prev').addEventListener('click',function(){if(st.page>0){st.page--;render();}});
$('ex-next').addEventListener('click',function(){st.page++;render();});
$('ex-csv').addEventListener('click',csv);
$('ex-reset').addEventListener('click',function(){['ex-q','ex-maxm','ex-minm'].forEach(function(i){$(i).value='';});
 ['ex-outcome','ex-class','ex-form','ex-lang'].forEach(function(i){$(i).value='(all)';});['ex-all','ex-dis','ex-chg'].forEach(function(i){$(i).checked=false;});
 st={sort:null,dir:1,page:0};render();});
render();})();
"""

EXPLORER = """
<h2 id="explorer">Numbers explorer</h2>
<p>One row per number found in the transcripts, with the reading that was chosen for it. Use the
controls to narrow the list, click a column header to sort (click again to reverse), and download
what you see as CSV. Hover any control or header for a short explanation.</p>
<details class="gloss" open><summary>How to use the controls</summary><dl>
 <dt>Method</dt><dd>Which scoring method's pick to show (compare report: tdt / ctc × segment / word / window; verbalize report: the one method used).</dd>
 <dt>Reference</dt><dd>Compare report only: which hypothesis set the pick is checked against (decides the Outcome).</dd>
 <dt>Outcome</dt><dd>With a reference — <b>success</b>: not flagged and the pick appears in the reference; <b>verify</b>: flagged needs_review; <b>failed</b>: not flagged, a candidate appears in the reference but not the pick (confident and wrong); <b>unchecked</b>: no candidate appears in the reference. Without a reference (verbalize) — <b>accepted</b>: not flagged, written as is; <b>verify</b>: flagged, worth a listen.</dd>
 <dt>Class</dt><dd>Number type in the text: integer, num_suffix (23-ന്), time, ordinal, percent, range, decimal, date, year, phone, …</dd>
 <dt>Form / Language</dt><dd>How the chosen reading is written (contracted, english, ordinal, hh_mm, …) and whether it is Malayalam (ml) or English (en) number words.</dd>
 <dt>Margin ≥ / ≤</dt><dd>Keep numbers whose margin (runner-up score − pick score, nats) is in this range. E.g. ≤ 2 shows the less certain picks.</dd>
 <dt>Search</dt><dd>Words to find (all must match) in the number, candidates, pick, sentence, hypothesis or file name.</dd>
 <dt>Rows per page</dt><dd>Page size; use ◀ Prev / Next ▶ below the table to move between pages.</dd>
 <dt>show every method's pick</dt><dd>Compare report: add a column per method so picks can be compared side by side.</dd>
 <dt>methods disagree</dt><dd>Compare report: only numbers where the methods picked different readings.</dd>
 <dt>pick ≠ default</dt><dd>Only numbers where the audio changed the reading away from the no-audio default.</dd>
 <dt>Reset / Download CSV</dt><dd>Clear all filters and sorting / save the filtered rows (all pages, visible columns) as a CSV.</dd>
 <dt>Counts line</dt><dd>How many numbers pass the filters, and how many of them fall in each outcome (with %).</dd>
 <dt>Colours</dt><dd>The Pick cell is green for success, amber for verify, red for failed.</dd>
</dl></details>
<div class="controls">
 <label title="Which scoring method's pick to show">Method<select id="ex-method"></select></label>
 <label title="Hypothesis set the pick is checked against (decides the Outcome)">Reference<select id="ex-ref"></select></label>
 <label title="success / verify / failed / unchecked (with a reference) or accepted / verify (without)">Outcome<select id="ex-outcome"></select></label>
 <label title="Number type in the text">Class<select id="ex-class"></select></label>
 <label title="How the chosen reading is written">Form (of pick)<select id="ex-form"></select></label>
 <label title="ml = Malayalam number words, en = English number words">Language (of pick)<select id="ex-lang"></select></label>
 <label title="Only numbers with margin at least this (nats)">Margin ≥<input id="ex-minm" type="number" step="0.1" placeholder="nats"></label>
 <label title="Only numbers with margin at most this (nats); small = less certain">Margin ≤<input id="ex-maxm" type="number" step="0.1" placeholder="nats"></label>
 <label title="All words must appear in number / candidates / pick / sentence / hypothesis / file">Search<input id="ex-q" placeholder="number, word, file…"></label>
 <label title="Rows shown per page">Rows per page<select id="ex-per"><option>50</option><option selected>100</option><option>250</option><option>1000</option></select></label>
</div>
<div class="tbar">
 <label title="Add a column with each method's pick"><input type="checkbox" id="ex-all"> show every method's pick</label>
 <label title="Only numbers where the methods picked different readings"><input type="checkbox" id="ex-dis"> methods disagree</label>
 <label title="Only numbers where the pick differs from the no-audio default reading"><input type="checkbox" id="ex-chg"> pick ≠ default</label>
 <button id="ex-reset" title="Clear all filters and sorting">Reset</button><button id="ex-csv" title="Save the filtered rows as CSV">Download CSV (filtered)</button>
</div>
<details class="gloss" open><summary>What the columns mean (also shown when you hover a header)</summary><dl id="ex-gloss"></dl></details>
<div class="stats" id="ex-stats"></div>
<div class="scroll" id="ex-table"></div>
<div class="pager"><button id="ex-prev">◀ Prev</button><span id="ex-page" class="muted"></span><button id="ex-next">Next ▶</button></div>
"""


def _inline(text: str) -> str:
    parts = text.split("`")
    out = []
    for i, p in enumerate(parts):
        if i % 2:
            out.append("<code>" + html.escape(p) + "</code>")
        else:
            p = html.escape(p)
            p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
            p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", p)
            out.append(p)
    return "".join(out)


def _cells(line: str):
    """Split a pipe-table row, ignoring pipes inside `code`."""
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    cells, cur, code = [], [], False
    for ch in s:
        if ch == "`":
            code = not code
        if ch == "|" and not code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def _slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") or "s"


def md_to_html(md: str):
    """Returns (body_html, [(id, heading)]) for the Markdown subset the reports use."""
    lines = md.split("\n")
    out, toc, i, para = [], [], 0, []

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            head = _cells(block[0])
            body = [_cells(b) for b in block[2:]] if len(block) > 1 and set(block[1].replace("|", "").strip()) <= set("-: ") else [_cells(b) for b in block[1:]]
            t = ['<div class="tbl"><div class="tbar"><input class="tfilter" placeholder="Filter rows…"><span class="tcount muted"></span></div>',
                 '<div class="scroll"><table class="sortable"><thead><tr>']
            t += [f"<th>{_inline(h)}</th>" for h in head]
            t.append("</tr></thead><tbody>")
            for r in body:
                t.append("<tr>" + "".join(
                    f'<td class="num">{_inline(c)}</td>' if re.match(r"^[-+]?\d[\d,.]*(\s|%|x|$)", c) else f"<td>{_inline(c)}</td>"
                    for c in r) + "</tr>")
            t.append("</tbody></table></div></div>")
            out.append("".join(t))
            continue
        if s.startswith("#"):
            flush()
            lvl = len(s) - len(s.lstrip("#"))
            txt = s[lvl:].strip()
            if lvl == 1:
                out.append(f"<h1>{_inline(txt)}</h1>")
            else:
                sid = _slug(txt)
                toc.append((sid, re.sub(r"`", "", txt)))
                out.append(f'<h{min(lvl, 4)} id="{sid}">{_inline(txt)}</h{min(lvl, 4)}>')
        elif s.startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append("<li>" + _inline(lines[i].strip()[2:]) + "</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return "\n".join(out), toc


def _finite(x):
    """Copy with NaN / +-inf floats replaced by None."""
    if isinstance(x, float):
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def page(title: str, md: str, numbers=None) -> str:
    """Full HTML document from Markdown-subset ``md``. ``numbers``: optional dict with ``columns``,
    ``rows`` (list of lists), ``methods``, ``refs``, ``default_method`` -> adds the explorer."""
    body, toc = md_to_html(md)
    extra = ""
    if numbers is not None:
        toc = [("explorer", "Numbers explorer")] + toc
        # JSON inside <script>: escape every '<' (no </script>, no <!-- script-state tricks) and
        # never emit NaN / Infinity (invalid JSON for JSON.parse)
        data = json.dumps(_finite(numbers), ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).replace("<", "\\u003c")
        extra = EXPLORER + f'<script type="application/json" id="numbers-data">{data}</script>'
    nav = '<nav class="toc">' + "".join(f'<a href="#{i}">{html.escape(t)}</a>' for i, t in toc) + "</nav>"
    # put the nav right after the <h1>
    if body.startswith("<h1>"):
        h1_end = body.index("</h1>") + 5
        body = body[:h1_end] + nav + body[h1_end:]
    else:
        body = nav + body
    if numbers is not None:          # explorer after the summary paragraph(s) + first table
        body = body + extra
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{body}</main>"
            f"<script>{JS}</script></body></html>")


def numbers_payload(rows, methods, refs, default_method):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    return {"columns": cols, "rows": [[r.get(c) for c in cols] for r in rows],
            "methods": methods, "refs": refs, "default_method": default_method}
