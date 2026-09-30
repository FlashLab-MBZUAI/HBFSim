#include "app/hbf_wear_report.hpp"
#include <atomic>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace hbfsim::app {
std::filesystem::path default_hbf_wear_prefix() {
    static std::atomic<unsigned> sequence{0};
    const auto clock = std::chrono::system_clock::now().time_since_epoch();
    return std::filesystem::absolute(std::filesystem::path("out/hbf-wear") /
        (std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(clock).count()) +
         "-" + std::to_string(getpid()) + "-" + std::to_string(sequence++)) / "wear");
}

void write_hbf_wear_report(const std::filesystem::path& prefix, std::string_view snapshot) {
    if (!prefix.parent_path().empty()) std::filesystem::create_directories(prefix.parent_path());
    std::ofstream json(prefix.string() + ".json");
    json << snapshot << '\n';
    json.close();
    if (!json) throw std::runtime_error("failed to write HBF wear JSON");
    std::ofstream html(prefix.string() + ".html");
    html << R"HTML(<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>HBF device wear</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#19283a;font:15px/1.5 system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:38px 40px}h1{font-size:34px;letter-spacing:-1px;margin:3px 0 9px}
h2{font-size:19px;margin:0 0 16px}.eyebrow{color:#53647b;letter-spacing:1.5px;font-size:12px}
p{color:#53647b;margin:8px 0 20px}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:26px 0}
.card,section{background:white;border:1px solid #e0e6ef;border-radius:12px;padding:20px}.card span{display:block;font-size:12px;color:#53647b}.card strong{font-size:25px;font-weight:620}
section{margin:18px 0;padding:25px}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:20px}
select{padding:8px;border:1px solid #d4dce7;border-radius:6px;background:white;color:#19283a}
.row{display:flex;align-items:center;margin:3px 0;gap:10px}.rowlabel{width:75px;flex:none;font-size:12px;color:#53647b}.cells{display:flex;gap:2px;flex:1;min-height:20px}.cell{min-width:1px;flex:1;border-radius:2px;min-height:20px;cursor:crosshair}.cell:hover{outline:2px solid #16283e;z-index:1}.stack{margin:18px 0 7px;font-weight:600;font-size:13px}
.legend{display:flex;align-items:center;gap:12px;font-size:12px;color:#53647b;margin-top:18px}.gradient{width:200px;height:11px;border-radius:3px;background:linear-gradient(90deg,#e3edf5,#92aec9,#777db6,#aa678e,#ed884e)}
#detail{min-height:66px;background:#f4f6fa;border-radius:8px;padding:12px 16px;margin-top:18px;font-size:13px;font-variant-numeric:tabular-nums}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e7edf3}th{color:#53647b;font-weight:500}.muted{font-size:12px;color:#65758a}footer{font-size:12px;color:#65758a;margin:28px 0}a{color:#425cb1}#status{padding:12px 16px;background:#edf3f8;border-radius:8px}
@media(max-width:800px){main{padding:20px}.cards{grid-template-columns:repeat(2,1fr)}section{padding:16px}.rowlabel{width:55px}.cells{gap:1px}.cell{min-height:16px}td,th{padding:6px}}
</style><main><div class="eyebrow">WORKLOAD RESULT · PHYSICAL MEDIA</div><h1>HBF device wear</h1>
<p>Physical P/E history stays with the NAND blocks when host software remaps zones.</p>
<div class="cards" id="cards"></div><div id="status"></div>
<section><h2>Wear across the device</h2><div class="controls"><label>Metric <select id="metric"><option value="mean">Lifetime mean P/E</option><option value="maximum">Lifetime maximum P/E</option><option value="workload">Workload erases / block</option></select></label><label>Stack <select id="stack"><option value="all">All stacks</option></select></label><span class="muted" id="geometry"></span></div><div id="heat"></div><div class="legend"><span>0</span><div class="gradient"></div><span id="scale"></span><span>One shared color scale across all stacks</span></div><div id="detail">Hover over a cell to inspect physical coordinates, wear and page occupancy.</div></section>
<section><h2>Most worn physical regions</h2><div style="overflow:auto"><table><thead><tr><th>Stack / channel</th><th>Physical zones</th><th>Max P/E</th><th>Mean P/E</th><th>Workload erases</th><th>Blocks</th></tr></thead><tbody id="hot"></tbody></table></div></section>
<section><h2>Host zone remapping</h2><p class="muted">Only changed assignments appear below. Every omitted local zone maps to the same physical zone. Remapping requires invalid data and a completed IO frontier; it performs no media copies.</p><div id="mapping"></div></section>
<footer><a href="https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf">OCP HBF v0.7.0 §11.4, Figure 46</a> · Model geometry and host timing are simulator assumptions. This report shows observed wear, not a calibrated lifetime prediction. Offline artifact; no external scripts.</footer>
<script id="snapshot" type="application/json">)HTML" << snapshot << R"HTML(</script><script>
'use strict';const d=JSON.parse(document.getElementById('snapshot').textContent),bins=d.bins;
const fmt=x=>Number(x).toLocaleString(undefined,{maximumFractionDigits:3});
const sum=k=>bins.reduce((n,b)=>n+b[k],0),maxpec=Math.max(0,...bins.map(b=>b.pec_max));
const cards=[['Workload block erases',sum('workload_erases')],['Lifetime P/E sum',sum('pec_sum')],['Maximum block P/E',maxpec],['Host zone swaps',d.host_zone_remaps],['Host GC roundtrip GiB',(d.host_gc_read_bytes+d.host_gc_write_bytes)/2**30]];
for(const [label,value] of cards){const c=document.createElement('div');c.className='card';const a=document.createElement('span');a.textContent=label;const b=document.createElement('strong');b.textContent=fmt(value);c.append(a,b);document.getElementById('cards').append(c)}
document.getElementById('status').textContent=sum('workload_erases')===0?'This workload issued no erase commands. Zero incremental wear is an observed result.':'Includes all erase commands issued during this workload and its terminal drain. Restored media history is shown separately in each cell.';
document.getElementById('geometry').textContent=`${d.stacks} stacks × ${d.channels_per_stack} channels · ${fmt(d.blocks)} blocks · ${fmt(d.zone_size_blocks)} blocks / zone`;
for(let i=0;i<d.stacks;i++){const o=document.createElement('option');o.value=i;o.textContent=`Stack ${i}`;document.getElementById('stack').append(o)}
const colors=[[227,237,245],[146,174,201],[119,125,182],[170,103,142],[237,136,78]];
function color(x){const t=Math.min(4,Math.max(0,x*4)),i=Math.min(3,Math.floor(t)),f=t-i;return `rgb(${colors[i].map((v,j)=>Math.round(v+(colors[i+1][j]-v)*f)).join(',')})`}
function render(){const metric=document.getElementById('metric').value,selected=document.getElementById('stack').value;const value=b=>metric==='maximum'?b.pec_max:(metric==='workload'?b.workload_erases:b.pec_sum)/(b.block_end-b.block_begin);const max=Math.max(0,...bins.map(value));document.getElementById('scale').textContent=fmt(max);const heat=document.getElementById('heat');heat.replaceChildren();
for(let s=0;s<d.stacks;s++){if(selected!=='all'&&Number(selected)!==s)continue;const title=document.createElement('div');title.className='stack';title.textContent=`STACK ${s}`;heat.append(title);for(let c=0;c<d.channels_per_stack;c++){const row=document.createElement('div');row.className='row';const label=document.createElement('div');label.className='rowlabel';label.textContent=`Channel ${c}`;const cells=document.createElement('div');cells.className='cells';for(const b of bins.filter(b=>b.stack===s&&b.channel===c)){const cell=document.createElement('div');cell.className='cell';cell.style.background=color(max?value(b)/max:0);cell.style.flexGrow=b.block_end-b.block_begin;const text=`Stack ${s} / Channel ${c} · Physical zones [${b.physical_zone_begin}, ${b.physical_zone_end}) · Blocks [${b.block_begin}, ${b.block_end}) | P/E min ${fmt(b.pec_min)}, mean ${fmt(b.pec_sum/(b.block_end-b.block_begin))}, max ${fmt(b.pec_max)} | Lifetime sum ${fmt(b.pec_sum)} = restored ${fmt(b.initial_pec_sum)} + workload ${fmt(b.workload_erases)} | Pages: valid ${fmt(b.valid_pages)}, invalid ${fmt(b.invalid_pages)}, free ${fmt(b.free_pages)}, pending ${fmt(b.pending_pages)} | Static blocks ${fmt(b.static_blocks)}; raw / zone blocks ${fmt(b.raw_blocks)}`;cell.title=text;cell.onmouseenter=()=>{document.getElementById('detail').textContent=text};cells.append(cell)}row.append(label,cells);heat.append(row)}}}
function tableRows(parent,rows){for(const row of rows){const tr=document.createElement('tr');for(const value of row){const td=document.createElement('td');td.textContent=value;tr.append(td)}parent.append(tr)}}
tableRows(document.getElementById('hot'),[...bins].sort((a,b)=>b.pec_max-a.pec_max||b.pec_sum-a.pec_sum).filter(b=>b.pec_max>0).slice(0,8).map(b=>[`${b.stack} / ${b.channel}`,`[${b.physical_zone_begin}, ${b.physical_zone_end})`,fmt(b.pec_max),fmt(b.pec_sum/(b.block_end-b.block_begin)),fmt(b.workload_erases),fmt(b.block_end-b.block_begin)]));
if(!maxpec)tableRows(document.getElementById('hot'),[['No block has accumulated a P/E cycle.']]);
const map=document.getElementById('mapping');if(!d.zone_remapping.length)map.textContent='All zone assignments are identity.';else{const table=document.createElement('table');tableRows(table,[['Stack','Channel','Local zone','Physical zone'],...d.zone_remapping.map(r=>[r.stack,r.channel,r.local_zone,r.physical_zone])]);map.append(table)}
document.getElementById('metric').onchange=render;document.getElementById('stack').onchange=render;render();
</script></main></html>)HTML";
    html.close();
    if (!html) throw std::runtime_error("failed to write HBF wear HTML");
}
} // namespace hbfsim::app
