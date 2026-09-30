#!/usr/bin/env python3
"""Verify that all three downloaded OCP v0.7.0 grades reach actual transport."""
from __future__ import annotations
import argparse,json,math,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulator',type=Path,required=True)
    p.add_argument('--config-root',type=Path,default=Path('configs'))
    p.add_argument('--registry',type=Path,default=Path('configs/parameter-provenance.json'))
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    registry=json.loads(a.registry.read_text())
    if registry['sources']['ocp_hbf_v070']['pages']!=130:raise AssertionError('unverified artifact')
    rows=[]
    for grade,channels,rate in [(1,8,8),(2,16,16),(3,16,32)]:
        path=a.output_dir/f'grade{grade}.json'
        command=[str(a.simulator.resolve()),'--config',str(a.config_root/'systems/eight-stack-baseline.cfg'),
                 '--config',str(a.config_root/f'overlays/hbf/ocp-v070-grade{grade}.cfg'),
                 '--hbf-stacks','1','--hbf-blocks-per-plane','16','--hbf-pages-per-block','16',
                 '--synthetic-sequential-read-bytes',str(16*4096),'--line-size','4096',
                 '--interarrival-ns','0','--max-hbf-outstanding-requests','16',
                 '--scenarios','direct-read','--static-direct-hbm-bytes','0',
                 '--summary-json',str(path)]
        run=subprocess.run(command,text=True,capture_output=True)
        if run.returncode:raise RuntimeError(run.stdout+run.stderr)
        data=json.loads(path.read_text());cfg=data['config']['hbf'];s=data['scenarios'][0]
        if data['schema']!={'name':'hbfsim.simulation.summary','version':19}:raise AssertionError('summary schema')
        per_channel=64*rate/8*0.75
        if cfg['hbio_bw_GBps']!=channels*per_channel:raise AssertionError('grade bandwidth')
        stats=s['hbf_stats'];read_bytes=stats['physical_read_bytes']
        if read_bytes!=16*4096 or not stats['accounting_verified']:raise AssertionError('media accounting')
        busy=s['time_breakdown']['resource_busy']['hbf_hbio_data']['busy_ns']
        # Busy work sums over channels; dividing by aggregate stack bandwidth
        # would lose channel resource multiplicity.
        if not math.isclose(busy,read_bytes/per_channel,rel_tol=1e-10):raise AssertionError('transport did not use grade')
        rows.append({'grade':grade,'payload_GBps_per_channel':per_channel,'payload_GBps_per_stack':channels*per_channel,'read_bytes':read_bytes,'channel_data_busy_ns':busy})
    (a.output_dir/'result.json').write_text(json.dumps({'source_sha256':registry['sources']['ocp_hbf_v070']['sha256'],'grades':rows},indent=2)+'\n')
    print('OCP v0.7.0 grades 1/2/3 reach independent channel transport: PASS')
    return 0
if __name__=='__main__':raise SystemExit(main())
