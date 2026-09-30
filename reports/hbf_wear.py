#!/usr/bin/env python3
"""Export a standalone PNG/SVG physical HBF wear figure from per-run JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def render(snapshot: dict, output: Path) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    if snapshot.get('schema') != 'hbfsim.hbf_wear.v1':
        raise ValueError('expected hbfsim.hbf_wear.v1')
    bins=snapshot['bins']
    groups=[(s,c) for s in range(snapshot['stacks']) for c in range(snapshot['channels_per_stack'])]
    rows=[[b for b in bins if (b['stack'],b['channel'])==key] for key in groups]
    if not rows or any(not row or len(row)!=len(rows[0]) for row in rows):
        raise ValueError('missing or unequal physical channel rows')
    if sum(b['block_end']-b['block_begin'] for b in bins)!=snapshot['blocks']:
        raise ValueError('physical block coverage diverged')
    for b in bins:
        if (b['pec_sum'] != b['initial_pec_sum']+b['workload_erases'] or
                b['pec_min'] > b['pec_max']):
            raise ValueError('physical PEC history diverged')
    cmap=LinearSegmentedColormap.from_list('hbf',['#e3edf5','#92aec9','#777db6','#aa678e','#ed884e'])
    height=max(5.3, 3+len(rows)*.34)
    fig,axes=plt.subplots(1,2,figsize=(13,height))
    fig.set_facecolor('#f4f6fa')
    fig.subplots_adjust(left=.13,right=.94,top=.66,bottom=.34,wspace=.42)
    total=sum(b['workload_erases'] for b in bins)
    maximum=max(b['pec_max'] for b in bins)
    fig.text(.065,.92,'HBF device wear',fontsize=25,color='#19283a',weight='bold')
    fig.text(.065,.85,f"{snapshot['stacks']} stacks  /  {snapshot['channels_per_stack']} channels per stack  /  "
             f"{snapshot['blocks']:,} physical blocks  /  {snapshot['zone_size_blocks']} blocks per zone",
             fontsize=11,color='#53647b')
    fig.text(.065,.77,f"{total:,} workload block erases     {maximum:,} maximum P/E     "
             f"{snapshot['host_zone_remaps']:,} host zone swaps",fontsize=13,color='#19283a')
    for ax,field,title in zip(axes,['pec_sum','workload_erases'],['Lifetime mean P/E','Workload erases / block']):
        values=np.array([[b[field]/(b['block_end']-b['block_begin']) for b in row] for row in rows])
        maximum_value=float(values.max())
        x_edges=np.array([b['physical_zone_begin'] for b in rows[0]]+[rows[0][-1]['physical_zone_end']])
        im=ax.pcolormesh(x_edges,np.arange(len(rows)+1)-.5,values,cmap=cmap,
                         vmin=0,vmax=maximum_value or 1,shading='flat',rasterized=True)
        ax.invert_yaxis()
        ax.set_title(title,loc='left',fontsize=13,pad=14,color='#19283a')
        ax.set_yticks(range(len(rows)),[f'Stack {s} / Ch {c}' for s,c in groups],fontsize=10)
        stride=max(1,len(rows[0])//8)
        ticks=list(range(0,len(rows[0]),stride))
        ax.set_xticks([(x_edges[i]+x_edges[i+1])/2 for i in ticks],
                      [rows[0][i]['physical_zone_begin'] for i in ticks],fontsize=10)
        ax.set_xlabel('Physical zone (cell start)',fontsize=10,labelpad=10)
        ax.tick_params(length=0,pad=8,colors='#53647b')
        for spine in ax.spines.values():spine.set_visible(False)
        bounds=ax.get_position()
        cb_axis=fig.add_axes([bounds.x0+.06,.19,bounds.width-.12,.025])
        cb=fig.colorbar(im,cax=cb_axis,orientation='horizontal')
        cb.outline.set_visible(False);cb.ax.tick_params(labelsize=9,length=0,colors='#53647b')
        if maximum_value==0:cb.set_ticks([0])
    fig.text(.065,.07,'P/E history stays with physical NAND blocks. Zone remapping does not move wear.\n'
             'Workload increments exclude restored history and include the terminal drain.',
             fontsize=10,color='#53647b')
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.suffix.lower() not in {'.png','.svg'}:
        parser.error('--output must be PNG or SVG')
    render(json.loads(args.input.read_text()),args.output)


if __name__=='__main__':main()
