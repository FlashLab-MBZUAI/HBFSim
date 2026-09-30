"""Fit an HBFSim host-DRAM timing overlay from host_link measurements.

Uses the same fit as the DANA A100 offload anchor
(evidence/hardware/dana_a100_offload/calibrate.py): the pipelined small-copy
interval becomes the controller issue time, a least-squares slope over block
size gives each direction's link bandwidth, and the median single-copy latency
residual becomes the fixed processing time. Repeated runs are combined by the
median of each (direction, block) point, so one contended run cannot move the
fit.

Usage:
  python -m evidence.hardware.host_link.fit --platform "..." --out OVERLAY.cfg RUN.json [RUN.json ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from evidence.hardware.dana_a100_offload.calibrate import fit_host, overlay_lines


def fit(documents):
    points = {}
    for document in documents:
        if document.get('schema') != 'hbfsim.host_link_measurement.v1':
            raise ValueError('expected host_link measurement documents')
        for row in document['rows']:
            points.setdefault((row['direction'], row['block_bytes']), []).append(row['measured'])
    groups = {key: dict(path='gpu_pinned_host', direction=key[0], block_bytes=key[1], measured=dict(
                  wall_time_ns_per_operation=statistics.median(m['wall_time_ns_per_operation'] for m in values),
                  latency_p50_ns=statistics.median(m['latency_p50_ns'] for m in values)))
              for key, values in points.items()}
    return fit_host(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--platform', required=True, help='one-line platform description for the overlay header')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    documents = [json.loads(path.read_text()) for path in args.runs]
    config = fit(documents)
    header = [
        f'# GPU <-> pinned host DRAM timing: {args.platform}.',
        '# Apply after the complete host-dram profile; usable capacity is unchanged.',
        f'# Fit of {len(documents)} evidence/hardware/host_link runs (median per direction and',
        '# block) with the DANA A100 offload method: small-copy issue interval, per-direction',
        '# link bandwidth slope, median fixed-latency residual. Media fields are',
        '# non-bottleneck sentinels: the directional link models the host transfer once.',
    ]
    args.out.write_text('\n'.join(header + overlay_lines(config)) + '\n')
    print(json.dumps({key: config[key] for key in ('controller_issue_ns', 'controller_processing_ns',
                                                    'm2s_bandwidth_GBps', 's2m_bandwidth_GBps')}, indent=2))


if __name__ == '__main__':
    main()
