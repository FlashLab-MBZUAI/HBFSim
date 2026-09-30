"""Measure GPU <-> pinned host DRAM copies through the CUDA driver API.

The benchmark needs only the NVIDIA driver (libcuda) and Python's standard
library, so it runs on any GPU node without a CUDA toolkit or PyTorch. It
records the same quantities as the DANA A100 offload anchor
(evidence/hardware/dana_a100_offload), so evidence/hardware/host_link/fit.py
can derive an HBFSim host-DRAM timing overlay with the same method:

- ``wall_time_ns_per_operation``: host wall time per copy while back-to-back
  asynchronous copies stream on one stream (the pipelined issue interval);
- ``latency_p50_ns``: host wall time of one copy plus stream synchronization.

Directions follow the offload anchor: ``offload_d2h`` (GPU to host) and
``restore_h2d`` (host to GPU).

Usage: python3 bench.py --out result.json [--device 0]
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import platform
import statistics
import time

SIZES = (4096, 65536, 1 << 20, 4 << 20, 16 << 20, 64 << 20)
DIRECTIONS = ('offload_d2h', 'restore_h2d')
MAX_STREAMS = 4


class Driver:
    def __init__(self, device, streams=MAX_STREAMS):
        self.lib = ctypes.CDLL('libcuda.so.1')
        c = ctypes
        for name, argtypes in {
            'cuInit': [c.c_uint],
            'cuDeviceGet': [c.POINTER(c.c_int), c.c_int],
            'cuDeviceGetName': [c.c_char_p, c.c_int, c.c_int],
            'cuDriverGetVersion': [c.POINTER(c.c_int)],
            'cuCtxCreate_v2': [c.POINTER(c.c_void_p), c.c_uint, c.c_int],
            'cuMemAllocHost_v2': [c.POINTER(c.c_void_p), c.c_size_t],
            'cuMemAlloc_v2': [c.POINTER(c.c_uint64), c.c_size_t],
            'cuStreamCreate': [c.POINTER(c.c_void_p), c.c_uint],
            'cuStreamSynchronize': [c.c_void_p],
            'cuEventCreate': [c.POINTER(c.c_void_p), c.c_uint],
            'cuEventRecord': [c.c_void_p, c.c_void_p],
            'cuEventSynchronize': [c.c_void_p],
            'cuEventElapsedTime': [c.POINTER(c.c_float), c.c_void_p, c.c_void_p],
            'cuMemcpyHtoDAsync_v2': [c.c_uint64, c.c_void_p, c.c_size_t, c.c_void_p],
            'cuMemcpyDtoHAsync_v2': [c.c_void_p, c.c_uint64, c.c_size_t, c.c_void_p],
        }.items():
            getattr(self.lib, name).argtypes = argtypes
            getattr(self.lib, name).restype = c.c_int
        self.call('cuInit', 0)
        self.device = c.c_int()
        self.call('cuDeviceGet', c.byref(self.device), device)
        name = c.create_string_buffer(256)
        self.call('cuDeviceGetName', name, 256, self.device)
        self.name = name.value.decode()
        version = c.c_int()
        self.call('cuDriverGetVersion', c.byref(version))
        self.driver_version = version.value
        self.context = c.c_void_p()
        self.call('cuCtxCreate_v2', c.byref(self.context), 0, self.device)
        # One disjoint host/device region per stream so concurrent copies never alias.
        self.host = c.c_void_p()
        self.call('cuMemAllocHost_v2', c.byref(self.host), max(SIZES) * streams)
        self.device_buffer = c.c_uint64()
        self.call('cuMemAlloc_v2', c.byref(self.device_buffer), max(SIZES) * streams)
        self.streams = []
        for _ in range(streams):
            stream = c.c_void_p()
            self.call('cuStreamCreate', c.byref(stream), 1)  # CU_STREAM_NON_BLOCKING
            self.streams.append(stream)
        self.stream = self.streams[0]
        self.start, self.stop = c.c_void_p(), c.c_void_p()
        self.call('cuEventCreate', c.byref(self.start), 0)
        self.call('cuEventCreate', c.byref(self.stop), 0)

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f'{name} failed with CUDA driver status {status}')

    def copy(self, direction, nbytes, lane=0):
        offset = lane * max(SIZES)
        host = ctypes.c_void_p(self.host.value + offset)
        device = self.device_buffer.value + offset
        if direction == 'restore_h2d':
            self.call('cuMemcpyHtoDAsync_v2', device, host, nbytes, self.streams[lane])
        else:
            self.call('cuMemcpyDtoHAsync_v2', host, device, nbytes, self.streams[lane])

    def gpu_elapsed_ns(self):
        milliseconds = ctypes.c_float()
        self.call('cuEventElapsedTime', ctypes.byref(milliseconds), self.start, self.stop)
        return milliseconds.value * 1e6


def measure(driver, direction, block, repeats, trial_bytes):
    for _ in range(8):
        driver.copy(direction, block)
    driver.call('cuStreamSynchronize', driver.stream)
    count = max(16, min(8192, trial_bytes // block))
    streams = []
    for _ in range(repeats):
        driver.call('cuEventRecord', driver.start, driver.stream)
        began = time.perf_counter_ns()
        for _ in range(count):
            driver.copy(direction, block)
        driver.call('cuEventRecord', driver.stop, driver.stream)
        driver.call('cuEventSynchronize', driver.stop)
        streams.append(((time.perf_counter_ns() - began) / count, driver.gpu_elapsed_ns() / count))
    singles = []
    for _ in range(max(repeats, 50)):
        driver.call('cuEventRecord', driver.start, driver.stream)
        began = time.perf_counter_ns()
        driver.copy(direction, block)
        driver.call('cuEventRecord', driver.stop, driver.stream)
        driver.call('cuStreamSynchronize', driver.stream)
        singles.append((time.perf_counter_ns() - began, driver.gpu_elapsed_ns()))
    wall = statistics.median(s[0] for s in streams)
    single_wall = sorted(s[0] for s in singles)
    return dict(path='gpu_pinned_host', direction=direction, block_bytes=block,
                operations_per_trial=count, trials=repeats,
                measured=dict(
                    wall_time_ns_per_operation=wall,
                    gpu_event_ns_per_operation=statistics.median(s[1] for s in streams),
                    wall_bandwidth_GBps=block / wall,
                    latency_p50_ns=statistics.median(single_wall),
                    latency_p90_ns=single_wall[int(.9 * (len(single_wall) - 1))],
                    gpu_event_latency_p50_ns=statistics.median(s[1] for s in singles)))


def concurrent_bandwidth(driver, direction, block, lanes, trial_bytes, repeats):
    """Aggregate GB/s with copies spread round-robin over ``lanes`` streams."""
    count = max(16 * lanes, min(8192, trial_bytes // block))
    results = []
    for _ in range(repeats):
        began = time.perf_counter_ns()
        for index in range(count):
            driver.copy(direction, block, index % lanes)
        for lane in range(lanes):
            driver.call('cuStreamSynchronize', driver.streams[lane])
        results.append(count * block / (time.perf_counter_ns() - began))
    return statistics.median(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--trial-bytes', type=int, default=1 << 30)
    args = parser.parse_args()
    driver = Driver(args.device)
    rows = [measure(driver, direction, block, args.repeats, args.trial_bytes)
            for direction in DIRECTIONS for block in SIZES]
    concurrency = [dict(direction=direction, block_bytes=block, streams=lanes,
                        aggregate_GBps=concurrent_bandwidth(driver, direction, block, lanes, args.trial_bytes, 5))
                   for direction in DIRECTIONS for block in (1 << 20, 4 << 20, 16 << 20)
                   for lanes in (1, 2, 4)]
    document = dict(schema='hbfsim.host_link_measurement.v1',
        measured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        host=platform.node(), gpu=driver.name, cuda_driver_version=driver.driver_version,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), python=platform.python_version(),
        method='CUDA driver API, page-locked host buffer, non-blocking streams; host wall clock per copy',
        rows=rows, concurrency=concurrency)
    with open(args.out, 'w') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')
    for row in rows:
        m = row['measured']
        print(f"{row['direction']:12s} {row['block_bytes']:>9d} B  stream {m['wall_bandwidth_GBps']:7.2f} GB/s  "
              f"single p50 {m['latency_p50_ns'] / 1e3:8.2f} us")
    for row in concurrency:
        print(f"{row['direction']:12s} {row['block_bytes']:>9d} B  {row['streams']} streams "
              f"{row['aggregate_GBps']:7.2f} GB/s")


if __name__ == '__main__':
    main()
