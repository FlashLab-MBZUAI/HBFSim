#!/usr/bin/env python3
"""Measure A100 offload to pinned host DRAM and compute-local NVMe.

The production path intentionally depends only on Python's standard library and
the cluster-provided PyTorch.  SSD measurements use page-aligned buffers with
O_DIRECT preadv/pwritev; an unsupported direct-I/O path is an error rather than
silently falling back to the page cache.
"""

import argparse
import ctypes
import datetime
import hashlib
import json
import math
import mmap
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
DIRECT_ALIGNMENT = 4096
DEFAULT_BLOCK_SIZES = (4096, 65536, 1048576, 16777216)
DEFAULT_LANES = (1, 8)
NORMAL_DRAM_TARGETS = {
    4096: 32 * 1024 * 1024,
    65536: 128 * 1024 * 1024,
    1048576: 512 * 1024 * 1024,
    16777216: 4 * 1024 * 1024 * 1024,
}
NORMAL_NVME_TARGETS = {
    4096: 64 * 1024 * 1024,
    65536: 256 * 1024 * 1024,
    1048576: 1024 * 1024 * 1024,
    16777216: 4 * 1024 * 1024 * 1024,
}
QUICK_DRAM_TARGETS = {
    4096: 1 * 1024 * 1024,
    65536: 4 * 1024 * 1024,
    1048576: 16 * 1024 * 1024,
    16777216: 128 * 1024 * 1024,
}
QUICK_NVME_TARGETS = {
    4096: 1 * 1024 * 1024,
    65536: 4 * 1024 * 1024,
    1048576: 16 * 1024 * 1024,
    16777216: 128 * 1024 * 1024,
}
KNOWN_NONLOCAL_FILESYSTEMS = {
    "9p",
    "afs",
    "ceph",
    "cifs",
    "fuse.sshfs",
    "gpfs",
    "lustre",
    "nfs",
    "nfs4",
    "overlay",
    "ramfs",
    "smb3",
    "tmpfs",
}


class BenchmarkError(RuntimeError):
    """The requested measurement contract could not be satisfied."""


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def run_command(argv: Sequence[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=15,
            check=False,
        )
        return {
            "argv": list(argv),
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": list(argv), "error": repr(error)}


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise BenchmarkError("cannot calculate a percentile of no samples")
    if not 0.0 <= quantile <= 1.0:
        raise BenchmarkError("quantile must be in [0, 1]")
    position = (len(sorted_values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def latency_summary(values_ns: Sequence[float]) -> Dict[str, Any]:
    if not values_ns:
        raise BenchmarkError("latency sample set is empty")
    ordered = sorted(float(value) for value in values_ns)
    result = {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
    }
    result["stdev"] = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return result


def throughput_summary(byte_count: int, wall_time_ns: int) -> Dict[str, Any]:
    if byte_count <= 0 or wall_time_ns <= 0:
        raise BenchmarkError("throughput requires positive bytes and wall time")
    seconds = wall_time_ns / 1.0e9
    return {
        "bytes": byte_count,
        "wall_time_ns": wall_time_ns,
        "GB_per_s": byte_count / seconds / 1.0e9,
        "GiB_per_s": byte_count / seconds / float(1024 ** 3),
    }


def parse_positive_csv(value: str, option_name: str) -> Tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("{} must contain integers".format(option_name))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("{} values must be positive".format(option_name))
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("{} contains duplicates".format(option_name))
    return parsed


def pattern_payload(label: str, size: int) -> bytes:
    seed = hashlib.sha256(label.encode("utf-8")).digest()
    repetitions, remainder = divmod(size, len(seed))
    return seed * repetitions + seed[:remainder]


class MmapBuffer:
    """Anonymous mmap with a buffer-protocol view suitable for O_DIRECT."""

    def __init__(self, size: int, alignment: int) -> None:
        self.mapping = mmap.mmap(-1, size)
        self.view = memoryview(self.mapping)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        if self.address % alignment:
            self.close()
            raise BenchmarkError(
                "anonymous mmap address is not {}-byte aligned".format(alignment)
            )

    def close(self) -> None:
        if getattr(self, "view", None) is not None:
            self.view.release()
            self.view = None
        if getattr(self, "mapping", None) is not None:
            self.mapping.close()
            self.mapping = None


class PinnedBuffer:
    """Aligned view into CUDA-pinned torch storage, also usable by preadv."""

    def __init__(self, torch_module: Any, size: int, alignment: int) -> None:
        self.torch = torch_module
        self.base = torch_module.empty(
            size + alignment, dtype=torch_module.uint8, pin_memory=True
        )
        base_address = int(self.base.data_ptr())
        offset = (-base_address) % alignment
        self.tensor = self.base.narrow(0, offset, size)
        self.address = int(self.tensor.data_ptr())
        if self.address % alignment:
            raise BenchmarkError(
                "pinned tensor address is not {}-byte aligned".format(alignment)
            )
        array_type = ctypes.c_ubyte * size
        self.array = array_type.from_address(self.address)
        self.view = memoryview(self.array).cast("B")

    def close(self) -> None:
        if getattr(self, "view", None) is not None:
            self.view.release()
            self.view = None
        self.array = None
        self.tensor = None
        self.base = None


def pwrite_all_direct(fd: int, view: memoryview, offset: int) -> None:
    completed = 0
    while completed < len(view):
        written = os.pwritev(fd, [view[completed:]], offset + completed)
        if written <= 0:
            raise BenchmarkError("pwritev made no forward progress")
        completed += written
        if completed < len(view) and completed % DIRECT_ALIGNMENT:
            raise BenchmarkError("pwritev returned a non-aligned partial direct write")


def pread_all_direct(fd: int, view: memoryview, offset: int) -> None:
    completed = 0
    while completed < len(view):
        received = os.preadv(fd, [view[completed:]], offset + completed)
        if received == 0:
            raise BenchmarkError(
                "unexpected EOF after {} of {} bytes".format(completed, len(view))
            )
        if received < 0:
            raise BenchmarkError("preadv returned a negative byte count")
        completed += received
        if completed < len(view) and completed % DIRECT_ALIGNMENT:
            raise BenchmarkError("preadv returned a non-aligned partial direct read")


def decode_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def mount_metadata(path: str) -> Dict[str, Any]:
    absolute = os.path.realpath(path)
    best: Optional[Dict[str, Any]] = None
    text = read_text("/proc/self/mountinfo")
    if text:
        for line in text.splitlines():
            fields = line.split()
            try:
                separator = fields.index("-")
            except ValueError:
                continue
            if len(fields) <= separator + 2:
                continue
            mount_point = decode_mount_field(fields[4])
            try:
                inside = os.path.commonpath([absolute, mount_point]) == mount_point
            except ValueError:
                inside = False
            if not inside:
                continue
            candidate = {
                "mount_point": mount_point,
                "major_minor": fields[2],
                "mount_options": fields[5].split(","),
                "filesystem_type": fields[separator + 1],
                "source": decode_mount_field(fields[separator + 2]),
                "super_options": fields[separator + 3].split(",")
                if len(fields) > separator + 3
                else [],
            }
            if best is None or len(mount_point) > len(best["mount_point"]):
                best = candidate
    if best is None:
        best = {
            "mount_point": None,
            "major_minor": "{}:{}".format(
                os.major(os.stat(absolute).st_dev), os.minor(os.stat(absolute).st_dev)
            ),
            "filesystem_type": None,
            "source": None,
        }
    stat = os.statvfs(absolute)
    best["statvfs"] = {
        "block_size": stat.f_bsize,
        "fragment_size": stat.f_frsize,
        "blocks": stat.f_blocks,
        "blocks_available": stat.f_bavail,
        "bytes_available": stat.f_bavail * stat.f_frsize,
    }
    return best


def block_device_metadata(path: str) -> Dict[str, Any]:
    device = os.stat(path).st_dev
    major_minor = "{}:{}".format(os.major(device), os.minor(device))
    entry = "/sys/dev/block/{}".format(major_minor)
    result: Dict[str, Any] = {
        "major_minor": major_minor,
        "sysfs_entry": entry,
        "sysfs_available": os.path.exists(entry),
        "devices": [],
        "is_nvme_backed": None,
    }
    if not os.path.exists(entry):
        return result

    visited = set()

    queue_fields = (
        "max_hw_sectors_kb",
        "max_sectors_kb",
        "optimal_io_size",
        "minimum_io_size",
        "logical_block_size",
        "physical_block_size",
        "read_ahead_kb",
        "nr_requests",
        "rotational",
    )

    def queue_metadata(device_path: str, device_name: str) -> Dict[str, Any]:
        queue_directory = os.path.join(device_path, "queue")
        parent = device_path
        while not os.path.isdir(queue_directory):
            next_parent = os.path.dirname(parent)
            if next_parent == parent or next_parent == "/sys":
                break
            parent = next_parent
            queue_directory = os.path.join(parent, "queue")
        sys_block_queue = "/sys/block/{}/queue".format(device_name)
        values: Dict[str, Any] = {
            "sysfs_path": queue_directory,
            "sys_block_path": sys_block_queue
            if os.path.isdir(sys_block_queue)
            else None,
        }
        for field in queue_fields:
            raw = read_text(os.path.join(queue_directory, field))
            if raw is None:
                values[field] = None
                continue
            try:
                values[field] = int(raw)
            except ValueError:
                values[field] = raw
        return values

    def visit(sysfs_path: str) -> None:
        real = os.path.realpath(sysfs_path)
        if real in visited:
            return
        visited.add(real)
        name = os.path.basename(real)
        queue = queue_metadata(real, name)
        parent = os.path.dirname(queue["sysfs_path"])
        model = read_text(os.path.join(real, "device", "model"))
        if model is None:
            model = read_text(os.path.join(parent, "device", "model"))
        device_record = {
            "name": name,
            "sysfs_path": real,
            "sys_block_path": "/sys/block/{}".format(name)
            if os.path.exists("/sys/block/{}".format(name))
            else None,
            "model": model,
            "queue": queue,
            "looks_like_nvme": name.startswith("nvme")
            or "/nvme" in real.lower(),
        }
        result["devices"].append(device_record)
        slaves = os.path.join(real, "slaves")
        if os.path.isdir(slaves):
            for slave in sorted(os.listdir(slaves)):
                visit(os.path.join(slaves, slave))

    visit(entry)
    result["is_nvme_backed"] = any(
        bool(item["looks_like_nvme"]) for item in result["devices"]
    )
    return result


def proc_status_subset() -> Dict[str, str]:
    wanted = {
        "Cpus_allowed_list",
        "Mems_allowed_list",
        "VmPeak",
        "VmPin",
        "VmRSS",
    }
    result: Dict[str, str] = {}
    text = read_text("/proc/self/status")
    if text:
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key in wanted:
                result[key] = value.strip()
    return result


def cpu_metadata() -> Dict[str, Any]:
    model = None
    text = read_text("/proc/cpuinfo")
    if text:
        for line in text.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    return {
        "logical_cpu_count": os.cpu_count(),
        "model_name": model,
        "affinity": affinity,
        "process_status": proc_status_subset(),
        "meminfo": read_text("/proc/meminfo"),
    }


def git_metadata(repo_root: str) -> Dict[str, Any]:
    revision = run_command(["git", "-C", repo_root, "rev-parse", "HEAD"])
    status = run_command(
        ["git", "-C", repo_root, "status", "--porcelain", "--untracked-files=no"]
    )
    return {
        "root": repo_root,
        "revision": revision.get("stdout") if revision.get("returncode") == 0 else None,
        "tracked_worktree_dirty": bool(status.get("stdout"))
        if status.get("returncode") == 0
        else None,
        "revision_command": revision,
        "status_command": status,
    }


def torch_environment(torch_module: Any) -> Dict[str, Any]:
    devices = []
    if torch_module.cuda.is_available():
        for index in range(torch_module.cuda.device_count()):
            properties = torch_module.cuda.get_device_properties(index)
            devices.append(
                {
                    "logical_index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "multiprocessor_count": properties.multi_processor_count,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    return {
        "version": torch_module.__version__,
        "cuda_build_version": torch_module.version.cuda,
        "cuda_available": torch_module.cuda.is_available(),
        "cuda_device_count": torch_module.cuda.device_count()
        if torch_module.cuda.is_available()
        else 0,
        "current_device": torch_module.cuda.current_device()
        if torch_module.cuda.is_available()
        else None,
        "devices": devices,
        "config": torch_module.__config__.show(),
    }


def base_environment(work_dir: str, torch_module: Any) -> Dict[str, Any]:
    script_path = os.path.realpath(__file__)
    repo_root = os.path.realpath(os.path.join(os.path.dirname(script_path), "../../.."))
    slurm_names = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_SUBMIT_DIR",
    )
    return {
        "host": {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": platform.platform(),
            "uname": list(platform.uname()),
        },
        "python": {
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "torch": torch_environment(torch_module),
        "cpu": cpu_metadata(),
        "slurm": {name: os.environ.get(name) for name in slurm_names},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,pci.bus_id,driver_version,pstate,clocks.current.sm,clocks.current.memory,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi_topology": run_command(["nvidia-smi", "topo", "-m"]),
        "storage": {
            "work_dir": os.path.realpath(work_dir),
            "mount": mount_metadata(work_dir),
            "block_device": block_device_metadata(work_dir),
            "direct_alignment_bytes": DIRECT_ALIGNMENT,
            "o_direct_value": getattr(os, "O_DIRECT", None),
        },
        "artifact": {
            "benchmark_path": script_path,
            "benchmark_sha256": sha256_file(script_path),
            "git": git_metadata(repo_root),
        },
    }


def require_environment(
    args: argparse.Namespace, torch_module: Any, environment: Dict[str, Any]
) -> None:
    if sys.version_info < (3, 8):
        raise BenchmarkError("Python 3.8 or newer is required")
    if args.expect_python and not platform.python_version().startswith(
        args.expect_python + "."
    ):
        raise BenchmarkError(
            "expected Python {}.x, found {}".format(
                args.expect_python, platform.python_version()
            )
        )
    if args.expect_torch and torch_module.__version__ != args.expect_torch:
        raise BenchmarkError(
            "expected torch {}, found {}".format(
                args.expect_torch, torch_module.__version__
            )
        )
    if not torch_module.cuda.is_available():
        raise BenchmarkError("CUDA is unavailable")
    device_name = torch_module.cuda.get_device_name(torch_module.cuda.current_device())
    if args.require_gpu_substring and args.require_gpu_substring.lower() not in device_name.lower():
        raise BenchmarkError(
            "GPU {!r} does not contain required name {!r}".format(
                device_name, args.require_gpu_substring
            )
        )
    if not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv") or not hasattr(
        os, "pwritev"
    ):
        raise BenchmarkError("Linux O_DIRECT, preadv, and pwritev are required")
    real_work = os.path.realpath(args.work_dir)
    if args.require_tmp:
        try:
            under_tmp = os.path.commonpath([real_work, "/tmp"]) == "/tmp"
        except ValueError:
            under_tmp = False
        if not under_tmp or real_work == "/tmp":
            raise BenchmarkError("work directory must be a job-specific child of /tmp")
    filesystem = environment["storage"]["mount"].get("filesystem_type")
    if args.require_local_storage and filesystem in KNOWN_NONLOCAL_FILESYSTEMS:
        raise BenchmarkError(
            "work directory filesystem {!r} is not a local block filesystem".format(
                filesystem
            )
        )
    is_nvme = environment["storage"]["block_device"].get("is_nvme_backed")
    if args.require_nvme and is_nvme is not True:
        raise BenchmarkError(
            "could not prove that {} is backed by an NVMe block device".format(
                real_work
            )
        )


def target_operations(target_bytes: int, block_bytes: int, lanes: int) -> Tuple[int, int]:
    operations_per_lane = max(1, int(math.ceil(target_bytes / float(block_bytes * lanes))))
    total_operations = operations_per_lane * lanes
    return operations_per_lane, total_operations


def measure_cuda_direction(
    torch_module: Any,
    direction: str,
    block_bytes: int,
    operations: int,
    latency_operations: int,
    warmups: int,
    host_source: PinnedBuffer,
    host_destination: PinnedBuffer,
    gpu_tensor: Any,
) -> Dict[str, Any]:
    stream = torch_module.cuda.Stream()

    def copy_once() -> None:
        if direction == "offload_d2h":
            host_destination.tensor.copy_(gpu_tensor, non_blocking=True)
        elif direction == "restore_h2d":
            gpu_tensor.copy_(host_source.tensor, non_blocking=True)
        else:
            raise BenchmarkError("unknown CUDA copy direction {}".format(direction))

    for _ in range(warmups):
        with torch_module.cuda.stream(stream):
            copy_once()
    stream.synchronize()

    device_start = torch_module.cuda.Event(enable_timing=True)
    device_end = torch_module.cuda.Event(enable_timing=True)
    torch_module.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    with torch_module.cuda.stream(stream):
        device_start.record(stream)
        for _ in range(operations):
            copy_once()
        device_end.record(stream)
    stream.synchronize()
    wall_end = time.perf_counter_ns()
    device_elapsed_ns = device_start.elapsed_time(device_end) * 1.0e6

    wall_latencies: List[float] = []
    device_latencies: List[float] = []
    latency_start_event = torch_module.cuda.Event(enable_timing=True)
    latency_end_event = torch_module.cuda.Event(enable_timing=True)
    for _ in range(latency_operations):
        sample_start = time.perf_counter_ns()
        with torch_module.cuda.stream(stream):
            latency_start_event.record(stream)
            copy_once()
            latency_end_event.record(stream)
        stream.synchronize()
        sample_end = time.perf_counter_ns()
        wall_latencies.append(float(sample_end - sample_start))
        device_latencies.append(
            latency_start_event.elapsed_time(latency_end_event) * 1.0e6
        )

    total_bytes = operations * block_bytes
    trial = {
        "tier": "host_dram",
        "path": "gpu_pinned_host",
        "direction": direction,
        "block_bytes": block_bytes,
        "lanes": 1,
        "max_outstanding_operations": 1,
        "throughput_operations": operations,
        "throughput": throughput_summary(total_bytes, wall_end - wall_start),
        "cuda_event_throughput": throughput_summary(
            total_bytes, max(1, int(round(device_elapsed_ns)))
        ),
        "latency_sample_operations": latency_operations,
        "latency_ns": latency_summary(wall_latencies),
        "cuda_event_latency_ns": latency_summary(device_latencies),
        "warmup_operations": warmups,
        "measurement_semantics": {
            "throughput": "all copies queued on one CUDA stream, one final stream synchronization; CPU wall clock includes enqueue overhead",
            "latency": "one copy followed by stream synchronization per sample",
        },
    }
    return trial


def run_dram_suite(
    torch_module: Any,
    block_sizes: Sequence[int],
    targets: Dict[int, int],
    quick: bool,
) -> List[Dict[str, Any]]:
    trials = []
    warmups = 2 if quick else 8
    torch_module.cuda.synchronize()
    for block_bytes in block_sizes:
        operations = max(1, int(math.ceil(targets[block_bytes] / float(block_bytes))))
        latency_operations = min(operations, 8 if quick else (64 if block_bytes >= 16777216 else 128))
        source = PinnedBuffer(torch_module, block_bytes, DIRECT_ALIGNMENT)
        destination = PinnedBuffer(torch_module, block_bytes, DIRECT_ALIGNMENT)
        try:
            payload = pattern_payload("dram:{}".format(block_bytes), block_bytes)
            source.view[:] = payload
            destination.view[:] = bytes(block_bytes)
            expected_digest = hashlib.sha256(source.view).hexdigest()
            gpu = torch_module.empty(block_bytes, dtype=torch_module.uint8, device="cuda")
            gpu.copy_(source.tensor, non_blocking=True)
            torch_module.cuda.synchronize()
            expected_gpu = gpu.clone()

            offload = measure_cuda_direction(
                torch_module,
                "offload_d2h",
                block_bytes,
                operations,
                latency_operations,
                warmups,
                source,
                destination,
                gpu,
            )
            destination_digest = hashlib.sha256(destination.view).hexdigest()
            offload["validation"] = {
                "passed": bool(torch_module.equal(destination.tensor, source.tensor)),
                "method": "exact torch.equal plus SHA-256 of the final destination",
                "expected_sha256": expected_digest,
                "actual_sha256": destination_digest,
            }
            if not offload["validation"]["passed"]:
                raise BenchmarkError("pinned DRAM D2H validation failed")
            trials.append(offload)

            gpu.zero_()
            torch_module.cuda.synchronize()
            restore = measure_cuda_direction(
                torch_module,
                "restore_h2d",
                block_bytes,
                operations,
                latency_operations,
                warmups,
                source,
                destination,
                gpu,
            )
            restore_equal = bool(torch_module.equal(gpu, expected_gpu))
            restore["validation"] = {
                "passed": restore_equal,
                "method": "exact device-side torch.equal against an immutable expected GPU tensor",
                "expected_sha256": expected_digest,
            }
            if not restore_equal:
                raise BenchmarkError("pinned DRAM H2D validation failed")
            trials.append(restore)
            del expected_gpu
            del gpu
        finally:
            destination.close()
            source.close()
        torch_module.cuda.empty_cache()
    return trials


def prepare_direct_file(path: str, size: int) -> Dict[str, Any]:
    if size % DIRECT_ALIGNMENT:
        raise BenchmarkError("direct-I/O file size must be alignment-multiple")
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    fd = os.open(path, flags, 0o600)
    method = "posix_fallocate"
    try:
        try:
            os.posix_fallocate(fd, 0, size)
        except AttributeError:
            method = "ftruncate"
            os.ftruncate(fd, size)
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"bytes": size, "allocation_method": method, "preallocated": True}


def open_direct_fds(path: str, lanes: int) -> List[int]:
    flags = os.O_RDWR | os.O_DIRECT
    descriptors = []
    try:
        for _ in range(lanes):
            descriptor = os.open(path, flags)
            descriptors.append(descriptor)
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise
    return descriptors


def parallel_lane_trial(
    lanes: int,
    worker: Callable[[int], Dict[str, List[float]]],
    prepare: Optional[Callable[[int], None]] = None,
) -> Tuple[int, Dict[str, List[float]], List[int]]:
    start_box: Dict[str, int] = {}

    def mark_start() -> None:
        start_box["ns"] = time.perf_counter_ns()

    barrier = threading.Barrier(lanes + 1, action=mark_start)
    results: List[Optional[Dict[str, List[float]]]] = [None] * lanes
    finish_times = [0] * lanes
    errors: List[Tuple[int, str]] = []
    error_lock = threading.Lock()

    def run_lane(lane: int) -> None:
        try:
            if prepare is not None:
                prepare(lane)
            barrier.wait(timeout=30)
            results[lane] = worker(lane)
        except BaseException:
            with error_lock:
                errors.append((lane, traceback.format_exc()))
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass
        finally:
            finish_times[lane] = time.perf_counter_ns()

    threads = [
        threading.Thread(target=run_lane, args=(lane,), name="offload-lane-{}".format(lane))
        for lane in range(lanes)
    ]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=30)
    except threading.BrokenBarrierError:
        for thread in threads:
            thread.join()
        if errors:
            lane, detail = errors[0]
            raise BenchmarkError("lane {} failed before start:\n{}".format(lane, detail))
        raise BenchmarkError("lane start barrier failed")
    for thread in threads:
        thread.join()
    if errors:
        lane, detail = errors[0]
        raise BenchmarkError("lane {} failed:\n{}".format(lane, detail))
    merged: Dict[str, List[float]] = {}
    operation_counts = []
    for result in results:
        if result is None:
            raise BenchmarkError("lane returned no result")
        lane_count = None
        for name, values in result.items():
            merged.setdefault(name, []).extend(values)
            if name == "total":
                lane_count = len(values)
        operation_counts.append(0 if lane_count is None else lane_count)
    wall_time = max(finish_times) - start_box["ns"]
    return wall_time, merged, operation_counts


def validate_direct_file(
    path: str,
    block_bytes: int,
    lanes: int,
    operations_per_lane: int,
    expected_block_digests: Sequence[str],
) -> Dict[str, Any]:
    buffer = MmapBuffer(block_bytes, DIRECT_ALIGNMENT)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    mismatches = []
    file_digest = hashlib.sha256()
    verified = 0
    start = time.perf_counter_ns()
    try:
        for lane in range(lanes):
            for operation in range(operations_per_lane):
                offset = (lane * operations_per_lane + operation) * block_bytes
                pread_all_direct(fd, buffer.view, offset)
                digest = hashlib.sha256(buffer.view).hexdigest()
                file_digest.update(buffer.view)
                verified += block_bytes
                if digest != expected_block_digests[lane] and len(mismatches) < 16:
                    mismatches.append(
                        {
                            "lane": lane,
                            "operation": operation,
                            "offset": offset,
                            "expected_sha256": expected_block_digests[lane],
                            "actual_sha256": digest,
                        }
                    )
    finally:
        os.close(fd)
        buffer.close()
    elapsed = time.perf_counter_ns() - start
    return {
        "passed": not mismatches,
        "method": "untimed O_DIRECT full-file read; SHA-256 checked for every block",
        "bytes_verified": verified,
        "blocks_verified": lanes * operations_per_lane,
        "validation_time_ns": elapsed,
        "actual_file_sha256": file_digest.hexdigest(),
        "mismatches": mismatches,
    }


def direct_trial_record(
    path_name: str,
    direction: str,
    block_bytes: int,
    lanes: int,
    operations_per_lane: int,
    wall_time_ns: int,
    components: Dict[str, List[float]],
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    operations = lanes * operations_per_lane
    trial = {
        "tier": "local_nvme",
        "path": path_name,
        "direction": direction,
        "block_bytes": block_bytes,
        "lanes": lanes,
        "max_outstanding_operations": lanes,
        "operations_per_lane": operations_per_lane,
        "operations": operations,
        "throughput": throughput_summary(operations * block_bytes, wall_time_ns),
        "latency_ns": latency_summary(components["total"]),
        "component_latency_ns": {
            name: latency_summary(values)
            for name, values in components.items()
            if name != "total"
        },
        "validation": validation,
        "measurement_semantics": {
            "concurrency": "one synchronous direct-I/O request per lane; aggregate outstanding depth is at most the lane count",
            "wall_clock": "barrier release through the final lane's final operation; setup, allocation, and validation excluded",
        },
    }
    return trial


def run_pure_nvme(
    work_dir: str,
    block_bytes: int,
    lanes: int,
    operations_per_lane: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = os.path.join(
        work_dir, "pure-{}-lane{}.direct".format(block_bytes, lanes)
    )
    size = block_bytes * lanes * operations_per_lane
    allocation = prepare_direct_file(path, size)
    buffers = [MmapBuffer(block_bytes, DIRECT_ALIGNMENT) for _ in range(lanes)]
    expected_digests = []
    try:
        for lane, buffer in enumerate(buffers):
            buffer.view[:] = pattern_payload(
                "pure:{}:{}".format(block_bytes, lane), block_bytes
            )
            expected_digests.append(hashlib.sha256(buffer.view).hexdigest())

        descriptors = open_direct_fds(path, lanes)
        try:
            def write_worker(lane: int) -> Dict[str, List[float]]:
                totals = []
                writes = []
                for operation in range(operations_per_lane):
                    offset = (lane * operations_per_lane + operation) * block_bytes
                    started = time.perf_counter_ns()
                    pwrite_all_direct(descriptors[lane], buffers[lane].view, offset)
                    finished = time.perf_counter_ns()
                    elapsed = float(finished - started)
                    totals.append(elapsed)
                    writes.append(elapsed)
                return {"total": totals, "ssd_pwritev": writes}

            write_wall, write_components, write_counts = parallel_lane_trial(
                lanes, write_worker
            )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)
        for buffer in buffers:
            buffer.view[:] = bytes(block_bytes)
        descriptors = open_direct_fds(path, lanes)
        try:
            def read_worker(lane: int) -> Dict[str, List[float]]:
                totals = []
                reads = []
                for operation in range(operations_per_lane):
                    offset = (lane * operations_per_lane + operation) * block_bytes
                    started = time.perf_counter_ns()
                    pread_all_direct(descriptors[lane], buffers[lane].view, offset)
                    finished = time.perf_counter_ns()
                    elapsed = float(finished - started)
                    totals.append(elapsed)
                    reads.append(elapsed)
                return {"total": totals, "ssd_preadv": reads}

            read_wall, read_components, read_counts = parallel_lane_trial(
                lanes, read_worker
            )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)
        full_validation = validate_direct_file(
            path,
            block_bytes,
            lanes,
            operations_per_lane,
            expected_digests,
        )
        if not full_validation["passed"]:
            raise BenchmarkError("pure NVMe full-file validation failed")
        final_buffer_match = [
            hashlib.sha256(buffer.view).hexdigest() == expected_digests[lane]
            for lane, buffer in enumerate(buffers)
        ]
        write_validation = dict(full_validation)
        write_validation["operation_counts_by_lane"] = write_counts
        write_validation["timing_order"] = (
            "full-file validation ran after the timed restore, so it did not "
            "pre-read or warm the restore workload"
        )
        write_trial = direct_trial_record(
            "ssd_direct",
            "offload_pwrite",
            block_bytes,
            lanes,
            operations_per_lane,
            write_wall,
            write_components,
            write_validation,
        )

        read_validation = dict(full_validation)
        read_validation["final_timed_buffer_match_by_lane"] = final_buffer_match
        read_validation["operation_counts_by_lane"] = read_counts
        read_validation["passed"] = bool(
            read_validation["passed"] and all(final_buffer_match)
        )
        if not read_validation["passed"]:
            raise BenchmarkError("pure NVMe read validation failed")
        read_trial = direct_trial_record(
            "ssd_direct",
            "restore_pread",
            block_bytes,
            lanes,
            operations_per_lane,
            read_wall,
            read_components,
            read_validation,
        )
        return [write_trial, read_trial], allocation
    finally:
        for buffer in buffers:
            buffer.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def run_e2e_nvme(
    torch_module: Any,
    work_dir: str,
    block_bytes: int,
    lanes: int,
    operations_per_lane: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = os.path.join(
        work_dir, "e2e-{}-lane{}.direct".format(block_bytes, lanes)
    )
    size = block_bytes * lanes * operations_per_lane
    allocation = prepare_direct_file(path, size)
    hosts = [
        PinnedBuffer(torch_module, block_bytes, DIRECT_ALIGNMENT) for _ in range(lanes)
    ]
    gpu_tensors = []
    expected_gpu = []
    streams = []
    expected_digests = []
    try:
        for lane, host in enumerate(hosts):
            host.view[:] = pattern_payload(
                "e2e:{}:{}".format(block_bytes, lane), block_bytes
            )
            expected_digests.append(hashlib.sha256(host.view).hexdigest())
            gpu = torch_module.empty(block_bytes, dtype=torch_module.uint8, device="cuda")
            gpu.copy_(host.tensor, non_blocking=True)
            gpu_tensors.append(gpu)
            expected_gpu.append(gpu.clone())
            streams.append(torch_module.cuda.Stream())
        torch_module.cuda.synchronize()

        descriptors = open_direct_fds(path, lanes)
        try:
            def prepare_offload_lane(lane: int) -> None:
                stream = streams[lane]
                with torch_module.cuda.stream(stream):
                    hosts[lane].tensor.copy_(gpu_tensors[lane], non_blocking=True)
                stream.synchronize()

            def offload_worker(lane: int) -> Dict[str, List[float]]:
                totals = []
                copies = []
                writes = []
                stream = streams[lane]
                for operation in range(operations_per_lane):
                    offset = (lane * operations_per_lane + operation) * block_bytes
                    total_start = time.perf_counter_ns()
                    copy_start = total_start
                    with torch_module.cuda.stream(stream):
                        hosts[lane].tensor.copy_(
                            gpu_tensors[lane], non_blocking=True
                        )
                    stream.synchronize()
                    copy_end = time.perf_counter_ns()
                    pwrite_all_direct(descriptors[lane], hosts[lane].view, offset)
                    write_end = time.perf_counter_ns()
                    totals.append(float(write_end - total_start))
                    copies.append(float(copy_end - copy_start))
                    writes.append(float(write_end - copy_end))
                return {
                    "total": totals,
                    "gpu_d2h_and_sync": copies,
                    "ssd_pwritev": writes,
                }

            offload_wall, offload_components, offload_counts = parallel_lane_trial(
                lanes, offload_worker, prepare=prepare_offload_lane
            )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)
        for gpu in gpu_tensors:
            gpu.zero_()
        torch_module.cuda.synchronize()
        for host in hosts:
            host.view[:] = bytes(block_bytes)
        descriptors = open_direct_fds(path, lanes)
        try:
            def prepare_restore_lane(lane: int) -> None:
                stream = streams[lane]
                with torch_module.cuda.stream(stream):
                    gpu_tensors[lane].copy_(hosts[lane].tensor, non_blocking=True)
                stream.synchronize()

            def restore_worker(lane: int) -> Dict[str, List[float]]:
                totals = []
                reads = []
                copies = []
                stream = streams[lane]
                for operation in range(operations_per_lane):
                    offset = (lane * operations_per_lane + operation) * block_bytes
                    total_start = time.perf_counter_ns()
                    pread_all_direct(descriptors[lane], hosts[lane].view, offset)
                    read_end = time.perf_counter_ns()
                    with torch_module.cuda.stream(stream):
                        gpu_tensors[lane].copy_(
                            hosts[lane].tensor, non_blocking=True
                        )
                    stream.synchronize()
                    copy_end = time.perf_counter_ns()
                    totals.append(float(copy_end - total_start))
                    reads.append(float(read_end - total_start))
                    copies.append(float(copy_end - read_end))
                return {
                    "total": totals,
                    "ssd_preadv": reads,
                    "gpu_h2d_and_sync": copies,
                }

            restore_wall, restore_components, restore_counts = parallel_lane_trial(
                lanes, restore_worker, prepare=prepare_restore_lane
            )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)
        full_validation = validate_direct_file(
            path,
            block_bytes,
            lanes,
            operations_per_lane,
            expected_digests,
        )
        if not full_validation["passed"]:
            raise BenchmarkError("GPU-to-NVMe full-file validation failed")
        offload_validation = dict(full_validation)
        offload_validation["operation_counts_by_lane"] = offload_counts
        offload_validation["timing_order"] = (
            "full-file validation ran after the timed restore, so it did not "
            "pre-read or warm the restore workload"
        )
        offload_trial = direct_trial_record(
            "gpu_pinned_host_ssd_direct",
            "offload_gpu_to_ssd",
            block_bytes,
            lanes,
            operations_per_lane,
            offload_wall,
            offload_components,
            offload_validation,
        )

        gpu_match = [
            bool(torch_module.equal(gpu_tensors[lane], expected_gpu[lane]))
            for lane in range(lanes)
        ]
        host_digest_match = [
            hashlib.sha256(host.view).hexdigest() == expected_digests[lane]
            for lane, host in enumerate(hosts)
        ]
        restore_validation = {
            "passed": bool(
                full_validation["passed"]
                and all(gpu_match)
                and all(host_digest_match)
            ),
            "method": "post-restore full-file direct validation, final direct-read host SHA-256, and exact device-side torch.equal per lane",
            "full_file_validation": full_validation,
            "gpu_exact_match_by_lane": gpu_match,
            "host_sha256_match_by_lane": host_digest_match,
            "operation_counts_by_lane": restore_counts,
        }
        if not restore_validation["passed"]:
            raise BenchmarkError("NVMe-to-GPU restore validation failed")
        restore_trial = direct_trial_record(
            "gpu_pinned_host_ssd_direct",
            "restore_ssd_to_gpu",
            block_bytes,
            lanes,
            operations_per_lane,
            restore_wall,
            restore_components,
            restore_validation,
        )
        return [offload_trial, restore_trial], allocation
    finally:
        for host in hosts:
            host.close()
        gpu_tensors[:] = []
        expected_gpu[:] = []
        streams[:] = []
        torch_module.cuda.empty_cache()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def run_nvme_suite(
    torch_module: Any,
    work_dir: str,
    block_sizes: Sequence[int],
    lanes_values: Sequence[int],
    targets: Dict[int, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trials = []
    allocations = []
    for block_bytes in block_sizes:
        for lanes in lanes_values:
            operations_per_lane, total_operations = target_operations(
                targets[block_bytes], block_bytes, lanes
            )
            pure_trials, pure_allocation = run_pure_nvme(
                work_dir, block_bytes, lanes, operations_per_lane
            )
            pure_allocation.update(
                {
                    "path_kind": "ssd_direct",
                    "block_bytes": block_bytes,
                    "lanes": lanes,
                    "operations": total_operations,
                }
            )
            allocations.append(pure_allocation)
            trials.extend(pure_trials)

            e2e_trials, e2e_allocation = run_e2e_nvme(
                torch_module, work_dir, block_bytes, lanes, operations_per_lane
            )
            e2e_allocation.update(
                {
                    "path_kind": "gpu_pinned_host_ssd_direct",
                    "block_bytes": block_bytes,
                    "lanes": lanes,
                    "operations": total_operations,
                }
            )
            allocations.append(e2e_allocation)
            trials.extend(e2e_trials)
    return trials, allocations


def atomic_write_json(path: str, document: Dict[str, Any], overwrite: bool) -> None:
    rendered = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path == "-":
        sys.stdout.write(rendered)
        sys.stdout.flush()
        return
    absolute = os.path.abspath(path)
    parent = os.path.dirname(absolute)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(absolute) and not overwrite:
        raise BenchmarkError(
            "output already exists (pass --overwrite deliberately): {}".format(absolute)
        )
    temporary = "{}.tmp.{}".format(absolute, os.getpid())
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, absolute)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def self_check() -> Dict[str, Any]:
    payload = pattern_payload("self-check", 4096)
    samples = [1.0, 2.0, 3.0, 4.0]
    buffer_alignment = None
    if os.name == "posix":
        buffer = MmapBuffer(4096, DIRECT_ALIGNMENT)
        try:
            buffer.view[:] = payload
            buffer_alignment = buffer.address % DIRECT_ALIGNMENT
            if hashlib.sha256(buffer.view).digest() != hashlib.sha256(payload).digest():
                raise BenchmarkError("mmap buffer digest mismatch")
        finally:
            buffer.close()
    summary = latency_summary(samples)
    if summary["p50"] != 2.5 or summary["p99"] <= summary["p95"]:
        raise BenchmarkError("percentile implementation failed self-check")
    operations_per_lane, operations = target_operations(1, 4096, 8)
    if operations_per_lane != 1 or operations != 8:
        raise BenchmarkError("lane operation rounding failed self-check")
    result = {
        "self_check": "PASS",
        "python": platform.python_version(),
        "pattern_sha256": hashlib.sha256(payload).hexdigest(),
        "mmap_alignment_remainder": buffer_alignment,
        "latency_summary": summary,
        "lane_rounding": {
            "operations_per_lane": operations_per_lane,
            "operations": operations,
        },
    }
    json.dumps(result, allow_nan=False)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="-",
        help="result JSON path, or - for stdout (default: -)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="job-exclusive direct-I/O directory (default: /tmp/hbfsim-offload-JOBID-or-PID)",
    )
    parser.add_argument(
        "--block-sizes",
        default=",".join(str(item) for item in DEFAULT_BLOCK_SIZES),
        help="comma-separated byte sizes",
    )
    parser.add_argument(
        "--lanes",
        default=",".join(str(item) for item in DEFAULT_LANES),
        help="comma-separated direct-I/O lane counts",
    )
    parser.add_argument("--quick", action="store_true", help="short remote smoke run")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="standard-library CPU checks only; no torch, GPU, or storage access",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-tmp", action="store_true")
    parser.add_argument("--require-local-storage", action="store_true")
    parser.add_argument("--require-nvme", action="store_true")
    parser.add_argument("--require-gpu-substring", default=None)
    parser.add_argument("--expect-python", default=None, metavar="MAJOR.MINOR")
    parser.add_argument("--expect-torch", default=None, metavar="VERSION")
    return parser


def resolve_targets(
    block_sizes: Sequence[int], quick: bool
) -> Tuple[Dict[int, int], Dict[int, int]]:
    source_dram = QUICK_DRAM_TARGETS if quick else NORMAL_DRAM_TARGETS
    source_nvme = QUICK_NVME_TARGETS if quick else NORMAL_NVME_TARGETS
    unsupported = [item for item in block_sizes if item not in source_dram]
    if unsupported:
        raise BenchmarkError(
            "no declared target byte count for block size(s): {}".format(unsupported)
        )
    return (
        {item: source_dram[item] for item in block_sizes},
        {item: source_nvme[item] for item in block_sizes},
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        print(json.dumps(self_check(), indent=2, sort_keys=True, allow_nan=False))
        return 0

    block_sizes = parse_positive_csv(args.block_sizes, "--block-sizes")
    lanes_values = parse_positive_csv(args.lanes, "--lanes")
    if any(size % DIRECT_ALIGNMENT for size in block_sizes):
        parser.error("every block size must be a multiple of {}".format(DIRECT_ALIGNMENT))
    dram_targets, nvme_targets = resolve_targets(block_sizes, args.quick)

    auto_work_dir = args.work_dir is None
    if args.work_dir is None:
        identifier = os.environ.get("SLURM_JOB_ID", "pid{}".format(os.getpid()))
        args.work_dir = "/tmp/hbfsim-offload-{}".format(identifier)
    args.work_dir = os.path.abspath(args.work_dir)
    os.makedirs(args.work_dir, mode=0o700, exist_ok=False if auto_work_dir else True)

    document: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "dana_a100_offload",
        "status": "running",
        "started_at_utc": utc_now(),
        "run": {
            "argv": [sys.executable] + sys.argv,
            "quick": args.quick,
            "block_sizes": list(block_sizes),
            "lanes": list(lanes_values),
            "dram_target_bytes": dram_targets,
            "nvme_target_bytes": nvme_targets,
            "clock": "time.perf_counter_ns (monotonic wall clock)",
        },
        "contract": {
            "dram": "CUDA copies between GPU uint8 tensors and page-aligned CUDA-pinned host tensors",
            "ssd": "preallocated file, O_DIRECT, page-aligned buffers, positional preadv/pwritev, no buffered fallback",
            "ssd_write_completion": "successful synchronous O_DIRECT pwritev return; no explicit power-loss flush",
            "validation_excluded_from_timing": True,
            "decimal_GB_per_s": True,
        },
        "environment": None,
        "allocations": [],
        "trials": [],
        "errors": [],
    }
    exit_code = 0
    try:
        try:
            import torch
        except ImportError as error:
            raise BenchmarkError("cluster-provided torch is required: {}".format(error))
        document["environment"] = base_environment(args.work_dir, torch)
        require_environment(args, torch, document["environment"])
        document["trials"].extend(
            run_dram_suite(torch, block_sizes, dram_targets, args.quick)
        )
        nvme_trials, allocations = run_nvme_suite(
            torch, args.work_dir, block_sizes, lanes_values, nvme_targets
        )
        document["trials"].extend(nvme_trials)
        document["allocations"].extend(allocations)
        for trial_index, trial in enumerate(document["trials"]):
            trial["trial_index"] = trial_index
            trial["trial_id"] = "{}.{}.block{}.lanes{}".format(
                trial["path"],
                trial["direction"],
                trial["block_bytes"],
                trial["lanes"],
            )
        validations = [
            bool(trial.get("validation", {}).get("passed"))
            for trial in document["trials"]
        ]
        expected_trial_count = len(block_sizes) * (2 + 4 * len(lanes_values))
        if len(document["trials"]) != expected_trial_count:
            raise BenchmarkError(
                "expected {} trials, produced {}".format(
                    expected_trial_count, len(document["trials"])
                )
            )
        if not validations or not all(validations):
            raise BenchmarkError("one or more trial validations failed")
        document["status"] = "passed"
        document["summary"] = {
            "passed": True,
            "trial_count": len(document["trials"]),
            "validation_count": len(validations),
            "all_validations_passed": True,
        }
    except BaseException as error:
        exit_code = 1
        document["status"] = "failed"
        document["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        document["summary"] = {
            "passed": False,
            "trial_count": len(document["trials"]),
            "all_validations_passed": False,
        }
    finally:
        document["finished_at_utc"] = utc_now()
        document["duration_ns"] = max(
            0,
            int(
                (
                    datetime.datetime.fromisoformat(document["finished_at_utc"])
                    - datetime.datetime.fromisoformat(document["started_at_utc"])
                ).total_seconds()
                * 1.0e9
            ),
        )
        try:
            atomic_write_json(args.output, document, args.overwrite)
        except BaseException as write_error:
            sys.stderr.write("failed to write result JSON: {}\n".format(write_error))
            exit_code = 1
        if auto_work_dir:
            try:
                os.rmdir(args.work_dir)
            except OSError:
                shutil.rmtree(args.work_dir, ignore_errors=True)
    if exit_code:
        sys.stderr.write("benchmark failed; inspect the JSON errors field\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
