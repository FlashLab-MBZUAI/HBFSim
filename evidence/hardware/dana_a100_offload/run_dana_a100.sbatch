#!/usr/bin/env bash
#SBATCH --job-name=hbfsim-a100-offload
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-51
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "This script must run inside a Slurm allocation." >&2
    exit 2
fi

readonly WORK_DIR="/tmp/hbfsim-offload-${SLURM_JOB_ID}"
case "${WORK_DIR}" in
    /tmp/hbfsim-offload-[0-9]*) ;;
    *)
        echo "Refusing unsafe work directory: ${WORK_DIR}" >&2
        exit 2
        ;;
esac

cleanup() {
    if [[ -d "${WORK_DIR}" ]]; then
        rm -rf --one-file-system -- "${WORK_DIR}"
    fi
}
trap cleanup EXIT INT TERM HUP

readonly BENCHMARK="${SLURM_SUBMIT_DIR}/evidence/hardware/dana_a100_offload/benchmark.py"
if [[ ! -f "${BENCHMARK}" ]]; then
    echo "Submit from the HBFSim repository root; missing ${BENCHMARK}" >&2
    exit 2
fi

quick_args=()
if [[ "${1:-}" == "--quick" ]]; then
    quick_args=(--quick)
    shift
fi
readonly OUTPUT_PATH="${1:-${SLURM_SUBMIT_DIR}/evidence/hardware/dana_a100_offload/results/dana-a100-offload-${SLURM_JOB_ID}.json}"
if [[ "$#" -gt 1 ]]; then
    echo "Usage: sbatch run_dana_a100.sbatch [--quick] [OUTPUT.json]" >&2
    exit 2
fi

mkdir -p -- "$(dirname -- "${OUTPUT_PATH}")"
mkdir -m 700 -- "${WORK_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname --fqdn 2>/dev/null || hostname)"
echo "work_dir=${WORK_DIR}"
echo "output=${OUTPUT_PATH}"
python3 --version
python3 -c 'import torch; print("torch=" + torch.__version__); print("cuda_build=" + str(torch.version.cuda)); print("cuda_available=" + str(torch.cuda.is_available()))'
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv,noheader

# Do not set or rewrite CUDA_VISIBLE_DEVICES: Slurm owns GPU visibility.
srun --ntasks=1 --cpus-per-task=8 \
    python3 "${BENCHMARK}" \
    --work-dir "${WORK_DIR}" \
    --output "${OUTPUT_PATH}" \
    --require-tmp \
    --require-local-storage \
    --require-nvme \
    --require-gpu-substring A100 \
    --expect-python 3.8 \
    --expect-torch 2.4.1+cu124 \
    "${quick_args[@]}"

echo "result=${OUTPUT_PATH}"
