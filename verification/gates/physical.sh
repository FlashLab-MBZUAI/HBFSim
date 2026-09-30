#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
probe="${1:-"${repo_root}/build/physical_probe"}"
out_dir="${2:-"${repo_root}/out/physical-guards"}"
mkdir -p "${out_dir}"
# Every contract throws on failure; string matching printed prose is redundant.
# The probe emits aggregate HBM channel and OCP HBF bank/payload/Host contracts.
"${probe}" all > "${out_dir}/physical-guards.log" 2>&1
"${probe}" hbf-standard --trace "${out_dir}/hbf-standard.trace.json" \
  --folded "${out_dir}/hbf-standard.folded" > "${out_dir}/hbf-standard.txt" 2>&1
printf 'Physical bank, interface, Host and HBM channel contracts: PASS\n'
