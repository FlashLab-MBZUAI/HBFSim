# Public H100 + Llama 2 70B timing anchor

This directory defines the first public timing-evidence anchor for the serving
path. It uses already published data; it does not require a local GPU, Nsight,
NVBit, or a new serving measurement campaign.

The anchor has two deliberately separate roles:

| Source | Pinned revision | Role | Included evidence |
|---|---|---|---|
| [Microsoft Vidur](https://github.com/microsoft/vidur/tree/8383d2935bc62723a212090baa9f98ada206fc14) | `8383d2935bc62723a212090baa9f98ada206fc14` | calibration fit input | H100 Llama 2 70B attention and MLP profiles plus H100-DGX all-reduce and send/recv profiles |
| [MLPerf Inference v5.0](https://github.com/mlcommons/inference_results_v5.0/tree/0bc17ab3f4b33f730d7869e04cc29cb70835e513) | `0bc17ab3f4b33f730d7869e04cc29cb70835e513` | held-out external reference only | NVIDIA result `5.0-0057`, DGX H100 system/precision/config and preprocessing source, accuracy records, and Server/Offline summary/detail logs |

MLPerf is prohibited from fitting. The held-out submission uses eight
H100-SXM-80GB GPUs, TensorRT 10.8/CUDA 12.8, and FP8 weights. Vidur's profiles
use FP16 and different measurement/runtime machinery. The mismatch is a
recorded validation limitation, not an adjustment factor.

Vidur's pinned `LICENSE` is included and verified as MIT. The pinned MLPerf
results revision has no top-level license file, so this project makes no claim
that those records may be redistributed under a particular license. The
importer reads them from the user's external checkout and writes only to a
local evidence output; the repository does not vendor the MLPerf records.

## Exact evidence boundary

The Vidur portion contains:

| Profile | Rows | Distinct feature keys | Repeated-key rows |
|---|---:|---:|---:|
| attention | 58,632 | 57,272 | 1,360 |
| MLP/linear operations | 1,044 | 1,036 | 8 |
| all-reduce | 2,982 | 2,979 | 3 |
| send/recv | 1,988 | 1,986 | 2 |

Repeated feature keys are present in the published files and can carry
different observations. The importer preserves and counts them; it never
silently deduplicates them. The profiled model geometry also uses Vidur's
32,768-entry padded vocabulary, distinct from the 32,000 tokenizer vocabulary.

The timing-method source files are part of the bundle. They establish that the
attention and collective tables use Kineto CUDA-activity timing, while the MLP
table uses record-function-correlated CUDA-kernel sums. The importer does not
rewrite those methods into a fictitious common `CUDA_EVENT` label and does not
manufacture missing KV-cache-save, CPU-overhead, or mixed-batch observations.

MLPerf contributes the following held-out scores:

| Scenario | Result | Tokens/s | Samples/s |
|---|---|---:|---:|
| Server | VALID | 31,106.28 | 106.21 |
| Offline | VALID | 31,306.8 | 107.256 |

The system, measurement precision, user configuration, and accuracy summaries
are digest-bound alongside the scores. A score copied into a spreadsheet is
not accepted as evidence.

## Build the immutable bundle

Prepare sparse, detached checkouts at the exact commits. The commands below
download published source data only; they do not execute either benchmark.

```bash
git init /path/to/vidur
git -C /path/to/vidur remote add origin https://github.com/microsoft/vidur.git
git -C /path/to/vidur sparse-checkout init --cone
git -C /path/to/vidur sparse-checkout set \
  data/profiling/compute/h100/meta-llama/Llama-2-70b-hf \
  data/profiling/network/h100_dgx \
  vidur/profiling/attention \
  vidur/profiling/mlp \
  vidur/profiling/collectives \
  vidur/profiling/common \
  vidur/profiling/utils
git -C /path/to/vidur fetch --depth=1 --filter=blob:none origin \
  8383d2935bc62723a212090baa9f98ada206fc14
git -C /path/to/vidur checkout --detach FETCH_HEAD

git init /path/to/mlperf-v5.0
git -C /path/to/mlperf-v5.0 remote add origin \
  https://github.com/mlcommons/inference_results_v5.0.git
git -C /path/to/mlperf-v5.0 sparse-checkout init --cone
git -C /path/to/mlperf-v5.0 sparse-checkout set \
  closed/NVIDIA/systems \
  closed/NVIDIA/configs/llama2-70b \
  closed/NVIDIA/code/llama2-70b/tensorrt \
  closed/NVIDIA/measurements/DGX-H100_H100-SXM-80GBx8_TRT/llama2-70b-99 \
  closed/NVIDIA/results/DGX-H100_H100-SXM-80GBx8_TRT/llama2-70b-99
git -C /path/to/mlperf-v5.0 fetch --depth=1 --filter=blob:none origin \
  0bc17ab3f4b33f730d7869e04cc29cb70835e513
git -C /path/to/mlperf-v5.0 checkout --detach FETCH_HEAD
```

Import into a new directory. The importer refuses an existing output, a wrong
origin/commit/tree, changed tracked source, a missing sparse-checkout file,
schema drift, malformed or non-finite timing, profile-domain drift, and any
digest mismatch.

```bash
python3 -B tools/import_h100_llama2_70b_anchor.py \
  --vidur-root /path/to/vidur \
  --mlperf-root /path/to/mlperf-v5.0 \
  --output-dir out/calibration/h100-llama2-70b-anchor
```

Reopen every copied artifact with the independent verifier:

```bash
python3 -B tools/verify_h100_llama2_70b_anchor.py \
  --bundle-dir out/calibration/h100-llama2-70b-anchor \
  --output out/calibration/h100-llama2-70b-anchor.verification.json
```

The verifier does not import the importer. It requires a byte-identical copy
of the tracked anchor manifest, reconstructs all four CSV contracts and all
16 MLPerf artifacts, rejects unattached files and symlinks, and fails if the
bundle attempts to promote its eligibility.

## Build the operator-calibration candidate

The calibration protocol is fixed in
`h100-llama2-70b-calibration-protocol.json`. It declares the split,
interpolator, thresholds, source roles, MLPerf applicability requirements, and
claim states before fitting. The fitter is standard-library-only. Pandas is
needed only once to decode the already published, digest-matched OpenOrca
pickle and extract request lengths; prompts, token IDs, and text are not
copied.

Download the official preprocessed dataset from MLCommons, then extract its
shape evidence. The extractor checks the exact byte count, MD5, and SHA-256
before it permits pandas to unpickle the file.

```bash
curl --fail --location \
  --output /path/to/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz \
  https://inference.mlcommons-storage.org/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz

python3 -B tools/extract_mlperf_openorca_lengths.py \
  --dataset /path/to/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz \
  --output-dir out/calibration/h100-llama2-70b-workload
```

Fit the Vidur-only candidate, then run the separately implemented verifier.
The verifier does not import the fitter: it reconstructs every split,
linear/bilinear prediction, metric, table row, MLPerf configuration fact, and
claim gate from the pinned bytes.

```bash
python3 -B tools/fit_h100_llama2_70b_calibration.py \
  --anchor-bundle-dir out/calibration/h100-llama2-70b-anchor \
  --anchor-verification \
    out/calibration/h100-llama2-70b-anchor.verification.json \
  --workload-evidence-dir out/calibration/h100-llama2-70b-workload \
  --output-dir out/calibration/h100-llama2-70b-candidate

python3 -B tools/verify_h100_llama2_70b_calibration.py \
  --anchor-bundle-dir out/calibration/h100-llama2-70b-anchor \
  --anchor-verification \
    out/calibration/h100-llama2-70b-anchor.verification.json \
  --candidate-dir out/calibration/h100-llama2-70b-candidate \
  --output out/calibration/h100-llama2-70b-candidate.verification.json
```

On the pinned public data, the protocol produces 49 model instances and
20,952 held-out profile keys. All 49 pass the declared per-model gates
(`WAPE <= 10%`, `P90 APE <= 20%`, and absolute signed bias `<= 5%`). The
aggregate WAPE is 2.935816164316%; the worst per-model WAPE is
4.597356723774%, and the worst per-model P90 APE is 18.888888888889%.
These are interpolation results inside the published Vidur domain, not HBFSim
end-to-end serving accuracy.

## Claim gate

A passing anchor import and verification receipt produce exactly this state:

```text
anchor_ready_not_calibrated
```

They establish that the public inputs are real, pinned, intact, and correctly
separated into fit and held-out roles. After a passing fit and independent
candidate verification, the narrower operator-level state is:

```text
operator_calibrated_external_validation_not_established
```

This closes a Vidur-domain interpolation candidate, not project-wide L4 and
not L5. The MLPerf applicability gate reads the held-out evidence only after
the operator evaluation is frozen. It emits no performance prediction because
the submission is TensorRT FP8 with an FP8 KV cache and configured GPU batch
size 1024, while the public Vidur fit is FP16, has decode batch size at most
128, and contains no TensorRT-kernel bridge, CPU/runtime overhead profile, or
scheduler/arrival trace. MLPerf also publishes aggregate generated-token
counts for this result, not the per-request generated-length sequence needed
for replay. Consequently `predictions` and `errors` remain JSON `null`, L5
remains `not_run_out_of_domain`, Frontier timing stays disabled, and the
candidate is not paper-result eligible.

The current Llama 3.1 70B W8A16 structural path cannot inherit this Llama 2
candidate by model-name or parameter-count similarity. A future end-to-end
claim needs a runtime/precision-matched profile or a separately validated
bridge, in-domain batching, scheduler and host-overhead evidence, and an
unfitted serving replay against the held-out targets.

Neither source contains GPU virtual/physical addresses or measured HBM
transactions. This anchor therefore cannot validate HBFSim's address mapping,
HBM traffic volume, or a Nsight/NVBit trace claim.
