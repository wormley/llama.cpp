# README_FORK.md — Intel Arc Pro B70 fork (Qwen3.8-Flash-Next + GLM-5.3-Flash)

## Description

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) maintained for
serving **Qwen3.8-Flash-Next** (qwen4exp) and **GLM-5.3-Flash** (glm5-next) on a
4× Intel Arc Pro B70 host (Xe2 Battlemage, 32 GB each, 128 GB VRAM / 128 GB RAM),
SYCL backend, podman. Deliverable image: `localhost/llama.cpp-intel:latest`
(manifest `sha256:a48a79214b716985326d6a02a78fadc26409532299739bdf9f7fcdb574667b07`).

- Base: upstream master `4c3c5810f` (2026-10-19) + the 6 patches below (branch
  `glm-rebase`, tip `9cd10bd17`). Validated on both models before promotion —
  prefill/decode A/B tables and gates in the companion results docs (`RESULTS.md`
  of the qwen and GLM test packages).
- Build flags that matter vs stock: `GGML_SYCL_DNN=ON` (oneDNN/XMX flash-attention
  path) on top of stock `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_BACKEND_DL=ON
  -DGGML_CPU_ALL_VARIANTS=ON`.
- Working set: hybrid-attention MoE models with ~50 % of blocks on CPU; the
  whole-layer split (`--n-gpu-layers`) beats `--n-cpu-moe` expert caching on this
  class of host (see "Open items").

## How to build

```bash
podman build --target server \
  -t localhost/llama.cpp-intel:latest \
  -f .devops/intel.Dockerfile .
podman push localhost/llama.cpp-intel:latest   # if a shared registry mirror is used
```

- Base image: Intel oneAPI components; build ~10–20 min with warm layer cache,
  ~40–60 min cold; needs ~40 GB free disk. Requires podman or docker with
  `/dev/dri` available at run time.
- The Dockerfile is preserved in-image at `/app/intel.Dockerfile` for rebuilds.
- Test image tags used during validation: `:glm-rebase` (the promoted content),
  `:glm-next` (adds the parked #30606 experiment — do not promote, see open items).

## How to run — current best settings

Env for both models: `GGML_SYCL_FA_ONEDNN=1 SYCL_CACHE_PERSISTENT=0
ZES_ENABLE_SYSMAN=1 CCL_P2P_ENABLE=1 CCL_D2D=p2p`. Run with the SYCL server
image, `--device /dev/dri`, host networking, and the model directory mounted
read-only.

### Qwen3.8-Flash-Next (validated best)

```
--model /models/Qwen3.8-Flash-Next-GGUF/Q8_0/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
--ctx-size 262144 --cache-type-k q8_0 --cache-type-v q8_0   # Q8 KV must stay full sized
--n-gpu-layers 30            # 176 GiB Q8 weights; ngl 46 OOMs at load
--spec-type draft-mtp --spec-draft-n-max 3 \
--model-draft /models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
--flash-attn on --fit off --batch-size 2048 --ubatch-size 512 \
--cache-ram 32768 --jinja --ctx-checkpoints 8 -np 1 --cache-reuse 512 \
--override-tensor "per_layer_token_embd=CPU,ple_ngram_embd=CPU" \
--reasoning-preserve --reasoning on
```

UD-Q6_K_XL variant: same but `--n-gpu-layers 34` (q8_0 KV unchanged) — best overall
prefill numbers measured on this rig.

### GLM-5.3-Flash (validated best)

```
--model /models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
--ctx-size 262144            # f16 KV (default; do NOT quantize — DSA cache untested with q8)
--n-gpu-layers 24            # 186 GiB weights; ngl 25/26 OOM at first compute (validated 2/2)
--spec-type draft-mtp --spec-draft-n-max 3    # embedded NextN (blk.45.nextn), no draft GGUF
--flash-attn on --fit off --batch-size 2048 --ubatch-size 512 \
--cache-ram 32768 --jinja --ctx-checkpoints 8 -np 1 --cache-reuse 512 \
--reasoning on
```

**GLM-only prerequisite — arch string:** Unsloth GGUFs of GLM-5.3-Flash built before
upstream PR #27773 naming use `glm5next` for `general.architecture` and all KV keys;
this tree expects `glm5-next`. Fix (no code change, numerics untouched): rewrite the
shard-1 metadata with the `gguf-rename-arch.py` script from the test package
(`gguf-rename-arch.py <in.gguf> <out.gguf> glm5next glm5-next`; shard 1 is
metadata-only, zero tensor bytes) and bind-mount the copy read-only over the
original file inside the container.

Do NOT use: `--n-cpu-moe`/`--moe-cache-mib` on GLM (−83 % decode measured, see
"Open items"). Keep `SYCL_CACHE_PERSISTENT=0` (SIGSEGV on Xe2 with persistent JIT
cache).

## Fixes applied (this tree, on top of upstream `4c3c5810f`)

| commit | what | link/notes |
|---|---|---|
| `b57261c74` | devops: intel image build with `GGML_SYCL_DNN=ON`, Dockerfile preserved in image | enables oneDNN/XMX FA path (B70: ~2× prefill) |
| `e4b5dfb2c` | qwen4exp: build MTP QSA pool inputs only when the MTP layer uses QSA | fixes upstream [#29811](https://github.com/ggml-org/llama.cpp/issues/29811) (server init abort with draft-mtp) |
| `39725fcd8` | sycl: lightning-indexer skip pool-mask-dropped rows | exact-preserving (−inf + finite = −inf); +9 % short prefill |
| `d2fc93cf2` → `13caeaf6b` | sycl: gate grouped MoE dequant-GEMM on routed-row count (`GGML_SYCL_GROUPED_MAX_ROWS`, default 512; reorder install inside the gate) | recovers the prefill regression from upstream's grouped-XMX path (#29245) while keeping its decode win: decode/MTP-verify batches pass the gate, 5120-row prefill ubatches use the library GEMM |

Inherited from upstream (already merged, required): GLM-5.3-Flash support
[#27773](https://github.com/ggml-org/llama.cpp/pull/27773), sparse-attention gather
removal [#30042](https://github.com/ggml-org/llama.cpp/pull/30042).

Rejected after A/B (kept only on branch `glm-next`): [#30606](https://github.com/ggml-org/llama.cpp/pull/30606)
(BMG D=256 FA vector-kernel decode routing) — −11 % GLM decode @2k, neutral @32k;
helps dense GQA D=256 models, not absorbed-MLA sparse attention.

## Open items / recommended next steps

1. **Rebase cadence:** upstream moves fast; rebase `glm-rebase` onto current master
   and re-run both gate clients (needle retrieval + arithmetic control at every
   context rung, per the test packages) before promoting. Judge decode changes by
   server-side slot timing, not client-side streaming numbers (SSE buffering
   distorts the latter).
2. **SYCL QSA/lightning-indexer prefill kernel** (upstream #29825/#29824 line):
   would close the remaining prefill regression (−34 % at short context vs the
   pre-regression build). Rebase to pick it up when merged.
3. **Port the HC-post fusion to SYCL** (#30602 CUDA / #29520 Vulkan precedent):
   `ggml/src/ggml-sycl/dsv4-hc.cpp` still runs SCALE→SIGMOID→SCALE→hc_post as 4
   kernels; ~276 launches/token saved. Mechanical, exact numerics.
4. **#30196** (sampling: read context outputs with one sync) — server decode
   overhead; test after merge.
5. **#30268** + the DFlash2 sidecar draft (HF `Anbeeld/GLM-5.3-Flash-DFlash2-GGUF`):
   enables `draft-dflash` on GLM. Likely a wash on a VRAM-split rig, more
   interesting when the model fits mostly in VRAM.
6. **#30606 for qwen**: parked on branch `glm-next`; A/B only if qwen's FA decode
   shapes match the routed case.
7. **Do not pursue** `--n-cpu-moe` on either model on a 50 %-CPU-resident host
   (GLM measured −83 % decode: the expert cache never warms without routing
   locality; qwen Q8_0 experts are 2× the bytes — worse). Revisit only where VRAM
   covers most of the expert set or the model has strong routing locality.
8. **Operations:** GPU-test hosts need a written run policy (unique container/port
   per test, evidence capture before cleanup, mandatory reboot after any
   device-lost/OOM event, all runs logged). Keep gate clients and result tables
   under version control next to the tree.
