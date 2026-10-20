# PLAN_HC_POST_SYCL_FUSION.md — Port the HC-post gate fusion to SYCL (README_FORK open item 3)

Status: **implemented on branch `glm-rebase` (see commit history), compile gate passed**
(icpx/DPC++, image `localhost/llama.cpp-intel:hc-post-fusion` = `72d5fa130df4`, pushed to
`registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion`; build stage pushed as
`:hc-post-fusion-build`; `test-backend-ops` compiles clean in-container). §7 is the
GPU-host validation session. §6 outcome: compile validation done; the opportunistic
CPU-device runtime check is **not possible** — the vendored dpct dev_mgr
(ggml/src/ggml-sycl/dpct/helper.hpp) throws `"can not find preferred GPU platform"` on a
cpu-only host regardless of `ONEAPI_DEVICE_SELECTOR`, so SYCL numerics tests need the GPU
host (see CONTAINERS.md).

---

## 1. Background and current state

**What the fusion is.** qwen4exp's `build_hc_combine()` (src/models/qwen4exp.cpp:369) builds
the hc_post scatter weights as a `SCALE → SIGMOID → SCALE` chain feeding
`ggml_dsv4_hc_post(..., w, nullptr)`:

```
w = ggml_scale(ctx0, inject, 1/hc);        // GGML_OP_SCALE, params {1/hc, 0}
w = ggml_sigmoid(ctx0, w);                 // GGML_OP_UNARY,  SIGMOID
w = ggml_scale(ctx0, w, 2.0f);             // GGML_OP_SCALE, params {2.0f, 0}
cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);   // w lands in src[2]
```

Upstream fused this 4-node chain into a single kernel:
- **CUDA #30602** — `ggml_cuda_try_fuse()` branch (ggml/src/ggml-cuda/ggml-cuda.cu:4535),
  kernel `dsv4_hc_post_f32<has_comb, gated>` (ggml/src/ggml-cuda/dsv4-hc.cu) reads the raw
  pre-gate weights (`scale_in->src[0]`) and computes `p = out*sigmoid(in*p)` inline.
- **Vulkan #29520** — same pattern (`hc_post_gate_pattern` in ggml-vulkan-types.h:540,
  `ggml_vk_can_fuse_hc_post_gate` at ggml-vulkan.cpp:13582, gate support in
  dsv4_hc_post.comp).

**What SYCL does today** — the unfused 4-kernel sequence: `ggml_sycl_scale` (scale_f32_sycl)
→ SIGMOID unary kernel → `ggml_sycl_scale` → `dsv4_hc_post_f32_sycl`, dispatched per node by
the graph loop in `ggml_backend_sycl_graph_compute_impl()` (ggml-sycl.cpp:6551).

**Payoff.** qwen4exp runs 2 `build_hc_combine` sites per layer + 2 in the MTP block
(qwen4exp.cpp:482/494/597/601) = 92 chains/token on a 45+1-layer model; 4 kernels → 1 saves
3 launches each ⇒ **~276 launches/token**, exactly the figure in README_FORK item 3. The win
is decode-side (launch-latency bound); prefill saves the same launches per ubatch but they
are noise next to GEMM time.

**Scope decision (important).** The pattern matches **only the qwen4exp scalar-SCALE form**.
GLM-5.3-Flash's post chain (glm5-next.cpp:470–485) is `view → MUL(affine) → ADD(affine) →
SIGMOID → SCALE → HC_POST`; its first chain nodes are binbcast MUL/ADD, not `GGML_OP_SCALE`,
so neither upstream precedent fuses it and this port does not change GLM (it is a no-op
there — verified by pattern, not assumption; see §7 step 4). Folding GLM's affine into the
kernel is a feasible, exact follow-up (§9), deliberately kept out of v1 to stay a faithful,
mechanical port of merged upstream code.

## 2. Target design

Mirror the CUDA implementation 1:1, using infrastructure the SYCL backend already has:

1. **Pattern match** in `ggml_sycl_can_fuse()` (ggml/src/ggml-sycl/fusion.cpp) — a dedicated
   branch for `{ GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }` +
   `{ GGML_UNARY_OP_SIGMOID }`, using `ggml_can_fuse_subgraph` + `ggml_check_edges`
   (both shared in ggml-impl.h) + the same tensor/param guards as CUDA.
2. **Fused kernel** — extend `dsv4_hc_post_f32_sycl` with a `gated` template parameter
   (mirroring `dsv4_hc_post_f32<has_comb, gated>` in dsv4-hc.cu).
3. **Op entry** — `ggml_sycl_op_dsv4_hc_post()` grows optional
   `gate_scale_in`/`gate_scale_out` args (default `nullptr`), exactly like
   `ggml_cuda_op_dsv4_hc_post()` in dsv4-hc.cuh.
4. **Dispatch** — one new branch in `ggml_backend_sycl_graph_compute_impl()` that calls the
   op directly with the two SCALE nodes and skips 3 nodes (`i += 3`).
5. **Memory-range guard** — promote the CUDA-derived `ggml_sycl_check_fusion_memory_ranges`
   helper (currently `static` in topk-moe.cpp:484) into fusion.cpp and share it.

The fusion is automatically gated by the existing `GGML_SYCL_ENABLE_FUSION` env
(g_ggml_sycl_enable_fusion, checked at the top of `ggml_sycl_can_fuse`), giving a free
kill-switch and A/B lever.

## 3. Critical implementation details (things that will bite if skipped)

- **Branch ordering in fusion.cpp.** `ggml_sycl_can_fuse()` calls generic
  `ggml_can_fuse()` (ggml-impl.h:711) *before* its per-pattern branches. Generic can_fuse
  requires each node's `src[0]`/`src[1]` to be the previous node and all shapes equal — both
  false here (the gate feeds `hc_post->src[2]`, shape `[hc,nt]` vs `[n_embd,hc,nt]`). The
  hc_post branch **must be placed before the generic `ggml_can_fuse()` call**, like the
  existing mul_mat+mul_mat+GLU subgraph branch. This is the one non-obvious failure mode;
  CUDA places its branch before the generic call for the same reason (ggml-cuda.cu:3552).
- **Edges to check** (same as CUDA:3559 / Vulkan hc_post_gate_edges):
  `{1,0,0}` sigmoid.src[0]==scale_in, `{2,0,1}` scale_out.src[0]==sigmoid,
  `{3,2,2}` hc_post.src[2]==scale_out.
- **Guards** (copy CUDA:3560–3563 verbatim): unary is SIGMOID; both SCALE bias params
  (`ggml_get_op_params_f32(scale, 1)`) are 0 (kernel implements no-bias only);
  `scale_in->src[0]` and `scale_out` are F32; `ggml_are_same_shape(scale_in->src[0],
  scale_out)`. Then `ggml_sycl_check_fusion_memory_ranges(cgraph, i, 4, {i+3}, 1)`.
- **Exact sigmoid** — the fused kernel must compute the gate exactly as the standalone
  kernels do: `1.0f / (1.0f + sycl::exp(-(gate_in * p)))` (matches `op_sigmoid` in
  element_wise.hpp:117 which uses `op_exp` = `sycl::exp` for F32, and matches the existing
  gated `dsv4_hc_pre_f32_sycl`). Use `sycl::exp`, not `expf`/`sycl::native::exp`.
- **Strides follow the raw tensor.** When gated, `post` data *and* strides (`nbp0/nbp1`)
  come from `gate_scale_in->src[0]` (same as CUDA dsv4-hc.cu:323 and Vulkan `p_src`).
  `ggml_are_same_shape(scale_in->src[0], scale_out)` guarantees the shape qwen4exp builds
  (`[hc, nt]` from `build_lora_mm(w_inject, xn)`).
- **Skipped-node bookkeeping.** `i += 3` in the graph loop (SYCL uses `i += skip`, CUDA's
  `try_fuse` returns 3). The three elided nodes are all `ggml_dup_tensor` outputs (SCALE and
  UNARY are never in-place here), single-use by construction (`ggml_can_fuse_subgraph`
  verifies uses), and the memory-range check covers any aliasing pathology.
- **Multi-GPU / split safety.** The chain is entirely f32 intermediates of one layer, so it
  always sits inside one backend's subgraph; if a sched split ever landed mid-chain the
  pattern simply wouldn't match. No cross-device hazard. SYCL graph-recording path
  (`GGML_SYCL_GRAPH`) uses the same `_impl` loop, so it inherits the fusion.

## 4. File-by-file changes

### 4.1 `ggml/src/ggml-sycl/dsv4-hc.cpp` — kernel + op entry

Kernel: add `gated` template param and two scalars (body identical to CUDA
dsv4-hc.cu:157–201):

```cpp
template <bool has_comb, bool gated>
static void dsv4_hc_post_f32_sycl(
        const float * x, const float * residual, const float * post, const float * comb, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        float gate_scale_in, float gate_scale_out,      // new
        queue_ptr stream) {
    ...
            float p = post[idst*sp0 + it*sp1];
            if constexpr (gated) {
                p = gate_scale_out * (1.0f / (1.0f + sycl::exp(-(gate_scale_in * p))));
            }
            float sum = x[i0*sx0 + it*sx1] * p;
    ...
}
```

Op entry: optional gate args, `post`/strides re-pointed when gated (mirror CUDA
dsv4-hc.cu:296–305), launch one of the four `<has_comb, gated>` instantiations:

```cpp
void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    ...
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];
    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;
    ...
}
```

### 4.2 `ggml/src/ggml-sycl/dsv4-hc.hpp` — declaration

```cpp
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);
```

The `case GGML_OP_DSV4_HC_POST:` dispatcher entry (ggml-sycl.cpp:6003) keeps compiling
unchanged (default args) and keeps serving the unfused path.

### 4.3 `ggml/src/ggml-sycl/fusion.cpp` — pattern branch

Insert **before** the generic `ggml_can_fuse(cgraph, node_idx, ops)` call (see §3):

```cpp
// scale_out*sigmoid(scale_in*post) as the hc_post weights (qwen4exp hc_combine);
// port of the CUDA/Vulkan hc_post_gate fusion. The gate feeds hc_post->src[2], so the
// generic ggml_can_fuse() chain check cannot express it — subgraph + explicit edges.
if (ops.size() == 4 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY &&
    ops.begin()[2] == GGML_OP_SCALE && ops.begin()[3] == GGML_OP_DSV4_HC_POST) {
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 3 })) {
        return false;
    }
    const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
    const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

    if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
        ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
        ggml_get_op_params_f32(scale_in,  1) == 0.0f &&
        ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
        scale_in->src[0]->type == GGML_TYPE_F32 && scale_out->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(scale_in->src[0], scale_out)) {
        int out_nodes[] = { node_idx + 3 };
        return ggml_sycl_check_fusion_memory_ranges(cgraph, node_idx, (int) ops.size(), out_nodes, 1);
    }
    return false;
}
```

### 4.4 `ggml/src/ggml-sycl/fusion.hpp` / `topk-moe.cpp` — helper promotion

Move `ggml_sycl_check_fusion_memory_ranges` from topk-moe.cpp (static, line 484) to
fusion.cpp; declare in fusion.hpp with `bool is_topk_moe = false` default; delete the
topk-moe copy and its comment ("ported from ggml_cuda_check_fusion_memory_ranges"). Keep the
`is_topk_moe` parameter so topk-moe call sites are untouched apart from the include.

### 4.5 `ggml/src/ggml-sycl/ggml-sycl.cpp` — dispatch branch

In `ggml_backend_sycl_graph_compute_impl()` (line ~6551), alongside the other
`ggml_sycl_can_fuse` branches (suggest: right after the UNARY+MUL / ADD+UNARY+MUL group,
before the L2_NORM batch branch):

```cpp
if (node->op == GGML_OP_SCALE &&
    ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST },
                       { GGML_UNARY_OP_SIGMOID })) {
    ggml_sycl_op_dsv4_hc_post(*sycl_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
    i += 3;
    continue;
}
```

No changes needed to: the op dispatcher switch, ggml.c graph builder, model code, CMake
(dsv4-hc.cpp already compiled; fusion.cpp already compiled).

Total diff: ~120 lines, four files (+1 file touched for the helper move).

## 5. Numerics-exactness argument

Per element the unfused sequence computes, in f32:

1. `t0 = s_in * p + 0.0f`   (scale_f32 kernel, bias 0)
2. `t1 = 1/(1 + sycl::exp(-t0))`  (op_sigmoid)
3. `t2 = s_out * t1 + 0.0f` (scale_f32 kernel, bias 0)
4. `sum = x*t2 + Σ residual*comb` (dsv4_hc_post_f32_sycl)

Fused computes `s_out * (1/(1 + sycl::exp(-(s_in * p))))` then the identical step-4 body.
Same ops, same order, same libdevice `sycl::exp` ⇒ bit-identical for every input except a
theoretical `-0.0` intermediate (`x + 0.0f` canonicalizes `-0.0` to `+0.0`; sigmoid of ±0 is
0.5 either way, and sigmoid output is never negative, so downstream values are identical).
This is the same exactness level the CUDA/Vulkan precedents accept, and the pre-gate kernel
in this same file already uses the identical expression. The DSV4_HC_POST gated case is
covered in tests/test-backend-ops.cpp (`test_dsv4_hc_post` with `gated=true` builds the
exact SCALE→SIGMOID→SCALE chain and runs the whole graph against the CPU reference via
`run_whole_graph()`), so backend-ops is the numerical referee — no new test math needed.

## 6. Validation without the GPU host (do now)

1. **Compile gate (podman, no GPU needed).** The intel.Dockerfile `build` stage compiles the
   SYCL backend with icpx; tests are off in the image config, so run a one-off:

   ```bash
   podman build --target build -t localhost/lcpp-intel-build:hcplan -f .devops/intel.Dockerfile .
   podman run --rm localhost/lcpp-intel-build:hcplan bash -c '
       cmake -B /app/build -DGGML_NATIVE=OFF -DGGML_SYCL=ON -DGGML_SYCL_DNN=ON \
             -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
             -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON \
       && cmake --build /app/build --target test-backend-ops -j"$(nproc)"'
   ```

   Catches DPC++ template/`[[sycl::reqd]]` issues; also build `llama-server` to cover the
   default config. (If a SYCL CPU/OpenCL device is visible inside the container via
   `sycl-ls`, `test-backend-ops -b <cpu-device>` can even run numerics here — opportunistic,
   not assumed.)
2. **Diff review against precedent** — line-by-line compare of the new branch vs
   ggml-cuda.cu:3552–3567 and the kernel body vs dsv4-hc.cu:157–201.
3. **Static check of pattern reachability** — grep that nothing else builds a
   SCALE→SIGMOID→SCALE→DSV4_HC_POST chain with different intent (only qwen4exp does today;
   deepseek4/glm5-next/hy-v4 chains don't match, dflash inherits deepseek4's form).

## 7. Validation on the GPU host (when free) — follow the README §8 ops policy

Unique container/port, evidence captured before cleanup, reboot after any device-lost/OOM,
all runs logged.

1. **Backend-ops numerics (minutes):**
   ```bash
   ./test-backend-ops -b SYCL0 -o 'DSV4_HC_POST'          # gated cases exercise the fusion
   ./test-backend-ops -b SYCL0 -o 'DSV4_HC.*'             # pre/comb regression
   GGML_SYCL_ENABLE_FUSION=0 ./test-backend-ops -b SYCL0 -o 'DSV4_HC_POST'   # unfused reference
   ```
   All three must pass (gated whole-graph vs CPU reference; the middle run also guards the
   helper-move refactor of topk-moe — include `-o 'TOPK_MOE.*'` once).
2. **Fusion-fires evidence.** Temporary `GGML_LOG_DEBUG` line in the new dispatch branch
   ("fused SCALE-SIGMOID-SCALE-HC_POST at node %d") during bring-up; capture log with the
   qwen recipe, then remove before commit (upstream logs nothing for this fusion). Counting
   2/layer + 2 in the MTP graph is the expected signature.
3. **Bitwise A/B on a real model (qwen4exp Q8_0 recipe from README_FORK):** fixed prompts,
   greedy sampling, `GGML_SYCL_ENABLE_FUSION=1` vs `=0` — identical output tokens expected
   (exact numerics). Any diff = implementation bug, stop.
4. **GLM-5.3-Flash no-op check:** run the GLM gate clients (needle retrieval + arithmetic
   control, per test package) on the fused build; results must match the promoted build
   (fusion cannot fire for GLM — verify with the §7.2 debug line absent for GLM graphs).
5. **Perf A/B, qwen4exp Q8_0 (`--n-gpu-layers 30`) and UD-Q6_K_XL (`34`):** server-side slot
   timing only (README §1 warning about SSE-buffered client numbers): decode tps at
   128/2048-token contexts, prefill tps at 2k/8k/32k rungs, plus MTP accept-rate (should be
   untouched). Expectation: decode +2–8 % from ~276 fewer launches/token; prefill ~neutral.
   Record in a RESULTS table next to the tree per repo convention.
6. **Promotion:** commit on `glm-rebase` as one commit ("sycl: fuse SCALE-SIGMOID-SCALE into
   hc_post gate (port of #30602/#29520)"), update README_FORK fixes table + strike open
   item 3, re-run both gate clients before promoting the image tag.

Rollback: `GGML_SYCL_ENABLE_FUSION=0` restores today's dispatch without rebuild; the commit
itself reverts cleanly (self-contained).

## 8. Risks

| Risk | Mitigation |
|---|---|
| Branch placed after generic `ggml_can_fuse()` → fusion silently never fires | §3 ordering note + §7.2 debug-line evidence |
| `sycl::exp` vs `expf`/native mismatch → non-bitwise gate | §3/§5: copy the pre-kernel's expression; backend-ops gated case catches it |
| topk-moe helper move breaks topk fusion | same TU set, signature preserved; `TOPK_MOE.*` backend-ops run |
| Pattern over-matches a future graph shape | guards copied from CUDA (no-bias, F32, same-shape, memory ranges); subgraph single-use |
| SYCL-graph recording path divergence | fusion lives in the shared `_impl` loop; covered by default server run |

## 9. Optional Phase 3 (follow-up, not in v1): GLM affine fold

GLM-5.3-Flash's chain `MUL(mixes_view, scale_post) → ADD(+base_post) → SIGMOID → SCALE(2) →
HC_POST` (5 kernels, ×92 sites ≈ 368 launches/token if fully folded, ~200 for the GPU-
resident half at ngl 24) could fuse as `{MUL, ADD, UNARY, SCALE, DSV4_HC_POST}` with the
kernel reading `mixes`/`scale_post`/`base_post` directly. Exact numerics (all f32
elementwise, same op order), but it is *new* kernel surface beyond both precedents, and the
affine operands are strided views of per-layer weights. Only pursue after v1 proves out on
qwen and GLM decode profiling shows launch overhead still dominates. Estimate +1 day incl.
a new `test_dsv4_hc_post` variant.

## 10. Effort and ordering

- Phase A (now, no GPU): implementation + §6 compile gate — ~0.5 day.
- Phase B (first GPU-host window): §7.1–7.4 correctness session — ~0.5 hour host time.
- Phase C (same or next window): §7.5 perf A/B + gates + promotion — ~1–2 hours host time.
- Phase D (optional): §9 GLM fold.

Sequencing note: land this before the next upstream rebase (open item 1) — the port touches
`try_fuse`-adjacent code that upstream actively reworks, and a small self-contained commit
rebases more easily than a pending local series.
