# CONTAINERS.md — image inventory for this tree

Per `/root/container.txt` policy: podman, images pushed to `registry.wormley.com/ai/`
(environment is ephemeral), Dockerfile preserved in-image, names recorded here.

## Images built from this tree (`.devops/intel.Dockerfile`)

| tag (local) | tag (registry) | purpose | status |
|---|---|---|---|
| `localhost/llama.cpp-intel:latest` | — | promoted deliverable (branch `glm-rebase`, see README_FORK.md) | existing, manifest `sha256:a48a79…b07` |
| `localhost/llama.cpp-intel:glm-next` | — | parked #30606 experiment — do not promote | existing |
| `localhost/llama.cpp-intel:hc-post-fusion` | `registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion` | hc_post fusion (qwen scalar + GLM affine gates) + probe fix; GPU-validated test image | built on this host, pulled to Apollo — id `fada1ff179f8a` (final; earlier `72d5fa…`, `e0f94d…`, `1ae3936…` superseded) |
| `localhost/llama.cpp-intel:hc-post-fusion-build` | `registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion-build` | build stage (oneAPI compile env + `/app/build`) | id `9444e99bcfee` |
| `localhost/llama.cpp-intel:hc-post-fusion-tests` | `registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion-tests` | test-runner (test-backend-ops compiled in, tests ON) | id `79a3cbe292ca` |

A persistent container `lcpp-hcpost-build` (from `hc-post-fusion-build`) holds a compiled
`/app/build/bin/test-backend-ops` for the GPU-host session; remove after use:
`podman rm -f lcpp-hcpost-build`.

Note on base image: `debian:trixie-slim` is the general-purpose preferred base, but this
build requires the Intel oneAPI toolchain (`icx`/`icpx`, DPC++, level-zero, oneDNN) which
only exists in `intel/oneapi-toolkit` / `intel/deep-learning-essentials`; the upstream
`.devops/intel.Dockerfile` is therefore used unmodified as the build recipe.

## Build

```bash
cd /root/code/llama.cpp-glm-next
podman build --target server \
  -t localhost/llama.cpp-intel:hc-post-fusion \
  -t registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion \
  -f .devops/intel.Dockerfile .
```

- `--target server` transitively builds the `build` stage (cmake with `-DGGML_SYCL=ON
  -DGGML_SYCL_DNN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_SYCL_F16=ON`),
  then assembles the server-only runtime image. ~40–60 min cold, 10–20 min warm, ~40 GB disk.
- The recipe is preserved in-image at `/app/intel.Dockerfile` (`COPY` in the server stage).
- Build log for the `hc-post-fusion` image: `/root/build-logs/hc-post-fusion-build.log`.

## Compile-validate test-backend-ops (no GPU needed)

The image build disables tests (`-DLLAMA_BUILD_TESTS=OFF`); re-enable incrementally in a
container from the build-stage image (done for `hc-post-fusion`: compiles clean):

```bash
podman run --rm registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion-build bash -c '
  cmake -B /app/build -DLLAMA_BUILD_TESTS=ON &&
  cmake --build /app/build --target test-backend-ops -j"$(nproc)"'
```

Known limitation: running SYCL kernels on a CPU-only host (OpenCL CPU device) is **not**
possible with this tree — the vendored `ggml/src/ggml-sycl/dpct/helper.hpp` dev_mgr throws
`"can not find preferred GPU platform"` when no platform exposes a GPU device, regardless
of `ONEAPI_DEVICE_SELECTOR`. Runtime (numerics) validation therefore requires the GPU host;
only compilation can be validated elsewhere.

## Run (GPU host — unique container name + port per the ops policy in README_FORK §8)

```bash
podman run --rm --name llama-hcpost-ab --network host --device /dev/dri \
  --group-add keep-groups \
  -v /models:/models:ro \
  localhost/llama.cpp-intel:hc-post-fusion \
  --model <model gguf> … # recipes: README_FORK.md "How to run"
```

**IMPORTANT:** `--group-add keep-groups` is mandatory on this rig — rootless podman maps
container root to the `llm` user, and without the host's video/render groups level-zero
fails with `ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS` and only the OpenCL CPU device
enumerates (symptom: "can not find preferred GPU platform"). The scripts in
`scripts/apollo-hcpost/` all include it.

Env for both models: `GGML_SYCL_FA_ONEDNN=1 SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1
CCL_P2P_ENABLE=1 CCL_D2D=p2p`. Fusion A/B lever: `GGML_SYCL_ENABLE_FUSION=0/1` (default 1).

Apollo session scripts (kept under version control here, synced to
`apollo:/home/llm/hcpost-tests/scripts/`): `hcpost-ops.sh` (backend-ops),
`hcpost-boot.sh` + `hcpost-bm.py` (A/B arms), `hcpost-fired.sh` (dispatch-count
evidence), `gpqa-boot.sh` + `gpqa_client.py` + `gpqa_diamond.jsonl` (GPQA-Diamond).

## Push (after GPU-host validation passes, or immediately as backup)

```bash
podman push registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion
```
