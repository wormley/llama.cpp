# CONTAINERS.md — image inventory for this tree

Per `/root/container.txt` policy: podman, images pushed to `registry.wormley.com/ai/`
(environment is ephemeral), Dockerfile preserved in-image, names recorded here.

## Images built from this tree (`.devops/intel.Dockerfile`)

| tag (local) | tag (registry) | purpose | status |
|---|---|---|---|
| `localhost/llama.cpp-intel:latest` | — | promoted deliverable (branch `glm-rebase`, see README_FORK.md) | existing, manifest `sha256:a48a79…b07` |
| `localhost/llama.cpp-intel:glm-next` | — | parked #30606 experiment — do not promote | existing |
| `localhost/llama.cpp-intel:hc-post-fusion` | `registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion` | SYCL HC-post gate fusion (open item 3) compile-validated test image, for GPU-host A/B | built, pushed — id `72d5fa130df4` |
| `localhost/llama.cpp-intel:hc-post-fusion-build` | `registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion-build` | build stage (full oneAPI compile env + `/app/build`), also used to build `test-backend-ops` | built, pushed — id `85a523c2070f` |

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
  -v /models:/models:ro \
  localhost/llama.cpp-intel:hc-post-fusion \
  --model <model gguf> … # recipes: README_FORK.md "How to run"
```

Env for both models: `GGML_SYCL_FA_ONEDNN=1 SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1
CCL_P2P_ENABLE=1 CCL_D2D=p2p`. Fusion A/B lever: `GGML_SYCL_ENABLE_FUSION=0/1` (default 1).

## Push (after GPU-host validation passes, or immediately as backup)

```bash
podman push registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion
```
