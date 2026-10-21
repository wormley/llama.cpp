#!/bin/bash
# hcpost-ops.sh — backend-op numerics tests for the hc_post fusion on Apollo.
# Runs the pulled tests image (no build on Apollo). Captures evidence per
# GPU_TESTING.md; containers are left in place (no --rm / no rm).
#
#   1. DSV4_HC.* with fusion on   — incl. new affine (GLM) variants
#   2. DSV4_HC.* with fusion off  — unfused reference
#   3. TOPK_MOE.* with fusion on  — regression (shared memory-ranges helper)
#   4. GGML_SYCL_DEBUG op-dispatch counts, fusion on vs off — proves the
#      fused kernels actually fire (fewer scale/mul/add/sigmoid dispatches)
#
# NOTE: the image's ENTRYPOINT wrapper breaks argv (vars.sh interaction) —
# invoke the binary directly via bash -c with %q-quoted args instead.
set -u
IMG="${IMG:-registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion-tests}"
BASE=/home/llm/hcpost-tests/ops
SUMMARY=/home/llm/hcpost-tests/summary.txt
mkdir -p "$BASE"

run() { # run <name> <extra env...> -- <test-backend-ops args...>
  local NAME="$1${ROUND:-}"; shift
  local ENV=()
  while [ "$1" != "--" ]; do ENV+=("-e" "$1"); shift; done; shift
  local QARGS="" a
  for a in "$@"; do QARGS+=" $(printf '%q' "$a")"; done
  podman run --name "$NAME" --group-add keep-groups --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    "${ENV[@]}" --entrypoint /bin/bash "$IMG" -c \
    "source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null 2>&1; exec /app/build/bin/test-backend-ops $QARGS" \
    > "$BASE/$NAME.log" 2>&1
  echo "hcpost-$NAME rc=$? $(grep -cE ' FAIL ' "$BASE/$NAME.log" || true) FAILs, $(grep -cE ' OK ' "$BASE/$NAME.log" || true) OKs" >> "$SUMMARY"
}

for c in $(podman ps --format '{{.Names}}' | grep -E '^hcpost-ops' || true); do
  podman stop "$c" >/dev/null 2>&1
done

run hcpost-ops-on  GGML_SYCL_ENABLE_FUSION=1 -- -b SYCL0 -o 'DSV4_HC.*'
run hcpost-ops-off GGML_SYCL_ENABLE_FUSION=0 -- -b SYCL0 -o 'DSV4_HC.*'
run hcpost-topk    GGML_SYCL_ENABLE_FUSION=1 -- -b SYCL0 -o 'TOPK_MOE'
# fusion-fires evidence: same tiny workload, dispatch-level op counts
run hcpost-dbg-on  GGML_SYCL_ENABLE_FUSION=1 GGML_SYCL_DEBUG=1 -- -b SYCL0 -o 'DSV4_HC_POST\(31,'
run hcpost-dbg-off GGML_SYCL_ENABLE_FUSION=0 GGML_SYCL_DEBUG=1 -- -b SYCL0 -o 'DSV4_HC_POST\(31,'

{
  for n in hcpost-dbg-on${ROUND:-} hcpost-dbg-off${ROUND:-}; do
    echo "hcpost-$n dispatch-counts:"
    grep -oE '\[SYCL\]\[OP\] call ggml_sycl_[a-z_0-9]+' "$BASE/$n.log" | sort | uniq -c | sort -rn | head -12 | sed 's/^/  /'
  done
} >> "$SUMMARY" 2>&1
