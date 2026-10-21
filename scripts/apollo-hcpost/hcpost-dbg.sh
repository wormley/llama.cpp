#!/bin/bash
# hcpost-dbg.sh — fusion-fires evidence: op-dispatch counts for the affine
# gated DSV4_HC_POST whole-graph test, fusion on vs off. Fewer scale/mul/add/
# sigmoid dispatches with fusion on (and equal hc_post calls) proves the
# fused kernels actually fire.
set -u
IMG="${IMG:-registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion-tests}"
BASE=/home/llm/hcpost-tests/ops
SUMMARY=/home/llm/hcpost-tests/summary.txt

for arm in on off; do
  if [ $arm = on ]; then FV=1; else FV=0; fi
  podman run --name hcpost-dbg3-$arm --group-add keep-groups --device /dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path -e SYCL_CACHE_PERSISTENT=0 -e ZES_ENABLE_SYSMAN=1 \
    -e GGML_SYCL_ENABLE_FUSION=$FV -e GGML_SYCL_DEBUG=1 \
    --entrypoint /bin/bash "$IMG" -c \
    "source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null 2>&1; exec /app/build/bin/test-backend-ops -b SYCL0 -o 'DSV4_HC_POST.*affine=1'" \
    > "$BASE/dbg3-$arm.log" 2>&1
  echo "hcpost-dbg3-$arm (affine=1 cases, FUSION=$FV) rc=$?:" >> $SUMMARY
  grep -oE '\[SYCL\]\[OP\] call ggml_sycl_[a-z_0-9]+' "$BASE/dbg3-$arm.log" | sort | uniq -c | sort -rn | head -10 | sed 's/^/  /' >> $SUMMARY
done
