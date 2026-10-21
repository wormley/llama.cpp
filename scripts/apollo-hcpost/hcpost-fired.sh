#!/bin/bash
# hcpost-fired.sh — decisive fusion-fires evidence on the real GLM graph:
# boot GLM twice (FUSION=1 / FUSION=0) with GGML_SYCL_DEBUG=1, issue ONE small
# request each, and compare SYCL op-dispatch counts. If the hc_post gate fusion
# fires, the f1 arm dispatches fewer scale/mul/add/sigmoid ops per hc chain.
set -u
IMG="${IMG:-registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion}"
BASE=/home/llm/hcpost-tests/fired
SUMMARY=/home/llm/hcpost-tests/summary.txt
mkdir -p "$BASE/cache"

PORT=18610
PATCH=/home/llm/qwen38-tests/glm53-lcpp/gguf-patched/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf

for FV in 1 0; do
  NAME="fired-glm-f$FV"
  podman rm -f "$NAME" >/dev/null 2>&1
  dmesg -wT > "$BASE/dmesg-f$FV.log" 2>&1 &
  DPID=$!
  podman run -d --name "$NAME" --group-add keep-groups \
    --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --network=host \
    --shm-size=2g --ulimit memlock=-1:-1 \
    -e GGML_SYCL_FA_ONEDNN=1 -e SYCL_CACHE_PERSISTENT=0 -e ZES_ENABLE_SYSMAN=1 \
    -e CCL_P2P_ENABLE=1 -e CCL_D2D=p2p \
    -e GGML_SYCL_ENABLE_FUSION=$FV -e GGML_SYCL_DEBUG=1 \
    -v /models/GLM-5.3-Flash-GGUF:/models/GLM-5.3-Flash-GGUF:ro \
    -v $PATCH:/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf:ro \
    -v "$BASE/cache:/cache:rw" \
    "$IMG" \
    --cache-ram 32768 --jinja \
    --model /models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --n-gpu-layers 24 --reasoning on --alias glm53 \
    --ctx-checkpoints 8 -np 1 --cache-reuse 512 \
    --ctx-size 8192 --fit off --batch-size 2048 --ubatch-size 512 \
    --flash-attn on --spec-type draft-mtp --spec-draft-n-max 3 \
    --host 0.0.0.0 --port $PORT --log-colors off \
    > "$BASE/run-f$FV.out" 2>&1
  podman logs -f "$NAME" > "$BASE/container-f$FV.log" 2>&1 &
  LPID=$!
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    podman ps --format '{{.Names}}' | grep -qx "$NAME" || break
    sleep 5
  done
  curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -H "Authorization: Bearer abc" \
    -d '{"model":"glm53","messages":[{"role":"user","content":"Compute 17*23. Reply with only the number."}],"temperature":0,"max_tokens":64,"stream":false,"chat_template_kwargs":{"reasoning_effort":"low"}}' \
    > "$BASE/req-f$FV.json" 2>&1
  podman logs "$NAME" > "$BASE/full-f$FV.log" 2>&1
  podman ps --format '{{.Names}}' | grep -qx "$NAME" && podman stop "$NAME" >/dev/null 2>&1
  kill $LPID $DPID 2>/dev/null || true
  echo "fired-glm-f$FV dispatch-counts:" >> "$SUMMARY"
  grep -oE '\[SYCL\]\[OP\] call ggml_sycl_[a-z_0-9]+' "$BASE/full-f$FV.log" | sort | uniq -c | sort -rn | head -14 | sed 's/^/  /' >> "$SUMMARY"
  grep -c "call ggml_sycl_op_dsv4_hc_post" "$BASE/full-f$FV.log" | sed 's/^/  dsv4_hc_post calls: /' >> "$SUMMARY"
done
