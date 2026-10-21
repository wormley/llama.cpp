#!/bin/bash
# gpqa-boot.sh — GPQA-Diamond run for one model on Apollo (fusion build).
# Usage: MODEL=glm|qwen8|qwen6 TAG=g1 bash gpqa-boot.sh
# Ops policy per GPU_TESTING.md: scratch under /home/llm/hcpost-tests/, unique
# container/port, evidence captured, no --rm, read-only model mounts.
set -u
MODEL="${MODEL:?glm|qwen8|qwen6}"
TAG="${TAG:?tag}"
IMG="${IMG:-registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion}"
BASEROOT=/home/llm/hcpost-tests
BASE="$BASEROOT/gpqa-$MODEL-$TAG"
NAME="gpqa-$MODEL-$TAG"
SUMMARY="$BASEROOT/summary.txt"
mkdir -p "$BASE/cache" "$BASE/tmp" "$BASE/outputs"; export TMPDIR="$BASE/tmp"

NN=${TAG//[!0-9]/}; [ -z "$NN" ] && NN=90
if [ "$MODEL" = glm ]; then PORT=$((18510 + NN)); ARGL=glm; else PORT=$((18520 + NN)); fi
if ss -ltn | grep -q ":$PORT "; then
  echo "gpqa-$NAME PORT $PORT IN USE — another TAG" | tee -a "$SUMMARY"; exit 1
fi
for c in $(podman ps --format '{{.Names}}' | grep -E '^(hcpost-|gpqa-)' || true); do
  podman stop "$c" >/dev/null 2>&1
done

if [ "$MODEL" = glm ]; then
  MOUNTS="-v /models/GLM-5.3-Flash-GGUF:/models/GLM-5.3-Flash-GGUF:ro"
  PATCH=/home/llm/qwen38-tests/glm53-lcpp/gguf-patched/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
  MOUNTS="$MOUNTS -v $PATCH:/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf:ro"
  MARGS="--model /models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
    --n-gpu-layers 24 --reasoning on --alias glm53"
  ALIAS=glm53 ; CTARGS='{"reasoning_effort":"low"}'
elif [ "$MODEL" = qwen8 ]; then
  MOUNTS="-v /models/Qwen3.8-Flash-Next-GGUF:/models/Qwen3.8-Flash-Next-GGUF:ro"
  MARGS="--model /models/Qwen3.8-Flash-Next-GGUF/Q8_0/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf \
    --model-draft /models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --n-gpu-layers 30 --cache-type-k q8_0 --cache-type-v q8_0 \
    --override-tensor per_layer_token_embd=CPU,ple_ngram_embd=CPU \
    --reasoning-preserve --reasoning on --alias qwen38"
  ALIAS=qwen38 ; CTARGS='{"enable_thinking":false}'
elif [ "$MODEL" = qwen6 ]; then
  MOUNTS="-v /models/Qwen3.8-Flash-Next-GGUF:/models/Qwen3.8-Flash-Next-GGUF:ro"
  MARGS="--model /models/Qwen3.8-Flash-Next-GGUF/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf \
    --model-draft /models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --n-gpu-layers 34 --cache-type-k q8_0 --cache-type-v q8_0 \
    --override-tensor per_layer_token_embd=CPU,ple_ngram_embd=CPU \
    --reasoning-preserve --reasoning on --alias qwen38"
  ALIAS=qwen38 ; CTARGS='{"enable_thinking":false}'
else
  echo "MODEL must be glm|qwen8|qwen6" | tee -a "$SUMMARY"; exit 1
fi

dmesg -wT > "$BASE/dmesg-live.log" 2>&1 &
DMESG_PID=$!

podman run -d --name "$NAME" --group-add keep-groups \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --network=host \
  --shm-size=2g --ulimit memlock=-1:-1 \
  -e GGML_SYCL_FA_ONEDNN=1 -e SYCL_CACHE_PERSISTENT=0 -e ZES_ENABLE_SYSMAN=1 \
  -e CCL_P2P_ENABLE=1 -e CCL_D2D=p2p \
  -e GGML_SYCL_ENABLE_FUSION=1 \
  $MOUNTS \
  -v "$BASE/cache:/cache:rw" \
  "$IMG" \
  --cache-ram 32768 --jinja \
  $MARGS \
  --ctx-checkpoints 8 -np 1 --cache-reuse 512 \
  --ctx-size 32768 \
  --fit off --batch-size 2048 --ubatch-size 512 \
  --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 0.0.0.0 --port "$PORT" --log-colors off \
  > "$BASE/run.out" 2>&1
podman logs -f "$NAME" > "$BASE/container.log" 2>&1 &
LOG_PID=$!

HB=""
for i in $(seq 1 720); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && HB=$((i*5)) && break
  podman ps --format '{{.Names}}' | grep -qx "$NAME" || break
  sleep 5
done
if [ -z "${HB:-}" ]; then
  echo "gpqa-$NAME BOOT_FAIL" >> "$SUMMARY"
  grep -E "error|Error|failed|DEVICE_LOST|UR_RESULT|out of memory|OOM|Exception" \
    "$BASE/container.log" | tail -12 | sed 's/^/  /' >> "$SUMMARY"
  podman logs "$NAME" > "$BASE/container-final.log" 2>&1
  dmesg -T > "$BASE/dmesg-exit.log" 2>/dev/null || true
  podman ps --format '{{.Names}}' | grep -qx "$NAME" && podman stop "$NAME" >/dev/null 2>&1
  kill $LOG_PID $DMESG_PID 2>/dev/null || true
  exit 1
fi
echo "gpqa-$NAME IMG=$IMG health=${HB}s" >> "$SUMMARY"
grep -E "fused DeepSeek V4 HC post (enabled|not supported)" "$BASE/container.log" \
  | sed 's/^/gpqa-'"$NAME"' FUSEDOP /' >> "$SUMMARY"

python3 "$(dirname "$0")/gpqa_client.py" "$PORT" "$NAME" "$ALIAS" "$CTARGS" "$BASE/outputs" \
  2>&1 | tee -a "$SUMMARY"

podman logs "$NAME" > "$BASE/container-final.log" 2>&1
dmesg -T > "$BASE/dmesg-exit.log" 2>/dev/null || true
echo "gpqa-$NAME UR=$(grep -c UR_RESULT "$BASE/container.log" || true) faults_live=$(grep -icE 'fault response|timedout|engine reset' "$BASE/dmesg-live.log" 2>/dev/null || true)" >> "$SUMMARY"

# mean server-side generation speed across GPQA requests
python3 - "$BASE/container.log" "$NAME" "$SUMMARY" <<'EOF'
import re, sys
log, name, summary = open(sys.argv[1], errors="replace").read(), sys.argv[2], sys.argv[3]
ev = [float(m) for m in re.findall(r"eval time =\s*[\d.]+ ms /\s*\d+ tokens \(.*?,\s*([\d.]+) tokens per second", log)]
# skip the first (JIT-cold) request when averaging
warm = ev[1:] if len(ev) > 3 else ev
if warm:
    with open(summary, "a") as f:
        f.write("gpqa-%s SERVER gen mean(warm)=%.1f tok/s over %d reqs (cold=%.1f)\n" %
                (name, sum(warm)/len(warm), len(warm), ev[0] if ev else 0))
EOF

podman ps --format '{{.Names}}' | grep -qx "$NAME" && podman stop "$NAME" >/dev/null 2>&1
kill $LOG_PID $DMESG_PID 2>/dev/null || true
