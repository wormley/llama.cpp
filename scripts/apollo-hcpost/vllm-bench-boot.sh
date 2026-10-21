#!/bin/bash
# vllm-bench-boot.sh — vLLM FP8 qwen3.8 benchmark boot on Apollo.
# Launch parameters identical to /home/llm/ai/vllm-qfnfp8.sh, detached (-d,
# no TTY over ssh) and with a unique container name per ops policy.
set -u
BASE=/home/llm/ai/scratch
MODEL=/models/Qwen3.8-Flash-Next-FP8
NAME=qfn38-bench
LOGDIR=/home/llm/hcpost-tests/vllm
mkdir -p "$LOGDIR" "$BASE/cache"

podman rm -f "$NAME" >/dev/null 2>&1
dmesg -wT > "$LOGDIR/dmesg-live.log" 2>&1 &
DPID=$!

podman run -d --name "$NAME" --group-add keep-groups \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --network=host \
  --shm-size=2g --ulimit memlock=-1:-1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e ZE_AFFINITY_MASK=0,1,2,3 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  -e VLLM_ENGINE_ITERATION_TIMEOUT_S=7200 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -v "$MODEL:/model:ro" -v "$BASE/cache:/cache:rw" \
  registry.wormley.com/qfn38-fp8-4xb70:latest /model --served-model-name qwen38-flash-next2 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --offload-backend uva --cpu-offload-gb 16 --cpu-offload-params w13_weight w2_weight \
  --language-model-only --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking":true}' \
  --max-model-len 262144 --max-num-seqs 1 --gpu-memory-utilization 0.88 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
  > "$LOGDIR/run.out" 2>&1
podman logs -f "$NAME" > "$LOGDIR/container.log" 2>&1 &
LPID=$!
echo "vllm booting, health wait..."
for i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1 && HB=$((i*5)) && break
  podman ps --format '{{.Names}}' | grep -qx "$NAME" || break
  sleep 5
done
if [ -z "${HB:-}" ]; then
  echo "VLLM BOOT_FAIL"; tail -20 "$LOGDIR/container.log"
  podman logs "$NAME" > "$LOGDIR/container-final.log" 2>&1
  dmesg -T > "$LOGDIR/dmesg-exit.log" 2>/dev/null || true
  podman ps --format '{{.Names}}' | grep -qx "$NAME" && podman stop "$NAME" >/dev/null 2>&1
  kill $LPID $DPID 2>/dev/null || true
  exit 1
fi
echo "VLLM health=${HB}s"
