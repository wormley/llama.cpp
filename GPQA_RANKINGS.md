# GPQA_RANKINGS.md — GPQA-Diamond on the Apollo rig (2026-10-21)

Rankings file for the hc-post-fusion validation session. Produced with the
llama.cpp fork `/root/code/llama.cpp-glm-next` (branch `glm-rebase`, tip
`578a44452`), image `registry.wormley.com/ai/llama.cpp-intel:hc-post-fusion`
(id `fada1ff179f8a`, built on the build host, pulled to Apollo — full
build→push→pull→run chain). Runner: `/root/code/llama.cpp-glm-next/scripts/apollo-hcpost/gpqa-boot.sh`
+ `gpqa_client.py`. Per-question evidence: `apollo:/home/llm/hcpost-tests/gpqa-<model>-<tag>/outputs/`.

## Rankings

| rank | model | recipe | correct | accuracy | mean gen tok/q | mean s/q | total time |
|---|---|---|---|---|---|---|---|
| 1 | **GLM-5.3-Flash** (UD-Q4_K_XL) | ngl 24, f16 KV, MTP3, reasoning on (effort low) | **145/198** | **73.23 %** | 202.7 | 42.6 s | 2 h 20 m |
| 2 | Qwen3.8-Flash-Next (UD-Q6_K_XL) | ngl 34, q8_0 KV, MTP3, thinking off | 103/198 | 52.02 % | 120.2 | 8.7 s | 29 m |
| 3 | Qwen3.8-Flash-Next (Q8_0) | ngl 30, q8_0 KV, MTP3, thinking off | 101/198 | 51.01 % | 113.2 | 9.0 s | 30 m |

## Methodology

- **Dataset:** GPQA-Diamond, 198 questions. Source: HF mirror
  `fingertap/GPQA-Diamond` (`test/gpqa_diamond.parquet`, sha256 prefix
  `fdd6e95117cdf870`) — the official `Idavidrein/gpqa` HF dataset is gated and no
  token was available; the official GitHub repo does not serve the CSVs directly.
  Converted to JSONL (`question`, `answer` letter) in
  `scripts/apollo-hcpost/gpqa_diamond.jsonl`. Answer distribution A:55 B:54 C:48 D:41.
- **Protocol:** `/v1/chat/completions`, temperature 0 (greedy), max_tokens 2048,
  sequential requests (np 1), fixed option order as published (identical prompt
  set for every model, so the ranking comparison is fair; no per-model shuffling —
  position bias, if any, affects all models equally).
- **Prompt:** the published question text + "\n\nAnswer with only the letter of
  the correct option." Qwen arms run with `enable_thinking:false`, GLM with
  `reasoning_effort:"low"` (chat template kwargs) — matched *budget*, not matched
  *thinking mode*; GLM's 73 % comes with roughly 2× the generation tokens.
- **Extraction:** last standalone A–D letter in the assistant content, falling
  back to the reasoning channel; per-question dumps retained for audit.
- **Constraints compliance:** quantizations untouched (UD-Q4_K_XL / Q8_0 / Q6_K_XL
  as published; GLM shard-1 metadata-only archfix overlay), KV full sized,
  no truncation of questions, models mounted read-only, UR_RESULT=0 and no dmesg
  faults in any arm.

## Caveats

- 198 questions ⇒ 1 σ ≈ 3.6 %. qwen Q8 vs Q6 (51.0 vs 52.0) is a statistical tie;
  GLM's lead (+21 pts) is far outside noise.
- GLM's effort level ("low") still emits ~200 reasoning tokens/question; its
  accuracy is achieved with substantially more test-time compute than the qwen
  arms (thinking disabled). A thinking-enabled qwen arm was not run (time budget)
  and would be the like-for-like comparison for a leaderboard-style number.
- Absolute values are **not** comparable to published GPQA scores (different
  prompt/scaffold/decoding budget). The deliverable is the ranking under a fixed,
  documented budget on this rig.
- All three arms ran on the fusion build (`GGML_SYCL_ENABLE_FUSION=1`) with the
  hc_post gate fusions active and zero GPU faults.
