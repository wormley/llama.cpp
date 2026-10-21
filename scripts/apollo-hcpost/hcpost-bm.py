#!/usr/bin/env python3
# hcpost-bm.py — gate + benchmark client for the SYCL hc_post fusion A/B on Apollo.
#
# Methodology (per glm53-lcpp-tests / qwen38-lcpp-tests clients):
#   temperature 0, streaming, unique-nonce sized prompts (defeats prefix cache),
#   arith gate 17*23=391, needle retrieval in every decode request,
#   decode 3x128 @2k ctx, prefill rungs 4k / 16k.
#
# Differences for this A/B:
#   - NONCES ARE FIXED (not random) so the fusion-on and fusion-off arms see
#     byte-identical prompts; greedy outputs must then be bit-identical and are
#     diffed across arms by the driver.
#   - Every assistant reply (reasoning + content) is dumped to OUTDIR/<name>.txt
#     for that identity check.
#   - Client-side tok/s is secondary (SSE buffering distorts it — README_FORK
#     item 1); the authoritative numbers are the server "prompt eval time"/
#     "eval time" lines parsed from the container log by hcpost-boot.sh.
#
# Usage: hcpost-bm.py <port> <tag> <alias> <chat_template_kwargs_json> <outdir>
import json, sys, time, urllib.request, os

port, tag, alias, tmpl, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
URL = "http://127.0.0.1:%s" % port
H = {"Content-Type": "application/json", "Authorization": "Bearer abc"}
F = "Routine inventory notes and maintenance logs from regional depots. "
KWT = json.loads(tmpl) if tmpl else {}

os.makedirs(OUTDIR, exist_ok=True)

def req(payload, timeout=7200):
    body = json.dumps(payload).encode()
    return urllib.request.urlopen(
        urllib.request.Request(URL + "/v1/chat/completions", data=body, headers=H), timeout=timeout)

def count(t):
    r = urllib.request.urlopen(urllib.request.Request(URL + "/tokenize",
        data=json.dumps({"content": t}).encode(), headers=H), timeout=600)
    return len(json.load(r)["tokens"])

def build(reps, code, nonce, question):
    b = F * reps; h = len(b) // 2
    return "[%s] " % nonce + b[:h] + "\nThe secret access code is " + code + ".\n" + b[h:] + "\n" + question

NEEDLE_Q = ("Question: What is the secret access code? First reply with only the code, "
            "then list the numbers from 1 to 25.")

def sized(nonce, target, question=NEEDLE_Q):
    lo, hi = 1, 400000
    for _ in range(22):
        m = (lo + hi) // 2
        if count(build(m, "X", nonce, question)) < target: lo = m + 1
        else: hi = m
    code = "AZ" + nonce
    return build(lo, code, nonce, question), count(build(lo, code, nonce, question)), code

def chat(msg, mx, name, timeout=7200):
    payload = {"model": alias, "messages": [{"role": "user", "content": msg}],
               "temperature": 0, "max_tokens": mx, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": KWT}
    t0 = time.time(); first = None; text = ""; ptok = 0; ntok = 0
    try:
        r = req(payload, timeout)
        for rl in r:
            line = rl.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: obj = json.loads(d)
            except Exception: continue
            if obj.get("usage"):
                u = obj["usage"]; ptok = u.get("prompt_tokens", ptok); ntok = u.get("completion_tokens", ntok)
            ch = obj.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                c = delta.get("content") or ""
                rr = delta.get("reasoning_content") or ""
                if (c or rr) and first is None: first = time.time() - t0
                text += ("[R]%s" % rr if rr else "") + (c if c else "")
    except Exception as e:
        print("%s REQ_EXC %s %s" % (tag, name, type(e).__name__ + " " + str(e)[:120]), flush=True)
        return None
    gen = time.time() - t0
    with open(os.path.join(OUTDIR, name + ".txt"), "w") as f:
        f.write(text)
    rate = (ntok / (gen if gen > 0 else 1e-9)) if mx > 1 else (ptok / (first if first else 1e-9))
    print("%s REQ %s ptok=%d ntok=%d client=%.1f tok/s" % (tag, name, ptok, ntok, rate), flush=True)
    return (ptok, ntok, text)

# 1. arith gate
_, _, r = chat("Compute 17*23. Reply with only the number.", 512, "arith")
print("%s ARITH %s" % (tag, "OK" if r and "391" in r else "FAIL"), flush=True)

# 2. decode: 3 x ~384-token generations at ~2k ctx, needle in every prompt
# (max_tokens high enough for reasoning models to finish; deterministic tail
# keeps the generation length usable for timing)
for i, nonce in enumerate(("d1", "d2", "d3"), 1):
    msg, ptok, code = sized(nonce, 2048)
    _, _, r = chat(msg, 384, "decode-" + nonce)
    ok = r is not None and code in r
    print("%s DECODE-%d %s ctx=%d needle=%s" % (tag, i, "OK" if r else "EXC", ptok,
          "PASS" if ok else "FAIL"), flush=True)

# 3. prefill rungs: 1 output token each (server 'prompt eval time' is the metric;
# the needle check is not applicable at 1 token)
for nonce, tgt in (("pA", 4096), ("pB", 16384)):
    msg, ptok, code = sized(nonce, tgt)
    _, _, r = chat(msg, 1, "prefill-" + nonce)
    print("%s PREFILL-%s ctx=%d needle=n/a" % (tag, nonce.upper(), ptok), flush=True)

print("%s DONE" % tag, flush=True)
