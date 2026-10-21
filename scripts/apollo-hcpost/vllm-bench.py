#!/usr/bin/env python3
# vllm-bench.py — same benchmark set as hcpost-bm.py against the vLLM FP8 server:
# arith gate + needle, decode 3x256 fixed @2k ctx (ignore_eos), prefill 4k/16k.
# vLLM has no llama.cpp-style slot timing log, so decode tok/s is measured from
# the SSE stream (first-chunk to last-chunk, excluding prefill) and prefill from
# TTFT — the same client-side methodology as the earlier vLLM test packages.
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
        data=json.dumps({"model": alias, "prompt": t}).encode(), headers=H), timeout=600)
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

def chat(msg, mx, name, ignore_eos=False, timeout=7200):
    payload = {"model": alias, "messages": [{"role": "user", "content": msg}],
               "temperature": 0, "max_tokens": mx, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": KWT}
    if ignore_eos:
        payload["ignore_eos"] = True
    t0 = time.time(); first = None; last = None; text = ""; ptok = 0; ntok = 0
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
                if (c or rr):
                    if first is None: first = time.time() - t0
                    last = time.time() - t0
                text += ("[R]%s" % rr if rr else "") + (c if c else "")
    except Exception as e:
        print("%s REQ_EXC %s %s" % (tag, name, type(e).__name__ + " " + str(e)[:120]), flush=True)
        return None
    decode_s = (last - first) if (first is not None and last and last > first) else 1e-9
    with open(os.path.join(OUTDIR, name + ".txt"), "w") as f:
        f.write(text)
    ttft = first if first is not None else 0
    dec_rate = (ntok - 1) / decode_s if ntok > 1 else 0
    prefill_rate = ptok / ttft if ttft > 0 else 0
    print("%s REQ %s ptok=%d ntok=%d ttft=%.1fs decode=%.1f tok/s prefill=%.0f tok/s" %
          (tag, name, ptok, ntok, ttft, dec_rate, prefill_rate), flush=True)
    return (ptok, ntok, text)

# 1. arith gate
_, _, r = chat("Compute 17*23. Reply with only the number.", 512, "arith")
print("%s ARITH %s" % (tag, "OK" if r and "391" in r else "FAIL"), flush=True)

# 2. decode: 3 x 256-token fixed-length generations at ~2k ctx
for i, nonce in enumerate(("d1", "d2", "d3"), 1):
    msg, ptok, code = sized(nonce, 2048)
    _, _, r = chat(msg, 256, "decode-" + nonce, ignore_eos=True)
    ok = r is not None and code in r
    print("%s DECODE-%d %s ctx=%d needle=%s" % (tag, i, "OK" if r else "EXC", ptok,
          "PASS" if ok else "FAIL"), flush=True)

# 3. prefill rungs: 1 output token each, TTFT is the prefill metric
for nonce, tgt in (("pA", 4096), ("pB", 16384)):
    msg, ptok, code = sized(nonce, tgt)
    _, _, r = chat(msg, 1, "prefill-" + nonce)
    print("%s PREFILL-%s ctx=%d needle=n/a" % (tag, nonce.upper(), ptok), flush=True)

print("%s DONE" % tag, flush=True)
