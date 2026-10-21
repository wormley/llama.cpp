#!/usr/bin/env python3
# gpqa_client.py — GPQA-Diamond (198 q) against a llama.cpp server on Apollo.
#
# Protocol: temperature 0, sequential (np 1), fixed option order (as published —
# same order for every model, so the ranking comparison is fair; noted in the
# results file), letter extraction from the final answer text, per-question dump
# for auditing. Server-side timings are parsed from the container log by the
# driver script.
#
# Usage: gpqa_client.py <port> <tag> <alias> <chat_template_kwargs_json> <outdir>
import json, sys, re, time, urllib.request, os, glob

port, tag, alias, tmpl, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
URL = "http://127.0.0.1:%s" % port
H = {"Content-Type": "application/json", "Authorization": "Bearer abc"}
KWT = json.loads(tmpl) if tmpl else {}

# dataset lives next to this script
DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpqa_diamond.jsonl")
QUESTIONS = [json.loads(l) for l in open(DS, encoding="utf-8")]

os.makedirs(OUTDIR, exist_ok=True)
RESULT_FILE = os.path.join(OUTDIR, "results.jsonl")

LETTER_RE = re.compile(r"\b([A-Da-d])\b(?![a-zA-Z])")

def extract_letter(text):
    if not text:
        return None
    matches = LETTER_RE.findall(text)
    return matches[-1].upper() if matches else None

def ask(msg, mx, timeout=1800):
    payload = {"model": alias, "messages": [{"role": "user", "content": msg}],
               "temperature": 0, "max_tokens": mx, "stream": False,
               "chat_template_kwargs": KWT}
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions",
                               data=json.dumps(payload).encode(), headers=H), timeout=timeout)
    obj = json.loads(r.read())
    ch = obj["choices"][0]["message"]
    content = ch.get("content") or ""
    reasoning = ch.get("reasoning_content") or ""
    u = obj.get("usage", {})
    return content, reasoning, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0

def main():
    done = {}
    if os.path.exists(RESULT_FILE):  # resume support
        for l in open(RESULT_FILE, encoding="utf-8"):
            done[json.loads(l)["id"]] = True
    n_correct = 0; n_answered = 0; n_total = 0
    t_start = time.time()
    with open(RESULT_FILE, "a", encoding="utf-8") as out:
        for q in QUESTIONS:
            n_total += 1
            if q["id"] in done:
                continue
            msg = (q["question"].strip() +
                   "\n\nAnswer with only the letter of the correct option.")
            try:
                content, reasoning, ptok, ntok, dt = ask(msg, 2048)
            except Exception as e:
                print("%s GPQA q%d EXC %s" % (tag, q["id"], type(e).__name__ + " " + str(e)[:100]), flush=True)
                continue
            pred = extract_letter(content) or extract_letter(reasoning)
            ok = (pred == q["answer"])
            n_answered += 1
            n_correct += 1 if ok else 0
            out.write(json.dumps({"id": q["id"], "pred": pred, "answer": q["answer"],
                                  "ok": ok, "ptok": ptok, "ntok": ntok, "sec": round(dt, 1)}) + "\n")
            out.flush()
            if (n_answered % 25) == 0:
                print("%s GPQA progress %d/%d acc=%.1f%% elapsed=%.0fs" %
                      (tag, n_answered, n_total, 100.0 * n_correct / max(n_answered, 1), time.time() - t_start), flush=True)
    # final score over all dumped results (incl. resumed)
    ids = {}; n_correct = 0
    for l in open(RESULT_FILE, encoding="utf-8"):
        r = json.loads(l); ids[r["id"]] = r
        n_correct += 1 if r["ok"] else 0
    n = len(ids)
    print("%s GPQA_SCORE answered=%d/198 correct=%d acc=%.2f%% total_time=%.0fs" %
          (tag, n, n_correct, 100.0 * n_correct / max(n, 1), time.time() - t_start), flush=True)

main()
