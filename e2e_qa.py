"""e2e_qa.py --- end-to-end QA harness: does better retrieval -> better answers?

Retrieval-stage metrics (Hit@k, NDCG) are a proxy. This harness closes the
loop: it feeds the top-k retrieved *sessions* to an LLM reader and scores the
generated answer against the LoCoMo gold answer, so we can compare answer
quality across retrievers (BM25 vs. BM25(+)max-sim fusion vs. oracle gold).

Reader: Anthropic Claude (messages API, with prompt caching on the shared
instruction block). Scorer: normalized token-F1 + exact-ish match, plus
abstention accuracy on adversarial (unanswerable) questions. No training.

REQUIRES: env var ANTHROPIC_API_KEY. Costs API tokens. Start small with
--sample. CPU for retrieval; the LLM call is the only network/billed part.

    python e2e_qa.py --method fusion --k 3 --sample 50
    python e2e_qa.py --method bm25   --k 3 --sample 50
    python e2e_qa.py --method oracle --k 3 --sample 50    # gold sessions
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import locomo
from tune7_bm25 import ConvStats, score_classic, _tokenize_text

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LOCOMO = HERE / "data" / "locomo" / "locomo10.json"
READER_MODEL = "claude-haiku-4-5"   # fast/cheap reader; override with --model


def _z(v):
    v = np.asarray(v, float); sd = v.std()
    return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd


# ----------------------------- scoring ------------------------------------
def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return [w for w in s.split() if w not in {"the", "a", "an", "of", "to", "is"}]


def token_f1(pred, gold):
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def is_abstention(pred):
    pred = (pred or "").lower()
    return any(t in pred for t in ["not answerable", "cannot", "can't", "no information",
                                   "not mentioned", "unanswerable", "don't know",
                                   "no relevant", "not enough information"])


# --------------------------- retrieval ------------------------------------
def rankings(exs, method, k):
    """Return list of top-k session-index lists per example."""
    tc = np.load(RES / "locomo_turn_cache.npz", allow_pickle=True)
    q_emb = np.load(RES / "locomo_bge_cache.npz", allow_pickle=True)["q_emb"]
    cs = {}
    out = []
    for i, e in enumerate(exs):
        ci = e["conv_idx"]
        sids = e["session_ids"]
        if method == "oracle":
            gold = [j for j, s in enumerate(sids) if s in e["gold_session_ids"]]
            out.append(gold[:k] if gold else list(range(min(k, len(sids)))))
            continue
        if ci not in cs:
            cs[ci] = ConvStats([_tokenize_text(s) for s in e["haystack_sessions"]])
        bm = score_classic(cs[ci], set(_tokenize_text(e["question"])), 1.5, 0.75)
        if method == "bm25":
            score = bm
        else:  # fusion
            qv = q_emb[i]
            S = len(sids)
            dense = np.full(S, -9.0)
            for j, sid in enumerate(sids):
                key = f"turns_{ci}_{int(sid[1:])}"
                if key in tc.files:
                    dense[j] = float((tc[key] @ qv).max())
            score = 0.6 * _z(bm) + 0.4 * _z(dense)
        out.append(list(np.argsort(-score)[:k]))
    return out


# ----------------------------- reader -------------------------------------
SYS = ("You answer questions about a long conversation between two people, using "
       "ONLY the provided excerpts. Answer in as few words as possible (a name, "
       "date, phrase). If the excerpts do not contain the answer, reply exactly "
       "'NOT ANSWERABLE'. Do not explain.")


def answer(client, model, context, question):
    msg = client.messages.create(
        model=model, max_tokens=64,
        system=[{"type": "text", "text": SYS,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user",
                   "content": f"Excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bm25", "fusion", "oracle"], default="fusion")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=READER_MODEL)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY to run the LLM reader. "
              "Retrieval is free/CPU; only the reader is billed.\n"
              "  export ANTHROPIC_API_KEY=sk-...   (PowerShell: $env:ANTHROPIC_API_KEY='sk-...')",
              file=sys.stderr)
        return 2
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        return 2
    client = anthropic.Anthropic()

    exs = [e for e in locomo.iter_examples(LOCOMO)]
    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(exs):
        idx = sorted(rng.choice(len(exs), size=args.sample, replace=False).tolist())
        exs = [exs[i] for i in idx]
    ranks = rankings(exs, args.method, args.k)

    rows, f1s, adv_correct, adv_n, ans_correct, ans_n = [], [], 0, 0, 0, 0
    for e, top in zip(exs, ranks):
        ctx = "\n\n".join(f"[Session {e['session_ids'][j]}]\n{e['haystack_sessions'][j]}"
                          for j in top)
        pred = answer(client, args.model, ctx, e["question"])
        is_adv = e.get("category") == 5 or e.get("answer") in (None, "None", "")
        if is_adv:
            adv_n += 1
            ok = is_abstention(pred)
            adv_correct += int(ok)
            f1 = float(ok)
        else:
            ans_n += 1
            f1 = token_f1(pred, str(e.get("answer", "")))
            ans_correct += int(f1 >= 0.5)
        f1s.append(f1)
        rows.append({"q": e["question"], "gold": e.get("answer"),
                     "pred": pred, "category": e.get("category"), "f1": f1,
                     "adversarial": is_adv})

    summary = {
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
        "method": args.method, "k": args.k, "n": len(exs), "model": args.model,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "answerable_acc@f1>=0.5": (ans_correct / ans_n) if ans_n else None,
        "adversarial_abstention_acc": (adv_correct / adv_n) if adv_n else None,
    }
    out = RES / f"e2e-{args.method}-k{args.k}-{summary['ts']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2e.json").write_text(json.dumps({"summary": summary, "rows": rows},
                                             indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out/'e2e.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
