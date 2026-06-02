"""
lme_maxsim.py — Generalization test of turn-level max-sim fusion on
LongMemEval-S.

Applies the LoCoMo champion recipe (BM25 (default) fused with
max-sim turn-level BGE-base dense, session score = max_t cos(q, turn))
to the public LongMemEval-S retrieval-challenge subset (500 questions
whose haystack has more sessions than the answer set).

This is the hard transfer test: prior work here found BM25 ALONE is very
strong on LongMemEval (Hit@1 ~0.86) because these queries have heavy
lexical overlap with the answer — the opposite regime from LoCoMo. If
turn-level fusion still helps (or at least does not hurt), the method
generalizes; if it only helped LoCoMo, that is an honest boundary.

Each question has its own haystack, so questions are independent and we
bootstrap over questions (not clusters). Reports BM25, session-mean
dense, max-sim dense, RRF, and fixed-alpha fusion (alpha chosen by
held-out CV over a question split to stay honest).

Output: results/lme-maxsim-<ts>/{lme.json, lme.md}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

DEFAULT_OUT = _HERE / "results"
MODEL = "BAAI/bge-base-en-v1.5"

# same tokenizer as the published BM25 baseline
_TOK_BAN = {"the", "and", "for", "with", "from", "this", "that", "what",
            "where", "when", "how", "why", "who", "are", "you", "was",
            "have", "has", "did", "do", "does", "is", "it", "be", "been",
            "an", "as", "we", "me", "my", "i", "your", "yours", "our",
            "of", "to", "on", "in", "at", "by", "or", "if", "but", "not",
            "a", "so", "all", "any", "no", "yes"}
_TOK_RE = re.compile(r"[a-z0-9][a-z0-9\-_']{1,30}")


def _tok(s):
    s = (s or "").lower()
    return [t for t in _TOK_RE.findall(s) if t not in _TOK_BAN and len(t) >= 3]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [_tok(d) for d in docs]
        self.dl = np.array([len(d) for d in self.docs], float)
        self.avgdl = self.dl.mean() if len(self.dl) else 1.0
        self.N = len(self.docs)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for c in self.tf:
            for t in c:
                df[t] += 1
        self.idf = {t: np.log((self.N - d + 0.5) / (d + 0.5) + 1.0)
                    for t, d in df.items()}

    def score(self, q):
        qt = set(_tok(q))
        s = np.zeros(self.N)
        for t in qt:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for j in range(self.N):
                f = self.tf[j].get(t, 0)
                if f:
                    den = f + self.k1 * (1 - self.b + self.b * self.dl[j]
                                         / max(self.avgdl, 1e-9))
                    s[j] += idf * f * (self.k1 + 1) / den
        return s


def _sess_text(session):
    return "\n".join(f"{t.get('role', t.get('speaker', '?'))}: "
                     f"{t.get('content', t.get('text', '')) or ''}"
                     for t in (session or []))


def _turn_texts(session):
    return [f"{t.get('role', t.get('speaker', '?'))}: "
            f"{t.get('content', t.get('text', '')) or ''}"
            for t in (session or [])]


def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd


def _rrf(v, k0=60.0):
    order = np.argsort(-v)
    rr = np.empty_like(v, float)
    for rank, idx in enumerate(order):
        rr[idx] = 1.0 / (k0 + rank + 1.0)
    return rr


def run(out_dir, challenge_only=True, n_boot=4000, sample=0, seed=0,
        model_id=MODEL, qpref="", ppref=""):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(out_dir) / f"lme-maxsim-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("loading LongMemEval-S...", flush=True)
    ds = load_dataset("LIXINYI33/longmemeval-s", split="train")
    exs = []
    for ex in ds:
        if challenge_only and not (len(ex["haystack_session_ids"])
                                   > len(ex["answer_session_ids"])):
            continue
        if not ex["haystack_sessions"] or not ex["answer_session_ids"]:
            continue
        exs.append(ex)
    if sample and sample < len(exs):
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(exs), size=sample, replace=False)
        exs = [exs[k] for k in sorted(pick.tolist())]
    print(f"  {len(exs)} examples (challenge_only={challenge_only}, "
          f"sample={sample})", flush=True)

    print(f"  encoder = {model_id} (qpref={qpref!r} ppref={ppref!r})",
          flush=True)
    model = SentenceTransformer(model_id)
    q_emb = model.encode([qpref + e["question"] for e in exs],
                         convert_to_numpy=True, normalize_embeddings=True,
                         batch_size=64, show_progress_bar=True).astype(np.float32)

    rows = []
    for i, e in enumerate(exs):
        if i % 50 == 0:
            print(f"  scoring {i}/{len(exs)}", flush=True)
        sessions = e["haystack_sessions"]
        sids = e["haystack_session_ids"]
        gold = set(e["answer_session_ids"])
        gold_idx = {j for j, s in enumerate(sids) if s in gold}
        texts = [_sess_text(s) for s in sessions]
        bm = BM25(texts).score(e["question"])
        qv = q_emb[i]
        S = len(sessions)
        dmax = np.full(S, -9.0)
        dmean = np.full(S, -9.0)
        # Embed ALL turns of this example in one batched call, then split
        # back per session (per-session encode() has crippling overhead).
        all_turns, spans = [], []
        for sess in sessions:
            tt = [ppref + t for t in _turn_texts(sess)]
            spans.append((len(all_turns), len(all_turns) + len(tt)))
            all_turns.extend(tt)
        if all_turns:
            temb = model.encode(all_turns, convert_to_numpy=True,
                                normalize_embeddings=True, batch_size=128,
                                show_progress_bar=False)
            sims_all = temb @ qv
            for j, (a, b) in enumerate(spans):
                if b > a:
                    seg = sims_all[a:b]
                    dmax[j] = float(seg.max())
                    dmean[j] = float(seg.mean())
        rows.append({"bm": bm, "dmax": dmax, "dmean": dmean,
                     "gold": gold_idx, "qtype": e.get("question_type", "?")})

    n = len(rows)

    def hit(score, gold):
        return int(int(np.argmax(score)) in gold) if score.size else 0

    def single(key):
        return np.array([hit(r[key], r["gold"]) for r in rows], np.int8)

    bm_h = single("bm")
    dmax_h = single("dmax")
    dmean_h = single("dmean")
    rrf_h = np.array([hit(_rrf(r["bm"]) + _rrf(r["dmax"]), r["gold"])
                      for r in rows], np.int8)

    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)

    def fuse(a, key):
        return np.array([hit(a * _z(r["bm"]) + (1 - a) * _z(r[key]), r["gold"])
                         for r in rows], np.int8)

    # honest alpha via 5-fold CV over questions (independent here)
    def cv_alpha(key, seed=0):
        rng = np.random.default_rng(seed)
        fold = rng.integers(0, 5, size=n)
        ah = {a: fuse(a, key) for a in alphas}
        pred = np.zeros(n, np.int8)
        for f in range(5):
            te = fold == f
            tr = ~te
            ba = max(alphas, key=lambda a: ah[a][tr].mean())
            pred[te] = ah[ba][te]
        ceil_a = max(alphas, key=lambda a: ah[a].mean())
        return pred, float(ah[ceil_a].mean()), float(ceil_a)

    fmax_pred, fmax_ceil, fmax_a = cv_alpha("dmax")

    methods = {"BM25": bm_h, "Dense-sessionmean": dmean_h,
               "Dense-maxsim": dmax_h, "RRF(BM25,maxsim)": rrf_h,
               "Fusion-maxsim(CV)": fmax_pred}
    h1 = {k: float(v.mean()) for k, v in methods.items()}
    for k in methods:
        print(f"  {k:24s} Hit@1={h1[k]:.4f}", flush=True)
    print(f"  [fusion ceiling a={fmax_a} -> {fmax_ceil:.4f}]", flush=True)

    def boot(a, b, seed=1):
        rng = np.random.default_rng(seed)
        d = np.empty(n_boot)
        for bi in range(n_boot):
            idx = rng.integers(0, n, n)
            d[bi] = a[idx].mean() - b[idx].mean()
        return {"diff": float(a.mean() - b.mean()),
                "lo": float(np.quantile(d, 0.025)),
                "hi": float(np.quantile(d, 0.975)),
                "p_one_sided": float(np.mean(d <= 0))}

    comps = {f"Fusion vs {b}": boot(fmax_pred, methods[b])
             for b in ["BM25", "Dense-maxsim", "RRF(BM25,maxsim)"]}
    for name, c in comps.items():
        print(f"  {name:30s} d={c['diff']*100:+.2f}pp "
              f"CI=[{c['lo']*100:+.2f},{c['hi']*100:+.2f}] "
              f"p={c['p_one_sided']:.4f}", flush=True)

    summary = {"ts": ts, "n_eval": n, "challenge_only": challenge_only,
               "n_boot": n_boot, "hit1": h1, "fusion_ceiling_alpha": fmax_a,
               "fusion_ceiling_h1": fmax_ceil, "comparisons": comps}
    (run_dir / "lme.json").write_text(json.dumps(summary, indent=2,
                                      default=str), encoding="utf-8")
    md = ["## LongMemEval-S: turn-level max-sim fusion transfer test", "",
          f"n_eval = {n} (challenge_only={challenge_only}). BGE-base turn "
          f"embeddings; bootstrap over questions (n_boot={n_boot}).", "",
          "| method | Hit@1 |", "|---|---|"]
    for k in methods:
        md.append(f"| {k} | {h1[k]:.4f} |")
    md += ["", f"Fusion full-data ceiling: a={fmax_a} -> {fmax_ceil:.4f}.",
           "", "### Fusion vs baselines", "",
           "| comparison | Δ pp | 95% CI | p |", "|---|---|---|---|"]
    for name, c in comps.items():
        md.append(f"| {name} | {c['diff']*100:+.2f} | "
                  f"[{c['lo']*100:+.2f}, {c['hi']*100:+.2f}] | "
                  f"{c['p_one_sided']:.4f} |")
    md += ["", "Raw: `lme.json`.", ""]
    (run_dir / "lme.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {run_dir/'lme.md'}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--all", action="store_true",
                    help="use all 1000, not just retrieval-challenge 500")
    ap.add_argument("--sample", type=int, default=0,
                    help="randomly subsample this many examples (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--qpref", default="")
    ap.add_argument("--ppref", default="")
    args = ap.parse_args()
    run(Path(args.out), challenge_only=not args.all, n_boot=args.n_boot,
        sample=args.sample, seed=args.seed, model_id=args.model,
        qpref=args.qpref, ppref=args.ppref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
