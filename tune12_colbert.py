"""
tune12_colbert.py — ColBERT token-level late interaction on LoCoMo.

The repo's win comes from late interaction at TURN granularity with one
vector per turn. ColBERT generalizes that to TOKEN granularity:

    score(query, turn) = sum_i  max_j  q_i . d_j          (MaxSim)

over query tokens i and turn tokens j. We keep the turn unit (encode
each turn as a ColBERT document, no truncation) and set the session
score to the max ColBERT score over its turns, then fuse with BM25 and
select the fusion weight by leave-one-conversation-out CV.

Question: does finer-than-turn (token-level) late interaction beat the
single-vector turn fusion (e5-large-v2: Hit@1 0.7518)?

Requires `pylate`. Model id via argv[1] (default a strong modern ColBERT;
falls back to colbert-ir/colbertv2.0).

Output: results/tune12-colbert-<ts>/{tune12.json, tune12.md}
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import locomo  # noqa: E402
from tune7_bm25 import ConvStats, score_classic, _tokenize_text  # noqa: E402

LOCOMO = _HERE / "data" / "locomo" / "locomo10.json"
RES = _HERE / "results"
DEFAULT_MODEL = "lightonai/GTE-ModernColBERT-v1"
FALLBACK_MODEL = "colbert-ir/colbertv2.0"


def _z(v):
    v = np.asarray(v, float)
    sd = v.std()
    return np.zeros_like(v) if sd < 1e-9 else (v - v.mean()) / sd


def _hit(s, g):
    return int(int(np.argmax(s)) in g) if s.size else 0


def maxsim(q_tok, d_tok):
    """ColBERT MaxSim: sum over query tokens of max over doc tokens."""
    if d_tok.shape[0] == 0 or q_tok.shape[0] == 0:
        return -9.0
    return float((q_tok @ d_tok.T).max(axis=1).sum())


def load_colbert(model_id):
    from pylate import models
    try:
        return models.ColBERT(model_name_or_path=model_id), model_id
    except Exception as e:  # noqa: BLE001
        print(f"  {model_id} failed ({e}); falling back to {FALLBACK_MODEL}",
              flush=True)
        return models.ColBERT(model_name_or_path=FALLBACK_MODEL), FALLBACK_MODEL


def run(model_id, n_boot=4000):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = RES / f"tune12-colbert-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(LOCOMO, encoding="utf-8"))
    # turn texts per (ci, si)
    turn_text = {}
    for ci, rec in enumerate(data):
        conv = rec["conversation"]
        for k in conv:
            if k.startswith("session_") and not k.endswith("_date_time"):
                si = int(k.split("_")[1])
                turn_text[(ci, si)] = [
                    f"{t.get('speaker','?')}: {t.get('text','') or ''}"
                    for t in conv[k]]

    exs = [e for e in locomo.iter_examples(LOCOMO) if e["gold_session_ids"]]
    n = len(exs)
    conv_of = np.array([e["conv_idx"] for e in exs])
    convs = sorted(set(conv_of.tolist()))

    print(f"loading ColBERT {model_id}...", flush=True)
    model, used = load_colbert(model_id)

    # Encode all unique turns once.
    keys = sorted(turn_text.keys())
    flat, spans = [], {}
    for key in keys:
        tt = turn_text[key]
        spans[key] = (len(flat), len(flat) + len(tt))
        flat.extend(tt)
    print(f"encoding {len(flat)} turns with ColBERT...", flush=True)
    d_tok = model.encode(flat, is_query=False, batch_size=64,
                        show_progress_bar=True, convert_to_numpy=True)
    # d_tok is a list of (n_tok, dim) arrays
    print("encoding queries...", flush=True)
    q_tok = model.encode([e["question"] for e in exs], is_query=True,
                        batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True)

    print("BM25 + ColBERT scoring...", flush=True)
    cs = {}
    bm_scores, cb_scores, golds = [], [], []
    for i, e in enumerate(exs):
        ci = e["conv_idx"]
        sids = e["session_ids"]
        if ci not in cs:
            cs[ci] = ConvStats([_tokenize_text(s) for s in e["haystack_sessions"]])
        bm_scores.append(score_classic(cs[ci],
                         set(_tokenize_text(e["question"])), 1.5, 0.75))
        qm = q_tok[i]
        sc = np.full(len(sids), -9.0)
        for j, sid in enumerate(sids):
            key = (ci, int(sid[1:]))
            if key in spans:
                a, b = spans[key]
                best = -9.0
                for ti in range(a, b):
                    s = maxsim(qm, d_tok[ti])
                    if s > best:
                        best = s
                sc[j] = best
        cb_scores.append(sc)
        sid2 = {s: j for j, s in enumerate(sids)}
        golds.append({sid2[g] for g in e["gold_session_ids"] if g in sid2})
        if i % 200 == 0:
            print(f"  {i}/{n}", flush=True)

    bm_hits = np.array([_hit(bm_scores[i], golds[i]) for i in range(n)], np.int8)
    cb_hits = np.array([_hit(cb_scores[i], golds[i]) for i in range(n)], np.int8)
    bm_h1, cb_h1 = float(bm_hits.mean()), float(cb_hits.mean())

    alphas = np.round(np.arange(0.0, 1.001, 0.05), 2)
    ah = {}
    for a in alphas:
        ah[a] = np.array([_hit(a * _z(bm_scores[i]) + (1 - a) * _z(cb_scores[i]),
                          golds[i]) for i in range(n)], np.int8)
    ceil_a = max(alphas, key=lambda a: ah[a].mean())
    loco = np.zeros(n, np.int8)
    choices = {}
    for c in convs:
        te = conv_of == c
        tr = ~te
        ba = max(alphas, key=lambda a: ah[a][tr].mean())
        choices[c] = float(ba)
        loco[te] = ah[ba][te]

    idx_by_conv = {c: np.where(conv_of == c)[0] for c in convs}

    def cboot(a, b, seed=1):
        rng = np.random.default_rng(seed)
        cl = np.array(convs)
        d = np.empty(n_boot)
        for bi in range(n_boot):
            pick = rng.choice(cl, size=len(cl), replace=True)
            ia = np.concatenate([idx_by_conv[c] for c in pick])
            d[bi] = a[ia].mean() - b[ia].mean()
        return {"diff": float(a.mean() - b.mean()),
                "lo": float(np.quantile(d, 0.025)),
                "hi": float(np.quantile(d, 0.975)),
                "p_one_sided": float(np.mean(d <= 0))}

    boot = cboot(loco, bm_hits)
    res = {"ts": ts, "model": used, "n_eval": n, "n_boot": n_boot,
           "bm25_h1": bm_h1, "colbert_maxsim_h1": cb_h1,
           "fusion_loco_h1": float(loco.mean()),
           "fusion_ceiling_h1": float(ah[ceil_a].mean()),
           "ceiling_alpha": float(ceil_a),
           "fusion_vs_bm25": boot, "loco_alpha_choices": choices}
    (run_dir / "tune12.json").write_text(json.dumps(res, indent=2, default=str),
                                         encoding="utf-8")
    print(f"\n  BM25={bm_h1:.4f}  ColBERT-maxsim={cb_h1:.4f}  "
          f"fusion(LOCO)={loco.mean():.4f}  d_vs_bm25={boot['diff']*100:+.2f}pp "
          f"p={boot['p_one_sided']:.4f}", flush=True)
    md = ["## tune12: ColBERT token-level late interaction (LoCoMo)", "",
          f"model = `{used}`. n_eval = {n}. Turn-level ColBERT MaxSim; "
          f"session score = max over turns; BM25 fusion weight by LOCO-CV; "
          f"conversation-cluster bootstrap (n_boot={n_boot}).", "",
          "| method | Hit@1 |", "|---|---|",
          f"| BM25 | {bm_h1:.4f} |",
          f"| ColBERT MaxSim (turn) alone | {cb_h1:.4f} |",
          f"| **Fusion BM25 ⊕ ColBERT (LOCO)** | **{loco.mean():.4f}** |", "",
          f"Fusion vs BM25: Δ = {boot['diff']*100:+.2f} pp, 95% CI "
          f"[{boot['lo']*100:+.2f}, {boot['hi']*100:+.2f}], "
          f"p = {boot['p_one_sided']:.4f}. Ceiling alpha={ceil_a} -> "
          f"{ah[ceil_a].mean():.4f}.", "",
          "Compare: e5-large-v2 single-vector turn fusion = 0.7518.", "",
          "Raw: `tune12.json`.", ""]
    (run_dir / "tune12.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {run_dir/'tune12.md'}", flush=True)
    return res


if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    run(mid)
