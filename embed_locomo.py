"""
embed_locomo.py — Cache BGE-base embeddings for LoCoMo once.

Sessions are shared within a conversation, so we embed each conversation's
sessions once and every query once. Output: results/locomo_bge_cache.npz
with:
  q_emb[i]       : query embedding for example i        (n, d)
  sess_emb_<ci>  : session embeddings for conversation ci (S_ci, d)
  meta (json)    : conv_idx per example, session counts

Normalised embeddings (cosine == dot). Runs on CPU.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import locomo  # noqa: E402

DEFAULT_LOCOMO = _HERE / "data" / "locomo" / "locomo10.json"
CACHE = _HERE / "results" / "locomo_bge_cache.npz"
MODEL = "BAAI/bge-base-en-v1.5"


def main() -> int:
    from sentence_transformers import SentenceTransformer

    examples = [e for e in locomo.iter_examples(DEFAULT_LOCOMO)
                if e["gold_session_ids"]]
    print(f"{len(examples)} examples", flush=True)

    by_conv: dict[int, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        by_conv[e["conv_idx"]].append(i)

    model = SentenceTransformer(MODEL)
    out: dict[str, np.ndarray] = {}

    # Session embeddings, once per conversation.
    for ci, idxs in sorted(by_conv.items()):
        sessions = examples[idxs[0]]["haystack_sessions"]
        emb = model.encode(sessions, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False,
                           batch_size=16)
        out[f"sess_emb_{ci}"] = emb.astype(np.float32)
        print(f"  conv {ci}: {emb.shape[0]} sessions embedded", flush=True)

    # Query embeddings, all at once.
    queries = [e["question"] for e in examples]
    q_emb = model.encode(queries, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True,
                         batch_size=32)
    out["q_emb"] = q_emb.astype(np.float32)

    meta = {"conv_idx": [e["conv_idx"] for e in examples],
            "model": MODEL, "n": len(examples)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, meta=json.dumps(meta), **out)
    print(f"wrote {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
