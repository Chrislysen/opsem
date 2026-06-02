"""
embed_turns.py — Cache TURN-level BGE-base embeddings for LoCoMo.

Session-level dense retrieval is catastrophic on LoCoMo (tune8: Hit@1
0.349) because embedding a whole multi-turn session into one vector
dilutes the answer turn. A question matches a specific turn. So we embed
each turn and let the session score be a late-interaction aggregate
(max-sim) over its turns.

Output: results/locomo_turn_cache.npz
  turns_<ci>_<si> : (T, d) turn embeddings for session si of conv ci
  shapes (json)   : turn counts per (ci, si)
Query embeddings are reused from results/locomo_bge_cache.npz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import locomo  # noqa: E402

DEFAULT_LOCOMO = _HERE / "data" / "locomo" / "locomo10.json"
CACHE = _HERE / "results" / "locomo_turn_cache.npz"
MODEL = "BAAI/bge-base-en-v1.5"


def main() -> int:
    from sentence_transformers import SentenceTransformer

    data = json.load(open(DEFAULT_LOCOMO, encoding="utf-8"))
    model = SentenceTransformer(MODEL)
    out: dict[str, np.ndarray] = {}
    shapes: dict[str, int] = {}

    for ci, rec in enumerate(data):
        conv = rec["conversation"]
        sess_keys = sorted([k for k in conv if k.startswith("session_")
                            and not k.endswith("_date_time")],
                           key=lambda k: int(k.split("_")[1]))
        for k in sess_keys:
            si = int(k.split("_")[1])
            turns = conv[k]
            texts = [f"{t.get('speaker','?')}: {t.get('text','') or ''}"
                     for t in turns]
            if not texts:
                continue
            emb = model.encode(texts, convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False, batch_size=32)
            out[f"turns_{ci}_{si}"] = emb.astype(np.float32)
            shapes[f"{ci}_{si}"] = emb.shape[0]
        print(f"  conv {ci}: {len(sess_keys)} sessions", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, shapes=json.dumps(shapes), **out)
    print(f"wrote {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB), "
          f"{len(out)} sessions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
