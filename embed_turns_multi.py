"""
embed_turns_multi.py — Cache turn + query embeddings for ANY encoder.

Used to show the turn-level max-sim granularity win is not specific to
BGE. Handles per-model prefixes (e5 needs "query:"/"passage:").

Usage: python embed_turns_multi.py <hf_model_id> [query_prefix] [passage_prefix]
Writes results/turns__<slug>.npz (keys turns_<ci>_<si>) and
results/q__<slug>.npy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import locomo  # noqa: E402

LOCOMO = _HERE / "data" / "locomo" / "locomo10.json"


def slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "-")


def main() -> int:
    from sentence_transformers import SentenceTransformer

    model_id = sys.argv[1]
    qpref = sys.argv[2] if len(sys.argv) > 2 else ""
    ppref = sys.argv[3] if len(sys.argv) > 3 else ""
    sl = slug(model_id)
    data = json.load(open(LOCOMO, encoding="utf-8"))
    try:
        m = SentenceTransformer(model_id, trust_remote_code=True)
    except Exception:
        m = SentenceTransformer(model_id)

    out = {}
    for ci, rec in enumerate(data):
        conv = rec["conversation"]
        for k in conv:
            if k.startswith("session_") and not k.endswith("_date_time"):
                si = int(k.split("_")[1])
                texts = [ppref + f"{t.get('speaker','?')}: {t.get('text','') or ''}"
                         for t in conv[k]]
                if not texts:
                    continue
                emb = m.encode(texts, convert_to_numpy=True,
                              normalize_embeddings=True, batch_size=64,
                              show_progress_bar=False)
                out[f"turns_{ci}_{si}"] = emb.astype(np.float32)
        print(f"  conv {ci} done", flush=True)
    np.savez_compressed(_HERE / "results" / f"turns__{sl}.npz", **out)

    exs = [e for e in locomo.iter_examples(LOCOMO) if e["gold_session_ids"]]
    q = m.encode([qpref + e["question"] for e in exs], convert_to_numpy=True,
                normalize_embeddings=True, batch_size=64,
                show_progress_bar=False).astype(np.float32)
    np.save(_HERE / "results" / f"q__{sl}.npy", q)
    print(f"wrote turns__{sl}.npz ({len(out)} sessions) + q__{sl}.npy "
          f"({len(q)} queries)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
