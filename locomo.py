"""
locomo.py — LoCoMo (Long Conversational Memory) adapter.

Source: Percena/locomo-mc10/raw/locomo10.json on HuggingFace (raw release of
Maharana et al. 2024, "Evaluating Very Long-Term Conversational Memory of
LLM Agents").

10 conversations, 272 sessions total, 1,986 QA pairs across 5 question
categories. The integer codes (verified from the data: cat 2 = "When did..."
temporal; cat 3 = commonsense/world-knowledge; cat 5 carries an
`adversarial_answer` field; cat 1 evidence spans ~2.7 sessions vs cat 4's 1.0)
are: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial.
Each turn has a dia_id like "D1:3" (session 1, turn 3); a question's evidence
is a list of dia_ids spanning one or more sessions.

We expose a session-level retrieval task analogous to LongMemEval:
  - haystack = all sessions in the conversation (string id: "D1", "D2", ...)
  - gold session ids = unique session prefixes from the evidence list
  - hit@1 = top-1 retrieved session intersects the gold session set
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOCOMO_JSON = (_REPO_ROOT / "research" / "opsmem"
                                  / "data" / "locomo" / "locomo10.json")


_DIA_RE = re.compile(r"^D(\d+):(\d+)$")


def _session_text_from_turns(turns: list[dict]) -> str:
    """Format a session (list of {speaker, dia_id, text}) as a single string."""
    parts = []
    for t in turns:
        sp = t.get("speaker", "?")
        tx = t.get("text", "") or ""
        parts.append(f"{sp}: {tx}")
    return "\n".join(parts)


def _conversation_to_haystack(conv: dict) -> tuple[list[str], list[list[dict]]]:
    """Return (session_ids, session_turns_per_session) for one conversation."""
    sess_keys = [k for k in conv.keys()
                  if k.startswith("session_")
                  and not k.endswith("_date_time")]
    # Sort by numeric suffix so D1, D2, ..., D19 come in order.
    sess_keys = sorted(sess_keys, key=lambda k: int(k.split("_")[1]))
    session_ids = []
    session_turns = []
    for k in sess_keys:
        idx = int(k.split("_")[1])
        session_ids.append(f"D{idx}")
        session_turns.append(conv[k])
    return session_ids, session_turns


def _parse_gold_sessions(evidence) -> set[str]:
    """Extract unique session ids from an evidence list like ['D1:3','D2:5']."""
    gold = set()
    if not isinstance(evidence, list):
        return gold
    for ev in evidence:
        if not isinstance(ev, str):
            continue
        m = _DIA_RE.match(ev.strip())
        if m:
            gold.add(f"D{int(m.group(1))}")
    return gold


def iter_examples(json_path: Path = DEFAULT_LOCOMO_JSON
                    ) -> Iterator[dict]:
    """Yield retrieval examples one per QA pair.

    Each example: {
        "conv_idx":     index into the LoCoMo conversation list (0..9)
        "sample_id":    conversation's sample_id field
        "speakers":     (speaker_a, speaker_b)
        "question":     the question text
        "answer":       the gold answer text (not used by retrievers)
        "category":     1..5 (LoCoMo question type)
        "session_ids":  ["D1", "D2", ...] in conversation order
        "haystack_sessions": list of session strings (same len as session_ids)
        "gold_session_ids": set of session ids the evidence points to
    }
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    for ci, conv_record in enumerate(data):
        conv = conv_record["conversation"]
        session_ids, session_turns = _conversation_to_haystack(conv)
        haystack_text = [_session_text_from_turns(t) for t in session_turns]
        speakers = (conv.get("speaker_a"), conv.get("speaker_b"))
        for qa in conv_record["qa"]:
            gold = _parse_gold_sessions(qa.get("evidence", []))
            yield {
                "conv_idx":           ci,
                "sample_id":          conv_record.get("sample_id"),
                "speakers":           speakers,
                "question":           qa["question"],
                "answer":             qa.get("answer"),
                "category":           qa.get("category"),
                "session_ids":        session_ids,
                "haystack_sessions":  haystack_text,
                "gold_session_ids":   gold,
            }


def load_all(json_path: Path = DEFAULT_LOCOMO_JSON) -> list[dict]:
    """List form, easier for downstream batching."""
    return list(iter_examples(json_path))


def smoke() -> None:
    examples = load_all()
    n = len(examples)
    print(f"loaded {n} LoCoMo retrieval examples")
    if n:
        ex = examples[0]
        print(f"first example:")
        print(f"  conv_idx     = {ex['conv_idx']}")
        print(f"  speakers     = {ex['speakers']}")
        print(f"  question     = {ex['question']!r}")
        print(f"  category     = {ex['category']}")
        print(f"  n_sessions   = {len(ex['session_ids'])}")
        print(f"  session_ids  = {ex['session_ids'][:8]}{'...' if len(ex['session_ids'])>8 else ''}")
        print(f"  gold         = {ex['gold_session_ids']}")
        print(f"  first session preview: {ex['haystack_sessions'][0][:200]!r}")
    # Stats
    from collections import Counter
    cats = Counter(ex["category"] for ex in examples)
    print(f"  category dist: {dict(cats)}")
    no_gold = sum(1 for ex in examples if not ex["gold_session_ids"])
    print(f"  questions with empty gold: {no_gold} ({no_gold/max(n,1)*100:.1f}%)")
    # haystack > 1
    multi = sum(1 for ex in examples if len(ex["session_ids"]) > 1)
    print(f"  multi-session haystacks: {multi}")


if __name__ == "__main__":
    smoke()
