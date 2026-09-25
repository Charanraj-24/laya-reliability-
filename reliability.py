"""Metamorphic reliability layer for Laya `choice` questions.

Laya runs unchanged; each question is also asked under transformations that
should preserve the correct answer (options reordered, options renamed to
A/B/C). An answer that changes under them is fragile. All variants go into the
same predict call: Laya encodes each question as its own row, so they are
scored independently in one forward pass.

    agent = ReliableAgent(laya.load("convaiinnovations/laya"))
    rel = agent.predict(state, questions)["answers"]["tone"]["reliability"]
    rel["decision"]        # "ACCEPT" | "VERIFY" | "ESCALATE"
    rel["soft_stability"]  # 0..1, higher = more trustworthy

`score` and `noul` questions pass through without a "reliability" entry.
"""
from __future__ import annotations

import random
import string
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ReliableAgent",
    "predict_reliable",
    "predict_reliable_batch",
    "make_variants",
    "decide",
    "calibrate",
    "DEFAULT_ACCEPT",
    "DEFAULT_ESCALATE",
    "VARIANT_PRESETS",
    "ALL_VARIANTS",
]

# Output of calibrate() on the 103-case development set ("full" variants).
# They do not transfer well between datasets; see README.
DEFAULT_ACCEPT = 0.77
DEFAULT_ESCALATE = 0.61

_SEP = "::rel"  # suffix for packed variant keys

# "full" is what the reported results use. "fast" is cheaper (3 rows per
# question instead of 8) and measurably weaker; see README.
VARIANT_PRESETS = {
    "full": None,  # every variant
    "fast": ["reversed", "letters_reversed"],
}
ALL_VARIANTS = ["reversed", "shuffle1", "shuffle2", "shuffle3",
                "letters", "letters_reversed", "letters_shuffle"]


Variant = Tuple[str, List[str], bool]  # (name, shown order of original labels, relabel?)


def make_variants(labels: Sequence[str], seed_key: str, n_shuffles: int = 3,
                  relabel: bool = True, include: Optional[Sequence[str]] = None) -> List[Variant]:
    """Meaning-preserving rewrites of a choice question, duplicates removed.

    The first variant is always the original question. Shuffles are seeded from
    `seed_key` so the same input always gets the same variants. Duplicates are
    dropped because Laya is deterministic: repeating an identical prompt would
    make an answer look more stable than it is.

    `include` keeps only the named variants (the original is always kept).
    """
    labels = list(labels)
    rng = random.Random(zlib.crc32(seed_key.encode("utf-8")))
    cands: List[Variant] = [("original", labels, False), ("reversed", labels[::-1], False)]
    for i in range(n_shuffles):
        order = labels[:]
        rng.shuffle(order)
        cands.append((f"shuffle{i + 1}", order, False))
    if relabel and len(labels) <= 26:
        cands.append(("letters", labels, True))
        cands.append(("letters_reversed", labels[::-1], True))
        order = labels[:]
        rng.shuffle(order)
        cands.append(("letters_shuffle", order, True))

    if include is not None:
        wanted = set(include)
        cands = [c for c in cands if c[0] == "original" or c[0] in wanted]

    seen, out = set(), []
    for name, order, rel in cands:
        key = (tuple(order), rel)
        if key not in seen:
            seen.add(key)
            out.append((name, order, rel))
    return out


def _variant_question(q: Dict[str, Any], order: List[str], relabel: bool):
    """Rebuild question `q` with options in `order`, optionally renamed A, B, C...
    Returns (question, map from shown label -> original label)."""
    shown = list(string.ascii_uppercase[: len(order)]) if relabel else list(order)
    new_q = dict(q)  # keeps instructions and any other fields
    new_q["criteria"] = {s: q["criteria"][orig] for s, orig in zip(shown, order)}
    return new_q, dict(zip(shown, order))


def decide(soft_stability: float, accept: float = DEFAULT_ACCEPT,
           escalate: float = DEFAULT_ESCALATE) -> str:
    """ACCEPT (trust it), VERIFY (double-check) or ESCALATE (send to a human or bigger model)."""
    if soft_stability >= accept:
        return "ACCEPT"
    if soft_stability < escalate:
        return "ESCALATE"
    return "VERIFY"


def calibrate(soft_scores: Sequence[float], correct: Sequence[bool],
              target_accuracy: float = 0.90, max_escalate_accuracy: float = 1 / 3,
              ) -> Tuple[float, float]:
    """Pick (accept, escalate) thresholds from your own labelled examples.

    accept   = the lowest cutoff where answers at or above it reach `target_accuracy`.
    escalate = the highest cutoff BELOW `accept` where answers below it are at
               most `max_escalate_accuracy` accurate (default 1/3: escalated
               answers are wrong at least 2 times out of 3). Keeping it below
               `accept` guarantees a VERIFY band between the two.
    Calibrate on one set of examples and check on a different one.
    If no cutoff reaches `target_accuracy`, accept is returned as 1.01 (accept nothing).
    """
    pairs = sorted(zip(soft_scores, correct))
    if not pairs:
        return DEFAULT_ACCEPT, DEFAULT_ESCALATE
    cuts = sorted({round(s, 4) for s, _ in pairs})

    accept = 1.01  # accept nothing if the target is never reached
    for t in cuts:
        kept = [c for s, c in pairs if s >= t]
        if kept and sum(kept) / len(kept) >= target_accuracy:
            accept = t
            break

    escalate = 0.0
    for t in cuts:
        if t >= accept:
            break
        below = [c for s, c in pairs if s < t]
        if below and sum(below) / len(below) <= max_escalate_accuracy:
            escalate = t
    return accept, escalate


def _resolve_variants(variants) -> Optional[List[str]]:
    """"full" / "fast" / a list of variant names -> list of names (None = all)."""
    if variants is None:
        return None
    if isinstance(variants, str):
        if variants not in VARIANT_PRESETS:
            raise ValueError(f"unknown variants preset {variants!r}; use one of {sorted(VARIANT_PRESETS)} "
                             f"or a list of names from {ALL_VARIANTS}")
        return VARIANT_PRESETS[variants]
    names = list(variants)
    bad = [n for n in names if n not in ALL_VARIANTS]
    if bad:
        raise ValueError(f"unknown variant names {bad}; choose from {ALL_VARIANTS}")
    if not names:
        raise ValueError("variants list is empty: nothing to compare the original against")
    return names

def _pack(questions: Dict[str, Dict[str, Any]], n_shuffles: int, relabel: bool, seed: str,
          include: Optional[List[str]] = None):
    """Build one questions dict holding the original questions plus every variant."""
    packed: Dict[str, Dict[str, Any]] = {}
    plan: Dict[str, List[Tuple[str, str, Dict[str, str]]]] = {}
    for qid, q in questions.items():
        if _SEP in qid:
            raise ValueError(f"question id {qid!r} must not contain {_SEP!r}")
        packed[qid] = q  # the untouched original: its answer is exactly what predict() returns
        if q.get("type") != "choice" or len(q.get("criteria", {})) < 2:
            continue
        variants = make_variants(list(q["criteria"]), f"{seed}|{qid}", n_shuffles, relabel, include)
        plan[qid] = [("original", qid, {k: k for k in q["criteria"]})]
        for i, (name, order, rel) in enumerate(variants[1:], 1):
            vq, back = _variant_question(q, order, rel)
            key = f"{qid}{_SEP}{i}"
            packed[key] = vq
            plan[qid].append((name, key, back))
    return packed, plan


def _unpack(raw: Dict[str, Any], questions, plan, accept, escalate) -> Dict[str, Any]:
    """Turn a packed result back into the caller's questions, adding `reliability`."""
    answers_in = raw.get("answers", {})
    answers_out = {}
    for qid in questions:
        ans = dict(answers_in[qid])
        if qid in plan:
            choice = ans["choice"]
            seen = []
            for name, key, back in plan[qid]:
                a = answers_in[key]
                probs = {back[k]: v for k, v in a.get("probabilities", {}).items()}
                seen.append((name, back.get(a["choice"], a["choice"]), probs))
            others = seen[1:]
            stability = (sum(c == choice for _, c, _ in others) / len(others)) if others else 1.0
            soft = sum(p.get(choice, 0.0) for _, _, p in seen) / len(seen)
            ans["reliability"] = {
                "decision": decide(soft, accept, escalate),
                "soft_stability": round(soft, 4),
                "stability": round(stability, 4),
                "n_variants": len(seen),
                "distinct_choices": sorted({c for _, c, _ in seen}),
                "variant_choices": {name: c for name, c, _ in seen},
            }
        answers_out[qid] = ans
    out = {k: v for k, v in raw.items() if k != "answers"}
    out["answers"] = answers_out
    return out


def _seed_of(state: Any) -> str:
    return state if isinstance(state, str) else repr(state)


def predict_reliable(agent, state, questions: Dict[str, Dict[str, Any]], *,
                     n_shuffles: int = 3, relabel: bool = True, variants="full",
                     accept: float = DEFAULT_ACCEPT, escalate: float = DEFAULT_ESCALATE,
                     **predict_kwargs) -> Dict[str, Any]:
    """Like `agent.predict(state, questions)`, plus a `reliability` entry on each
    choice answer. Uses a single predict call."""
    packed, plan = _pack(questions, n_shuffles, relabel, _seed_of(state), _resolve_variants(variants))
    raw = agent.predict(state, packed, **predict_kwargs)
    return _unpack(raw, questions, plan, accept, escalate)


def predict_reliable_batch(agent, states: List[Any], questions: Dict[str, Dict[str, Any]], *,
                           n_shuffles: int = 3, relabel: bool = True, variants="full",
                           accept: float = DEFAULT_ACCEPT, escalate: float = DEFAULT_ESCALATE,
                           **batch_kwargs) -> List[Dict[str, Any]]:
    """Batched version via `agent.predict_batch`. Variants are seeded per question
    (not per state) here, since predict_batch shares one questions dict across states."""
    packed, plan = _pack(questions, n_shuffles, relabel, "batch", _resolve_variants(variants))
    raws = agent.predict_batch(states, packed, **batch_kwargs)
    return [_unpack(r, questions, plan, accept, escalate) for r in raws]


class ReliableAgent:
    """Wraps a loaded Laya agent. `predict` / `predict_batch` return Laya's normal
    output plus `reliability` on every choice answer. Other attributes pass through."""

    def __init__(self, agent, *, n_shuffles: int = 3, relabel: bool = True, variants="full",
                 accept: float = DEFAULT_ACCEPT, escalate: float = DEFAULT_ESCALATE):
        _resolve_variants(variants)  # fail early on a bad value
        self.agent = agent
        self.opts = dict(n_shuffles=n_shuffles, relabel=relabel, variants=variants,
                         accept=accept, escalate=escalate)

    def predict(self, state, questions, **kwargs):
        return predict_reliable(self.agent, state, questions, **self.opts, **kwargs)

    def predict_batch(self, states, questions, **kwargs):
        return predict_reliable_batch(self.agent, states, questions, **self.opts, **kwargs)

    def __getattr__(self, name):
        return getattr(self.agent, name)


if __name__ == "__main__":
    import json
    import laya

    tone = {
        "tone": {
            "type": "choice",
            "instructions": "What is the speaker expressing?",
            "criteria": {
                "indifferent": "The speaker says they do not care or have no preference.",
                "grateful": "The speaker thanks someone or shows appreciation.",
                "uncertain": "The speaker says they do not know or are unsure.",
                "apologetic": "The speaker apologizes or admits a mistake.",
                "agreeing": "The speaker agrees to or accepts a plan.",
                "other": "None of the above.",
            },
        }
    }
    agent = ReliableAgent(laya.load("convaiinnovations/laya"))
    for msg in ["Thank you so much!", "Either option works, I have no preference.",
                "I couldn't agree less.", "Gee, thanks for the help, genius."]:
        a = agent.predict(msg, tone)["answers"]["tone"]
        r = a["reliability"]
        print(f"{msg:45} -> {a['choice']:12} conf={a['answer_confidence']:.2f} "
              f"soft={r['soft_stability']:.2f} {r['decision']:8} {r['distinct_choices']}")
