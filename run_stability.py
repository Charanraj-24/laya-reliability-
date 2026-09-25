"""Stability experiment: each case asked several ways that should not change the answer.

Variants: original order, reversed, seeded random orders, and options renamed
to A, B, C... in several orders. Each variant is a separate predict call (the
packed single-call version is in reliability.py). Compares stability with
Laya's confidence fields at separating right from wrong answers.

    python run_stability.py [--tests laya_tests.json] [--only T] [--shuffles 3]
"""
import argparse
import csv
import json
import random
import string
import time
import zlib
from datetime import datetime
from pathlib import Path

import laya
from run_benchmark import auroc, build_question, fmt

HERE = Path(__file__).resolve().parent


def make_variants(criteria, case_id, n_shuffles):
    """Return a list of (name, order, relabel) variants, duplicates removed.

    order   = list of original labels in the order they are shown
    relabel = True -> keys become A, B, C... (descriptions unchanged)
    """
    labels = list(criteria)
    rng = random.Random(zlib.crc32(case_id.encode()))  # same shuffles every run

    candidates = [("original", labels, False), ("reversed", labels[::-1], False)]
    for i in range(n_shuffles):
        order = labels[:]
        rng.shuffle(order)
        candidates.append((f"shuffle{i + 1}", order, False))
    candidates.append(("letters", labels, True))
    candidates.append(("letters_reversed", labels[::-1], True))
    order = labels[:]
    rng.shuffle(order)
    candidates.append(("letters_shuffle", order, True))

    # Laya is deterministic, so a repeated variant would always agree with
    # itself and inflate stability. Keep unique ones only.
    seen, unique = set(), []
    for name, order, relabel in candidates:
        key = (tuple(order), relabel)
        if key not in seen:
            seen.add(key)
            unique.append((name, order, relabel))
    return unique


def build_variant_question(base_q, order, relabel):
    """Rebuild the question with options in `order`, optionally renamed.
    Returns the question and a map from shown label -> original label."""
    criteria = base_q["criteria"]
    if relabel:
        shown = list(string.ascii_uppercase[: len(order)])
    else:
        shown = order
    new_criteria = {s: criteria[orig] for s, orig in zip(shown, order)}
    back = dict(zip(shown, order))
    q = {"type": "choice", "instructions": base_q["instructions"], "criteria": new_criteria}
    return {"q": q}, back


def run(test_file, model, only, n_shuffles):
    data = json.loads(Path(test_file).read_text(encoding="utf-8"))
    cases = data["cases"]
    if only:
        cases = [c for c in cases if c["id"].startswith(only.upper())]

    print(f"Loading {model} ...")
    agent = laya.load(model)

    rows, start = [], time.time()
    for n, case in enumerate(cases, 1):
        base_q = build_question(case, data["schemas"])["q"]
        variants = make_variants(base_q["criteria"], case["id"], n_shuffles)
        state = {"message": case["statement"]}

        answers = []  # one dict per variant, answers mapped back to original labels
        for name, order, relabel in variants:
            questions, back = build_variant_question(base_q, order, relabel)
            try:
                ans = agent.predict(state, questions)["answers"]["q"]
            except Exception as e:
                print(f"  {case['id']} {name}: error {e!r}")
                continue
            probs = {back[k]: v for k, v in ans.get("probabilities", {}).items()}
            answers.append({
                "variant": name,
                "choice": back.get(ans["choice"], ans["choice"]),
                "confidence": ans.get("confidence"),
                "answer_confidence": ans.get("answer_confidence"),
                "probs": probs,
            })

        if not answers or answers[0]["variant"] != "original":
            print(f"  {case['id']}: original variant failed, skipping")
            continue

        orig = answers[0]
        others = answers[1:]
        top_choice = orig["choice"]
        ranked = sorted(orig["probs"].values(), reverse=True)

        stability = (sum(a["choice"] == top_choice for a in others) / len(others)) if others else 1.0
        # soft: mean probability of the original answer across all variants
        soft = sum(a["probs"].get(top_choice, 0.0) for a in answers) / len(answers)
        distinct = sorted({a["choice"] for a in answers})

        row = {
            "id": case["id"],
            "category": case["category"],
            "statement": case["statement"],
            "expected": " | ".join(case["expected"]),
            "ambiguous": case["ambiguous"],
            "predicted": top_choice,
            "correct": None if case["ambiguous"] else top_choice in case["expected"],
            "confidence": orig["confidence"],
            "answer_confidence": orig["answer_confidence"],
            "margin": round(ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0), 4) if ranked else None,
            "num_variants": len(answers),
            "stability": round(stability, 4),
            "soft_stability": round(soft, 4),
            "num_distinct_answers": len(distinct),
            "distinct_answers": " | ".join(distinct),
            "variant_answers": json.dumps({a["variant"]: a["choice"] for a in answers}),
        }
        # Fixed before the first run; not tuned on any results.
        if row["answer_confidence"] is not None:
            row["combined"] = round(row["answer_confidence"] * row["soft_stability"], 4)
        rows.append(row)

        mark = {True: "OK   ", False: "WRONG", None: "  -  "}[row["correct"]]
        flip = "" if stability == 1.0 else f"  FLIPS -> {row['distinct_answers']}"
        print(f"[{n:3}/{len(cases)}] {mark} {case['id']:4} pred={top_choice:16} "
              f"conf={fmt(row['confidence'], 2)} stab={stability:.2f}{flip}")

    print(f"\nFinished in {time.time() - start:.1f} s")
    return rows


def accept_rate_table(scored, signal, add):
    """Accept the top X% of cases by `signal`, report accuracy of what was accepted."""
    ordered = sorted(scored, key=lambda r: r[signal], reverse=True)
    parts = []
    for cov in (0.5, 0.7, 0.8, 0.9, 1.0):
        k = max(1, round(len(ordered) * cov))
        kept = ordered[:k]
        parts.append(f"{int(cov * 100)}%: {sum(r['correct'] for r in kept) / k:.1%}")
    add(f"  {signal:18} " + "   ".join(parts))


def summarize(rows):
    lines = []
    add = lines.append
    scored = [r for r in rows if r["correct"] is not None]
    labels = [1 if r["correct"] else 0 for r in scored]

    add("=" * 70)
    add("STABILITY SUMMARY")
    add("=" * 70)
    if scored:
        add(f"Accuracy on original prompts: {sum(labels) / len(labels):.1%}  ({len(scored)} cases)")

    add("\nAUROC (predicting right vs wrong; 0.5 = useless, 1.0 = perfect):")
    signals = ["confidence", "answer_confidence", "margin", "stability", "soft_stability", "combined"]
    for s in signals:
        vals = [r.get(s) for r in scored]
        if vals and all(v is not None for v in vals):
            add(f"  {s:18} {fmt(auroc(vals, labels))}")

    add("\nAccuracy of the top X% of cases, ranked by each signal:")
    for s in signals:
        if scored and all(r.get(s) is not None for r in scored):
            accept_rate_table(scored, s, add)

    add("\nWrong answers with confidence >= 0.70, and whether stability flagged them:")
    conf_wrong = [r for r in scored if not r["correct"] and (r["confidence"] or 0) >= 0.7]
    caught = [r for r in conf_wrong if r["stability"] < 1.0]
    add(f"  Wrong with confidence >= 0.70: {len(conf_wrong)}   of those, unstable: {len(caught)}")
    for r in conf_wrong:
        tag = "CAUGHT " if r["stability"] < 1.0 else "missed "
        add(f"  {tag}{r['id']:4} conf={r['confidence']:.2f} stab={r['stability']:.2f} "
            f"answers={r['distinct_answers']:28} {r['statement'][:35]}")

    right = [r for r in scored if r["correct"]]
    false_alarms = [r for r in right if r["stability"] < 1.0]
    add(f"\n  False alarms (right answer, but unstable): {len(false_alarms)} of {len(right)}")
    for r in false_alarms:
        add(f"    {r['id']:4} conf={r['confidence']:.2f} stab={r['stability']:.2f} "
            f"answers={r['distinct_answers']:28} {r['statement'][:35]}")

    wrong_low = [r for r in scored if not r["correct"] and (r["confidence"] or 0) < 0.7]
    if wrong_low:
        unstable = sum(r["stability"] < 1.0 for r in wrong_low)
        add(f"\n  Wrong with confidence < 0.70 (confidence already flags these): {len(wrong_low)}, unstable: {unstable}")

    amb = [r for r in rows if r["ambiguous"]]
    if amb:
        add("\nAmbiguous cases (want LOW stability or LOW confidence):")
        for r in amb:
            add(f"  {r['id']:4} conf={r['confidence']:.2f} stab={r['stability']:.2f} "
                f"answers={r['distinct_answers']:28} {r['statement'][:35]}")

    add("\nNote: with ~100 cases, differences of a few AUROC points can be noise.")
    return "\n".join(lines)


def save(rows, summary, out_dir):
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"stability_{stamp}.csv"
    json_path = out_dir / f"stability_{stamp}.json"
    txt_path = out_dir / f"stability_{stamp}_summary.txt"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(summary, encoding="utf-8")
    print(f"\nSaved:\n  {csv_path}\n  {json_path}\n  {txt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=str(HERE / "laya_tests.json"))
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--only", default="", help="run only IDs starting with this letter, e.g. T")
    ap.add_argument("--shuffles", type=int, default=3, help="random option orders per case")
    args = ap.parse_args()

    rows = run(args.tests, args.model, args.only, args.shuffles)
    summary = summarize(rows)
    print("\n" + summary)
    save(rows, summary, HERE / "results")
