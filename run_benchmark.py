"""Baseline run: each test case through Laya once.

Records the answer, both confidence fields and the probability distribution,
prints accuracy and AUROC per signal, and writes CSV, JSON and a text summary
to results/.

    python run_benchmark.py [--tests laya_tests.json] [--only H]
"""
import argparse
import csv
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import laya

HERE = Path(__file__).resolve().parent


def build_question(case, schemas):
    """Turn one test case into the questions dict Laya expects."""
    if case["schema"] == "Q3":
        instructions = case["question"]
        criteria = {opt: f"The answer is {opt}." for opt in case["options"]}
    else:
        schema = schemas[case["schema"]]
        instructions = schema["instructions"]
        criteria = dict(schema["criteria"])
    return {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}


def entropy(probs):
    """Normalized entropy: 0 = all probability on one option, 1 = spread evenly."""
    p = [x for x in probs if x > 0]
    if len(probs) < 2:
        return 0.0
    h = -sum(x * math.log(x) for x in p)
    return h / math.log(len(probs))


def auroc(scores, labels):
    """How well a score separates right (1) from wrong (0) answers.
    1.0 = perfect, 0.5 = no better than a coin flip. Ties count as half."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def fmt(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}f}"


def run(test_file, model, only):
    data = json.loads(Path(test_file).read_text(encoding="utf-8"))
    cases = data["cases"]
    if only:
        cases = [c for c in cases if c["id"].startswith(only.upper())]
    print(f"Loaded {len(cases)} cases from {test_file}")

    print(f"Loading {model} ...")
    agent = laya.load(model)

    rows = []
    start = time.time()
    for n, case in enumerate(cases, 1):
        questions = build_question(case, data["schemas"])
        row = {
            "id": case["id"],
            "category": case["category"],
            "statement": case["statement"],
            "question": questions["q"]["instructions"],
            "expected": " | ".join(case["expected"]),
            "ambiguous": case["ambiguous"],
            "pair": case["pair"],
            "note": case["note"],
        }
        try:
            t0 = time.time()
            result = agent.predict({"message": case["statement"]}, questions)
            latency_ms = (time.time() - t0) * 1000
            ans = result["answers"]["q"]
            probs = ans.get("probabilities", {})
            ranked = sorted(probs.values(), reverse=True)
            top = ranked[0] if ranked else None
            second = ranked[1] if len(ranked) > 1 else 0.0
            row.update({
                "predicted": ans["choice"],
                "correct": None if case["ambiguous"] else ans["choice"] in case["expected"],
                "confidence": ans.get("confidence"),
                "answer_confidence": ans.get("answer_confidence"),
                "top_probability": top,
                "second_probability": second,
                "margin": None if top is None else round(top - second, 4),
                "entropy": round(entropy(list(probs.values())), 4) if probs else None,
                "num_options": len(questions["q"]["criteria"]),
                "latency_ms": round(latency_ms, 1),
                "probabilities": json.dumps(probs),
                "error": "",
            })
        except Exception as e:
            row.update({"predicted": "", "correct": None, "error": repr(e)})
        rows.append(row)

        mark = {True: "OK   ", False: "WRONG", None: "  -  "}[row.get("correct")]
        print(f"[{n:3}/{len(cases)}] {mark} {case['id']:4} "
              f"pred={row.get('predicted', ''):16} conf={fmt(row.get('confidence'), 2)}  "
              f"{case['statement'][:50]}")

    print(f"\nFinished in {time.time() - start:.1f} s")
    return rows


def summarize(rows):
    scored = [r for r in rows if r.get("correct") is not None]
    lines = []
    add = lines.append

    add("=" * 60)
    add("SUMMARY")
    add("=" * 60)
    if scored:
        acc = sum(r["correct"] for r in scored) / len(scored)
        add(f"Overall accuracy (excl. ambiguous): {acc:.1%}  ({len(scored)} cases)")

    add("\nAccuracy by category:")
    by_cat = defaultdict(list)
    for r in scored:
        by_cat[r["category"]].append(r["correct"])
    for cat, vals in by_cat.items():
        add(f"  {cat:16} {sum(vals) / len(vals):6.1%}  ({sum(vals)}/{len(vals)})")

    wrong = [r for r in scored if not r["correct"]]
    confident_wrong = [r for r in wrong if (r.get("confidence") or 0) >= 0.9]
    add(f"\nWrong answers: {len(wrong)}")
    add(f"Confidently wrong (confidence >= 0.90): {len(confident_wrong)}")
    for r in confident_wrong:
        add(f"  {r['id']:4} conf={r['confidence']:.2f} pred={r['predicted']:16} "
            f"expected={r['expected']:16} {r['statement'][:40]}")

    add("\nAUROC (how well each signal predicts correctness; 0.5 = useless, 1.0 = perfect):")
    labels = [1 if r["correct"] else 0 for r in scored]
    for signal in ["confidence", "answer_confidence", "margin"]:
        vals = [r.get(signal) for r in scored]
        if all(v is not None for v in vals):
            add(f"  {signal:18} {fmt(auroc(vals, labels))}")
    ent = [r.get("entropy") for r in scored]
    if all(v is not None for v in ent):
        add(f"  {'1 - entropy':18} {fmt(auroc([1 - v for v in ent], labels))}")

    # Cases sharing a pair ID mean the same thing and should get the same answer.
    pairs = defaultdict(list)
    for r in rows:
        if r["pair"].startswith("P") and r.get("predicted"):
            pairs[r["pair"]].append(r)
    if pairs:
        consistent = [p for p, rs in pairs.items() if len({r["predicted"] for r in rs}) == 1]
        add(f"\nPair consistency: {len(consistent)}/{len(pairs)} pairs got the same answer")
        for p, rs in pairs.items():
            if p not in consistent:
                detail = ", ".join(f"{r['id']}={r['predicted']}" for r in rs)
                add(f"  {p:4} inconsistent: {detail}")

    amb = [r for r in rows if r["ambiguous"] and r.get("predicted")]
    if amb:
        add("\nAmbiguous cases (ideally LOW confidence):")
        for r in amb:
            add(f"  {r['id']:4} conf={fmt(r.get('confidence'), 2)} pred={r['predicted']:16} {r['statement'][:40]}")

    errors = [r for r in rows if r.get("error")]
    if errors:
        add(f"\nErrors: {len(errors)}")
        for r in errors:
            add(f"  {r['id']}: {r['error']}")
    return "\n".join(lines)


def save(rows, summary, out_dir):
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"baseline_{stamp}.csv"
    json_path = out_dir / f"baseline_{stamp}.json"
    txt_path = out_dir / f"baseline_{stamp}_summary.txt"

    fields = list(dict.fromkeys(k for r in rows for k in r))
    # utf-8-sig: Excel on Windows needs the BOM to read UTF-8 correctly.
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
    ap.add_argument("--only", default="", help="run only IDs starting with this letter, e.g. H")
    args = ap.parse_args()

    rows = run(args.tests, args.model, args.only)
    summary = summarize(rows)
    print("\n" + summary)
    save(rows, summary, HERE / "results")
