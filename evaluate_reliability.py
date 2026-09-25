"""Evaluate reliability.py on a test set.

Reports AUROC (answer_confidence vs stability), accuracy per decision bucket,
and time per case against a plain predict call. With --calibrate-on, the
cutoffs are fitted on one file and applied to another.

    python evaluate_reliability.py --tests laya_tests.json --variants full
    python evaluate_reliability.py --tests laya_tests_v2.json --calibrate-on laya_tests.json
"""
import argparse
import json
import time
from pathlib import Path

import laya
from reliability import DEFAULT_ACCEPT, DEFAULT_ESCALATE, ReliableAgent, calibrate, decide
from run_benchmark import auroc, build_question

HERE = Path(__file__).resolve().parent


def run_set(path, base, agent, label):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows, t_plain, t_rel, same = [], 0.0, 0.0, 0
    print(f"Running {label}: {path} ({len(data['cases'])} cases)")
    for n, case in enumerate(data["cases"], 1):
        q = build_question(case, data["schemas"])
        state = {"message": case["statement"]}
        t0 = time.perf_counter()
        plain = base.predict(state, q)["answers"]["q"]
        t_plain += time.perf_counter() - t0
        t0 = time.perf_counter()
        a = agent.predict(state, q)["answers"]["q"]
        t_rel += time.perf_counter() - t0
        same += a["choice"] == plain["choice"]
        rows.append({"id": case["id"], "ambiguous": case["ambiguous"],
                     "correct": None if case["ambiguous"] else a["choice"] in case["expected"],
                     "answer_confidence": a["answer_confidence"], **a["reliability"]})
        if n % 25 == 0:
            print(f"  {n}/{len(data['cases'])}")
    return rows, t_plain, t_rel, same


def report(path, rows, t_plain, t_rel, same, accept, escalate, variants):
    scored = [r for r in rows if r["correct"] is not None]
    y = [int(r["correct"]) for r in scored]
    for r in rows:
        r["decision"] = decide(r["soft_stability"], accept, escalate)
    print("\n" + "=" * 64)
    print(f"Test set: {path}  ({len(scored)} scored, {len(rows) - len(scored)} ambiguous)")
    print(f"Variants: {variants}  ({rows[0]['n_variants']} rows per question)")
    print(f"Thresholds: ACCEPT >= {accept:.2f}, ESCALATE < {escalate:.2f}")
    print(f"Original answer identical to plain predict: {same}/{len(rows)}")
    print(f"AUROC answer_confidence: {auroc([r['answer_confidence'] for r in scored], y):.3f}")
    print(f"AUROC soft_stability:    {auroc([r['soft_stability'] for r in scored], y):.3f}")
    print(f"AUROC stability:         {auroc([r['stability'] for r in scored], y):.3f}")

    print("\nDecisions (accuracy of the answers in each bucket):")
    for d in ("ACCEPT", "VERIFY", "ESCALATE"):
        b = [r for r in scored if r["decision"] == d]
        acc = f"{sum(r['correct'] for r in b) / len(b):.1%}" if b else "n/a"
        print(f"  {d:8} {len(b):4} cases ({len(b) / len(scored):5.1%})  accuracy {acc}")
    k = sum(r["decision"] == "ACCEPT" for r in scored)
    if k:
        top = sorted(scored, key=lambda r: r["answer_confidence"], reverse=True)[:k]
        print(f"  (answer_confidence accepting the same {k} cases: "
              f"{sum(r['correct'] for r in top) / k:.1%})")

    amb = [r for r in rows if r["ambiguous"]]
    if amb:
        print(f"\nAmbiguous cases marked ACCEPT (should be few): "
              f"{sum(r['decision'] == 'ACCEPT' for r in amb)}/{len(amb)}")
    print(f"\nSpeed: plain {1000 * t_plain / len(rows):.0f} ms/case, "
          f"reliable {1000 * t_rel / len(rows):.0f} ms/case "
          f"({t_rel / max(t_plain, 1e-9):.1f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=str(HERE / "laya_tests_v2.json"))
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--variants", default="full", help="'full', 'fast', or comma-separated names")
    ap.add_argument("--calibrate-on", default="",
                    help="set thresholds on this (development) file first, then apply them to --tests")
    args = ap.parse_args()

    variants = args.variants if args.variants in ("full", "fast") else args.variants.split(",")
    base = laya.load(args.model)
    agent = ReliableAgent(base, variants=variants)

    accept, escalate = DEFAULT_ACCEPT, DEFAULT_ESCALATE
    if args.calibrate_on:
        dev_rows, *_ = run_set(args.calibrate_on, base, agent, "calibration set")
        dev = [r for r in dev_rows if r["correct"] is not None]
        accept, escalate = calibrate([r["soft_stability"] for r in dev], [r["correct"] for r in dev])
        print(f"\nCalibrated on {args.calibrate_on}: ACCEPT >= {accept:.2f}, ESCALATE < {escalate:.2f}\n")
        if accept > 1.0:
            print("WARNING: no cutoff reached the 90% accuracy target on the calibration set, "
                  "so nothing will be ACCEPTed. Lower target_accuracy in calibrate() if needed.\n")

    rows, t_plain, t_rel, same = run_set(args.tests, base, agent, "evaluation set")
    report(args.tests, rows, t_plain, t_rel, same, accept, escalate, args.variants)

    tag = args.variants if isinstance(variants, str) else "custom"
    out = HERE / "results" / f"reliability_eval_{Path(args.tests).stem}_{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"accept": accept, "escalate": escalate, "variants": args.variants,
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
