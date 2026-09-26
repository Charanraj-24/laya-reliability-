"""Compare predict_reliable_batch with single predict_reliable, and time both.

Uses the tone (Q1) cases, since they all share one question, which is what
predict_batch needs. Checks that batched and single runs agree (answers,
decisions, stability scores) and that the answers match plain Laya.

    python run_batch_check.py
    python run_batch_check.py --batch-size 32
"""
import argparse
import json
import time
from pathlib import Path

import laya
from reliability import predict_reliable, predict_reliable_batch
from run_benchmark import build_question

HERE = Path(__file__).resolve().parent
TOL = 1e-3


def load_cases(paths):
    states, question = [], None
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for case in data["cases"]:
            if case["schema"] != "Q1":
                continue
            q = build_question(case, data["schemas"])
            if question is None:
                question = q
            elif q != question:
                raise SystemExit(f"{case['id']}: Q1 question differs between files")
            states.append({"message": case["statement"]})
    return states, question


def timed(label, fn):
    print(f"  {label} ...", end="", flush=True)
    t0 = time.perf_counter()
    out = fn()
    el = time.perf_counter() - t0
    print(f" done in {el:.0f} s", flush=True)
    return out, el


def each(fn, items, label):
    """Run fn over items, printing progress every 20 items."""
    out = []
    for i, x in enumerate(items, 1):
        out.append(fn(x))
        if i % 20 == 0 or i == len(items):
            print(f"\r  {label}: {i}/{len(items)}", end="", flush=True)
    print(flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+",
                    default=[str(HERE / "laya_tests.json"), str(HERE / "laya_tests_v2.json")])
    ap.add_argument("--batch-size", type=int, default=16, help="states per forward pass")
    ap.add_argument("--model", default="convaiinnovations/laya")
    args = ap.parse_args()

    states, q = load_cases(args.tests)
    n, bs = len(states), args.batch_size
    print(f"{n} tone cases, one shared question with {len(q['q']['criteria'])} options, batch size {bs}")

    base = laya.load(args.model)
    print("Warming up ...", flush=True)
    base.predict(states[0], q)  # so model loading is not timed
    base.predict_batch(states[:2], q, batch_size=bs)

    print("Running 4 passes over all cases:", flush=True)
    plain_single, t_ps = timed("1/4 plain, single",
                               lambda: each(lambda s: base.predict(s, q), states, "    plain single"))
    plain_batch, t_pb = timed("2/4 plain, batched", lambda: base.predict_batch(states, q, batch_size=bs))
    rel_single, t_rs = timed("3/4 reliable, single",
                             lambda: each(lambda s: predict_reliable(base, s, q, seed="batch"),
                                          states, "    reliable single"))
    rel_batch, t_rb = timed("4/4 reliable, batched",
                            lambda: predict_reliable_batch(base, states, q, batch_size=bs))

    def ans(results, i):
        return results[i]["answers"]["q"]

    plain_same = sum(ans(plain_single, i)["choice"] == ans(plain_batch, i)["choice"] for i in range(n))
    plain_diff = max(abs(ans(plain_single, i)["answer_confidence"] - ans(plain_batch, i)["answer_confidence"])
                     for i in range(n))
    rel_same = sum(ans(rel_single, i)["choice"] == ans(rel_batch, i)["choice"]
                   and ans(rel_single, i)["reliability"]["decision"] == ans(rel_batch, i)["reliability"]["decision"]
                   for i in range(n))
    rel_diff = max(abs(ans(rel_single, i)["reliability"]["soft_stability"]
                       - ans(rel_batch, i)["reliability"]["soft_stability"]) for i in range(n))
    orig_same = sum(ans(rel_batch, i)["choice"] == ans(plain_single, i)["choice"] for i in range(n))

    ok = lambda good: "OK" if good else "CHECK"
    lines = [
        "=" * 66,
        f"Batch path check: {n} cases, batch size {bs}, device {getattr(base, 'device', '?')}",
        "=" * 66,
        "Correctness:",
        f"  plain batch vs single, same answer:        {plain_same}/{n}  "
        f"(max answer_confidence diff {plain_diff:.2e})  {ok(plain_same == n and plain_diff < TOL)}",
        f"  reliable batch vs single, same answer and decision: {rel_same}/{n}  "
        f"(max soft_stability diff {rel_diff:.2e})  {ok(rel_same == n and rel_diff < TOL)}",
        f"  reliable batch answer == plain answer:     {orig_same}/{n}  {ok(orig_same == n)}",
        "",
        "Time per case:",
        f"  plain, single predict     {1000 * t_ps / n:7.0f} ms",
        f"  plain, predict_batch      {1000 * t_pb / n:7.0f} ms",
        f"  reliable, single          {1000 * t_rs / n:7.0f} ms   ({t_rs / t_ps:.1f}x plain single)",
        f"  reliable, batched         {1000 * t_rb / n:7.0f} ms   ({t_rb / t_ps:.1f}x plain single, "
        f"{t_rb / t_pb:.1f}x plain batched)",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    out = HERE / "results" / f"batch_check_bs{bs}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(summary, encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
