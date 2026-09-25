"""Run reliability.py on MASSIVE intent (mteb/amazon_massive_intent).

Cases come from Laya's own harness (research/eval/laya_eval.py), so the
utterances, the 20 options and the gold labels match the published tables.

Reports accuracy on the first 100 cases (published: 0.82 for en), the
one-shuffle flip rate (BENCHMARKS.md: 0.150), AUROC with a bootstrap
interval, and decision-bucket accuracy with cutoffs fitted on the first half
of the cases and applied to the second half.

Needs `pip install datasets` and a checkout of the Laya repository.

    python run_massive.py --repo path/to/laya --n 300
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

# Import the installed laya first: the harness prepends the repo root to
# sys.path, which would otherwise shadow it with the repo's source tree.
import laya

from reliability import ReliableAgent, calibrate, decide

HERE = Path(__file__).resolve().parent


def find_repo(path: Path) -> Path:
    """Return the folder containing research/eval/laya_eval.py.

    Unzipping on Windows often nests the repo (laya-main/laya-main/), so also
    look one and two levels down."""
    path = path.resolve()
    marker = Path("research") / "eval" / "laya_eval.py"
    candidates = [path] + sorted(p for p in path.glob("*") if p.is_dir()) + \
        sorted(p for p in path.glob("*/*") if p.is_dir())
    for c in candidates:
        if (c / marker).is_file():
            if c != path:
                print(f"Using repo folder: {c}")
            return c
    contents = ", ".join(sorted(p.name for p in path.glob("*"))[:15]) if path.exists() else "(folder not found)"
    sys.exit(f"Could not find research/eval/laya_eval.py under {path}\n"
             f"That folder contains: {contents}\n"
             f"Point --repo at the folder that contains 'research' and 'laya'.")


def auroc(scores, labels):
    """Rank-based AUROC (ties get average rank). 1 = right answers score higher."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(1 for _, y in pairs if y)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    r_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y)
    return (r_pos - pos * (pos + 1) / 2) / (pos * neg)


def bootstrap_diff(rows, a, b, n_boot=2000, seed=0):
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        s = [rng.choice(rows) for _ in rows]
        y = [r["correct"] for r in s]
        x, z = auroc([r[a] for r in s], y), auroc([r[b] for r in s], y)
        if x is not None and z is not None:
            diffs.append(x - z)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    return sum(diffs) / len(diffs), lo, hi


def buckets(rows, accept, escalate):
    out = {}
    for d in ("ACCEPT", "VERIFY", "ESCALATE"):
        b = [r for r in rows if decide(r["soft_stability"], accept, escalate) == d]
        out[d] = (len(b), sum(r["correct"] for r in b) / len(b) if b else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the extracted laya-main folder")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n", type=int, default=300, help="number of cases (harness default is 100)")
    ap.add_argument("--variants", default="full")
    ap.add_argument("--model", default="convaiinnovations/laya")
    args = ap.parse_args()

    repo = find_repo(Path(args.repo))
    sys.path.insert(1, str(repo))  # after the installed laya
    from research.eval import laya_eval as harness

    print(f"laya {laya.__version__} from {Path(laya.__file__).parent}")
    print(f"Loading {harness.DATASET} ({args.lang}) ...")
    data = harness.load_language(args.lang)
    cases, gold, keys = harness.build_suite(
        data, sorted({r["label_text"] for r in data}), args.n, harness.N_OPTS, harness.SEED)
    print(f"{len(cases)} cases, {harness.N_OPTS} options each")

    base = laya.load(args.model)
    agent = ReliableAgent(base, variants=args.variants)

    rows, t0 = [], time.perf_counter()
    for i, (state, question) in enumerate(cases):
        a = agent.predict(state, question)["answers"]["intent"]
        r = a["reliability"]
        rows.append({
            "index": i, "utterance": state["utterance"], "gold": keys[i][gold[i]],
            "pred": a["choice"], "correct": int(a["choice"] == keys[i][gold[i]]),
            "answer_confidence": a["answer_confidence"],
            "soft_stability": r["soft_stability"], "stability": r["stability"],
            "variant_choices": r["variant_choices"],
        })
        if (i + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(cases)}  ({el / (i + 1):.1f} s/case, ~{el / (i + 1) * (len(cases) - i - 1) / 60:.0f} min left)")

    y = [r["correct"] for r in rows]
    lines = []
    add = lines.append
    add("=" * 66)
    add(f"MASSIVE intent ({args.lang}), {len(rows)} cases, {harness.N_OPTS} options, variants={args.variants}")
    add("=" * 66)
    first = rows[:100]
    add(f"Accuracy, first {len(first)} cases: {sum(r['correct'] for r in first) / len(first):.3f}"
        f"   (harness publishes 0.82 for en, n=100)")
    add(f"Accuracy, all {len(rows)} cases:   {sum(y) / len(y):.3f}")
    flips = [r["variant_choices"].get("shuffle1") for r in rows]
    flip_n = [r for r, f in zip(rows, flips) if f is not None]
    if flip_n:
        rate = sum(r["variant_choices"]["shuffle1"] != r["pred"] for r in flip_n) / len(flip_n)
        add(f"Option-order flip rate (one shuffle): {rate:.3f}   (BENCHMARKS.md: 0.150 for en)")

    add("\nAUROC (predicting right vs wrong; 0.5 = useless):")
    for s in ("answer_confidence", "soft_stability", "stability"):
        v = auroc([r[s] for r in rows], y)
        add(f"  {s:18} {v:.3f}" if v is not None else f"  {s:18} n/a")
    if 0 < sum(y) < len(y):
        m, lo, hi = bootstrap_diff(rows, "soft_stability", "answer_confidence")
        add(f"  soft_stability - answer_confidence: {m:+.3f}  (95% interval {lo:+.3f} to {hi:+.3f})")

    half = len(rows) // 2
    cal, ev = rows[:half], rows[half:]
    acc_t, esc_t = calibrate([r["soft_stability"] for r in cal], [bool(r["correct"]) for r in cal])
    add(f"\nCutoffs calibrated on cases 1-{half}: ACCEPT >= {acc_t:.2f}, ESCALATE < {esc_t:.2f}")
    add(f"Applied to cases {half + 1}-{len(rows)}:")
    for d, (n, acc) in buckets(ev, acc_t, esc_t).items():
        add(f"  {d:8} {n:4} cases  accuracy {acc:.1%}" if acc is not None else f"  {d:8}    0 cases")
    k = buckets(ev, acc_t, esc_t)["ACCEPT"][0]
    if k:
        top = sorted(ev, key=lambda r: r["answer_confidence"], reverse=True)[:k]
        add(f"  (answer_confidence accepting the same {k}: {sum(r['correct'] for r in top) / k:.1%})")
    add(f"\nTime: {(time.perf_counter() - t0) / len(rows):.1f} s/case")
    summary = "\n".join(lines)
    print("\n" + summary)

    out = HERE / "results" / f"massive_{args.lang}_{args.n}_{args.variants}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "laya_version": laya.__version__,
                               "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out.with_suffix(".txt")).write_text(summary, encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
