"""Run reliability.py on MASSIVE intent, using the cases from Laya's own harness
(research/eval/laya_eval.py) so they match the published tables.

Reports AUROC split by transformation (reorder / rename / all), how often the
head budget truncates option text, and decision-bucket accuracy. --options
changes the option count; at 6 nothing is truncated (see Laya #543).

Needs `pip install datasets` and a checkout of the Laya repository.

    python run_massive.py --repo path/to/laya --n 300
    python run_massive.py --repo path/to/laya --n 300 --options 6
"""
import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

# Import the installed laya first: the harness prepends the repo root to
# sys.path, which would otherwise shadow it with the repo's source tree.
import laya

from reliability import ReliableAgent, calibrate, decide

HERE = Path(__file__).resolve().parent
REORDER = ["reversed", "shuffle1", "shuffle2", "shuffle3"]
RENAME = ["letters", "letters_reversed", "letters_shuffle"]


def find_repo(path: Path) -> Path:
    """Folder containing research/eval/laya_eval.py. Unzipping on Windows often
    nests the repo (laya-main/laya-main/), so also look two levels down."""
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
    """Rank-based AUROC; ties get the average rank."""
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
    return sum(diffs) / len(diffs), diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def option_budget(tok, texts, head_max_len):
    """Mirror laya.common.build_sequence: tokens each option needs vs keeps.
    Returns (truncated?, mean tokens needed, mean tokens kept) per option."""
    need = [1 + len(tok(" " + t, add_special_tokens=False, truncation=True, max_length=48)["input_ids"])
            for t in texts]
    if head_max_len - sum(need) >= 16:
        return False, sum(need) / len(need), sum(need) / len(need)
    per = max(4, (head_max_len - 16) // max(1, len(need)))
    kept = [min(n, per) for n in need]
    return True, sum(need) / len(need), sum(kept) / len(kept)


def option_texts(criteria, relabel):
    """Option strings as Laya renders them ("key: description")."""
    keys = list(string.ascii_uppercase[: len(criteria)]) if relabel else list(criteria)
    return [f"{k}: {v}" for k, v in zip(keys, criteria.values())]


def subset_scores(r, names):
    """Soft and plain stability over a subset of variants, excluding the original."""
    names = [n for n in names if n in r["variant_choices"]]
    if not names:
        return None, None
    soft = sum(r["variant_support"][n] for n in names) / len(names)
    plain = sum(r["variant_choices"][n] == r["pred"] for n in names) / len(names)
    return soft, plain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="folder containing the Laya repository")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n", type=int, default=300, help="number of cases")
    ap.add_argument("--options", type=int, default=0, help="options per question (default: harness, 20)")
    ap.add_argument("--variants", default="full")
    ap.add_argument("--model", default="convaiinnovations/laya")
    args = ap.parse_args()

    sys.path.insert(1, str(find_repo(Path(args.repo))))
    from research.eval import laya_eval as harness

    n_opts = args.options or harness.N_OPTS
    print(f"laya {laya.__version__} from {Path(laya.__file__).parent}")
    print(f"Loading {harness.DATASET} ({args.lang}) ...")
    data = harness.load_language(args.lang)
    cases, gold, keys = harness.build_suite(
        data, sorted({r["label_text"] for r in data}), args.n, n_opts, harness.SEED)
    print(f"{len(cases)} cases, {n_opts} options each")

    base = laya.load(args.model)
    head_max_len = getattr(base, "cfg", {}).get("head_max_len", 192)
    tok = getattr(base, "tok", None)
    agent = ReliableAgent(base, variants=args.variants)

    rows, t0 = [], time.perf_counter()
    for i, (state, question) in enumerate(cases):
        a = agent.predict(state, question)["answers"]["intent"]
        r = a["reliability"]
        row = {
            "index": i, "utterance": state["utterance"], "gold": keys[i][gold[i]],
            "pred": a["choice"], "correct": int(a["choice"] == keys[i][gold[i]]),
            "answer_confidence": a["answer_confidence"],
            "soft_stability": r["soft_stability"], "stability": r["stability"],
            "variant_choices": r["variant_choices"], "variant_support": r["variant_support"],
        }
        for name, sub in (("reorder", REORDER), ("rename", RENAME)):
            row[f"soft_{name}"], row[f"plain_{name}"] = subset_scores(row, sub)
        if tok is not None:
            crit = question["intent"]["criteria"]
            for name, relabel in (("original", False), ("renamed", True)):
                cut, need, kept = option_budget(tok, option_texts(crit, relabel), head_max_len)
                row[f"trunc_{name}"], row[f"need_{name}"], row[f"kept_{name}"] = cut, need, kept
        rows.append(row)
        if (i + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(cases)}  ({el / (i + 1):.1f} s/case, "
                  f"~{el / (i + 1) * (len(cases) - i - 1) / 60:.0f} min left)")

    y = [r["correct"] for r in rows]
    lines = []
    add = lines.append
    add("=" * 70)
    add(f"MASSIVE intent ({args.lang}), {len(rows)} cases, {n_opts} options, "
        f"variants={args.variants}, head_max_len={head_max_len}")
    add("=" * 70)
    if n_opts == harness.N_OPTS:
        first = rows[:100]
        add(f"Accuracy, first {len(first)} cases: {sum(r['correct'] for r in first) / len(first):.3f}"
            f"   (published: 0.82 for en, n=100)")
    add(f"Accuracy, all {len(rows)} cases:   {sum(y) / len(y):.3f}")

    if tok is not None:
        add("\nOption text vs head budget (mean tokens per option):")
        for name in ("original", "renamed"):
            cut = sum(r[f"trunc_{name}"] for r in rows) / len(rows)
            need = sum(r[f"need_{name}"] for r in rows) / len(rows)
            kept = sum(r[f"kept_{name}"] for r in rows) / len(rows)
            add(f"  {name:9} truncated in {cut:6.1%} of cases; needs {need:4.1f}, keeps {kept:4.1f}")

    add("\nAUROC (right vs wrong; 0.5 = useless). Subsets exclude the original question:")
    signals = [("answer_confidence", "answer_confidence"),
               ("soft_stability", "soft stability, all (incl. original)"),
               ("soft_reorder", "soft stability, reorder only"),
               ("soft_rename", "soft stability, rename only"),
               ("stability", "plain stability, all"),
               ("plain_reorder", "plain stability, reorder only"),
               ("plain_rename", "plain stability, rename only")]
    for key, label in signals:
        vals = [r[key] for r in rows]
        v = auroc(vals, y) if all(x is not None for x in vals) else None
        add(f"  {label:38} {v:.3f}" if v is not None else f"  {label:38} n/a")

    if 0 < sum(y) < len(y):
        add("\nDifference vs answer_confidence, bootstrap 95% interval:")
        for key in ("soft_stability", "soft_reorder", "soft_rename"):
            if all(r[key] is not None for r in rows):
                m, lo, hi = bootstrap_diff(rows, key, "answer_confidence")
                add(f"  {key:16} {m:+.3f}  ({lo:+.3f} to {hi:+.3f})")

    add("\nHow often variants changed the answer:")
    for name, sub in (("reorder", REORDER), ("rename", RENAME)):
        flips = [sum(r["variant_choices"][k] != r["pred"] for k in sub if k in r["variant_choices"])
                 / max(1, sum(k in r["variant_choices"] for k in sub)) for r in rows]
        right = [f for f, c in zip(flips, y) if c]
        wrong = [f for f, c in zip(flips, y) if not c]
        add(f"  {name} flip rate: right answers {sum(right) / max(1, len(right)):.3f}, "
            f"wrong answers {sum(wrong) / max(1, len(wrong)):.3f}")

    half = len(rows) // 2
    cal, ev = rows[:half], rows[half:]
    acc_t, esc_t = calibrate([r["soft_stability"] for r in cal], [bool(r["correct"]) for r in cal])
    add(f"\nCutoffs fitted on cases 1-{half}: ACCEPT >= {acc_t:.2f}, ESCALATE < {esc_t:.2f}; "
        f"applied to cases {half + 1}-{len(rows)}:")
    for d in ("ACCEPT", "VERIFY", "ESCALATE"):
        b = [r for r in ev if decide(r["soft_stability"], acc_t, esc_t) == d]
        add(f"  {d:8} {len(b):4} cases  accuracy {sum(r['correct'] for r in b) / len(b):.1%}"
            if b else f"  {d:8}    0 cases")
    add(f"\nTime: {(time.perf_counter() - t0) / len(rows):.1f} s/case")
    summary = "\n".join(lines)
    print("\n" + summary)

    out = HERE / "results" / f"massive_{args.lang}_{args.n}_k{n_opts}_{args.variants}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "laya_version": laya.__version__,
                               "head_max_len": head_max_len, "options": n_opts, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".txt").write_text(summary, encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
