# laya-reliability

Reliability testing for [Laya](https://github.com/NandhaKishorM/laya) to catch answers that are wrong even when Laya gives them high confidence.

## Why I built this

While testing Laya, I noticed that `answer_confidence` isn't always a good indication that an answer is actually correct.

On Laya's MASSIVE intent benchmark, 37 of the 65 wrong answers (out of 300 total cases) still had an `answer_confidence` of 0.90 or higher.

That means just using a confidence threshold isn't enough to catch a lot of the mistakes.

The idea behind this project is pretty simple: ask Laya the same choice question in slightly different ways that shouldn't change the correct answer and see if Laya stays consistent.

For example:

- reorder the choices
- rename the choices to A/B/C while keeping their descriptions the same

If the answer changes just because the options were reordered or renamed, that's a sign that the prediction probably isn't very stable.

I use this to calculate a **soft stability** score for each answer.

The important part is that this doesn't modify Laya or replace its original prediction. It wraps around Laya and adds reliability information to the result.

## Results

Tested with:

- Laya `0.3.20`
- checkpoint: `convaiinnovations/laya`
- English
- CPU

AUROC is used to measure how well each score separates correct answers from incorrect ones. `0.5` is basically random and `1.0` is perfect separation.

| | MASSIVE `en` (300 cases, 20 options) | Hand-written held-out set (280 cases) |
|---|---|---|
| AUROC `answer_confidence` | 0.831 | 0.647 |
| AUROC soft stability | **0.907** | **0.714** |
| Difference, bootstrap 95% interval | +0.076 (+0.035 to +0.123) | +0.068 (+0.031 to +0.111)* |

\*The held-out difference is from an earlier run of the same method with different shuffles (AUROC 0.715).

### MASSIVE results at different coverage levels

This shows what happens if Laya only handles the cases it trusts the most.

| Cases Laya handles | Ranked by `answer_confidence` | Ranked by soft stability |
|---|---|---|
| 50% | 94.7% | **98.0%** |
| 70% | 91.0% | **94.3%** |
| 80% | 89.2% | **90.4%** |

At 50% coverage, using soft stability resulted in about **60% fewer wrong answers**.

A few other things I found:

- Out of the 37 wrong answers with `answer_confidence >= 0.90`, stability flagged 29.
- MASSIVE examples are generated using Laya's own `research/eval/laya_eval.py` harness.
- Accuracy on the first 100 MASSIVE cases was `0.820`, which matches Laya's published `0.82`.
- Laya's original answers are not changed. The original predictions were identical to plain `predict`; this module only adds a `reliability` field.
- Renaming choices to A/B/C gave a stronger signal than just reordering them. On the held-out set, plain stability AUROC was `0.703` for renaming compared with `0.611` for reordering. This result is exploratory.

## How it works

For every `choice` question, the wrapper creates up to 7 additional versions:

1. reversed option order
2. 3 seeded random option orders
3. 3 versions where the options are renamed to A/B/C in different orders

After Laya answers them, the renamed options are mapped back to the original choices before comparing the results.

Duplicate variants are removed because Laya is deterministic, so running an identical version more than once would artificially increase the stability score.

All of the variants are added as extra questions to the **same `predict` call**.

Laya encodes each question as its own row, so the variants can be evaluated independently in one forward pass.

The original Laya answer is still the answer returned to the caller. The wrapper just adds extra information about how reliable that answer seems to be.

### Reliability values

**Soft stability**

Average probability Laya gives its original answer across all of the variants.

Higher values mean the answer stayed stronger across the different versions.

**Plain stability**

The percentage of variants where Laya kept the same original answer.

This doesn't depend on temperature because temperature doesn't change which option wins.

**Decision**

Each answer is assigned one of:

```text
ACCEPT
VERIFY
ESCALATE
```

The decision is based on soft stability and two configurable cutoffs.

`score` and `noul` questions aren't modified and pass through normally.

## Installation

Install Laya:

```bash
pip install laya
```

Then copy `reliability.py` into your project.

## Usage

```python
import laya
from reliability import ReliableAgent

agent = ReliableAgent(laya.load("convaiinnovations/laya"))

questions = {
    "tone": {
        "type": "choice",
        "instructions": "What is the speaker expressing?",
        "criteria": {
            "grateful": "The speaker thanks someone or shows appreciation.",
            "agreeing": "The speaker agrees to or accepts a plan.",
            "other": "None of the above.",
        },
    }
}

out = agent.predict("I couldn't agree less.", questions)

answer = out["answers"]["tone"]

answer["choice"]
answer["reliability"]["decision"]
answer["reliability"]["soft_stability"]
answer["reliability"]["distinct_choices"]
```

The original Laya result is still available through:

```python
answer["choice"]
```

The additional reliability information is under:

```python
answer["reliability"]
```

### Fast mode

The default version uses up to 8 rows per question: the original question plus up to 7 variants.

If speed matters more, use:

```python
agent = ReliableAgent(base, variants="fast")
```

Fast mode uses 3 rows per question instead of 8.

It is faster, but the reliability signal is weaker.

### Batch prediction

`predict_batch` is also supported:

```python
agent.predict_batch(states, questions)
```

It works like Laya's normal `predict_batch`.

## Calibration

The ACCEPT and ESCALATE thresholds should be calibrated for the dataset you're actually using.

```python
calibrate(soft_scores, correct)
```

I wouldn't recommend relying on the default cutoffs for a real application.

During testing, the best cutoffs didn't transfer reliably between datasets. They even changed between the two halves of MASSIVE.

## Cost

These measurements were done on CPU.

| Setting | Rows per question | Time vs plain `predict` |
|---|---|---|
| `full` (default) | up to 8 | 5.1–5.5× |
| `fast` | 3 | 2.4× |

On MASSIVE with 20 options, `full` took around **6.0 seconds per case**.

I haven't measured GPU performance yet. Since the additional variants are separate rows in the same forward pass, some of the extra work may parallelize better on a GPU.

## What this doesn't catch

This isn't able to catch every confident mistake.

### Consistent misreadings

For example:

```text
"Gee, thanks for the help, genius."
```

Laya predicts `grateful` with `1.00` confidence and continues predicting `grateful` under every variant.

Since the model is consistently wrong, stability doesn't detect anything unusual.

This happened pretty often with keyword-driven errors in the hand-written held-out set:

```text
"thanks" -> grateful
"sorry"  -> apologetic
"kill"   -> threat
```

Stability only caught 6 of 27 confident mistakes like these.

The MASSIVE errors behaved differently. They were more often confusions between similar intents, and those predictions were more likely to change when the options changed.

### Ambiguous inputs

Stability also isn't a good ambiguity detector by itself.

Out of 20 deliberately ambiguous held-out examples, 14 were still marked `ACCEPT`.

Examples included inputs like:

```text
"ok."
"sure"
```

## Limitations

There are a few limitations to keep in mind when looking at these results.

- The two hand-written datasets were written and labelled by me specifically to test Laya's weak spots. They're intentionally harder than normal examples, so their raw accuracy shouldn't be treated as a general Laya benchmark.
- The MASSIVE experiment currently covers one language (English), one checkpoint and 300 cases.
- MASSIVE has 20 choices per question. Laya's `choice:11+` temperature clamp therefore affects those experiments, while the hand-written datasets only contain 2–6 choices.
- `calibrate()` had a bug in its first version where the ESCALATE cutoff was placed at the ACCEPT cutoff, which meant there was no VERIFY range. The fix was chosen using the development set only.

## Reproducing the results

Install the dependencies:

```bash
pip install -U laya pytest datasets
```

Run the tests:

```bash
python -m pytest test_reliability.py -q
```

Run the hand-written development set:

```bash
python evaluate_reliability.py --tests laya_tests.json --variants full
```

Run the held-out set using the development set for calibration:

```bash
python evaluate_reliability.py \
    --tests laya_tests_v2.json \
    --variants full \
    --calibrate-on laya_tests.json
```

Run the MASSIVE benchmark:

```bash
python run_massive.py --repo path/to/laya --n 300
```

This requires a local checkout of the Laya repository.

## Repository files

| File | Purpose |
|---|---|
| `reliability.py` | Main reliability wrapper, variants, soft stability, decisions and `calibrate()` |
| `test_reliability.py` | Unit tests using fake agents; doesn't require downloading the model |
| `evaluate_reliability.py` | Evaluates a test set and reports AUROC, decisions and speed |
| `run_massive.py` | Runs the reliability layer against Laya's MASSIVE benchmark using Laya's own harness |
| `run_benchmark.py` | Original baseline experiments |
| `run_stability.py` | Original stability experiments |
| `laya_tests.json` | Development set with 111 hand-written cases |
| `laya_tests_v2.json` | Hand-written held-out set; 300 cases total, with 280 used for the reported held-out results |
| `results/` | Per-case results and summaries from the experiments |

## Related Laya issues

- [#361](https://github.com/NandhaKishorM/laya/issues/361) - confidence-based abstention using `min_confidence`. Stability could be used as an additional signal.
- [#394](https://github.com/NandhaKishorM/laya/issues/394) - confidence thresholds have trouble separating correct and incorrect answers when there are 20 options. I reproduced this on 300 MASSIVE cases.
- [#244](https://github.com/NandhaKishorM/laya/issues/244) - Laya's existing option-order invariance test in `research/eval/metamorphic.py`. That implementation tests option reordering but leaves label renaming for future work.

## License

Apache License 2.0, same as Laya.
