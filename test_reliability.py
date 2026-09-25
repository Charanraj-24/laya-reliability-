"""Tests for reliability.py. Uses fake agents, so no model download is needed.

Run:  python -m pytest test_reliability.py -q
"""
import pytest

from reliability import (ReliableAgent, calibrate, decide, make_variants,
                         predict_reliable, predict_reliable_batch)

Q = {"tone": {"type": "choice", "instructions": "What is expressed?",
              "criteria": {"grateful": "Thanks someone.", "apologetic": "Apologizes.",
                           "other": "None of the above."}}}


class MeaningAgent:
    """Answers from option descriptions only: order and label names never matter."""
    def predict(self, state, questions, **kw):
        ans = {}
        for qid, q in questions.items():
            if q.get("type") != "choice":
                ans[qid] = {"type": q.get("type"), "confidence": 0.9, "answer_confidence": 0.9}
                continue
            keys = list(q["criteria"])
            probs = {k: (0.8 if "Thanks" in q["criteria"][k] else 0.2 / (len(keys) - 1)) for k in keys}
            best = max(probs, key=probs.get)
            ans[qid] = {"type": "choice", "choice": best, "probabilities": probs,
                        "confidence": 0.5, "answer_confidence": probs[best]}
        return {"model": "fake", "answers": ans}

    def predict_batch(self, states, questions, **kw):
        return [self.predict(s, questions) for s in states]


class FirstOptionAgent(MeaningAgent):
    """Always picks whatever option is listed first: maximally fragile."""
    def predict(self, state, questions, **kw):
        ans = {}
        for qid, q in questions.items():
            keys = list(q["criteria"])
            probs = {k: (0.9 if i == 0 else 0.1 / (len(keys) - 1)) for i, k in enumerate(keys)}
            ans[qid] = {"type": "choice", "choice": keys[0], "probabilities": probs,
                        "confidence": 0.9, "answer_confidence": 0.9}
        return {"model": "fake", "answers": ans}


def test_variants_unique_deterministic_and_original_first():
    v1 = make_variants(["a", "b", "c"], "seed")
    v2 = make_variants(["a", "b", "c"], "seed")
    assert v1 == v2
    assert v1[0] == ("original", ["a", "b", "c"], False)
    assert len({(tuple(o), r) for _, o, r in v1}) == len(v1)


def test_two_options_have_at_most_four_variants():
    assert len(make_variants(["x", "y"], "s", n_shuffles=10)) == 4


def test_stable_agent_is_accepted_and_answer_unchanged():
    out = predict_reliable(MeaningAgent(), "thank you", Q)
    a = out["answers"]["tone"]
    assert a["choice"] == "grateful"
    assert a["reliability"]["stability"] == 1.0
    assert a["reliability"]["decision"] == "ACCEPT"
    assert set(out["answers"]) == {"tone"}  # variant keys are hidden from the caller


def test_fragile_agent_is_not_accepted():
    r = predict_reliable(FirstOptionAgent(), "thank you", Q)["answers"]["tone"]["reliability"]
    assert r["stability"] < 1.0
    assert r["decision"] != "ACCEPT"
    assert len(r["distinct_choices"]) > 1


def test_single_predict_call():
    calls = []
    class Counting(MeaningAgent):
        def predict(self, state, questions, **kw):
            calls.append(len(questions))
            return super().predict(state, questions)
    predict_reliable(Counting(), "hi", Q)
    assert len(calls) == 1 and calls[0] > 1


def test_non_choice_questions_pass_through():
    q = dict(Q, urgent={"type": "noul", "instructions": "Urgent?", "criteria": {}})
    out = predict_reliable(MeaningAgent(), "hi", q)
    assert "reliability" not in out["answers"]["urgent"]
    assert "reliability" in out["answers"]["tone"]


def test_batch_matches_single():
    agent = MeaningAgent()
    batch = predict_reliable_batch(agent, ["a", "b"], Q)
    assert len(batch) == 2
    assert all(b["answers"]["tone"]["reliability"]["decision"] == "ACCEPT" for b in batch)


def test_wrapper_passes_through_attributes():
    inner = MeaningAgent()
    inner.model_id = "fake-id"
    assert ReliableAgent(inner).model_id == "fake-id"


def test_reserved_separator_in_question_id_rejected():
    with pytest.raises(ValueError):
        predict_reliable(MeaningAgent(), "hi", {"bad::rel1": Q["tone"]})


def test_decide_thresholds():
    assert decide(0.9) == "ACCEPT"
    assert decide(0.7) == "VERIFY"
    assert decide(0.3) == "ESCALATE"


def test_calibrate_finds_sensible_thresholds():
    scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.85, 0.9, 0.95, 0.97, 0.99]
    correct = [False, False, True, False, True, True, True, True, True, True]
    accept, escalate = calibrate(scores, correct, target_accuracy=0.9)
    assert escalate <= accept
    kept = [c for s, c in zip(scores, correct) if s >= accept]
    assert sum(kept) / len(kept) >= 0.9


def test_fast_preset_uses_three_rows():
    seen = []
    class Counting(MeaningAgent):
        def predict(self, state, questions, **kw):
            seen.append(len(questions))
            return super().predict(state, questions)
    out = predict_reliable(Counting(), "thank you", Q, variants="fast")
    assert seen == [3]
    assert out["answers"]["tone"]["reliability"]["n_variants"] == 3


def test_custom_variant_list_and_bad_names():
    r = predict_reliable(MeaningAgent(), "hi", Q, variants=["letters"])["answers"]["tone"]["reliability"]
    assert set(r["variant_choices"]) == {"original", "letters"}
    with pytest.raises(ValueError):
        predict_reliable(MeaningAgent(), "hi", Q, variants=["nope"])
    with pytest.raises(ValueError):
        predict_reliable(MeaningAgent(), "hi", Q, variants="turbo")
    with pytest.raises(ValueError):
        ReliableAgent(MeaningAgent(), variants=[])


def test_full_is_default_and_matches_old_behaviour():
    a = predict_reliable(MeaningAgent(), "hi", Q)["answers"]["tone"]["reliability"]
    b = predict_reliable(MeaningAgent(), "hi", Q, variants="full")["answers"]["tone"]["reliability"]
    assert a == b and a["n_variants"] >= 5  # up to 8; duplicate shuffles are dropped


def test_calibrate_keeps_a_verify_band():
    # Regression: pooled accuracy stays under 50% up to the accept cutoff,
    # which once made escalate == accept and left no VERIFY band.
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    correct = [False, False, False, True, False, True, False, True, True, True, True, True]
    accept, escalate = calibrate(scores, correct, target_accuracy=0.9)
    assert escalate < accept
    below = [c for s, c in zip(scores, correct) if s < escalate]
    assert not below or sum(below) / len(below) <= 1 / 3
