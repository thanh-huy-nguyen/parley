"""The evaluation kit. CPU, no weights."""

import numpy as np
import pytest

from parley import edits as E
from parley.evalkit import (
    Case,
    DialogueRecord,
    TimeCosts,
    cluster_bootstrap,
    gain,
    gain_stop,
    inject_word_errors,
    jitter_click,
    reach,
    run_dialogue,
    run_study,
    simulated_time,
    summarize,
    summary_markdown,
)
from parley.evalkit.clinician import SimulatedClinician
from parley.evalkit.toy import ToyClinician


def rec(ious, words=None, clicks=None, clinician="c", case="i"):
    n = len(ious) - 1
    return DialogueRecord(clinician, case, "d", "a", ious=list(ious), words=words or [5] * n, clicks=clicks or [0] * n)


def disk(r, cx=100, cy=100, n=200):
    yy, xx = np.mgrid[:n, :n]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def test_reach_at_any_turn_including_a_later_fall():
    assert reach(rec([0.5, 0.9, 0.6]))
    assert not reach(rec([0.5, 0.84]))
    assert reach(rec([0.5, 0.85]))


def test_reach_stops_at_the_turn_limit():
    assert not reach(rec([0.5] * 11 + [0.9]), max_turns=10)
    assert reach(rec([0.5] * 10 + [0.9]), max_turns=10)


def test_reach_counts_a_good_start_and_the_subset_can_exclude_it():
    r = rec([0.9])
    assert reach(r)
    assert summarize([r, rec([0.5, 0.6])], needs_correction=True, n_boot=20)["n"] == 1


def test_gain_is_last_minus_start_even_after_a_peak():
    assert gain(rec([0.5, 0.9, 0.6])) == pytest.approx(0.1)


def test_simulated_time_uses_the_stated_costs():
    r = rec([0.1, 0.2, 0.3], words=[5, 10], clicks=[1, 0])
    assert simulated_time(r) == pytest.approx((0.4 * 5 + 0.5 + 2.0 + 4.0) + (0.4 * 10 + 0.5 + 4.0))


def test_click_only_turn_has_no_recognizer_cost():
    assert simulated_time(rec([0.1, 0.2], words=[0], clicks=[1])) == pytest.approx(2.0 + 4.0)


def test_costs_can_be_changed_and_time_can_stop_at_reach():
    r = rec([0.1, 0.9, 0.95], words=[1, 1], clicks=[0, 0])
    assert simulated_time(r, TimeCosts(per_word=1, recognizer=0, per_click=0, review=1)) == pytest.approx(4)
    assert simulated_time(
        r, TimeCosts(per_word=1, recognizer=0, per_click=0, review=1), until_reach=0.85
    ) == pytest.approx(2)


def test_summary_shares_and_means():
    rows = [rec([0.5, 0.9], case="a"), rec([0.5, 0.4], case="b"), rec([0.5, 0.5], case="c"), rec([0.5, 0.6], case="d")]
    s = summarize(rows, n_boot=50)
    assert s["n"] == 4 and s["images"] == 4 and s["clinicians"] == 1
    assert s["improved"]["mean"] == 0.5 and s["worse"]["mean"] == 0.25 and s["reach"]["mean"] == 0.25
    assert s["gain"]["mean"] == pytest.approx(0.1)
    assert s["gain"]["lo"] <= s["gain"]["mean"] <= s["gain"]["hi"]
    assert "| a |" in summary_markdown(dict(a=s))


def test_bootstrap_is_seeded_and_collapses_with_one_cluster():
    assert cluster_bootstrap([0.5, 0.5], ["c", "c"], ["i", "i"], n_boot=50) == (0.5, 0.5)
    v, c, i = np.arange(30.0), [str(k % 5) for k in range(30)], [str(k % 6) for k in range(30)]
    assert cluster_bootstrap(v, c, i, n_boot=100, seed=3) == cluster_bootstrap(v, c, i, n_boot=100, seed=3)


def test_bootstrap_widens_when_values_are_clustered():
    """Dialogues of one clinician that share a clinician effect: the interval must be wider than a bootstrap that treats
    dialogues as independent.
    """
    rng = np.random.default_rng(0)
    n_c, n_i = 12, 20
    effect = rng.normal(0, 1, n_c)
    v, c, i = [], [], []
    for a in range(n_c):
        for b in range(n_i):
            v.append(effect[a] + rng.normal(0, 0.1))
            c.append(a)
            i.append(b)
    lo, hi = cluster_bootstrap(np.array(v), c, i, n_boot=400)
    ilo, ihi = cluster_bootstrap(np.array(v), list(range(len(v))), [0] * len(v), n_boot=400)
    assert (hi - lo) > 3 * (ihi - ilo)


def test_bootstrap_covers_the_truth_on_two_way_data():
    """Coverage of the clinician-and-image interval on data with both effects, over repeated studies."""
    hits = 0
    for s in range(60):
        rng = np.random.default_rng(100 + s)
        n_c, n_i = 15, 15
        ce, ie = rng.normal(0, 1, n_c), rng.normal(0, 1, n_i)
        v = np.array([ce[a] + ie[b] + rng.normal(0, 1) for a in range(n_c) for b in range(n_i)])
        c = [a for a in range(n_c) for _ in range(n_i)]
        i = [b for _ in range(n_c) for b in range(n_i)]
        lo, hi = cluster_bootstrap(v, c, i, n_boot=200, seed=s)
        hits += lo <= 0 <= hi
    # At least 80 percent in 60 runs at a nominal 95 percent: the two-way bootstrap is known to be slightly
    # conservative or liberal in small samples.
    assert hits >= 48


def test_word_errors_hit_about_the_requested_rate():
    rng = np.random.default_rng(0)
    text = "pull the top edge in a little bit please " * 20
    assert inject_word_errors(text, 0.0, rng) == " ".join(text.split())
    changed = sum(a != b for a, b in zip(text.split(), inject_word_errors(text, 0.3, rng).split()))
    assert changed > 0
    with pytest.raises(ValueError):
        inject_word_errors(text, 2.0, rng)


def test_click_jitter_and_drop():
    rng = np.random.default_rng(0)
    assert jitter_click(None, 3, rng) is None
    assert jitter_click((5, 5), 3, rng, drop=1.0) is None
    assert jitter_click((5, 5), 0, rng) == (5.0, 5.0)


class Scripted:
    """An editor that shrinks or grows the mask toward nothing in particular, for runner tests."""

    def __init__(self, amount=0.05):
        self.amount, self.ended, self.sessions = amount, [], []

    def begin_session(self, clinician_id, dataset):
        self.sessions.append((clinician_id, dataset))

    def begin_image(self, case):
        return case.start_mask

    def step(self, mask, text, click):
        return E.apply(mask, E.Edit("shrink", "global", amount=-self.amount)), dict(types=["shrink"])

    def end_image(self, mask, reached, accept_text):
        self.ended.append((reached, accept_text))


def case(r_start=30, r_true=40, cid="x"):
    return Case(cid, "d", np.zeros((200, 200, 3), np.uint8), disk(r_start), disk(r_true))


def test_runner_stops_at_accept_and_reports_it():
    ed = Scripted()
    r = run_dialogue(ed, ToyClinician(), case())
    assert r.accepted and r.ious[-1] >= 0.85 and ed.ended == [(True, "that looks good")]
    assert len(r.ious) == len(r.words) + 1 == len(r.clicks) + 1


def test_runner_respects_the_turn_cap():
    ed = Scripted(amount=0.0)
    r = run_dialogue(ed, ToyClinician(), case(), max_turns=3)
    assert not r.accepted and len(r.words) == 3 and ed.ended == [(False, None)]


def test_runner_accepts_a_good_start_with_no_turns():
    r = run_dialogue(Scripted(), ToyClinician(), case(40, 40))
    assert r.accepted and r.words == [] and r.ious == [1.0]


class GiveUp(SimulatedClinician):
    def target(self, case):
        return case.reference

    def accept(self, obs):
        return False

    def next_utterance(self, obs):
        return None

    def click(self, obs, utterance):
        return None


def test_runner_records_giving_up():
    r = run_dialogue(Scripted(), GiveUp(), case())
    assert r.gave_up and not r.accepted and r.words == []


def test_abstract_clinician_cannot_be_built_half_way():
    class Half(SimulatedClinician):
        def target(self, case):
            return case.reference

    with pytest.raises(TypeError):
        Half()


def test_study_runs_one_session_per_clinician_and_dataset():
    ed = Scripted()
    rows = run_study(
        ed,
        [ToyClinician("a"), ToyClinician("b", margin=0.02)],
        dict(d1=[case(cid="1"), case(cid="2")], d2=[case(cid="3")]),
        max_turns=2,
    )
    assert len(rows) == 6 and ed.sessions == [("a", "d1"), ("a", "d2"), ("b", "d1"), ("b", "d2")]
    assert summarize(rows, n_boot=20)["clinicians"] == 2


def test_toy_clinician_asks_for_the_right_direction():
    sim = ToyClinician()
    c = case(30, 40)
    from parley.evalkit.clinician import Observation

    assert sim.next_utterance(Observation(c, disk(36), 0)).startswith("push")
    assert sim.next_utterance(Observation(c, disk(46), 0)).startswith("bring")
    assert sim.next_utterance(Observation(c, np.zeros((200, 200), bool), 0)) is not None


def test_toy_clinician_clicks_on_a_missing_piece():
    from parley.evalkit.clinician import Observation

    two = disk(25, 60, 60) | disk(25, 140, 140)
    c = Case("x", "d", np.zeros((200, 200, 3), np.uint8), disk(25, 60, 60), two)
    sim = ToyClinician()
    obs = Observation(c, disk(25, 60, 60), 0)
    u = sim.next_utterance(obs)
    x, y = sim.click(obs, u)
    assert "missing" in u and abs(x - 140) < 3 and abs(y - 140) < 3


def test_toy_module_says_it_is_an_example():
    import parley.evalkit.toy as T

    assert "EXAMPLE ONLY" in T.__doc__ and "EXAMPLE ONLY" in T.ToyClinician.__doc__


def test_giving_up_is_charged_the_remaining_turns():
    took_one = rec([0.5, 0.6], words=[5], clicks=[0])
    cost = 0.4 * 5 + 0.5 + 4.0
    assert simulated_time(took_one) == pytest.approx(cost)
    took_one.gave_up = True
    assert simulated_time(took_one) == pytest.approx(10 * cost)
    assert simulated_time(took_one, max_turns=4) == pytest.approx(4 * cost)
    went_to_the_cap = rec([0.5] * 11, words=[5] * 10, clicks=[0] * 10)
    assert summarize([took_one], n_boot=10)["time_s"]["mean"] == pytest.approx(simulated_time(went_to_the_cap))


def test_giving_up_before_any_turn_is_charged_the_review():
    r = rec([0.5])
    r.gave_up = True
    assert simulated_time(r) == pytest.approx(10 * 4.0)


def test_giving_up_after_reaching_stops_at_the_reach_when_asked():
    r = rec([0.5, 0.9, 0.7], words=[5, 5], clicks=[0, 0])
    r.gave_up = True
    assert simulated_time(r, until_reach=0.85) == pytest.approx(0.4 * 5 + 0.5 + 4.0)


class OffsetEditor:
    """Shows a larger mask than it was given, as an editor with a start offset does."""

    def __init__(self, shown):
        self.shown = shown

    def begin_session(self, clinician_id, dataset):
        pass

    def begin_image(self, case):
        return self.shown

    def step(self, mask, text, click):
        return mask, {}

    def end_image(self, mask, reached, accept_text):
        pass


class SilentClinician(SimulatedClinician):
    clinician_id = "silent"
    accept_phrase = "fine"

    def __init__(self, truth):
        self.truth = truth

    def target(self, case):
        return self.truth

    def accept(self, obs):
        return False

    def next_utterance(self, obs):
        return None

    def click(self, obs, utterance):
        return None


def test_slice_and_gain_use_the_given_start_not_the_offset_mask():
    truth, given, shown = disk(50), disk(35), disk(49)
    case = Case("i", "d", np.zeros((200, 200, 3), np.uint8), given, truth)
    r = run_dialogue(OffsetEditor(shown), SilentClinician(truth), case)
    assert r.start_iou < 0.85 <= r.ious[0]
    assert gain(r) == pytest.approx(r.ious[0] - r.start_iou) and gain(r) > 0
    s = summarize([r], needs_correction=True, n_boot=10)
    assert s["n"] == 1 and s["reach"]["mean"] == 1.0
    r.start_iou, r.ious = r.ious[0], [0.5]
    assert summarize([r], needs_correction=True, n_boot=10)["n"] == 0


def test_gain_stop_takes_the_first_mask_at_the_threshold_else_the_last():
    assert gain_stop(rec([0.5, 0.9, 0.6])) == pytest.approx(0.4)
    assert gain_stop(rec([0.5, 0.7, 0.6])) == pytest.approx(0.1)


def test_time_to_reach_charges_a_dialogue_accepted_below_the_threshold_to_the_cap():
    r = rec([0.5, 0.7], words=[5], clicks=[0])
    r.accepted = True
    cost = 0.4 * 5 + 0.5 + 4.0
    assert simulated_time(r) == pytest.approx(cost)
    assert simulated_time(r, until_reach=0.85) == pytest.approx(10 * cost)
    s = summarize([r], n_boot=10)
    assert s["time_s"]["mean"] == pytest.approx(cost) and s["time_reach"]["mean"] == pytest.approx(10 * cost)


def test_time_to_reach_stops_at_the_first_reach_even_after_a_give_up():
    r = rec([0.5, 0.9, 0.7], words=[5, 5], clicks=[0, 0])
    r.gave_up = True
    cost = 0.4 * 5 + 0.5 + 4.0
    assert simulated_time(r, until_reach=0.85) == pytest.approx(cost)
    assert simulated_time(r) == pytest.approx(10 * cost)
    assert summarize([r], n_boot=10)["time_reach"]["mean"] == pytest.approx(cost)


def test_summary_table_shows_both_times_and_the_stopping_gain():
    table = summary_markdown(
        {"a": summarize([rec([0.5, 0.9, 0.6])], n_boot=10), "b": dict(n=0, clinicians=0, images=0)}
    )
    head, rule, a, b = table.split("\n")
    assert "Gain, stopping rule" in head and "Time spent" in head and "Time to reach" in head
    assert head.count("|") == rule.count("|") == a.count("|") == b.count("|")
