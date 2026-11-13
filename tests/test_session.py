"""Protocol, undo and session state on a random-weight network with stub encoders. CPU, no weights."""

import json
import warnings

import numpy as np
import pytest
import torch

from parley import listener as L
from parley import weights as W
from parley.errors import SessionStateError, WeightsHashMismatchError, WeightsNotFoundError


def begin(ed, phantom):
    rgb, mask = phantom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ed.begin_image(rgb, mask)


def add_shrink_event(ed, mask, sign="shrink", size="medium"):
    ed.engine.listener.observe_turn(mask, dict(type="shrink", scope="global", sign=sign, size=size))


def test_result_fields_and_shape(stub_editor_factory, phantom):
    ed = stub_editor_factory()
    begin(ed, phantom)
    r = ed.edit(text="make it a bit bigger", click=(10, 10))
    assert r.mask.shape == phantom[1].shape and r.mask.dtype == bool
    assert r.status in ("edited", "unchanged", "accepted")
    assert set(r.programs[0]) >= {
        "type",
        "scope",
        "place",
        "move_dir",
        "sign",
        "size",
        "amount_D",
        "move_D",
        "confidence",
    }


def test_undo_restores_mask_and_belief(stub_editor_factory, phantom):
    ed = stub_editor_factory("listener")
    m0 = begin(ed, phantom).copy()
    logp0 = ed.engine.listener.belief.logp.copy()
    ed.edit(text="make it a bit bigger")
    add_shrink_event(
        ed, ed.current, "grow", "large"
    )  # guarantees that the belief moved even if the random network parsed something else
    assert ed.can_undo
    assert (ed.undo() == m0).all()
    assert (ed.current == m0).all() and np.array_equal(ed.engine.listener.belief.logp, logp0)
    assert ed.undo() is None
    assert ed.engine.turns == []


def test_low_confidence_does_not_run_or_teach(stub_editor_factory, phantom):
    ed = stub_editor_factory("listener", min_confidence=1.01)
    m0 = begin(ed, phantom).copy()
    logp0 = ed.engine.listener.belief.logp.copy()
    r = ed.edit(text="make it a bit bigger")
    assert r.status == "low_confidence" and (r.mask == m0).all() and not ed.can_undo
    assert np.array_equal(ed.engine.listener.belief.logp, logp0) and ed.engine.turns == []


def test_non_edit_turn_does_not_reach_the_adapters(stub_editor_factory, phantom):
    ed = stub_editor_factory("adapter")
    begin(ed, phantom)
    ed.edit(text="hello there", is_edit=False)
    assert ed.engine.turns == []
    ed.edit(text="make it bigger")
    assert len(ed.engine.turns) == 1


def test_listener_state_round_trip_is_exact(stub_editor_factory, phantom, tmp_path):
    a = stub_editor_factory("listener")
    a.begin_session("c7", "ultrasound")
    begin(a, phantom)
    add_shrink_event(a, phantom[1], "grow", "large")
    a.engine.listener.n_images = 3
    meta = a.save_state(tmp_path / "s.npz")
    assert meta["clinician_id"] == "c7" and meta["dataset"] == "ultrasound"
    b = stub_editor_factory("listener")
    b.load_state(tmp_path / "s.npz")
    assert np.array_equal(a.engine.listener.belief.logp, b.engine.listener.belief.logp)
    assert b.clinician_id == "c7" and b.engine.listener.n_images == 3
    assert b.belief()["mean"] == a.belief()["mean"]
    begin(b, phantom)
    begin(a, phantom)
    add_shrink_event(a, phantom[1], "shrink", "small")
    add_shrink_event(b, phantom[1], "shrink", "small")
    assert np.array_equal(a.engine.listener.belief.logp, b.engine.listener.belief.logp)


def test_begin_session_resets_the_listener(stub_editor_factory, phantom):
    ed = stub_editor_factory("listener")
    ed.begin_session("c", "a")
    begin(ed, phantom)
    add_shrink_event(ed, phantom[1], "grow", "large")
    assert abs(ed.belief()["mean"]) > 0
    ed.begin_session("c", "b")
    assert ed.belief()["mean"] == pytest.approx(L.MarginBelief().mean()) and ed.belief()["n_events"] == 0


def test_a_new_dataset_refuses_old_state(stub_editor_factory, phantom, tmp_path):
    a = stub_editor_factory("listener")
    a.begin_session("c", "ultrasound")
    a.save_state(tmp_path / "s.npz")
    b = stub_editor_factory("listener")
    with pytest.raises(SessionStateError, match="new session"):
        b.load_state(tmp_path / "s.npz", dataset="colonoscopy")
    b.load_state(tmp_path / "s.npz", dataset="ultrasound")


def test_state_mode_must_match(stub_editor_factory, tmp_path):
    a = stub_editor_factory("listener")
    a.begin_session()
    a.save_state(tmp_path / "s.npz")
    with pytest.raises(SessionStateError, match="mode"):
        stub_editor_factory("base").load_state(tmp_path / "s.npz")


def test_bad_state_file_is_a_clear_error(stub_editor_factory, tmp_path):
    p = tmp_path / "bad.npz"
    p.write_bytes(b"not a state file")
    with pytest.raises(SessionStateError):
        stub_editor_factory("base").load_state(p)


def test_adapter_state_round_trip_is_exact(stub_editor_factory, phantom, tmp_path):
    a = stub_editor_factory("adapter")
    a.begin_session("c", "d")
    with torch.no_grad():
        a.engine.adapters.mods[0].B.add_(0.25)
    a.save_state(tmp_path / "s.npz")
    b = stub_editor_factory("adapter")
    b.load_state(tmp_path / "s.npz")
    for p, q in zip(a.engine.adapters.params, b.engine.adapters.params):
        assert torch.equal(p, q)
    c = stub_editor_factory("adapter", seed=1)
    c.seed = 1
    with pytest.raises(SessionStateError, match="seed"):
        c.load_state(tmp_path / "s.npz")


def test_adapter_fit_on_an_accepted_image_changes_the_matrices(stub_editor_factory, phantom):
    ed = stub_editor_factory("adapter")
    ed.begin_session()
    m = begin(ed, phantom)
    typ = ed.engine.net.typ
    with torch.no_grad():  # force a shrink parse on the random network
        typ.bias[:] = -100.0
        typ.bias[3] = 100.0
    ed.edit(text="pull it in")
    before = [p.clone() for p in ed.engine.adapters.params]
    ed.end_image(m, accepted=True, accept_text=None)
    assert ed.engine.adapters.n_fits == 1 and any(
        not torch.equal(a, b) for a, b in zip(before, ed.engine.adapters.params)
    )


def test_unaccepted_image_teaches_nothing(stub_editor_factory, phantom):
    ed = stub_editor_factory("adapter")
    ed.begin_session()
    m = begin(ed, phantom)
    ed.edit(text="pull it in")
    before = [p.clone() for p in ed.engine.adapters.params]
    ed.end_image(m, accepted=False)
    assert all(torch.equal(a, b) for a, b in zip(before, ed.engine.adapters.params))


def test_compound_second_slot_is_described(stub_editor_factory, phantom):
    ed = stub_editor_factory()
    begin(ed, phantom)
    with torch.no_grad():
        ed.engine.net.typ.bias[:] = -100.0
        ed.engine.net.typ.bias[3] = 100.0
        ed.engine.net.present.bias[:] = 100.0
    r = ed.edit(text="pull it in, and shift it left")
    assert len(r.programs) == 2


def test_weights_lookup_order(tmp_path, monkeypatch):
    env_dir = tmp_path / "env"
    arg_dir = tmp_path / "arg"
    env_dir.mkdir()
    arg_dir.mkdir()
    (env_dir / W.file_name(0)).write_bytes(b"e")
    (arg_dir / W.file_name(0)).write_bytes(b"a")
    monkeypatch.setenv(W.ENV_WEIGHTS, str(env_dir))
    assert W.locate(0, arg_dir) == arg_dir / W.file_name(0)
    assert W.locate(0) == env_dir / W.file_name(0)
    assert W.locate(0, arg_dir / W.file_name(0)) == arg_dir / W.file_name(0)
    monkeypatch.setenv(W.ENV_WEIGHTS, str(tmp_path / "missing"))
    monkeypatch.setattr(W, "DEFAULT_DIR", tmp_path / "nothing")
    with pytest.raises(WeightsNotFoundError, match="PARLEY_WEIGHTS"):
        W.locate(0)


def test_hash_check_stops_and_explains_the_opt_out(tmp_path, monkeypatch):
    p = tmp_path / W.file_name(0)
    p.write_bytes(b"not the released weights")
    monkeypatch.delenv(W.ENV_NO_VERIFY, raising=False)
    with pytest.raises(WeightsHashMismatchError, match="retrained"):
        W.resolve(0, p)
    assert W.resolve(0, p, verify_hash=False) == (p, None)
    monkeypatch.setenv(W.ENV_NO_VERIFY, "1")
    assert W.resolve(0, p) == (p, None)
    monkeypatch.setenv(W.ENV_NO_VERIFY, "0")
    with pytest.raises(WeightsHashMismatchError):
        W.resolve(0, p)


def test_manifest_lists_three_seeds_and_three_models():
    m = W.manifest()
    assert set(m["weights"]) == {"0", "1", "2"} and all(len(v["sha256"]) == 64 for v in m["weights"].values())
    assert set(m["hf"]) == {W.SAM2_REPO, W.TEXT_REPO, W.ASR_REPO}


def test_state_file_needs_no_pickle(stub_editor_factory, tmp_path):
    a = stub_editor_factory("listener")
    a.begin_session("c", "d")
    a.save_state(tmp_path / "s.npz")
    z = np.load(tmp_path / "s.npz", allow_pickle=False)
    assert json.loads(str(z["meta"]))["format"] == 1
