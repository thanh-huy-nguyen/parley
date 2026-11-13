"""Command line. Parsing and exit codes run on the CPU with no weights; the marked test runs the real models."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from parley import cli


def write_case(tmp_path):
    yy, xx = np.mgrid[:128, :160]
    d = np.hypot(xx - 80, yy - 64)
    Image.fromarray(np.stack([np.clip(200 - 100 * (d < 30), 0, 255).astype(np.uint8)] * 3, -1)).save(tmp_path / "x.png")
    Image.fromarray(((d < 26) * 255).astype(np.uint8)).save(tmp_path / "m.png")
    return tmp_path / "x.png", tmp_path / "m.png"


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip()


def test_missing_weights_gives_exit_code_3(tmp_path, monkeypatch, capsys):
    x, m = write_case(tmp_path)
    monkeypatch.setenv("PARLEY_WEIGHTS", str(tmp_path / "missing.pt"))
    monkeypatch.setattr("parley.weights.DEFAULT_DIR", tmp_path / "nothing")
    assert (
        cli.main(
            [
                "edit",
                "--image",
                str(x),
                "--mask",
                str(m),
                "--text",
                "a",
                "--out",
                str(tmp_path / "o.png"),
                "--device",
                "cpu",
            ]
        )
        == 3
    )
    assert "PARLEY_WEIGHTS" in capsys.readouterr().err


def test_wrong_hash_gives_exit_code_3(tmp_path, capsys):
    x, m = write_case(tmp_path)
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"x")
    assert cli.main(["weights", "--weights", str(bad)]) == 3
    assert "no-verify" in capsys.readouterr().err


def test_weights_no_verify_prints_the_digest(tmp_path, capsys):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"x")
    assert cli.main(["weights", "--weights", str(bad), "--no-verify"]) == 0
    assert "not compared" in capsys.readouterr().out


def _stub(monkeypatch, stub_editor_factory):
    import parley.editor as ED

    real = ED.ParleyEditor
    monkeypatch.setattr(
        ED,
        "ParleyEditor",
        lambda mode="base", seed=0, device="cpu", weights=None, verify=True, min_confidence=None: stub_editor_factory(
            mode, seed, min_confidence=min_confidence
        ),
    )
    return real


def test_edit_writes_a_mask_and_prints_the_program(tmp_path, monkeypatch, stub_editor_factory, capsys):
    x, m = write_case(tmp_path)
    _stub(monkeypatch, stub_editor_factory)
    out = tmp_path / "o.png"
    assert (
        cli.main(
            [
                "edit",
                "--image",
                str(x),
                "--mask",
                str(m),
                "--text",
                "pull the top in a little",
                "--out",
                str(out),
                "--click",
                "10",
                "12",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "transcript: pull the top in a little" in text and "program 1:" in text
    arr = np.array(Image.open(out))
    assert arr.shape == (128, 160) and set(np.unique(arr)) <= {0, 255}


def test_edit_json_output(tmp_path, monkeypatch, stub_editor_factory, capsys):
    x, m = write_case(tmp_path)
    _stub(monkeypatch, stub_editor_factory)
    assert (
        cli.main(
            ["edit", "--image", str(x), "--mask", str(m), "--text", "a", "--out", str(tmp_path / "o.png"), "--json"]
        )
        == 0
    )
    assert set(json.loads(capsys.readouterr().out)) >= {
        "transcript",
        "status",
        "programs",
        "changed_fraction",
        "output",
    }


def test_empty_text_is_a_no_op_with_a_warning(tmp_path, monkeypatch, stub_editor_factory, capsys):
    x, m = write_case(tmp_path)
    _stub(monkeypatch, stub_editor_factory)
    out = tmp_path / "o.png"
    assert cli.main(["edit", "--image", str(x), "--mask", str(m), "--text", "", "--out", str(out)]) == 0
    cap = capsys.readouterr()
    assert "status: no_input" in cap.out and "empty transcript" in cap.err
    assert np.array_equal(np.array(Image.open(out)), np.array(Image.open(m)))


def test_state_file_is_written_and_read(tmp_path, monkeypatch, stub_editor_factory):
    x, m = write_case(tmp_path)
    _stub(monkeypatch, stub_editor_factory)
    s = tmp_path / "state.npz"
    args = [
        "edit",
        "--image",
        str(x),
        "--mask",
        str(m),
        "--text",
        "a",
        "--out",
        str(tmp_path / "o.png"),
        "--mode",
        "listener",
        "--state",
        str(s),
    ]
    assert cli.main(args) == 0 and s.exists()
    assert cli.main(args) == 0


def test_bad_mask_path_gives_exit_code_2(tmp_path, monkeypatch, stub_editor_factory, capsys):
    x, _ = write_case(tmp_path)
    _stub(monkeypatch, stub_editor_factory)
    assert (
        cli.main(
            [
                "edit",
                "--image",
                str(x),
                "--mask",
                str(tmp_path / "none.png"),
                "--text",
                "a",
                "--out",
                str(tmp_path / "o.png"),
            ]
        )
        == 2
    )
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.weights
def test_cli_with_real_models(tmp_path):
    """Runs the installed entry point on a synthetic image, on the device named by PARLEY_TEST_DEVICE (default cpu)."""
    x, m = write_case(tmp_path)
    out = tmp_path / "o.png"
    env = dict(os.environ, PARLEY_PROCTITLE="parley-cli-test")
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "parley.cli",
            "edit",
            "--image",
            str(x),
            "--mask",
            str(m),
            "--text",
            "pull the top in a little",
            "--out",
            str(out),
            "--device",
            os.environ.get("PARLEY_TEST_DEVICE", "cpu"),
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout)
    assert (
        res["programs"][0]["type"] == "shrink"
        and res["programs"][0]["scope"] == "top"
        and res["programs"][0]["sign"] == "shrink"
    )
    assert np.array(Image.open(out)).shape == (128, 160)
