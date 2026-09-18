import json

import pytest

from sous.tune.rundir import RunDir


def test_new_run_dir_is_named_by_the_clock_and_created(tmp_path):
    rd = RunDir.new(tmp_path / "tune", clock=lambda: 1_800_000_000.0)
    assert rd.path.is_dir() and rd.path.parent == tmp_path / "tune"
    assert rd.run_id == rd.path.name and len(rd.run_id) == 15  # YYYYmmdd-HHMMSS


def test_rows_round_trip_by_kind_and_survive_a_reopen(tmp_path):
    rd = RunDir.new(tmp_path / "tune")
    rd.append("bench", {"label": "a", "decode_tps_1k": 25.5})
    rd.append("bench", {"label": "b", "decode_tps_1k": None})
    rd.append("note", {"text": "x"})
    again = RunDir.existing(tmp_path / "tune", rd.run_id)
    assert again.rows("bench") == [
        {"label": "a", "decode_tps_1k": 25.5},
        {"label": "b", "decode_tps_1k": None},
    ]
    assert again.rows("note") == [{"text": "x"}]
    lines = (rd.path / "results.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["kind"] == "bench"


def test_existing_refuses_an_unknown_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunDir.existing(tmp_path / "tune", "nope")


def test_json_and_text_files_land_in_the_run_dir(tmp_path):
    rd = RunDir.new(tmp_path / "tune")
    p = rd.write_json("hardware.json", {"chip": "M2"})
    assert json.loads(p.read_text()) == {"chip": "M2"}
    assert rd.write_text("report.md", "# hi\n").read_text() == "# hi\n"
