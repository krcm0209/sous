from types import SimpleNamespace

from sous.tune.hub import Download, ask_consent, fetch, is_cached, plan_downloads, snapshot_bytes


class _Api:
    def __init__(self, files):
        self.files = files

    def model_info(self, repo_id, files_metadata=False):
        assert files_metadata is True
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in self.files.items()]
        )


def test_snapshot_bytes_sums_the_hubs_safetensors_sizes_when_not_cached():
    api = _Api({"model.safetensors": 10, "model-2.safetensors": 5, "config.json": 999})
    assert snapshot_bytes("x/y", api=api, cached_path=lambda rid: None) == 15


def test_snapshot_bytes_prefers_the_on_disk_size_of_a_cached_snapshot(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "config.json").write_bytes(b"{}")
    api = _Api({"model.safetensors": 10})
    assert snapshot_bytes("x/y", api=api, cached_path=lambda rid: tmp_path) == 102


def test_snapshot_bytes_is_none_offline():
    class Down:
        def model_info(self, *a, **k):
            raise OSError("no network")

    assert snapshot_bytes("x/y", api=Down(), cached_path=lambda rid: None) is None


def test_is_cached_follows_the_local_only_lookup(tmp_path):
    assert is_cached("x/y", cached_path=lambda rid: tmp_path) is True
    assert is_cached("x/y", cached_path=lambda rid: None) is False


def test_plan_collapses_duplicates_drops_cached_and_keeps_order():
    plan = plan_downloads(
        [
            ("org/big", "candidate (27b-dense tier)", "its arms"),
            ("org/draft", "drafter for org/big", "org/big runs without a drafter"),
            ("org/big", "reference", "its arms"),
            ("org/cached", "candidate (9b tier)", "its arms"),
        ],
        cached=lambda rid: rid == "org/cached",
        size=lambda rid: {"org/big": 20 * 10**9, "org/draft": None}[rid],
    )
    assert [d.repo_id for d in plan] == ["org/big", "org/draft"]
    assert plan[0] == Download("org/big", 20 * 10**9, "candidate (27b-dense tier)", "its arms")
    assert plan[1].bytes is None


def test_consent_prints_the_block_up_front_then_asks_each_separately(capsys):
    plan = [
        Download("org/big", 20 * 10**9, "candidate (35b-moe tier)", "its arms"),
        Download("org/draft", 8 * 10**8, "drafter for org/big", "org/big runs without a drafter"),
    ]
    answers = iter(["y", "n"])
    approved = ask_consent(plan, free_bytes=412 * 2**30, ask=lambda prompt: next(answers))
    out = capsys.readouterr().out
    assert out.index("org/big") < out.index("org/draft") < out.index("Download #1?")
    assert "20.0 GB" in out and "0.8 GB" in out and "412.0 GiB" in out
    assert "org/big runs without a drafter" in out
    assert approved == {"org/big"}


def test_consent_treats_eof_and_anything_but_yes_as_no():
    plan = [Download("a/b", 1, "candidate (2b tier)", "its arms")]

    def eof(prompt):
        raise EOFError

    assert ask_consent(plan, free_bytes=1, ask=eof, out=lambda *a, **k: None) == set()
    assert ask_consent(plan, free_bytes=1, ask=lambda p: "maybe", out=lambda *a, **k: None) == set()
    assert ask_consent(plan, free_bytes=1, ask=lambda p: "YES", out=lambda *a, **k: None) == {"a/b"}


def test_fetch_downloads_each_approved_snapshot_once(capsys):
    seen = []
    fetch(["a/b", "c/d", "a/b"], download=lambda rid: seen.append(rid))
    assert seen == ["a/b", "c/d"]
    assert "a/b" in capsys.readouterr().out
