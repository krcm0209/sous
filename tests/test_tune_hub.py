import json
from pathlib import Path
from types import SimpleNamespace

from sous.tune.hub import (
    Download,
    _cached_path,
    ask_consent,
    fetch,
    is_cached,
    plan_downloads,
    snapshot_bytes,
)


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
    # Only the weights count, on disk as on the Hub branch: config.json's 2
    # bytes must not inflate the figure the fit arithmetic budgets against.
    assert snapshot_bytes("x/y", api=api, cached_path=lambda rid: tmp_path) == 100


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


def test_consent_prints_the_block_up_front_then_asks_each_separately():
    plan = [
        Download("org/big", 20 * 10**9, "candidate (35b-moe tier)", "its arms"),
        Download("org/draft", 8 * 10**8, "drafter for org/big", "org/big runs without a drafter"),
    ]
    events: list[tuple[str, str]] = []
    answers = iter(["y", "n"])

    def out(*args, **kwargs):
        events.append(("out", " ".join(str(a) for a in args)))

    def ask(prompt):
        events.append(("ask", prompt))
        return next(answers)

    approved = ask_consent(
        plan, free_bytes=412 * 2**30, hub_cache="/mnt/data/hub", ask=ask, out=out
    )
    # Every line of the block precedes the first question, so the user
    # decides each download knowing the whole plan.
    assert [kind for kind, _ in events] == ["out", "out", "out", "ask", "ask"]
    block = "\n".join(text for kind, text in events if kind == "out")
    assert "20.0 GB" in block and "0.8 GB" in block and "412.0 GiB" in block
    assert "/mnt/data/hub" in block
    assert "org/big runs without a drafter" in block
    assert events[3][1].startswith("Download #1") and "org/big" in events[3][1]
    assert events[4][1].startswith("Download #2") and "org/draft" in events[4][1]
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


def _cache_snapshot(
    cache_dir: Path, repo_id: str, files: dict[str, bytes], sha: str = "a" * 40
) -> Path:
    """A slice of a real Hugging Face cache layout for repo_id at sha: a
    refs/main file holding the commit sha as plain text, and the given files
    under snapshots/<sha>/ — what try_to_load_from_cache actually reads."""
    org, name = repo_id.split("/")
    repo_dir = cache_dir / f"models--{org}--{name}"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(sha)
    snapshot = repo_dir / "snapshots" / sha
    snapshot.mkdir(parents=True)
    for name_, content in files.items():
        (snapshot / name_).write_bytes(content)
    return snapshot


def _index(*shard_names: str) -> bytes:
    weight_map = {f"tensor.{i}": name for i, name in enumerate(shard_names)}
    return json.dumps({"weight_map": weight_map}).encode()


def test_cached_path_is_none_for_a_metadata_only_snapshot(tmp_path):
    # candidates.describe() fetches config.json alone to size a candidate;
    # that lookup must never itself read back as a cached, loadable model.
    _cache_snapshot(tmp_path, "org/model", {"config.json": b"{}"})
    assert _cached_path("org/model", cache_dir=tmp_path) is None


def test_cached_path_finds_a_single_file_checkpoint(tmp_path):
    snapshot = _cache_snapshot(
        tmp_path, "org/model", {"config.json": b"{}", "model.safetensors": b"x" * 10}
    )
    assert _cached_path("org/model", cache_dir=tmp_path) == snapshot


def test_cached_path_is_none_when_an_indexed_shard_is_missing(tmp_path):
    shards = (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    _cache_snapshot(
        tmp_path,
        "org/model",
        {
            "config.json": b"{}",
            "model.safetensors.index.json": _index(*shards),
            shards[0]: b"x",
            shards[1]: b"x",
            # shards[2] is never written: the shard list names it, disk lacks it.
        },
    )
    assert _cached_path("org/model", cache_dir=tmp_path) is None


def test_cached_path_finds_a_sharded_checkpoint_with_no_readme(tmp_path):
    # mlx-vlm downloads with allow_patterns, so a fully usable snapshot it
    # produced never has a README.md; the Hub's own completeness check
    # disagrees and this must not.
    shards = (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    snapshot = _cache_snapshot(
        tmp_path,
        "org/model",
        {
            "config.json": b"{}",
            "model.safetensors.index.json": _index(*shards),
            **{name: b"x" for name in shards},
        },
    )
    assert _cached_path("org/model", cache_dir=tmp_path) == snapshot


def test_cached_path_is_none_for_an_absent_repo(tmp_path):
    assert _cached_path("org/nope", cache_dir=tmp_path) is None
