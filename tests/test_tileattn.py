"""The attention tile (sous.engine.tileattn).

The schedule is pure Python. The kernel entries run on any Metal GPU with a
plain-mlx stand-in for the Metal kernel, which CI's GPU (macos-15) cannot
compile; the `nax`-marked tests run the real kernel where the tensor units are.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import tileattn  # noqa: E402 — after the importorskip guard

N0 = tileattn.N0
SPLIT_TARGETS = [10, 16, 20, 40]


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_below_n0_is_the_stock_path(splits):
    assert tileattn.chunk_for(N0 - 1, splits) == 0
    assert tileattn.chunk_for(N0, splits) > 0
    assert tileattn.step_of(N0 - 1, splits) == (1, N0 - 1, 0)


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_every_step_past_the_first_is_32_s_keys_wide(splits):
    marks = tileattn.boundaries(262_144, splits)
    for mark in marks[1:]:
        first, last, chunk = tileattn.step_of(mark, splits)
        assert (first, last - first + 1) == (mark, 32 * splits), mark
        assert chunk == tileattn.chunk_for(mark, splits) == tileattn.chunk_for(last, splits)
        assert tileattn.chunk_for(mark - 1, splits) == chunk - tileattn.GRAN


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_no_row_gets_more_than_s_splits(splits):
    for n in range(N0, 262_145):
        assert -(-n // tileattn.chunk_for(n, splits)) <= splits, n


def test_twenty_splits_from_the_tail_of_step_nineteen():
    assert -(-11_552 // tileattn.chunk_for(11_552, 20)) == 19
    assert all(-(-n // tileattn.chunk_for(n, 20)) == 20 for n in range(11_553, 262_145))


@pytest.mark.parametrize(
    ("splits", "first_four"),
    [
        (20, [1536, 1921, 2561, 3201]),
        (10, [1536, 1601, 1921, 2241]),
        (16, [1536, 1537, 2049, 2561]),  # N0 is a step of one key at 16 splits
        (40, [1536, 2561, 3841, 5121]),
    ],
)
def test_boundaries_start_at_n0(splits, first_four):
    assert tileattn.boundaries(262_144, splits)[:4] == first_four


def test_the_m5_pros_schedule_has_408_steps_to_262k():
    assert len(tileattn.boundaries(262_144, 20)) == 408


def test_step_of_either_side_of_the_57600_boundary():
    assert tileattn.step_of(57_600, 20) == (56_961, 57_600, 2_880)
    assert tileattn.step_of(57_601, 20) == (57_601, 58_240, 2_912)


@pytest.mark.parametrize(
    ("prefix", "t", "runs"),
    [
        # rows 1531..1535 are stock; 1536..1543 fill a run; 1544..1546 continue the step
        (1530, 16, [(0, 5, 0), (5, 13, 96), (13, 16, 96)]),
        # a run cut at 8 rows, then the step boundary at 57601 inside the call
        (57_590, 16, [(0, 8, 2880), (8, 10, 2880), (10, 16, 2912)]),
        (5000, 12, [(0, 8, 256), (8, 12, 256)]),
    ],
)
def test_verify_plan_cuts_at_n0_steps_and_eight_rows(prefix, t, runs):
    assert tileattn.verify_plan(prefix, t, 20) == runs


@pytest.mark.parametrize("splits", SPLIT_TARGETS)
def test_verify_plan_runs_cover_every_row_at_its_own_chunk(splits):
    for prefix in range(N0 - 20, N0 + 3 * 32 * splits + 1):
        for t in range(1, 17):
            runs = tileattn.verify_plan(prefix, t, splits)
            assert [j for j, _, _ in runs] == [0, *[k for _, k, _ in runs[:-1]]]
            assert runs[-1][1] == t
            for j, k, chunk in runs:
                assert 1 <= k - j <= tileattn.MAX_RUN
                assert all(tileattn.chunk_for(prefix + r + 1, splits) == chunk for r in range(j, k))
            for (j, k, chunk), (_, _, following) in zip(runs, runs[1:], strict=False):
                assert chunk != following or k - j == tileattn.MAX_RUN, (prefix, t, runs)
