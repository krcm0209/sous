"""Exact grouped verify attention (sous.engine.verifyattn).

The plan and grouping tables are pure Python. The GPU tests run on any Metal
GPU (CI is macos-15), and the slow tests repeat them in subprocesses with
MLX_METAL_GPU_ARCH set, so the 's' and 'd' dispatch tables the M5 Pro takes are
proven bit-exact on whatever GPU runs the suite.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine import verifyattn  # noqa: E402 — after the importorskip guard

S, D, G = "applegpu_g13s", "applegpu_g13d", "applegpu_g14g"


@pytest.mark.parametrize(
    ("arch", "n_keys", "gqa", "q_len", "expected"),
    [
        # the vector path's limits
        (S, 5000, 6, 9, ("fallback", 0)),
        (S, 5000, 6, 6, ("fallback", 0)),  # 6 x 6 > 32
        (S, 2, 6, 3, ("fallback", 0)),  # more queries than keys
        # 's': two passes from 1024 keys; more blocks only past 1024 and n_simds > 4
        (S, 1023, 6, 1, ("1pass", 0)),
        (S, 1024, 6, 1, ("2pass", 64)),
        (S, 1025, 6, 1, ("2pass", 128)),
        (S, 8192, 6, 1, ("2pass", 128)),
        (S, 8193, 6, 1, ("2pass", 256)),
        (S, 32768, 6, 1, ("2pass", 256)),
        (S, 32769, 6, 1, ("2pass", 512)),
        (S, 65536, 6, 1, ("2pass", 512)),
        (S, 65537, 6, 1, ("2pass", 1024)),
        (S, 5000, 4, 1, ("2pass", 64)),  # n_simds 4 is not > 4
        (S, 5000, 4, 2, ("2pass", 128)),
        # 'd'
        (D, 1023, 6, 1, ("1pass", 0)),
        (D, 1024, 6, 1, ("2pass", 128)),
        (D, 16383, 6, 1, ("2pass", 128)),
        (D, 16384, 6, 1, ("2pass", 512)),
        (D, 65535, 6, 1, ("2pass", 512)),
        (D, 65536, 6, 1, ("2pass", 1024)),
        (D, 8193, 2, 1, ("2pass", 256)),  # n_simds <= 2 past 8192
        (D, 20000, 4, 1, ("2pass", 128)),  # n_simds 4: neither branch
        # every other suffix: two passes only with GQA from 4096 keys
        (G, 4095, 6, 1, ("1pass", 0)),
        (G, 4096, 6, 1, ("2pass", 64)),
        (G, 4096, 2, 1, ("2pass", 32)),
        (G, 100000, 1, 1, ("1pass", 0)),
    ],
)
def test_plan_mirrors_mlx_0_32_2_dispatch(arch, n_keys, gqa, q_len, expected):
    assert verifyattn.plan(arch, n_keys, gqa, q_len) == expected


@pytest.mark.parametrize(
    ("arch", "prefix", "t", "gqa", "expected"),
    [
        (G, 4093, 3, 6, [(0, 2), (2, 3)]),  # rows at 4094, 4095 | 4096
        (G, 4093, 8, 6, [(0, 2), (2, 7), (7, 8)]),  # past 5 rows the call falls back
        (S, 1021, 4, 6, [(0, 2), (2, 3), (3, 4)]),  # 1024 and 1025 are two transitions
        (S, 8190, 8, 6, [(0, 2), (2, 7), (7, 8)]),
        (S, 32766, 3, 6, [(0, 2), (2, 3)]),
        (S, 57000, 8, 6, [(0, 5), (5, 8)]),
        (S, 57000, 1, 6, [(0, 1)]),
    ],
)
def test_groups_split_at_every_plan_transition(arch, prefix, t, gqa, expected):
    assert verifyattn.groups(arch, prefix, t, gqa) == expected


@pytest.mark.parametrize(
    ("arch", "gqa", "expected"),
    [
        (S, 6, (1024, 1025, 8193, 32769, 65537)),
        (D, 6, (1024, 16384, 65536)),
        (G, 6, (4096,)),
        (G, 4, (4096,)),
    ],
)
def test_plan_transitions_are_the_dispatch_thresholds(arch, gqa, expected):
    assert verifyattn.plan_transitions(arch, gqa) == expected


def test_probe_cases_straddle_every_transition_and_stop_past_the_last():
    cases = verifyattn.probe_cases(S, 6)
    assert max(p + t for p, t in cases) == 65537 + 7
    for n in verifyattn.plan_transitions(S, 6):
        for t in range(verifyattn.MIN_ROWS, verifyattn.MAX_ROWS + 1):
            # some row sits at n - 1 and the next at n
            assert any(p + 1 <= n - 1 and n <= p + t for p, tt in cases if tt == t), (n, t)
    assert all(verifyattn.MIN_ROWS <= t <= verifyattn.MAX_ROWS for _, t in cases)
    assert max(p + t for p, t in verifyattn.probe_cases(G, 6)) == 4096 + 7
