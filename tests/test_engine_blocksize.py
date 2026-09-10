"""A fixed speculative block size only takes effect when the drafter is told
to prefer it: mlx-vlm's DFlash controller otherwise re-picks the depth from
recent acceptance every round and ignores the request. Exercised without mlx
on a stand-in drafter."""

from sous.engine.vlm import _pin_block_size


class Drafter:
    prefer_requested_block_size = False  # mlx-vlm's class default


def test_a_fixed_block_size_pins_the_drafter_to_the_requested_depth():
    d = Drafter()
    _pin_block_size(d, 3)
    assert d.prefer_requested_block_size is True


def test_block_size_zero_leaves_the_drafters_own_policy_in_charge():
    d = Drafter()
    _pin_block_size(d, 0)
    assert d.prefer_requested_block_size is False


def test_pinning_nothing_is_a_no_op():
    # The engine runs without a drafter when the configured one cannot load.
    _pin_block_size(None, 3)
