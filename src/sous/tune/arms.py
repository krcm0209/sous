"""One arm is one configuration the bench measures: a model, a drafter or
none, a block size, and the window it fits at."""

from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass

from sous.config import (
    MIN_CONTEXT_TOKENS,
    SPECULATIVE_BLOCK_KERNEL,
    SPECULATIVE_BLOCK_MAX,
    SPECULATIVE_BLOCK_TILE,
    SousConfig,
)
from sous.tune.candidates import Candidate, Checkpoint, Fit, drafter_compatible, fit

# The verify depths worth measuring per machine, up to the config's clamp: 5
# pays on the M5 Pro where the projection kernel runs and 4 where it does not;
# 2 and 3 are for machines where a verify row costs more.
BLOCK_SIZES = tuple(range(2, SPECULATIVE_BLOCK_MAX + 1))


@dataclass(frozen=True)
class Arm:
    label: str
    config: SousConfig
    model_id: str
    drafter_id: str
    block_size: int
    # The model's measurement window, shared by every arm of the model and
    # lowered for its heaviest drafter, so no block size is measured at a
    # window another block size could not run at.
    window: int
    tier: str
    current: bool
    # The window this arm alone fits at — its own drafter or none — which is
    # what a proposal writes: the drafterless arm must not carry the
    # reservation made for a drafter its configuration never loads.
    fit_window: int | None = None
    # The quality-affecting dimensions the winner stage measures, mirrored
    # from `config` so a row can be keyed without reading the config back.
    int8_prefill: bool = False
    greedy: bool = False
    # Mirrored the same way but never measured: every arm inherits the
    # user's setting. It still keys suite rows, because tile and stock
    # attention differ in their last bits.
    attention_tile: bool = True
    # Inherited and keyed the same way, for the same reason: kernel and stock
    # projections differ in their last bits.
    projection_kernel: bool = True
    # True only for the winner stage's own int8 arm, built to *measure*
    # INT8 prefill: the engine refusing it must fail that arm. Every other
    # arm's int8_prefill is merely inherited from the user's config (every
    # quick_arms arm mirrors it) — the daemon would run that arm's
    # checkpoint on the stock path with one warning rather than refuse it,
    # so the suite runner must do the same instead of treating an inherited
    # setting as a hard requirement.
    int8_under_test: bool = False

    @property
    def key(self) -> tuple[str, str, int]:
        """What identifies an arm across a run, its rows and a resume. The
        label is for people: two orgs' checkpoints with one repo basename
        share it."""
        return (self.model_id, self.drafter_id, self.block_size)

    @property
    def suite_key(self) -> tuple[str, str, int, bool, bool, bool, bool]:
        """What identifies an arm across suite runs and a resume: the bench
        key, the two settings only the suite may change, and the attention
        tile and projection kernel every arm inherits."""
        return (
            *self.key,
            self.int8_prefill,
            self.greedy,
            self.attention_tile,
            self.projection_kernel,
        )


@dataclass(frozen=True)
class Refusal:
    model_id: str
    drafter_id: str
    reason: str


def _short(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def label_for(model_id: str, drafter_id: str, block_size: int) -> str:
    if not drafter_id:
        return _short(model_id)
    depth = "auto" if block_size == 0 else str(block_size)
    return f"{_short(model_id)} + {_short(drafter_id)} @{depth}"


def _with_current(user: SousConfig, candidates: list[Candidate]) -> list[Candidate]:
    """The user's own model and drafter always among the candidates: a model
    the table lacks is added as a candidate of its own, and a model the
    table lists gains the user's drafter when the table pairs it with
    others — otherwise the configured arm is never measured and there is no
    baseline to propose against."""
    mine = user.speculative_draft_id
    out: list[Candidate] = []
    found = False
    for cand in candidates:
        if cand.id == user.model_id:
            found = True
            if mine and mine not in cand.drafters:
                cand = dataclasses.replace(cand, drafters=(*cand.drafters, mine))
        out.append(cand)
    if not found:
        drafters = (mine,) if mine else ()
        out.insert(0, Candidate(user.model_id, "current", drafters, "the configured model"))
    return out


def drafter_kind(config: dict) -> str:
    """The round-loop kind mlx-vlm resolves at load for a drafter with this
    config.json and no kind given, by its own resolver, so the static check
    below needs no download. A resolver this cannot reach, or one that raises,
    reads as "unresolved", which the check refuses like any kind but dflash:
    the resolver is private to mlx-vlm and the check only advises, so its
    drift must not abort the tune."""
    try:
        drafters = importlib.import_module("mlx_vlm.speculative.drafters")
    except ImportError:
        return "unresolved"
    expected = getattr(drafters, "_expected_drafter_kind", None)
    default = getattr(drafters, "DEFAULT_DRAFTER_KIND", None)
    if expected is None or not isinstance(default, str):
        return "unresolved"
    model_type = config.get("model_type") or config.get("speculators_model_type")
    try:
        return str(expected(model_type, config) or default)
    except Exception:  # noqa: BLE001 — any drift in a private resolver
        return "unresolved"


def resolve_block(user: SousConfig, checkpoints: dict[str, Checkpoint]) -> SousConfig:
    """The user's configuration with the verify block the daemon would run.
    An unset speculative_block_size resolves at load to
    SPECULATIVE_BLOCK_KERNEL where the projection kernel goes active,
    SPECULATIVE_BLOCK_TILE where only the attention tile does and stays at the
    default elsewhere; the tune cannot load a model to find out, so it puts
    the two static checks to the checkpoints it already described.
    The current-arm mark, the user's own block among the measured ones and
    every proposed change then compare against what the daemon runs. A
    refusal the static check cannot see (a failed probe, a drifted source
    pin) leaves the mark a block or two off; the arms themselves are unaffected,
    since each pins the block its label names."""
    if user.speculative_block_explicit or not user.speculative_draft_id:
        return user
    target = checkpoints.get(user.model_id)
    drafter = checkpoints.get(user.speculative_draft_id)
    if target is None or drafter is None:
        return user
    # The daemon runs this pair undrafted (the factory drops the drafter on
    # the mlx-lm backend, the engine one that cannot read the target), so no
    # block is resolved for it.
    if target.backend != "vlm" or not drafter_compatible(target, drafter)[0]:
        return user
    from sous.engine import projkernel, tileattn

    kind = drafter_kind(drafter.config)
    if projkernel.static_reason(user, target.config, kind) is None:
        return dataclasses.replace(user, speculative_block_size=SPECULATIVE_BLOCK_KERNEL)
    if tileattn.static_reason(user, target.config, kind) is None:
        return dataclasses.replace(user, speculative_block_size=SPECULATIVE_BLOCK_TILE)
    return user


def _arm(
    user: SousConfig,
    cand: Candidate,
    drafter_id: str,
    block: int,
    f: Fit,
    own: Fit,
    *,
    current: bool | None = None,
) -> Arm:
    config = dataclasses.replace(
        user,
        model_id=cand.id,
        speculative_draft_id=drafter_id,
        speculative_block_size=block if drafter_id else user.speculative_block_size,
        # Explicit, or the copy of the user's unset flag would let the engine
        # resolve the block at load: where the projection kernel runs, every
        # drafter arm would measure block 5 whatever its label says.
        speculative_block_explicit=True if drafter_id else user.speculative_block_explicit,
        max_context_tokens=min(user.max_context_tokens, f.window),
    )
    if current is None:
        current = (
            cand.id == user.model_id
            and drafter_id == user.speculative_draft_id
            and (not drafter_id or block == user.speculative_block_size)
        )
    return Arm(
        label=label_for(cand.id, drafter_id, block),
        config=config,
        model_id=cand.id,
        drafter_id=drafter_id,
        block_size=block if drafter_id else 0,
        window=min(user.max_context_tokens, f.window),
        tier=cand.tier,
        current=current,
        fit_window=min(user.max_context_tokens, own.window),
        int8_prefill=user.int8_prefill,
        greedy=user.temperature == 0,
        attention_tile=user.attention_tile,
        projection_kernel=user.projection_kernel,
    )


def quick_arms(
    user: SousConfig,
    candidates: list[Candidate],
    checkpoints: dict[str, Checkpoint],
    *,
    working_set_bytes: int,
) -> tuple[list[Arm], list[Refusal]]:
    """Every quality-neutral arm of every candidate that fits, the user's own
    configuration among them. All arms of a model share one window: the one
    the fit yields with the model's heaviest drafter, so a block size is never
    measured at a window another block size could not run at."""
    window = user.max_context_tokens
    floor = MIN_CONTEXT_TOKENS
    arms: list[Arm] = []
    refusals: list[Refusal] = []
    for cand in _with_current(user, candidates):
        target = checkpoints.get(cand.id)
        if target is None:
            refusals.append(Refusal(cand.id, "", "checkpoint unknown (offline?)"))
            continue
        drafters: list[Checkpoint] = []
        # The configured drafter refused as unusable with this model: the
        # daemon runs the model undrafted (the engine degrades rather than
        # fails), so the no-drafter arm is the configuration in use.
        users_drafter_unusable = False
        for drafter_id in cand.drafters:
            drafter = checkpoints.get(drafter_id)
            if drafter is None:
                refusals.append(Refusal(cand.id, drafter_id, "drafter unknown (offline?)"))
                continue
            if drafter.bytes is None:
                # Unresolvable cost, not incompatibility: reaching fit() with
                # it would refuse the whole model, drafterless arm included.
                refusals.append(Refusal(cand.id, drafter_id, "drafter size unknown (offline?)"))
                continue
            if target.backend != "vlm":
                # Speculative decoding is an mlx-vlm feature; the factory
                # drops the drafter for the text-only backend without a word.
                ok, why = False, "the text-only backend has no speculative decoding"
            else:
                ok, why = drafter_compatible(target, drafter)
            if ok:
                drafters.append(drafter)
                continue
            refusals.append(Refusal(cand.id, drafter_id, why))
            if cand.id == user.model_id and drafter_id == user.speculative_draft_id:
                users_drafter_unusable = True
        heaviest = max(drafters, key=lambda d: d.bytes or 0, default=None)
        f = fit(target, heaviest, window=window, floor=floor, working_set_bytes=working_set_bytes)
        if not f.fits:
            refusals.append(Refusal(cand.id, "", f.detail))
            continue
        own = fit(target, None, window=window, floor=floor, working_set_bytes=working_set_bytes)
        arms.append(
            _arm(
                user,
                cand,
                "",
                0,
                f,
                own,
                current=True if users_drafter_unusable else None,
            )
        )
        for drafter in drafters:
            own = fit(
                target, drafter, window=window, floor=floor, working_set_bytes=working_set_bytes
            )
            is_users_own_pair = cand.id == user.model_id and drafter.id == user.speculative_draft_id
            # The user's own block size may sit outside the curated set (0 is
            # the drafter's adaptive policy; anything else they set by hand),
            # and it must still be measurable as `current` — every other pair
            # sticks to the curated set.
            blocks = (
                sorted({*BLOCK_SIZES, user.speculative_block_size})
                if is_users_own_pair
                else BLOCK_SIZES
            )
            for block in blocks:
                arms.append(_arm(user, cand, drafter.id, block, f, own))
    return arms, refusals


def winner_stage_arms(winner: Arm, *, nax: bool, checkpoint: Checkpoint) -> list[Arm]:
    """One extra arm per quality-affecting setting the winner does not have
    yet: INT8 prefill where the tensor units and the checkpoint's quantization
    allow it (the engine refuses anything else with a status, and the arm
    would measure the stock path under the int8 label), and greedy sampling
    for a winner that samples. Each is judged on its own against the winner."""
    from sous.engine.int8prefill import SUPPORTED_MODEL_TYPES

    arms: list[Arm] = []
    routable = (
        nax and checkpoint.model_type in SUPPORTED_MODEL_TYPES and checkpoint.quant.int8_routable
    )
    if routable and not winner.int8_prefill:
        arms.append(
            dataclasses.replace(
                winner,
                label=f"{winner.label} + int8 prefill",
                config=dataclasses.replace(winner.config, int8_prefill=True),
                current=False,
                int8_prefill=True,
                int8_under_test=True,
            )
        )
    if not winner.greedy:
        arms.append(
            dataclasses.replace(
                winner,
                label=f"{winner.label} greedy",
                config=dataclasses.replace(winner.config, temperature=0.0),
                current=False,
                greedy=True,
            )
        )
    return arms
