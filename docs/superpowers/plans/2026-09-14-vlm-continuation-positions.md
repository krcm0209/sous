# VLM Continuation Positions — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every token the VLM backend feeds behind a warm cache — a warm turn's new messages, the tail behind a fork boundary, the generation block every decode starts with — is encoded at its real position in the conversation, on the drafter path and off it, and a test that would have caught the current mispositioning runs on every model-test pass.

**Architecture:** One helper on `VLMEngine` builds explicit rotary positions for a call — a `position_ids` array covering the cache from position 0 through the tokens being appended, and a zero `rope_deltas` — and both cache hooks (`prefill`, `decode`) hand it to mlx-vlm alongside the ids, for the model families whose text-only embedding helper returns positions of its own (probed once per engine). No change to the prompt cache, the drafter, the LM backend, or any protocol. The dependency floor moves to the mlx-vlm release whose `generate_step` honours caller positions over the embedding helper's, and CLAUDE.md gains the gotcha that explains why the kwargs are load-bearing.

**Tech Stack:** Python 3.14; mlx-vlm `>=0.7.0` (0.7.0 locked; 0.7.1 released 2026-09-14 carries the same code, verified); pytest with `object.__new__`-built engines and a `sys.modules` stub for the two mlx-vlm entry points (no mlx-vlm import in CI); one `model`-marked test with two parametrisations on the small VLMs already used by `tests/test_engine_vlm.py`, run on the M5 Pro over ssh.

**Spec:** GitHub issue #86 ("engine: speculative decode positions the generation-prompt suffix at 0..N-1 instead of the cache offset") as corrected by the measurements in *What was measured* below — the defect is wider than the issue's (every continuation, prefill and decode, drafter or not, on 0.6.17 and 0.7.0 alike), and on the locked 0.7.0 its shape is different (position 0, not suffix-local). Related: #87 and #88 change the same decode call and should land after this; #91's second correction (the mRoPE-priming note is path-dependent) is closed by this plan's docs task; its first (cache classes attributed to mlx-lm) is already gone from CLAUDE.md.

## What was measured

Throwaway instrumentation on the M5 Pro (2026-09-14, `mlx-community/Qwen3.8-27B-4bit` + `z-lab/Qwen3.8-27B-DFlash2`, mlx-vlm 0.7.0, mlx 0.32.2; a 4381-token stable render `P` and its 7-token generation block `N` = `<|im_start|>assistant\n<think>\n\n</think>\n\n`). Never committed; the facts it established are the spec.

**Mechanism on 0.7.0.** `Qwen3_5Model.get_input_embeddings` (text-only branch) returns `position_ids, rope_deltas = get_rope_index(input_ids)` — positions **local to the ids it is handed**, `0..n-1`. `generate_step` (`mlx_vlm/generate/ar.py`) merges the embedding output into `kwargs` and re-applies `explicit_prompt_metadata`, so those local positions reach `LanguageModel.__call__` as explicit `position_ids` on every prompt forward (the chunk loop's and `_step`'s). The language model then does `if position_ids.shape[-1] > seq_length: position_ids = position_ids[..., cache_offset : cache_offset + seq_length]` — `generate_step` chunks any prompt of more than one token into steps of `prefill_step_size` (2048) followed by a final single token, so for any continuation the last chunk, and usually every chunk, is shorter than the helper's array and the guard fires; for a call that starts at cache offset `P ≥ n` the slice is **empty**, shape `(1, 0)`. The fused MRoPE Metal kernel (`rope_utils._mrope_apply_kernel`) indexes `position_ids[b * q_len + t]` with no bounds check; the out-of-bounds read returns 0, so **every continued token is rotated to position 0**. When `0 < P < n` the first `n − P` tokens get positions `P..n-1` and the rest 0. (An unchunked forward would be wrong differently: the guard does not fire and the tokens keep their local `0..n-1`.) The `position_ids is None` branch the issue analysed (state nulled by a drafter run → `get_rope_index` → `0..N-1`) is never reached because the kwarg is never `None`. `_prime_cached_prefix_rope_state` (`dispatch.py`), which sets full-prompt positions for exactly this case, runs only on mlx-vlm's own `prompt_cache_state`/APC paths — sous drives `prompt_cache` + `input_ids` directly and never enters them. Qwen2-VL's language model carries the identical slice.

**Who is affected.** Every sous call at a non-zero cache offset: `prefill(cache, stable_ids[reuse:boundary])` and `prefill(cache, stable_ids[reuse:])` on warm and forked turns (the whole conversation behind a tools fork — system text and every message — and a warm turn's new tool results), and the decode of the generation block wherever it starts behind a non-empty cache: on the non-trimmable (hybrid) path, which the default model takes, that is `decode(cache, full_ids[anchor:])` on every turn, cold ones included, because the stable render is always prefilled first; on the all-trimmable (pure-attention) path prefill and decode fuse into one `decode(cache, full_ids[reuse:])`, so only a warm or forked turn's decode is affected. A fork copied from a mispositioned continuation publishes the mispositioning to every session that starts from it. Unaffected: a cold prefill from an empty cache (positions `0..n-1` are right at offset 0), the mlx-lm backend (its Qwen3.5 text model takes positions from the cache offset), the drafter's exactness (a single-call run with the drafter matches one without, token for token).

**Keys, not text, are the witness.** Cached keys are rotated by position before they are stored, so the key a token gets behind a warm cache must equal the key the same token gets in a one-pass prefill of the whole prompt, up to the rounding a different chunk shape costs. Max over the 16 full-attention layers of `|k − truth| / max|truth|`, per generation-block position, against one explicit-position forward of the block over a copy of the same prefix cache:

| path | positions 0..6 |
|---|---|
| explicit `P..P+N-1`, chunked `N−1` then 1 (generate_step's shape) | 0, 0, 0, 0, 0, 0, **0.017** (the drift floor over an identical prefix) |
| explicit empty `(1, 0)` — what the slice leaves today | 1.70, 1.41, 1.28, 1.29, 1.38, 1.30, 1.30 |
| explicit all-zero `(1, N)` | bit-identical to the empty row |
| suffix-local `0..N-1` — the issue's prediction | 1.70, 1.42, 1.27, 1.29, 1.38, 1.32, 1.30 |
| no kwarg, state nulled as after a drafter run | = suffix-local |
| no kwarg, state `rope_deltas = 0` as after a plain prefill | 0 everywhere (a bare forward; never what `generate_step` does, on either version) |
| **sous `prefill()` of the block** | = empty |
| **sous `decode()` of the block, drafter on** | = empty |
| **sous `decode()` of the block, drafter off** | = empty |
| `stream_generate` with explicit `arange(P+N)` + zero `rope_deltas`, drafter on (**the fix**) | 0, 0, 0, 0, 0, 0, 0.017 |
| same, drafter off | 0, 0, 0, 0, 0, 0, 0.017 |
| partial: 7 tokens behind a 4-token cache, sous `prefill()` | 0, 0, 0, 1.05, 0.70, 1.41, 0.71 |

**Greedy text is not a witness.** A mispositioned split reproduced the one-pass reference's 24 greedy tokens exactly (cold, warm, drafter on and off), so the split-versus-single text comparison the issue proposed passes on the defect. No text comparison of a fully fixed split against the reference was run: the one probe that corrected the decode positions left the 17-token tail behind the fork boundary at position 0, and its divergent answer says nothing about the fix. The test in Task 1 therefore compares keys.

**Small models, and where the test reads.** The plan's model test was run ahead of time on the two small VLMs `tests/test_engine_vlm.py` uses, with the fix applied by wrapping the hooks: a 9-token continuation behind a 431-token (Qwen2-VL-2B) and a 471-token (Qwen3.5-9B) cache against a one-pass prefill of the whole prompt, metric `max |got − want| / max |want|` per layer. **Qwen2-VL-2B (28 attention layers, no recurrence):** unfixed 1.23 as the max over all layers, every tail position above 0.87, but the first layer alone only 0.02; fixed 0.004 over all layers. So on a pure-attention model every layer is comparable and the first layer alone is not a witness. **Qwen3.5-9B (8 attention layers among 32):** at the first attention layer (cache index 3), unfixed 1.08 and fixed 0.009 against the one-pass reference; deeper attention layers differ from that reference by up to 0.89 *fixed*, and by the same order (0.74 at the third attention layer) between two explicit-position references that merely chunk the prefix differently — recurrent-state drift compounding through 4-bit layers, not positions. So on a hybrid only the first attention layer is comparable, and there the defect still shows as >1.0. The test therefore reads every layer on a pure-attention model and the first attention layer on a hybrid, keyed on the cache shape it already asserts, with a bound of 0.1 an order of magnitude from both sides on both models. (Against a forward over an identical prefix, the fixed 9B continuation is within 0.012 at the first attention layer and 0.31 at its worst position over all layers.)

**Why full-length positions, and why `rope_deltas` too.** `generate_step` hands the same `kwargs` to every prompt chunk and the model slices at its own cache offset, so the array must be indexed by absolute position from 0 — a suffix-only array is sliced past its end exactly as the helper's is. `rope_deltas` travels with the positions so the engine, not the embedding helper, owns the delta the model positions generated tokens from: `LanguageModel.__call__` adopts the kwarg into `self._rope_deltas` (the state a drafter run nulls) and positions every post-prompt token at `cache_offset + that state`. On 0.7.0 the helper's own text-only delta is already zero, so the kwarg pins a value rather than repairing one; it is what keeps the contract from depending on the helper's output.

**Which models.** The defect needs a text-only embedding helper that returns positions, and in mlx-vlm 0.7.0 that is the Qwen lineage — `qwen2_vl`, `qwen2_5_vl`, `qwen3_5` and its MoE and derived variants, `qwen3_vl`, `qwen3_vl_moe`, `qwen3_omni_moe` — every member of which slices a caller's `position_ids` at its cache offset, so the absolute array is right for all of them (verified on `qwen3_5` and `qwen2_vl`; the rest share the code). Families that reuse a Qwen model class (`qwen3_5_moe`, `minicpmv4_6` and kin) inherit both the helper and the slice and get the kwargs the same way. The multi-axis-rope families with their own code — `glm4v`, `glm4v_moe`, `glm_ocr`, `paddleocr_vl`, `ernie4_5_moe_vl`, `minimax_m3_vl`, `hunyuan_vl`, and `cohere_compass`/`falcon_ocr`, which build positions only on their image-bearing branch — return no positions from the text-only helper and derive `cache_offset + rope_deltas` themselves, so the probe leaves them alone; the GLM/ERNIE/MiniMax group among them consumes a caller's array verbatim in its own rank, so an absolute rank-2 array would be an index error in its rotary embedding on a cold prefill as much as a warm one. The helper's behaviour is therefore the predicate, probed once per engine with a one-token call (verified to return positions on both small test models); `get_rope_index` alone is not — every one of those families has it or its equivalent.

**0.6.17 — the defect predates the bump, and the fix needs 0.7.0.** The same script in a venv with mlx-vlm 0.6.17 (mlx 0.32.2, same weights): sous `prefill()`, `decode()` with the drafter and `decode()` without it all put the block at suffix-local positions `0..N-1` (the issue's shape — 1.70, 1.42, 1.27, 1.29, 1.38, 1.32, 1.30 — on the prefill path as well, which the issue thought unaffected), and `stream_generate` with explicit `arange(P+N)` + zero `rope_deltas` produced the **same** wrong row: 0.6.17's `generate_step` merges the embedding helper's output over the caller's kwargs with no re-application, so the caller's positions are silently overwritten. Hence the floor in Task 2: below 0.7.0 the fix is inert. Every VLM continuation sous has made on 0.6.17 and later was mispositioned (earlier releases were not measured); the 0.7.0 bump (#84, 2026-09-12) changed the shape of the error, not its existence, and every gate since the 0.6.17 bump (#67, 2026-09-05) ran on one shape or the other.

**0.7.1.** The `explicit_prompt_metadata` block, the language model's slice and the text-only `get_input_embeddings` are byte-identical between 0.7.0 and 0.7.1; the fix holds on both, and Dependabot's next bump changes nothing here.

**Drafter throughput, one sample each.** 256 greedy tokens of prose on the 27B with DFlash2 behind the same 4381-token prefix: 27.9–28.4 tok/s mispositioned, 25.4–25.9 tok/s fixed — decoding *different* text (the fix changes the story), so this is directional, not an A/B. A mispositioned context flattens the target's distribution and repetitive text is easier to draft, so acceptance can fall while output quality rises. Task 3 records acceptance and tokens/s on both branches; neither gates the merge.

## Global Constraints

- Python `>=3.14`; `except A, B:` without parentheses is valid 3.14 syntax — do not "fix" it.
- Type-suppression pragmas are `# ty: ignore[rule]`, never `# type: ignore`; `ty` flags an *unused* pragma as its own diagnostic — add one only when `ty check` names the rule. The three in Task 1's test helper are named there because `ty` reports them (mlx-vlm ships `py.typed`, so `VLMEngine._model`, `_processor` and `_sampler` carry the types `load()` and `make_sampler()` return).
- `mlx` / `mlx_lm` / `mlx_vlm` imports stay function-local (the lint job runs on ubuntu without them). The new helpers import `mlx.core` inside their bodies like every other `VLMEngine` method; the non-model tests stub `mlx_vlm` in `sys.modules` and never import the real package, and take `mlx.core` through a module-level `pytest.importorskip` exactly as `tests/test_int8prefill.py` does (the tests job runs on macos-15 where it is installed; the guard is for parity with that precedent, not for CI).
- No thread that touches mlx is added or changed; the helpers run on the calling thread inside a hook that already holds mlx arrays.
- Tests never touch the real `~/.sous` or `~/.claude`; the non-model tests build engines with `object.__new__` the way `tests/test_engine_unloaded.py` does and load no weights. The `model`-marked test downloads the two small VLMs `tests/test_engine_vlm.py` already uses (cached on the M5 Pro) and runs there over ssh, never in CI.
- Never edit, reformat or "sync" anything under `docs/superpowers/**` (this plan included) once committed.
- **No plan language in committed code.** Code and test comments/docstrings must never cite this plan, the issue by number as a rationale, a task or a step. State the fact instead (the mechanism above is the fact). Sweep before the PR: `git diff main -- src tests CLAUDE.md pyproject.toml | grep -nE '^\+.*(Task [0-9]|Step [0-9]|#(86|87|88|91)([^0-9]|$)|docs/superpowers|the plan|this plan)'` must print nothing.
- Never loosen an existing assertion to make a test pass. Every existing test in `tests/test_engine_vlm.py` keeps its expected values; the two fork bit-exactness tests compare a fork copy against a split prefill and stay bit-exact with the fix because both sides now carry the same positions.
- Commits: Conventional Commits, imperative lowercase subject, *why* in the body, trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>` — model-less, whatever the harness suggests. No session link anywhere. No person's name or home path anywhere committed or posted (say "the maintainer").
- Verification before every commit: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3` (never pipe without `pipefail`; `pyproject.toml` already sets `-q`, so never add another `-q`, and use `-vv` where a per-test listing is wanted — a single `-v` only cancels it), `uv run ty check`, `uv run ruff format .` then `uv run ruff check . && uv run ruff format --check .`, and after Task 2 `uv lock --check`. Line length is 100 and `E501` is enforced. No commit on this branch carries a failing test: the model test's RED run happens from the working tree (Task 1 Step 4), not from a commit.
- The suite on `main` at `a36246f` under `-m "not model"` on this Mac (macOS 15.5, no tensor units, eight `tests/test_int8prefill.py` cases skip): `1104 passed, 8 skipped, 29 deselected in 109.78s (0:01:49)`. Each task's "Expected" line gives the delta.
- Branch: `fix/vlm-continuation-positions` off `main` at `a36246f`. `main` is protected: PR with all four CI jobs green. Pushing the feature branch to origin is routine here and is how the M5 Pro checkout sees it; nothing is ever pushed to `main`.
- The maintainer's daemon on the M5 Pro is not restarted by any task; the model tests and the gate run in the `~/code/personal/sous-remote` checkout there, never in the maintainer's working copy. After the PR merges, the daemon must be reinstalled and restarted: its in-memory slots were built by the mispositioned path and nothing but a restart drops them.

---

## File structure

| File | Responsibility |
|---|---|
| `src/sous/engine/vlm.py` | `VLMEngine._positions(cache, n_tokens)`: the explicit-position kwargs, or `{}` for a model whose embedding helper returns no positions; `_helper_returns_positions()`: the one-token probe, memoised in `_positional`; `prefill()` and `decode()` spread the kwargs into their mlx-vlm calls; the stale "priming turns out to be bit-identical" comment in `decode()` is replaced by the real contract. |
| `tests/test_engine_positions.py` (new) | Non-model: the helper's values (coverage from 0, dtype, the offset read off the attention layers, the empty-cache case), the probe (positions returned → kwargs; none → `{}`; helper raises → `{}`; probed once), and both hooks handing the kwargs to mlx-vlm, via a `sys.modules` stub. |
| `tests/test_engine_vlm.py` | `model`-marked `test_vlm_a_continuation_is_positioned_behind_its_cache`, parametrised over the pure-attention and hybrid small VLMs: cached keys behind a warm cache (every layer on the pure-attention model, the first attention layer on the hybrid) match a one-pass prefill for both `prefill()` and `decode()`. Fails on `main`. |
| `pyproject.toml`, `uv.lock` | `mlx-vlm>=0.7.0` (the release whose `generate_step` re-applies caller positions over the embedding helper's); `uv lock` refreshes the recorded requirement, the resolution is unchanged. |
| `CLAUDE.md` | One Gotchas bullet: the kwargs are load-bearing, for which models, why, and the restart note. |

---

### Task 1: Explicit continuation positions on the VLM backend

**Files:**
- Modify: `src/sous/engine/vlm.py` (`__init__`, `prefill`, `decode`; new `_positions`, `_helper_returns_positions`)
- Create: `tests/test_engine_positions.py`
- Modify: `tests/test_engine_vlm.py` (one new `model`-marked test after `test_vlm_snapshot_restore_is_bit_exact`)

**Interfaces:**
- Consumes: `CacheHooks.prefill(cache, token_ids)` / `decode(cache, token_ids, max_tokens, on_delta)` as `PrefixCache` calls them (`src/sous/engine/promptcache.py`); each attention layer's `offset` (an `int` on mlx-vlm's `KVCache`; recurrent `ArraysCache` layers have none), read with `getattr(c, "offset", 0) or 0` exactly as `promptcache.snapshot`/`trim_to` read it — the max over layers is the number the language model itself slices at, since every attention layer of a cache built by the model's own `make_cache` holds the same offset and the model reads one of them (`cache[self.model.fa_idx]`); `model.get_input_embeddings(input_ids)` (the wrapper model's text-only embedding helper) as the probe.
- Produces: `VLMEngine._positions(cache: list, n_tokens: int) -> dict[str, Any]` — `{"position_ids": mx.arange(offset + n_tokens, dtype=mx.int32)[None], "rope_deltas": mx.zeros((1, 1), dtype=mx.int32)}` when the helper returns positions, `{}` otherwise; `VLMEngine._helper_returns_positions() -> bool`; the attribute `_positional: bool | None` (None until probed). Nothing outside `vlm.py` calls them; the tests do.

- [ ] **Step 1: Write the failing non-model tests**

Create `tests/test_engine_positions.py`:

```python
"""The VLM backend hands mlx-vlm explicit positions on every prefill and
decode. mlx-vlm's text-only embedding helper returns positions local to the
ids it is given, generate_step merges them into the model's kwargs, and a
Qwen-style language model slices that array at its cache offset — past its
end for any continuation generate_step chunks — so the fused MRoPE kernel
reads a zero-length buffer out of bounds and every continued token lands at
position 0. The kwargs are the cure, for exactly the models whose helper
returns positions; exercised here without mlx-vlm by stubbing its two entry
points, the way tests/test_engine_unloaded.py builds engines without weights."""

import sys
import threading
import types

import pytest

mx = pytest.importorskip("mlx.core")

from sous.engine.vlm import VLMEngine  # noqa: E402 — after the importorskip guard


class _Layer:
    """An attention layer carries an int offset; a recurrent layer carries none."""

    def __init__(self, offset: int | None = None):
        if offset is not None:
            self.offset = offset


class _Embeddings:
    def __init__(self, position_ids):
        self.position_ids = position_ids


class _Model:
    """The wrapper model, reduced to its text-only embedding helper: it either
    returns rotary positions of its own (the Qwen lineage), returns none (a
    family that positions from the cache offset itself), or cannot run
    text-only at all."""

    def __init__(self, positions: bool | None = True):
        self.positions = positions
        self.probes = 0
        self.language_model = object()
        self.config = types.SimpleNamespace(eos_token_id=0)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        self.probes += 1
        if self.positions is None:
            raise RuntimeError("needs pixels")
        return _Embeddings(input_ids if self.positions else None)


def _engine(model: _Model) -> VLMEngine:
    from sous.engine.promptcache import PrefixCache, PromptMemo

    engine = object.__new__(VLMEngine)
    engine.model_id = "test/model"
    engine._model = model  # ty: ignore[invalid-assignment]
    stopping = types.SimpleNamespace(reset=lambda eos: None)
    engine._processor = types.SimpleNamespace(  # ty: ignore[invalid-assignment]
        tokenizer=types.SimpleNamespace(stopping_criteria=stopping)
    )
    engine._sampler = object()  # ty: ignore[invalid-assignment]
    engine._positional = None
    engine._memo = PromptMemo()
    engine._tokenize_lock = threading.Lock()
    engine._draft = None
    engine._draft_kind = ""
    engine._draft_block_size = 0
    engine._cache = PrefixCache(engine, enabled=True)
    return engine


def test_positions_cover_the_cache_from_zero_through_the_new_tokens():
    # Recurrent layers first, as the default model lays them out: the offset
    # comes from the attention layers, whichever index they sit at.
    cache = [_Layer(), _Layer(), _Layer(offset=40), _Layer(offset=40)]
    kw = _engine(_Model())._positions(cache, 7)
    assert kw["position_ids"].shape == (1, 47)
    assert kw["position_ids"].dtype == mx.int32
    assert kw["position_ids"][0, 40:].tolist() == list(range(40, 47))
    assert kw["rope_deltas"].shape == (1, 1)
    assert kw["rope_deltas"].item() == 0


def test_an_empty_cache_positions_from_zero():
    kw = _engine(_Model())._positions([_Layer(offset=0), _Layer()], 3)
    assert kw["position_ids"].tolist() == [[0, 1, 2]]


def test_a_helper_that_returns_no_positions_means_no_kwargs():
    # Such a model derives cache_offset + rope_deltas itself and consumes a
    # caller's array verbatim in its own rank: leaving it alone is the fix.
    assert _engine(_Model(positions=False))._positions([_Layer(offset=40)], 7) == {}


def test_a_helper_that_cannot_run_text_only_means_no_kwargs():
    assert _engine(_Model(positions=None))._positions([_Layer(offset=40)], 7) == {}


def test_the_helper_is_probed_once_per_engine():
    model = _Model()
    engine = _engine(model)
    for _ in range(3):
        engine._positions([_Layer(offset=40)], 7)
    assert model.probes == 1
    assert engine._positional is True


def test_prefill_and_decode_hand_mlx_vlm_the_positions(monkeypatch):
    calls: dict[str, dict] = {}

    def generate(model, processor, prompt, **kwargs):
        calls["prefill"] = kwargs

    def stream_generate(model, processor, prompt, **kwargs):
        calls["decode"] = kwargs
        return iter(())

    stub = types.SimpleNamespace(generate=generate, stream_generate=stream_generate)
    monkeypatch.setitem(sys.modules, "mlx_vlm", stub)
    engine = _engine(_Model())
    cache = [_Layer(), _Layer(offset=40)]
    engine.prefill(cache, [1, 2, 3])
    engine.decode(cache, [4, 5], 8)
    assert calls["prefill"]["position_ids"].tolist() == [list(range(43))]
    assert calls["prefill"]["rope_deltas"].tolist() == [[0]]
    assert calls["prefill"]["max_tokens"] == 0
    assert calls["decode"]["position_ids"].tolist() == [list(range(42))]
    assert calls["decode"]["rope_deltas"].tolist() == [[0]]
    assert calls["decode"]["max_tokens"] == 8


def test_prefill_of_nothing_asks_mlx_vlm_for_nothing(monkeypatch):
    # The existing early return keeps its meaning: no ids, no call, no probe.
    called = []
    stub = types.SimpleNamespace(generate=lambda *a, **k: called.append(k), stream_generate=None)
    monkeypatch.setitem(sys.modules, "mlx_vlm", stub)
    model = _Model()
    _engine(model).prefill([_Layer(offset=40)], [])
    assert called == []
    assert model.probes == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_engine_positions.py -vv`
Expected: six tests fail with `AttributeError: 'VLMEngine' object has no attribute '_positions'` / `KeyError: 'position_ids'`; `test_prefill_of_nothing_asks_mlx_vlm_for_nothing` passes already (the early return exists) — that is fine, it pins the early return against the change in Step 5.

- [ ] **Step 3: Write the failing model test**

Add to `tests/test_engine_vlm.py`, directly after `test_vlm_snapshot_restore_is_bit_exact`:

```python
@pytest.mark.parametrize(
    ("model_id", "is_hybrid"),
    [
        pytest.param(TINY_VLM, False, id="pure-attention"),
        pytest.param(HYBRID_VLM, True, id="linear-attention-hybrid"),
    ],
)
def test_vlm_a_continuation_is_positioned_behind_its_cache(model_id, is_hybrid):
    """A key is rotated by its position before it is cached, so a token fed
    behind a warm cache must get the key the same token gets when the whole
    prompt is prefilled in one pass from position 0. Both continuation paths
    must hold it: the prefill of a suffix (a warm turn's new tokens, the tail
    behind a fork boundary) and the decode of the generation block.

    Greedy text is no witness — a mispositioned generation block has
    reproduced the one-pass reference's words exactly — so this compares
    cached keys, measured as max |got − want| / max |want| over the tail's
    positions. Which layers depends on the cache shape. A pure-attention
    model has nothing but attention layers between a one-pass and a split
    prefill, so every layer is comparable: 0.004 fixed against 1.23 unfixed
    on Qwen2-VL-2B (its first layer alone barely moves, 0.02 unfixed — every
    other layer does). A hybrid's recurrent layers drift between the two
    prefills for reasons that have nothing to do with positions, and the
    deeper attention layers compound that drift into tens of percent, so
    there only the first attention layer is read: 0.009 fixed against 1.08
    unfixed on Qwen3.5-9B. The bound sits an order of magnitude above the
    fixed values and below the defect on both."""
    import mlx.core as mx

    from sous.engine.promptcache import fork_copy
    from sous.engine.vlm import VLMEngine

    e = VLMEngine(model_id, temperature=0.0)
    ids = e._encode("def f(x):\n    return x + 1\n" * 40)
    cut = len(ids) - 9
    head, tail = ids[:cut], ids[cut:]

    whole = e.new_cache()
    e.prefill(whole, ids)
    warm = e.new_cache()
    e.prefill(warm, head)

    def attention_keys(cache):
        # Only an attention layer holds keys; a recurrent layer's state has no
        # positions to check. On the hybrid, only the first attention layer.
        layers = [c for c in cache if getattr(c, "keys", None) is not None]
        if is_hybrid:
            assert len(layers) < len(cache)
            layers = layers[:1]
        else:
            assert len(layers) == len(cache)
        return [c.keys[..., cut : len(ids), :].astype(mx.float32) for c in layers]

    want = attention_keys(whole)

    def worst(cache):
        return max(
            (mx.abs(got - w).max() / mx.abs(w).max()).item()
            for got, w in zip(attention_keys(cache), want, strict=True)
        )

    continued = e.new_cache()
    fork_copy(warm, continued, e.copy_array)
    e.prefill(continued, tail)
    assert worst(continued) < 0.1

    decoded = e.new_cache()
    fork_copy(warm, decoded, e.copy_array)
    e.decode(decoded, tail, 1)
    assert worst(decoded) < 0.1
    e.unload()
```

- [ ] **Step 4: Run the model test on the M5 Pro to verify it fails on the current code**

Nothing is committed for this: the test file's diff is applied to the remote checkout at `main`, run, and reverted (`git diff HEAD` so a staged file still produces the patch; `-vv`, not `-v`, because `addopts` already carries `-q`, which cancels a single `-v`).

```bash
git diff HEAD -- tests/test_engine_vlm.py | ssh m5pro 'cd ~/code/personal/sous-remote && git checkout -q main && git pull -q --ff-only && git apply && HF_HUB_OFFLINE=1 uv run pytest -m model tests/test_engine_vlm.py -k positioned -vv 2>&1 | tail -12; git checkout -q -- tests'
```

Expected: both parametrisations FAIL on the `continued` assertion with a worst error above 1.0 (position-0 keys); record both `worst` values from the assertion messages. If a parametrisation fails elsewhere (a download, a missing weight), fix the environment, not the test. Confirm afterwards that `ssh m5pro 'cd ~/code/personal/sous-remote && git status --short'` prints nothing; the checkout stays on `main` until Step 7 checks the pushed branch out.

- [ ] **Step 5: Implement the probe and the helper, and hand the kwargs over**

In `src/sous/engine/vlm.py`: in `__init__`, directly after `self._model, self._processor = load(model_id)`, add

```python
        # Whether the text-only embedding helper returns rotary positions —
        # decided by _helper_returns_positions on the first prefill or decode.
        self._positional: bool | None = None
```

then replace `prefill` and `decode`, adding the two helpers between them:

```python
    def prefill(self, cache: list, token_ids: list[int]) -> None:
        import mlx.core as mx
        from mlx_vlm import generate

        model, processor = self._loaded()
        if not token_ids:
            return
        # max_tokens=0 is prefill-only: dispatch has an explicit
        # `if not generated_tokens:` branch that yields a result and returns
        # without touching any cache state.
        generate(
            model,
            processor,
            "",
            max_tokens=0,
            verbose=False,
            prompt_cache=cache,
            input_ids=mx.array(token_ids)[None],
            **self._positions(cache, len(token_ids)),
        )

    def _positions(self, cache: list, n_tokens: int) -> dict[str, Any]:
        """Explicit rotary positions for `n_tokens` appended behind `cache`,
        or nothing for a model whose embedding helper returns none.

        Load-bearing, not belt and braces: mlx-vlm's text-only embedding
        helper returns positions local to the ids it is given, generate_step
        merges them into the model's kwargs, and a Qwen-style language model
        slices that array at its cache offset — past its end for any
        continuation generate_step chunks — so the fused MRoPE kernel reads a
        zero-length buffer out of bounds and every continued token lands at
        position 0 (keys off by 130–170 % on the default model). The array
        covers the cache from 0 because the same kwargs reach every prompt
        chunk and the model slices each at its own offset; a suffix-only
        array is sliced past its end the same way. `rope_deltas` travels
        with it so the engine, not the helper, owns the delta the model
        adopts into the state a drafter run nulls and positions the
        generated tokens from."""
        import mlx.core as mx

        if self._positional is None:
            self._positional = self._helper_returns_positions()
        if not self._positional:
            return {}
        # The attention layers share one offset; a recurrent layer has none
        # (the same read promptcache.snapshot makes).
        offset = max((int(getattr(c, "offset", 0) or 0) for c in cache), default=0)
        return {
            "position_ids": mx.arange(offset + n_tokens, dtype=mx.int32)[None],
            "rope_deltas": mx.zeros((1, 1), dtype=mx.int32),
        }

    def _helper_returns_positions(self) -> bool:
        """Whether this model's text-only embedding helper returns rotary
        positions, probed once with a single token. The families that do (the
        Qwen lineage) are exactly the ones whose language model slices a
        caller's positions at its cache offset; the others derive
        `cache_offset + rope_deltas` themselves and consume a caller's array
        verbatim in their own rank, so handing them an absolute rank-2 array
        would be an index error, not a repair."""
        import mlx.core as mx

        model, _ = self._loaded()
        try:
            out = model.get_input_embeddings(mx.zeros((1, 1), dtype=mx.int32))
        except Exception:  # noqa: BLE001 — a helper that needs pixels positions itself
            return False
        return getattr(out, "position_ids", None) is not None

    def decode(
        self, cache: list, token_ids: list[int], max_tokens: int, on_delta: OnDelta | None = None
    ) -> str:
        import mlx.core as mx
        from mlx_vlm import stream_generate

        model, processor = self._loaded()
        # Speculative decoding rides on the decode call only: prefill has no
        # tokens to draft, and generate_step captures the hidden states the
        # drafter needs during its own prefill of these input_ids. block size
        # 0 means None — let the drafter's own policy pick the depth.
        draft_kwargs = (
            {
                "draft_model": self._draft,
                "draft_kind": self._draft_kind,
                "draft_block_size": self._draft_block_size or None,
            }
            if self._draft is not None
            else {}
        )
        # generate() resets the tokenizer's shared stopping criteria before
        # every call and stream_generate does not; mirror it so a criteria
        # left mutated by another caller cannot change where this turn stops.
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)
        chunks: list[str] = []
        # prompt_cache plus input_ids, not prompt_cache_state: sous owns the
        # cache outright rather than driving mlx-vlm's reuse path, and so
        # also owns the positions the suffix is encoded at (_positions).
        for r in stream_generate(
            model,
            processor,
            "",
            max_tokens=max_tokens,
            sampler=self._sampler,
            verbose=False,
            prompt_cache=cache,
            input_ids=mx.array(token_ids)[None],
            **self._positions(cache, len(token_ids)),
            **draft_kwargs,
        ):
            # Draft rows are the speculator's proposals, not accepted output;
            # generate() skips them the same way.
            if r.is_draft:
                continue
            chunks.append(r.text)
            if on_delta is not None:
                on_delta(Delta(r.text, r.generation_tokens, r.finish_reason))
        return "".join(chunks)
```

`Any` is already imported at the top of `vlm.py` (`from typing import Any, cast`). The `# noqa: BLE001` form is the one `vlm.py` already uses for the drafter's load.

- [ ] **Step 6: Run the non-model tests and the suite**

Run: `uv run pytest tests/test_engine_positions.py`
Expected: 7 passed.

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3`
Expected: `1111 passed, 8 skipped, 31 deselected` — seven more passes (the new non-model file) and two more deselections than the baseline (the two parametrisations of the `model`-marked test Step 3 added); the 8 skips are unchanged.

Run: `uv run ty check && uv run ruff format . && uv run ruff check . && uv run ruff format --check .`
Expected: all clean. If `ty` complains that `cache` items have no `offset`, the `getattr` form above is what it accepts (it mirrors `promptcache.snapshot`).

- [ ] **Step 7: Commit, push, and run the model tests on the M5 Pro to verify the new one passes and the rest still do**

```bash
git add src/sous/engine/vlm.py tests/test_engine_positions.py tests/test_engine_vlm.py
git commit -m "fix(engine): position every VLM continuation behind its cache" -m "mlx-vlm's text-only embedding helper returns positions local to the ids it is handed; generate_step merges them into the model's kwargs and the Qwen language models slice that array at the cache offset — past its end for any continuation generate_step chunks — so the fused MRoPE kernel read a zero-length buffer out of bounds and every token sous fed behind a warm cache (a warm turn's new messages, the tail behind a fork, the generation block of every decode) was rotated to position 0 on mlx-vlm 0.7, and to its suffix-local index on 0.6.17. Measured on the default model: keys off by 130–170 %, on the drafter path and off it. The engine now hands both hooks explicit positions covering the cache from 0 through the new tokens plus a zero rope delta, for the model families whose helper returns positions (probed once), which the model slices correctly; keys match a one-pass prefill within the chunk-rounding floor. The new model test compares cached keys, not greedy text, because the mispositioned block reproduced the reference's words exactly." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
git push -u origin fix/vlm-continuation-positions
```

Then:

```bash
ssh m5pro 'cd ~/code/personal/sous-remote && git fetch -q origin && git checkout -q fix/vlm-continuation-positions && git pull -q --ff-only && uv sync -q && HF_HUB_OFFLINE=1 uv run pytest -m model tests/test_engine_vlm.py -vv 2>&1 | tail -25'
```

(`-vv`, not `-v`: `addopts` already carries `-q`, which cancels a single `-v`.) Expected: every test in the file passes, including both parametrisations of `test_vlm_a_continuation_is_positioned_behind_its_cache` (record the two observed `worst` values by adding a temporary `print` if you want them in the report; do not commit it), the two `fork_copy … bit_for_bit` tests (both sides of each now carry the same positions, so bit-exactness holds) and `test_vlm_a_turn_from_the_tools_fork_matches_its_cold_run_token_for_token`. `test_vlm_drafter_speeds_up_generation_on_default_model` loads the 27B model; let it run — it proves only that the drafter still engages and still produces output, not speed; the acceptance and tokens/s pair comes from Task 3, on both branches. Record the whole tail in the task report.

---

### Task 2: Dependency floor and the gotcha

**Files:**
- Modify: `pyproject.toml` (the `mlx-vlm` requirement), `uv.lock` (via `uv lock`)
- Modify: `CLAUDE.md` (Gotchas)

**Interfaces:**
- Consumes: Task 1's `_positions` contract.
- Produces: nothing programmatic.

- [ ] **Step 1: Raise the floor**

In `pyproject.toml` change `"mlx-vlm>=0.6.16",` to `"mlx-vlm>=0.7.0",`. Then:

Run: `uv lock && uv lock --check && git diff --stat uv.lock`
Expected: `uv lock` rewrites only the `requires-dist` line for mlx-vlm in the `sous-mcp` package entry (the resolved version is already 0.7.0); `--check` is clean; the stat shows a two-line change in `uv.lock`.

Why 0.7.0 and not the locked-only view: `generate_step` re-applies caller-supplied `position_ids`/`rope_deltas` after the embedding helper's only from 0.7.0 (`explicit_prompt_metadata`); on 0.6.17 the helper's local positions overwrite the engine's and the fix is inert — measured, not inferred (*What was measured*, the 0.6.17 paragraph). The floor makes the contract installable, not merely locked.

- [ ] **Step 2: The gotcha**

In `CLAUDE.md`, under `## Gotchas`, add this as a new top-level bullet (a `- ` at column 0) between the bullet that begins `- Prompt-cache slots (\`engine/promptcache.py\`) are owned by the thread that` — whose last line ends `…the valve evicted the forks a cold turn had just made.` — and the bullet that begins `- Prompt-cache per-turn gauges`:

```markdown
- Every `prefill()` and `decode()` on the VLM backend hands mlx-vlm explicit
  `position_ids` covering the cache from 0 through the tokens being appended,
  plus a zero `rope_deltas` (`VLMEngine._positions`), for the model families
  whose text-only embedding helper returns positions of its own (probed once
  per engine: the Qwen lineage — `qwen2_vl`, `qwen2_5_vl`, `qwen3_5`,
  `qwen3_vl`, `qwen3_vl_moe`, `qwen3_omni_moe` and the families that reuse
  their model classes; the other mRoPE families return none, position from
  the cache offset themselves, and several would fail on an array of another
  rank). Load-bearing: mlx-vlm's helper
  returns positions local to the ids it is given and `generate_step` merges
  them into the model's kwargs, and the Qwen language models slice that
  array at the cache offset — past its end for any continuation
  `generate_step` chunks — so on 0.7.x the fused MRoPE Metal kernel reads a
  zero-length buffer out of bounds and every continued token lands at
  position 0 (on 0.6.17 the same tokens landed at `0..n-1`; measured
  2026-09-14 on the default model: keys off by 130–170 % either way, prefill
  and decode, drafter or not). Only from 0.7.0 does `generate_step` re-apply
  a caller's positions over the helper's — hence the `mlx-vlm>=0.7.0` floor;
  below it the kwargs are silently overwritten. A suffix-only array is sliced
  the same way, so the array is always absolute from 0; `rope_deltas` goes
  with it so the engine owns the delta the model adopts into the state a
  drafter run nulls and positions the generated tokens from. mlx-vlm's own
  `_prime_cached_prefix_rope_state` does this only on its
  `prompt_cache_state`/APC paths, which sous never enters.
  `tests/test_engine_vlm.py` pins it by comparing cached keys, not greedy
  text — a mispositioned block has reproduced the reference's words exactly
  — reading every layer on a pure-attention model but only the first
  attention layer on a hybrid, whose deeper layers drift by tens of percent
  between a one-pass and a split prefill for reasons that have nothing to
  do with positions. The LM backend needs none
  of this (mlx-lm's text models take positions from the cache offset).
  Slots built before this fix hold mispositioned keys: only a daemon restart
  drops them.
```

- [ ] **Step 3: Verify and commit**

Run: `set -o pipefail; uv run pytest -m "not model" 2>&1 | tail -3 && uv run ty check && uv run ruff check . && uv run ruff format --check . && uv lock --check`
Expected: the Task 1 count, all clean.

```bash
git add pyproject.toml uv.lock CLAUDE.md
git commit -m "chore(deps): require mlx-vlm 0.7.0, document the position contract" -m "The engine's explicit positions are honoured only from 0.7.0, where generate_step re-applies caller metadata over the embedding helper's; on 0.6.17 they are overwritten and the fix is inert (measured). The CLAUDE.md gotcha records why the kwargs exist and which models get them, so a later change cannot drop them as redundant or widen them to a family they would break." -m "Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

---

### Task 3: Default-model gate on the M5 Pro

The small-VLM test proves the mechanism on the two model families; this task proves it on the shipped model with the drafter, which no CI machine can load, and runs one real delegated task through the worker loop on the VLM backend. Orchestrator-run, over ssh, in `~/code/personal/sous-remote` on the branch Task 2 pushed. It installs nothing over the maintainer's daemon and commits nothing.

**Files:**
- None committed. The scripts below live in the SDD workspace and are deleted with it.

- [ ] **Step 1: The key check with the drafter, and the drafter's numbers**

With `WORKSPACE` set to this plan's SDD workspace directory (the one `scripts/sdd-workspace` printed), write `$WORKSPACE/gate_positions.py`:

```python
"""Default model + drafter: keys of the generation block behind a warm cache
versus one explicit-position forward of the block over a copy of the same
cache, plus the drafter's acceptance and tokens/s on a 128-token decode.
Throwaway. PASS/FAIL is the key error alone; the rest is recorded."""

import os
import sys
import time

import mlx.core as mx
from mlx_vlm.speculative.common import speculative_stats_since, speculative_stats_snapshot

sys.path.insert(0, os.path.expanduser("~/code/personal/sous-remote/src"))
from sous.engine.promptcache import fork_copy  # noqa: E402
from sous.engine.vlm import VLMEngine  # noqa: E402

RULES = "\n".join(
    f"{i}. Kitchen rule number {i}: every plate that leaves the pass is checked twice, "
    f"wiped once, and called by the expediter before the runner takes it."
    for i in range(1, 121)
)
MESSAGES = [
    {"role": "system", "content": "You are a terse line cook. Follow the rules below.\n\n" + RULES},
    {"role": "user", "content": "Write a 300-word story about a busy dinner service."},
]
e = VLMEngine(
    "mlx-community/Qwen3.8-27B-4bit",
    temperature=0.0,
    prompt_cache=True,
    draft_id="z-lab/Qwen3.8-27B-DFlash2",
    draft_block_size=3,
    cache_budget=8 << 30,
)
assert e._draft is not None, "the drafter must be on for this gate"
model, _ = e._loaded()
lm = model.language_model
full_ids = e._ids("full", MESSAGES, [])
P = len(e._ids("stable", MESSAGES, []))
N = len(full_ids) - P
ids = mx.array(full_ids[P:])[None]
base = e.new_cache()
e.prefill(base, full_ids[:P])


def fresh():
    c = e.new_cache()
    fork_copy(base, c, e.copy_array)
    return c


def keys(cache):
    return [
        c.keys[..., P : P + N, :].astype(mx.float32)
        for c in cache
        if getattr(c, "keys", None) is not None
    ]


truth = fresh()
out = lm(
    ids,
    cache=truth,
    position_ids=mx.arange(P, P + N, dtype=mx.int32)[None],
    rope_deltas=mx.zeros((1, 1), dtype=mx.int32),
)
mx.eval(out.logits)
want = keys(truth)

decoded = fresh()
seen = []
before = speculative_stats_snapshot(e._draft)
t = time.perf_counter()
text = e.decode(decoded, full_ids[P:], 128, lambda d: seen.append(d.output_tokens))
dt = time.perf_counter() - t
rounds, accepted, drafted = speculative_stats_since(e._draft, before)
gen = seen[-1] if seen else 0
worst = max(
    (mx.abs(g - w).max() / mx.abs(w).max()).item() for g, w in zip(keys(decoded), want, strict=True)
)
print(
    f"P={P} N={N} worst_relative_key_error={worst:.4f} gen_tokens={gen} "
    f"elapsed={dt:.2f}s tokens_per_s={gen / dt:.1f} rounds={rounds} accepted={accepted} "
    f"drafted={drafted} text={text[:60]!r}"
)
print("PASS" if worst < 0.1 else "FAIL")
```

Copy and run it on the branch, then once on `main` for the before/after pair:

```bash
ssh m5pro 'mkdir -p /tmp/sous-gate && cat > /tmp/sous-gate/gate_positions.py' < "$WORKSPACE"/gate_positions.py
ssh m5pro 'cd ~/code/personal/sous-remote && git checkout -q fix/vlm-continuation-positions && git pull -q --ff-only && HF_HUB_OFFLINE=1 uv run python /tmp/sous-gate/gate_positions.py 2>&1 | grep -v "^Fetching" | tail -3'
ssh m5pro 'cd ~/code/personal/sous-remote && git checkout -q main && HF_HUB_OFFLINE=1 uv run python /tmp/sous-gate/gate_positions.py 2>&1 | grep -v "^Fetching" | tail -3; git checkout -q fix/vlm-continuation-positions'
```

Expected on the branch: `PASS` — the script's gate is `worst_relative_key_error < 0.1`, an order of magnitude below the defect — with an observed value around 0.02 (the measured floor over an identical prefix is 0.017 at the block's last token); a PASS between 0.02 and 0.1 needs explaining in the report before the merge. On `main`: ≈1.7 and `FAIL`. Record both lines whole. The acceptance (`accepted / drafted`) and `tokens_per_s` pair is recorded, not gated: the two runs decode different text (the fix changes the story), so it is directional evidence only, and a lower acceptance after the fix is a live outcome, not a contradiction — the tool-loop A/B in #87's harness is where that question gets answered.

- [ ] **Step 2: One real delegated task through the worker loop, on the VLM backend**

`scripts/e2e_smoke.py` runs the agent loop in-process — a worker thread, an in-process `EngineManager`, a temporary project and data dir — which is why it cannot touch the maintainer's daemon or `~/.sous`; but its model is a text-only 0.6B that `select_backend` routes to the LM backend, which this plan does not touch. The gate runs a copy of it pointed at the hybrid small VLM instead, so the run goes through `VLMEngine`, its warm turns continue the cache through the fixed hooks, and the generation-touching worker thread exits through `release_mlx_thread_state()` — the hazard the CLAUDE.md rule about mlx threads exists for, which Task 1's model test (main thread only) does not exercise.

The second `sed` blanks the drafter: `SousConfig` defaults it to the 27B's DFlash2, which the VLM branch would now try to load against the 9B and reject with a `sous: speculative drafter … unavailable` warning — harmless, but noise the gate does not need.

```bash
sed -e 's#mlx-community/Qwen3-0.6B-4bit#mlx-community/Qwen3.5-9B-MLX-4bit#' -e 's#model_id=TINY,#model_id=TINY, speculative_draft_id="",#' scripts/e2e_smoke.py > "$WORKSPACE"/smoke_vlm.py
grep -c 'speculative_draft_id=""' "$WORKSPACE"/smoke_vlm.py
ssh m5pro 'cat > /tmp/sous-gate/smoke_vlm.py' < "$WORKSPACE"/smoke_vlm.py
ssh m5pro 'cd ~/code/personal/sous-remote && git checkout -q fix/vlm-continuation-positions && git pull -q --ff-only && HF_HUB_OFFLINE=1 uv run python /tmp/sous-gate/smoke_vlm.py 2>&1 | tail -15'
```

Expected: the `grep -c` prints `1` (both substitutions applied); the loop reaches a finished state (`done`, `failed` or `budget-exhausted` — the script's docstring explains why the state alone is not the signal) with no traceback and no segfault, and the printed `prompt_cache` stats show `reused_tokens` above 0 with `snapshot_bytes` above 0 (warm turns on a hybrid cache went through the fixed continuation path). Record the tail.

- [ ] **Step 3: Record, clean up**

Append all three outputs to the ledger. `ssh m5pro 'rm -rf /tmp/sous-gate'`. Leave `~/code/personal/sous-remote` on the branch; the post-merge reinstall returns it to `main`.

---

## Not in this plan

- The tool-loop A/B the issue's step 3 asks for (parse failures, outcomes, acceptance, tokens/s, fixed and unfixed). The key measurement is the proof of correctness; a quality A/B belongs to the harness #87 defines, which compares sampler settings on the same suite and should run on the fixed positions. Task 3 records the drafter's acceptance and tokens/s on both branches so that harness starts with a number.
- Passing positions for image-bearing prompts (`get_rope_index` over the full prompt). sous is text-only; `_positions` is the text-only answer and says so.
- An upstream mlx-vlm report. The maintainer decides whether to file one; the mechanism paragraph above is the report.
- Reinstalling the maintainer's daemon (post-merge, by the maintainer or with their go-ahead).
