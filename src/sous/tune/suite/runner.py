"""One suite run is one task through the suite's own loop — its scratch
tools, its budget, its transcript — on an engine the arm loaded once for
every task; graded afterwards, recorded as it finishes, and released the
way the bench releases its arm."""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from sous.engine.base import (
    Delta,
    Engine,
    EngineManager,
    ManagedEngine,
    OnDelta,
    ReplaySafe,
    default_engine_factory,
    release_mlx_thread_state,
)
from sous.tune.arms import Arm
from sous.tune.bench import (
    _GIB,
    _INFLIGHT_WAIT_SECONDS,
    _UNLOAD_SLACK_BYTES,
    BenchRow,
    _active_memory,
    _check_drafter,
    release,
    settle_resident,
)
from sous.tune.suite import SuiteTask
from sous.tune.suite.grading import grade_task
from sous.tune.suite.loop import EVENT_MALFORMED, EVENT_TOOL, Budget, Transcript, run_loop
from sous.tune.suite.tools import ScratchTools

# The ETA's picture of one run — the turns a task takes and what each turn
# prefills and decodes — from the M5 Pro's delegated tasks of 2026-09.
ETA_TURNS = 8
ETA_PROMPT_TOKENS = 3000
ETA_OUTPUT_TOKENS = 300

# A suite row's projection_tile: a state a load reported, or, on a row written
# before rows recorded it, the setting as written.
TileState = bool | str


@dataclass(frozen=True)
class SuiteRun:
    task: str
    index: int
    label: str
    model_id: str
    drafter_id: str
    block_size: int
    int8_prefill: bool
    greedy: bool
    attention_tile: bool
    projection_kernel: bool
    kv_int8: bool
    # What the load reported (loaded_tile): "all", "mlp", "unavailable" or
    # "off", never the setting, since true picks a scope at load and the two
    # scopes verify differently. A row written before that holds the setting as
    # written (true or false, or "mlp" or "all").
    projection_tile: bool | str
    window: int
    state: str
    outcome: str | None
    turns: int
    seconds: float
    output_tokens: int
    malformed: int
    repetitions: int
    approvals_denied: int
    grade: float
    grade_detail: str
    error: str | None
    transcript_path: str | None

    @property
    def key(self) -> tuple[str, str, int, bool, bool, bool, bool, bool, bool | str]:
        """The arm this run measured — `Arm.suite_key`, never the label — with
        the projection tile as its load reported it (arm_runs matches the
        two)."""
        return (
            self.model_id,
            self.drafter_id,
            self.block_size,
            self.int8_prefill,
            self.greedy,
            self.attention_tile,
            self.projection_kernel,
            self.kv_int8,
            self.projection_tile,
        )

    @property
    def completed(self) -> bool:
        return self.state == "done"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SuiteRun:
        # A row written before a setting existed ran without what it switches
        # on — stock attention, stock projections, bf16 K/V, no projection
        # tile — on every machine, so it reads False, and a resume under an
        # arm that has the setting runs the task again rather than pool the
        # two. Without the fallback, resuming such a run would raise KeyError.
        d = {
            "attention_tile": False,
            "projection_kernel": False,
            "kv_int8": False,
            "projection_tile": False,
            **d,
        }
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


def loaded_tile(status: dict | None) -> str:
    """The projection tile as a load reported it, as a suite row records it: the
    scope ("all" or "mlp") while active, else the state ("unavailable", "off");
    an engine with no status has no tile, which is "off"."""
    status = status or {}
    state = status.get("state") or "off"
    return (status.get("scope") or state) if state == "active" else state


def _tile_states(setting: TileState) -> frozenset[TileState]:
    """The row values a load under this projection_tile setting can have written:
    every state a load can report under it, and the setting itself, as rows
    written before the state was recorded hold it. An engine without the tile
    reports "off" under any setting; under false nothing else."""
    if setting is False:
        return frozenset({"off", False})
    scopes = {"all", "mlp"} if setting is True else {setting}
    return frozenset({*scopes, "unavailable", "off", setting})


def done_pairs(rows: Iterable[SuiteRun]) -> set[tuple[str, int]]:
    """The (task, index) pairs these rows finished. Only the latest attempt at a
    pair decides — a retry leaves its earlier row in the run's rows too,
    appended rather than replaced — and a runner-only "error" row (TaskState has
    no such state) is not a result to count as done; a task that ran and failed
    is."""
    latest: dict[tuple[str, int], SuiteRun] = {}
    for r in rows:
        latest[(r.task, r.index)] = r
    return {key for key, r in latest.items() if r.state != "error"}


def arm_runs(
    arm: Arm, runs: Iterable[SuiteRun]
) -> tuple[dict[TileState, list[SuiteRun]], TileState | None]:
    """The rows of ``runs`` this arm measured, by the tile state each ran under,
    and the state a reader takes (None when there is no row): the arm's key but
    for the tile, its window, and a state a load of its setting reports. Rows
    of two states never pool. The state taken is the one whose rows finished
    the most (task, index) pairs, the latest of those tied: a resume loads only
    while no state is complete, so a finished measurement is never set aside
    for a later load's partial one. Under false a row written before the state
    was recorded ran off too, and joins the "off" rows; under any other setting
    such a row's scope is unknown, and it keeps a group of its own."""
    allowed = _tile_states(arm.projection_tile)
    groups: dict[TileState, list[SuiteRun]] = {}
    last: dict[TileState, int] = {}
    for i, r in enumerate(runs):
        if r.window != arm.window or r.key[:-1] != arm.suite_key[:-1]:
            continue
        if r.projection_tile not in allowed:
            continue
        state = "off" if r.projection_tile is False else r.projection_tile
        groups.setdefault(state, []).append(r)
        last[state] = i
    if not groups:
        return groups, None

    def finished(state: TileState) -> tuple[int, int]:
        return len(done_pairs(groups[state])), last[state]

    return groups, max(groups, key=finished)


@dataclass(frozen=True)
class SuiteOutcome:
    runs: list[SuiteRun]
    # False when the weights stayed resident after the arm: the caller must
    # stop, as it does for a bench row, rather than load a second model.
    released: bool
    error: str | None


class CountingEngine:
    """The real engine behind a callback that sums what every generate
    produced: run_loop passes no on_delta of its own, so this is the one
    reader of Delta.output_tokens on a suite run. The callback is
    ReplaySafe — nothing it sees leaves the process — so a warm attempt
    that fails may still be retried cold, as with no callback at all."""

    def __init__(self, inner: Engine):
        self._inner = inner
        self.output_tokens = 0
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def __getattr__(self, name: str):
        # drafter, draft_block, positions, int8_prefill_status,
        # verify_attention_status, draft_context_status, attention_tile_status,
        # projection_kernel_status, projection_kernel_drafter_status,
        # projection_tile_status, kv_int8_status: whatever the backend has,
        # read through getattr(..., None) by ManagedEngine.
        return getattr(self._inner, name)

    def generate(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        on_delta: OnDelta | None = None,
    ) -> str:
        produced = [0]

        def count(d: Delta) -> None:
            # The count restarts on a cold retry; the attempt that finished
            # is the largest one seen.
            produced[0] = max(produced[0], d.output_tokens)
            if on_delta is not None:
                on_delta(d)

        already_safe = on_delta is None or isinstance(on_delta, ReplaySafe)
        callback = ReplaySafe(count) if already_safe else count
        text = self._inner.generate(messages, tools, max_tokens, callback)
        with self._lock:
            self.output_tokens += produced[0]
        return text

    def count_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return self._inner.count_tokens(messages, tools)

    def reset_prompt_cache(self, owner: threading.Thread | None = None) -> None:
        self._inner.reset_prompt_cache(owner)

    def prompt_cache_stats(self, owner: threading.Thread | None = None) -> dict:
        return self._inner.prompt_cache_stats(owner)

    def unload(self) -> None:
        self._inner.unload()


def metrics_from_transcript(path: Path) -> tuple[int, int]:
    """(malformed tool calls, repetition incidents) from a task's
    transcript. An incident is the model looping on one tool: three identical
    consecutive executed calls, counted once per streak of three or more. A
    `tool` event is exactly one executed call with its arguments, so nothing
    needs re-parsing; a `finish` reaches the transcript as a tool event only
    when it lacked a summary, and three of those in a row are a loop too."""
    if not path.is_file():
        return 0, 0
    malformed = repetitions = streak = 0
    last: tuple | None = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("event") if isinstance(event, dict) else None
        if kind == EVENT_MALFORMED:
            malformed += 1
        elif kind == EVENT_TOOL:
            call = (event.get("name"), json.dumps(event.get("arguments"), sort_keys=True))
            streak = streak + 1 if call == last else 1
            last = call
            if streak == 3:
                repetitions += 1
    return malformed, repetitions


@contextlib.contextmanager
def interpreter_first(python: Path) -> Iterator[None]:
    """`python` on PATH resolves to the tune's own interpreter for the span:
    ScratchTools.run_command inherits this process's environment untouched,
    and the test runner a task invokes must be the Python the graders use,
    with the standard library and nothing else."""
    before = os.environ.get("PATH")
    os.environ["PATH"] = os.pathsep.join([str(python.parent), *([before] if before else [])])
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = before


def _error_run(task: SuiteTask, index: int, arm: Arm, error: str, projection_tile: str) -> SuiteRun:
    return SuiteRun(
        task=task.name,
        index=index,
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        int8_prefill=arm.int8_prefill,
        greedy=arm.greedy,
        attention_tile=arm.attention_tile,
        projection_kernel=arm.projection_kernel,
        kv_int8=arm.kv_int8,
        projection_tile=projection_tile,
        window=arm.window,
        state="error",
        outcome=None,
        turns=0,
        seconds=0.0,
        output_tokens=0,
        malformed=0,
        repetitions=0,
        approvals_denied=0,
        grade=0.0,
        grade_detail="",
        error=error,
        transcript_path=None,
    )


def run_one(
    task: SuiteTask,
    index: int,
    arm: Arm,
    engine: ManagedEngine,
    counting: CountingEngine,
    scratch: Path,
    *,
    python: Path,
    work: Path | None = None,
    grade_timeout: float = 120.0,
) -> SuiteRun:
    """One task, once, through the suite loop, then the grade over what it
    left in the scratch copy of the project.

    The copy runs at `work` when given and is moved to `scratch / "project"`
    afterwards, beside the transcript: the system prompt names the project
    root, so a root built from the arm's label or the run index would hand
    each arm, and each run, a prompt of its own (#149). The run samples from
    seed `index`, so run i of every arm draws from the same stream."""
    archive = scratch / "project"
    project = work or archive
    shutil.copytree(task.project, project)
    transcript = Transcript(scratch / "transcript.jsonl")
    tools = ScratchTools(project, task.verify_commands)
    before = counting.output_tokens
    try:
        with interpreter_first(python):
            result = run_loop(
                instructions=task.instructions,
                context_files=task.context_files,
                root=project,
                engine=engine,
                window=arm.window,
                budget=Budget(turns=task.max_turns, minutes=task.max_minutes),
                tools=tools,
                transcript=transcript,
                seed=index,
            )
        grade = grade_task(task, project, python=python, timeout=grade_timeout)
    finally:
        if project != archive:
            shutil.move(project, archive)
    malformed, repetitions = metrics_from_transcript(transcript.path)
    failed = result.error is not None
    return SuiteRun(
        task=task.name,
        index=index,
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        int8_prefill=arm.int8_prefill,
        greedy=arm.greedy,
        attention_tile=arm.attention_tile,
        projection_kernel=arm.projection_kernel,
        kv_int8=arm.kv_int8,
        projection_tile=loaded_tile(engine.projection_tile_status),
        window=arm.window,
        state="failed" if failed else "done",
        outcome=None if failed else result.outcome,
        turns=result.turns,
        seconds=result.seconds,
        output_tokens=counting.output_tokens - before,
        malformed=malformed,
        repetitions=repetitions,
        approvals_denied=tools.denied,
        grade=grade.score,
        grade_detail=grade.detail,
        error=result.error,
        transcript_path=str(transcript.path),
    )


def _check_int8(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """Like the drafter check: the engine refuses INT8 prefill with a status
    rather than an error. The winner stage's own int8 arm exists to
    *measure* int8 prefill, so the engine refusing it must fail that arm —
    a run under the int8 label over the stock path would put that setting
    in the config on stock numbers. Every other arm only inherited
    int8_prefill from the user's config; the daemon would run it stock, with
    the reason in its status, rather than refuse it, so this prints a notice
    instead and lets the arm run."""
    if not arm.int8_prefill:
        return
    status = engine.int8_prefill_status or {}
    if status.get("state") == "active":
        return
    reason = status.get("reason") or "no status"
    if arm.int8_under_test:
        raise RuntimeError(
            f"int8 prefill requested, engine reports {status.get('state', 'nothing')}: {reason}"
        )
    out(
        f"  {arm.label}: int8 prefill unavailable here ({reason}); "
        "the engine runs stock, as the daemon would"
    )


def _notice_unavailable(
    arm: Arm,
    what: str,
    status: dict | None,
    out: Callable[..., None],
    *,
    runs: str = "stock",
) -> None:
    """The notice for a setting the arm inherits and the engine reports
    unavailable; any other state, or no status, prints nothing. `runs` names
    what the engine falls back to."""
    status = status or {}
    if status.get("state") != "unavailable":
        return
    reason = status.get("reason") or "no reason given"
    out(
        f"  {arm.label}: {what} unavailable here ({reason}); "
        f"the engine runs {runs}, as the daemon would"
    )


def _check_tile(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """Every arm inherits the attention tile from the user's config and none
    exists to measure it, so unlike int8 there is nothing to fail: where the
    engine reports the tile unavailable, the daemon would serve this
    checkpoint with stock attention, and the arm runs the same way after a
    notice. `off`, or no status, is an engine with no tile (the mlx-lm
    backend), not a machine the tile cannot run on: no notice."""
    if arm.attention_tile:
        _notice_unavailable(arm, "attention tile", engine.attention_tile_status, out)


def _check_proj(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """The projection kernel is inherited and unmeasured like the tile, so
    the same holds: where the engine reports it unavailable, the daemon would
    serve this checkpoint with stock projections, and the arm runs the same
    way after a notice. Its block is unaffected — every drafter arm pins the
    one its label names. The drafter's own linears ride on the kernel under a
    status of their own, so an arm whose drafter route is unavailable gets a
    notice of its own and drafts on stock projections. `off`, or no status,
    is the setting switched off or an engine without the kernel: no notice."""
    if arm.projection_kernel:
        _notice_unavailable(arm, "projection kernel", engine.projection_kernel_status, out)
        _notice_unavailable(
            arm,
            "projection kernel drafter",
            engine.projection_kernel_drafter_status,
            out,
            runs="the drafter's own projections stock",
        )


def _check_projtile(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """The projection tile is inherited and unmeasured like the kernel, so the
    same holds: where the engine reports it unavailable, the daemon would
    serve this checkpoint without it, and the arm runs the same way after a
    notice. Except for `no drafter`: the tile loads only beside a drafter,
    so every drafterless arm on a machine it runs on gets that refusal, and
    a notice there would say nothing. `off`, or no status, is
    the setting switched off or an engine without the tile: no notice."""
    if not arm.projection_tile:
        return
    status = engine.projection_tile_status or {}
    if (status.get("reason") or "").startswith("no drafter"):
        return
    _notice_unavailable(arm, "projection tile", status, out, runs="without it")


def _check_kv8(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """int8 K/V is inherited and unmeasured like the tile, so the same holds:
    where the engine reports it unavailable, the daemon would serve this
    checkpoint's attention from the bf16 cache, and the arm runs the same way
    after a notice. The fallback is bf16 K/V, not stock: where only kv8
    refused, the tile still serves. `off`, or no status, is the setting
    switched off or an engine without it: no notice."""
    if arm.kv_int8:
        _notice_unavailable(arm, "int8 K/V", engine.kv_int8_status, out, runs="bf16 K/V")


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)


_DRAIN_POLL_SECONDS = 1.0


def _drain(
    engine: ManagedEngine,
    *,
    label: str,
    out: Callable[..., None],
    wait: float = _INFLIGHT_WAIT_SECONDS,
) -> None:
    """Wait out this run's own abandoned generation before the next run's
    budget clock starts. run_loop can give up on a stalled generation at its
    wall-clock deadline (recorded budget-exhausted) while the engine's
    session thread keeps decoding under ManagedEngine._gen_lock until it
    reaches its token cap — minutes on a slow model. Left alone, the next
    run's first generate() would block on that lock for free, spending its
    own budget waiting rather than doing anything."""
    deadline = time.monotonic() + wait
    waited = False
    while engine.generation_in_flight() and time.monotonic() < deadline:
        if not waited:
            out(f"  {label}: waiting for an abandoned generation to end before the next run")
            waited = True
        time.sleep(_DRAIN_POLL_SECONDS)


def _run_suite(
    arm: Arm,
    tasks: list[SuiteTask],
    runs: int,
    done: set[tuple[str, int]],
    done_by_tile: Mapping[TileState, set[tuple[str, int]]] | None,
    record: Callable[[SuiteRun], None],
    scratch: Path,
    out: Callable[..., None],
    factory: Callable[[str], Engine] | None,
    python: Path,
    active_memory: Callable[[], int],
    work: Path,
) -> SuiteOutcome:
    baseline = active_memory()
    base = factory or default_engine_factory(arm.config, forks=False)
    counters: list[CountingEngine] = []

    def wrapped(model_id: str) -> Engine:
        counters.append(CountingEngine(base(model_id)))
        return counters[-1]

    manager = EngineManager(arm.config, engine_factory=wrapped)
    engine: ManagedEngine | None = None
    load_error = ""
    try:
        engine = manager.get()
    except Exception as e:  # noqa: BLE001 — the arm is skipped, named
        load_error = f"load failed: {type(e).__name__}: {e}"
    if engine is None:
        # Judged only here, outside the except block: the exception's
        # traceback held the loader thread's frames, and with them the
        # half-built engine and its arrays, so a reading taken while it was
        # bound would have called every failed load "still resident". A
        # load that failed partway (a drafter that would not fit beside a
        # target already mapped) can still leave weights behind through
        # other references, and the next arm must not load beside them.
        gc.collect()
        resident = settle_resident(active_memory, baseline)
        if resident > _UNLOAD_SLACK_BYTES:
            out(f"  {arm.label}: {resident / _GIB:.1f} GiB still resident after the failed load")
            load_error += f"; memory not released: {resident / _GIB:.1f} GiB still resident"
        return SuiteOutcome([], resident <= _UNLOAD_SLACK_BYTES, load_error)
    results: list[SuiteRun] = []
    check_error: str | None = None
    loop_error: str | None = None
    # Everything that can raise after the load — the checks, the whole task
    # loop, `record`, `out`, the scratch-directory bookkeeping — is inside
    # this try, so the finally below always releases the weights: a raise
    # anywhere here used to skip release() and still report released=True.
    try:
        try:
            _check_drafter(arm, engine)
            _check_int8(arm, engine, out)
            _check_tile(arm, engine, out)
            _check_proj(arm, engine, out)
            _check_projtile(arm, engine, out)
            _check_kv8(arm, engine, out)
        except RuntimeError as e:
            check_error = str(e)
        tile = loaded_tile(engine.projection_tile_status)
        # Only the rows of the state this load reported count as done: rows of
        # another scope verified another way, and the runs go again under this
        # one rather than pool with them.
        if done_by_tile is not None:
            done = done_by_tile.get(tile, set())
        if check_error is None:
            for task in tasks:
                for index in range(runs):
                    if (task.name, index) in done:
                        continue
                    out(f"  {arm.label}: {task.name} #{index + 1} ...")
                    run_scratch = scratch / _slug(arm.label) / f"{task.name}-{index + 1}"
                    # A resumed run may have died mid-task here; nothing in a
                    # half-run scratch is worth keeping.
                    for stale in (run_scratch, work):
                        if stale.exists():
                            shutil.rmtree(stale)
                    run_scratch.mkdir(parents=True)
                    work.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        result = run_one(
                            task,
                            index,
                            arm,
                            engine,
                            counters[-1],
                            run_scratch,
                            python=python,
                            work=work,
                        )
                    except Exception as e:  # noqa: BLE001 — one run's failure, recorded; goes on
                        result = _error_run(task, index, arm, f"{type(e).__name__}: {e}", tile)
                    # Before record(): a budget-exhausted run's generation can
                    # still be decoding under the engine's lock, and the next
                    # run's own budget clock must not start against it.
                    _drain(engine, label=arm.label, out=out)
                    # Appended before record() runs, so a run already
                    # finished is kept even when the callback itself raises.
                    results.append(result)
                    record(result)
                    out(
                        f"    {task.name} #{index + 1}: {result.state}, grade {result.grade:.2f}, "
                        f"{result.seconds:.0f} s, {result.turns} turn(s)"
                    )
    except Exception as e:  # noqa: BLE001 — the arm's own failure; still released in finally
        loop_error = f"{type(e).__name__}: {e}"
    finally:
        teardown = release(
            manager, None, baseline=baseline, active_memory=active_memory, label=arm.label, out=out
        )
    return SuiteOutcome(results, teardown is None, teardown or loop_error or check_error)


def run_suite(
    arm: Arm,
    tasks: list[SuiteTask],
    *,
    runs: int,
    done: set[tuple[str, int]],
    record: Callable[[SuiteRun], None],
    scratch: Path,
    out: Callable[..., None] = print,
    factory: Callable[[str], Engine] | None = None,
    python: Path | None = None,
    active_memory: Callable[[], int] | None = None,
    work: Path | None = None,
    done_by_tile: Mapping[TileState, set[tuple[str, int]]] | None = None,
) -> SuiteOutcome:
    """Every (task, run index) of the arm not in `done`, on a thread of its
    own that loads the engine once, hands each run to `record` as it
    finishes, and releases its mlx state on the way out. With `done_by_tile`
    (the done set of each tile state, arm_runs' groups), the done set is the
    one of the state the load reports instead, since a resume's load can pick
    another scope than the rows' did. Every run's project runs at `work`, by
    default `project` beside `scratch` (outside every arm's archive), before
    it is archived under `scratch`: arms compared on one `work` see one
    prompt."""
    outcome: list[SuiteOutcome | BaseException] = []

    def run() -> None:
        try:
            outcome.append(
                _run_suite(
                    arm,
                    tasks,
                    runs,
                    done,
                    done_by_tile,
                    record,
                    scratch,
                    out,
                    factory,
                    python or Path(sys.executable),
                    active_memory or _active_memory,
                    work or scratch.parent / "project",
                )
            )
        except BaseException as e:  # noqa: BLE001 — becomes the outcome's error
            outcome.append(e)
        finally:
            release_mlx_thread_state()

    worker = threading.Thread(target=run, name=f"sous-tune-suite-{arm.label}", daemon=True)
    worker.start()
    worker.join()
    result = outcome[0]
    if isinstance(result, BaseException):
        # _run_suite now releases on every ordinary-Exception path and only
        # lets a BaseException (an interrupt) past that release, so only one
        # can reach here — and whether the weights actually came off is then
        # unknown, not "definitely not resident": the caller must stop
        # rather than load a second model over an uncertain interrupt.
        return SuiteOutcome([], False, f"{type(result).__name__}: {result}")
    return result


def estimate_seconds(
    arms: list[Arm], rows: list[BenchRow], *, tasks: int, runs: int
) -> float | None:
    """A rough suite duration from the quick stage's speeds: every arm's
    short-context prefill and decode over ETA_TURNS turns of a typical
    task's prompt and answer, times the tasks and runs. None when an arm has
    no usable row — a guess would be read as a measurement."""
    by_key = {(r.key, r.window): r for r in rows if r.ok}
    total = 0.0
    for arm in arms:
        row = by_key.get((arm.key, arm.window))
        if row is None or not row.prefill_tps_2k or not row.decode_tps_1k:
            return None
        per_turn = ETA_PROMPT_TOKENS / row.prefill_tps_2k + ETA_OUTPUT_TOKENS / row.decode_tps_1k
        total += ETA_TURNS * per_turn * tasks * runs
    return total
