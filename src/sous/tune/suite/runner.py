"""One suite run is one task through sous's own worker loop — the sandbox,
the allowlist, the approval hook, the budgets, the verify commands — on an
engine the arm loaded once for every task; graded afterwards, recorded as
it finishes, and released the way the bench releases its arm."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sous.config import DEFAULT_ALLOWLIST, SousConfig
from sous.context import ContextDecision
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
from sous.tasks import TaskState, TaskStore
from sous.tune.arms import Arm
from sous.tune.bench import (
    _INFLIGHT_WAIT_SECONDS,
    BenchRow,
    _active_memory,
    _check_drafter,
    release,
)
from sous.tune.suite import SuiteTask
from sous.tune.suite.grading import grade_task
from sous.worker import run_task

SUITE_ALLOWLIST = (*DEFAULT_ALLOWLIST, "python -m unittest", "python3 -m unittest")
_POLL_SECONDS = 0.1
# The ETA's picture of one run — the turns a task takes and what each turn
# prefills and decodes — from the M5 Pro's delegated tasks of 2026-09.
ETA_TURNS = 8
ETA_PROMPT_TOKENS = 3000
ETA_OUTPUT_TOKENS = 300


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
    def key(self) -> tuple[str, str, int, bool, bool]:
        """The arm this run measured — `Arm.suite_key`, never the label."""
        return (self.model_id, self.drafter_id, self.block_size, self.int8_prefill, self.greedy)

    @property
    def completed(self) -> bool:
        return self.state == TaskState.DONE

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SuiteRun:
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls)})


@dataclass(frozen=True)
class SuiteOutcome:
    runs: list[SuiteRun]
    # False when the weights stayed resident after the arm: the caller must
    # stop, as it does for a bench row, rather than load a second model.
    released: bool
    error: str | None


class CountingEngine:
    """The real engine behind a callback that sums what every generate
    produced: run_task passes no on_delta of its own, so this is the one
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
        # drafter, positions, int8_prefill_status: whatever the backend has,
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
        if kind == "malformed":
            malformed += 1
        elif kind == "tool":
            call = (event.get("name"), json.dumps(event.get("arguments"), sort_keys=True))
            streak = streak + 1 if call == last else 1
            last = call
            if streak == 3:
                repetitions += 1
    return malformed, repetitions


@contextlib.contextmanager
def interpreter_first(python: Path) -> Iterator[None]:
    """`python` on the worker's PATH resolves to the tune's own interpreter
    for the span: the sandbox passes PATH through (toolexec.scrubbed_env)
    and the `python -m unittest` the worker runs must be the Python the
    graders use, with the standard library and nothing else."""
    before = os.environ.get("PATH")
    os.environ["PATH"] = os.pathsep.join([str(python.parent), *([before] if before else [])])
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = before


class _Denier:
    """Denies every approval the worker asks for the moment it appears, and
    counts it: a suite run has no human, and a command outside the
    allowlist must cost the run a denial, not the approval timeout."""

    def __init__(self, store: TaskStore, task_id: str, poll: float):
        self._store, self._task_id, self._poll = store, task_id, poll
        self.denied = 0
        # Set once a store call raises: TaskStore._conn's own comment notes
        # the WAL pragma can raise SQLITE_BUSY under contention. A run whose
        # denier died silently would let every approval ride out the full
        # timeout instead, mislabelling a budget-exhausted run as measured.
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="sous-tune-denier", daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._store.get(self._task_id)
                # respond_approval is true only for the decision that took
                # effect, so a request seen twice before the worker polls it
                # is counted once.
                if (
                    task is not None
                    and task.state == TaskState.AWAITING_APPROVAL
                    and self._store.respond_approval(self._task_id, approve=False)
                ):
                    self.denied += 1
            except Exception as e:  # noqa: BLE001 — transient; keep looping, don't kill the denier
                self.error = f"{type(e).__name__}: {e}"
            self._stop.wait(self._poll)

    def __enter__(self) -> _Denier:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(5.0)


def _scratch_config(arm: Arm, scratch: Path, task: SuiteTask) -> SousConfig:
    """The arm's config over a scratch control directory: the worker reads
    the allowlist from config_path on every command, so the file has to
    exist, and the data dir holds this run's store and transcript."""
    config_path = scratch / "config.toml"
    entries = ", ".join(json.dumps(e) for e in SUITE_ALLOWLIST)
    config_path.write_text(f"[commands]\nallowlist = [{entries}]\n")
    return dataclasses.replace(
        arm.config,
        data_dir=scratch / "data",
        config_path=config_path,
        max_turns=task.max_turns,
        max_minutes=task.max_minutes,
        context_mode="fixed",
        max_context_tokens=arm.window,
    )


def _error_run(task: SuiteTask, index: int, arm: Arm, error: str) -> SuiteRun:
    return SuiteRun(
        task=task.name,
        index=index,
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        int8_prefill=arm.int8_prefill,
        greedy=arm.greedy,
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
    poll: float = _POLL_SECONDS,
    grade_timeout: float = 120.0,
) -> SuiteRun:
    """One task, once, through run_task — the loop the daemon runs, minus
    the queue polling around it — then the grade over what it left."""
    project = scratch / "project"
    shutil.copytree(task.project, project)
    config = _scratch_config(arm, scratch, task)
    store = TaskStore(scratch / "tasks.db")
    queued = store.enqueue(
        title=task.title,
        instructions=task.instructions,
        project_root=str(project),
        context_files=list(task.context_files),
        verify_commands=list(task.verify_commands),
    )
    claimed = store.claim_next()
    if claimed is None or claimed.id != queued.id:
        raise RuntimeError("the scratch store handed back another task")
    before = counting.output_tokens
    with _Denier(store, claimed.id, poll) as denier, interpreter_first(python):
        run_task(claimed, store, engine, config, context=ContextDecision(arm.window, "tune"))
    final = store.get(claimed.id)
    if final is None:
        raise RuntimeError("the task vanished from the scratch store")
    transcript = config.data_dir / "tasks" / claimed.id / "transcript.jsonl"
    malformed, repetitions = metrics_from_transcript(transcript)
    grade = grade_task(task, project, python=python, timeout=grade_timeout)
    finished = final.finished_at if final.finished_at is not None else time.time()
    started = final.started_at if final.started_at is not None else final.created_at
    report = final.report or {}
    raw_budget = report.get("budget")
    budget = raw_budget if isinstance(raw_budget, dict) else {}
    # run_task counts turns itself in the report's budget block; the store's
    # turns_used is bumped only on tool calls, so it misses the finish turn.
    turns = int(budget.get("turns", final.turns_used))
    state = final.state
    error = None if final.state == TaskState.DONE else str(report.get("error") or final.state)
    if denier.error is not None:
        # The denier is the only thing answering approvals in a suite run;
        # if its thread died mid-run, every request rode out the full
        # approval timeout instead of being denied at once, so this run
        # measured a wedged approval loop, not the arm. Every other field
        # (grade included) still stands — only the label changes.
        state = "error"
        error = f"approval denier failed: {denier.error}"
    return SuiteRun(
        task=task.name,
        index=index,
        label=arm.label,
        model_id=arm.model_id,
        drafter_id=arm.drafter_id,
        block_size=arm.block_size,
        int8_prefill=arm.int8_prefill,
        greedy=arm.greedy,
        window=arm.window,
        state=state,
        outcome=final.outcome,
        turns=turns,
        seconds=finished - started,
        output_tokens=counting.output_tokens - before,
        malformed=malformed,
        repetitions=repetitions,
        approvals_denied=denier.denied,
        grade=grade.score,
        grade_detail=grade.detail,
        error=error,
        transcript_path=str(transcript),
    )


def _check_int8(arm: Arm, engine: ManagedEngine, out: Callable[..., None]) -> None:
    """Like the drafter check: the engine refuses INT8 prefill with a status
    rather than an error. The winner stage's own int8 arm exists to
    *measure* int8 prefill, so the engine refusing it must fail that arm —
    a run under the int8 label over the stock path would put that setting
    in the config on stock numbers. Every other arm only inherited
    int8_prefill from the user's config; the daemon would run it stock with
    one warning rather than refuse it, so this prints a notice instead and
    lets the arm run."""
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
    budget clock starts. run_task can give up on a stalled generation at its
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
    record: Callable[[SuiteRun], None],
    scratch: Path,
    out: Callable[..., None],
    factory: Callable[[str], Engine] | None,
    python: Path,
    active_memory: Callable[[], int],
    poll: float,
) -> SuiteOutcome:
    baseline = active_memory()
    base = factory or default_engine_factory(arm.config)
    counters: list[CountingEngine] = []

    def wrapped(model_id: str) -> Engine:
        counters.append(CountingEngine(base(model_id)))
        return counters[-1]

    manager = EngineManager(arm.config, engine_factory=wrapped)
    try:
        engine = manager.get()
    except Exception as e:  # noqa: BLE001 — the arm is skipped, named; nothing is resident
        return SuiteOutcome([], True, f"load failed: {type(e).__name__}: {e}")
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
        except RuntimeError as e:
            check_error = str(e)
        if check_error is None:
            for task in tasks:
                for index in range(runs):
                    if (task.name, index) in done:
                        continue
                    out(f"  {arm.label}: {task.name} #{index + 1} ...")
                    run_scratch = scratch / _slug(arm.label) / f"{task.name}-{index + 1}"
                    # A resumed run may have died mid-task here; nothing in a
                    # half-run scratch is worth keeping.
                    if run_scratch.exists():
                        shutil.rmtree(run_scratch)
                    run_scratch.mkdir(parents=True)
                    try:
                        result = run_one(
                            task,
                            index,
                            arm,
                            engine,
                            counters[-1],
                            run_scratch,
                            python=python,
                            poll=poll,
                        )
                    except Exception as e:  # noqa: BLE001 — one run's failure, recorded; goes on
                        result = _error_run(task, index, arm, f"{type(e).__name__}: {e}")
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
    poll: float = _POLL_SECONDS,
) -> SuiteOutcome:
    """Every (task, run index) of the arm not in `done`, on a thread of its
    own that loads the engine once, hands each run to `record` as it
    finishes, and releases its mlx state on the way out."""
    outcome: list[SuiteOutcome | BaseException] = []

    def run() -> None:
        try:
            outcome.append(
                _run_suite(
                    arm,
                    tasks,
                    runs,
                    done,
                    record,
                    scratch,
                    out,
                    factory,
                    python or Path(sys.executable),
                    active_memory or _active_memory,
                    poll,
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
