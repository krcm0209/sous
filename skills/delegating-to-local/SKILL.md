---
name: delegating-to-local
description: Use when coding work is mechanical, repetitive, volume-heavy, and low-risk (boilerplate, test scaffolding, bulk renames/migrations, docstrings, lint-fix sweeps, fixtures) - delegates it to the local sous MLX worker via MCP so you stay focused on reasoning, architecture, and review, and the volume output costs none of the user's Claude plan
---

# Delegating to the local sous worker

sous runs a local MLX model that executes self-contained coding tasks in a
sandboxed tool loop. You are the head chef: design, decide, review. sous is
prep: volume and repetition. Every line the worker generates is output the
user's plan didn't pay for, so prefer delegating work that qualifies.
**Its output is a draft, never a merge.**

## When to delegate

- Mechanical + repetitive + low-risk: boilerplate, test scaffolding, bulk
  renames/migrations, docstring/comment sweeps, lint fixes, fixture generation.
- Pays only when a short spec yields a large diff. If you find yourself
  authoring the content inside the instructions (taste-heavy prose, judgment
  calls), do it inline — the spec would cost more than it saves.
- NOT: architecture, subtle debugging, security-sensitive code, anything that
  needs conversation context, API design, or taste.

## How to delegate

1. Write **self-contained** instructions — the worker sees nothing of this
   chat. State the *what*: goal, scope limits (files it must not touch),
   explicit acceptance criteria. Trust the worker with the *how*: don't
   pre-author content, don't restate conventions a `context_files` entry
   already carries, don't prescribe what `verify_commands` will catch
   mechanically. Worker attempts are free and your prompt is not — start
   lean, review the miss, re-delegate narrower.
2. Call `delegate_to_local_model` with `project_root` (absolute path),
   `context_files` (files it should read first, including convention docs
   like CLAUDE.md), and `verify_commands` (allowlisted test/lint commands
   proving the work).
3. **Mirror into your native task list**: create a task (TaskCreate) named
   "sous: <title>" when you delegate, update it as status changes, complete
   it when you collect the result.
4. If your next step depends on the worker's result, block on
   `sous wait <task_id>` in the foreground — an executor that backgrounds
   the wait with nothing else to do just ends its turn and stalls.
   Otherwise keep doing your own work and check `task_status` between your
   own steps, or park `sous wait <task_id>` in a background shell. Never a tight
   loop, and never read `~/.sous/tasks.db` directly (internal schema, not
   a contract).

## Executing an implementation plan

A written plan is the ideal delegation source: the spec already exists, so
the marginal prompt cost is near zero.

- Batch adjacent mechanical plan tasks that share a pattern into ONE
  delegation — one worker round-trip, one review.
- Lift scope limits straight from the plan's task boundaries ("Task 3 is
  out of scope — do not touch `__init__.py`").
- Keep the plan's judgment tasks (API design, naming, trade-off calls)
  inline — delegating them just moves the thinking into your prompt.
- Review at the plan's own checkpoints: the worker's diff must meet the
  task's verify criterion before the next task starts.

## While it runs

- `awaiting_approval` state: the worker wants to run the command in
  `pending_command`. Relay it to the human verbatim and ask approve once /
  add to allowlist / deny. Answer with `respond_to_command_request`.
  Unanswered requests auto-deny after a timeout, so relay promptly.
  Note: allowlisting any command that executes repo-resident code (test
  runners especially) grants the worker arbitrary local execution via files
  it writes — prefer approve-once unless the human clearly wants it standing.
- Don't edit files the running task is touching (`last_activity` shows where
  it is working).

## Collecting results

- `task_result` with `include_diff=true`. **Review the diff** like a PR from
  an eager junior: check acceptance criteria, run your own verification.
- Keep the plan-side output lean — it's the other half of the economics. A
  clean diff earns a two-sentence acceptance; re-narrating good work spends
  the tokens delegation just saved. Report what you *changed or rejected*,
  not what you merely confirmed.
- On a miss, re-delegate a narrower **self-contained** task scoped to just
  the flaws. Each task starts a fresh worker with no memory of the last one,
  so "fix these two contracts" means nothing to it: name the file, the
  specific defects, and the acceptance criteria for the fix. The follow-up
  stays short because the scope shrank, not because context was omitted.
  Say only that you're re-instructing the worker before moving on.
- `budget-exhausted` outcome = partial work; review what landed, then either
  finish it yourself or delegate a narrower follow-up.
- The full transcript path is in the report if you need to audit behavior.

## If the MCP server is unreachable

Self-heal before giving up, via Bash:

```bash
sous status
```

Not running? Boot it (first try launchd, then direct):

```bash
launchctl kickstart gui/$(id -u)/com.sous.daemon || (nohup sous serve >/dev/null 2>&1 &)
```

Wait ~3 seconds, then retry the MCP call. If `sous` isn't installed at all,
tell the human instead of improvising.
