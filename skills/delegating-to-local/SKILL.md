---
name: delegating-to-local
description: Use when coding work is mechanical, repetitive, volume-heavy, and low-risk (boilerplate, test scaffolding, bulk renames/migrations, docstrings, lint-fix sweeps, fixtures) - delegates it to the local sous MLX worker via MCP so you stay focused on reasoning, architecture, and review
---

# Delegating to the local sous worker

sous runs a local MLX model that executes self-contained coding tasks in a
sandboxed tool loop. You are the head chef: design, decide, review. sous is
prep: volume and repetition. **Its output is a draft, never a merge.**

## When to delegate

- Mechanical + repetitive + low-risk: boilerplate, test scaffolding, bulk
  renames/migrations, docstring/comment sweeps, lint fixes, fixture generation.
- NOT: architecture, subtle debugging, security-sensitive code, anything that
  needs conversation context, API design, or taste.

## How to delegate

1. Write **self-contained** instructions — the worker sees nothing of this
   chat. Include: goal, constraints, target files/patterns, explicit
   acceptance criteria.
2. Call `delegate_task` with `project_root` (absolute path), `context_files`
   (files it should read first), and `verify_commands` (allowlisted test/lint
   commands proving the work).
3. **Mirror into your native task list**: create a task (TaskCreate) named
   "sous: <title>" when you delegate, update it as status changes, complete
   it when you collect the result.
4. Keep doing your own work. Poll `task_status` between your own steps, not
   in a tight loop.

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
