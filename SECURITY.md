# Security Policy

sous runs a local model in an autonomous tool loop against your source tree.
Its sandbox — path confinement, the command allowlist, environment scrubbing,
the process-group kill — is the security boundary, and bugs in it are worth
reporting.

## Supported versions

sous is pre-1.0 and has no released versions yet. Fixes land on `main`; there
is nothing to backport to. When releases start, this section will say which
ones get security fixes.

## Reporting a vulnerability

**Please don't open a public issue.** Use GitHub's private reporting:

> **Security** tab → **Report a vulnerability**

That opens a private advisory only you and the maintainer can see.

This is a single-maintainer project. Reports are handled on a best-effort
basis — expect an acknowledgement within about a week rather than same-day,
and no fixed timeline for a fix. If a report is valid, the fix and the
advisory are published together; if it is declined, you get the reasoning, not
silence.

Credit in the advisory unless you would rather stay anonymous.

## What's in scope

Anything that breaks a guarantee in the
[Security model](README.md#security-model). Concretely:

- **Path-confinement escapes** — a worker writing outside its task's
  `project_root`, into `.git/`, or into the sous control directory (`~/.sous/`).
  Symlink and path-normalisation tricks count.
- **Allowlist bypasses** — running a command that is not allowlisted and was
  never approved, including argv-splitting tricks that make a command look
  like an allowlisted one.
- **Environment leaks** — secrets surviving `scrubbed_env()` into a command's
  environment.
- **Audit evasion** — a command modifying files without the change appearing
  in the report, other than by the privileged routes noted below.
- **Descendant survival** — processes outliving the timeout kill and writing
  files after the audit, other than by the documented double-fork route.
- **Approval bypasses** — anything that gets a command run without the human
  approval it should have required.
- **The MCP endpoint** — it binds to `127.0.0.1`; reachability beyond that, or
  anything exploitable through it, is in scope.

## What isn't

These are documented design limits, not vulnerabilities. They are described in
the [Security model](README.md#security-model), and reporting them is
reporting the README:

- **Allowlisting a command that runs repo-resident code grants arbitrary code
  execution.** Any test runner — `pytest`, `npm test`, `make test` — executes
  code the worker just wrote. This is inherent to the feature, which is why
  the README tells you to calibrate the allowlist and review diffs.
- **There is no network sandbox.** An allowlisted or approved command can
  reach the network if it does so itself.
- **A descendant that double-forks and calls `setsid()` survives the group
  kill.** Closing that needs OS-level confinement macOS does not offer.
- **The stat-based audit does not defend against a privileged attacker.** Root
  access or system-clock manipulation between snapshots can hide a change; the
  audit is a safety net against the sandboxed worker, not against root.
- **The model producing wrong, low-quality, or malicious-looking code.** That
  is the expected failure mode the human review step exists for. Review the
  diff.

If you think one of these is worse in practice than the README claims — for
instance a *reliable* way to reach the double-fork escape from an ordinary
delegated task — that is worth reporting. The limitation being documented does
not make a sharp exploitation path uninteresting.
