# Security Policy

sous executes nothing. It is an HTTP daemon on `127.0.0.1` that answers a
Claude Code subagent's turns from a local model and forwards every other
request to Anthropic's API — so its security surface is that daemon and the
credential it carries: your Claude Code subscription token, on every
forwarded request. Leaking, storing, misrouting or logging it is worth
reporting, as is anything that reaches the daemon from off the loopback
interface. What the local model *does* is governed by Claude Code's own
permission system, not by sous.

## Supported versions

sous is pre-1.0. Only the latest release is supported: fixes land on `main`
and ship in the next release, and nothing is backported to earlier versions.
If you are running an older one, upgrade before reporting — the bug may
already be gone.

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

- **The loopback guard** (`src/sous/loopback.py`) — the daemon binds to
  `127.0.0.1` and refuses foreign `Host`/`Origin` values and any browser
  request whose `Sec-Fetch-Site` is not `none` or `same-origin` (a page's
  `<iframe>` or no-cors GET carries no `Origin` to refuse). The check runs on
  every route, forwarded ones included. Reachability from anything but a
  loopback client, or a request that slips past the check, is in scope.
- **Credential handling** — sous forwards the `Authorization` header Claude
  Code sends to `[server].upstream_url` and nowhere else, stores it nowhere,
  and adds no credential of its own (no `~/.netrc`, no proxy environment). A
  credential sent anywhere but the configured upstream, or persisted to disk,
  is in scope.
- **Anything logged that shouldn't be** — a request body, a header value or a
  query string reaching a log at any level, debug included, is a
  vulnerability, not a papercut. The same goes for prompt text, tool names or
  file paths reaching the status document, the event stream, `sous top` or
  `sous statusline`.
- **A `/sous/` path reaching the upstream.** The daemon's own routes
  (`GET /sous/status`, `GET /sous/events`, `POST /sous/hold`,
  `POST /sous/unload`) are mounted before the forwarder precisely so that no
  path under `/sous` can be proxied out. One that is, is in scope — as is
  anything of a hold body beyond a pid and a start time being acted on (the
  pid is logged to attribute the hold and its release; the start time and the
  raw body never are).
- **A request body that crashes the daemon**, or that reaches the upstream
  re-serialized rather than byte for byte. A forwarded request must be altered
  only in `Host`, the hop-by-hop headers and, for the two Messages routes
  whose body sous reads, a recomputed `Content-Length`.
- **`sous claude` setting a credential, a tier or a permission variable.** It
  must never set `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, an
  `ANTHROPIC_DEFAULT_*_MODEL` or a Claude Code permission mode: the first two
  move your billing, the third pulls the main loop off the upstream, and the
  last would weaken a boundary that is the user's to set.
- **The daemon writing outside `~/.sous`.** Its log, its lock and its config
  live there and nowhere else; it edits no source tree.

## What isn't

These are documented design limits, not vulnerabilities. They are described in
the [Security model](README.md#security-model), and reporting them is
reporting the README:

- **What the local model does through Claude Code's tools.** sous returns
  `tool_use` blocks and never runs one; Claude Code executes every tool under
  its own permission mode — auto mode's frontier classifier, the allow/deny
  rules, and the optional sandbox. A gap there is a Claude Code issue, and
  belongs to Anthropic's own reporting channels, not here.
- **A plaintext upstream on loopback.** `[server].upstream_url` accepts
  `http://` for `127.0.0.1`, `localhost` and `::1` so tests and local
  front-ends can sit in between; that traffic never leaves the machine.
- **The model producing wrong, low-quality, or malicious-looking code.** That
  is the expected failure mode a frontier main loop reviewing the subagent's
  work exists for. Review the diff.

If you think one of these is worse in practice than the README claims — a
concrete path from a local turn to something sous itself should have
prevented, say — that is worth reporting. A limitation being documented does
not make a sharp exploitation path uninteresting.
