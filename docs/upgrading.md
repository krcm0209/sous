# Upgrading from 0.6

0.7.0 removed the MCP server, the task queue, the worker and its sandbox.

- Remove the MCP registration: `claude mcp remove sous`, and any `sous mcp`
  entry in `claude_desktop_config.json`. Claude Desktop is no longer supported.
- Config: `[gateway]` keys moved — `local_models`, `upstream_url` and
  `generation_timeout_minutes` to `[server]`, `max_context_tokens` to
  `[model]`; `enabled` is gone (the endpoint is always on); `[budgets]`,
  `[commands]`, `[context]` and `[tasks]` are gone. The daemon warns once at
  startup about any of them and starts anyway. A `[model].max_context_tokens`
  written for the worker (32768) is clamped to the 49152 floor; set it to the
  window you want.
- `~/.sous/tasks/` and `~/.sous/tasks.db` are no longer read and can be
  deleted.
- `sous wait` and `sous mcp` are gone; `sous status` now prints the engine and
  the turns.
- Earlier releases split the daemon's output into `daemon.log` and
  `daemon.err.log`; re-run `sous install-launchd` once to fold them into one
  ([Observability](observability.md#the-daemon-log) has the details).
