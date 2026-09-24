# Design records

Specs and plans whose work has shipped leave `docs/superpowers/`; git history
keeps them. Each link below opens the record as it was when it left the tree.
Records that still govern the code, or still have planned work, stay in
`docs/superpowers/`.

## Specs

| Date | Spec | Shipped in |
|---|---|---|
| 2026-08-14 | [The original design: an MCP server delegating to a sandboxed local worker](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/specs/2026-08-14-sous-design.md) | the first implementation (pre-PR history); removed by #115 |
| 2026-08-19 | [Cross-turn prompt cache reuse](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/specs/2026-08-19-prompt-cache-reuse-design.md) | #33 |
| 2026-08-20 | [One generation thread per task, so the cache survives between turns](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/specs/2026-08-20-per-task-generation-thread-design.md) | #35 |
| 2026-09-05 | [Cross-session reuse through a tools-boundary fork](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/specs/2026-09-05-gateway-phase3c-tools-fork-design.md) | #72 |
| 2026-09-10 | [In-place system messages, retained turn slots, preload and hold](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/specs/2026-09-10-gateway-cache-continuity-design.md) | #93, #94 |
| 2026-09-11 | [INT8-activation prefill on the M5 tensor units](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/specs/2026-09-11-int8-activation-prefill-design.md) | #78 |
| 2026-09-21 | [The tools fork persisted to disk](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/specs/2026-09-21-fork-persistence-design.md) | #120, #122 |

## Plans

| Date | Plan | Executed in |
|---|---|---|
| 2026-08-14 | [The first implementation](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-08-14-sous.md) | pre-PR history; removed by #115 |
| 2026-08-19 | [Cross-turn prompt cache reuse](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-08-19-prompt-cache-reuse.md) | #33 |
| 2026-08-20 | [One generation thread per task](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-08-20-per-task-generation-thread.md) | #35 |
| 2026-08-26 | [Gateway phase 0: the spike gates](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-08-26-gateway-phase0-spikes.md) | #49 (the gates ran on 2026-08-27, under #41) |
| 2026-09-02 | [Gateway phase 1: the Anthropic-compatible endpoint](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-02-gateway-phase1-endpoint.md) | #63 |
| 2026-09-04 | [Gateway phase 2: routing and `sous claude`](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-04-gateway-phase2-routing.md) | #64 |
| 2026-09-04 | [Gateway phase 3a: keyed cache slots and header forks](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-04-gateway-phase3a-keyed-cache.md) | #71 |
| 2026-09-05 | [Gateway phase 3c: the tools-boundary fork](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-05-gateway-phase3c-tools-fork.md) | #72 |
| 2026-09-10 | [Observability PR 1: attributed turns, one daemon log](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-10-gateway-observability-pr1.md) | #85 |
| 2026-09-11 | [INT8-activation prefill](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-11-int8-activation-prefill.md) | #78 |
| 2026-09-13 | [Continuity A: in-place system messages, retained turn slots](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-13-gateway-continuity-a.md) | #93 |
| 2026-09-13 | [Continuity B: preload and hold](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-13-gateway-continuity-b.md) | #94 |
| 2026-09-14 | [Observability PR 2: live status, `/sous/events`, `sous top`](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-14-gateway-observability-pr2.md) | #95 |
| 2026-09-14 | [VLM continuation positions](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-14-vlm-continuation-positions.md) | #99 |
| 2026-09-15 | [The idle cost of `/sous/events`](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-15-events-idle-cost.md) | #102 |
| 2026-09-17 | [`sous tune --quick`](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-17-sous-tune-quick.md) | #106 |
| 2026-09-18 | [`sous tune`, full run](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-18-sous-tune-full.md) | #107 |
| 2026-09-19 | [Removing the MCP delegate path](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-19-remove-mcp-path.md) | #115 |
| 2026-09-22 | [The tools fork on disk](https://github.com/krcm0209/sous/blob/b00099bc86f984be2f31c824a64ef4259964f6ee/docs/superpowers/plans/2026-09-22-fork-persistence.md) | #120 |

## Still in `docs/superpowers/specs/`

- `2026-08-26-hybrid-gateway-design.md` and `2026-09-19-remove-mcp-path-design.md`:
  the security boundary `CLAUDE.md` and the README point to. The first also
  holds phase 3b (batched serving), which is parked.
- `2026-09-10-gateway-observability-design.md`: its PR 3 (persisted turns and
  `sous turns`) has not shipped.
- `2026-09-17-sous-tune-design.md`: `--discover` has not shipped.
