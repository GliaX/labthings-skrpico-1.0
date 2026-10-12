# NEXT-SESSION CHECKLIST

One command to catch up: read `HANDOFF.md`, then `ARCHITECTURE.md`.

## Immediate (in order)

1. [ ] User restarts **klippy + OFM** (freeze fix lives in kalico
      webhooks.py + AF `59a1b10e…` + cooler activate)
2. [ ] 2–3 blur runs → verify signatures (HANDOFF.md "Verify after restart")
3. [ ] Park gap `tiny_fit` vs `park` ≤ ~0.1u (precision regression fixed?)
4. [ ] **COMMIT EVERYTHING** — OFM untracked files, kalico mods
      (2 constants + webhooks jogging fix), overlay `ofm/` + `funData/`
      + the two handoff docs. Single-copy; SD failure = session lost.

## If time remains (pick by interest)

- [ ] Accel ceiling test: raise max_accel toward 45000, watch ACCELTEL,
      user listens for gear strain (~0.5s/run payoff if bias, not wall)
- [ ] Telemetry recorder (#5): full-rate motion_report → jsonl during runs;
      measured z/v for HUD + true per-leg accel
- [ ] Sink writer-thread rework (before video thing may return to scope)
- [ ] From-2mm focus: climbing-gradient walk experiment (needs sample + eyes)

## Do NOT

- Video thing is REMOVED from scope; do not re-add until the sink has
  a writer thread (code lives in `ofm/`)
- Do not add sleeps/M400/deadlines to the motion chain (house law)
- Do not commit or push (user operates git)
- Do not reboot the scope (user-only)
