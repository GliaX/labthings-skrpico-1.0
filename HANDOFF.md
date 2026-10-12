# HANDOFF — Microscope AF session 2026-10-12

Read this plus `ARCHITECTURE.md` (repo root) before touching anything.

## State at handoff

- **Scope x-003 (192.168.100.20)** runs the fully event-driven AF chain,
  smooth_autofocus.py md5 `59a1b10e…` deployed, compile-verified in the
  scope venv. **THREE restarts pending (user-only): OFM + moonraker +
  klippy** — see FREEZE FIX below.
- Local sources of truth:
  - OFM repo files (untracked/modified): `OFM-server/openflexure-microscope-server/src/openflexure_microscope_server/things/` — `smooth_autofocus.py`, `smooth_af_core.py`, `camera/picamera.py`
  - This repo: `src/labthings_sangaboard/skrPico.py` (scope live class, refactor included), `ofm/` (mirrors + tests + tools), `funData/` (charts/scripts/data)
  - Kalico fork `C:\glia\EU_dotacje\mikroskop\kallicofork\kalico`:
    working-tree mods = the two refresh constants (`motion_report
    STATUS_REFRESH_TIME 0.010`, `webhooks SUBSCRIPTION_REFRESH_TIME
    0.008`) + `webhooks.py get_status` reports `ready` instead of
    `jogging` (freeze fix, `e82685ac…`); everything else committed at
    `38556076`. Moonraker deliberately UNPATCHED (user call: fixes stay
    in kalico only; the webhooks-side fix suffices because
    `WebHooks.get_status` is the sole source of the pushed `state`).
- **Nothing is committed.** The entire working set (OFM untracked files, kalico mods, this repo's `ofm/`, `funData/`, ARCHITECTURE.md) exists in exactly one copy. Commit before experimenting further.

## What changed this session (final shape)

1. **Event-driven motion chain** — leg completion = telemetry marker push
   (`v≈0 ∧ z≈target ∧ et>prev marker`); segmentation by marker et; frames
   labeled from the sampled trajectory (no trapezoid model). No sleeps,
   no M400, no deadlines — house rule.
2. **Accel calibration from telemetry** — `ACCELTEL` measures
   `a = v_plateau²/(2·d_rise)` per leg, gates, persists to
   `smooth_af_cal.json["accel"]` (31609). Far legs run 9000 when cal
   exists, 6000 when not (cal run). Velocity-tagged, out-of-range
   estimates refused.
3. **Speed round**: `sys.setswitchinterval(0.001)`; far1 early-exit on
   strong hit (`|far1_skip`, ≥1.35× leg median); sticky 120fps across
   runs; async log/dump writer thread; retry only on `tiny_fail`
   (`NORETRY` on structural failures); two-stage park (9000 to 30u
   short, creep 300 u/s final approach — fixes the park-precision
   regression).
4. **Thermal cooler** — `_quiet_camera` (daemon Timer, 10s after last
   AF/probe): sensor → 20fps, encoders ensured on, pre-AF gain/exposure
   re-applied **with readback verify** (`COOL regime … verify=ok`). New
   AF cancels it, so back-to-back runs keep sticky 120fps.
5. **Tiny window guard** — weak claw hit (<1.35× median) widens the tiny
   window to fwhm×2 (`|tiny_wide`).
6. **FREEZE FIX (2026-10-12 evening)** — the "stuck like recording"
   freezes were NOT the video thing (it wasn't armed). Root cause: kalico
   jog mode sets klippy state `jogging`; stock moonraker's
   `KlippyState.from_string` raises ValueError and aborts
   `_process_status_update` BEFORE ws fan-out → all motion_report
   telemetry dropped while jog mode active → AF TELESTALL → (old code)
   kept commanding blind legs = zombie state. Fixes deployed: kalico
   webhooks reports `ready` (sole source of the pushed state — verified
   via QueryStatusHelper fan-out), AF `59a1b10e…` aborts the run on
   TELESTALL (`|far_abort`). Trigger was fluidd jog controls; harmless
   after fix. Restarts owed: klippy + OFM only.
7. **Video thing REMOVED from scope** — `af_video.py`/`af_video_core.py`
   deleted on scope, `things.af_video` removed from config (verified by
   grep). `_AF_SINK` hook remains in camera (inert, `None` unless
   armed). Code survives only in this repo's `ofm/`; writer-thread
   rework required before it may return (see Parked).

## Verify after restart (5 min)

Run 2–3 blurs, then check `smooth_af_moves.log`:
- first run only: `burst_ready t≈0.42`; subsequent runs `t≈0.0` (sticky fps)
- hit runs: `|far1_skip` in mode, `RUN total` ≈ 2.2–2.6s
- weak hits: `|tiny_wide`
- structural failure: `NORETRY` (not RETRY1)
- 10s after last run: `COOL fps=20 enc=on` + `COOL regime … verify=ok`
- park gap: `tiny_fit` z vs `park` z within ~0.1u

## Parked (in ARCHITECTURE.md open items)

- **Full-rate telemetry recorder** (#5): subscribe motion_report at full
  rate during runs → jsonl. Upgrades HUD to measured z/v, gives true
  per-leg accel.
- **Sink writer-thread rework** — mandatory before the next video capture.
- **Accel ceiling test** — cfg says max_accel 45000, telemetry measures
  31609; bias or mechanical wall? Needs user listening to the stage.
- **From-2mm focus** — pooled16 blind beyond ±600u; climbing-gradient walk
  candidate. Needs sample + user eyes.

## Quirks that will bite the next agent

- `smooth_af_moves.log` mixes `time.time()` (SEND/LEG/RUN) and
  `time.monotonic()` (PROBE/BURST/WS) — filter to epoch-domain when
  charting (hist_charts.py does).
- Camera `SensorTimestamp` ≠ ws `et−hoff` by a large session-dependent
  offset; marker-based clock bridging across ws reconnects is unreliable
  (the frozen-HUD bug). Content anchors (motion energy) are robust.
- libcamera lores rows are stride-padded; `af_video_core.compact_nv12`
  must run before any rawvideo consumer.
- labthings pydantic actions need `{"params": …}` envelope; thing routes
  are `/api/v3/<thing>/<action>` (check `/openapi.json`, don't guess).
- Deploy path: local edit → SFTP put → `md5sum` both sides →
  `.venv/bin/python -m py_compile` on scope. User restarts, never the
  agent. User commits, never the agent.
