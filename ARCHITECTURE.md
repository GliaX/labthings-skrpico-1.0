# Smooth Autofocus — Architecture & Implementation

Scope x-003 · OFM server (openflexure-microscope-server) + Kalico/Klippy stage
over Moonraker · fully event-driven chain; validated end-to-end from blur:
far legs at 9000 u/s, park on peak, zero sleeps / M400 / clock-based waits
anywhere in the motion path.

## What it does

`smooth_focus` acquires focus from an arbitrary starting defocus:

1. **near** — ±110u sweep at the calibrated knee velocity; if the sample's
   focus is inside, fit + park directly (rarely taken at high mag).
2. **far** — on near-miss, ±1024u legs at **9000 u/s** when
   `smooth_af_cal.json["accel"]` exists, **6000 u/s** otherwise (the
   calibration run). A strong leg-1 hit (≥1.35× leg median) skips leg 2
   (`|far1_skip`); weak hits run both + widen tiny (`|tiny_wide`).
3. **claw** — pooled-region hits, or sharpness hits over a far-field bar
   (0.45 × calibrated contrast × median). Blur-tolerant by design.
4. **tiny** — ±(fwhm × 1.25, min 60u; ×2 on weak claw hits) sweep,
   gaussian fit through the gate (absolute bar, curve-relative bar, edge
   margin, r²/SNR). A fit both verifies the hit and **re-measures FWHM on
   the go** (persisted, tagged `|fwhm…` — stale calibrations self-heal
   under current optics).
5. **park** — two-stage: 9000 u/s to 30u short, creep 300 u/s for the
   final approach; OFM cache synced.

Any refusal returns the stage to its start; retries happen only on
`tiny_fail` (`NORETRY` on structural failures); the pre-AF camera regime
is restored 10s after the last run by the cooler, with readback verify.

## The motion chain (event-driven, the heart of it)

The only wait is the telemetry reader thread's push-arrival Condition.
`et` (klippy motion-report encoder time) does all sequencing; host
arrival order is treated as unreliable (pushes arrive 0.3–0.5s late and
bundled under load).

1. **Telemetry** (`skrPico.start_motion_telemetry`): one websocket to
   Moonraker, `motion_report` pushed on change (10ms STATUS_REFRESH_TIME,
   8ms webhooks SUBSCRIPTION_REFRESH_TIME — both tuned 2026-10-11). Pushes
   carry only changed fields — absent `live_velocity` mid-cruise is
   *carried* from the previous push and flagged, never read as 0.
2. **Completion marker**: a leg is over when a push arrives with
   `v≈0 ∧ z≈target ∧ et > previous marker et`. Waited on via the push
   Condition. A stale bundle from the previous leg cannot satisfy it (its
   z is the wrong target).
3. **Segmentation**: leg N's samples = pushes with `et ∈ (marker[N-1],
   marker[N]]`. `et` slicing is exact regardless of delivery timing.
4. **Labeling**: frames interpolate the *sampled* (et−hoff, z) trajectory
   — no rebuilt trapezoid, no accel model in the label path.
5. **Starvation guard**: 3 consecutive event-waits with no new pushes
   (`TELESTALL`) abort the run (`|far_abort`).
6. **Calibration from telemetry** (`_accel_from_telemetry`): per far leg,
   plateau = max|v|, rise distance to ≥90% plateau, `a = v_p²/(2·d_rise)`;
   both legs averaged, range-gated, persisted to `smooth_af_cal.json`
   (`ACCELTEL … SAVED`). Estimates that disagree wildly are refused and
   the persisted cal kept. The persisted cal is what flips far speed to
   9000.

## Camera metric

- `crop990` mode (1332×990 @120fps), lores YUV420 350×350 via
  `make_array("lores")` — straight from the ISP, encoder not in the path.
- **Sharpness**: full-plane gradient energy on the luma.
- **pooled16**: defocus-tolerant structure channel for far direction.
  Blind beyond ~±600u true defocus on pale H&E at this mag.
- **Regime**: gain 2.0, exposure = pre-AF exposure / 2, clamped to
  100–8000µs (halving keeps motion smear at ~4.5u during the 9000 u/s
  far legs; the clamp exists because letting AE run once collapsed to
  200µs → black field).
- **Thermal housekeeping** (`_quiet_camera`): sticky 120fps during AF
  sessions; a daemon timer 10s after the last run restores 20fps, ensures
  both MJPEG encoders run, and re-applies the pre-AF gain/exposure with
  readback verify (`COOL regime … verify=ok` in `smooth_af_moves.log`).
  A new run cancels the pending timer, so back-to-back runs keep
  sticky 120fps.
- `_AF_SINK` hook: an optional external sink receives every pre-callback
  frame + sensor timestamp — the way to see the AF's point of view, since
  the encoder is paused during AF.

## Camera hardware: IMX477 clock overclock

The sensor driver runs Open360Camera-style overclocked clocks, doubling the
full-res readout: **2800×2800 @ 12-bit window mode measures 21.99 fps vs
10 fps stock** (kernel 6.18.29+rpt-rpi-v8). This lifts the `default`
full-resolution mode used for stills and calibration; the crop990 120 fps
AF mode is unchanged.

Driver and device tree are a matched pair, both kept in `kernel/`:

| parameter | stock | this build |
|---|---|---|
| MIPI link freq (`IMX477_DEFAULT_LINK_FREQ`) | 450 MHz | 640 MHz |
| `IMX477_IOP_PREDIV` (sensor reg 0x030d) | 0x02 | 0x03 |
| pixel rate (`IMX477_PIXEL_RATE`) | 840 MHz | 984 MHz |
| mode table | stock modes | + `mode_2800x2800_regs` (window 628,120, frm_length 2848) |

- `kernel/imx477-ofm2800.c` / `imx477-ofm2800.ko` — driver source and built
  module (installed as
  `/lib/modules/…/kernel/drivers/media/i2c/imx477.ko`, md5-verified against
  the scope).
- `kernel/imx477-640M.dtbo` — the installed DT overlay
  (`/boot/firmware/overlays/imx477.dtbo`; stock kept on the scope as
  `imx477.dtbo.stock`). `kernel/dtso/` holds the stock rpi-6.18.y overlay
  sources; the 640M dtbo is those sources with 450M→640M in
  `imx477_378.dtsi`.
- Rebuild + install commands are in `kernel/README.md`. Constraints: the
  dtbo is rebuilt from source only (binary decompile/recompile does not
  round-trip phandle fixups), and driver/DT must move together — the
  PREDIV=3 driver accepts only a 640M DT; a stock driver with a 640M DT
  fails probe (−22) and the camera is absent at boot.

## Freeze-motion strobe (`freeze_motion` thing)

Global-shutter imitation for histo scans: the integration window spans the
whole frame period (E pinned at 45000 µs) and the LED delivers a short
pulse (τ = 300 µs), so every row integrates the same instant. Frame timing
is the native mode's 45.474 ms period (21.99 fps).

- LED: Bridgelux Thrive BXEN-50S on the **Pi's** BCM 24 (header pins 18+20,
  3.3 V through the existing resistor). The SKR Pico gpio24 / `[pwm_tool]`
  path is retired: klippy PWM reprogramming emitted a full-ON glitch frame
  and tore every frame in flight. Viewing light = pin HIGH; the pin is dark
  between pulses in strobe mode.
- Carrier: **pigpio wave engine** (built from source; not in trixie apt).
  `pigpiod` runs Type=oneshot + RemainAfterExit (it double-forks), with a
  SIGCONT-early-return patch in its signal handler. Free-running pulse
  train at the camera period; retunes swap waves with REPEAT_SYNC at cycle
  boundaries — glitch-free.
- Frame-accuracy chain: the `imx477` driver carries the 2800×2800 overclock
  mode AND a clamp — in that mode frame length ≤ 2848 and exposure ≤
  frame − 22 (the exposure clamp prevents an IPA-requested overflow that
  wedges the sensor). `picamera.py` pins `FrameDurationLimits` [45000,
  45000] at every stream creation: a loose FDL lets the rpi IPA stretch
  vblank to ~500 lines and the frame to 53 ms; metadata FrameDuration lies
  about it — only SensorTimestamp deltas are truth.
- Band geometry: the straddle band (rows being read out during the pulse)
  wraps at the END of the frame into a wide invisible zone (~200 sensor
  rows of end-of-frame blanking). Drift law, stopwatch-calibrated: 0.169
  preview-rows/s per µs of period error; the null sits at ≈45474.5 µs,
  between the wave engine's 1 µs ticks, so 45474 (−0.085 rows/s) is the
  best integer pin.
- Hold loop (`_pll_loop` in `things/freeze_motion.py`): walk band down at
  45535 µs (~40 rows/s) → the instant the hook loses it (row = None, band
  inside the end blanking) pin 45474 → every 40 s hidden, a 1.5 s burst at
  45415 (≈ −14 rows backward) pushes it deeper into the blanking, then
  re-pin. The band never re-enters the visible frame. A manual
  `pin_period(us)` overrides the loop absolutely; `set_phase(µs)` exists
  but the wave-swap phase actuator measured ~300× weaker than designed
  (unresolved).
- API: `strobe_on(tau=0.0003)` arms everything (AE off, E 45000, gain 1.0,
  wave start, band hook, hold loop). `strobe_off` restores CW viewing
  light. Properties: `last_band_row`, `period_us`, `locked`, `firing`.
- Static artifact bands (dark top rows, bright ~1818–2020) are lens-shading
  tables from the stock mode; LST recalibration pending.

## Calibration (`smooth_af_cal.json`)

| key | meaning | derived |
|---|---|---|
| `accel` | measured stage accel (u/s²) | telemetry rise-fit; gates far speed 6000→9000 |
| `fwhm_u` | optics PSF width | re-measured by every accepted tiny fit; near_v = 12·FWHM; tiny span = 1.25·FWHM |
| `contrast` | peak/floor | claw bar 0.45× (far), gate bar 0.85× (near) |

Loaded lazily at first far sweep; written by the telemetry solver and the
tiny fit.

## Position integrity

- `__enter__` retries homing + `update_position()` up to 60s (boot race).
- The OFM cache is re-synced via `update_position()` after legs that did
  not complete on a certified marker (fallback paths only).
- UI-equivalent moves only for point tests: `move_relative`/`move_absolute`
  through the OFM API — raw kalico POSTs bypass position tracking.
- Known open behavior: anchor drifts ~50u/day, direction consistent,
  cause unknown (no thermal correlate — the microscope has no fans or
  thermal probes).

## Hard rules encoded

- Happens-before only: waits are data-arrival events (push Condition);
  frame-count waits and clock waits are banned in the motion chain.
- User-stated focus is ground truth (±20u); failures are pipeline failures.
- Full-res data always; nothing subsampled.
- Git is the operator's; deploys are file copies, commits are manual.

## Files (repo shape)

- `src/openflexure_microscope_server/things/` (deployed to the scope):
  - `smooth_autofocus.py` — the thing: flow, marker waits, telemetry
    accel cal, claw/tiny gates, scanner-compat shim.
  - `smooth_af_core.py` — log-gaussian fit + median filter (pure math).
  - `camera/picamera.py` — camera thing: 120fps crop990, AF metric
    callback, `_AF_SINK` hook, sensor-timestamp interpolation.
  - `stage/skrPico.py` — stage thing (scope-live; also mirrored at
    `src/labthings_sangaboard/skrPico.py`): Moonraker client, telemetry
    websocket reader (push Condition), cal persistence, `_stamp_for`.
- `kernel/` — IMX477 overclocked driver artifacts (source, built module, dtbo,
  rebuild recipes; see Camera hardware above).
- `ofm/` (this repo) — deployed copies of the above + unit tests.
- `funData/` — motion-data playground: `scripts/` (story chart, history
  charts, HUD slowmo burner + its test), `data/` (pulled artifacts),
  `charts/` (outputs).
- Kalico side: user fork `apopanda/kalico`, base a7e74c5c; live mods vs
  upstream are the two refresh constants (motion_report 10ms, webhooks
  8ms) + `webhooks.py get_status` reports `ready` while in jog mode
  (jog mode sets an internal state string stock moonraker cannot parse;
  translating at this single source keeps every websocket client fed).

## Known interop constraint

Klippy printer-state strings cross a language boundary into moonraker's
`KlippyState` enum. Any new state a fork introduces must be either added
to that enum or translated in `webhooks.get_status` (the sole source of
the pushed `state` field) — an unparseable state aborts moonraker's
status fan-out, starving every websocket client of telemetry mid-motion.

## Open items

- Full-rate telemetry recorder (jsonl dump of every motion_report push
  during a capture) — would give measured per-frame z/v for the HUD and
  true per-leg accel stats; parked by operator request.
- External sink durability: a sink that writes from the pre-callback must
  use a queued writer thread; direct SD writes from the callback stall
  frame delivery (observed: every ~8th frame doubled its interval).
