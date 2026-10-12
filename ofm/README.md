# ofm/ — OpenFlexure microscope files kept in sync with the scope

Deployed copies of the OFM server files this project touches, plus one-off
tools. The live authority is always the scope
(`192.168.100.20:/home/admin/openflexure-microscope-server/...`); these are
mirrors so the work survives outside the device.

## Files

| file | lives on scope at | purpose |
|---|---|---|
| `af_video.py` | `things/af_video.py` | AfVideoThing: arms an NV12 sink in the camera AF pre-callback, captures the AF point-of-view frames (encoder is paused during AF, preview shows nothing), encodes to mp4 via ffmpeg |
| `af_video_core.py` | `things/af_video_core.py` | pure helpers for the above (stride compaction, ts parsing); importable without labthings |
| `test_af_video_core.py` | — | unit tests incl. real ffmpeg lossless round-trip; run: `python test_af_video_core.py` |
| `af_story_chart.py` | — | one-shot chart from the AF artifacts (`python af_story_chart.py <dir-with-logs>`): sharpness mountain + leg marker times + telemetry trapezoid |
| `hist_charts.py` | — | history charts from `smooth_af_moves.log` (`python hist_charts.py <dir>`): accel-cal health, focus drift over time, run-duration by chain era |
| `af_hud_video.py` | — | burns a telemetry HUD (leg, z, v, leg progress) onto the AF point-of-view video; clock-bridges `moves.log` epochs to frame `.tsv` monotonics via LEG MARKER et+t. `python af_hud_video.py <dir>` |
| `test_af_hud_video.py` | — | synthetic end-to-end test for the above: ground-truth z assertion + HUD-band render check + ffmpeg decode; run: `python test_af_hud_video.py` |
| `smooth_autofocus.py` | `things/smooth_autofocus.py` | event-driven AF (marker waits, telemetry accel cal), refactored 2026-10-12 |
| `smooth_af_core.py` | `things/smooth_af_core.py` | pure curve math (gaussian fit, median filter, clustering) used by the AF; unchanged |
| `picamera.py` | `things/camera/picamera.py` | AF metric hook + `_AF_SINK` capture hook, refactored 2026-10-12 |
| `skrPico.py` | (scope stage class; also patched at `src/labthings_sangaboard/skrPico.py`) | skrPico stage: ws motion_report reader, marker/et samples, cal load |
| `test_smooth_af_core.py` | `tests/unit_tests/` | AF curve-fit/cluster unit tests; run: `python test_smooth_af_core.py` (needs numpy) |

## Using the AF video recorder

On the scope (after restart): `POST /af_video/arm/` (optional `max_frames`,
default 1200 = 10 s at 120 fps), run autofocus, then `POST /af_video/disarm/`
and `POST /af_video/encode/` (optional `fps`=120, `crf`=18). Output lands at
`/home/admin/af_view_<epoch>.mp4` with a `.tsv` of sensor timestamps per
frame. Disarm is safe to forget — capture auto-stops at the frame cap.
