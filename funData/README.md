# funData — motion-data playground for the microscope

Charts and video experiments built from the AF artifacts the scope
already writes (`smooth_af_moves.log`, `smooth_af_seekpts_*.json`,
`smooth_af_legdump.json`, `af_view_*.nv12/.tsv`). Nothing here runs on
the scope; point the scripts at a folder containing the pulled
artifacts.

## scripts/

| script | what it does |
|---|---|
| `af_story_chart.py <dir>` | one run, three panels: sharpness-vs-z mountain with leg markers, leg wall-times (marker spacing), motion-report trapezoid of the last labeled leg |
| `hist_charts.py <dir>` | history over all runs in `smooth_af_moves.log`: accel-cal health, focus anchor drift over time, run duration + **throughput** (sweep distance / run time) by chain era |
| `af_hud_video.py <dir>` | burns a telemetry HUD (leg, z, v, t+, progress) onto the AF point-of-view NV12 capture; clock-bridges `moves.log` epochs to frame `.tsv` monotonics via per-leg LEG MARKER knots (survives ws re-handshakes) |
| `test_af_hud_video.py` | synthetic end-to-end test for the HUD: ground-truth z assertion, chroma round-trip, ffmpeg decode. Run from anywhere: `python test_af_hud_video.py` |

Era split for `hist_charts.py` lives in the script (`T_6000`, `T_EVENT`);
bump them when the next chain change lands.

## charts/

Latest generated outputs (regenerate with the scripts, they're cheap):

- `hist_accel_health.png` — solver estimates vs persisted cal
- `hist_focus_drift.png` — focus anchor z over time (thermal drift watch)
- `hist_run_eras.png` — duration scatter + throughput histogram by era
- `af_motion_story.png` — the single-run mountain chart
- `af_hud.mp4` — HUD-burned AF point-of-view video

## ofm/ (sibling) — server-side capture

`af_video.py` / `af_video_core.py` (deployed to the scope as
`things/`) arm the camera sink that produces `af_view_*.nv12/.tsv`.
Known issue before next use: the sink writes synchronously inside the
camera callback at 120 fps — rework to a queued writer thread before
long captures.
