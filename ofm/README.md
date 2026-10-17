# OFM mirror + agent changes (safekeeping repo)

Layout (user-set 2026-10-17):

- This repo (skrpico overlay, user-controlled) keeps every change that would
  otherwise land in the openflexure-microscope-server repo. That repo is an
  upstream PR target judged by others, so its diff stays minimal; the full
  copies live here for safekeeping.
- Klippy-side changes live in the kalico fork only
  (kallicofork/kalico, klippy/extras/) — nothing duplicated here.
- `ofm/` mirrors the deployed OFM files byte-identical to the scope.
- `scripts/` = deploy/verify tooling (run with local Python 3.14, paramiko).

## 2026-10-17: scan focus-plane mesh (sample surface follow)

- `ofm/scan_focus_plane.py` — NEW thing: measure_pass (3x3 relative grid,
  +-30 sweeps with approved claw to -90, dual-metric sharp/brenner
  agreement gate on main-plane YUV420, pooled presence -> anchored-0,
  skip empty; installs SAMPLE_MESH)
- `ofm/picamera.py` — metric hook plane param (default lores = button path
  byte-exact) + brenner channel; crop990 main stream now YUV420
- `ofm/smooth_autofocus.py` — additive: brenner storage, monitor plane param
- `ofm_config_full.json` (repo root) — adds the scan_focus_plane thing
- `printer.cfg` (repo root) — scope config incl [bed_mesh] + [sample_mesh]
- kalico side: kllicofork/kalico klippy/extras/sample_mesh.py
  (SAMPLE_MESH / SAMPLE_MESH_CLEAR / SAMPLE_MESH_STATUS + jog z-follow),
  md5 8e6b0b0a641e481246a774421011fc3e

Verified end-to-end on scope x-003 2026-10-17: first 3x3 install live
(3 peaks + 6 anchored), klippy profile=sample active, jog follow armed.
