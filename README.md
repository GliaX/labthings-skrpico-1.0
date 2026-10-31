# labthings-skrpico-1.0

A LabThings-FastAPI Thing for working with the SKR Pico v1.0 motion controller

Contact: adam@glia.org

Based on code from https://openflexure.org/, sangaboard.py and labthings-sangaboard
https://gitlab.com/openflexure/openflexure-microscope-server/-/blob/v3/src/openflexure_microscope_server/things/stage/sangaboard.py
https://github.com/labthings/labthings-sangaboard
authored by Richard Bowman, Julian Stirling, Joe Knapper from the OFM team.

## ofm/ mirror (2026-10-31)

Byte-identical mirrors of the deployed OFM server files (scope x-003,
`/home/admin/openflexure-microscope-server/src/openflexure_microscope_server/things/`), kept here as the canonical
safekeeping copy per the layout law. Test files are mirrored at `ofm/test_*.py` (source: `tests/unit_tests/`).

CSM conventions (as deployed):

- `camera_stage_mapping.py`: the stored
  `image_to_stage_displacement_matrix` is **[stage][image]** (stage rows, image columns). `csm_img_to_stage` /
  `csm_stage_to_img` keep the legacy pixel-argument forms; the scan workflow and the UI image-jog (including its
  preview-rotation compensation) depend on those forms.
- `stage_measure.py`: the recenter/ROM probe path converts explicitly with
  `stage = M @ (px_x, px_y)` (`_img_to_stage_rel`) and moves the stage directly, so each labelled axis drives its own
  flexure.
- `leg_length_mm` setting on StageMeasurementThing: flexure leg length in mm. None/60 = stock probe geometry; other
  values scale probe spans by sqrt (leg/60) and gate turning-point acceptance on probe z-span.
