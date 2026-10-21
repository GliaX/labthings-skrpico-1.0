# IMX477 camera power management (no-sleep) fix

## Symptom

Camera mode switches (crop990 <-> default <-> full_resolution) are slow and
flaky: intermittent `Device or resource busy` when setting controls (seen at
the start of stage-camera auto-calibration), first frames after a switch
arriving late or degraded.

## Root cause

The IMX477 sensor's i2c device has runtime power management enabled by
default (`power/control = auto`). Whenever libcamera is not actively holding
the device, it autosuspends. The next mode switch has to wake it over i2c,
which adds latency and occasionally races the control write.

## Fix

Pin the sensor's i2c device to `power/control = on` via a udev rule
(survives reboots) plus a direct sysfs write for the current boot (udev
`trigger` does not re-evaluate ATTR matches on an already-bound device).

## Apply

On the scope:

    bash scripts/apply-imx477-no-pm.sh

Expected output: `control: on`, `status: active`, and the rule echoed back.

## Verify anytime

    cat /sys/bus/i2c/devices/10-001a/power/control         # on
    cat /sys/bus/i2c/devices/10-001a/power/runtime_status  # active
    cat /etc/udev/rules.d/98-imx477-no-pm.rules            # rule text

Device path note: `10-001a` is this board's i2c address for the imx477
(check `/sys/bus/i2c/devices/*/name` if a future board differs).

## When it was needed

2026-10-21: stage-camera auto-calibration intermittently failed with
`Device or resource busy` during camera reconfiguration after AF runs left
the camera switching between crop990 and default modes. Related history:
sticky-crop990 AF streaming mode and the COOL mode restore.
