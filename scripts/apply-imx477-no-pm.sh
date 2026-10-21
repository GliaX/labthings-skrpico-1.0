#!/bin/bash
# apply-imx477-no-pm.sh -- keep the IMX477 camera sensor's I2C interface awake.
#
# WHY: with runtime PM on (power/control=auto), the sensor's i2c device
# autosuspends whenever libcamera isn't holding it. Waking it during camera
# mode switches (crop990 <-> default <-> full_resolution) causes slow/flaky
# reconfigs and intermittent "Device or resource busy" errors at calibration
# start. Pinning power/control=on removes that penalty.
#
# Installs a udev rule (survives reboots; re-applies at device add) and sets
# the control directly on the live device (udev trigger does NOT re-evaluate
# ATTR matches on an already-bound device, so the direct write is needed for
# the current boot).
#
# Run on the microscope (scope) as a user with sudo:
#   bash apply-imx477-no-pm.sh
#
# Verify:
#   cat /sys/bus/i2c/devices/10-001a/power/control         -> on
#   cat /sys/bus/i2c/devices/10-001a/power/runtime_status  -> active

set -e

DEV=/sys/bus/i2c/devices/10-001a
RULE=/etc/udev/rules.d/98-imx477-no-pm.rules
RULE_LINE='ACTION=="add", SUBSYSTEM=="i2c", ATTR{name}=="imx477", ATTR{power/control}="on"'

if [ "$(cat "$DEV/name" 2>/dev/null)" != "imx477" ]; then
    echo "WARNING: $DEV is not imx477 on this board - adjust DEV path." >&2
fi

echo "$RULE_LINE" | sudo tee "$RULE" > /dev/null
sudo udevadm control --reload-rules

# udev trigger does not re-match ATTR on bound devices; set it directly:
echo on | sudo tee "$DEV/power/control" > /dev/null

echo "control:  $(cat "$DEV/power/control")"
echo "status:   $(cat "$DEV/power/runtime_status")"
echo "rule:     $(cat "$RULE")"
