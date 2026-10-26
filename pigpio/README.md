# pigpio/ — sync-edge timing for the strobe calibration (x-003)

pigpiod gives the Pi GPIO edge counting; the OFM strobe thing
(`ofm/strobe_control.py`) counts CAM1_SYNC edges on GPIO18 over a 2 s
window to measure the camera's true frame period in every streaming
mode (`_period_us`). The MCU pulse counter is only a fallback: it
multi-counts when the flash is shorter than the sync HIGH time, so the
edge count is the measurement the calibration trusts.

## Contents

- `pigpio.c.patched` — pigpio source with the one custom patch:
  line ~5588 `if (signum == SIGCONT) return;` at the top of `sigHandler()`.
  Without it systemd's SIGCONT kills pigpiod ("Unhandled signal 18").
- `pigpiod.service` — systemd unit. pigpiod double-forks out of its cgroup,
  so it MUST be Type=oneshot + RemainAfterExit=yes (not Type=forking —
  systemd SIGTERMs the straggler and the daemon dies on "Unhandled signal 15").
- `pigpio.py` — the Python binding vendored into the OFM venv
  (`site-packages/pigpio.py`); upstream module, md5 9ae9c1e3.

## Rebuild on the Pi

    git clone https://github.com/joan2937/pigpio/ /tmp/pigpio
    cp pigpio/pigpio.c.patched /tmp/pigpio/pigpio.c
    make -C /tmp/pigpio
    sudo make -C /tmp/pigpio install
    sudo cp pigpio/pigpiod.service /etc/systemd/system/
    sudo systemctl enable pigpiod
    cp pigpio/pigpio.py <ofm-venv>/site-packages/

Installed-binary md5s on x-003 (for verification):
    pigpiod            fe9f8c7c
    libpigpio.so.1     ed458a8e

pigpiod listens on localhost:8888 and must be running before the server
starts; calibration raises a clear error if the daemon is unreachable.
