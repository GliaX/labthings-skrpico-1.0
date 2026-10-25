# pigpio/ — patched pigpio for the strobe carrier (x-003)

The strobe PWM carrier runs on the Pi's BCM 24 via the pigpio wave engine
(apt's pigpio package does not exist on trixie). Everything here is what the
scope needs to reproduce it.

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
    cp kernel/pigpio/pigpio.c.patched /tmp/pigpio/pigpio.c
    make -C /tmp/pigpio
    sudo make -C /tmp/pigpio install
    sudo cp kernel/pigpio/pigpiod.service /etc/systemd/system/
    sudo systemctl enable pigpiod
    cp kernel/pigpio/pigpio.py <ofm-venv>/site-packages/

Installed-binary md5s on x-003 (for verification):
    pigpiod            fe9f8c7c
    libpigpio.so.1     ed458a8e

Usage in `freeze_motion.py`: wave_send_using_mode with
pigpio.WAVE_MODE_REPEAT_SYNC; retunes create-then-delete waves (wave ids
tracked in `_waves`) — never leak wave_create handles, the daemon has a
finite pool (~250) and silent failure past it.
