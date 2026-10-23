# kernel/ — IMX477 overclocked driver artifacts (x-003)

2800x2800 @ 12-bit window mode + Open360Camera-style clock overclock (640M MIPI link, 984M pixel rate).
Measured 21.99 fps vs 10 fps stock on x-003 (kernel 6.18.29+rpt-rpi-v8).

## Files

- `imx477-ofm2800.c` — driver source. Diffs vs upstream 6.18 `imx477.c`:
  - `IMX477_DEFAULT_LINK_FREQ` 450M -> 640M
  - `IMX477_IOP_PREDIV` 0x02 -> 0x03 (sensor reg 0x030d)
  - `IMX477_PIXEL_RATE` 840M -> 984M
  - new `mode_2800x2800_regs` table (window 628,120 out 2800x2800, frm_length 2848) + mode struct entry
- `imx477-ofm2800.ko` — built module (renamed to `imx477.ko` when installing)
- `imx477-build-Makefile` — `obj-m := imx477.o` (name it `Makefile` in the build dir)
- `imx477-640M.dtbo` — device tree overlay with `link-frequencies = <640000000>`; stock backed up on scope as `imx477.dtbo.stock`
- `dtso/` — stock rpi-6.18.y overlay sources (450M); the 640M dtbo = these with `450000000 -> 640000000` in `imx477_378.dtsi`

## Rebuild module (on the Pi)

    make -C /lib/modules/$(uname -r)/build M=$PWD modules

## Rebuild dtbo (on the Pi, needs cpp + dtc)

    cpp -nostdinc -I . -x assembler-with-cpp imx477-overlay.dts imx477-pre.dts
    dtc -@ -Hepapr -I dts -O dtb -o imx477-640M.dtbo imx477-pre.dts

Edit the 450M value in `imx477_378.dtsi` BEFORE preprocessing. Decompile-edit-recompile of the binary dtbo does NOT work (phandle fixups don't round-trip).

## Install

    sudo cp imx477.ko /lib/modules/$(uname -r)/kernel/drivers/media/i2c/imx477.ko
    sudo cp imx477-640M.dtbo /boot/firmware/overlays/imx477.dtbo
    sudo depmod -a

Driver and DT must match: PREDIV=3 driver only accepts 640M DT; stock driver + 640M DT = camera dead at boot (probe -22).
