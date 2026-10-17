import hashlib
import json
import pathlib
import shutil

import paramiko

OVERLAY = pathlib.Path(r"C:\glia\EU_dotacje\mikroskop\python\labthings-skrpico-1.0")
OFM_SRC = pathlib.Path(
    r"C:\glia\EU_dotacje\mikroskop\OFM-server\openflexure-microscope-server\src\openflexure_microscope_server\things")
STAGE = pathlib.Path(r"C:\glia\EU_dotacje\mikroskop\OFM-server\openflexure-microscope-server\.agent-work")

pw = (pathlib.Path(r"C:\Users\Maalus\.ssh\microscope\x-003_microscope_pass.txt")).read_text().strip()
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.100.20", username="admin", password=pw, timeout=25, banner_timeout=25, auth_timeout=25)
sftp = c.open_sftp()

sftp.get("/home/admin/printer_data/config/printer.cfg", str(STAGE / "printer.cfg.from-scope"))
sftp.close()
c.close()


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


shutil.copyfile(STAGE / "printer.cfg.from-scope", OVERLAY / "printer.cfg")
print("printer.cfg -> overlay:", md5(OVERLAY / "printer.cfg"), "(scope copy:", md5(STAGE / "printer.cfg.from-scope"),
      ")")

cfgp = OVERLAY / "ofm_config_full.json"
cfg_text = cfgp.read_text()
if "scan_focus_plane" not in cfg_text:
    cfg_text = cfg_text.replace(
        '"smooth_autofocus": "openflexure_microscope_server.things.smooth_autofocus:SmoothAutofocusThing",',
        '"smooth_autofocus": "openflexure_microscope_server.things.smooth_autofocus:SmoothAutofocusThing",\n        "scan_focus_plane": "openflexure_microscope_server.things.scan_focus_plane:ScanFocusPlaneThing",',
    )
    cfgp.write_text(cfg_text)
print("ofm_config_full.json has scan_focus_plane:", "scan_focus_plane" in cfgp.read_text())
print("  json valid:", bool(json.loads(cfgp.read_text())))

mirror = OVERLAY / "ofm"
mirror.mkdir(exist_ok=True)
for name, src in (
        ("picamera.py", OFM_SRC / "camera" / "picamera.py"),
        ("smooth_autofocus.py", OFM_SRC / "smooth_autofocus.py"),
        ("scan_focus_plane.py", OFM_SRC / "scan_focus_plane.py"),
        ("smooth_af_core.py", OFM_SRC / "smooth_af_core.py"),
):
    dst = mirror / name
    shutil.copyfile(src, dst)
    same = md5(src) == md5(dst)
    print(f"ofm/{name}: {md5(dst)} mirror==repo {same}")

kpath = r"C:\glia\EU_dotacje\mikroskop\kallicofork\kalico\klippy\extras\sample_mesh.py"
print("kalico sample_mesh.py:", md5(kpath))
