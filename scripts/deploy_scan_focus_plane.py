import hashlib
import pathlib

import paramiko

LOCAL = (
    r"C:\glia\EU_dotacje\mikroskop\OFM-server\openflexure-microscope-server"
    r"\src\openflexure_microscope_server\things\scan_focus_plane.py"
)
REMOTE = (
    "/home/admin/openflexure-microscope-server/src/openflexure_microscope_server"
    "/things/scan_focus_plane.py"
)
CFG = "/home/admin/ofm/openflexure/settings/ofm_config_full.json"

pw = pathlib.Path(r"C:\Users\Maalus\.ssh\microscope\x-003_microscope_pass.txt").read_text().strip()
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.100.20", username="admin", password=pw, timeout=25, banner_timeout=25, auth_timeout=25)
sftp = c.open_sftp()
data = pathlib.Path(LOCAL).read_bytes()
sftp.put(LOCAL, REMOTE)
_, out, _ = c.exec_command(f"md5sum {REMOTE}", timeout=25)
remote_md5 = out.read().decode().split()[0]
local_md5 = hashlib.md5(data).hexdigest()
print("MD5", "OK" if local_md5 == remote_md5 else "MISMATCH", remote_md5)

_, out, _ = c.exec_command(
    "EXE=$(readlink /proc/$(systemctl show -p MainPID --value "
    "openflexure-microscope-server)/exe); echo EXE=$EXE; "
    "\"$EXE\" -c \"import sys; sys.path.insert(0, '/home/admin/openflexure-microscope-server/src'); "
    "import openflexure_microscope_server.things.scan_focus_plane as m; "
    "import json, pathlib; "
    "cfg = json.loads(pathlib.Path('" + CFG + "').read_text()); "
    "print('LIVE_WIRING autofocus=', cfg['things'].get('autofocus')); "
    "print('LIVE_WIRING scan_focus_plane=', cfg['things'].get('scan_focus_plane')); "
    "print('SCOPE_IMPORT_OK')\"",
    timeout=25,
)
print(out.read().decode(errors="replace"))
sftp.close()
c.close()
