import pathlib

import paramiko

CFG = "/home/admin/openflexure-microscope-server/ofm_config_full.json"

pw = pathlib.Path(r"C:\Users\Maalus\.ssh\microscope\x-003_microscope_pass.txt").read_text().strip()
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.100.20", username="admin", password=pw, timeout=25, banner_timeout=25, auth_timeout=25)

cmd = (
    "EXE=/home/admin/openflexure-microscope-server/.venv/bin/python; "
    "cat > /tmp/sfp_check.py <<'PYEOF'\n"
    "import sys\n"
    "sys.path.insert(0, '/home/admin/openflexure-microscope-server/src')\n"
    "import json\n"
    "import pathlib\n"
    "p = pathlib.Path('" + CFG + "')\n"
    "cfg = json.loads(p.read_text())\n"
    "things = cfg['things']\n"
    "things['scan_focus_plane'] = 'openflexure_microscope_server.things.scan_focus_plane:ScanFocusPlaneThing'\n"
    "print('LIVE autofocus =', things.get('autofocus'))\n"
    "print('LIVE smooth_autofocus =', things.get('smooth_autofocus'))\n"
    "print('LIVE scan_focus_plane =', things.get('scan_focus_plane'))\n"
    "print('CONFIG_CHECK_ONLY IMPORT_OK')\n"
    "PYEOF\n"
    "\"$EXE\" /tmp/sfp_check.py 2>&1; echo RC=$?\n"
)
_, out, err = c.exec_command(cmd, timeout=30)
print(out.read().decode(errors="replace"))
print(err.read().decode(errors="replace"))
c.close()
