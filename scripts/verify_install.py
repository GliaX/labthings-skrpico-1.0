import json
import pathlib

import paramiko

pw = pathlib.Path(r"C:\Users\Maalus\.ssh\microscope\x-003_microscope_pass.txt").read_text().strip()
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.100.20", username="admin", password=pw, timeout=25, banner_timeout=25, auth_timeout=25)

def run(cmd):
    _, out, err = c.exec_command(cmd, timeout=25)
    return out.read().decode(errors="replace"), err.read().decode(errors="replace")

inv, _ = run("curl -s http://127.0.0.1:5000/api/v3/action_invocations/e64ca883-9271-4de6-9c9e-d2fcfefbadba")
try:
    d = json.loads(inv)
    print("status:", d.get("status"))
    r = d.get("output") or {}
    print("installed:", r.get("installed"))
    print("install_response:", json.dumps(r.get("install_response"))[:500])
except Exception as e:
    print("parse err", e, inv[:300])

status, _ = run("curl -s -X POST http://127.0.0.1:7125/printer/gcode/script -H 'Content-Type: application/json' -d '{\"script\": \"SAMPLE_MESH_STATUS\"}'")
print("status cmd:", status.strip())

log, _ = run("awk '/Start printer/{n=NR} END{print n}' /home/admin/printer_data/logs/klippy.log")
start_line = int(log.strip() or 0)
g, _ = run(f"grep -n 'sample_mesh' /home/admin/printer_data/logs/klippy.log | awk -F: '$1 > {start_line}'")
print("--- klippy.log sample_mesh (this boot) ---")
print(g)
c.close()
