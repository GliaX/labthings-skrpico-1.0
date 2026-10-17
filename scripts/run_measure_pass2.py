import json
import pathlib
import re
import time

import paramiko

pw = pathlib.Path(r"C:\Users\Maalus\.ssh\microscope\x-003_microscope_pass.txt").read_text().strip()
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.100.20", username="admin", password=pw, timeout=25, banner_timeout=25, auth_timeout=25)


def curl(args):
    _, out, err = c.exec_command("curl -s -X POST " + args, timeout=30)
    return out.read().decode(errors="replace"), err.read().decode(errors="replace")


body, _err = curl(
    "http://127.0.0.1:5000/api/v3/scan_focus_plane/measure_pass "
    "-H 'Content-Type: application/json' -d '{\"grid\":3,\"step\":150.0,\"install\":true}'"
)
m = re.search(r'"id"\s*:\s*"([0-9a-f-]{36})"', body)
if not m:
    print("NO_INVOCATION_ID")
    print(body[:2000])
    raise SystemExit(1)
inv = m.group(1)
print("INVOCATION", inv)

url = f"http://127.0.0.1:5000/api/v3/action_invocations/{inv}"
for i in range(40):
    time.sleep(3)
    _, out, _ = c.exec_command(f"curl -s {url}", timeout=30)
    st = json.loads(out.read().decode(errors="replace"))
    status = st.get("status")
    print(f"poll {i} status={status}")
    if status in ("completed", "error"):
        print("--- log ---")
        for entry in st.get("log", []):
            print(entry.get("levelname"), entry.get("message"))
        print("--- output ---")
        _, out, _ = c.exec_command(f"curl -s {url}/output", timeout=30)
        print(out.read().decode(errors="replace")[:3000])
        break
c.close()
