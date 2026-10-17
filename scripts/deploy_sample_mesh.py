import hashlib

import paramiko

HOST = "192.168.100.20"
USER = "admin"
PASS_PATH = r"C:\Users\Maalus\.ssh\microscope\x-003_microscope_pass.txt"
LOCAL = r"C:\glia\EU_dotacje\mikroskop\kallicofork\kalico\klippy\extras\sample_mesh.py"
REMOTE = "/home/admin/klipper/klippy/extras/sample_mesh.py"
VENV_PY = "/home/admin/openflexure-microscope-server/.venv/bin/python"
CFG_CANDIDATES = [
    "/home/admin/printer_data/config/printer.cfg",
    "/home/admin/klipper/printer.cfg",
]

password = open(PASS_PATH).read().strip()
cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, username=USER, password=password, timeout=15)
sftp = cli.open_sftp()

sftp.put(LOCAL, REMOTE)
local_md5 = hashlib.md5(open(LOCAL, "rb").read()).hexdigest()
remote_md5 = hashlib.md5(sftp.open(REMOTE, "rb").read()).hexdigest()
print("MD5 local  :", local_md5)
print("MD5 remote :", remote_md5)
print("MD5 MATCH  :", local_md5 == remote_md5)

cmd = (
    "import sys; sys.path.insert(0, '/home/admin/klipper'); "
    "import klippy.extras.sample_mesh; print('SCOPE_IMPORT_OK')"
)
stdin, stdout, stderr = cli.exec_command(
    f"{VENV_PY} -c '{cmd}'", timeout=30
)
print("IMPORT rc:", stdout.channel.recv_exit_status())
print("IMPORT out:", stdout.read().decode().strip())
err = stderr.read().decode().strip()
if err:
    print("IMPORT err (tail):", err[-800:])

for cfg in CFG_CANDIDATES:
    stdin, stdout, stderr = cli.exec_command(
        f"grep -n '\\[bed_mesh\\]\\|\\[sample_mesh\\]' {cfg} 2>/dev/null; "
        f"echo rc=$?",
        timeout=10,
    )
    out = stdout.read().decode().strip()
    print("CFG", cfg, "->", out.replace("\n", " | "))

sftp.close()
cli.close()
print("DEPLOY_DONE")
