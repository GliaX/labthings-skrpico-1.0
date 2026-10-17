import hashlib
import pathlib

import paramiko

FILES = [
    (
        r"C:\glia\EU_dotacje\mikroskop\OFM-server\openflexure-microscope-server"
        r"\src\openflexure_microscope_server\things\smooth_autofocus.py",
        "/home/admin/openflexure-microscope-server/src/openflexure_microscope_server/things/smooth_autofocus.py",
    ),
    (
        r"C:\glia\EU_dotacje\mikroskop\OFM-server\openflexure-microscope-server"
        r"\src\openflexure_microscope_server\things\smooth_af_core.py",
        "/home/admin/openflexure-microscope-server/src/openflexure_microscope_server/things/smooth_af_core.py",
    ),
    (
        r"C:\glia\EU_dotacje\mikroskop\OFM-server\openflexure-microscope-server"
        r"\src\openflexure_microscope_server\things\camera\picamera.py",
        "/home/admin/openflexure-microscope-server/src/openflexure_microscope_server/things/camera/picamera.py",
    ),
    (
        r"C:\glia\EU_dotacje\mikroskop\OFM-server\openflexure-microscope-server"
        r"\src\openflexure_microscope_server\things\stage\skrPico.py",
        "/home/admin/openflexure-microscope-server/src/openflexure_microscope_server/things/stage/skrPico.py",
    ),
]

pw = pathlib.Path(r"C:\Users\Maalus\.ssh\microscope\x-003_microscope_pass.txt").read_text().strip()
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.100.20", username="admin", password=pw, timeout=25, banner_timeout=25, auth_timeout=25)
sftp = c.open_sftp()
for local, remote in FILES:
    data = pathlib.Path(local).read_bytes()
    sftp.put(local, remote)
    _, out, _ = c.exec_command(f"md5sum {remote}", timeout=25)
    remote_md5 = out.read().decode().split()[0]
    local_md5 = hashlib.md5(data).hexdigest()
    status = "OK " if local_md5 == remote_md5 else "MISMATCH"
    print(f"{status} {remote_md5} {remote}")
sftp.close()

_, out, _ = c.exec_command(
    "EXE=$(readlink /proc/$(systemctl show -p MainPID --value "
    "openflexure-microscope-server)/exe); echo EXE=$EXE; "
    "\"$EXE\" -c \"import sys; sys.path.insert(0, '/home/admin/openflexure-microscope-server/src'); "
    "import openflexure_microscope_server.things.smooth_af_core; "
    "import openflexure_microscope_server.things.smooth_autofocus; "
    "import openflexure_microscope_server.things.camera.picamera; "
    "print('SCOPE_IMPORT_OK')\"",
    timeout=25,
)
print(out.read().decode(errors="replace"))
c.close()
