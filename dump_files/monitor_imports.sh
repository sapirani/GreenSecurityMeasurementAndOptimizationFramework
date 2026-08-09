#!/bin/bash
# Monitor import processes on 159 and 184, send email when each finishes.
# 184 is monitored only after its export (PID 3193137) completes.
# Usage: nohup ./monitor_imports.sh > ./monitor_imports.log 2>&1 &

source activate py310_modelenv 2>/dev/null || conda activate py310_modelenv 2>/dev/null

python3 << 'PYEOF'
import paramiko
import smtplib
import time
import os
from email.mime.text import MIMEText

HOSTS = {
    "132.72.80.159": {"done": False, "label": "159", "started": True},
    "132.72.81.184": {"done": False, "label": "184", "started": False},
}
EXPORT_184_PID = 3193137
SSH_USER = "splunk"
SSH_PASS = "splunk1q2w#E$R"
REMOTE_DIR = "/opt/splunk/var/splunk_dumps"
EMAIL = "shouei@post.bgu.ac.il"


def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL
    msg["To"] = EMAIL
    try:
        with smtplib.SMTP("smtp.bgu.ac.il", 25, timeout=30) as server:
            server.ehlo()
            server.send_message(msg)
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")


def is_pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_host(host):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=SSH_USER, password=SSH_PASS, timeout=15)

        stdin, stdout, stderr = ssh.exec_command("ps aux | grep oneshot_import.py | grep -v grep")
        proc = stdout.read().decode().strip()

        stdin, stdout, stderr = ssh.exec_command(f"tail -5 {REMOTE_DIR}/import.log 2>/dev/null")
        log = stdout.read().decode().strip()

        stdin, stdout, stderr = ssh.exec_command(f"ls {REMOTE_DIR}/splunk_export_*.jsonl 2>/dev/null | wc -l")
        remaining = stdout.read().decode().strip()

        ssh.close()
        return bool(proc), log, remaining
    except Exception as e:
        print(f"  Error checking {host}: {e}")
        return True, "", "?"


print("Monitoring imports on 159 and 184...")
print(f"184 import will be monitored after export PID {EXPORT_184_PID} finishes.")

while not all(h["done"] for h in HOSTS.values()):
    # Check if 184 export is done — then mark 184 as started
    # (the watcher script will start the import automatically)
    if not HOSTS["132.72.81.184"]["started"]:
        if not is_pid_alive(EXPORT_184_PID):
            print(f"[{time.strftime('%H:%M:%S')}] 184 export finished. Waiting 3 min for import to start...")
            time.sleep(180)  # give watcher time to start import
            HOSTS["132.72.81.184"]["started"] = True

    for host, info in HOSTS.items():
        if info["done"] or not info["started"]:
            continue

        running, log, remaining = check_host(host)
        print(f"[{time.strftime('%H:%M:%S')}] {info['label']}: running={running}, files_left={remaining}")

        if not running and remaining == "0":
            info["done"] = True
            send_email(
                f"Splunk import on {info['label']} ({host}) COMPLETED",
                f"Import into new_main on {host} has finished.\n\nLast log lines:\n{log}")
        elif not running and log:
            # Process not running but has log = it ran and stopped
            # Check if it's a real failure (files remain) vs hasn't started yet
            if int(remaining) > 0:
                info["done"] = True
                send_email(
                    f"Splunk import on {info['label']} ({host}) FAILED",
                    f"Import process on {host} stopped but {remaining} files remain.\n\nLast log lines:\n{log}")

    time.sleep(120)

print("All imports finished.")
PYEOF
