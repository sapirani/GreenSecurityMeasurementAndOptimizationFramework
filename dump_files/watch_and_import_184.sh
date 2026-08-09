#!/bin/bash
# Watch the 184 export process, then start import on 184 when done.
# Usage: nohup ./watch_and_import_184.sh > ./watch_184.log 2>&1 &

EXPORT_PID=3193137
REMOTE_HOST="132.72.81.184"
REMOTE_USER="splunk"
REMOTE_PASS='splunk1q2w#E$R'
REMOTE_DIR="/opt/splunk/var/splunk_dumps"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Watching export PID $EXPORT_PID ..."

# Wait for export to finish
while kill -0 $EXPORT_PID 2>/dev/null; do
    sleep 60
done

echo "Export finished at $(date). Checking result..."
tail -5 "$SCRIPT_DIR/dump_files/export_scp_184.log"

echo ""
echo "Starting import on $REMOTE_HOST ..."

# Use python+paramiko to start import remotely
source activate py310_modelenv 2>/dev/null || conda activate py310_modelenv 2>/dev/null
python3 -c "
import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('$REMOTE_HOST', username='$REMOTE_USER', password='$REMOTE_PASS', timeout=30)

# Upload latest import script
sftp = ssh.open_sftp()
sftp.put('$SCRIPT_DIR/SplunkResearch/src/oneshot_import.py', '$REMOTE_DIR/oneshot_import.py')
sftp.close()
print('Uploaded import script')

# Start import
stdin, stdout, stderr = ssh.exec_command('nohup python3 $REMOTE_DIR/oneshot_import.py --index new_main --dump-dir $REMOTE_DIR > $REMOTE_DIR/import.log 2>&1 & echo PID=\$!')
print('184 import:', stdout.read().decode().strip())
ssh.close()
"

echo "Import started on $REMOTE_HOST at $(date)"
echo "Monitor with: ssh splunk@$REMOTE_HOST 'tail -10 $REMOTE_DIR/import.log'"
