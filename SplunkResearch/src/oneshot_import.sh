#!/bin/bash
# Import JSONL dump files into Splunk via REST API
# Extracts _raw from JSON and posts to /services/receivers/simple
# Run on destination Splunk VMs (184, 159)
#
# Usage: nohup ./oneshot_import.sh [index_name] [dump_dir] > /opt/splunk/var/splunk_dumps/oneshot.log 2>&1 &

INDEX="${1:-new_main}"
DUMP_DIR="${2:-/opt/splunk/var/splunk_dumps}"
SPLUNK="/opt/splunk/bin/splunk"
AUTH_USER="splunk"
AUTH_PASS='splunk1q2w#E$R'
LOG="$DUMP_DIR/oneshot_import.log"

echo "=== Import started at $(date) ===" | tee -a "$LOG"
echo "Index: $INDEX, Dump dir: $DUMP_DIR" | tee -a "$LOG"

# Create index if it doesn't exist
$SPLUNK add index "$INDEX" -auth "$AUTH_USER:$AUTH_PASS" 2>/dev/null

GRAND_TOTAL=0
FILE_COUNT=0

for f in $(ls "$DUMP_DIR"/splunk_export_*.jsonl 2>/dev/null | sort); do
    [ -f "$f" ] || continue
    BASENAME=$(basename "$f")
    FILE_COUNT=$((FILE_COUNT + 1))
    echo "[$FILE_COUNT] Processing: $BASENAME ..." | tee -a "$LOG"

    # Use python to read JSONL and post events via REST
    COUNT=$(python3 << 'PYEOF'
import json, sys, urllib.request, urllib.error, ssl

index = "__INDEX__"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

count = 0
errors = 0

with open("__FILE__") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        raw = event.get("_raw", "")
        if not raw:
            continue
        host = event.get("host", "unknown")
        source = event.get("source", "WinEventLog")
        sourcetype = event.get("sourcetype", "WinEventLog")

        url = f"https://localhost:8089/services/receivers/simple?index={index}&host={host}&source={source}&sourcetype={sourcetype}"
        req = urllib.request.Request(url, data=raw.encode("utf-8"), method="POST")
        req.add_header("Authorization", "Basic " + __import__("base64").b64encode(b"splunk:splunk1q2w#E$R").decode())

        try:
            urllib.request.urlopen(req, context=ctx, timeout=30)
            count += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"Error: {e}", file=sys.stderr)

        if count % 10000 == 0 and count > 0:
            print(f"  {count} events...", file=sys.stderr)

print(count)
PYEOF
)

    # Replace placeholders
    COUNT=$(python3 -c "
import json, sys, urllib.request, urllib.error, ssl, base64

index = '$INDEX'
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
auth = base64.b64encode(b'$AUTH_USER:$AUTH_PASS').decode()

count = 0
errors = 0

with open('$f') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        raw = event.get('_raw', '')
        if not raw:
            continue
        host = event.get('host', 'unknown')
        source = event.get('source', 'WinEventLog')
        sourcetype = event.get('sourcetype', 'WinEventLog')

        url = f'https://localhost:8089/services/receivers/simple?index={index}&host={host}&source={source}&sourcetype={sourcetype}'
        req = urllib.request.Request(url, data=raw.encode('utf-8'), method='POST')
        req.add_header('Authorization', 'Basic ' + auth)

        try:
            urllib.request.urlopen(req, context=ctx, timeout=30)
            count += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'Error: {e}', file=sys.stderr)

        if count % 10000 == 0 and count > 0:
            print(f'  {count} events...', file=sys.stderr)

print(count)
" 2>&1 | tee -a "$LOG" | tail -1)

    GRAND_TOTAL=$((GRAND_TOTAL + COUNT))
    rm -f "$f"
    echo "  Done: $COUNT events. Grand total: $GRAND_TOTAL. Deleted." | tee -a "$LOG"
done

echo "=== Finished at $(date). Total: $GRAND_TOTAL events ===" | tee -a "$LOG"
