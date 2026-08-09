#!/usr/bin/env python3
"""Import JSONL dump files into local Splunk index.

Uses urllib to POST to /services/receivers/stream (persistent connection).
No external dependencies required — uses only stdlib.

Usage (on the Splunk VM):
    nohup python3 oneshot_import.py [--index new_main] [--dump-dir /opt/splunk/var/splunk_dumps] \
        > /opt/splunk/var/splunk_dumps/import.log 2>&1 &
"""
import argparse
import base64
import json
import http.client
import ssl
import sys
import time
import urllib.parse
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def stream_events(host, port, auth_header, index, events, src_host, source, sourcetype):
    """Stream a batch of raw events via /services/receivers/stream."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    params = urllib.parse.urlencode({
        "index": index, "host": src_host, "source": source, "sourcetype": sourcetype
    })
    path = f"/services/receivers/stream?{params}"

    conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=120)
    # Chunked transfer encoding for streaming
    conn.putrequest("POST", path)
    conn.putheader("Authorization", auth_header)
    conn.putheader("Transfer-Encoding", "chunked")
    conn.putheader("x-splunk-input-mode", "streaming")
    conn.endheaders()

    for raw in events:
        data = (raw + "\n").encode("utf-8")
        # Send as chunked
        conn.send(f"{len(data):X}\r\n".encode())
        conn.send(data)
        conn.send(b"\r\n")

    # End chunked transfer
    conn.send(b"0\r\n\r\n")
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp.status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="new_main")
    parser.add_argument("--dump-dir", default="/opt/splunk/var/splunk_dumps")
    parser.add_argument("--splunk-host", default="localhost")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--username", default="splunk")
    parser.add_argument("--password", default="splunk1q2w#E$R")
    args = parser.parse_args()

    auth = base64.b64encode(f"{args.username}:{args.password}".encode()).decode()
    auth_header = f"Basic {auth}"

    dump_path = Path(args.dump_dir)
    files = sorted(dump_path.glob("splunk_export_*.jsonl"))
    print(f"Found {len(files)} dump files in {dump_path}")
    if not files:
        return

    grand_total = 0
    chunk_size = 5000

    for fi, f in enumerate(files):
        print(f"\n[{fi+1}/{len(files)}] {f.name}")

        # Read and group events by (host, source, sourcetype)
        groups = {}
        line_count = 0
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                raw = event.get("_raw", "")
                if not raw:
                    continue
                key = (event.get("host", "unknown"),
                       event.get("source", "WinEventLog"),
                       event.get("sourcetype", "WinEventLog"))
                if key not in groups:
                    groups[key] = []
                groups[key].append(raw)
                line_count += 1

        # Stream each group
        file_injected = 0
        for (host, source, sourcetype), raws in groups.items():
            for ci in range(0, len(raws), chunk_size):
                chunk = raws[ci:ci + chunk_size]
                for attempt in range(5):
                    try:
                        status = stream_events(
                            args.splunk_host, args.port, auth_header,
                            args.index, chunk, host, source, sourcetype)
                        if status < 300:
                            file_injected += len(chunk)
                            break
                        else:
                            print(f"  HTTP {status} (attempt {attempt+1})")
                            time.sleep(5 * (attempt + 1))
                    except Exception as e:
                        wait = 10 * (2 ** attempt)
                        print(f"  Error (attempt {attempt+1}): {e}, retry in {wait}s")
                        time.sleep(wait)
                        if attempt == 4:
                            print("  FATAL: giving up")
                            raise

        grand_total += file_injected
        f.unlink()
        print(f"  Done: {file_injected}/{line_count} events. Grand total: {grand_total}. Deleted.")

    print(f"\n=== Complete! Total: {grand_total} events ===")


if __name__ == "__main__":
    main()
