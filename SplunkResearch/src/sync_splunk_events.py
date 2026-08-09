#!/usr/bin/env python3
"""
Sync Windows events between two Splunk instances.

1. Query Splunk 1 (destination) for the latest event timestamp
2. Export events from Splunk 2 (source) since that timestamp using the
   streaming export endpoint (bypasses the 50K result cap)
3. Index those events into Splunk 1

Usage:
    # Full export+import with checkpoint and email on completion:
    python -m SplunkResearch.src.sync_splunk_events \
        --dst-host 132.72.81.184 --dump-dir ./dump_files --import-only --notify

    # Resume after interruption (automatically skips already-indexed events):
    python -m SplunkResearch.src.sync_splunk_events \
        --dst-host 132.72.81.184 --dump-dir ./dump_files --import-only --notify
"""

import argparse
import json
import smtplib
import ssl
import time
import traceback
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import sys
import urllib.request
import urllib.error
import splunklib.client as client
import splunklib.results as results

# Force unbuffered output for tmux/nohup
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEFAULT_PORT = 8089
DEFAULT_USERNAME = "splunk"
DEFAULT_PASSWORD = "splunk1q2w#E$R"
DEFAULT_INDEX = "main"
DEFAULT_SOURCETYPE = "WinEventLog"
EXCLUDED_HOSTS = {"132.72.81.150", "dt-splunk"}
DUMP_CHUNK_SIZE = 100000
HEC_PORT = 8088
HEC_BATCH_SIZE = 500  # events per HEC request
HEC_TOKENS = {
    "132.72.81.184": "dbded243-069e-48eb-a2d6-7f96db39a1f7",
    "132.72.80.159": "638087d2-2941-4de0-9ea1-d3bc106dad3f",
}


def send_email(subject, body):
    """Send notification email."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "shouei@post.bgu.ac.il"
    msg["To"] = "shouei@post.bgu.ac.il"
    try:
        with smtplib.SMTP("smtp.bgu.ac.il", 25, timeout=30) as server:
            server.ehlo()
            server.send_message(msg)
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")


def connect(host, port=DEFAULT_PORT, username=DEFAULT_USERNAME, password=DEFAULT_PASSWORD):
    print(f"Connecting to {host}:{port} ...")
    service = client.connect(
        host=host, port=port,
        username=username, password=password,
        autologin=True,
    )
    print(f"Connected to {host}")
    return service


def get_latest_timestamp(service, index=DEFAULT_INDEX, sourcetype=DEFAULT_SOURCETYPE):
    query = f'search index={index} sourcetype="{sourcetype}" | head 1 | fields _time'
    job = service.jobs.create(query, **{"earliest_time": "0", "latest_time": "now", "output_mode": "json"})
    while not job.is_done():
        time.sleep(1)
    latest = None
    for result in results.JSONResultsReader(job.results(output_mode="json")):
        if isinstance(result, dict) and "_time" in result:
            latest = result["_time"]
    job.cancel()
    if latest:
        print(f"Latest event in destination: {latest}")
    else:
        print("No existing events in destination — will export ALL events.")
    return latest


def _export_time_range(service, earliest, latest, index, sourcetype, dump_path,
                       ts, file_idx_start):
    """Export a single time range. Returns (total, skipped, next_file_idx)."""
    host_filter = " ".join(f'host!={h}' for h in EXCLUDED_HOSTS)
    query = f'search index={index} sourcetype="{sourcetype}" {host_filter}'
    kwargs = {
        "earliest_time": earliest,
        "latest_time": latest,
        "output_mode": "json",
        "search_mode": "normal",
    }

    print(f"  Exporting {earliest} -> {latest} ...")
    export_stream = service.jobs.export(query, **kwargs)

    total = 0
    skipped = 0
    file_idx = file_idx_start
    chunk_count = 0
    current_file = dump_path / f"splunk_export_{ts}_{file_idx:04d}.jsonl"
    fh = open(current_file, "w")

    try:
        for result in results.JSONResultsReader(export_stream):
            if not isinstance(result, dict):
                continue

            host = result.get("host", "")
            if host in EXCLUDED_HOSTS:
                skipped += 1
                continue

            fh.write(json.dumps(result) + "\n")
            total += 1
            chunk_count += 1

            if chunk_count >= DUMP_CHUNK_SIZE:
                fh.close()
                print(f"    wrote {current_file.name} ({chunk_count} events)")
                file_idx += 1
                chunk_count = 0
                current_file = dump_path / f"splunk_export_{ts}_{file_idx:04d}.jsonl"
                fh = open(current_file, "w")

            if total % 10000 == 0:
                print(f"    exported {total} events (skipped {skipped} fake) ...")
    finally:
        fh.close()

    if chunk_count == 0 and file_idx > file_idx_start:
        current_file.unlink()
    elif chunk_count > 0:
        print(f"    wrote {current_file.name} ({chunk_count} events)")
        file_idx += 1

    print(f"  Range done: {total} events, {skipped} skipped.")
    return total, skipped, file_idx


def _generate_day_ranges(start_year, start_month, start_day=1, end_date=None):
    """Generate (earliest, latest) daily boundaries from start to end_date (or now)."""
    from datetime import datetime as dt, timedelta
    current = dt(start_year, start_month, start_day)
    now = dt(*end_date) if end_date else dt.now()
    while current < now:
        next_day = min(current + timedelta(days=1), now)
        yield current.strftime("%Y-%m-%dT%H:%M:%S"), next_day.strftime("%Y-%m-%dT%H:%M:%S")
        current = next_day


def export_events(service, since_time, index=DEFAULT_INDEX, sourcetype=DEFAULT_SOURCETYPE,
                  dump_dir=None):
    """
    Export events week-by-week using the streaming jobs/export endpoint.
    Streams directly to JSONL dump files to avoid OOM.
    Filters out fake injected logs (EXCLUDED_HOSTS).
    """
    dump_path = Path(dump_dir) if dump_dir else Path("./dump_files")
    dump_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Filtering out hosts: {EXCLUDED_HOSTS}")
    print(f"Exporting day-by-day ...")

    grand_total = 0
    grand_skipped = 0
    file_idx = 0

    for earliest, latest in _generate_day_ranges(2025, 5, 8):
        total, skipped, file_idx = _export_time_range(
            service, earliest, latest, index, sourcetype, dump_path, ts, file_idx)
        grand_total += total
        grand_skipped += skipped

    print(f"Exported {grand_total} events total, skipped {grand_skipped} fake logs.")
    return dump_path


def export_scp_delete(src_host, dst_hosts, port=DEFAULT_PORT, username=DEFAULT_USERNAME,
                      password=DEFAULT_PASSWORD, src_index=DEFAULT_INDEX,
                      sourcetype=DEFAULT_SOURCETYPE, start_date=(2024, 3, 1),
                      end_date=None, ssh_user="splunk", ssh_password="splunk1q2w#E$R",
                      remote_dir="/data/splunk_dumps"):
    """Export day-by-day from src, SCP to destinations, delete local dump.

    Creates fresh connections per day to avoid splunklib memory leaks.
    Dump files are SCPed to remote_dir on each dst host, then deleted locally.
    """
    import gc
    dump_path = Path("./dump_files")
    dump_path.mkdir(parents=True, exist_ok=True)

    ckpt_file = dump_path / ".pipeline_scp_ckpt.json"
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        last_date = ckpt.get("last_completed_date", "")
        grand_total = ckpt.get("grand_total", 0)
        print(f"Resuming from after {last_date}, grand_total={grand_total}")
    else:
        last_date = ""
        grand_total = 0

    days = list(_generate_day_ranges(*start_date, end_date=end_date))
    print(f"Export-SCP pipeline: {len(days)} days, src={src_host}/{src_index} -> {dst_hosts}")

    import paramiko

    def _scp_file(local_path, remote_host, remote_path):
        """SCP a file using paramiko SFTP."""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(remote_host, username=ssh_user, password=ssh_password, timeout=30)
        sftp = ssh.open_sftp()
        sftp.put(str(local_path), remote_path)
        sftp.close()
        ssh.close()

    def _ssh_cmd(host, cmd):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=ssh_user, password=ssh_password, timeout=30)
        stdin, stdout, stderr = ssh.exec_command(cmd)
        stdout.read()
        ssh.close()

    # Create remote dirs on each host
    for dh in dst_hosts:
        _ssh_cmd(dh, f"mkdir -p {remote_dir}")
        print(f"  Created {remote_dir} on {dh}")

    for day_idx, (earliest, latest) in enumerate(days):
        if earliest <= last_date:
            continue

        print(f"\n=== Day {day_idx+1}/{len(days)}: {earliest} -> {latest} ===")

        src_svc = connect(src_host, port, username, password)
        ts = earliest.replace("-", "").replace(":", "").replace("T", "_")

        total, skipped, _ = _export_time_range(
            src_svc, earliest, latest, src_index, sourcetype, dump_path, ts, 0)
        del src_svc
        gc.collect()

        if total == 0:
            print(f"  No events, skipping.")
            for ef in dump_path.glob(f"splunk_export_{ts}_*.jsonl"):
                ef.unlink()
            with open(ckpt_file, "w") as f:
                json.dump({"last_completed_date": earliest, "grand_total": grand_total}, f)
            continue

        # SCP each file to each destination, then delete local
        day_files = sorted(dump_path.glob(f"splunk_export_{ts}_*.jsonl"))
        for df in day_files:
            for dh in dst_hosts:
                for attempt in range(3):
                    try:
                        _scp_file(df, dh, f"{remote_dir}/{df.name}")
                        print(f"  SCP {df.name} -> {dh} OK")
                        break
                    except Exception as e:
                        print(f"  SCP {df.name} -> {dh} failed (attempt {attempt+1}): {e}")
                        if attempt < 2:
                            time.sleep(5)
                        else:
                            raise
            df.unlink()
            print(f"  Deleted local {df.name}")

        grand_total += total
        print(f"  Day done: {total} events. Grand total: {grand_total}")

        with open(ckpt_file, "w") as f:
            json.dump({"last_completed_date": earliest, "grand_total": grand_total}, f)

    ckpt_file.unlink(missing_ok=True)
    print(f"\nExport-SCP pipeline complete! Total events: {grand_total}")
    return grand_total


# ── Checkpoint-based import ──────────────────────────────────────────────────

def _checkpoint_path(dump_dir, dst_host, index):
    """Unique checkpoint file per destination+index combo."""
    return Path(dump_dir) / f".checkpoint_{dst_host}_{index}.json"


def _load_checkpoint(ckpt_file):
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        print(f"Resuming from checkpoint: file={ckpt['file_idx']}, line={ckpt['line_idx']}, "
              f"total_indexed={ckpt['total_indexed']}")
        return ckpt
    return {"file_idx": 0, "line_idx": 0, "total_indexed": 0}


def _save_checkpoint(ckpt_file, file_idx, line_idx, total_indexed):
    with open(ckpt_file, "w") as f:
        json.dump({"file_idx": file_idx, "line_idx": line_idx,
                    "total_indexed": total_indexed}, f)


def _hec_post(dst_host, hec_token, batch):
    """Post a batch of events to Splunk HEC. Returns True on success."""
    # Build newline-delimited JSON payload
    payload = ""
    for event in batch:
        hec_event = {
            "event": event.get("_raw", json.dumps(event)),
            "host": event.get("host", "unknown"),
            "source": event.get("source", "WinEventLog"),
            "sourcetype": event.get("sourcetype", DEFAULT_SOURCETYPE),
        }
        # Preserve original timestamp if available
        if "_time" in event:
            try:
                from datetime import datetime as dt
                # Try ISO format first
                t = event["_time"]
                if "T" in t:
                    parsed = dt.fromisoformat(t.replace("Z", "+00:00"))
                    hec_event["time"] = parsed.timestamp()
            except Exception:
                pass
        payload += json.dumps(hec_event) + "\n"

    url = f"http://{dst_host}:{HEC_PORT}/services/collector/event"
    req = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={
            "Authorization": f"Splunk {hec_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            body = json.loads(resp.read())
            if body.get("code") != 0:
                print(f"  HEC warning: {body}")
            return True
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            print(f"  HEC error (attempt {attempt + 1}): {e.code} {err_body}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  HEC error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False


def index_events_with_checkpoint(service, dump_dir, dst_host,
                                  index=DEFAULT_INDEX, sourcetype=DEFAULT_SOURCETYPE):
    """Index events from JSONL dump files via HEC batch with checkpoint for resume."""
    hec_token = HEC_TOKENS.get(dst_host)
    if not hec_token:
        raise ValueError(f"No HEC token configured for {dst_host}")

    files = sorted(Path(dump_dir).glob("splunk_export_*.jsonl"))
    if not files:
        print("No dump files found!")
        return 0

    ckpt_file = _checkpoint_path(dump_dir, dst_host, index)
    ckpt = _load_checkpoint(ckpt_file)
    start_file = ckpt["file_idx"]
    start_line = ckpt["line_idx"]
    total = ckpt["total_indexed"]

    print(f"Indexing into {index} on {dst_host} via HEC (batch={HEC_BATCH_SIZE}, "
          f"{len(files)} dump files) ...")

    batch = []
    batch_start_line = start_line

    for fi, f in enumerate(files):
        if fi < start_file:
            continue

        print(f"Processing {f.name} (file {fi + 1}/{len(files)}) ...")
        with open(f) as fh:
            for li, line in enumerate(fh):
                if fi == start_file and li < start_line:
                    continue

                line = line.strip()
                if not line:
                    continue

                batch.append(json.loads(line))

                if len(batch) >= HEC_BATCH_SIZE:
                    if not _hec_post(dst_host, hec_token, batch):
                        print(f"  FATAL: HEC post failed after retries at total={total}")
                        _save_checkpoint(ckpt_file, fi, batch_start_line, total)
                        return total
                    total += len(batch)
                    batch = []
                    batch_start_line = li + 1

                    if total % 50000 == 0:
                        _save_checkpoint(ckpt_file, fi, li + 1, total)
                        print(f"  indexed {total} ...")

        # Flush remaining batch for this file
        if batch:
            if _hec_post(dst_host, hec_token, batch):
                total += len(batch)
            batch = []

        _save_checkpoint(ckpt_file, fi + 1, 0, total)
        batch_start_line = 0

    ckpt_file.unlink(missing_ok=True)
    print(f"Indexed {total} events total.")
    return total


def index_events_splunklib(service, dump_dir, dst_host, index_name="new_main",
                            sourcetype=DEFAULT_SOURCETYPE, delete_after=True):
    """Index events from JSONL dump files via splunklib index.submit().

    Slower than HEC but works when HEC port is blocked.
    Deletes each dump file after successful injection to save disk space.
    """
    files = sorted(Path(dump_dir).glob("splunk_export_*.jsonl"))
    if not files:
        print("No dump files found!")
        return 0

    ckpt_file = _checkpoint_path(dump_dir, dst_host, index_name)
    ckpt = _load_checkpoint(ckpt_file)
    start_file = ckpt["file_idx"]
    start_line = ckpt["line_idx"]
    total = ckpt["total_indexed"]

    idx = service.indexes[index_name]
    print(f"Indexing into '{index_name}' on {dst_host} via splunklib "
          f"({len(files)} dump files, resume from file={start_file} line={start_line}) ...")

    for fi, f in enumerate(files):
        if fi < start_file:
            continue

        file_count = 0
        print(f"Processing {f.name} (file {fi + 1}/{len(files)}) ...")
        with open(f) as fh:
            for li, line in enumerate(fh):
                if fi == start_file and li < start_line:
                    continue

                line = line.strip()
                if not line:
                    continue

                event = json.loads(line)
                raw = event.get("_raw", json.dumps(event))

                for attempt in range(3):
                    try:
                        idx.submit(
                            raw.encode("utf-8"),
                            host=event.get("host", "unknown"),
                            source=event.get("source", "WinEventLog"),
                            sourcetype=event.get("sourcetype", sourcetype),
                        )
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                        else:
                            print(f"  FATAL: submit failed at total={total}: {e}")
                            _save_checkpoint(ckpt_file, fi, li, total)
                            return total

                total += 1
                file_count += 1

                if total % 10000 == 0:
                    _save_checkpoint(ckpt_file, fi, li + 1, total)
                    print(f"  indexed {total} (file_count={file_count}) ...")

        _save_checkpoint(ckpt_file, fi + 1, 0, total)
        print(f"  Finished {f.name}: {file_count} events. Total={total}")

        if delete_after:
            f.unlink()
            print(f"  Deleted {f.name}")

    ckpt_file.unlink(missing_ok=True)
    print(f"Indexed {total} events total.")
    return total


def iter_dump_files(dump_dir):
    """Iterate events from JSONL dump files, yielding one event at a time."""
    files = sorted(Path(dump_dir).glob("splunk_export_*.jsonl"))
    if not files:
        print("No dump files found!")
        return
    for f in files:
        print(f"Reading {f.name} ...")
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Sync Windows events between Splunk instances")
    parser.add_argument("--src-host", default="132.72.81.150", help="Source Splunk host")
    parser.add_argument("--dst-host", default="132.72.81.184", help="Destination Splunk host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--sourcetype", default=DEFAULT_SOURCETYPE)
    parser.add_argument("--dump-dir", help="Directory for dump files")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument("--import-splunklib", action="store_true",
                        help="Import via splunklib (slower, works when HEC is blocked)")
    parser.add_argument("--delete-after", action="store_true",
                        help="Delete dump files after successful injection")
    parser.add_argument("--pipeline", action="store_true",
                        help="Export-inject-delete pipeline (day by day)")
    parser.add_argument("--export-scp", action="store_true",
                        help="Export day-by-day, SCP to destinations, delete local")
    parser.add_argument("--dst-hosts", nargs="+", default=[],
                        help="Destination hosts for SCP (e.g. 132.72.81.184 132.72.80.159)")
    parser.add_argument("--remote-dir", default="/data/splunk_dumps",
                        help="Remote dir for SCP dumps")
    parser.add_argument("--dst-index", default="new_main", help="Destination index name")
    parser.add_argument("--start-date", default="2024-03-01",
                        help="Start date for pipeline (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None,
                        help="End date for pipeline (YYYY-MM-DD), default: now")
    parser.add_argument("--since", help="Override earliest time (e.g. '2026-03-01T00:00:00')")
    parser.add_argument("--notify", action="store_true", help="Send email on completion/failure")
    args = parser.parse_args()

    try:
        if args.export_scp:
            if not args.dst_hosts:
                parser.error("--export-scp requires --dst-hosts")
            parts = args.start_date.split("-")
            sd = (int(parts[0]), int(parts[1]), int(parts[2]))
            ed = tuple(int(x) for x in args.end_date.split("-")) if args.end_date else None
            total = export_scp_delete(
                src_host=args.src_host, dst_hosts=args.dst_hosts,
                port=args.port, username=args.username, password=args.password,
                src_index=args.index, sourcetype=args.sourcetype,
                start_date=sd, end_date=ed, remote_dir=args.remote_dir)
            if args.notify:
                send_email(
                    f"Export-SCP pipeline completed",
                    f"Exported {total} events from {args.src_host} and SCPed to {args.dst_hosts}.")
            return

        if args.pipeline:
            parts = args.start_date.split("-")
            sd = (int(parts[0]), int(parts[1]), int(parts[2]))
            total = export_inject_delete(
                src_host=args.src_host or args.dst_host,
                dst_host=args.dst_host,
                port=args.port, username=args.username, password=args.password,
                src_index=args.index, dst_index=args.dst_index,
                sourcetype=args.sourcetype, start_date=sd)
            if args.notify:
                send_email(
                    f"Pipeline to {args.dst_host}/{args.dst_index} completed",
                    f"Exported+injected {total} events into {args.dst_index} on {args.dst_host}.")
            return

        if args.import_splunklib:
            if not args.dump_dir or not args.dst_host:
                parser.error("--import-splunklib requires --dump-dir and --dst-host")
            dst = connect(args.dst_host, args.port, args.username, args.password)
            total = index_events_splunklib(
                dst, args.dump_dir, args.dst_host, args.index, args.sourcetype,
                delete_after=args.delete_after)
            if args.notify:
                send_email(
                    f"Splunk sync to {args.dst_host} completed",
                    f"Indexed {total} events into index={args.index} on {args.dst_host}.")
            return

        if args.import_only:
            if not args.dump_dir or not args.dst_host:
                parser.error("--import-only requires --dump-dir and --dst-host")
            dst = connect(args.dst_host, args.port, args.username, args.password)
            total = index_events_with_checkpoint(
                dst, args.dump_dir, args.dst_host, args.index, args.sourcetype)
            if args.notify:
                send_email(
                    f"Splunk sync to {args.dst_host} completed",
                    f"Indexed {total} events into index={args.index} on {args.dst_host}.")
            return

        if not args.src_host or not args.dst_host:
            parser.error("--src-host and --dst-host are required (unless --import-only)")

        dst = connect(args.dst_host, args.port, args.username, args.password)
        src = connect(args.src_host, args.port, args.username, args.password)

        since = args.since if args.since else get_latest_timestamp(dst, args.index, args.sourcetype)
        dump_path = export_events(src, since, args.index, args.sourcetype,
                                  args.dump_dir or "./dump_files")

        if not args.export_only:
            total = index_events_with_checkpoint(
                dst, dump_path, args.dst_host, args.index, args.sourcetype)
            if args.notify:
                send_email(
                    f"Splunk sync to {args.dst_host} completed",
                    f"Indexed {total} events into index={args.index} on {args.dst_host}.")
        else:
            print("Export-only mode — skipping import.")

        print("Done.")

    except Exception as e:
        msg = f"Error: {e}\n\n{traceback.format_exc()}"
        print(msg)
        if args.notify:
            send_email(f"Splunk sync to {args.dst_host} FAILED", msg)
        raise


if __name__ == "__main__":
    main()