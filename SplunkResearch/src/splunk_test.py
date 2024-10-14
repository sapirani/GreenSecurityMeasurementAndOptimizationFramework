import os
import splunklib.client as client
import splunklib.results as results
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import psutil

def monitor_search_resources(host, port, username, password, search_query, duration_minutes):
    # Connect to Splunk
    service = client.connect(
        host=host,
        port=port,
        username=username,
        password=password
    )

    # Start the search job
    job = service.jobs.create(search_query)

    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)

    while datetime.now() < end_time:
        # Get the latest statistics
        job.refresh()
        stats = job.content
        
        pid = int(stats.get('pid', 0))
        print('pid: ', pid)
        cpu_percent = 0
        cpu_num = 0
        cpu_times = 0
        memory_info = 0
        io_counters = 0

        if pid != 0:
            try:
                process = psutil.Process(pid)
                cpu_num = psutil.cpu_count()
                
                with process.oneshot():
                    while process.is_running():
                        process = psutil.Process(pid)
                        
                        job.refresh()
                        stats = job.content
                        # Extract relevant resource metrics
                        scan_count = stats.get('scanCount', 0)
                        event_count = stats.get('eventCount', 0)
                        result_count = stats.get('resultCount', 0)
                        disk_usage = stats.get('diskUsage', 0)
                        run_duration = stats.get('runDuration', 0)
                        
                        cpu_percent = process.cpu_percent(interval=.1)/cpu_num
                        cpu_times = process.cpu_times()
                        memory_info = process.memory_info()
                        io_counters = process.io_counters()
                        
                        # print("cpu test: ", psutil.Process(1824).cpu_percent(interval=.1))
                        
                        # Print or store the metrics as needed
                        # print(chr(27) + "[2J")
                        
                        print(f"Timestamp: {datetime.now()}")
                        print(f"Scan Count: {scan_count}")
                        print(f"Event Count: {event_count}")
                        print(f"Result Count: {result_count}")
                        print(f"Disk Usage: {disk_usage} bytes")
                        print(f"Run Duration: {run_duration} seconds")
                        print(f"CPU Usage: {cpu_percent} %")
                        print(f"CPU Num: {cpu_num}")
                        print(f"CPU Times: {cpu_times}")
                        print(f"Memory Info: {memory_info}")
                        print(f"IO Counters: {io_counters}")
                        # print(f"Network In: {network_bytes_in} bytes")
                        # print(f"Network Out: {network_bytes_out} bytes")
                        print("---")
                        
            except psutil.NoSuchProcess:
                return None
            except psutil.AccessDenied:
                print(f"Access denied to process with PID {pid}.")
                return None

        
        
    # Cancel the job when done
    job.cancel()

# Example usage
if __name__ == "__main__":
    load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/.env')
    monitor_search_resources(
        host=os.getenv("SPLUNK_HOST"),
        port=os.getenv("SPLUNK_PORT"),
        username=os.getenv("SPLUNK_USERNAME"),
        password=os.getenv("SPLUNK_PASSWORD"),
        # search_query='search `wineventlog_security` EventCode=5140 ShareName IN("\\\\*\\ADMIN$","\\\\*\\C$","*\\\\*\\IPC$") AccessMask= 0x1  earliest=1717437600.0 latest=1717452000.0| stats min(_time) as firstTime max(_time) as lastTime count by ShareName SourceAddress ObjectType AccountName AccountDomain SourcePort AccessMask Computer | `security_content_ctime(firstTime)` | `security_content_ctime(lastTime)` | `network_share_discovery_via_dir_command_filter`',
        search_query='search `wineventlog_system` EventCode=7045 Service_Name IN ("SecurityCenterIBM", "WinCheckDRVs")  earliest=1714723200.0 latest=1714730400.0| stats count min(_time) as firstTime max(_time) as lastTime by dest EventCode Service_File_Name Service_Name Service_Start_Type Service_Type | `security_content_ctime(firstTime)` | `security_content_ctime(lastTime)` | `clop_ransomware_known_service_name_filter`',
        duration_minutes=3
    )