from SplunkResearch.src.splunk_tools import SplunkTools
import subprocess
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name",
                 'Windows AD Replication Request Initiated from Unsanctioned Location',
                 'ESCU Windows Rapid Authentication On Multiple Hosts Rule']
splunk_tools = SplunkTools(savedsearches)

for savedsearch in savedsearches:
    # enable the saved search
    print(f"Enabling {savedsearch}")
    splunk_tools.enable_search(savedsearch)
    # update cron schedule to run every 5 minutes
    splunk_tools.update_search_cron_expression(savedsearch, "*/5 * * * *")

# run scan
subprocess.run(["python3", "scanner.py"])

for savedsearch in savedsearches:
    # enable the saved search
        print(f"Disabling {savedsearch}")
        splunk_tools.disable_search(savedsearch)
