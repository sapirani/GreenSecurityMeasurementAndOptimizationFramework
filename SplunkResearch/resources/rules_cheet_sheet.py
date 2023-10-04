rules = [
    {
        "name": "Monitor for Registry Changes",
        "search": 'index=main LogName=Security (EventCode=4657) Object_Name="*\\Run*" | table _time, host, Security_ID, Account_Name, Account_Domain, Operation_Type, Object_Name, Object_Value_Name, Process_Name, New_Value',
        "description": "Adding auditing to known exploited registry keys is a great way to catch malicious\
                activity. Registry keys should not change very often unless something is installed or updated. The goal is to look\
                for NEW items and changes to known high risk items like the Run and RunOnce keys. ",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Suspicious Network IP’s",
        "search": 'index=main LogName=Security EventCode=4663 host=* (Accesses="WriteData (or AddFile)" AND\
                Object_Name="*.*") NOT (Security_ID="NT AUTHORITY\\SYSTEM") NOT (Object_Name="*\\FireFoxProfile\\*" OR\
                Object_Name="*.tmp*" OR Object_Name="*.xml" OR Object_Name="*Thumbs.db" OR\
                Object_Name="\\Device\\HarddiskVolumeShadowCopy*") NOT (Object_Name="*:Zone.Identifier" OR\
                Object_Name="*.part*") | stats count values(Object_Name), values(Accesses) by Security_ID | where count > 1000 ',
        "description": "Setting auditing on a File Server Share will allow large amounts of file changes from a\
                    crypto event to be detected. Look at a large quantity of changes > 1000 in 1 hour to detect the event. Use the\
                    same settings as above as you only need to monitor for NEW files. It is obvious when an event occurs!",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Suspicious Network IP’s",
        "search": 'index=main LogName=Security EventCode=5156 NOT (Source_Address="239.255.255.250" OR\
                Source_Address="224.0.0.*" OR Source_Address="::1" OR Source_Address="ff02::*" OR Source_Address="fe80::*" OR\
                Source_Address="255.255.255.255" OR Source_Address=192.168.1.255) NOT (Destination_Address="127.0.0.1" OR\
                Destination_Address="239.255.255.250" OR Destination_Address="10.*.*.255" OR\
                Destination_Address="224.0.0.25*") NOT (Destination_Port="0") NOT (Application_Name="\\<some process name>\\"\
                OR Application_Name="*\\bin\\splunkd.exe") | dedup Destination_Address Destination_Port | table _time, host,\
                Application_Name, Direction, Source_Address, Source_Port, Destination_Address, Destination_Port | sort Direction Destination_Port',
        "description": 'This does require the use of the Windows Firewall. In networks where this is\
                    normally not used, you can use Group Policy to set the Windows Firewall to an Any/Any configuration so no\
                    blocking occurs, yet the traffic is captured in the logs and more importantly what process made the connection.\
                    You can create exclusions by IP addresses (such as broadcast IP’s) and by process names to reduce the output and\
                    make it more actionable. The "Lookup" command will benefit this query tremendously by excluding items.',
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Suspicious/Administrative Processes",
        "search": 'index=main LogName=Security EventCode=4688 NOT (Account_Name=*$) (arp.exe OR at.exe OR bcdedit.exe OR bcp.exe OR\
                chcp.exe OR cmd.exe OR cscript.exe OR csvde OR dsquery.exe OR ipconfig.exe OR mimikatz.exe OR nbtstat.exe OR nc.exe OR\
                netcat.exe OR netstat.exe OR nmap OR nslookup.exe OR netsh OR OSQL.exe OR ping.exe OR powershell.exe OR powercat.ps1 OR\
                psexec.exe OR psexecsvc.exe OR psLoggedOn.exe OR procdump.exe OR qprocess.exe OR query.exe OR rar.exe OR reg.exe OR\
                route.exe OR runas.exe OR rundll32 OR schtasks.exe OR sethc.exe OR sqlcmd.exe OR sc.exe OR ssh.exe OR sysprep.exe OR\
                systeminfo.exe OR system32\\net.exe OR reg.exe OR tasklist.exe OR tracert.exe OR vssadmin.exe OR whoami.exe OR winrar.exe\
                OR wscript.exe OR "winrm.*" OR "winrs.*" OR wmic.exe OR wsmprovhost.exe OR wusa.exe) | eval Message=split(Message,".") |\
                eval Short_Message=mvindex(Message,0) | table _time, host, Account_Name, Process_Name, Process_ID,\
                Process_Command_Line, New_Process_Name, New_Process_ID, Creator_Process_ID, Short_Message',
        "description": "This list is based on built-in Windows administrative utilities and\
                    known hacking utilities that are often seen used in exploitation. Expand this list as needed to add utilities used in\
                    hacking attacks. You do not need to alert on all processes launching, just suspicious ones or ones known to be used\
                    in hacking attacks. Some administrative tools are very noisy and normally used or automatically executed regularly\
                    and should NOT be included to make your alert more actionable and accurate that something suspicious has\
                    occurred.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Whitelisting bypass attempts",
        "search": 'index=main LogName=Security (EventCode=4688) NOT (Account_Name="Something_good") (iexec.exe OR InstallUtil.exe OR\
                Regsrv32.exe OR Regasm.exe OR Regsvcs.exe OR MSBuild.exe) | eval Message=split(Message,".") | eval\
                Short_Message=mvindex(Message,0) | table _time, host, Account_Name, Process_Name, Process_ID, Process_Command_Line,\
                New_Process_Name, New_Process_ID, Creator_Process_ID, Short_Message',
        "description": "Hackers will often use PowerShell to exploit a system due to the\
                capability of PowerShell to avoid using built-in utilities and dropping additional malware files on disk. Watching for\
                policy and profile bypasses will allow you to detect this hacking activity.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": " Monitor for File Shares being accessed:",
        "search": 'index=main LogName=Security EventCode=5140 (Share_Name="*\\C$" OR Share_Name="*D$" OR\
                Share_Name="*E$" OR Share_Name="*F$" OR Share_Name="*U$") NOT Source_Address="::1" | eval\
                Destination_Sys1=trim(host,"1") | eval Destination_Sys2=trim(host,"2") | eval Dest_Sys1=lower(Destination_Sys1) |\
                eval Dest_Sys2=lower(Destination_Sys2) | rename host AS Destination | rename Account_Domain AS Domain | where\
                Account_Name!=Dest_Sys1 | where Account_Name!=Dest_Sys2 | stats count values(Domain) AS Domain,\
                values(Source_Address) AS Source_IP, values(Destination) AS Destination, dc(Destination) AS Dest_Count,\
                values(Share_Name) AS Share_Name, values(Share_Path) AS Share_Path by Account_Name',
        "description": "Once a system is compromised, hackers will connect or jump to other\
                systems to infect and/or to steal data. Watch for accounts crawling across file shares. Some management\
                accounts will do this normally so exclude these to the systems they normally connect. Other activity from\
                management accounts such as new processes launching will alert you to malicious behavior when excluded in this alert.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for File Shares",
        "search": 'index=main LogName=Security EventCode=5140 (Share_Name=\"*\\\\C$\" OR Share_Name=\"*D$\" OR Share_Name=\"*E$\" OR Share_Name=\"*F$\" OR Share_Name=\"*U$\") NOT Source_Address=\"::1\" | eval Destination_Sys1=trim(host,\"1\") | eval Destination_Sys2=trim(host,\"2\") | eval Dest_Sys1=lower(Destination_Sys1) | eval Dest_Sys2=lower(Destination_Sys2) | rename host AS Destination | rename Account_Domain AS Domain | where Account_Name!=Dest_Sys1 | where Account_Name!=Dest_Sys2 | stats count values(Domain) AS Domain, values(Source_Address) AS Source_IP, values(Destination) AS Destination, dc(Destination) AS Dest_Count, values(Share_Name) AS Share_Name, values(Share_Path) AS Share_Path by Account_Name',
        "description": "Once a system is compromised, hackers will connect or jump to other systems to infect and/or to steal data. Watch for accounts crawling across file shares.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for New Service Installs",
        "search": 'index=main LogName=System EventCode=7045 NOT (Service_Name=mgmt_service) | eval Message=split(Message,\".\") | eval Short_Message=mvindex(Message,0) | table _time host Service_Name, Service_Type, Service_Start_Type, Service_Account, Short_Message',
        "description": "Monitoring for a new service install is crucial. Hackers often use a new service to gain persistence for their malware when a system restarts.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Service State Changes",
        "search": 'index=main LogName=System EventCode=7040 NOT Message=\"*Windows Modules Installer service*\" OR Message=\"*Background Intelligent Transfer Service service*\") | table _time, host, User, Message',
        "description": "Monitoring for a service state changes can show when a service is altered. Hackers often use an existing service to avoid new service detection and modify the ServiceDll to point to a malicious payload gaining persistence for their malware when a system restarts.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },

    {
        "name": "Monitor for PowerShell bypass attempts",
        "search": "index=main LogName=Security EventCode=4688 (powershell* AND (–ExecutionPolicy OR –Exp)) OR (powershell* AND bypass) OR (powershell* AND (-noprofile OR -nop)) | eval Message=split(Message,\".\") | eval Short_Message=mvindex(Message,0) | table _time, host, Account_Name, Process_Name, Process_ID, Process_Command_Line, New_Process_Name, New_Process_ID, Creator_Process_ID, Short_Message",
        "description": "Hackers will often use PowerShell to exploit a system due to the capability of PowerShell to avoid using built-in utilities and dropping additional malware files on disk. Watching for policy and profile bypasses will allow you to detect this hacking activity",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": "Medium"
    },
    {
        "name": "Monitor for all processes excluding trusted/known processes",
        "search": "index=main LogName=Security EventCode=4688 NOT (Account_Name=*$) NOT [ inputlookup Trusted_processes.csv | fields Process_Name ] | eval Message=split(Message,\".\") | eval Short_Message=mvindex(Message,0) | table _time, host, Account_Name, Process_Name, Process_ID, Process_Command_Line, New_Process_Name, New_Process_ID, Creator_Process_ID, Short_Message",
        "description": "You can create reports for any or all processes starting (4688) and filter out the known good ones to create a more actionable report and alert.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": "Medium"
    },
    {
        "name": "Monitor for Logon Success",
        "search": 'index=main LogName=Security EventCode=4624 NOT (host="DC1" OR host="DC2" OR host="DC…") NOT (Account_Name="*$" OR Account_Name="ANONYMOUS LOGON") NOT (Account_Name="Service_Account") | eval Account_Domain=(mvindex(Account_Domain,1)) | eval Account_Name=if(Account_Name="-\n",(mvindex(Account_Name,1)), Account_Name) | eval Account_Name=if(Account_Name="*$",(mvindex(Account_Name,1)), Account_Name) | eval Time=strftime(_time,\'%Y/%m/%d %T\') | stats count values(Account_Domain) AS Domain, values(host) AS Host, dc(host) AS Host_Count, values(Logon_Type) AS Logon_Type, values(Workstation_Name) AS WS_Name, values(Source_Network_Address) AS Source_IP, values(Process_Name) AS Process_Name by Account_Name | where Host_Count > 2',
        "description": "Logging for failed logons seems obvious, but when a user credential gets compromised and their credentials used for exploitation, successful logins will be a major indicator of malicious activity and system crawling.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Logon Failures",
        "search": "index=main LogName=Security EventCode=4625 | table _time, Workstation_Name, Source_Network_Address, host, Account_Name",
        "description": "Watch for excessive logon failures, especially Internet facing systems and systems that contain confidential data. This will also detect brute force attempts and users who have failed to changed their passwords on additional devices such as smartphones.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Administrative and Guest Logon Failures",
        "search": "index=main LogName=Security EventCode=4625 (Account_Name=administrator OR Account_Name=guest) | stats count values(Workstation_Name) AS Workstation_Name, Values(Source_Network_Address) AS Source_IP_Address, values(host) AS Host by Account_Name | where count > 5",
        "description": "Hackers and malware often try to brute force known accounts, such as Administrator and Guest. This alert will monitor and alert if configured for attempts > 5.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Additions to Firewall Rules",
        "search": "index=main LogName=Security EventCode=2004 | table _time, host, Rule_Name, Origin, Active, Direction, Profiles, Action, Application_Path, Service_Name, Protocol, Security_Options, Edge_Traversal, Modifying_User, Modifying_Application, Rule_ID",
        "description": "Malware and hackers will often add a firewall rule to allow access to some Windows service or application.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    },
    {
        "name": "Monitor for Changes to Firewall Rules",
        "search": "index=main LogName=Security EventCode=2005 | table _time, host, Rule_Name, Origin, Active, Direction, Profiles, Action, Application_Path, Service_Name, Protocol, Security_Options, Edge_Traversal, Modifying_User, Modifying_Application, Rule_ID",
        "description": "Malware and hackers will often modify a firewall rule to allow access to some Windows service or application.",
        "cron_schedule": "*/15 * * * *",
        "dispatch.earliest_time": "-15m@m",
        "dispatch.latest_time": "now",
        "alert.expires": "90m",
        "severity": 4
    }


]
