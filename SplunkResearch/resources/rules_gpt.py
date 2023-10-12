rules = [
    {
        "name": "Failed Logins Followed by Successful Login",
        "search": 'index=main source="WinEventLog:security" (EventCode=4625 OR EventCode=4624) | stats count by EventCode,Account_Name | where count > 3',
        "description": "Alerts when there are multiple failed logins for a user, followed by a successful login, which could be a sign of a successful brute force attack.",
        "cron_schedule": "*/15 * * * *",
        "severity": 4
    },
    {
        "name": "High Number of Security Log Deletions",
        "search": 'index=main source="WinEventLog:security" EventCode=1102',
        "description": "Alerts when the number of security log deletions is unusually high, which could be an attempt to hide activity.",
        "cron_schedule": "*/15 * * * *",
        "severity": 3
    },
    # Continue from the previous rules list
    {
        "name": "New User Created",
        "search": 'index=main source="WinEventLog:security" EventCode=4720',
        "description": "Alerts when a new user is created.",
        "cron_schedule": "*/15 * * * *",
        "severity": 2
    },
    {
        "name": "DLL Load Failed",
        "search": 'index=main source="WinEventLog:Application" EventCode=1000 | where Status_Code=0xc0000135',
        "description": "Alerts when a DLL load fails, which could be a sign of DLL sideloading.",
        "cron_schedule": "*/15 * * * *",
        "severity": 4
    },
    {
        "name": "Non-Interactive PowerShell Session",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=1 | where CommandLine="-NonI -W Hidden -NoP -Exec Bypass"',
        "description": "Alerts when a non-interactive PowerShell session is started, which could be a sign of a script running.",
        "cron_schedule": "*/5 * * * *",
        "severity": 3
    },
    {
        "name": "Multiple Network Connections to Same Port on External Hosts",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=3 | stats count by Destination_Port,Destination_Ip | where count > 5',
        "description": "Alerts when there are multiple network connections to the same port on external hosts, which could be a sign of data exfiltration or command and control.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    {
        "name": "Potential Data Compressed for Exfiltration",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=11 | where Target_File_Name LIKE "%.zip" OR Target_File_Name LIKE "%.rar" OR Target_File_Name LIKE "%.tar" OR Target_File_Name LIKE "%.gz" | stats sum(Write_Bytes) as total_bytes by User,Target_File_Name | where total_bytes > 10000000',
        "description": "Alerts when there is a large amount of data written to a compressed file, which could be a sign of data being prepared for exfiltration.",
        "cron_schedule": "*/15 * * * *",
        "severity": 4
    },
    {
        "name": "Potential DCSync Attack",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=3 | where Destination_Port=389 OR Destination_Port=636 OR Destination_Port=3268 OR Destination_Port=3269',
        "description": "Alerts when there is a network connection to the Active Directory Domain Services replication service, which could be a sign of a DCSync attack.",
        "cron_schedule": "*/15 * * * *",
        "severity": 5
    },
    {
        "name": "Suspicious Process Access",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=10 | where CallTrace LIKE "%UNKNOWN%" OR CallTrace LIKE "%wow64%" OR CallTrace LIKE "%unknown%"',
        "description": "Alerts when a process accesses another process in a suspicious way, which could be a sign of process injection or credential dumping.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    # ... Continue with the remaining rules using this same format
    # Continue from the previous rules list
    {
        "name": "Disabled Security Tool",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=16',
        "description": "Alerts when a security tool is disabled, which could be a sign of an attempt to evade detection.",
        "cron_schedule": "*/5 * * * *",
        "severity": 5
    },
    {
        "name": "Process Opened a Network Connection",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=3',
        "description": "Alerts when a process opens a network connection, which could be a sign of malware communication or data exfiltration.",
        "cron_schedule": "*/5 * * * *",
        "severity": 3
    },
    {
        "name": "Suspicious File Creation",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=11 | where Target_File_Name LIKE "%.exe" OR Target_File_Name LIKE "%.dll"',
        "description": "Alerts when a suspicious file (like .exe or .dll) is created, which could be a sign of malware delivery or execution.",
        "cron_schedule": "*/15 * * * *",
        "severity": 4
    },
    {
        "name": "Detected Named Pipe Creation",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=17',
        "description": "Alerts when a named pipe is created, which could be a sign of inter-process communication commonly used by malware.",
        "cron_schedule": "*/5 * * * *",
        "severity": 2
    },
    {
        "name": "Potential Pass-the-Hash Attack",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=10 | where CallTrace LIKE "%samlib.dll%" OR CallTrace LIKE "%vaultcli.dll%"',
        "description": "Alerts when a process accesses the SAM or Vault DLLs, which could be a sign of a Pass-the-Hash attack.",
        "cron_schedule": "*/5 * * * *",
        "severity": 5
    },
    {
        "name": "Potential Process Injection",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=8 | where CallTrace LIKE "%UNKNOWN%" OR CallTrace LIKE "%wow64%" OR CallTrace LIKE "%unknown%"',
        "description": "Alerts when a process exhibits behavior that could be indicative of Process Injection.",
        "cron_schedule": "*/5 * * * *",
        "severity": 5
    },
    {
        "name": "Suspicious PowerShell Command Line",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=1 | where CommandLine LIKE "%EncodedCommand%" OR CommandLine LIKE "%ep bypass%" OR CommandLine LIKE "%-NoP%"',
        "description": "Alerts when a suspicious PowerShell command line is detected, which could be a sign of a script running.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    {
        "name": "Modification of Executable File",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=11 | where Target_File_Name LIKE "%.exe" OR Target_File_Name LIKE "%.dll"',
        "description": "Alerts when an executable file is modified, which could be a sign of malware delivery or execution.",
        "cron_schedule": "*/15 * * * *",
        "severity": 4
    },
    {
        "name": "Detected Registry Modification",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=13 OR EventCode=14',
        "description": "Alerts when a registry modification is detected, which could be a sign of system or security software manipulation.",
        "cron_schedule": "*/5 * * * *",
        "severity": 3
    },
    {
        "name": "Multiple Failed Logins from the Same Source",
        "search": 'index=main source="WinEventLog:security" EventCode=4625 | stats count by Account_Name,Workstation_Name | where count > 3',
        "description": "Alerts when there are multiple failed logins from the same source, which could be a sign of a brute force attack.",
        "cron_schedule": "*/15 * * * *",
        "severity": 4
    },
    {
        "name": "System or Application Error",
        "search": 'index=main source="WinEventLog:System" OR source="WinEventLog:Application" EventCode=6008 OR EventCode=1074 OR EventCode=1000 OR EventCode=1002',
        "description": "Alerts when a system or application error occurs, which could be a sign of system instability or potential malware activity.",
        "cron_schedule": "*/15 * * * *",
        "severity": 2
    },
    {
        "name": "Malware Often Changes File Creation Time",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=2 | where Previous_Creation_Time!=Creation_Time',
        "description": "Detects file creation time changes, often associated with malware attempting to hide itself.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    {
        "name": "Suspicious Remote Thread Creation",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=8',
        "description": "Detects remote thread creation, often associated with process injection techniques.",
        "cron_schedule": "*/5 * * * *",
        "severity": 5
    },
    {
        "name": "Detect Network Connections to Non-Standard Ports",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=3 | where Destination_Port!=80 AND Destination_Port!=443',
        "description": "Detects network connections to non-standard ports, which could be a sign of malware communication or data exfiltration.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    {
        "name": "Detect Network Connections from Non-Browser or Non-Email Client",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=3 | where Image!="C:\\Program Files (x86)\\Internet Explorer\\iexplore.exe" AND Image!="C:\\Program Files\\Internet Explorer\\iexplore.exe" AND Image!="C:\\Program Files\\Mozilla Firefox\\firefox.exe" AND Image!="C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe" AND Image!="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe" AND Image!="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" AND Image!="C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe" AND Image!="C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe" AND Image!="C:\\Program Files\\Microsoft Office\\root\\Office16\\OUTLOOK.EXE" AND Image!="C:\\Program Files (x86)\\Microsoft Office\\root\\Office16\\OUTLOOK.EXE"',
        "description": "Detects network connections initiated by non-browser or non-email client applications, which could indicate Command and Control (C2) activities or other suspicious behaviors.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    {
        "name": "Detect Use of Tools like PsExec",
        "search": 'index=main source="XmlWinEventLog:Microsoft-Windows-Sysmon/Operational" EventCode=1 | where Image LIKE "%\\services.exe" AND Parent_Image LIKE "%\\PsExec.exe"',
        "description": "Detects the use of tools like PsExec, which are often used in lateral movement.",
        "cron_schedule": "*/5 * * * *",
        "severity": 5
    },
    {
        "name": "Clearing of Windows Logs",
        "search": 'index=main source="WinEventLog:Security" EventCode=1102',
        "description": "Detects the clearing of Windows logs, which could indicate malicious activity aiming to hide tracks.",
        "cron_schedule": "*/5 * * * *",
        "severity": 5
    },
    {
        "name": "Successful Login After Multiple Failures",
        "search": 'index=main source="WinEventLog:Security" (EventCode=4625 OR EventCode=4624)',
        "description": "Detects successful login after multiple failures, which might indicate a successful brute force attack.",
        "cron_schedule": "*/5 * * * *",
        "severity": 3
    },
    {
        "name": "Creation of a New Local User Account",
        "search": 'index=main source="WinEventLog:Security" EventCode=4720',
        "description": "Detects the creation of a new local user account, which could indicate malicious activity.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    {
        "name": "An Account Was Logged off",
        "search": 'index=main source="WinEventLog:Security" EventCode=4634',
        "description": "Detects when an account was logged off, which can be used for troubleshooting and detecting potentially malicious activity.",
        "cron_schedule": "*/5 * * * *",
        "severity": 2
    },
    {
        "name": "Account Locked Out",
        "search": 'index=main source="WinEventLog:Security" EventCode=4740',
        "description": "Detects when an account is locked out, which could indicate a brute force attempt.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    {
        "name": "Change in System Time",
        "search": 'index=main source="WinEventLog:Security" EventCode=520',
        "description": "Detects changes in system time, which could indicate evasion tactics by malware or other malicious activities.",
        "cron_schedule": "*/5 * * * *",
        "severity": 3
    },
    {
        "name": "Change in User Account Control",
        "search": 'index=main source="WinEventLog:Security" EventCode=4675',
        "description": "Detects changes in User Account Control (UAC), which could indicate privilege escalation or other suspicious activity.",
        "cron_schedule": "*/5 * * * *",
        "severity": 3
    },
    {
        "name": "New Kernel Filter Driver",
        "search": 'index=main source="WinEventLog:Security" EventCode=6',
        "description": "Detects a new kernel filter driver, which could indicate potential rootkit behavior.",
        "cron_schedule": "*/5 * * * *",
        "severity": 5
    },
    {
        "name": "Detected a New Service Installation",
        "search": 'index=main source="WinEventLog:Security" EventCode=601',
        "description": "Detects a new service installation, which could be related to a persistence mechanism.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    },
    {
        "name": "User Added to Privileged Group",
        "search": 'index=main source="WinEventLog:Security" EventCode=4728 OR EventCode=4732 OR EventCode=4756',
        "description": "Detects when a user is added to a privileged group (Administrators, Power Users, etc.), which could indicate privilege escalation.",
        "cron_schedule": "*/5 * * * *",
        "severity": 4
    }
    # End of the rules list
]

