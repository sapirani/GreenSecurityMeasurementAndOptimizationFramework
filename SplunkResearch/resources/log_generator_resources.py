wineventlog_log = "07/18/2023 01:32:02 PM LogName=Security EventCode=4634 EventType=0 ComputerName=LB-111-4.auth.ad.bgu.ac.il SourceName=Microsoft Windows security auditing. Type=Information RecordNumber=704156 Keywords=Audit Success TaskCategory=Logoff OpCode=Info Message=An account was logged off. Subject: Security ID:		NT AUTHORITY\SYSTEM Account Name:		LB-111-4$                            Account Domain:		BGU-USERS                            Logon ID:		0x5775CFDA Logon Type:			3   This event is generated when a logon session is destroyed. It may be positively correlated with a logon event using the Logon ID value. Logon IDs are only unique between reboots on the same computer."

sysmon_log = "<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System><Provider Name='Microsoft-Windows-Sysmon' Guid='{5770385F-C22A-43E0-BF4C-06F5698FFBD9}'/><EventID>1</EventID><Version>5</Version><Level>4</Level><Task>1</Task><Opcode>0</Opcode><Keywords>0x8000000000000000</Keywords><TimeCreated SystemTime='2023-07-18T10:32:02.142779900Z'/><EventRecordID>187271</EventRecordID><Correlation/><Execution ProcessID='2524' ThreadID='3976'/><Channel>Microsoft-Windows-Sysmon/Operational</Channel><Computer>dts12-vm2.auth.ad.bgu.ac.il</Computer><Security UserID='S-1-5-18'/></System><EventData><Data Name='RuleName'>-</Data><Data Name='UtcTime'>2023-07-18 10:32:02.135</Data><Data Name='ProcessGuid'>{1601DCC0-6A22-64B6-F999-000000006600}</Data><Data Name='ProcessId'>7960</Data><Data Name='Image'>C:\Program Files\SplunkUniversalForwarder\\bin\splunk-admon.exe</Data><Data Name='FileVersion'>9.0.4</Data><Data Name='Description'>Active Directory monitor</Data><Data Name='Product'>splunk Application</Data><Data Name='Company'>Splunk Inc.</Data><Data Name='OriginalFileName'>splunk-admon.exe</Data><Data Name='CommandLine'>\"C:\Program Files\SplunkUniversalForwarder\\bin\splunk-admon.exe\"</Data><Data Name='CurrentDirectory'>C:\Windows\system32\</Data><Data Name='User'>NT AUTHORITY\SYSTEM</Data><Data Name='LogonGuid'>{1601DCC0-B9E0-6491-E703-000000000000}</Data><Data Name='LogonId'>0x3e7</Data><Data Name='TerminalSessionId'>0</Data><Data Name='IntegrityLevel'>System</Data><Data Name='Hashes'>MD5=B1D294E045EF534E34B0EE5F96D83EBA,SHA256=2440A9E5C3FBFD3F0CE3370178A0277F2F228FC82C46E5B0389AFC22360B9AF5,IMPHASH=0B3917FB5306BF16B3F2D78E6FD4AC61</Data><Data Name='ParentProcessGuid'>{1601DCC0-F6DF-64B3-CA57-000000006600}</Data><Data Name='ParentProcessId'>5692</Data><Data Name='ParentImage'>C:\Program Files\SplunkUniversalForwarder\\bin\splunkd.exe</Data><Data Name='ParentCommandLine'>\"C:\Program Files\SplunkUniversalForwarder\\bin\splunkd.exe\" service</Data><Data Name='ParentUser'>NT AUTHORITY\SYSTEM</Data></EventData></Event>"
# Define the fields that can be replaced


# Define some possible replacement values
replacement_values_wineventlog_security = {
# 'EventCode': [str(num) for num in range(1000, 1005)],
'ComputerName': ['LB-111-4.auth.ad.bgu.ac.il', 'LB-222-8.auth.ad.bgu.ac.il', 'LB-333-2.auth.ad.bgu.ac.il'],
'RecordNumber': [str(num) for num in range(700000, 700005)],
'TaskCategory': ['Logoff', 'Logon', 'System', 'Security']
}
replacement_values_wineventlog_system = {
# 'EventCode': [str(num) for num in range(1000, 1005)],
'ComputerName': ['LB-111-4.auth.ad.bgu.ac.il', 'LB-222-8.auth.ad.bgu.ac.il', 'LB-333-2.auth.ad.bgu.ac.il'],
'RecordNumber': [str(num) for num in range(700000, 700005)],
'TaskCategory': ['Logoff', 'Logon', 'System', 'Security']
}
replacement_values_wineventlog_application = {
# 'EventCode': [str(num) for num in range(1000, 1005)],
'ComputerName': ['LB-111-4.auth.ad.bgu.ac.il', 'LB-222-8.auth.ad.bgu.ac.il', 'LB-333-2.auth.ad.bgu.ac.il'],
'RecordNumber': [str(num) for num in range(700000, 700005)],
'TaskCategory': ['Logoff', 'Logon', 'System', 'Security']
}
# Define some possible replacement values
replacement_values_sysmon = {
# 'EventID': ['1', '2', '3', '4', '5'],
'Version': ['1', '2', '3', '4', '5'],
'Level': ['1', '2', '3', '4', '5'],
'Task': ['1', '2', '3', '4', '5'],
'Opcode': ['1', '2', '3', '4', '5'],
'Keywords': ['4294967296', '1073741824', '8589934592', '2147483648', '17179869184'],
'Computer': ['dts14-vm4.auth.ad.bgu.ac.il', 'dts12-vm2.auth.ad.bgu.ac.il', 'dts13-vm3.auth.ad.bgu.ac.il'],
'Provider': ['Microsoft-Windows-Kernel-Power', 'Microsoft-Windows-Sysmon', 'Microsoft-Windows-Wininit'],
'RuleName': ['rule1', 'rule2', 'rule3', 'rule4', 'rule5'],
'Image': ['C:\\Program Files\\App1\\app1.exe', 'C:\\Program Files\\App2\\app2.exe', 'C:\\Program Files\\App3\\app3.exe'],
'Description': ['description1', 'description2', 'description3', 'description4', 'description5'],
'Product': ['product1', 'product2', 'product3', 'product4', 'product5'],
'Company': ['company1', 'company2', 'company3', 'company4', 'company5'],
'OriginalFileName': ['file1.exe', 'file2.exe', 'file3.exe', 'file4.exe', 'file5.exe']
}

replacement_dicts = {'wineventlog:security': replacement_values_wineventlog_security,
                    'wineventlog:system': replacement_values_wineventlog_system,
                    'wineventlog:application': replacement_values_wineventlog_application,
                    'xmlwineventlog:microsoft-windows-sysmon/operational': replacement_values_sysmon
                    }