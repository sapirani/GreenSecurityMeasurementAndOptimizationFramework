from gymnasium.envs.registration import register
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
from src.splunk_tools import SplunkTools
from src.log_generator import LogGenerator
from resources.section_logtypes import section_logtypes

# fake_start_datetime = "04/30/2023:08:00:00"
# savedsearches = ["Detect New Local Admin account", "ESCU Network Share Discovery Via Dir Command Rule", "Known Services Killed by Ransomware", "Non Chrome Process Accessing Chrome Default Dir"]
# relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
# relevant_logtypes.append(('wineventlog:security', '4624'))
# splunk_tools_instance = SplunkTools(savedsearches)
# log_generator_instance = LogGenerator(relevant_logtypes, splunk_tools_instance)
# additional_percentage = 0.1
# register(id='splunk-v0',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300, "additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 0.3
# register(id='splunk-v1',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300, "additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# savedsearches = ["Windows Event For Service Disabled","Detect New Local Admin account", "ESCU Network Share Discovery Via Dir Command Rule", "Known Services Killed by Ransomware", "Non Chrome Process Accessing Chrome Default Dir"]
# relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
# relevant_logtypes.append(('wineventlog:security', '4624'))
# splunk_tools_instance = SplunkTools(savedsearches)
# log_generator_instance = LogGenerator(relevant_logtypes, splunk_tools_instance)
# additional_percentage = 0.1
# register(id='splunk-v2',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 0.3
# register(id='splunk-v3',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 0.1
# register(id='splunk-v4',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':60, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 0.3
# register(id='splunk-v5',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':60, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})

# #######################################
# additional_percentage = 1
# register(id='splunk-v6',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':60, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# savedsearches = ["Windows Event For Service Disabled",
#                  "Detect New Local Admin account",
#                  "ESCU Network Share Discovery Via Dir Command Rule",
#                  "Known Services Killed by Ransomware",
#                  "Non Chrome Process Accessing Chrome Default Dir",
#                  "Kerberoasting spn request with RC4 encryption",
#                  "Clop Ransomware Known Service Name"]
# relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
# relevant_logtypes.append(('wineventlog:security', '4624'))
# splunk_tools_instance = SplunkTools(savedsearches)
# log_generator_instance = LogGenerator(relevant_logtypes, splunk_tools_instance)
# additional_percentage = 0.1
# register(id='splunk-v7',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':30, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 0.5
# register(id='splunk-v8',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':30, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 0.2
# fake_start_datetime = "04/26/2024:13:00:00"
# register(id='splunk-v9',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':30, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 0.3
# fake_start_datetime = "04/26/2024:13:00:00"
# register(id='splunk-v10',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':30, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 0.4
# fake_start_datetime = "04/26/2024:13:00:00"
# register(id='splunk-v11',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':30, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime})
# #######################################
# additional_percentage = 1
# fake_start_datetime = "04/26/2024:13:00:00"
# register(id='splunk-v12',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':30, 'span_size':60,
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime,
#                  "env_id": "splunk-v12"})
# #######################################
# additional_percentage = 0.1
# fake_start_datetime = "04/26/2024:13:00:00"
# register(id='splunk-v13',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':120, 'span_size':120, # search_window in minutes and span_size in seconds
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime,
#                  "env_id": "splunk-v13"})
# #######################################
# additional_percentage = 0.1
# fake_start_datetime = "05/03/2024:13:00:00"
# register(id='splunk-v14',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':120, 'span_size':120, # search_window in minutes and span_size in seconds
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime,
#                  "env_id": "splunk-v14"})
# #######################################
# additional_percentage = 0.5
# fake_start_datetime = "05/03/2024:13:00:00"
# register(id='splunk-v15',
#          entry_point='custom_splunk.envs:SplunkEnv',
#          kwargs={'rule_frequency':1, 'search_window':120, 'span_size':120, # search_window in minutes and span_size in seconds
#                  'splunk_tools_instance':splunk_tools_instance,
#                  "log_generator_instance": log_generator_instance,
#                  "relevant_logtypes": relevant_logtypes,
#                  "num_of_searches": len(savedsearches),
#                  "logs_per_minute": 300,"additional_percentage":additional_percentage,
#                  "fake_start_datetime": fake_start_datetime,
#                  "env_id": "splunk-v15"})
#######################################
#######################################
fake_start_datetime = "05/03/2024:13:00:00"
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
register(id='splunk_train-v0',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_train-v0'
        })
#######################################
#######################################
fake_start_datetime = "06/03/2024:13:00:00"
register(id='splunk_eval-v0',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_eval-v0'
        })
#######################################
#######################################
savedsearches = [
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
fake_start_datetime = "05/03/2024:13:00:00"
env_id = "splunk_train-v1"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
fake_start_datetime = "06/03/2024:13:00:00"
env_id = "splunk_eval-v1"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
fake_start_datetime = "09/03/2023:13:00:00"
env_id = "splunk_train-v2"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
fake_start_datetime = "09/03/2023:13:00:00"
env_id = "splunk_eval-v2"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
savedsearches = [
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
fake_start_datetime = "09/03/2023:13:00:00"
env_id = "splunk_train-v3"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
fake_start_datetime = "09/03/2023:13:00:00"
env_id = "splunk_eval-v3"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
savedsearches = [
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
fake_start_datetime = "12/03/2023:13:00:00"
env_id = "splunk_train-v30"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
savedsearches = [
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
fake_start_datetime = "04/01/2024:00:00:00"
env_id = "splunk_train-v31"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name",
                 'Windows AD Replication Request Initiated from Unsanctioned Location',
                 'ESCU Windows Rapid Authentication On Multiple Hosts Rule',
                #  'WinEvent Scheduled Task Created Within Public Path',
                #  'Monitor for Administrative and Guest Logon Failures'
                 ]
fake_start_datetime = "12/01/2024:00:00:00"#"07/01/2024:00:00:00"
env_id = "splunk_train-v32"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
        })
#######################################
#######################################

fake_start_datetime = "08/01/2025:00:00:00"
env_id = "splunk_eval-v32"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
        })

#######################################
#######################################
fake_start_datetime = "05/03/2024:13:00:00"
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
register(id='splunk_train-v4',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_train-v4'
        })
#######################################
#######################################
fake_start_datetime = "06/03/2024:13:00:00"
register(id='splunk_eval-v4',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_eval-v4'
        })
#######################################
#######################################
fake_start_datetime = "05/03/2024:13:00:00"
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir"
                 ]
register(id='splunk_train-v5',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_train-v5'
        })
#######################################
#######################################
fake_start_datetime = "06/03/2024:13:00:00"
register(id='splunk_eval-v5',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_eval-v5'
        })
#######################################
#######################################
fake_start_datetime = "09/03/2024:13:00:00"
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir"
                 ]
register(id='splunk_train-v6',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_train-v6'
        })
#######################################
#######################################
fake_start_datetime = "09/03/2024:13:00:00"
register(id='splunk_eval-v6',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_eval-v6'
        })
#######################################
#######################################
fake_start_datetime = "09/03/2024:13:00:00"
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
        ]
register(id='splunk_train-v7',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_train-v7'
        })
#######################################
#######################################
fake_start_datetime = "09/03/2024:13:00:00"
register(id='splunk_eval-v7',
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':'splunk_eval-v7'
        })
#######################################
#######################################
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
fake_start_datetime = "09/03/2023:13:00:00"
env_id = "splunk_train-v8"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })
#######################################
#######################################
fake_start_datetime = "09/03/2023:13:00:00"
env_id = "splunk_eval-v8"
register(id=env_id,
        entry_point='custom_splunk.envs:SplunkEnv', 
        kwargs={
                'savedsearches':savedsearches,
                'fake_start_datetime':fake_start_datetime,
                'env_id':env_id
        })