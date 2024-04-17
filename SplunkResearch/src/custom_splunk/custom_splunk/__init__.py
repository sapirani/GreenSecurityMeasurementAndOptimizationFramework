from gym.envs.registration import register
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
from src.splunk_tools import SplunkTools
from src.log_generator import LogGenerator
from resources.section_logtypes import section_logtypes

fake_start_datetime = "04/30/2023:08:00:00"
savedsearches = ["Detect New Local Admin account", "ESCU Network Share Discovery Via Dir Command Rule", "Known Services Killed by Ransomware", "Non Chrome Process Accessing Chrome Default Dir"]
relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
relevant_logtypes.append(('wineventlog:security', '4624'))
splunk_tools_instance = SplunkTools(savedsearches)
log_generator_instance = LogGenerator(relevant_logtypes, splunk_tools_instance)
additional_percentage = 0.1
register(id='splunk-v0',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300, "additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})
#######################################
additional_percentage = 0.3
register(id='splunk-v1',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300, "additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})
#######################################
savedsearches = ["Windows Event For Service Disabled","Detect New Local Admin account", "ESCU Network Share Discovery Via Dir Command Rule", "Known Services Killed by Ransomware", "Non Chrome Process Accessing Chrome Default Dir"]
relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
relevant_logtypes.append(('wineventlog:security', '4624'))
splunk_tools_instance = SplunkTools(savedsearches)
log_generator_instance = LogGenerator(relevant_logtypes, splunk_tools_instance)
additional_percentage = 0.1
register(id='splunk-v2',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300,"additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})
#######################################
additional_percentage = 0.3
register(id='splunk-v3',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300,"additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})
#######################################
additional_percentage = 0.1
register(id='splunk-v4',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':60, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300,"additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})
#######################################
additional_percentage = 0.3
register(id='splunk-v5',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':60, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300,"additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})

#######################################
additional_percentage = 1
register(id='splunk-v6',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':60, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300,"additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})
#######################################
savedsearches = ["Windows Event For Service Disabled",
                 "Detect New Local Admin account",
                 "ESCU Network Share Discovery Via Dir Command Rule",
                 "Known Services Killed by Ransomware",
                 "Non Chrome Process Accessing Chrome Default Dir",
                 "Kerberoasting spn request with RC4 encryption",
                 "Clop Ransomware Known Service Name"]
relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
relevant_logtypes.append(('wineventlog:security', '4624'))
splunk_tools_instance = SplunkTools(savedsearches)
log_generator_instance = LogGenerator(relevant_logtypes, splunk_tools_instance)
additional_percentage = 0.1
register(id='splunk-v7',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':30, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300,"additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})
#######################################
additional_percentage = 0.5
register(id='splunk-v8',
         entry_point='custom_splunk.envs:SplunkEnv',
         kwargs={'rule_frequency':1, 'search_window':30, 'span_size':60,
                 'splunk_tools_instance':splunk_tools_instance,
                 "log_generator_instance": log_generator_instance,
                 "relevant_logtypes": relevant_logtypes,
                 "num_of_searches": len(savedsearches),
                 "logs_per_minute": 300,"additional_percentage":additional_percentage,
                 "fake_start_datetime": fake_start_datetime})