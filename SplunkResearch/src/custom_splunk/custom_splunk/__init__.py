from gym.envs.registration import register
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')
from src.splunk_tools import SplunkTools
from src.log_generator import LogGenerator
from resources.section_logtypes import section_logtypes

savedsearches = ["Detect New Local Admin account", "ESCU Network Share Discovery Via Dir Command Rule", "Known Services Killed by Ransomware", "Non Chrome Process Accessing Chrome Default Dir"]
relevant_logtypes = sorted(list({logtype  for rule in savedsearches for logtype  in section_logtypes[rule]})) #[(x[0], str(x[1])) for x in state_span]
relevant_logtypes.append(('wineventlog:security', '4624'))
splunk_tools_instance = SplunkTools()
log_generator_instance = LogGenerator(relevant_logtypes, splunk_tools_instance)
register(id='splunk-v0', entry_point='custom_splunk.envs:SplunkEnv', kwargs={'rule_frequency':1, 'search_window':10, 'span_size':60, 'splunk_tools_instance':splunk_tools_instance, "log_generator_instance": log_generator_instance, "relevant_logtypes": relevant_logtypes, "num_of_searches": len(savedsearches)})