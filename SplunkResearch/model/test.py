


import os
import random
from log_generator import LogGenerator
from config import replacement_dicts
import sys
from logtypes import logtypes
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/splunk_tools')
from SplunkResearch.splunk_tools import SplunkTools

if __name__ == '__main__':

    splunk_tools = SplunkTools()
    # print(splunk_tools.get_rules_pids(60))
    # print(splunk_tools.extract_logs('WinEventLog:Security', '4624'))
    # test insertion with fake logs of wineventlog
    log_generator = LogGenerator()
    replacement_dicts = {field.lower():{key: random.choice(value) for key, value in replacement_dicts[field].items()} for field in replacement_dicts}    
    time_range = ('06/12/2023:20:00:00', '06/15/2023:08:00:00')
    if not os.path.exists('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/logs_to_duplicate_files'):
        for logtype in logtypes:
            source, eventcode = logtype
            source = source.lower()
            # extract real logs
            logs = splunk_tools.extract_logs(source, eventcode=eventcode, time_range=("-96h@h", "now"), limit=10)
            # replace fields in logs
            fake_logs = [log_generator.replace_fields_in_log(log, source, time_range, replacement_dicts[source]) for log in logs]
            # load fake logs from file
            # # insert fake logs
            # for fake_log in fake_logs:
            #     splunk_tools.insert_log(fake_log, source)
    fake_logs = splunk_tools.load_logs_to_duplicate_dict(logtypes)
    for logtype,logs in fake_logs.items():
        print(logtype)
        for log in logs:
            print(log)
            print('\n\n')
