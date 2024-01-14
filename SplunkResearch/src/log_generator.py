
from datetime import datetime, timedelta
import logging
import random
import re
from xml.etree import ElementTree as ET
from faker import Faker
import sys
sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')

class LogGenerator:
    
    def __init__(self, logtypes, big_replacement_dicts, splunk_tools_instance):
        self.big_replacement_dicts = big_replacement_dicts
        self.splunk_tools = splunk_tools_instance
        self.logs_to_duplicate_dict = self.init_logs_to_duplicate_dict(logtypes)

    
    def replace_fields_in_log(self, log, log_source, time_range, replacement_dict):
        # random time from time_range
        start_date, end_date=time_range
        time = self.generate_fake_time(start_date, end_date)      
        if log_source.split(':')[0] == 'wineventlog':
            new_log = re.sub(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", time.strftime("%m/%d/%Y %I:%M:%S %p"), log, flags=re.MULTILINE)
            new_log += '\nIsFakeLog=True'
            for field, new_value in replacement_dict.items():
                new_log = re.sub(f"{field}=\S+", f"{field}={new_value}", new_log, flags=re.MULTILINE)
        else:
            ET.register_namespace('', "http://schemas.microsoft.com/win/2004/08/events/event")
            
            try:
                xml = ET.fromstring(log)
            except ET.ParseError:
                logging.info('ParseError', log)
                return None
            # data_elem = xml.find('EventData')
            # if data_elem is None:
            #     logging.info('No EventData', log)
            #     return None
            #     try:
            #         data_elem.set(field, new_value)
            #         print(field, new_value)
            #     except ValueError:
            #         logging.info('ValueError', log)
            #         continue
            for field, new_value in replacement_dict.items():
                for elem in xml.iter():
                    if elem.attrib.get('Name') == field:
                        elem.text = new_value
            for elem in xml.iter():
                if elem.attrib.get('Name') == 'UtcTime' or elem.attrib.get('Name') == 'CreationUtcTime':
                    elem.text = time.isoformat()
            # Find the 'TimeCreated' element and set the 'SystemTime' attribute
            system_elem = xml.find('{http://schemas.microsoft.com/win/2004/08/events/event}System')
            if system_elem is not None:
                time_created_elem = system_elem.find('{http://schemas.microsoft.com/win/2004/08/events/event}TimeCreated')
                if time_created_elem is not None:
                    time_created_elem.set('SystemTime', time.isoformat())
                ET.SubElement(system_elem, 'IsFakeLog').text = 'True'
            # ET.register_namespace('', "http://schemas.microsoft.com/win/2004/08/events/event")
            new_log = ET.tostring(xml, encoding='unicode')
            # print(new_log)
            # print(log)
            # print(xml)
        return log,time.timestamp()

    # def generate_fake_time(self, start_date, end_date):
    #     time = Faker().date_time_between(start_date, end_date, tzinfo=None)
    #     return time
    
    def generate_fake_time(self, start_date, end_date):
        time_delta = end_date - start_date
        random_days = random.randint(0, time_delta.days)
        random_seconds = random.randint(0, time_delta.seconds-1)
        # random_microseconds = random.randint(0, time_delta.microseconds)

        random_date_time = start_date + timedelta(
            days=random_days,
            seconds=random_seconds,
            # microseconds=random_microseconds
        )

        return random_date_time
    
    def init_logs_to_duplicate_dict(self, logtypes):
        # logs_to_duplicate_dict = {(logtype[0].lower(), logtype[1]): self.splunk_tools.generate_log(logtype[0].lower(), logtype[1]) for logtype in logtypes}
        # # logs_to_duplicate_dict = {(logtype[0].lower(), logtype[1]): self.splunk_tools.extract_logs(logtype[0].lower(),time_range=("1", "now"), eventcode=logtype[1], limit=100) for logtype in logtypes}
        return self.splunk_tools.load_logs_to_duplicate_dict(logtypes)
        
    def generate_log(self, logsource, eventcode, replacement_dict, time_range):
        # logs_to_duplicate_dict = self.logs_to_duplicate_dict[(logsource, eventcode)]
        # if logs_to_duplicate_dict is None or len(logs_to_duplicate_dict) == 0:
        #     return None
        # log = random.choice(logs_to_duplicate_dict)
        # return self.replace_fields_in_log(log, logsource, time_range, replacement_dict)
        log = self.logs_to_duplicate_dict[logsource, eventcode][0]
        start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S') 
        end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S') 
        return log, self.generate_fake_time(start_date,end_date).timestamp()
    
    def get_replacement_values(self, logsource):
        replacement_dict = self.random_replacement_values()    
        return replacement_dict[logsource]

    def random_replacement_values(self):
        replacement_dict = {field.lower():{key: random.choice(value) for key, value in self.big_replacement_dicts[field].items()} for field in self.big_replacement_dicts}
        return replacement_dict
        
    def generate_logs(self, logsource, eventcode, time_range, num_logs):
        logsource_replacement_dict = self.get_replacement_values(logsource)
        logs = []
        for i in range(num_logs):
            log = self.generate_log(logsource, eventcode, logsource_replacement_dict, time_range)
            if log is not None:
                logs.append(log)
        return logs
    
# if __name__=='__main__':
#     log_generator = LogGenerator()
#     # test generation of sysmon log:
#     log = """<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event">
#     <System>
#         <Provider Name="Microsoft-Windows-Sysmon" Guid="{5770385F-C22A-43E0-BF4C-06F5698FFBD9}" />
#         <EventID>1</EventID>
#         <Version>5</Version>
#         <Level>4</Level>
#         <Task>1</Task>
#         <Opcode>0</Opcode>
#         <Keywords>0x8000000000000000</Keywords>
#         <TimeCreated SystemTime="2021-06-15T08:00:00.0000000Z" />
#         <EventRecordID>1</EventRecordID>
#         <Correlation />
#         <Execution ProcessID="4" ThreadID="8" />
#         <Channel>Microsoft-Windows-Sysmon/Operational</Channel>
#         <Computer>DESKTOP-1</Computer>
#         <Security UserID="S-1-5-18" />
#     </System>
#     <EventData>
#         <Data Name="RuleName">CreateRemoteThread</Data>
#         <Data Name="UtcTime">2021-06-15 08:00:00.000</Data>
#         <Data Name="ProcessGuid">{00000000-0000-0000-0000-000000000000}</Data>
#         <Data Name="ProcessId">0</Data>
#         <Data Name="Image">-</Data>
#         <Data Name="TargetProcessGuid">{00000000-0000-0000-0000-000000000000}</Data>
#         <Data Name="TargetProcessId">0</Data>
#         <Data Name="TargetImage">-</Data>
#         <Data Name="NewThreadId">0</Data>
#         <Data Name="StartAddress">-</Data>
#         <Data Name="StartModule">-</Data>
#         <Data Name="StartFunction">-</Data>
#         <Data Name="StartFunctionName">-</Data>
#         <Data Name="Type">-</Data>
#         <Data Name="TypeDescription">-</Data>
#         <Data Name="CreationUtcTime">-</Data>
#         <Data Name="CreationTime">-</Data>
#         <Data Name="CreationProcessGuid">{00000000-0000-0000-0000-000000000000}</Data>
#         <Data Name="CreationProcessId">0</Data>
#         <Data Name="CreationImage">-</Data>
#         <Data Name="CreationCommandLine">-</Data>
#         <Data Name="CreationCurrentDirectory">-</Data>
#         <Data Name="CreationUser">-</Data>
#     </EventData>
#     </Event>"""
#     log_source = 'xmlwineventlog:Microsoft-Windows-Sysmon/Operational'
#     time_range = ('06/12/2023:20:00:00', '06/15/2023:08:00:00')
#     replacement_dicts = {field.lower():{key: random.choice(value) for key, value in replacement_dicts[field].items()} for field in replacement_dicts}    
#     logging.info(replacement_dicts)
#     logging.info(log_generator.replace_fields_in_log(log, log_source, time_range, replacement_dicts['xmlwineventlog:microsoft-windows-sysmon/operational']))