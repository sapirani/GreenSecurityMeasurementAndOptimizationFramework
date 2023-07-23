
from datetime import datetime
import re
from xml.etree import ElementTree as ET
from faker import Faker


class LogGenerator:

    def replace_fields_in_log(self, log, log_source, time_range, replacements_dict):
        # random time from time_range
        start_date, end_date=time_range
        start_date = datetime.strptime(start_date, '%m/%d/%Y:%H:%M:%S') 
        end_date = datetime.strptime(end_date, '%m/%d/%Y:%H:%M:%S') 
        time = Faker().date_time_between(start_date, end_date, tzinfo=None)      
                        
        if log_source == 'WinEventLog':
            new_log = re.sub(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", time.strftime("%m/%d/%Y %I:%M:%S %p"), log, flags=re.MULTILINE)
            new_log += '\nIsFakeLog=True'
            for field, new_value in replacements_dict.items():
                new_log = re.sub(f"{field}=\S+", f"{field}={new_value}", new_log, flags=re.MULTILINE)
        else:
            xml = ET.fromstring(log)
            for field, new_value in replacements_dict.items():
                for elem in xml.iter():
                    if elem.attrib.get('Name') == field:
                        elem.text = new_value
            for elem in xml.iter():
                if elem.attrib.get('Name') == 'UtcTime':
                    elem.text = time.isoformat() 
            # Find the 'TimeCreated' element and set the 'SystemTime' attribute
            time_created_elem = xml.find('{http://schemas.microsoft.com/win/2004/08/events/event}System')
            if time_created_elem is not None:
                time_created_elem.set('SystemTime', time.isoformat())  # 'Z' is added to indicate UTC time
       
            # Register the namespace with a prefix
            ET.register_namespace('', 'http://schemas.microsoft.com/win/2004/08/events/event')
            # Use the registered prefix in your XPath query
            event_data_elem = xml.find('{http://schemas.microsoft.com/win/2004/08/events/event}EventData')
            if event_data_elem is not None:
                ET.SubElement(event_data_elem, 'Data', {'Name': 'IsFakeLog'}).text = 'True'
                print('added')
            new_log = ET.tostring(xml, encoding='unicode')
        return new_log
    
    def compare_distributions(self, dist1, dist2):
        # Placeholder for your distribution comparison function
        # This could use a metric like KL divergence
        pass
    
    def get_reward(self, alerts_status, rules_energy_df, dist_distance):
        pass
    
    def perform_action(self, action, log, log_type, replacement_dict, time_range):
        # TODO according to the action, generate logs and insert them to splunk
        pass
