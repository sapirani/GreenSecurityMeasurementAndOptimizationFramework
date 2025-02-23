
from datetime import datetime, timedelta, timezone
import logging
import random
import re
from xml.etree import ElementTree as ET
from faker import Faker
import sys
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

sys.path.insert(1, '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch')

class LogGenerator:
    
    def __init__(self, logtypes, splunk_tools_instance, big_replacement_dicts=None):
        self.big_replacement_dicts = big_replacement_dicts
        self.splunk_tools = splunk_tools_instance
        self.logs_to_duplicate_dict = self.init_logs_to_duplicate_dict(logtypes)
        # self.optional_process_names = ['C:\\Windows\\chrome.exe', 'C:\\Windows\\explorer.exe', 'C:\\Windows\\sql']
    
    # def generate_fake_time(self, start_date, end_date):
    #     time = Faker().date_time_between(start_date, end_date, tzinfo=None)
    #     return time
    
    def generate_fake_time(self, start_date, end_date):
        # Get total seconds between dates for more precise randomization
        time_delta = end_date - start_date
        total_seconds = time_delta.days * 86400 + time_delta.seconds
        
        # Generate random number of seconds between 0 and total_seconds
        random_seconds = random.randint(0, total_seconds)
        
        # Create the random datetime by adding seconds to start_date
        random_date_time = start_date + timedelta(seconds=random_seconds)
        return random_date_time
    
    def init_logs_to_duplicate_dict(self, logtypes):
        # logs_to_duplicate_dict = {(logtype[0].lower(), logtype[1]): self.splunk_tools.generate_log(logtype[0].lower(), logtype[1]) for logtype in logtypes}
        # # logs_to_duplicate_dict = {(logtype[0].lower(), logtype[1]): self.splunk_tools.extract_logs(logtype[0].lower(),time_range=("1", "now"), eventcode=logtype[1], limit=100) for logtype in logtypes}
        return self.splunk_tools.load_logs_to_duplicate_dict(logtypes)
    


    def generate_log(self, logsource, eventcode, istrigger, time_range, variation_id=None):

        log = self.logs_to_duplicate_dict[f"{logsource.lower()}_{eventcode}_{istrigger}"][0]
        
        # Generate time
        start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
        end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
        time = self.generate_fake_time(start_date, end_date)
        
        # Replace time
        log = re.sub(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", 
                    time.strftime("%m/%d/%Y %I:%M:%S %p"), 
                    log, 
                    flags=re.MULTILINE)
        
        # Add diversity if variation_id is provided
        if variation_id is not None:
            log = self.add_variation(log, logsource, eventcode, variation_id)
        
        # Add fake flag
        insert_line = "is_fake=1"
        field_pattern = re.compile(r'(\w+=.*?)(?=\n+Message=)')
        modified_log = re.sub(field_pattern, fr'\g<0>\n{insert_line}', log)
        
        return modified_log
    
    def add_variation(self, log, logsource, eventcode, variation_id):
        variations = {
            ('wineventlog:security', '4732'): {  # Admin group modification
                'ComputerName': f'dt-{variation_id}.auth.ad.bgu.ac.il',
                'Security ID': f'S-1-5-21-1220750395-818509756-262303683-{variation_id}',
                'Account Name': f'DT-{variation_id}$',
                'Account Domain': f'BGU-USERS-{variation_id}',
                # 'Group Name': 'Administrator'
            },
            
            ('wineventlog:security', '4769'): {  # Kerberoasting
                'ComputerName': f'win-dc-{variation_id}.attackrange.local',
                'Account Name': f'Administrator@ATTACKRANGE{variation_id}.LOCAL',
                'Account Domain': f'ATTACKRANGE{variation_id}.LOCAL',
                # 'Service Name': f'srv_smb{variation_id}$',
                'Service ID': f'ATTACKRANGE\\srv_smb{variation_id}',
                'Logon GUID': f'{{154B8810-5DFB-8AB3-16CA-210CAFC9{variation_id}}}',
                # 'Ticket Options': '0x40800000',
                # 'Ticket Encryption Type': '0x17'
            },

            ('wineventlog:security', '4663'): {  # Chrome access
                'ComputerName': f'user-PC-{variation_id}.domain.com',
                'Security ID': f'S-1-5-21-1234567890-1234567890-1234567890-{variation_id}',
                'Account Name': f'user_{variation_id}',
                'Account Domain': f'DOMAIN_{variation_id}',
                # 'Object Name': f'C:\\Users\\user_{variation_id}\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Cache\\data_{variation_id}',
                # 'Process Name': self.optional_process_names[variation_id%len(self.optional_process_names)]  # Not explorer.exe to trigger
            },

            ('wineventlog:security', '5140'): {  # Network share
                'Computer': f'ar-win-{variation_id}.attackrange.local',
                'SecurityID': f'ATTACKRANGE\\Admin_{variation_id}',
                'AccountName': f'Admin_{variation_id}',
                'AccountDomain': 'ATTACKRANGE',
                'SourceAddress': f'10.0.1.{variation_id}',
                # 'ShareName': f'\\\\MININT-LDTUS6A\\C${variation_id}',  # Admin shares to trigger
                # 'AccessMask': '0x1'
            },

            ('wineventlog:system', '7040'): {  # Service disabled
                'ComputerName': f'server-{variation_id}.domain.com',
                'User': f'DOMAIN\\User_{variation_id}',
                'Sid': f'S-1-5-21-1234567890-1234567890-1234567890-{variation_id}',
                # 'Message': 'The start type of the critical_service service was changed from auto start to disabled.'  # Changed to match rule
            },

            ('wineventlog:system', '7036'): {  # Service stopped
                'ComputerName': f'user-PC-{variation_id}.domain.com',
                # 'Message': 'The Volume Shadow Copy service entered the stopped state.'  # Changed to match rule
            },

            ('wineventlog:system', '7045'): {  # CLOP service
                'ComputerName': f'win-dc-{variation_id}.attackrange.local',
                'Sid': f'S-1-5-21-3730028101-1805993102-2296611634-{variation_id}',
                # 'Service Name': 'SecurityCenterDT',  # Keep constant to match rule
                'Service File Name': f'c:\\Users\\Public\\clop_{variation_id}.exe'
            }
        }
        
        if (logsource, eventcode) in variations:
            log_variations = variations[(logsource, eventcode)]
            for field, value in log_variations.items():
                # Escape backslashes in the value
                escaped_value = value.replace('\\', '\\\\')
                
                if field == 'Message':
                    # Special handling for Message field
                    pattern = r'Message=.*?(?=\n\n|\Z)'
                    replacement = f'Message={escaped_value}'
                    log = re.sub(pattern, replacement, log, flags=re.DOTALL)
                elif field in ['Security ID', 'SecurityID']:
                    # Handle security IDs
                    pattern = r'Security ID:\s*S-1-5-\d+(-\d+)*|SecurityID:\s*[^\n]+'
                    replacement = f'{field}:\t{escaped_value}'
                    log = re.sub(pattern, replacement, log)
                else:
                    # Check if field is in Message block (using colon)
                    if re.search(f'{field}:\s*[^\n]+', log):
                        pattern = f'{field}:\s*[^\n]+'
                        replacement = f'{field}:\t{escaped_value}'
                    else:
                        # Handle normal fields (using equals)
                        pattern = f'{field}=[^\n]+'
                        replacement = f'{field}={escaped_value}'
                    log = re.sub(pattern, replacement, log)
        return log

        
    def generate_logs(self, logsource, eventcode, istrigger, time_range, num_logs, diversity=0, max_workers=12):
        """
        Fully parallelized log generation with two-level parallelization:
        1. Parallel template preparation 
        2. Parallel log generation within batches
        """
        diversity = min(diversity + 1, num_logs)  # Include original template
        
        # Parse date range once
        if isinstance(time_range[0], str):
            start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
            end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
        else:
            start_date, end_date = time_range
        
        # Pre-compile regex patterns (used for all logs)
        time_pattern = re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", re.MULTILINE)
        field_pattern = re.compile(r'(\w+=.*?)(?=\n+Message=)')
        
        # Step 1: Prepare variation templates in parallel
        def prepare_template(variation_id):
            # Get the base log template
            base_log = self.logs_to_duplicate_dict[f"{logsource.lower()}_{eventcode}_{istrigger}"][0]
            
            # Apply variation-specific changes
            if variation_id > 0:  # Skip for the original template
                base_log = self.add_variation(base_log, logsource, eventcode, variation_id)
                
            # Add fake flag to the template
            insert_line = "is_fake=1"
            modified_template = re.sub(field_pattern, fr'\g<0>\n{insert_line}', base_log)
            
            return modified_template
        
        # Generate templates in parallel
        variation_templates = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(diversity, max_workers)) as executor:
            future_to_template = {
                executor.submit(prepare_template, var_id): var_id 
                for var_id in range(diversity)
            }
            
            for future in concurrent.futures.as_completed(future_to_template):
                var_id = future_to_template[future]
                variation_templates[var_id] = future.result()
        
        # Calculate distribution of logs across variations
        logs_per_variation = num_logs // diversity
        remaining_logs = num_logs % diversity
        
        distribution = {}
        for d in range(diversity):
            count = logs_per_variation + (1 if d < remaining_logs else 0)
            if count > 0:
                distribution[d] = count
        
        # Step 2: Generate logs for each variation with internal parallelization
        all_logs = []
        
        def generate_single_log(template):
            # Generate time
            time_obj = self.generate_fake_time(start_date, end_date)
            time_str = time_obj.strftime("%m/%d/%Y %I:%M:%S %p")
            
            # Apply time to the template
            return time_pattern.sub(time_str, template)
        
        # Function to generate logs in sub-batches
        def generate_variation_batch(variation_id, count):
            template = variation_templates[variation_id]
            
            # For larger batches, parallelize internally
            if count > 100:  # Threshold for internal parallelization
                # Create a list of the same template repeated count times
                templates = [template] * count
                
                # Determine sub-batch worker count - don't use all workers for small batches
                sub_workers = min(max(1, max_workers // 4), count // 50)
                
                # Generate logs in parallel within this batch
                with concurrent.futures.ThreadPoolExecutor(max_workers=sub_workers) as sub_executor:
                    return list(sub_executor.map(generate_single_log, templates))
            else:
                # For smaller batches, process sequentially
                return [generate_single_log(template) for _ in range(count)]
        
        # Process variation batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(generate_variation_batch, var_id, count): var_id 
                for var_id, count in distribution.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_logs = future.result()
                all_logs.extend(batch_logs)
        
        return all_logs
        
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
#     log:source = 'xmlwineventlog:Microsoft-Windows-Sysmon/Operational'
#     time_range = ('06/12/2023:20:00:00', '06/15/2023:08:00:00')
#     replacement_dicts = {field.lower():{key: random.choice(value) for key, value in replacement_dicts[field].items()} for field in replacement_dicts}    
#     logging.info(replacement_dicts)
#     logging.info(log_generator.replace_fields_in_log(log, log:source, time_range, replacement_dicts['xmlwineventlog:microsoft-windows-sysmon/operational']))