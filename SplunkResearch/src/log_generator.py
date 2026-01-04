import re
import random
import concurrent.futures
import time
import logging
import os
import pickle
from datetime import datetime, timedelta
from functools import lru_cache
import logging
logger = logging.getLogger(__name__)
PREFIX_PATH = '/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/'

def _worker_generate_log(task):
    """
    Top-level function for ProcessPoolExecutor.
    """
    template, timestamp, var_id, field_pattern, time_pattern, log_source, real_ts = task
    
    # Apply variation ID injection
    # Note: Ensure field_pattern is valid regex. 
    # If field_pattern matches the whole event, \g<0> appends the var_id.
    template_with_var = field_pattern.sub(rf'\g<0>\r\nvar_id={var_id}\r\nreal_ts={real_ts}', template)
    
    # Apply timestamp injection
    return time_pattern.sub(timestamp, template_with_var), log_source

class LogGenerator:
    
    def __init__(self, logtypes, big_replacement_dicts=None, cache_dir='./template_cache'):
        logger.info("Initializing OptimizedLogGenerator")
        start_time = time.time()
        
        self.big_replacement_dicts = big_replacement_dicts
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Load templates once and store them
        self.logs_to_duplicate_dict = {}
        self.variation_cache = {}
        
        # Pre-compile regex patterns
        logger.info("Compiling regex patterns")
        pattern_start = time.time()
        self.time_pattern = re.compile(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", re.MULTILINE)
        self.field_pattern = re.compile(r'(\w+=.*?)(?=\n+Message=)')
        # logger.info(f"Regex compilation completed in {time.time() - pattern_start:.3f} seconds")
        
        # Initialize log templates on demand (lazy loading)
        self._init_variation_templates()
        self.fake_splunk_state = {} # a dict of time range to event codes and variation ids quantities
        self.logs_to_delete = {}
        # logger.info(f"LogGenerator initialization completed in {time.time() - start_time:.3f} seconds")
        
    def _init_variation_templates(self):
        """Initialize the variation templates for faster substitution"""
        self.variations = {
            ('wineventlog:security', '4732'): {  # Admin group modification
                'ComputerName': 'dt-{}.auth.ad.bgu.ac.il',
                'Security ID': 'S-1-5-21-1220750395-818509756-262303683-{}',
                'Account Name': 'DT-{}$',
                'Account Domain': 'BGU-USERS-{}',
            },

            ('wineventlog:security', '4769'): {  # Kerberoasting
                'ComputerName': 'win-dc-{}.attackrange.local',
                'Account Name': 'Administrator@ATTACKRANGE{}.LOCAL',
                'Account Domain': 'ATTACKRANGE{}.LOCAL',
                'Service ID': 'ATTACKRANGE\\srv_smb{}',
                'Logon GUID': '{{154B8810-5DFB-8AB3-16CA-210CAFC9{}}}',
            },

            ('wineventlog:security', '4663'): {  # Chrome access
                'ComputerName': 'user-PC-{}.domain.com',
                'Security ID': 'S-1-5-21-1234567890-1234567890-1234567890-{}',
                'Account Name': 'user_{}',
                'Account Domain': 'DOMAIN_{}',
            },

            ('wineventlog:security', '5140'): {  # Network Share Discovery
                'ComputerName': 'user-PC-{}.domain.com',
                'Security ID': 'S-1-5-21-1234567890-1234567890-1234567890-{}',
                'Account Name': 'user_{}',
                'Account Domain': 'DOMAIN_{}',
                'Share Name': '\\\\user-PC-{}.domain.com\\C$'
            },

            ('wineventlog:security', '4624'): {  # Rapid Auth on Multiple Hosts
                'ComputerName': 'node-{}.net.corp',
                'Account Name': 'jcrawford',
                'Account Domain': 'net.corp',
                'Source Network Address': '192.168.{}.77',
                'Logon ID': '0x4E5B3A{}',
            },

            ('wineventlog:security', '4662'): {  # AD Replication Request
                'ComputerName': 'replica-{}.ad.internal',
                'Account Name': 'replica_user{}',
                'Account Domain': 'INTERNALDOM{}',
                'Security ID': 'S-1-5-21-999999999-888888888-777777777-{}',
                'Object Name': 'CN=Configuration,DC=internal,DC=dom{}',
            },

            ('wineventlog:system', '7040'): {  # Service disabled
                'ComputerName': 'server-{}.domain.com',
                'User': 'DOMAIN\\User_{}',
                'Sid': 'S-1-5-21-1234567890-1234567890-1234567890-{}',
            },

            ('wineventlog:system', '7036'): {  # Service stopped
                'ComputerName': 'user-PC-{}.domain.com',
            },

            ('wineventlog:system', '7045'): {  # CLOP service
                'ComputerName': 'win-dc-{}.attackrange.local',
                'Sid': 'S-1-5-21-3730028101-1805993102-2296611634-{}',
                'Service File Name': 'c:\\Users\\Public\\clop_{}.exe'
            }
        }
                   
    def load_logs_to_duplicate_dict(self, logtypes):
        dir_name = 'logs_resource'
        # load the logs to duplicate from disk
        logs_to_duplicate_dict = {f"{logtype[0].lower()}_{logtype[1]}_{istrigger}": [] for istrigger,_ in enumerate(['notrigger', 'trigger']) for logtype in logtypes}
        for logtype in logtypes:
            source = logtype[0].lower()
            eventcode = logtype[1]
            for istrigger, istrigger_string in enumerate(['notrigger', 'trigger']):
                path = f'{PREFIX_PATH}{dir_name}/{source.replace("/", "__")}_{eventcode}_{istrigger_string}.txt'
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    text = f.read()
                    results = text.split('\n[EOF]\n')
                    # results = self.split_logs(source, text)   
                    for log in results:
                         if log != '':
                             logs_to_duplicate_dict[f"{logtype[0].lower()}_{logtype[1]}_{istrigger}"].append(log)
        return logs_to_duplicate_dict   

    def _get_template(self, logsource, eventcode, istrigger):
        """Lazy-loading templates to avoid loading everything at once"""
        key = f"{logsource.lower()}_{eventcode}_{istrigger}"
        
        if key not in self.logs_to_duplicate_dict:
            # Load just this template
            logger.info(f"Loading template for {key}")
            start_time = time.time()
            templates = self.load_logs_to_duplicate_dict([(logsource, eventcode, istrigger)])
            self.logs_to_duplicate_dict.update(templates)
            # logger.info(f"Template loading for {key} completed in {time.time() - start_time:.3f} seconds")
        
        return self.logs_to_duplicate_dict[key][0]
    
    def _generate_timestamp_pool(self, start_date, end_date, size=1000):
        """Pre-generate a pool of timestamps for better performance"""
        format_start = time.time()
        
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%m/%d/%Y:%H:%M:%S')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%m/%d/%Y:%H:%M:%S')
        
        calc_start = time.time()
        time_delta = end_date - start_date
        total_seconds = time_delta.days * 86400 + time_delta.seconds
        # logger.info(f"Time delta calculation completed in {time.time() - calc_start:.3f} seconds")
        
        generate_start = time.time()
        timestamp_pool = []
        for _ in range(size):
            random_seconds = random.randint(0, total_seconds)
            random_date_time = start_date + timedelta(seconds=random_seconds)
            timestamp_pool.append(random_date_time.strftime("%m/%d/%Y %I:%M:%S %p"))
        
        # logger.info(f"Timestamp generation completed in {time.time() - generate_start:.3f} seconds")
        # logger.info(f"Total timestamp pool generation completed in {time.time() - format_start:.3f} seconds")
        
        return timestamp_pool

    def _apply_variations(self, log, logsource, eventcode, variation_id):
        """Apply variations to a log template and cache the result"""
        if (logsource, eventcode) not in self.variations:
            return log
            
        # Create helper functions for different field formats
        def replace_message(log, value):
            pattern = r'Message=.*?(?=\n\n|\Z)'
            return re.sub(pattern, f'Message={value}', log, flags=re.DOTALL)
            
        def replace_security_id(log, value):
            pattern = r'Security ID:\s*S-1-5-\d+(-\d+)*'
            return re.sub(pattern, f'Security ID:\t{value}', log)
            
        def replace_security_id_alt(log, value):
            pattern = r'SecurityID:\s*[^\n]+'
            return re.sub(pattern, f'SecurityID:\t{value}', log)
            
        def replace_field_equals(log, field, value):
            pattern = f'{re.escape(field)}=[^\\n]+'
            return re.sub(pattern, f'{field}={value}', log)
            
        def replace_field_colon(log, field, value):
            pattern = f'{re.escape(field)}:\\s*[^\\n]+'
            return re.sub(pattern, f'{field}:\t{value}', log)
        
        log_variations = self.variations[(logsource, eventcode)]
        start_time = time.time()
        
        for field, template in log_variations.items():
            # Format the value with the variation ID
            value = template.format(variation_id)
            # Escape backslashes
            escaped_value = value.replace('\\', '\\\\')
            
            # Apply the replacement based on field type
            if field == 'Message':
                log = replace_message(log, escaped_value)
            elif field == 'Security ID':
                log = replace_security_id(log, escaped_value)
            elif field == 'SecurityID':
                log = replace_security_id_alt(log, escaped_value)
            else:
                # Try equals format first
                if re.search(f'{re.escape(field)}=[^\\n]+', log):
                    log = replace_field_equals(log, field, escaped_value)
                else:
                    # Then try colon format
                    if re.search(f'{re.escape(field)}:\\s*[^\\n]+', log):
                        log = replace_field_colon(log, field, escaped_value)
        
        total_time = time.time() - start_time
        if total_time > 0.01:  # Only log if it took more than 10ms
            logger.info(f"Applied variations in {total_time:.3f}s for {logsource}:{eventcode}:{variation_id}")
        
        # Add fake flag
        return log
    
    # def _prepare_base_template(self, logsource, eventcode, istrigger):
    #     """Get and prepare a base template with caching for efficiency"""
    #     start_time = time.time()
        
    #     # Check cache first
    #     cache_key = f"{logsource.lower()}_{eventcode}_{istrigger}_base"
    #     cache_path = os.path.join(self.cache_dir, f"{cache_key}.template")
        
    #     if os.path.exists(cache_path):
    #         # logger.info(f"Loading base template from cache: {cache_key}")
    #         with open(cache_path, 'r') as f:
    #             result = f.read()
    #     else:
    #         # logger.info(f"Creating base template for {cache_key}")
    #         template = self._get_template(logsource, eventcode, istrigger)
            
    #         # Add fake flag
    #         result = self.field_pattern.sub(r'\g<0>\nis_fake=1', template)
            
    #         # Cache the result
    #         with open(cache_path, 'w') as f:
    #             f.write(result)
        
    #     # logger.info(f"Base template preparation completed in {time.time() - start_time:.3f} seconds")
    #     return result

    
    def _get_variation_template(self, logsource, eventcode, istrigger, variation_id, base_template=None):
        """Get a variation template, creating and caching it if necessary"""
        # If it's variation 0, return the base template
        if variation_id == 0:
            return self._prepare_base_template(logsource, eventcode, istrigger)
            
        # Check if we have this variation in cache
        cache_key = f"{logsource.lower()}_{eventcode}_{istrigger}_{variation_id}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.template")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading variation template from cache: {cache_key}")
            with open(cache_path, 'r') as f:
                return f.read()
        
        # If not in cache, create it
        # logger.info(f"Creating variation template for {cache_key}")
        
        # Get base template if not provided
        if base_template is None:
            base_template = self._prepare_base_template(logsource, eventcode, istrigger)
            
        # Apply variations
        result = self._apply_variations(base_template, logsource, eventcode, variation_id)
        
        # Cache the result
        with open(cache_path, 'w') as f:
            f.write(result)
            
        return result
    
    # def prepare_tasks(self, logsource, eventcode, istrigger, time_range, num_logs, diversity=1, injection_id=0, max_workers=12):
    #     if time_range not in self.fake_splunk_state:
    #         self.fake_splunk_state[time_range] = {}
    #     if time_range not in self.logs_to_delete:
    #         self.logs_to_delete[time_range] = {}
    #     log_type_key = f"{logsource.lower()}_{eventcode}_{istrigger}"
    #     if log_type_key not in self.fake_splunk_state[time_range]:
    #         self.fake_splunk_state[time_range][log_type_key] = {}
    #     if log_type_key not in self.logs_to_delete[time_range]:
    #         self.logs_to_delete[time_range][log_type_key] = {}
    #     """Generate logs with optimized performance using pre-cached templates"""
    #     # logger.info(f"Generating {num_logs} logs with diversity={diversity}, max_workers={max_workers}")
        
    #     # Ensure diversity is at least 1 and no more than num_logs
    #     # diversity = max(1, min(diversity, num_logs))
        
    #     # Parse time range once
    #     if isinstance(time_range[0], str):
    #         start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
    #         end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
    #     else:
    #         start_date, end_date = time_range
    #     # logger.info(f"Time parsing completed in {time.time() - time_parsing_start:.3f} seconds")
        
    #     # Pre-generate timestamp pool for better performance
    #     timestamp_pool_size = min(1000, num_logs * 2)  # Create some extra timestamps for randomness
    #     timestamp_pool = self._generate_timestamp_pool(start_date, end_date, timestamp_pool_size)
    #     # logger.info(f"Timestamp pool generation ({timestamp_pool_size} timestamps) completed in {time.time() - pool_start:.3f} seconds")
        
    #     # Prepare templates if they don't exist
    #     # Prepare base template
    #     base_template = self._prepare_base_template(logsource, eventcode, istrigger)
        
    #     # Check if we need to prepare variations
    #     missing_templates = []
    #     is_trigger = int(log_type_key.split('_')[-1])
        
    #     for var_id in range(is_trigger, diversity+is_trigger):
    #         cache_key = f"{logsource.lower()}_{eventcode}_{istrigger}_{var_id}"
    #         cache_path = os.path.join(self.cache_dir, f"{cache_key}.template")
    #         if not os.path.exists(cache_path):
    #             missing_templates.append(var_id)
        
    #     # Prepare missing templates if needed
    #     if missing_templates:
    #         logger.debug(f"Preparing {len(missing_templates)} missing variation templates")
    #         for var_id in missing_templates:
    #             self._get_variation_template(logsource, eventcode, istrigger, var_id, base_template)
        
    #     # logger.info(f"Template preparation completed in {time.time() - templates_start:.3f} seconds")
        
    #     # Calculate logs per variation
    #     logs_per_variation = num_logs // diversity
    #     remaining_logs = num_logs % diversity
        
    #     # Create a list of variation IDs based on distribution
    #     variation_assignments = []
    #     max_old_diversity = max([*self.fake_splunk_state[time_range][log_type_key].keys(), 0])
        
    #     for var_id in range(is_trigger, diversity+is_trigger):
    #         new_count = logs_per_variation + (1 if (var_id-is_trigger) < remaining_logs else 0)
    #         if var_id not in self.fake_splunk_state[time_range][log_type_key]:
    #             self.fake_splunk_state[time_range][log_type_key][var_id] = 0
    #         # if var_id not in self.logs_to_delete[time_range][log_type_key]:
    #         #     self.logs_to_delete[time_range][log_type_key][var_id] = 0
    #         old_count = self.fake_splunk_state[time_range][log_type_key][var_id]
    #         count_to_update = new_count - old_count
    #         if count_to_update < 0: # need to delete logs
    #             self.logs_to_delete[time_range][log_type_key][var_id] = - count_to_update
    #         self.fake_splunk_state[time_range][log_type_key][var_id] = new_count
    #         if count_to_update > 0: # only add if we need logs of this variation
    #             variation_assignments.extend([var_id] * count_to_update)
        
    #     if diversity  <  max_old_diversity:
    #         for var_id in range(diversity + is_trigger, max_old_diversity + is_trigger,):
    #             old_count = self.fake_splunk_state[time_range][log_type_key].get(var_id, 0)
    #             if old_count > 0:
    #                 self.logs_to_delete[time_range][log_type_key][var_id] = old_count
    #                 self.fake_splunk_state[time_range][log_type_key][var_id] = 0
    #         # logging.info(f"old_count: {old_count}, new_count: {new_count}, count_to_update: {count_to_update}")
        
    #     # if need to delete logs, check if entire var_ids can be delete or it's need to be cherugial
    #     # if need to be cherugial, we will delete all logs of this log type and time range, and regenerate them
    #     # else we will just delete the entire var_id logs
    #     for var_id, delete_count in self.logs_to_delete[time_range][log_type_key].items():
    #         if delete_count > 0:
    #             self.logs_to_delete[time_range][log_type_key] = f"RemoveAll_{var_id}"
    #             # update variation assignments to regenerate all logs of this type
    #             for var_id2 in range(var_id, diversity+is_trigger):
    #                 new_count = self.fake_splunk_state[time_range][log_type_key][var_id2]
    #                 variation_assignments.extend([var_id2] * new_count)
    #             break

            
    #     # Shuffle to distribute variations evenly
    #     random.shuffle(variation_assignments)
    #     # logger.info(f"Task distribution calculation completed in {time.time() - distribution_start:.3f} seconds")
        
    #     # Load all templates we need
    #     template_loading_start = time.time()
    #     templates = {}
    #     for var_id in range(is_trigger, diversity+is_trigger):
    #         if var_id == is_trigger:
    #             templates[var_id] = base_template
    #         else:
    #             cache_key = f"{logsource.lower()}_{eventcode}_{istrigger}_{var_id}"
    #             cache_path = os.path.join(self.cache_dir, f"{cache_key}.template")
    #             with open(cache_path, 'r') as f:
    #                 templates[var_id] = f.read()
    #     logger.debug(f"Template loading completed in {time.time() - template_loading_start:.3f} seconds")
        
    #     # # Determine optimal batch size based on num_logs
    #     # batch_size = max(100, min(1000, num_logs // (max_workers * 2)))
        
    #     # # Create tasks with templates and timestamps
    #     # tasks_start = time.time()
    #     real_ts = datetime.now().strftime("%m_%d_%Y_%I_%M")
    #     generation_tasks = [
    #         (templates[var_id], random.choice(timestamp_pool), var_id, self.field_pattern, self.time_pattern, logsource, real_ts) 
    #         for var_id in variation_assignments
    #     ]

        
    #     return generation_tasks


############### version with delete logic - didnt work well ###############
    # def prepare_tasks(self, logsource, eventcode, istrigger, time_range, num_logs, diversity=1, injection_id=0, max_workers=12):

    #     # --- Initialization ---
    #     if time_range not in self.fake_splunk_state:
    #         self.fake_splunk_state[time_range] = {}
    #     if time_range not in self.logs_to_delete:
    #         self.logs_to_delete[time_range] = {}
            

            
            
    #     is_trigger_int = int(istrigger)
    #     log_type_key = f"{logsource.lower()}_{eventcode}_{is_trigger_int}"
        
    #     if log_type_key not in self.fake_splunk_state[time_range]:
    #         self.fake_splunk_state[time_range][log_type_key] = {}
    #     # We don't init logs_to_delete[log_type_key] as {} yet, because we might overwrite it with a string string later
    #     if log_type_key not in self.logs_to_delete[time_range]:
    #         self.logs_to_delete[time_range][log_type_key] = {}

    #     # --- Time Parsing (Existing Logic) ---
    #     if isinstance(time_range[0], str):
    #         start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
    #         end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
    #     else:
    #         start_date, end_date = time_range

    #     # --- Template & Timestamp Prep ---
    #     timestamp_pool_size = min(1000, num_logs * 2)
    #     timestamp_pool = self._generate_timestamp_pool(start_date, end_date, timestamp_pool_size)
    #     base_template = self._prepare_base_template(logsource, eventcode, istrigger)
        
    #     # Prepare Missing Templates
    #     start_var = is_trigger_int
    #     end_var = diversity + is_trigger_int
    #     missing_templates = []
    #     for var_id in range(start_var, end_var):
    #         cache_key = f"{logsource.lower()}_{eventcode}_{is_trigger_int}_{var_id}"
    #         if not os.path.exists(os.path.join(self.cache_dir, f"{cache_key}.template")):
    #             missing_templates.append(var_id)
    #     if missing_templates:
    #         for var_id in missing_templates:
    #             self._get_variation_template(logsource, eventcode, istrigger, var_id, base_template)

    #     # --- CORE LOGIC START ---
        
    #     logs_per_variation = num_logs // diversity
    #     remaining_logs = num_logs % diversity
        
    #     variation_assignments = []
        
    #     # We need to find the lowest var_id that causes a "Cutoff" (deletion).
    #     # Once found, ALL subsequent vars must be fully regenerated.
    #     cutoff_var_id = None 

    #     # 1. Loop through required diversity
    #     for var_id in range(start_var, end_var):
    #         # Calculate Target
    #         new_count = logs_per_variation + (1 if (var_id - start_var) < remaining_logs else 0)
            
    #         # Get Current State
    #         if var_id not in self.fake_splunk_state[time_range][log_type_key]:
    #             self.fake_splunk_state[time_range][log_type_key][var_id] = 0
    #         old_count = self.fake_splunk_state[time_range][log_type_key][var_id]

    #         # Logic Switch: Are we safely adding, or have we triggered a wipe?
            
    #         if cutoff_var_id is not None:
    #             # CASE A: Cutoff already triggered by a lower var_id.
    #             # This var_id is implicitly wiped by "RemoveAll_X".
    #             # We must regenerate the FULL count.
    #             variation_assignments.extend([var_id] * new_count)
    #             self.fake_splunk_state[time_range][log_type_key][var_id] = new_count
                
    #         elif new_count < old_count:
    #             # CASE B: This var_id requires reduction.
    #             # This triggers the Cutoff.
    #             cutoff_var_id = var_id
                
    #             # Since we are wiping this var (and all above), we regenerate FULL count.
    #             variation_assignments.extend([var_id] * new_count)
    #             self.fake_splunk_state[time_range][log_type_key][var_id] = new_count
                
    #         else:
    #             # CASE C: No cutoff yet, and new_count >= old_count.
    #             # Safe to just add the delta.
    #             count_to_add = new_count - old_count
    #             variation_assignments.extend([var_id] * count_to_add)
    #             self.fake_splunk_state[time_range][log_type_key][var_id] = new_count

    #     # 2. Check for "Zombie" vars (Reduction in Diversity)
    #     # If we lowered diversity (e.g. 5 -> 3), we must ensure vars 3 and 4 are removed.
    #     # This might trigger a cutoff if one wasn't triggered yet.
        
    #     # Check if any old vars exist beyond our new 'end_var'
    #     existing_vars = [k for k in self.fake_splunk_state[time_range][log_type_key].keys() if isinstance(k, int)]
    #     if existing_vars:
    #         max_old_var = max(existing_vars)
            
    #         if max_old_var >= end_var:
    #             # We have leftover vars to delete.
    #             # If we haven't already wiped from a lower point, wipe from end_var onwards.
    #             if cutoff_var_id is None:
    #                 cutoff_var_id = end_var
                
    #             # Update state to remove zombies
    #             for old_id in range(end_var, max_old_var + 1):
    #                 if old_id in self.fake_splunk_state[time_range][log_type_key]:
    #                     self.fake_splunk_state[time_range][log_type_key][old_id] = 0

    #     # 3. Apply the Deletion Command
    #     if cutoff_var_id is not None:
    #         # Overwrite the dictionary with the flag string as requested
    #         self.logs_to_delete[time_range][log_type_key] = f"RemoveAll_{cutoff_var_id}"

    #     logger.debug(f"Prepared variation assignments with cutoff at var_id={cutoff_var_id}")
    #     logger.debug(f"New Counts: {self.fake_splunk_state[time_range][log_type_key]}")

        
    #     # --- (Shuffle, Template Loading, Task Generation - Same as original) ---
    #     random.shuffle(variation_assignments)
        
    #     template_loading_start = time.time()
    #     templates = {}
    #     for var_id in range(start_var, end_var):
    #         if var_id == is_trigger_int:
    #             templates[var_id] = base_template
    #         else:
    #             cache_key = f"{logsource.lower()}_{eventcode}_{is_trigger_int}_{var_id}"
    #             with open(os.path.join(self.cache_dir, f"{cache_key}.template"), 'r') as f:
    #                 templates[var_id] = f.read()
        
    #     real_ts = datetime.now().strftime("%m_%d_%Y_%I_%M")
    #     generation_tasks = [
    #         (templates[var_id], random.choice(timestamp_pool), var_id, self.field_pattern, self.time_pattern, logsource, real_ts) 
    #         for var_id in variation_assignments
    #     ]

    #     return generation_tasks
    
    def prepare_tasks(self, logsource, eventcode, istrigger, time_range, num_logs, diversity=1, injection_id=0, max_workers=12):
        
        # --- 1. Time Parsing ---
        if isinstance(time_range[0], str):
            start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
            end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
        else:
            start_date, end_date = time_range

        # --- 2. Timestamp Pool Generation ---
        # Pre-generate a pool of random timestamps within the range for speed
        timestamp_pool_size = min(1000, num_logs * 2)
        timestamp_pool = self._generate_timestamp_pool(start_date, end_date, timestamp_pool_size)
        
        # --- 3. Template Preparation ---
        base_template = self._prepare_base_template(logsource, eventcode, istrigger)
        is_trigger_int = int(istrigger)
        
        start_var = is_trigger_int
        end_var = diversity + is_trigger_int
        
        # Check cache for missing templates and generate them if needed
        missing_templates = []
        for var_id in range(start_var, end_var):
            cache_key = f"{logsource.lower()}_{eventcode}_{is_trigger_int}_{var_id}"
            if not os.path.exists(os.path.join(self.cache_dir, f"{cache_key}.template")):
                missing_templates.append(var_id)
                
        if missing_templates:
            for var_id in missing_templates:
                self._get_variation_template(logsource, eventcode, istrigger, var_id, base_template)

        # --- 4. Stateless Distribution Calculation ---
        # Simply divide the logs evenly among the requested diversity
        logs_per_variation = num_logs // diversity
        remaining_logs = num_logs % diversity
        
        variation_assignments = []
        
        for i in range(diversity):
            var_id = start_var + i
            # Add 1 extra log to the first few variations if there's a remainder (to hit exact num_logs)
            count = logs_per_variation + (1 if i < remaining_logs else 0)
            variation_assignments.extend([var_id] * count)

        random.shuffle(variation_assignments)
        
        # --- 5. Load Templates & Build Tasks ---
        templates = {}
        for var_id in range(start_var, end_var):
            if var_id == is_trigger_int:
                templates[var_id] = base_template
            else:
                cache_key = f"{logsource.lower()}_{eventcode}_{is_trigger_int}_{var_id}"
                with open(os.path.join(self.cache_dir, f"{cache_key}.template"), 'r') as f:
                    templates[var_id] = f.read()
        
        real_ts = datetime.now().strftime("%m_%d_%Y_%I_%M")
        generation_tasks = [
            (templates[var_id], random.choice(timestamp_pool), var_id, self.field_pattern, self.time_pattern, logsource, real_ts) 
            for var_id in variation_assignments
        ]

        return generation_tasks
    
    def generate_massive_stream(self, all_configs, batch_size=50000):
        task_buffer = []
        for config in all_configs:
            new_tasks = self.prepare_tasks(**config)
            task_buffer.extend(new_tasks)
            
            while len(task_buffer) >= batch_size:
                current_batch = task_buffer[:batch_size]
                task_buffer = task_buffer[batch_size:]
                
                # CHANGE: Cast to list and yield the whole chunk
                # This forces the workers to finish this batch before moving on
                yield list(self._process_batch(current_batch))
        
        # Flush remaining
        if task_buffer:
            yield list(self._process_batch(task_buffer))

    def _process_batch(self, tasks):
        # Check if tasks is empty to avoid errors
        if not tasks:
            return []
            
        # Use ProcessPool for CPU bound regex
        # Chunksize optimization
        chunk = max(1, len(tasks) // (os.cpu_count() * 4))
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Calls the global function, not a nested one
            return executor.map(_worker_generate_log, tasks, chunksize=chunk)
            

    
    @lru_cache(maxsize=32)
    def _prepare_base_template(self, logsource, eventcode, istrigger):
        """Get and prepare a base template with caching for efficiency"""
        start_time = time.time()
        
        template = self._get_template(logsource, eventcode, istrigger)
        
        # Add fake flag
        template_start = time.time()
        # result = self.field_pattern.sub(r'\g<0>\nis_fake=1', template)
        result = template
        # logger.info(f"Base template modification completed in {time.time() - template_start:.3f} seconds")
        # logger.info(f"Total base template preparation completed in {time.time() - start_time:.3f} seconds")
        
        return result
    
    def _prepare_variation_template(self, logsource, eventcode, istrigger, variation_id):
        """Prepare a template for a specific variation"""
        base_template = self._prepare_base_template(logsource, eventcode, istrigger)
        
        if variation_id > 0:
            return self._apply_variations(base_template, logsource, eventcode, variation_id)
        return base_template
    
