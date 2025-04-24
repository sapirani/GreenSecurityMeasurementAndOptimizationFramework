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


class LogGenerator:
    
    def __init__(self, logtypes, splunk_tools_instance, big_replacement_dicts=None, cache_dir='./template_cache'):
        logger.info("Initializing OptimizedLogGenerator")
        start_time = time.time()
        
        self.big_replacement_dicts = big_replacement_dicts
        self.splunk_tools = splunk_tools_instance
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
        
        logger.info(f"LogGenerator initialization completed in {time.time() - start_time:.3f} seconds")
        
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
                'ComputerName': 'sharehost-{}.lab.internal',
                'Account Name': 'user_{}',
                'Account Domain': 'LABDOM{}',
                'Workstation Name': 'CLIENT-{}',
                'Source Network Address': '10.0.0.{}',
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

    def _get_template(self, logsource, eventcode, istrigger):
        """Lazy-loading templates to avoid loading everything at once"""
        key = f"{logsource.lower()}_{eventcode}_{istrigger}"
        
        if key not in self.logs_to_duplicate_dict:
            # Load just this template
            logger.info(f"Loading template for {key}")
            start_time = time.time()
            templates = self.splunk_tools.load_logs_to_duplicate_dict([(logsource, eventcode, istrigger)])
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
    
    def _get_cached_variation_path(self, logsource, eventcode, istrigger, variation_id):
        """Get the path to a cached variation template"""
        return os.path.join(self.cache_dir, f"{logsource.lower()}_{eventcode}_{istrigger}_{variation_id}.template")
    
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
        return self.field_pattern.sub(r'\g<0>\nis_fake=1', log)
    
    def _prepare_base_template(self, logsource, eventcode, istrigger):
        """Get and prepare a base template with caching for efficiency"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{logsource.lower()}_{eventcode}_{istrigger}_base"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.template")
        
        if os.path.exists(cache_path):
            # logger.info(f"Loading base template from cache: {cache_key}")
            with open(cache_path, 'r') as f:
                result = f.read()
        else:
            # logger.info(f"Creating base template for {cache_key}")
            template = self._get_template(logsource, eventcode, istrigger)
            
            # Add fake flag
            result = self.field_pattern.sub(r'\g<0>\nis_fake=1', template)
            
            # Cache the result
            with open(cache_path, 'w') as f:
                f.write(result)
        
        # logger.info(f"Base template preparation completed in {time.time() - start_time:.3f} seconds")
        return result
    
    def prepare_variation_templates(self, logsource, eventcode, istrigger, max_variations=100):
        """Pre-prepare and cache all variation templates for a log type"""
        # logger.info(f"Preparing variation templates for {logsource}:{eventcode}:{istrigger} (max={max_variations})")
        start_time = time.time()
        
        # Get the base template
        base_template = self._prepare_base_template(logsource, eventcode, istrigger)
        
        # Create and cache variations
        if (logsource, eventcode) in self.variations:
            for var_id in range(1, max_variations + 1):
                self._get_variation_template(logsource, eventcode, istrigger, var_id, base_template)
                
        # logger.info(f"Prepared {max_variations} variation templates in {time.time() - start_time:.3f} seconds")
    
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
    
    def generate_logs(self, logsource, eventcode, istrigger, time_range, num_logs, diversity=0, max_workers=12):
        """Generate logs with optimized performance using pre-cached templates"""
        # logger.info(f"Generating {num_logs} logs with diversity={diversity}, max_workers={max_workers}")
        total_start_time = time.time()
        
        # Ensure diversity is at least 1 and no more than num_logs
        diversity = max(1, min(diversity + 1, num_logs))
        
        # Parse time range once
        time_parsing_start = time.time()
        if isinstance(time_range[0], str):
            start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
            end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
        else:
            start_date, end_date = time_range
        # logger.info(f"Time parsing completed in {time.time() - time_parsing_start:.3f} seconds")
        
        # Pre-generate timestamp pool for better performance
        pool_start = time.time()
        timestamp_pool_size = min(1000, num_logs * 2)  # Create some extra timestamps for randomness
        timestamp_pool = self._generate_timestamp_pool(start_date, end_date, timestamp_pool_size)
        # logger.info(f"Timestamp pool generation ({timestamp_pool_size} timestamps) completed in {time.time() - pool_start:.3f} seconds")
        
        # Prepare templates if they don't exist
        templates_start = time.time()
        # Prepare base template
        base_template = self._prepare_base_template(logsource, eventcode, istrigger)
        
        # Check if we need to prepare variations
        missing_templates = []
        for var_id in range(1, diversity):
            cache_key = f"{logsource.lower()}_{eventcode}_{istrigger}_{var_id}"
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.template")
            if not os.path.exists(cache_path):
                missing_templates.append(var_id)
        
        # Prepare missing templates if needed
        if missing_templates:
            logger.info(f"Preparing {len(missing_templates)} missing variation templates")
            for var_id in missing_templates:
                self._get_variation_template(logsource, eventcode, istrigger, var_id, base_template)
        
        # logger.info(f"Template preparation completed in {time.time() - templates_start:.3f} seconds")
        
        # Calculate logs per variation
        distribution_start = time.time()
        logs_per_variation = num_logs // diversity
        remaining_logs = num_logs % diversity
        
        # Create a list of variation IDs based on distribution
        variation_assignments = []
        for var_id in range(diversity):
            count = logs_per_variation + (1 if var_id < remaining_logs else 0)
            variation_assignments.extend([var_id] * count)
        
        # Shuffle to distribute variations evenly
        random.shuffle(variation_assignments)
        # logger.info(f"Task distribution calculation completed in {time.time() - distribution_start:.3f} seconds")
        
        # Load all templates we need
        template_loading_start = time.time()
        templates = {}
        for var_id in range(diversity):
            if var_id == 0:
                templates[var_id] = base_template
            else:
                cache_key = f"{logsource.lower()}_{eventcode}_{istrigger}_{var_id}"
                cache_path = os.path.join(self.cache_dir, f"{cache_key}.template")
                with open(cache_path, 'r') as f:
                    templates[var_id] = f.read()
        logger.info(f"Template loading completed in {time.time() - template_loading_start:.3f} seconds")
        
        # Determine optimal batch size based on num_logs
        batch_size = max(100, min(1000, num_logs // (max_workers * 2)))
        
        # Create tasks with templates and timestamps
        tasks_start = time.time()
        generation_tasks = [
            (templates[var_id], random.choice(timestamp_pool)) 
            for var_id in variation_assignments
        ]
        # logger.info(f"Task generation completed in {time.time() - tasks_start:.3f} seconds")
        
        # Function to apply timestamp to a template
        def apply_timestamp(task):
            template, timestamp = task
            return self.time_pattern.sub(timestamp, template)
        
        # # Generate logs in batches using a single thread pool
        # all_logs = []
        # batch_count = 0
        
        # generation_start = time.time()
        # for i in range(0, len(generation_tasks), batch_size):
        #     batch_start = time.time()
        #     batch = generation_tasks[i:i+batch_size]
        return list(map(apply_timestamp, generation_tasks))
    
    def generate_log(self, logsource, eventcode, istrigger, time_range, variation_id=None):
        """Generate a single log (optimized version)"""
        # Get the template
        template = self._prepare_base_template(logsource, eventcode, istrigger)
        
        # Apply variation if needed
        if variation_id is not None:
            template = self._apply_variations(template, logsource, eventcode, variation_id)
        
        # Generate time
        if isinstance(time_range[0], str):
            start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
            end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
        else:
            start_date, end_date = time_range
        
        time_delta = end_date - start_date
        total_seconds = time_delta.days * 86400 + time_delta.seconds
        random_seconds = random.randint(0, total_seconds)
        time = start_date + timedelta(seconds=random_seconds)
        time_str = time.strftime("%m/%d/%Y %I:%M:%S %p")
        
        # Apply timestamp
        return self.time_pattern.sub(time_str, template)

    def _get_template(self, logsource, eventcode, istrigger):
        """Lazy-loading templates to avoid loading everything at once"""
        key = f"{logsource.lower()}_{eventcode}_{istrigger}"
        
        if key not in self.logs_to_duplicate_dict:
            # Load just this template
            logger.info(f"Loading template for {key}")
            start_time = time.time()
            templates = self.splunk_tools.load_logs_to_duplicate_dict([(logsource, eventcode, istrigger)])
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
    
    # def _apply_variations(self, log, logsource, eventcode, variation_id):
    #     """Apply variations to a log template using more efficient string operations"""
    #     start_time = time.time()
        
    #     if (logsource, eventcode) not in self.variations:
    #         return log
        
    #     log_variations = self.variations[(logsource, eventcode)]
    #     field_count = 0
    #     regex_time = 0
        
    #     for field, template in log_variations.items():
    #         field_start = time.time()
    #         field_count += 1
            
    #         # Format the value with the variation ID
    #         value = template.format(variation_id)
    #         # Escape backslashes
    #         escaped_value = value.replace('\\', '\\\\')
            
    #         # Determine pattern and replacement based on field type
    #         if field == 'Message':
    #             pattern = self.field_patterns['Message'][0]
    #             replacement = self.field_patterns['Message'][1].format(escaped_value)
    #             log = pattern.sub(replacement, log)
    #         elif field in ['Security ID', 'SecurityID']:
    #             pattern = self.field_patterns.get(field, (None, None))[0]
    #             if pattern:
    #                 replacement = self.field_patterns.get(field, (None, None))[1].format(escaped_value)
    #                 log = pattern.sub(replacement, log)
    #         else:
    #             regex_start = time.time()
    #             # First try with equals sign format
    #             field_pattern = r'{}=[^\n]+'.format(re.escape(field))
    #             if re.search(field_pattern, log):
    #                 log = re.sub(field_pattern, f'{field}={escaped_value}', log)
    #             else:
    #                 # Then try with colon format
    #                 field_pattern = r'{}:\s*[^\n]+'.format(re.escape(field))
    #                 if re.search(field_pattern, log):
    #                     log = re.sub(field_pattern, f'{field}:\t{escaped_value}', log)
    #             regex_time += time.time() - regex_start
        
    #     total_time = time.time() - start_time
    #     if total_time > 0.01:  # Only log if it took more than 10ms
    #         logger.info(f"Applied {field_count} variations in {total_time:.3f}s (regex: {regex_time:.3f}s) for {logsource}:{eventcode}:{variation_id}")
        
    #     return log
    
    @lru_cache(maxsize=32)
    def _prepare_base_template(self, logsource, eventcode, istrigger):
        """Get and prepare a base template with caching for efficiency"""
        start_time = time.time()
        
        template = self._get_template(logsource, eventcode, istrigger)
        
        # Add fake flag
        template_start = time.time()
        result = self.field_pattern.sub(r'\g<0>\nis_fake=1', template)
        
        # logger.info(f"Base template modification completed in {time.time() - template_start:.3f} seconds")
        # logger.info(f"Total base template preparation completed in {time.time() - start_time:.3f} seconds")
        
        return result
    
    def _prepare_variation_template(self, logsource, eventcode, istrigger, variation_id):
        """Prepare a template for a specific variation"""
        base_template = self._prepare_base_template(logsource, eventcode, istrigger)
        
        if variation_id > 0:
            return self._apply_variations(base_template, logsource, eventcode, variation_id)
        return base_template
    
    # def generate_logs(self, logsource, eventcode, istrigger, time_range, num_logs, diversity=0, max_workers=12):
    #     """Generate logs with optimized performance, creating variations on-demand"""
    #     # logger.info(f"Generating {num_logs} logs with diversity={diversity}, max_workers={max_workers}")
    #     total_start_time = time.time()
        
    #     # Ensure diversity is at least 1 and no more than num_logs
    #     diversity = max(1, min(diversity + 1, num_logs))
        
    #     # Parse time range once
    #     time_parsing_start = time.time()
    #     if isinstance(time_range[0], str):
    #         start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
    #         end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
    #     else:
    #         start_date, end_date = time_range
    #     # logger.info(f"Time parsing completed in {time.time() - time_parsing_start:.3f} seconds")
        
    #     # Pre-generate timestamp pool for better performance
    #     pool_start = time.time()
    #     timestamp_pool_size = min(1000, num_logs * 2)  # Create some extra timestamps for randomness
    #     timestamp_pool = self._generate_timestamp_pool(start_date, end_date, timestamp_pool_size)
    #     # logger.info(f"Timestamp pool generation ({timestamp_pool_size} timestamps) completed in {time.time() - pool_start:.3f} seconds")
        
    #     # Get the base template once
    #     template_start = time.time()
    #     base_template = self._prepare_base_template(logsource, eventcode, istrigger)
    #     # logger.info(f"Base template preparation completed in {time.time() - template_start:.3f} seconds")
        
    #     # Calculate logs per variation
    #     distribution_start = time.time()
    #     logs_per_variation = num_logs // diversity
    #     remaining_logs = num_logs % diversity
        
    #     # Create a list of variation IDs based on distribution
    #     variation_assignments = []
    #     for var_id in range(diversity):
    #         count = logs_per_variation + (1 if var_id < remaining_logs else 0)
    #         variation_assignments.extend([var_id] * count)
        
    #     # Shuffle to distribute variations evenly
    #     random.shuffle(variation_assignments)
    #     # logger.info(f"Task distribution calculation completed in {time.time() - distribution_start:.3f} seconds")
        
    #     # Determine optimal batch size based on num_logs
    #     batch_size = max(100, min(1000, num_logs // (max_workers * 2)))
        
    #     # Create tasks with variation IDs instead of pre-generated templates
    #     tasks_start = time.time()
    #     generation_tasks = [
    #         (var_id, random.choice(timestamp_pool)) 
    #         for var_id in variation_assignments
    #     ]
    #     logger.info(f"Task generation completed in {time.time() - tasks_start:.3f} seconds")
        
    #     # Function to generate a single log with variation
    #     def generate_log_with_variation(task):
    #         var_id, timestamp = task
            
    #         # Apply variation on-demand
    #         if var_id > 0:
    #             template = self._apply_variations(base_template, logsource, eventcode, var_id)
    #         else:
    #             template = base_template
                
    #         # Apply timestamp
    #         return self.time_pattern.sub(timestamp, template)
        
    #     # Generate logs in batches using a single thread pool
    #     all_logs = []
    #     batch_count = 0
        
    #     generation_start = time.time()
    #     for i in range(0, len(generation_tasks), batch_size):
    #         batch_start = time.time()
    #         batch = generation_tasks[i:i+batch_size]
    #         batch_count += 1
            
    #         # logger.info(f"Processing batch {batch_count} ({len(batch)} tasks)")
            
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             batch_results = list(executor.map(generate_log_with_variation, batch))
    #             all_logs.extend(batch_results)
            
    #         logger.info(f"Batch {batch_count} completed in {time.time() - batch_start:.3f} seconds")
        
    #     # logger.info(f"Log generation completed in {time.time() - generation_start:.3f} seconds")
    #     # logger.info(f"Total generation time for {num_logs} logs: {time.time() - total_start_time:.3f} seconds")
        
    #     return all_logs
    
    # def generate_log(self, logsource, eventcode, istrigger, time_range, variation_id=None):
    #     """Generate a single log (optimized version)"""
    #     # Get the template
    #     template = self._prepare_base_template(logsource, eventcode, istrigger)
        
    #     # Apply variation if needed
    #     if variation_id is not None:
    #         template = self._apply_variations(template, logsource, eventcode, variation_id)
        
    #     # Generate time
    #     if isinstance(time_range[0], str):
    #         start_date = datetime.strptime(time_range[0], '%m/%d/%Y:%H:%M:%S')
    #         end_date = datetime.strptime(time_range[1], '%m/%d/%Y:%H:%M:%S')
    #     else:
    #         start_date, end_date = time_range
        
    #     time_delta = end_date - start_date
    #     total_seconds = time_delta.days * 86400 + time_delta.seconds
    #     random_seconds = random.randint(0, total_seconds)
    #     time = start_date + timedelta(seconds=random_seconds)
    #     time_str = time.strftime("%m/%d/%Y %I:%M:%S %p")
        
    #     # Apply timestamp
    #     return self.time_pattern.sub(time_str, template)