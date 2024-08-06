
from datetime import timedelta
import datetime
import random
import re
from splunk_tools import SplunkTools
import json
# plot a px chart of duration vs. percentage with absolute volume as color
import plotly.express as px
import pandas as pd

rules_path = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/energy_profile_resources/rules.json"
with open(rules_path, 'r') as f:
    rules = json.load(f)


splunk_tools = SplunkTools(rules.keys())

def generate_fake_time(start_date, end_date):
    # convert timestamps to datetime objects
    start_date = datetime.datetime.fromtimestamp(start_date)
    end_date = datetime.datetime.fromtimestamp(end_date)
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

def generate_logs(log, time_range, num_logs):
    logs = []
    for i in range(num_logs):
        fake_time = generate_fake_time(time_range[0], time_range[1])
        log = re.sub(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", fake_time.strftime("%m/%d/%Y %I:%M:%S %p"), log, flags=re.MULTILINE)
        logs.append(log)
    return logs
    
def run_rule(rule, time_range):
    results, rule_duration = splunk_tools.query_splunk(rule, *time_range)
    return len(results), rule_duration

def get_total_log_volume(time_range):
    results, rule_duration = splunk_tools.query_splunk('index=main', *time_range)
    return len(results)

def delete_fake_logs(logsource):
    results, rule_duration = splunk_tools.query_splunk('index=main host="dt-splunk" | delete', *time_range)
    with open (f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/{logsource}.txt", "w") as f:
        f.write('')
        
def get_buckets_time_range(time_range, bucket_span):
    start_date = datetime.datetime.fromtimestamp(time_range[0])
    end_date = datetime.datetime.fromtimestamp(time_range[1])
    time_delta = end_date - start_date
    bucket_span_seconds = float(bucket_span.split('m')[0])*60
    bucket_span = datetime.timedelta(seconds=bucket_span_seconds)
    bucket_tuples = []
    for i in range(round(time_delta.seconds*1000000/(bucket_span_seconds*1000000))):
        bucket_start = start_date + i*bucket_span
        bucket_end = bucket_start + bucket_span
        bucket_tuples.append((bucket_start, bucket_end))
    return bucket_tuples

def get_random_computer_names(amount):
    computer_names = []
    for i in range(amount):
        computer_names.append(f'computer_{random.randint(1, 1000)}')
    return computer_names

def increase_targeted_log(log, logsource, time_range, amount, threshold=1, threshold_field=None, bucket_span=None, is_triggered=True):
    if threshold_field and bucket_span:
        logs = []
        if not is_triggered:
            threshold -= 1
        
        bucket_tuples = get_buckets_time_range(time_range, bucket_span)
        for i in range(len(bucket_tuples)):
            computer_names = get_random_computer_names(threshold)
            bucket_start, bucket_end = bucket_tuples[i]   
            time_delta = bucket_end - bucket_start
            sub_bucket_span = str(time_delta.seconds/(60*threshold)) + 'm'     
            sub_bucket_tuples = get_buckets_time_range((bucket_start.timestamp(), bucket_end.timestamp()), sub_bucket_span)
            for j in range(len(sub_bucket_tuples)):
                sub_bucket_start, sub_bucket_end = sub_bucket_tuples[j]
                fake_time = generate_fake_time(sub_bucket_start.timestamp(), sub_bucket_end.timestamp())
                log = re.sub(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", fake_time.strftime("%m/%d/%Y %I:%M:%S %p"), log, flags=re.MULTILINE)
                log = re.sub(r"ComputerName=[a-zA-Z0-9_]+", f'ComputerName={computer_names[j]}', log)
                logs.append(log)
    else:
        logs = generate_logs(log, time_range, amount)
    splunk_tools.write_logs_to_monitor(logs, logsource)
    
def decrease_targeted_log(rule, logsource, time_range, amount):
    results, rule_duration = splunk_tools.query_splunk(f'{rule} host="dt-splunk" | head {amount}', *time_range)
    with open (f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/{logsource}.txt", "w") as f:
        f.write('')
        
time_windows = list([0.5, 1, 3,6,12,18,24]) # hours
current_time = datetime.datetime.strptime('2024-07-13 17:00:00', '%Y-%m-%d %H:%M:%S')
for rule in ["Windows Rapid Authentication On Multiple Hosts"]: #rules:
    volumes = []
    percentages = []
    rule_durations = []
    is_triggered_list = []
    alerts = []
    query = rules[rule]['query']
    events_query = query.split('| stats')[0]
    for is_triggered in [True, False]:
        if is_triggered or ('bucket_span' in rules[rule]):
            log = rules[rule]['trigger_log']
        else:
            if 'non_trigger_log' not in rules[rule]:
                continue
            log = rules[rule]['non_trigger_log']
            non_trigger_query = rules[rule]['non_trigger_query']            
        for time_window in time_windows:
            print(f'Running rule {rule} for time window {time_window} hours')
            time_range = (current_time - timedelta(hours=time_window), current_time)
            time_range = tuple([time.timestamp() for time in time_range])
            delete_fake_logs(rules[rule]['logsource'].lower())
            
            total_log_volume = get_total_log_volume(time_range)
            p_c = 0
            i = 0
            if 'bucket_span' in rules[rule]:
                range_pc = range(49,50)
            else:
                range_pc = range(0, 50, 12)
            for pc in range_pc:  
                add_pc = int( pc  - 100*p_c)
                if add_pc > 0:
                    amount = int(total_log_volume * add_pc / 100)    
                    print(f'{rule}: Adding {amount} logs to the targeted log')     
                    increase_targeted_log(log, rules[rule]['logsource'].lower(), time_range, amount, threshold=rules[rule]['threshold'], threshold_field=rules[rule]['threshold_field'], bucket_span=rules[rule]['bucket_span'], is_triggered=is_triggered)
                elif add_pc < 0:
                    amount = int(total_log_volume * abs(add_pc) / 100)
                    print(f'{rule}: Removing {amount} logs from the targeted log')
                    if is_triggered:
                        decrease_targeted_log(events_query, rules[rule]['logsource'].lower(), time_range, amount)
                    else:
                        decrease_targeted_log(non_trigger_query, rules[rule]['logsource'].lower(), time_range, amount)
                volumes.append(total_log_volume)
                
                current_rule_duration = datetime.timedelta(0)
                for i in range(5):
                    targeted_log_volume, rule_duration = run_rule(events_query, time_range)
                    current_rule_duration += rule_duration
                if not is_triggered and 'bucket_span' not in rules[rule]:
                    targeted_log_volume, rule_duration = run_rule(non_trigger_query, time_range) # in case of non trigger log check its amount
                num_of_alerts, _ = run_rule(query, time_range) # any way we need to check the amount of alerts
                alerts.append(num_of_alerts)
                rule_duration = current_rule_duration / 5   
                p_c = targeted_log_volume / total_log_volume
                percentages.append(p_c*100)
                rule_durations.append(rule_duration)
                is_triggered_list.append(is_triggered)
            print(f'deleting fake logs for rule {rule} and time window {time_window} hours')
            delete_fake_logs(rules[rule]['logsource'].lower())
            




    df = pd.DataFrame({'duration': rule_durations, 'percentage': percentages, 'volume': volumes, 'is_triggered': is_triggered_list, 'alerts': alerts})
    df.to_csv(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/energy_profile_resources/duration_vs_percentage_{rule}.csv')
    # rule = "Windows Rapid Authentication On Multiple Hosts"
    # df = pd.read_csv(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/energy_profile_resources/duration_vs_percentage_{rule}.csv')
    fig = px.scatter(df, y='duration', x='percentage', color='volume', title=f'Duration vs. Percentage for rule {rule}', facet_col='is_triggered')
    # save the plot to a file
    fig.write_html(f'duration_vs_percentage_{rule}.html')


# '''
# for each rule in rules:
#     for each time_range in time_ranges:
#         # get the current percentage of the targeted log from the total log volume as p_c
#         targeted_log_volume = get_targeted_log_volume(rule, time_range)
#         total_log_volume = get_total_log_volume(time_range)
#         p_c = targeted_log_volume / total_log_volume
#         percentages = range(p_c, 50, 5)
#         for each percentage in percentages:
#             # add logs to the targeted log until the percentage of the targeted log reaches the percentage
#             increase_targeted_log(rule, time_range, percentage)
#             # measure the duration of the rule execution as t
#             t = get_rule_duration(rule, time_range)
# plot a px chart of duration vs. percentage with absolute volume as color

# '''
# from datetime import timedelta
# import datetime
# import random
# import re
# from splunk_tools import SplunkTools
# import json
# # plot a px chart of duration vs. percentage with absolute volume as color
# import plotly.express as px
# import pandas as pd

# rules_path = "/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/energy_profile_resources/rules.json"
# with open(rules_path, 'r') as f:
#     rules = json.load(f)


# splunk_tools = SplunkTools(rules.keys())

# def generate_fake_time(start_date, end_date):
#     # convert timestamps to datetime objects
#     start_date = datetime.datetime.fromtimestamp(start_date)
#     end_date = datetime.datetime.fromtimestamp(end_date)
#     time_delta = end_date - start_date
#     random_days = random.randint(0, time_delta.days)
#     random_seconds = random.randint(0, time_delta.seconds-1)
#     # random_microseconds = random.randint(0, time_delta.microseconds)

#     random_date_time = start_date + timedelta(
#         days=random_days,
#         seconds=random_seconds,
#         # microseconds=random_microseconds
#     )

#     return random_date_time

# def generate_logs(log, time_range, num_logs):
#     logs = []
#     for i in range(num_logs):
#         fake_time = generate_fake_time(time_range[0], time_range[1])
#         log = re.sub(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} [APM]{2}", fake_time.strftime("%m/%d/%Y %I:%M:%S %p"), log, flags=re.MULTILINE)
#         logs.append(log)
#     return logs
    
# def run_rule(rule, time_range):
#     results, rule_duration = splunk_tools.query_splunk(rule, *time_range)
#     return len(results), rule_duration

# def get_total_log_volume(time_range):
#     results, rule_duration = splunk_tools.query_splunk('index=main', *time_range)
#     return len(results)

# def delete_fake_logs(logsource):
#     results, rule_duration = splunk_tools.query_splunk('index=main host="dt-splunk" | delete', *time_range)
#     with open (f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/{logsource}.txt", "w") as f:
#         f.write('')

# def increase_targeted_log(log, logsource, time_range, amount):
#     logs = generate_logs(log, time_range, amount)
#     splunk_tools.write_logs_to_monitor(logs, logsource)
    
# def decrease_targeted_log(rule, logsource, time_range, amount):
#     results, rule_duration = splunk_tools.query_splunk(f'{rule} host="dt-splunk" | head {amount}', *time_range)
#     with open (f"/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/monitor_files/{logsource}.txt", "w") as f:
#         f.write('')
        
# time_windows = list(range(1,25,5)) # hours
# current_time = datetime.datetime.strptime('2024-07-13 17:00:00', '%Y-%m-%d %H:%M:%S')
# for rule in rules:
#     volumes = []
#     percentages = []
#     rule_durations = []
#     is_triggered_list = []
#     query = rules[rule]['query']
#     query = query.split('| stats')[0]
#     for is_triggered in [True, False]:
#         if is_triggered:
#             log = rules[rule]['trigger_log']
#         else:
#             if 'non_trigger_log' not in rules[rule]:
#                 continue
#             log = rules[rule]['non_trigger_log']
#             non_trigger_query = rules[rule]['non_trigger_query']            
#         for time_window in time_windows:
#             print(f'Running rule {rule} for time window {time_window} hours')
#             time_range = (current_time - timedelta(hours=time_window), current_time)
#             time_range = tuple([time.timestamp() for time in time_range])
#             delete_fake_logs(rules[rule]['logsource'].lower())
            
#             total_log_volume = get_total_log_volume(time_range)
#             p_c = 0
#             i = 0
#             for pc in range(0, 50, 10):  
#                 add_pc = int( pc  - 100*p_c)
#                 if add_pc > 0:
#                     amount = int(total_log_volume * add_pc / 100)    
#                     print(f'{rule}: Adding {amount} logs to the targeted log')     
#                     increase_targeted_log(log, rules[rule]['logsource'].lower(), time_range, amount)
#                 elif add_pc < 0:
#                     amount = int(total_log_volume * abs(add_pc) / 100)
#                     print(f'{rule}: Removing {amount} logs from the targeted log')
#                     if is_triggered:
#                         decrease_targeted_log(query, rules[rule]['logsource'].lower(), time_range, amount)
#                     else:
#                         decrease_targeted_log(non_trigger_query, rules[rule]['logsource'].lower(), time_range, amount)
#                 volumes.append(total_log_volume)
                
#                 current_rule_duration = datetime.timedelta(0)
#                 for i in range(5):
#                     targeted_log_volume, rule_duration = run_rule(query, time_range)
#                     current_rule_duration += rule_duration
#                 if not is_triggered:
#                     targeted_log_volume, rule_duration = run_rule(non_trigger_query, time_range)
#                 rule_duration = current_rule_duration / 5   
#                 p_c = targeted_log_volume / total_log_volume
#                 percentages.append(p_c*100)
#                 rule_durations.append(rule_duration)
#                 is_triggered_list.append(is_triggered)
#             print(f'deleting fake logs for rule {rule} and time window {time_window} hours')
#             delete_fake_logs(rules[rule]['logsource'].lower())
            




#     df = pd.DataFrame({'duration': rule_durations, 'percentage': percentages, 'volume': volumes, 'is_triggered': is_triggered_list, 'time_window': time_window})
#     fig = px.scatter(df, y='duration', x='percentage', color='volume', title=f'Duration vs. Percentage for rule {rule}', facet_col='is_triggered')

#     # save the plot to a file
#     fig.write_html(f'duration_vs_percentage_{rule}.html')
    
#     df.to_csv(f'/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/energy_profile_resources/duration_vs_percentage_{rule}.csv')