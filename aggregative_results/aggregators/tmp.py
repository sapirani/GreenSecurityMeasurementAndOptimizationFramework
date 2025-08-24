from datetime import datetime
from time import sleep

from aggregative_results.DTOs.raw_results_dtos.iteration_info import IterationMetadata
from aggregative_results.DTOs.raw_results_dtos.process_raw_results import ProcessRawResults
from aggregative_results.DTOs.raw_results_dtos.system_process_raw_results import ProcessSystemRawResults
from aggregative_results.DTOs.raw_results_dtos.system_raw_results import SystemRawResults
from aggregative_results.aggregators.energy_model_aggregator import EnergyModelAggregator

if __name__ == "__main__":
    start_time = datetime.now()
    process_results_1 = ProcessRawResults(pid=1, process_name="session1",
                                          arguments=None,
                                          cpu_percent_sum_across_cores=40,
                                          cpu_percent_mean_across_cores=20,
                                          threads_num=1,
                                          used_memory_mb=2000,
                                          used_memory_percent=5,
                                          disk_read_count=1030,
                                          disk_write_count=1050,
                                          disk_read_kb=2000,
                                          disk_write_kb=2000,
                                          page_faults=8,
                                          network_kb_sent=3000,
                                          packets_sent=300,
                                          network_kb_received=3000,
                                          packets_received=300,
                                          process_of_interest=True)

    system_results_1 = SystemRawResults(cpu_percent_mean_across_cores=80,
                                        cpu_percent_sum_across_cores=160,
                                        number_of_cores=2,
                                        total_memory_gb=5,
                                        total_memory_percent=20,
                                        disk_read_count=6000,
                                        disk_write_count=6000,
                                        disk_read_kb=2000,
                                        disk_write_kb=2000,
                                        disk_read_time=50,
                                        disk_write_time=60,
                                        packets_sent=90,
                                        packets_received=70,
                                        network_kb_sent=20.5,
                                        network_kb_received=80,
                                        battery_percent=None,
                                        battery_remaining_capacity_mWh=None,
                                        battery_voltage_mV=None,
                                        core_percents=[80,80])

    iteration_1 = IterationMetadata(timestamp=datetime.now(), start_date=start_time,
                                    hostname="first", session_id="session1")
    raw_results_1 = ProcessSystemRawResults(process_results_1, system_results_1)

    process_results_2 = ProcessRawResults(pid=1, process_name="session1",
                                          arguments=None,
                                          cpu_percent_sum_across_cores=50,
                                          cpu_percent_mean_across_cores=25,
                                          threads_num=1,
                                          used_memory_mb=2000,
                                          used_memory_percent=5,
                                          disk_read_count=1030,
                                          disk_write_count=1050,
                                          disk_read_kb=2000,
                                          disk_write_kb=2000,
                                          page_faults=8,
                                          network_kb_sent=3000,
                                          packets_sent=300,
                                          network_kb_received=3000,
                                          packets_received=300,
                                          process_of_interest=True)

    system_results_2 = SystemRawResults(cpu_percent_mean_across_cores=50,
                                        cpu_percent_sum_across_cores=100,
                                        number_of_cores=2,
                                        total_memory_gb=8,
                                        total_memory_percent=30,
                                        disk_read_count=6000,
                                        disk_write_count=6000,
                                        disk_read_kb=2000,
                                        disk_write_kb=2000,
                                        disk_read_time=50,
                                        disk_write_time=60,
                                        packets_sent=90,
                                        packets_received=70,
                                        network_kb_sent=20.5,
                                        network_kb_received=80,
                                        battery_percent=None,
                                        battery_remaining_capacity_mWh=None,
                                        battery_voltage_mV=None,
                                        core_percents=[80,80])

    sleep(3)
    iteration_2 = IterationMetadata(timestamp=datetime.now(), start_date=start_time,
                                    hostname="first",session_id="session1")


    raw_results_2 = ProcessSystemRawResults(process_results_2, system_results_2)
    agg = EnergyModelAggregator()

    agg1_features = agg.extract_features(raw_results_1,iteration_1)
    agg1_predict = agg.process_sample(agg1_features)

    agg2_features = agg.extract_features(raw_results_2,iteration_2)
    agg2_predict = agg.process_sample(agg2_features)
    print("Hello")
