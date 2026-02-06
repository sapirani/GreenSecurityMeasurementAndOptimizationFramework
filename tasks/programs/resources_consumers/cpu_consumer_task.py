import argparse
import time
import psutil
import multiprocessing

initial_time = time.time()

def consume_cpu(percentage: float, total_duration: int):
    curr_process = psutil.Process()
    curr_process.cpu_percent()

    curr_process.nice(psutil.HIGH_PRIORITY_CLASS)
    adjustments = 0
    inter_num = 0
    desired_consumption_between_0_to_1 = percentage / 100
    while True:
        inter_num += 1
        if time.time() - initial_time > total_duration:   # exit program
            return

        start_time = time.time()
        a = 0
        # Perform CPU-intensive computations
        while (time.time() - start_time) < desired_consumption_between_0_to_1:
            # Perform computations that consume CPU cycles
            # For example, calculate Fibonacci sequence
            a += 1  # Adjust the size of the Fibonacci sequence as needed
        cpu_percent = curr_process.cpu_percent() / psutil.cpu_count(logical=False)
        if cpu_percent > percentage:
            adjustments += 0.01
        else:
            adjustments -= 0.01

        sleep_time = ((1 - desired_consumption_between_0_to_1) / 5) + adjustments
        if sleep_time >= 0:
            time.sleep(sleep_time)
        else:
            raise RuntimeError("Received a negative sleep time. The Rate value is too high.")


def divide_consumption_to_processes(cpu_consumption: float):
    processes_num = int(cpu_consumption / 13) + 1
    process_cpu = cpu_consumption / processes_num
    if cpu_consumption <= 60:
        return processes_num, process_cpu
    elif cpu_consumption <= 80:
        return processes_num + 1, process_cpu
    elif cpu_consumption < 88:
        return processes_num + 2, process_cpu
    else:
        return processes_num + 3, process_cpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program is a dummy task that only consumes CPU"
    )

    parser.add_argument("-t", "--duration",
                        type=int,
                        required=True,
                        help="The duration of executing the task")

    parser.add_argument("-c", "--cpu_consumption",
                        type=float,
                        required=True,
                        help="The cpu usage that the dummy should consume")

    args = parser.parse_args()

    cpu_consumption = args.cpu_consumption
    duration = args.duration

    processes = []
    num_of_processes, cpu_consumption_per_process = divide_consumption_to_processes(cpu_consumption)
    for _ in range(num_of_processes):
        p = multiprocessing.Process(target=consume_cpu, args=(cpu_consumption_per_process, duration))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
