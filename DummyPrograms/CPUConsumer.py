# SuperFastPython.com
# example of a program that uses all cpu cores
import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

import psutil

# define a cpu-intensive task
"""def task(arg):
    return sum([math.sqrt(i) for i in range(1, arg)])


# protect the entry point
if __name__ == '__main__':
    # report a message
    print('Starting task...')
    # create the process pool
    with ProcessPoolExecutor(8) as exe:
        # perform calculations
        results = exe.map(task, range(1, 250000))
    # report a message
    print('Done.')"""

import time
import psutil
import multiprocessing


initial_time = time.time()

def consume_cpu(percentage):
    curr_process = psutil.Process()
    curr_process.cpu_percent()
    #cpu_num = psutil.cpu_count()
    curr_process.nice(psutil.HIGH_PRIORITY_CLASS)
    adjustments = 0
    inter_num = 0
    desired_consumption_between_0_to_1 = percentage / 100
    while True:
        inter_num += 1
        if time.time() - initial_time > int(sys.argv[2]):   # exit program -
            return

        start_time = time.time()
        a = 0
        # Perform CPU-intensive computations
        while (time.time() - start_time) < desired_consumption_between_0_to_1:
            # Perform computations that consume CPU cycles
            # For example, calculate Fibonacci sequence
            a += 1  # Adjust the size of the Fibonacci sequence as needed
        #time.sleep((1 - desired_consumption_between_0_to_1) / 5)
        cpu_percent = curr_process.cpu_percent() / psutil.cpu_count(logical=False)
        #print(cpu_percent)
        if cpu_percent > percentage:
            adjustments += 0.01
        else:
            adjustments -= 0.01

        sleep_time = ((1 - desired_consumption_between_0_to_1) / 5) + adjustments
        if sleep_time > 0:
            time.sleep(sleep_time)


def divide_consumption_to_processes(cpu_consumption):
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


if __name__ == '__main__':
    processes = []
    num_of_processes, cpu_consumption_per_process = divide_consumption_to_processes(int(sys.argv[1]))
    #print("num_of_processes", num_of_processes)
    #print("cpu_consumption_per_process", cpu_consumption_per_process)
    for _ in range(num_of_processes):
        p = multiprocessing.Process(target=consume_cpu, args=(cpu_consumption_per_process,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
