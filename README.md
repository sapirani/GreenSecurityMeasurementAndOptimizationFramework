# GreenSecurity-FirstExperiment

The main goal of this project is to understand the energy consumption of different anti virus scans, on each of the hardware components separately.
This project measures various metrics of computer resource usage (such as: CPU usage, memory, disk and battery) while running anti virus scans. 
We measure the resource usage of the whole system, and of each process separately.

There are 3 modes of execution:
1. No scan - the code measures the resource usage without running any anti virus scan. This is the baseline consumption of other processes running in the background.
2. One scan - the code starts a scan using windows defender. Meanwhile, the code keeps measuring the resource usage. It is possible to configure what type of scan to perform, as shown below
3. Continuous scan - the code runs a scan using windows defender, and when the scan is over it immediately starts a new one. The code ends when at least a certain amount of time has elapsed and at least a certain amount of battery capcity has been decreased. It is possible to configure these parameters and the type of scan, as shown below.
