import sys
import os
from activities import *
import pandas as pd


def dataframe_append(df, element):
    df.loc[len(df.index)] = element


class Scheduler:

    def __init__(self, tasks, results_path):
        self.tasks_list = tasks
        self.results_path = results_path
        self.tasks_times = pd.DataFrame(columns=["task name", "start time", "finished time", "duration"])

    def run_tasks(self):
        start_time = time.time()
        for task in self.tasks_list:
            task_start_time = time.time() - start_time
            # run task
            task.run_task()
            task_end_time = time.time() - start_time
            dataframe_append(self.tasks_times, [task.get_name(), task_start_time, task_end_time,
                                                task_end_time - task_start_time])
            # wait for 20 seconds
            #time.sleep(3)

    def get_list(self):
        return self.tasks_list

    def save_tasks_times(self):
        self.tasks_times.to_csv(self.results_path, index=False)


def main():
    print(sys.argv)
    if len(sys.argv) != 2:
        raise Exception("Expecting exactly one argument - scan path")

    # Yonatan's code
    user_activity = Scheduler(ACTIVITIES.git_cnn_whatsapp_google, sys.argv[1])
    user_activity.run_tasks()
    user_activity.save_tasks_times()


if __name__ == "__main__":
    main()

