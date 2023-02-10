import sys
import os
from activities import *
import pandas as pd


def dataframe_append(df, element):
    """Append a row to a dataframe"""
    df.loc[len(df.index)] = element


class Scheduler:

    def __init__(self, tasks, results_path):
        """_summary_: _description_

        Args:
            tasks (list): list of tasks to run
            results_path (list[str]): array of paths to save the results
        """
        self.tasks_list = tasks
        self.results_path = results_path
        self.tasks_times = pd.DataFrame(columns=["task name", "start time", "finished time", "duration"])

    def run_tasks(self):
        """_summary_: take times of each task and append it to a dataframe
        """
        start_time = time.time()
        for task in self.tasks_list:
            task_start_time = time.time() - start_time
            # run task
            task.run_task()
            task_end_time = time.time() - start_time
            dataframe_append(self.tasks_times, [task.get_name(), task_start_time, task_end_time,
                                                task_end_time - task_start_time])
            # wait for 20 seconds
            time.sleep(3)

    def get_list(self):
        return self.tasks_list

    def save_tasks_times(self):
        """_summary_: save the dataframe to a csv file
        """
        self.tasks_times.to_csv(self.results_path, index=False)


def main():
    if len(sys.argv) != 2:
        raise Exception("Expecting exactly one argument - scan path")

    # Yonatan's code
    user_activity = Scheduler(ACTIVITIES.activities_flow, sys.argv[1])
    user_activity.run_tasks()
    user_activity.save_tasks_times()


if __name__ == "__main__":
    main()

