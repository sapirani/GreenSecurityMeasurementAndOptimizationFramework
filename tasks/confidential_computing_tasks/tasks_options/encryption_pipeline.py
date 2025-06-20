from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.tasks_options.execute_pipeline import execute_pipeline

if __name__ == "__main__":
    valid_messages = execute_pipeline(ActionType.FullPipeline)
    print("Homomorphic Full Pipeline Process Ended")
    print("Num of Valid Messages: {}".format(len(valid_messages)))
    print("All Valid Messages: \n{}".format(valid_messages))
