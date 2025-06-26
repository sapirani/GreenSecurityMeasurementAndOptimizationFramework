from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.tasks_options.execute_pipeline import execute_operation_pipeline


if __name__ == '__main__':
    execute_operation_pipeline(ActionType.Addition)
