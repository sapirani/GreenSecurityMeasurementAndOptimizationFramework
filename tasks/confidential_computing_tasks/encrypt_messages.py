from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.execute_pipeline import execute_pipeline

if __name__ == "__main__":

    encrypt_messages = execute_pipeline(ActionType.Encryption)
    print("Encryption Process Ended")
    print("Num of Encrypted Messages: {}".format(len(encrypt_messages)))
    print("Encrypted Messages: {}".format(encrypt_messages))
