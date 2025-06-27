from tasks.confidential_computing_tasks.action_type import ActionType
from tasks.confidential_computing_tasks.tasks_options.execute_pipeline import execute_regular_pipeline

if __name__ == "__main__":
    decrypt_messages = execute_regular_pipeline(ActionType.Decryption)
    print("Decryption Process Ended")
    print("Num of Decrypted Messages: {}".format(len(decrypt_messages)))
    print("Decrypted Messages: {}".format(decrypt_messages))
