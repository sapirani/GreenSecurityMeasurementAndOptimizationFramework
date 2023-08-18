import os
import shutil
import datetime

class ExperimentManager:
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def create_experiment_dir(self):
        """Creates a new directory for the experiment based on the current timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(self.base_dir, f"exp_{timestamp}")
        os.makedirs(exp_dir)
        return exp_dir

    def archive_old_experiments(self, days_old=30):
        """Archives experiments that are older than the specified number of days."""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days_old)
        for exp in os.listdir(self.base_dir):
            exp_path = os.path.join(self.base_dir, exp)
            if os.path.isdir(exp_path):
                creation_time = datetime.datetime.fromtimestamp(os.path.getctime(exp_path))
                if creation_time < cutoff_time:
                    shutil.make_archive(exp_path, 'zip', exp_path)
                    shutil.rmtree(exp_path)

    def list_experiments(self):
        """Lists all experiments."""
        return [os.path.join(self.base_dir, exp) for exp in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, exp))]

if __name__ == "__main__":
    manager = ExperimentManager()

    # Create a new experiment directory
    new_dir = manager.create_experiment_dir()
    print(f"New experiment directory: {new_dir}")

    # Archive experiments older than 30 days
    manager.archive_old_experiments(30)

    # List all current experiments
    print("Current experiments:")
    for exp in manager.list_experiments():
        print(exp)
