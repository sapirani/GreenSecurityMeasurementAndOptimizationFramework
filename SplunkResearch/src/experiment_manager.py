import logging
import os
import pickle
import shutil
import datetime
import smtplib
from email.message import EmailMessage
import ssl
from dotenv import load_dotenv
import os
load_dotenv('/home/shouei/GreenSecurity-FirstExperiment/SplunkResearch/src/.env')

class ExperimentManager:
    def __init__(self, base_dir="experiments"):
        self.logger = None
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def setup_logging(self, log_file):
        """Sets up logging to write to the specified log file."""
        self.logger = logging.getLogger("my_app")    
         # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        # Set the formatter for the file handler
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        return self.logger
    
    def save_experiment(self, experiment_object, experiment_dir):
        # save experiment object in a pickle file
        with open(f'{experiment_dir}/experiment.pkl', 'wb') as f:
            pickle.dump(experiment_object, f)
    
    def load_experiment(self, experiment_dir):
        # load experiment object from a pickle file
        with open(f'{experiment_dir}/experiment.pkl', 'rb') as f:
            return pickle.load(f)

    
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
    
    def get_last_experiment_dir(self):
        """Returns the last experiment directory. by name parsed as date"""
        experiments = self.list_experiments()
        # names are date. parse them as date and sort them
        experiments.sort(key=lambda x: datetime.datetime.strptime(' '.join(x.split('_')[1:]), '%Y%m%d %H%M%S'))
        return experiments[-1]

    def read_last_x_mb(self, file_path, x_mb):
        with open(file_path, 'rb') as file:
            file_size = os.path.getsize(file_path)
            x_bytes = x_mb * 1024 * 1024  # Convert megabytes to bytes
            if file_size < x_bytes:
                # If the file is smaller than x MB, read the entire file
                content = file.read()
            else:
                # Move the file pointer to the position x bytes from the end
                file.seek(-x_bytes, os.SEEK_END)
                # Read the last x bytes of the file
                content = file.read()

        return content

    def send_email(self, log_file):
        my_email = os.getenv('EMAIL')
        email_password = os.getenv('EMAIL_PASSWORD')
        print(my_email)
        print(email_password)
        msg = EmailMessage()
        msg['Subject'] = 'Experiment is done - Log File is Attached' 
        msg['From'] = my_email
        msg['To'] = my_email
        with open(log_file, 'rb') as f:
            # file_data = f.read()
            file_data = self.read_last_x_mb(log_file, 25)
            msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=f.name)
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(my_email, email_password)
            smtp.send_message(msg)
    
if __name__ == "__main__":
    manager = ExperimentManager()

    # Create a new experiment directory
    new_dir = manager.create_experiment_dir()
    print(f"New experiment directory: {new_dir}")
    # create log file
    with open(f'{new_dir}/log_train.txt', 'w') as f:
        f.write("test")
    # send email
    manager.send_email(f'{new_dir}/log_train.txt')
    # Archive experiments older than 30 days
    manager.archive_old_experiments(30)

    # List all current experiments
    print("Current experiments:")
    for exp in manager.list_experiments():
        print(exp)
