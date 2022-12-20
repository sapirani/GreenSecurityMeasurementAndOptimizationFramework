import time
from tasks import *
import os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class Task:

    def __init__(self):
        self.task_name = ''
        self.start_time = None
        self.end_time = None
        self.url = ''
        self.final_list = []
        self.google_url = 'https://www.google.com/'
        self.cnn_url = 'https://us.cnn.com/'
        self.whatsapp_url = 'https://web.whatsapp.com/'
        self.driver = self.create_driver()
        self.tasks_list = [
            GoogleSearch(self.driver),
            OpenLink(self.driver),
            GetLink(self.driver),
            Git(self.driver),
            NewTab(self.driver, self.cnn_url),
            ScrollDown(self.driver),
            NewTab(self.driver, self.whatsapp_url),
            NewTab(self.driver, self.google_url),
        ]

    def _run_tasks(self):
        for task in self.tasks_list:
            self.start_time = time.time()
            # print(f'Task {self.task_name} start. Start time: {self.start_time}')
            # run task
            task.run_task()
            self.end_time = time.time()
            self.final_list.append((task.get_name(), self.start_time, self.end_time))
            # wait for 20 seconds
            time.sleep(20)
        self.driver.quit()

    def _get_list(self):
        return self.tasks_list

    def create_driver(self):
        driver = webdriver.Chrome()
        driver.get(url=self.google_url)
        #self.url = driver.open_driver('https://www.google.com/')
        return driver


if __name__ == "__main__":
    # Yonatan's code
    user_activity = Task()
    user_activity._run_tasks()
    print(user_activity._get_list())

    # for task in experiment:
    #   print(f"Task name: {task[0]}\nStart run time: {task[1]}\nEnd run time: {task[1]}\nExceptions: {task[1]}")
