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
        self.total_start_time = None
        self.total_end_time = None
        self.start_time = None
        self.end_time = None
        self.search_key = ''
        self.github_key = 'github sign in'
        self.stack_overflow = 'python selenium code example stack overflow'
        self.nn_key = 'neural network architecture diagram'
        self.url = ''
        self.final_list = []
        self.google_url = 'https://www.google.com/'
        self.cnn_url = 'https://us.cnn.com/'
        self.whatsapp_url = 'https://web.whatsapp.com/'
        self.photos_url = 'https://photos.google.com/'
        self.javascr_url = 'https://observablehq.com/@d3/gallery'
        self.video = 'neural+network+explained+statquest+'
        self.username = 'sdvsdfv'
        self.password = 'sdfvsdfv'
        self.driver = self.create_driver()
        
        self.tasks_list = [
            GoogleSearch(self.driver, self.github_key),
            OpenFirstLink(self.driver),
            GitLogIn(self.driver, self.username, self.password),
            NewTab(self.driver, self.cnn_url),
            ScrollDown(self.driver),
            NewTab(self.driver, self.whatsapp_url),
            NewTab(self.driver, self.google_url),
            GoogleSearch(self.driver, self.stack_overflow),
            LoopOverLinks(self.driver),
            NewTab(self.driver, self.google_url),
            GoogleSearch(self.driver, self.nn_key),
            OpenFirstLink(self.driver),
            ScreenShot(self.driver),
            NewTab(self.driver, self.javascr_url),      
            YouTubeVideo(self.driver, self.video),

        ]

    def _run_tasks(self):
        self.total_start_time = time.time()
        for task in self.tasks_list:
            self.start_time = time.time()
            # print(f'Task {self.task_name} start. Start time: {self.start_time}')
            # run task
            task.run_task()
            self.end_time = time.time()
            self.final_list.append((task.get_name(), self.start_time, self.end_time))
            print(f'Task {self.task_name} end. End time: {self.end_time}')
        self.total_end_time = time.time()
        print(f'total user activity running time: {self.total_end_time - self.total_start_time}')
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
