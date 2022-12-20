import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



class AbstractTask:
    def __init__(self, driver):
        self.driver = driver

    def run_task(self):
        pass

    def get_name(self):
        return __class__.__name__


"""class GoogleChrome(AbstractTask):
    
    def __init__(self, driver):
        super(GoogleChrome, self).__init__(driver)
        self.url = self.driver.open_driver('https://www.google.com/')"""


class GoogleSearch(AbstractTask):

    def __init__(self, driver):
        super(GoogleSearch, self).__init__(driver)
        self.search_key = 'github sign in'

    def run_task(self):
        # find the element that's name attribute is q (the Google search box)
        elem = self.driver.find_element(By.NAME, "q")
        # clear the text in it
        elem.clear()
        # type in search
        elem.send_keys(self.search_key)
        # submit the form (although google automatically searches now without submitting)
        elem.send_keys(Keys.RETURN)


class OpenLink(AbstractTask):

    def __init__(self, driver):
        super(OpenLink, self).__init__(driver)

    def run_task(self):
        # open first link from the search results
        # make try adn except if there is no link
        # try:
        result = self.driver.find_element(By.TAG_NAME, 'h3')
        result.click()
        # except:
        #    pass


class GetLink(AbstractTask):

    def __init__(self, driver):
        super(GetLink, self).__init__(driver)


    def run_task(self):
        # get the link from the current page
        # try:
        ahref = self.driver.find_element(By.TAG_NAME, 'a')
        link = ahref.get_attribute('href')
        link.click()
        # except:


class Git(AbstractTask):

    def __init__(self, driver):
        super(Git, self).__init__(driver)

    def run_task(self):
        username = 'adyon@post.bgu.ac'
        password = 'aDyon5522'
        # login
        uname = self.driver.find_element("id", "login_field")
        uname.send_keys(username)

        pword = self.driver.find_element("id", "password")
        pword.send_keys(password)

        self.driver.find_element("name", "commit").click()

        # Wait for login process to complete.
        WebDriverWait(driver=self.driver, timeout=10).until(
            lambda x: x.execute_script("return document.readyState === 'complete'")
        )

        # Verify that the login was successful.
        error_message = "Incorrect username or password."

        # Retrieve any errors found.
        errors = self.driver.find_elements(By.CLASS_NAME, "flash-error")

        # When errors are found, the login will fail.
        if any(error_message in e.text for e in errors):
            print("[!] Login failed")


class NewTab(AbstractTask):

    def __init__(self, driver, url):
        super(NewTab, self).__init__(driver)
        self.url = url
        self.key = self.url.split('.')[1]  # unused

    def run_task(self):
        # Open a new window
        self.driver.execute_script("window.open('');")
        # Switch to the new window and open new URL
        self.driver.switch_to.window(self.driver.window_handles[1])
        self.driver.get(self.url)


class ScrollDown(AbstractTask):

    def __init__(self, driver):
        super(ScrollDown, self).__init__(driver)

    def run_task(self):
        scroll_pause_time = 0.5

        # Get scroll height
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(scroll_pause_time)

            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        # ??
        self.driver.back()

