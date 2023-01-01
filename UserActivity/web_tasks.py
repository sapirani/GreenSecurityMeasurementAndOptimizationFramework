import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from abc import ABCMeta, abstractmethod
from PIL import Image


class AbstractTask:
    def __init__(self, driver):
        self.driver = driver

    @abstractmethod
    def run_task(self):
        pass

    def get_name(self):
        return self.__class__.__name__


class CloseDriver(AbstractTask):
    def run_task(self):
        self.driver.quit()


"""class GoogleChrome(AbstractTask):
    
    def __init__(self, driver):
        super(GoogleChrome, self).__init__(driver)
        self.url = self.driver.open_driver('https://www.google.com/')"""


class GoogleSearch(AbstractTask):

    def __init__(self, driver, search_key):
        super(GoogleSearch, self).__init__(driver)
        self.search_key = search_key

    def run_task(self):
        # find the element that's name attribute is q (the Google search box)
        elem = self.driver.find_element(By.NAME, "q")
        # clear the text in it
        elem.clear()
        # type in search
        elem.send_keys(self.search_key)
        # submit the form (although google automatically searches now without submitting)
        elem.send_keys(Keys.RETURN)


class OpenFirstLink(AbstractTask):

    def __init__(self, driver):
        super(OpenFirstLink, self).__init__(driver)

    def run_task(self):
        # open first link from the search results
        # make try adn except if there is no link
        try:
            result = self.driver.find_element(By.TAG_NAME, 'h3')
            result.click()
        except:
            pass

class LoopOverLinks(AbstractTask):

    def __init__(self, driver):
        super(LoopOverLinks, self).__init__(driver)

    def run_task(self):
        def find_links(driver):
                # find all links in the current page
                return driver.find_elements(By.TAG_NAME, 'h3')
        try:
            links = find_links(self.driver)
            for link in range(5):
                links[link].click()
                time.sleep(20)
                ScrollDown(self.driver)
                time.sleep(5)
                self.driver.back()
                time.sleep(5)
        
        except:
            pass

class GitLogIn(AbstractTask):

    def __init__(self, driver, username, password):
        super(GitLogIn, self).__init__(driver)
        self.username = username
        self.password = password

    def run_task(self):
        # login
        uname = self.driver.find_element("id", "login_field")
        uname.send_keys(self.username)

        pword = self.driver.find_element("id", "password")
        pword.send_keys(self.password)

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

        time.sleep(30)

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
        time.sleep(10)


class ScrollDown(AbstractTask):

    def __init__(self, driver):
        super(ScrollDown, self).__init__(driver)

    def run_task(self):
        scroll_pause_time = 1

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

class YouTubeVideo(AbstractTask):
    def __init__(self, driver, video):
        super(YouTubeVideo, self).__init__(driver)
        self.video = video
    def run_task(self):
        # Navigate to url with video being appended to search_query
        self.driver.get('https://www.youtube.com/results?search_query={}'.format(str(self.video)))
        links = self.driver.find_elements(By.CLASS_NAME, "style-scope ytd-video-renderer")
        if links:
            links[0].click()
            time.sleep(1134)

class ScreenShot(AbstractTask):
    def __init__(self, driver):
        super(ScreenShot, self).__init__(driver)

    def run_task(self):
        self.driver.save_screenshot('ss.png')
        screenshot = Image.open('ss.png')
        # download the screenshot
        screenshot.save('ss.png')
        time.sleep(5)