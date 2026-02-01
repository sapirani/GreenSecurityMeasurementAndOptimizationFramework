from web_tasks import *
from office_tasks import *

"""driver = create_driver()
time.sleep(10)
driver.close()"""


class ACTIVITIES:
    """
    list of all the activities that will be performed
    """
    @staticmethod
    def activities_flow():
        driver = create_driver()
        return [
            Word(),
            Excel(),
            PowerPoint(),
            # Outlook(),
            GoogleSearch(driver, github_key),
            OpenFirstLink(driver),
            GitLogIn(driver, username, password),
            NewTab(driver, cnn_url),
            ScrollDown(driver),
            NewTab(driver, whatsapp_url),
            NewTab(driver, google_url),
            GoogleSearch(driver, stack_overflow),
            LoopOverLinks(driver),
            NewTab(driver, google_url),
            GoogleSearch(driver, nn_key),
            OpenFirstLink(driver),
            ScreenShot(driver),
            NewTab(driver, javascr_url),
            YouTubeVideo(driver, video),
            CloseDriver(driver)
        ]
