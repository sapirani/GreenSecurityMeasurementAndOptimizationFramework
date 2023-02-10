from web_tasks import *
from office_tasks import *

google_url = 'https://www.google.com/'
cnn_url = 'https://us.cnn.com/'
whatsapp_url = 'https://web.whatsapp.com/'
photos_url = 'https://photos.google.com/'
javascr_url = 'https://observablehq.com/@d3/gallery'
video = 'neural+network+explained+statquest'
username = 'sdvsdfv'
password = 'sdfvsdfv'
github_key = 'github sign in'
stack_overflow = 'python selenium code example stack overflow'
nn_key = 'neural network architecture diagram'


def create_driver():
    """
    function to create a driver connection to the browser ('Chrome')

    Returns:
        driver: the driver connects to google chrome page loaded with the google url
    """   
    driver = webdriver.Chrome()
    driver.get(url=google_url)
    # self.url = driver.open_driver('https://www.google.com/')
    return driver


driver = create_driver()


class ACTIVITIES:
    """
    list of all the activities that will be performed
    """
    activities_flow = [
        Word(),
        Excel(),
        PowerPoint(),
        Outlook(),
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
       
        
    ]
