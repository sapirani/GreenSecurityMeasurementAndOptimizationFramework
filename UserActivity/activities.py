from tasks import *

google_url = 'https://www.google.com/'
cnn_url = 'https://us.cnn.com/'
whatsapp_url = 'https://web.whatsapp.com/'


def create_driver():
    driver = webdriver.Chrome()
    driver.get(url=google_url)
    # self.url = driver.open_driver('https://www.google.com/')
    return driver


driver = create_driver()


class ACTIVITIES:
    git_cnn_whatsapp_google = [
        #GoogleSearch(driver),
        #OpenLink(driver),
        #GetLink(driver),
        #Git(driver),
        NewTab(driver, cnn_url),
        ScrollDown(driver),
        NewTab(driver, whatsapp_url),
        NewTab(driver, google_url),
        CloseDriver(driver)
    ]
