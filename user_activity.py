import time 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def chrome_driver(url,task_operated):

    # create a new Chrome session
    driver = webdriver.Chrome()  
    driver.get(url = url);
    task_operated.append('Chrome driver is opened')
    return driver
    
def search_google(search_key, driver, task_operated):
    # find the element that's name attribute is q (the google search box)
    elem = driver.find_element(By.NAME, "q")
    
    # clear the text in it
    elem.clear()
    # type in search
    elem.send_keys(search_key)
    # submit the form (although google automatically searches now without submitting)
    elem.send_keys(Keys.RETURN)
    
    task_operated.append(f'found {search_key}')
    
def open_first_link(driver, task_operated):
    # open first link from the search results
    # make try adn except if there is no link
    try:
        result = driver.find_element(By.TAG_NAME, 'h3')
        result.click()
        task_operated.append(f'entered first link')
    except:
        task_operated.append(f'no links found')

def find_links(driver):
    # find all links in the current page
    return driver.find_elements(By.TAG_NAME, 'h3')
    
def get_aherf_link(driver, task_operated):
    # get the link from the current page
    try:
        ahref = driver.find_element(By.TAG_NAME, 'a')
        link = ahref.get_attribute('href')
        link.click()
        task_operated.append(f'entered {link} link')
        
    except:
        task_operated.append(f'no links found')


def login_github(driver, task_operated):
    username = 'adyon@post.bgu.ac'
    password = 'aDyon5522'
    # login
    uname = driver.find_element("id", "login_field") 
    uname.send_keys(username)
    
    pword = driver.find_element("id", "password") 
    pword.send_keys(password)
    
    driver.find_element("name", "commit").click()
    
    # Wait for login process to complete. 
    WebDriverWait(driver=driver, timeout=10).until(
        lambda x: x.execute_script("return document.readyState === 'complete'")
    )
    
    # Verify that the login was successful.
    error_message = "Incorrect username or password."
    
    # Retrieve any errors found. 
    errors = driver.find_elements(By.CLASS_NAME, "flash-error")

    # When errors are found, the login will fail. 
    if any(error_message in e.text for e in errors): 
        print("[!] Login failed")
    else:
        task_operated.append(f'github account is logged in')

def open_new_tab(driver, url, key, task_operated):
    # Open a new window
    driver.execute_script("window.open('');")
  
    # Switch to the new window and open new URL
    driver.switch_to.window(driver.window_handles[1])
    driver.get(url)
    task_operated.append(f'{key} opened in new tab')

def get_name(url):
    # extract name of site from url
    return url.split('.')[1] 

def close_tab(driver, key, task_operated):
    # close tab
    driver.close()
    task_operated.append(f'{key} tab is closed')

def scroll_down(driver, task_operated):
    SCROLL_PAUSE_TIME = 0.5

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def privous_tab(driver, task_operated):
    # switch to privous tab
    driver.back()
    task_operated.append(f'back to privous tab')

# main
if __name__ == "__main__":
    
    task_operated = []
    search_key1 =  'github sign in'
    search_key2 = 'python selenium code example stack overflow'
    google_url = 'https://www.google.com/'
    cnn_url ='https://us.cnn.com/'
    whatsapp_url = 'https://web.whatsapp.com/'
    key1 = get_name(cnn_url)
    key2 = get_name(whatsapp_url)
    
    
    # flow
    chrome_driver(google_url, task_operated)
    driver = chrome_driver(google_url, task_operated)
    search_google(search_key1, driver, task_operated)
    open_first_link(driver, task_operated)
    login_github(driver, task_operated)
    open_new_tab(driver, cnn_url,  key1, task_operated)
    scroll_down(driver, task_operated)
    open_new_tab(driver, whatsapp_url, key2, task_operated)
    open_new_tab(driver, google_url, key2, task_operated)
    search_google(search_key2, driver, task_operated)
    links = find_links(driver)
    print(f'length:{len(links)}, list: {links}')
    for link in links:
        link.click()
        task_operated.append(f'entered {link} link')
        time.sleep(2)
        driver.back()
        task_operated.append(f'back to privous tab')
        time.sleep(2)
    

    # quit chrome driver
    driver.quit()
    # print(task_operated) 
