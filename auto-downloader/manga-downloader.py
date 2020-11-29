from selenium import webdriver
import time
import urllib.request
import os  
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

# 山立漫畫 - 無良公會

local_path = 'H:\無良工會'
filename = 1
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.86 Safari/537.36"}
# headers = {'User-Agent':str(ua.chrome)}

# 啟動chrome瀏覽器
chromeDriver = 'D:\github\chromedriver.exe' # chromedriver檔案放的位置，請自行下載 chromeDriver， google 搜尋 "chromeDriver" 即可，請下載當前 chrome 版本的 Driver
driver = webdriver.Chrome(chromeDriver) 
chapters = {}
driver.get('https://www.setnmh.com/comic-lvcnh-%E7%84%A1%E8%89%AF%E5%85%AC%E6%9C%83')
driver.find_element_by_xpath("//span[.='點擊此處繼續閱讀']").click()
driver.find_element_by_xpath("//span[.='全部目錄']").click()

print("Current Page Title is : %s" %driver.title)

time.sleep(10)

for element in driver.find_elements_by_xpath('//ul[@id="ul_chapter1"]/li/a'):
    url_chapter = element.get_attribute('href')
    folderName = element.get_attribute('title').rstrip()
    print(url_chapter)
    chapters.setdefault(folderName, url_chapter)
    # driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 't')
    # # driver.get(url_chapter)
    # # driver.execute_script("window.open('');")
    # # driver.switch_to.window(driver.window_handles[1])
    # driver.get(url_chapter)

    # print("Current Page Title is : %s" %driver.title)

    # if not os.path.exists(local_path + '\\' + folderName):
    #     os.makedirs(local_path + '\\' + folderName)

    # time.sleep(10)

    # # driver.close()
    # driver.get('https://www.setnmh.com/comic-lvcnh-%E7%84%A1%E8%89%AF%E5%85%AC%E6%9C%83')

    # # driver.switch_to.window(driver.window_handles[0])
    # print("Current Page Title is : %s" %driver.title)

for keys_chapter in chapters.keys():
    driver.get(chapters[keys_chapter])
    time.sleep(5)

    download_dir = local_path + '\\' + keys_chapter
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    count = 1
    while True:
        target = driver.find_element_by_xpath("//div[@class='ptview']/img")
        url_img = target.get_attribute('src')
        print(url_img)
        # urllib.request.urlretrieve(url_img, os.path.join(local_path , filename))
        filename = str(count) + '.jpg'
        print(download_dir + '\\' + filename)

    
        req = urllib.request.Request(url_img, headers=headers)
        data = urllib.request.urlopen(req).read()
        with open(download_dir + '\\' + filename, 'wb') as f:
            f.write(data)
            f.close()

        count = count + 1
               
        try:
            driver.find_element_by_xpath("//a[.='下一頁']").click()
        except Exception as e:
            break
        
        time.sleep(5)

        # opener=urllib.request.build_opener()
        # opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        # urllib.request.install_opener(opener)
        # try:
        #     target_name = local_path + '\\' + folderName
        #     urllib.request.urlretrieve(url_img, os.path.join(target_name , filename))
        # except Exception as e:
        #     print(e)
    # 



# driver.close()