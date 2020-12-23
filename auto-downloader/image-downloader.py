from selenium import webdriver
import time
import urllib.request
import os

# 存圖位置
local_path = 'D:\machine-learning\auto-downloader\landscape_imgs'
# 爬取頁面網址 
url = 'https://pic.sogou.com/pics?query=%EFL%BE%B0&di=2&_asf=pic.sogou.com&w=05009900'  
  
# 目標元素的xpath
xpath = '//div[@id="imgid"]/ul/li/a/img'
# 啟動chrome瀏覽器
chromeDriver = 'D:\github\chromedriver.exe' # chromedriver檔案放的位置，請自行下載 chromeDriver， google 搜尋 "chromeDriver" 即可，請下載當前 chrome 版本的 Driver
driver = webdriver.Chrome(chromeDriver) 
  
# 最大化窗口，因為每一次爬取只能看到視窗内的圖片  
driver.maximize_window()
  
# 紀錄下載過的圖片網址，避免重複下載  
img_url_dic = {}  
  
# 瀏覽器打開爬取頁面
driver.get(url)  
  
# 模擬滾動視窗瀏覽更多圖片
pos = 0  
m = 0 # 圖片編號 
for i in range(100):  # 100 的話有 2393 張好像是極限了
    pos += i*500 # 每次下滾500(不知道這個單位是甚麼)  
    js = "document.documentElement.scrollTop=%d" % pos  
    driver.execute_script(js)  
    time.sleep(1)

    for element in driver.find_elements_by_xpath(xpath):
        try:
            img_url = element.get_attribute('src')
            
            # 保存圖片到指定路徑
            if img_url != None and not img_url in img_url_dic:
                img_url_dic[img_url] = ''  
                m += 1
                # print(img_url)
                ext = img_url.split('/')[-1]
                # print(ext)
                filename = str(m) + '_' + 'landscape' + '_' + ext +'.jpg'
                print(filename)
                
                # 保存圖片
                urllib.request.urlretrieve(img_url, os.path.join(local_path , filename))
                
        except OSError:
            print('發生OSError!')
            print(pos)
            break;
            
driver.close()