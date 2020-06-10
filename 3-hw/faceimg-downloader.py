
import urllib.request
import os
from urllib.error import HTTPError

# Public Figures Face Database(哥倫比亞大學公眾人物臉部數據庫)

# 存圖位置
path = 'data_pubfig/'

count = 1
with open('pubfig.txt') as object_pubfig:
    for line in object_pubfig:
      li=line.strip()
      if not li.startswith("#"):
        #print(line.split("\t")[2])
        filename = ("%06d" % count) + '.jpg'
        print(filename)
        # 保存圖片
        try:
          urllib.request.urlretrieve(line.split("\t")[2], os.path.join(path , filename))
          count += 1
        except e:
          print(e)
          pass

