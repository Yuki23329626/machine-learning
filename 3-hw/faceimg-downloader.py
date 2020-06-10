
# Public Figures Face Database(哥倫比亞大學公眾人物臉部數據庫)

# 存圖位置
path = 'data_pubfig/'

with open('pubfig.txt') as object_pubfig:
    for line in object_pubfig:
      print(line.split("\t")[2])

