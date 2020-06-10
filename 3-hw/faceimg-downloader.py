text_file = open("pubfig.txt", "r")
lines = text_file.read().split(' ')
print lines
print len(lines)
text_file.close()