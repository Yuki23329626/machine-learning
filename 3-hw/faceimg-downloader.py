text_file = open("pubfig.txt", "r")
lines = text_file.read().split(' ')
print(lines)
text_file.close()