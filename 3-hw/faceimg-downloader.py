text_file = open("pubfig.txt", "r")
lines = text_file.readlines()
print(lines[0]).split('\t')
text_file.close()