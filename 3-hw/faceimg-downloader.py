text_file = open("pubfig.txt", "r")
lines = text_file.readlines()
print(lines[2].split("\t")[3])
text_file.close()