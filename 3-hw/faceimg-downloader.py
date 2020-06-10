from numpy import loadtxt
lines = loadtxt("pubfig.txt", comments="#", delimiter="\t", unpack=False)
print(lines)