import os
import matplotlib.pyplot as plt
import psutil
import Levenshtein
from googletrans import Translator
import re
import sys


nksize=0
nextkey=[]
nextvalue=[]
datakun = dict()
                
with open('./nextkey.txt') as f:
    for line in f:
        nksize+=1
        line.rstrip('\n')
        nextkey.append(str(line))

with open('./nextvalue.txt') as f:
    for line in f:
        line.rstrip('\n')
        nextvalue.append(str(line))

nfile = open("ndict.txt", "wt")

for i in range(nksize):
    nfile.write(str(nextkey[i]).rstrip('\n')+"@"+str(float(nextvalue[i].rstrip('\n')))+"\n")