import os
import matplotlib.pyplot as plt
import psutil
import Levenshtein
from googletrans import Translator
import re
import sys

s="Water"
g="Light"

silver=""
gold=""

ddd = dict()
NoAns = dict()

def is_sp(s):

    sp=0
    
    if s == ".":
        sp=1
    elif s == ",":
        sp=1 
    elif s == "?":
        sp=1            
    elif s == '"':
        sp=1        
    elif s == "'":
        sp=1
    elif s == "[":
        sp=1
    elif s == "]":
        sp=1
    elif s == "(":
        sp=1            
    elif s == ")":
        sp=1            
    elif s == ":":
        sp=1            
    elif s == ";":
        sp=1            
    elif s == "/":
        sp=1
    elif s == "!":
        sp=1
    elif s == "#":
        sp=1            
    elif s == "&":
        sp=1
    elif s == "$":
        sp=1
    elif s == "%":
        sp=1
    elif s == "-":
        sp=1 
    return sp


meta=""
    

with open("./metadata.txt") as f:
    for line in f:
       meta=meta+line

meta=meta.split()

z=""

for l in range(len(meta)):

    if z != "":
        break
    
    strr=""
    
    title = meta[l]
    
    with open("./"+title+".txt") as f:
        for line in f:
            strr=strr+line
    strr2=""

    for k in range(len(strr)):   
        if is_sp(strr[k]) > 0:
            strr2+=" "+strr[k]+" "
        else:
            strr2+=strr[k]
    
    talk = strr2.split()

    for x in talk:
        if x not in NoAns:
            NoAns[x]=1
        else:
            NoAns[x]+=1

    hit=0
    for i in range(len(talk)):
        if hit >0:
            break
        for j in range(len(spt1)):
            if talk[i]==spt1[j] and NoAns[talk[i]] < 1000:
                hit+=1
    if hit == 0:
        print(title)
        for i in range(len(talk)):
            z+=talk[i]+" "

with open("./"+s+".txt") as f:
    for line in f:
        silver=silver+line
            
with open("./"+g+".txt") as f:
    for line in f:
        gold=gold+line

spsilver=""
spgold=""

for k in range(len(silver)):
    if is_sp(silver[k]) > 0:
        spsilver+=" "+silver[k]+" "
    else:
        spsilver+=silver[k]

for k in range(len(gold)):
    if is_sp(gold[k]) > 0:
        spgold+=" "+gold[k]+" "
    else:
        spgold+=gold[k]

goldsilver=gold+" "+silver
spgoldsilver=""

for k in range(len(goldsilver)):
    if is_sp(goldsilver[k]) > 0:
        spgoldsilver+=" "+goldsilver[k]+" "
    else:
        spgoldsilver+=goldsilver[k]


spt1=spgold.split()
spt2=spsilver.split()
AND=""


for i in range(len(spt1)):
    for j in range(len(spt2)):
        if spt1[i]==spt2[j] and NoAns[spt1[i]] < 1000:
            if spt1[i] not in ddd:
                ddd[spt1[i]]=1
                AND+=spt1[i]+" "
            else:
                ddd[spt1[i]]+=1

spt3=AND.split()
spt4=spgoldsilver.split()

era=[]

for i in range(len(spt4)):
    if spt4[i] in ddd:
        if ddd[spt4[i]] > 0:
            era.append(i)
            ddd[spt4[i]]-=1

era.append(len(spt4))

ans=""
cnt=0
for i in range(len(spt4)):
    if i==era[cnt]:
        cnt+=1
        continue
    ans+=" "+spt4[i]
        

#file = open("NOT"+str(g)+".txt","wt")
#file.write(str(z)+"\n")
#file = open(str(g)+"OR"+str(s)+".txt", "wt")
#file.write(str(ans)+"\n")
#file = open(str(g)+"AND"+str(s)+".txt", "wt")
#file.write(str(AND)+"\n")
