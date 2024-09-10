import os
import matplotlib.pyplot as plt
import psutil
import Levenshtein
from googletrans import Translator
import re
import sys

s="Light"
g="Water"

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

for l in range(len(meta)):
    
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
spt4=spgoldsilver.split()


def NOTgate(s1):
    
    z=""
    
    for l in range(len(meta)):
        
        if z != "":
            return z
            
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
        hit=0
        for i in range(len(talk)):
            if hit > 0:
                break
            for j in range(len(s1)):
                if talk[i] in NoAns:
                    if talk[i]==s1[j] and NoAns[talk[i]] <= 1000:
                        hit+=1
                        break        
        if hit == 0:
            print(title)
            for i in range(len(talk)):
                z+=talk[i]+" "
                
    return z

NOTg=NOTgate(spt1)
NOTs=NOTgate(spt2)

def ANDgateORgate(s1,s2,s4):
    AND=""
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] in NoAns:
                if s1[i]==s2[j] and NoAns[s1[i]] <= 1000:
                    if s1[i] not in ddd:
                        ddd[s1[i]]=1
                        AND+=s1[i]+" "
                    else:
                        ddd[s1[i]]+=1

    OR=""

    era=[]
    for i in range(len(s4)):
        if s4[i] in ddd:
            if ddd[s4[i]] > 0:
                era.append(i)
                ddd[s4[i]]-=1
    era.append(len(s4))

    cnt=0
    for i in range(len(s4)):
        if i==era[cnt]:
            cnt+=1
            continue
        OR+=s4[i]+" "
                    
    return AND,OR

ANDgs,ORgs = ANDgateORgate(spt1,spt2,spt4)

spt3=ANDgs.split()

def DIFFgate(s1,s3):

    diff=""
    for i in range(len(s1)):
        hit=0
        for j in range(len(s3)):
            if s1[i]==s3[j]:
                hit=1
                break
        if hit == 0:
            diff+=s1[i]+" "
            
    return diff

DIFF=DIFFgate(spt1,spt3)

spt5=ORgs.split()


def XORgate(s5,s3):
    xor=""
    for i in range(len(s5)):
        hit=0
        for j in range(len(s3)):
            if s5[i] in NoAns:
                if s5[i]==s3[j] and NoAns[s5[i]] <= 1000:
                    hit+=1
                    break
        if hit == 0:
            xor+=s5[i]+" "
    
    return xor

XOR=XORgate(spt5,spt3)

print("XOR="+str(XOR))
print("AND="+str(ANDgs))
print("OR="+str(ORgs))
print("DIFF="+str(DIFF))
print("HA="+str(ANDgs)+str(XOR))
            
#file = open("DIFF"+str(g)+".txt","wt")
#file.write(str(DIFF)+"\n")
#file = open(str(g)+"XOR"+str(s)+".txt","wt")
#file.write(str(XOR)+"\n")    
#file = open("NOT"+str(g)+".txt","wt")
#file.write(str(NOTg)+"\n")
#file = open(str(g)+"OR"+str(s)+".txt", "wt")
#file.write(str(ORgs)+"\n")
#file = open(str(g)+"AND"+str(s)+".txt", "wt")
#file.write(str(ANDgs)+"\n")
