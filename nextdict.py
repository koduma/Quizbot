import os
import matplotlib.pyplot as plt
import psutil
import Levenshtein
from googletrans import Translator
import re
import sys

strr=""
meta=""

train = dict()
train_num = dict()
datakun = dict()
NoAns = dict()

with open("./metadata.txt") as f:
    for line in f:
       meta=meta+line

meta=meta.split()

counter=1
now=[1]

PROBLEM = len(meta)
TABOO = 1000

translator = Translator()

def is_ja(s):
    return True if re.search(r'[ぁ-んァ-ン]', s) else False

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

looked = dict()

for l in range(len(meta)):
    title = meta[l]
    title=title.upper()
    ex=title in looked
    if ex==True and title!="LATEX":
        print(title)
    else:
        looked[title]=1
        
#sys.exit()

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

    n=0

    train[title]=counter
    train_num[counter]=title
    NoAns[title]=0
    counter+=1
    n=counter
    
    for x in talk:
        ex=x in train.keys()
        if ex==False:
            train[x]=counter
            train_num[counter]=x
            NoAns[x]=1
            counter=counter+1
            n=counter
        else:
            NoAns[x]+=1
    now.append(n)       
    for c in talk:
        tmp5 = title+","+str(c)
        tmp6 = str(c)+","+str(title)
        if tmp5 in datakun:
            datakun[tmp5]+=2
        else:
            datakun[tmp5]=2

        if tmp6 in datakun:
            datakun[tmp6]+=2
        else:
            datakun[tmp6]=2

#for c in range(counter-2):
#    tmp = str(train_num[c+1])+","+str(train_num[c+2])
#    if tmp in data:
#        data[tmp]+=1
#    else:
#        data[tmp]=1
    
    for c in range(now[l]-1,counter-1):
        for c2 in range(c+1,counter-1):
            tmp = str(train_num[c+1])+","+str(train_num[c2+1])
            #if train_num[c+1]=="Empiricism":
                #print(tmp+","+str(l))
            if tmp in datakun:
                datakun[tmp]+=1
                #if str(train_num[c+1])=="Empiricism":
                    #print(str(tmp)+"="+str(datakun[tmp]))
            else:
                datakun[tmp]=1
            tmp3 = str(train_num[c2+1])+","+str(train_num[c+1])
            if tmp3 in datakun:
                datakun[tmp3]+=1
            else:
                datakun[tmp3]=1

for l in range(len(meta)):
    if NoAns[meta[l]] > TABOO:
        print("ERROR:"+str(meta[l]))
        #s=str(meta[l])
        #os.rename(s+".txt",s.upper()+".txt")

nksize=0
nextkey=[]
nextvalue=[]

try:
    with open('./nextkey.txt') as f:
        for line in f:
            nksize+=1
            line.rstrip('\n')
            nextkey.append(str(line))
            
    with open('./nextvalue.txt') as f:
        for line in f:
            line.rstrip('\n')
            nextvalue.append(str(line))
    
    for i in range(nksize):
        datakun[str(nextkey[i]).rstrip('\n')]=float(nextvalue[i].rstrip('\n'))

except FileNotFoundError:
    print("fileNoExist")

dic3 = dict()    

for loop in range(PROBLEM):
    print("loop="+str(loop+1)+"/"+str(PROBLEM))
    quiz=""
    print(str(meta[loop]))
    with open("./"+str(meta[loop])+".txt") as f:
        for line in f:
            quiz=quiz+line
    quiz3=""
    
    for k in range(len(quiz)):   
        if is_sp(quiz[k]) > 0:
            quiz3+=" "+quiz[k]+" "
        else:
            quiz3+=quiz[k]
    quiz=quiz3    
    quiz2 = quiz.split()
    dic2 = dict()

    #print(str(quiz2))
    
    for xx in range(len(meta)):
        sum=1.0
        for xxx in quiz2:
            if str(xxx)=="?":
                continue
            dist = Levenshtein.distance(str(meta[xx]).upper(), str(xxx).upper())                
            if dist < 1:
                sum=1.0
                break 
            tmp2=str(meta[xx])+","+str(xxx)
            if tmp2 in datakun:
                sum*=datakun[tmp2]
                if NoAns[str(meta[xx])] > TABOO or NoAns[xxx] > TABOO:
                    sum/=datakun[tmp2]
        dic2[str(meta[xx])]=sum
    g = sorted(dic2.items(), key=lambda x: x[1], reverse=True)[:5]
    x_all, y_all = zip(*g)        
    for i in range(5):
        print(str(x_all[i]))
        dic3[str(loop)+","+str(i)]=str(x_all[i])

for loop in range(PROBLEM):
    print("loop="+str(loop+1)+"/"+str(PROBLEM))
    quiz=""
    print(str(meta[loop]))
    with open("./"+str(meta[loop])+".txt") as f:
        for line in f:
            quiz=quiz+line
    quiz3=""
    
    for k in range(len(quiz)):   
        if is_sp(quiz[k]) > 0:
            quiz3+=" "+quiz[k]+" "
        else:
            quiz3+=quiz[k]
    quiz=quiz3    
    quiz2 = quiz.split()
    for i in range(5):
        for x in quiz2:
            tmp1=str(x)+","+str(dic3[str(loop)+","+str(i)])
            tmp2=str(dic3[str(loop)+","+str(i)])+","+str(x)
            if tmp1 in datakun:
                datakun[tmp1]+=0.1
            else:
                datakun[tmp1]=1
            
            if tmp2 in datakun:
                datakun[tmp2]+=0.1
            else:
                datakun[tmp2]=1
            
key_file = open("nextkey.txt", "wt")
value_file = open("nextvalue.txt", "wt")
for mykey, myvalue in datakun.items():
    key_file.write(str(mykey)+"\n")
    value_file.write(str(myvalue)+"\n")    
