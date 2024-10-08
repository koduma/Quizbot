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

PROBLEM = 26
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

for l in range(len(meta)):
    if NoAns[meta[l]] > TABOO:
        print("ERROR:"+str(meta[l]))
        #s=str(meta[l])
        #os.rename(s+".txt",s.upper()+".txt")


nksize=0

try:
   with open('./ndict.txt') as f:
    for line in f:
        nksize+=1
        key=""
        value=""
        number=False
        stopper=0
        for i in range(len(line)):
            if line[i]=="@":
                stopper=i
        for i in range(len(line)):
            if i==stopper:
                number=True
                continue
            if number == False:
                key+=line[i]
            else:
                value+=line[i]       
        datakun[str(key).rstrip('\n')]=float(str(value).rstrip('\n'))

except FileNotFoundError:
    print("fileNoExist")


ok=0
ng=0
mode=""

print("testcase?(y/n)=")

mode=input()

if mode=="n":
    PROBLEM=1
else:
    PROBLEM=26

for loop in range(PROBLEM):
    quiz=""

    if mode=="y":
        with open('./testcase/quiz'+str(loop+1)+'.txt') as f:
            for line in f:
                quiz=quiz+line
    else:
        with open('./quiz.txt') as f:
            for line in f:
                quiz=quiz+line
    if is_ja(quiz) == False:
        quiz_ja = translator.translate(quiz, src='en', dest='ja').text
    else:
        quiz_ja=quiz
        quiz=translator.translate(quiz_ja, src='ja', dest='en').text

    quiz3=""
    
    for k in range(len(quiz)):   
        if is_sp(quiz[k]) > 0:
            quiz3+=" "+quiz[k]+" "
        else:
            quiz3+=quiz[k]
    quiz=quiz3    
    quiz2 = quiz.split()
    print("Quiz_ja:\n"+quiz_ja)
    print("\n")
    print("Quiz_en:\n"+quiz)
    sumsum=0
    maxsum=0
    ans=""
    dic2 = dict()
    
    for xx in range(counter-1):
        sum=1.0
        for xxx in quiz2:
            if str(xxx)=="?":
                continue
            dist = Levenshtein.distance(str(train_num[xx+1]).upper(), str(xxx).upper())                
            if dist < 1:
                sum=0
                break 
            tmp2=str(train_num[xx+1])+","+str(xxx)
            if tmp2 in datakun:
                sum*=datakun[tmp2]
                if NoAns[train_num[xx+1]] > TABOO or NoAns[xxx] > TABOO:
                    sum/=datakun[tmp2]
                #else:
                    #if str(train_num[xx+1])=="LionelMessi":
                        #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))
                    #if str(train_num[xx+1])=="Recursion(ComputerScience)":
                        #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))                            
                #if str(train_num[xx+1])=="Printer(Computing)":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))
                #if str(train_num[xx+1])=="CHINA":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))    
                #if str(train_num[xx+1])=="USSR":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))  
                #if str(train_num[xx+1])=="EiffelTower":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))      
                #if str(train_num[xx+1])=="Recursion":
                    #print(str(tmp2)+",score="+str(sum))
                #if str(train_num[xx+1])=="BeamStackSearch":
                    #print(str(tmp2)+",score="+str(sum))
                #if str(train_num[xx+1])=="BeamSearch":
                    #print(str(tmp2)+",score="+str(sum))
        dic2[str(train_num[xx+1])]=sum
        if sum>maxsum:
            maxsum=sum
            ans=train_num[xx+1]
    g = sorted(dic2.items(), key=lambda x: x[1], reverse=True)[:5]
    print(g)
    x_all, y_all = zip(*g)
    for i in range(5):
        sumsum+=y_all[i]
    ans_ja = translator.translate(str(ans), src='en', dest='ja').text
    print("Answer_ja:"+ans_ja)    
    print("Answer_en:"+str(ans))

    if mode == "y":
        truth=""
        with open('./testcase/ans'+str(loop+1)+'.txt') as f:
            for line in f:
                truth=truth+line
        if str(truth)==str(ans):
            ok+=1
        else:
            ng+=1
    score=maxsum/(sumsum+1)
    print("Score:"+'{:.3f}'.format(score))
    if score < 0.3:
        print("Eval:F") 
    else:
        if score < 0.4:
            print("Eval:C")
        else:
            if score < 0.6:
                print("Eval:A")
            else:
                print("Eval:S")               
    print("Words:"+str(counter))
    mem = psutil.virtual_memory() 
    print("Mem:"+str(mem.percent))
    print("Num:"+str(len(meta)))
    #plt.figure(figsize= (15,6))
    #plt.bar(x_all, y_all)
    #plt.show()

if mode=="y":
    print(str("AC=")+str(ok)+",WA="+str(ng))    
