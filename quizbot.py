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

PROBLEM = 1
TABOO = 500

translator = Translator()

def is_ja(s):
    return True if re.search(r'[ぁ-んァ-ン]', s) else False

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
            
    talk = strr.split()

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

for loop in range(PROBLEM):
    quiz=""
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
        sp=0
        if quiz[k] == ".":
            sp=1
        elif quiz[k] == ",":
            sp=1 
        elif quiz[k] == "?":
            sp=1            
        elif quiz[k] == '"':
            sp=1        
        elif quiz[k] == "'":
            sp=1
        elif quiz[k] == "[":
            sp=1
        elif quiz[k] == "]":
            sp=1
        elif quiz[k] == "(":
            sp=1            
        elif quiz[k] == ")":
            sp=1            
        elif quiz[k] == ":":
            sp=1            
        elif quiz[k] == ";":
            sp=1            
        elif quiz[k] == "/":
            sp=1
        elif quiz[k] == "!":
            sp=1
        elif quiz[k] == "#":
            sp=1            
        elif quiz[k] == "&":
            sp=1
        elif quiz[k] == "$":
            sp=1
        elif quiz[k] == "%":
            sp=1

        if sp > 0:
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
        sum=0
        for xxx in quiz2:
            dist = Levenshtein.distance(str(train_num[xx+1]).upper(), str(xxx).upper())                
            if dist < 1:
                sum=0
                break 
            tmp2=str(train_num[xx+1])+","+str(xxx)
            if tmp2 in datakun:
                sum+=datakun[tmp2]
                if NoAns[train_num[xx+1]] > TABOO or NoAns[xxx] > TABOO:
                    sum -= datakun[tmp2]
                #else:
                    #if str(train_num[xx+1])=="LionelMessi":
                        #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))
                    #if str(train_num[xx+1])=="Recursion(ComputerScience)":
                        #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))                            
                #if str(train_num[xx+1])=="NetworkFlowProblem":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))
                #if str(train_num[xx+1])=="Cucumber":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))    
                #if str(train_num[xx+1])=="Tomato":
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
    plt.figure(figsize= (15,6))
    plt.bar(x_all, y_all)
    plt.show()
