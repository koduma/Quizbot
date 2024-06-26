import os
import matplotlib.pyplot as plt

import psutil

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
TABOO = 20

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
        #print(str("c="+str(c)))
        #if str(title)=="Empiricism":
            #print("AAAAAAAAAAAAAAAAAAAAAAAAA"+tmp5+"====="+str(c))
        if tmp5 in datakun:
            datakun[tmp5]+=2
        else:
            datakun[tmp5]=2

#for c in range(counter-2):
#    tmp = str(train_num[c+1])+","+str(train_num[c+2])
#    if tmp in data:
#        data[tmp]+=1
#    else:
#        data[tmp]=1
    
    for c in range(now[l]-1,counter-1):
        for c2 in range(c+1,counter-1):
            tmp = str(train_num[c+1])+","+str(train_num[c2+1])
            #print(tmp)
            if tmp in datakun:
                datakun[tmp]+=1
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

def scoring(s1,s2):
    sum=0
    for x in s1:
        for y in s2:
         tmp=str(x)+","+str(y)  
         if tmp in datakun:
             sum+=datakun[tmp]
             if NoAns[str(x)] > TABOO or NoAns[str(y)] > TABOO:
                 sum-=datakun[tmp]

    return sum

for loop in range(PROBLEM):
    quiz=""
    with open('./quiz.txt') as f:
        for line in f:
            quiz=quiz+line            
    quiz2 = quiz.split()
    print("Quiz:"+quiz)
    sumsum=0
    maxsum=0
    ans=""
    dic2 = dict()
    
    for xx in range(len(meta)):
        if str(meta[xx]) in quiz2:
            continue
        xi=""
        with open("./"+str(meta[xx])+".txt") as f:
            for line in f:
                xi=xi+line
        xi=xi.split()
        ret=scoring(xi,quiz2)/(len(xi)+1.0)            
        dic2[str(meta[xx])]=ret
        if ret>maxsum:
            maxsum=ret
            ans=str(meta[xx])
    g = sorted(dic2.items(), key=lambda x: x[1], reverse=True)[:5]
    print(g)
    x_all, y_all = zip(*g)
    for i in range(5):
        sumsum+=y_all[i]
    print("Answer:"+str(ans))
    score=maxsum/(sumsum+0.001)
    print("Score:"+'{:.3f}'.format(score))               
    print("Words:"+str(counter))
    mem = psutil.virtual_memory() 
    print("Mem:"+str(mem.percent))
    print("Num:"+str(len(meta)))
    plt.figure(figsize= (10,6))
    plt.bar(x_all, y_all)
    plt.show()
