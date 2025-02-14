import math
import os
import matplotlib.pyplot as plt
import psutil
import Levenshtein
from googletrans import Translator
import re
import sys
import random
import warnings

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
WA=[]

PROBLEM = 75
TABOO = 1000

translator = Translator()

warnings.simplefilter('ignore')

def fix_expression(s: str) -> str:
    s = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', s)
    s = re.sub(r'\s*([*/+\-^])\s*', r'\1', s)
    s = re.sub(r'(?<!\d)\s*\.\s*(?!\d)', r' . ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

def calculator(s):
    ev = 0
    ret = ""
    ans = None

    safe_globals = {
        "__builtins__": None,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "abs": abs
    }

    for i in range(len(s)):
        st = s[i]
        for j in range(i+1, len(s)):
            st += s[j]
            try:
                if "=" in st:
                    parts = st.split("=")
                    if len(parts) != 2:
                        continue

                    left_expr = parts[0].strip()
                    right_expr = parts[1].strip()

                    if not left_expr or not right_expr:
                        continue

                    left_val = eval(left_expr, safe_globals, {})
                    right_val = eval(right_expr, safe_globals, {})

                    if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                        if abs(left_val - right_val) < 1e-9:
                            if j - i >= ev:
                                ev = j - i
                                ret = st
                                if ans != "False":
                                    ans = "True"
                        else:
                            if j - i >= ev:
                                ev = j - i
                                ret = st
                                ans = "False"                
                else:
                    result = eval(st, safe_globals, {})
                    if isinstance(result, (int, float, complex)):
                        if j - i >= ev:
                            ev = j - i
                            ret = st
                            ans = result
            except Exception:
                continue

    return ev, ans
def is_ja(s):
    return True if re.search(r'[ぁ-んァ-ン]', s) else False

def is_include(s,t):
    #if s!="SolarSystem":
        #return False
    if s not in NoAns or t not in NoAns:
        #print(str(0)+",s="+s+",t="+t)
        return False
    if NoAns[s]>TABOO or NoAns[t]>TABOO:
        #print(str(1)+",s="+s+",t="+t)
        return False
    if (s+","+t) not in datakun or (t+","+s) not in datakun:
        #print(str(2)+",s="+s+",t="+t)
        return False        
    if datakun[s+","+t] >5 or datakun[t+","+s]>5:
        if len(s)>=len(t):
            #print(str(3)+",s="+s+",t="+t)
            return (t.upper() in s.upper())
        else:
            #print(str(4)+",s="+s+",t="+t)
            return (s.upper() in t.upper())
    #print(str(5)+",s="+s+",t="+t)
    return False

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
    elif s == "{":
        sp=1
    elif s == "}":
        sp=1
    elif s=="“":
        sp=1
    elif s=="’":
        sp=1
    elif s=="”":
        sp=1
    elif s==">":
        sp=1
    elif s=="<":
        sp=1
    elif s=="+":
        sp=1
    elif s=="*":
        sp=1
    elif s=="|":
        sp=1
    elif s=="¥":
        sp=1
    elif s=="@":
        sp=1
    elif s=="=":
        sp=1
    elif s=="^":
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

    if str(title)=="300000kms":
        title="300000km/s"
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

    #for i in range(len(talk)):
        #talk[i]=str(talk[i]).lower()
    
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
        tmp5 = str(title)+","+str(c)
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
            #if train_num[c+1]=="206":
                #print(tmp)
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
                

    if (l+1)%100 == 0 or (l+1)==len(meta):
        print("params="+str(len(datakun))+",train="+str(l+1)+"/"+str(len(meta)))
        #b = sys.getsizeof(datakun)
        #b += sum(map(sys.getsizeof, datakun.values())) + sum(map(sys.getsizeof, datakun.keys()))
        #kb = b / 1024
        #mb = kb / 1024
        #gb = mb / 1024
        #gb2 = format(gb, '.2f')
        #print("params="+str(len(datakun))+","+str(gb2)+"GB"+",train="+str(l+1)+"/"+str(len(meta)))

ok=0
ng=0
mode=""

print("mode?(1:keyboard,2:txt,3:testcase,4:generator)=",end="")

mode=input()

if mode=="3":
    PROBLEM=75
else:
    PROBLEM=1

def solve(loop,o,add,q):

    global ok,ng
    
    quiz=""

    if mode=="3":
        with open('./testcase/quiz'+str(loop+1)+'.txt') as f:
            for line in f:
                quiz=quiz+line
    elif mode=="2":
        with open('./quiz.txt') as f:
            for line in f:
                quiz=quiz+line
    elif mode=="1" or mode=="4":
        quiz=q
    if len(quiz)==0:
        sys.exit()
        
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
    if len(add)>0:
        quiz+=add
    quiz2 = quiz.split()

    #for i in range(len(quiz2)):
        #quiz2[i]=str(quiz2[i]).lower()
    
    if o==True:
        print("Quiz_ja:\n"+quiz_ja)
        print("\n")
        print("Quiz_en:\n"+quiz)
        print("\n")
    sumsum=0
    maxsum=0
    ans=""
    dic2 = dict()

    hint=""
    maxhit=1

    #for xxx in range(len(quiz2)):
        #if quiz2[xxx] in NoAns:
            #if NoAns[quiz2[xxx]]<=TABOO:
                #print(str(quiz2[xxx])+"="+str(NoAns[quiz2[xxx]]))

    for xxx in range(len(quiz2)):
        hit=0
        if len(quiz2[xxx])>1:
            for x in range(len(quiz2)):
                if quiz2[xxx] in quiz2[x]:
                    if quiz2[xxx] in NoAns:
                        if NoAns[quiz2[xxx]] <= TABOO:
                            hit+=1            
        if hit>maxhit:
            maxhit=hit
            hint=quiz2[xxx]
    #print("hint="+str(hint)+",maxhit="+str(maxhit))

    include=dict()
    
    for xx in range(counter-1):
        sum=1.0
        tmp=str(train_num[xx+1])+","+str(hint)
        if tmp in datakun:
            sum=float(pow(2,maxhit-1))
            #if str(train_num[xx+1])=="IrreversibleProcess":
                #print("sum="+str(sum))
        for xxx in quiz2:
            if str(xxx)=="?" or str(xxx)=="!":
                continue
            dist = Levenshtein.distance(str(train_num[xx+1]).upper(), str(xxx).upper())                
            if dist < 1:
                sum=1.0
                break
            bbb=False
            if is_include(str(train_num[xx+1]),str(xxx))==True:
                #include[str(train_num[xx+1])]=True
                for w1 in range(len(quiz2)):
                    if bbb==True:
                        break
                    for w2 in range(w1+1,len(quiz2)):
                        w3=str(quiz2[w1])+str(quiz2[w2])
                        w4=str(quiz2[w2])+str(quiz2[w1])
                        d1=Levenshtein.distance(str(train_num[xx+1]).upper(), str(w3).upper())
                        d2=Levenshtein.distance(str(train_num[xx+1]).upper(), str(w4).upper())
                        if d1 <=1 or d2 <=1:
                            include[str(train_num[xx+1])]=True
                            bbb=True
            tmp2=str(train_num[xx+1])+","+str(xxx)
            if tmp2 not in datakun:
                sum/=1.2
            if tmp2 in datakun:
                sum*=datakun[tmp2]
                if NoAns[train_num[xx+1]] > TABOO or NoAns[xxx] > TABOO:
                    if xxx != "water":
                        sum/=datakun[tmp2]        
                #else:
                    #if str(train_num[xx+1])=="IrreversibleProcess":
                        #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))                            
                #if str(train_num[xx+1])=="Synchronicity":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))
                #if str(train_num[xx+1])=="Metaphysics":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))
                #if str(train_num[xx+1])=="Ones'Complement":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))
                #if str(train_num[xx+1])=="PhilosophyOfLogic":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))    
        dic2[str(train_num[xx+1])]=round(sum,2)
        if sum>maxsum:
            maxsum=sum
            ans=train_num[xx+1]
    tmp_quiz=quiz        
    tmp_quiz2=fix_expression(tmp_quiz)
    i1,i2=calculator(tmp_quiz2)
    #print("i1="+str(i1)+",len(quiz)="+str(len(quiz)))
    if i1>=3:
        sco=0.0
        if float(i1/len(tmp_quiz2))<0.1:
            sco=float(pow(2.0,i1*10/len(tmp_quiz2)))
        else:
            sco=float(pow(2.0,i1*100/len(tmp_quiz2)))
        dic2[str(i2)]=round(sco,2)
        #print("i2="+str(i2)+",i1="+str(i1)+",len="+str(len(quiz))+",sco="+str(sco))
        if sco>maxsum:
            maxsum=round(sco,2)
            ans=str(i2)
    g = sorted(dic2.items(), key=lambda x: x[1], reverse=True)[:5]
    x_all, y_all = zip(*g)
    if str(x_all[0]) in include:
        return -1,str(x_all[0])
    print("Top5:",end="")    
    print(g)
    if y_all[0] < 1.01:
        ans="Unknown"
    ans_ja = translator.translate(str(ans), src='en', dest='ja').text
    print("Answer_ja:"+ans_ja)    
    print("Answer_en:"+str(ans))
    if mode == "3":
        truth=""
        with open('./testcase/ans'+str(loop+1)+'.txt') as f:
            for line in f:
                truth=truth+line
        if str(truth).lower()==str(ans).lower():
            ok+=1
            print("State:AC")
        else:
            WA.append(loop+1)
            ng+=1
            print("State:WA")
            print("Truth:"+str(truth))
    score=maxsum/(len(quiz2)+1)
    print("Score:"+'{:.3f}'.format(score))
    if score < 1.0:
        print("Eval:F") 
    else:
        if score < 10.0:
            print("Eval:C")
        else:
            if score < 100.0:
                print("Eval:A")
            else:
                if mode=="4":
                    for yyy in quiz2:
                        tp1 = str(ans)+","+str(yyy)
                        tp2 = str(yyy)+","+str(ans)
                        if tp1 in datakun:
                            datakun[tp1]+=1
                        else:
                            datakun[tp1]=1
                        if tp2 in datakun:
                            datakun[tp2]+=1
                        else:
                            datakun[tp2]=1
                print("Eval:S")               
    print("Words:"+str(counter))
    mem = psutil.virtual_memory() 
    print("Mem:"+str(mem.percent))
    print("Num:"+str(len(meta)))
    if mode=="2":
        plt.figure(figsize= (15,6))
        plt.bar(x_all, y_all)
        plt.show()

    return 0,"end"


if mode=="2":
    o=True
    add=""
    q=""
    while True:
        a,b=solve(0,o,add,q)
        if a==0:
            break
        else:
            o=False
            add+=" "+str(b)

elif mode=="3":

    for l in range(PROBLEM):
        print("------------------------------------------------------------------")
        print(str("Problem:")+str(l+1)+"/"+str(PROBLEM))
        o=True
        add=""
        q=""
        while True:
            a,b=solve(l,o,add,q)
            if a==0:
                break
            else:
                o=False
                add+=" "+str(b)
elif mode=="1":
    o=True
    add=""
    while True:
        if o==True:
            print("------------------------------------------------------------------")
            print("Input_Quiz:")
            q=input()
        a,b=solve(0,o,add,q)
        if a!=0:
            o=False
            add+=" "+str(b)
        else:
            o=True
            add=""
            
elif mode=="4":
    o=True
    add=""
    while True:
        if o==True:
            print("------------------------------------------------------------------")
            q=""
            d=dict()
            c=0
            while True:
                if c >=10:
                    break
                r=random.randint(0, len(meta)-1)
                if r not in d:
                    q+=meta[r]+" "
                    d[r]=1
                    c+=1
            print("Input_Quiz:")
            print(q)
        a,b=solve(0,o,add,q)
        if a!=0:
            o=False
            add+=" "+str(b)
        else:
            o=True
            add=""            


if mode=="3":
    print(str("AC=")+str(ok)+",WA="+str(ng))
    if len(WA)!=0:
        print("WA_Problem:",end="")
        for i in range(len(WA)-1):
            print(str(WA[i])+",",end="")
        print(str(WA[len(WA)-1]))
