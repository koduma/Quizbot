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
import sympy
from sympy import sympify, Eq, solve

strr=""
meta=""

train = dict()
train_num = dict()
datakun = dict()
NoAns = dict()
Topk = dict()

with open("./metadata2.txt") as f:
    for line in f:
        meta=meta+line

meta=meta.split()

counter=1
now=[1]
WA=[]

PROBLEM = 91
TABOO = 1000#7000
RARE = 140#400
docs = 0
perfect = 1000000

translator = Translator()

warnings.simplefilter('ignore')

def match_score(q,definition):
    q1=q.lower().split()
    for i in range(len(q1)):
        if str(q1[i]) in NoAns:
            if NoAns[str(q1[i])] > TABOO:
                q = q.replace(str(q1[i]), "")
            
    d1=definition.lower().split()
    for i in range(len(d1)):
        if str(d1[i]) in NoAns:
            if NoAns[str(d1[i])] > TABOO:
                definition = definition.replace(str(d1[i]), "")

    q_words = set(q.lower().split())
    d_words = set(definition.lower().split())
    retx=len(q_words&d_words)
    if retx <= 1:
        retx=1
    return retx

def is_english_word(text):
    if re.fullmatch(r'[a-zA-Z]+', text):
        return 1
    elif re.fullmatch(r'[0-9]+', text):
        return 0
    else:
        return 0

def extract_unit(s: str) -> str:
    pattern = r'\d+(?:\.\d+)?\s*([a-zA-Z]+)'
    match = re.search(pattern, s)
    return match.group(1) if match else ""

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

    lower_s = s.lower()
    lower_s = lower_s.replace('wide', 'width')
    lower_s = lower_s.replace('high', 'height')
    lower_s = lower_s.replace('sizes', 'size')
    lower_s = lower_s.replace('diameters', 'diameter')

    pattern = r"((?:\d+\.\d+)|(?:\d+/\d+)|(?:\d+))(?:\s*(cm|m))?"

    matches = re.findall(pattern, s)

    xx=0

    if all(word in lower_s for word in ["rectangle", "area", "height", "width"]) or all(word in lower_s for word in ["rectangle", "area","size"]):
        if matches:
            xx=1
        
    if all(word in lower_s for word in ["rectangle", "perimeter", "height", "width"]) or all(word in lower_s for word in ["rectangle", "perimeter", "size"]):
        if matches:
            xx=2

    if all(word in lower_s for word in ["circle", "area", "diameter"]):
        if matches:
            xx=3

    if all(word in lower_s for word in ["circle", "area", "radius"]):
        if matches:
            xx=4

    if all(word in lower_s for word in ["circle", "circumference", "radius"]):
        if matches:
            xx=5

    if all(word in lower_s for word in ["circle", "circumference", "diameter"]):
        if matches:
            xx=6

    meter=False
    centimeter=False
    ls=lower_s.split()
    for i in range(len(ls)):
        if ls[i]=='meters':
            meter=True
        if ls[i]=='meter':
            meter=True
        if ls[i]=='metre':
            meter=True
        if ls[i]=='metres':
            meter=True
        if ls[i]=='centimeters':
            centimeter=True
        if ls[i]=='centimeter':
            centimeter=True
        if ls[i]=='centimetre':
            centimeter=True
        if ls[i]=='centimetres':
            centimeter=True 
    if xx ==1 or xx==2:
        first_match = matches[0]
        second_match = matches[1]
        number1, unit1 = first_match[0], first_match[1]
        number2, unit2 = second_match[0], second_match[1]
        n1 = eval(number1, safe_globals, {})
        n2 = eval(number2, safe_globals, {})
        number1=float(n1)
        number2=float(n2)
        if unit1=='cm':
            number1=number1/100.0
        if unit2=='cm':
            number2=number2/100.0
        ca=0
        if xx == 1:
            ca=number1*number2
            if meter==True:
                return len(s), f"{ca} m^2"
            elif centimeter==True:
                return len(s), f"{ca*10000.0} cm^2"
            else:
                return len(s),f"{ca}"
        if xx == 2:
            ca=2.0*(number1+number2)
            if meter==True:
                return len(s), f"{ca} m"
            elif centimeter==True:
                return len(s), f"{ca*100.0} cm"
            else:
                return len(s),f"{ca}"
    if xx == 3 or xx == 4:            
        first_match = matches[0]
        number1, unit1 = first_match[0], first_match[1]
        n1 = eval(number1, safe_globals, {})
        number1=float(n1)
        if unit1=='cm':
            number1=number1/100.0
        ca=0    
        if xx == 3:
            ca=(1.0/2.0)*(number1)*(1.0/2.0)*(number1)
        if xx == 4:
            ca=number1*number1
        if meter==True:
            return len(s), f"{ca}π m^2"
        elif centimeter==True:
            return len(s), f"{ca*10000.0}π cm^2"
        else:
            return len(s),f"{ca}π"

    if xx == 5 or xx == 6:            
        first_match = matches[0]
        number1, unit1 = first_match[0], first_match[1]
        n1 = eval(number1, safe_globals, {})
        number1=float(n1)
        if unit1=='cm':
            number1=number1/100.0
        ca=0    
        if xx == 5:
            ca=2.0*number1
        if xx == 6:
            ca=number1
        if meter==True:
            return len(s), f"{ca}π m"
        elif centimeter==True:
            return len(s), f"{ca*100.0}π cm"
        else:
            return len(s),f"{ca}π"
    
    #print(s)

    pt=0

    if 'of' in s:
        s = s.replace('of','*')
    if 'divisible' in s:
        s = s.replace('divisible','%')
        pt=1
    if 'by' in s:
        s = s.replace('by',' ')
    if 'divided' in s:
        s = s.replace('divided','*1.0/')
    unit=extract_unit(s)
    if unit in s and len(unit)>0:
        s = s.replace(unit,'')

    for i in range(len(s)):
        st = s[i]
        for j in range(i+1, len(s)):
            add=""
            if s[j]=='%':
                add="/100.0"
            else:
                add=s[j]    
            st += add    
            try:
                if "=" in st:
                    parts = st.split("=")
                    if len(parts) != 2:
                        if ans == None:
                            ans="False"
                        continue

                    left_expr = parts[0].strip()
                    right_expr = parts[1].strip()

                    if not left_expr or not right_expr:
                        continue
                        
                    try:
                        left_sym = sympify(left_expr)
                        right_sym = sympify(right_expr)
                        free_vars = left_sym.free_symbols.union(right_sym.free_symbols)
                        if not free_vars:
                            try:
                                left_val = float(left_sym.evalf())
                                right_val = float(right_sym.evalf())
                                if j - i >= ev:
                                    ev = j - i
                                    ret = st
                                    if abs(left_val - right_val) < 1e-9:
                                        ans = "True"
                                    else:
                                        ans = "False"
                            except Exception:
                                pass
                        else:
                            eq = Eq(left_sym, right_sym)
                            try:
                                sol = solve(eq, list(free_vars))
                                if j - i >= ev:
                                    ev = j - i
                                    ret = st
                                    if isinstance(sol, list):
                                        ans = sol[0]
                                    else:
                                        ans = sol
                            except Exception:
                                pass
                    except Exception:
                        pass

                    left_val = eval(left_expr, safe_globals, {})
                    right_val = eval(right_expr, safe_globals, {})

                    if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                        #print("l="+str(left_expr)+",r="+str(right_expr))
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
                        #print("st="+str(st)+",result="+str(result)+",j-i="+str(j-i)+",pt="+str(pt))
                        if j - i >= ev:
                            ev = j - i
                            ret = st
                            if pt==0:
                                #print(str("st=")+str(st)+",result="+str(result))
                                ans = result
                            else:
                                if result==0:
                                    ans="True"
                                else:
                                    ans="False"
            except Exception:
                continue

    ans=str(ans)
    if len(unit)>0:
        ans+=str(unit)
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

#print("load?(y/n)=",end="")
#ld=input()

for l in range(len(meta)):

    #if ld=='y':
        #nksize=0
        #nextkey=[]
        #nextvalue=[]
        #try:
            #with open('./nextkey.txt') as f:
                #for line in f:
                    #nksize+=1
                    #line.rstrip('\n')
                    #nextkey.append(str(line))
            #with open('./nextvalue.txt') as f:
                #for line in f:
                    #line.rstrip('\n')
                    #nextvalue.append(str(line))
            #for i in range(nksize):
                #datakun[str(nextkey[i]).rstrip('\n')]=float(nextvalue[i].rstrip('\n'))
        #except FileNotFoundError:
            #print("fileNoExist")
        #break

    strr=""
    
    title = meta[l]

    if l<=10040:
        docs=docs+1
        with open("./"+title+".txt") as f:
            for line in f:
                strr=strr+line
    elif l<70000:
        break
        docs=docs+1
        with open("./getdata/"+title+".txt") as f:
            for line in f:
                strr=strr+line
    else:
        break
        
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
        #print("params="+str(len(datakun))+",train="+str(l+1)+"/"+str(len(meta)))
        b = sys.getsizeof(datakun)
        b += sum(map(sys.getsizeof, datakun.values())) + sum(map(sys.getsizeof, datakun.keys()))
        kb = b / 1024
        mb = kb / 1024
        gb = mb / 1024
        gb2 = format(gb, '.2f')
        print("params="+str(len(datakun))+",mem="+str(gb2)+"GB"+",train="+str(l+1)+"/"+str(len(meta)))


#nk_w=""
#nv_w=""
#siz=0

#for key, value in datakun.items():
    #sr1=str(key)+'\n'
    #sr2=str(value)+'\n'
    #nk_w+=sr1
    #nv_w+=sr2
    #if siz % 1000 == 0:
        #with open("nextkey.txt", mode="a", encoding="utf-8") as f:
            #f.write(nk_w)
        #with open("nextvalue.txt", mode="a", encoding="utf-8") as f:
            #f.write(nv_w)
        #nk_w=""
        
    #siz+=1

ok=0
ng=0
mode=""

print("mode?(1:keyboard,2:txt,3:testcase,4:generator)=",end="")

mode=input()

if mode=="3":
    PROBLEM=91
else:
    PROBLEM=1

def quiz_solve(loop,o,add,q):

    global ok,ng

    global Topk
    
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

    quiz_ja=""

    try:
        if is_ja(quiz) == False:
            quiz_ja = translator.translate(quiz, src='en', dest='ja').text
        else:
            quiz_ja=quiz
            quiz=translator.translate(quiz_ja, src='ja', dest='en').text

    except Exception:
        pass

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

    printed = [False] * 15

    ngram = dict()

    for k1 in range(len(quiz2)):
        if k1+2 < len(quiz2):
            kt=str(quiz2[k1])+str(quiz2[k1+1])+str(quiz2[k1+2])
            ngram[str(kt).lower()]=1

    for xx in range(counter-1):
        per=xx/(counter+1)
        if per < 0.1:
            idx = 0
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.2:
            idx = 1
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.3:
            idx = 2
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.4:
            idx = 3
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.5:
            idx = 4
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.6:
            idx = 5
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.7:
            idx = 6
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.8:
            idx = 7
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.9:
            idx = 8
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        else:
            idx = 9
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        if xx == counter-2:
            print("complete")
        sum=1.0
        if str(train_num[xx+1]) in NoAns:
            if NoAns[str(train_num[xx+1])] > TABOO:
                continue
        cnt=-1
        tmp=str(train_num[xx+1])+","+str(hint)
        if tmp in datakun:
            sum=float(pow(2,maxhit-1))
            #if str(train_num[xx+1])=="IrreversibleProcess":
                #print("sum="+str(sum))
        strl=str(train_num[xx+1]).lower()
        if strl in ngram:
            sum=1.0
            continue
        for xxx in quiz2:
            cnt+=1
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
            if tmp2 in datakun and (str(xxx) in NoAns):
                weight=1.0
                if cnt < 5:
                    weight=3.0
                sum*=weight*datakun[tmp2]#float(datakun[tmp2]+cnt)                    
                if NoAns[xxx] > TABOO:
                    if xxx != "water":
                        sum/=(weight*datakun[tmp2])#float(datakun[tmp2]+cnt)
                if NoAns[xxx] <= RARE and is_english_word(str(xxx)) == 1:
                    sum*=3.0
                #else:
                    #if str(train_num[xx+1])=="Neanderthal":
                        #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))                            
                #if str(train_num[xx+1])=="Apostrophe":
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
    if i1>=2:
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
    s7=100
    gg = sorted(dic2.items(), key=lambda item: item[1], reverse=True)
    if len(gg) < 100:
        s7=len(gg)
    for ab in range(s7):
        keyx,value=gg[ab]
        sd=""
        sd2=""
        try:
            with open("./"+str(keyx)+".txt") as f:
                for line in f:
                    sd=sd+line
            for hj in range(len(sd)):
                if is_sp(sd[hj]) > 0:
                    sd2+=" "+sd[hj]+" "
                else:
                    sd2+=sd[hj]
            nscore=round(value*match_score(quiz,sd2),2)
            dic2[str(keyx)]=nscore
            if nscore>maxsum:
                maxsum=nscore
                ans=str(keyx)
        except FileNotFoundError:
            pass
    g = sorted(dic2.items(), key=lambda x: x[1], reverse=True)[:5]
    x_all, y_all = zip(*g)
    if str(x_all[0]) in include:
        return -1,str(x_all[0])
    #print("len(dic2)=", len(dic2))
    Topk.clear()
    Topk=dic2.copy()
    #print("Top5:",end="")    
    #print(g)
    #ans_ja=""
    #if y_all[0] < 1.01:
        #ans="Unknown"
    #try:
        #ans_ja = translator.translate(str(ans), src='en', dest='ja').text
    #except Exception:
        #pass
        
    #print("Answer_ja:"+ans_ja)    
    #print("Answer_en:"+str(ans))
    #if mode == "3":
        #truth=""
        #with open('./testcase/ans'+str(loop+1)+'.txt') as f:
            #for line in f:
                #truth=truth+line
        #if str(truth).lower()==str(ans).lower():
            #ok+=1
            #print("State:AC")
        #else:
            #WA.append(loop+1)
            #ng+=1
            #print("State:WA")
            #print("Truth:"+str(truth))
    imp=1.0
    for ik in range(len(quiz2)):
        if str(quiz2[ik]) in NoAns:
            if NoAns[str(quiz2[ik])] <= TABOO:
                imp+=1.0
    score=maxsum/imp

    return 0,"end"

def first_floor(qn,aaa,adder,doprint):#qnは記号分離したクイズ文
    global Topk
    quiz=qn
    quiz2=qn.split()
    maxk=500
    if len(Topk)<500:
        maxk=len(Topk)
    g = sorted(Topk.items(), key=lambda x: x[1], reverse=True)[:maxk]
    #print("len(g)="+str(len(g))+",len(Topk)="+str(len(Topk))+",maxk="+str(maxk))

    hint=""
    maxhit=1

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
            
    ngram = dict()

    for k1 in range(len(quiz2)):
        if k1+2 < len(quiz2):
            kt=str(quiz2[k1])+str(quiz2[k1+1])+str(quiz2[k1+2])
            ngram[str(kt).lower()]=1

    maxsum=0
    ans=""
    dic2 = dict()
    #include=dict()
    sum=1.0
    
    for xx in range(maxk):
        keyx,value=g[xx]
        #if (xx+1)%10==0:
            #print(str(xx+1)+"/"+str(maxk))
        sum=1.0
        if str(keyx) in NoAns:
            if NoAns[str(keyx)] > TABOO:
                continue
        cnt=-1
        tmp=str(keyx)+","+str(hint)
        if tmp in datakun:
            sum=float(pow(2,maxhit-1))
        strl=str(keyx).lower()
        if strl in ngram:
            sum=1.0
            continue
        for xxx in quiz2:
            cnt+=1
            if str(xxx)=="?" or str(xxx)=="!":
                continue
            #dist = Levenshtein.distance(str(keyx).upper(), str(xxx).upper())                
            #if dist < 1:
                #sum=1.0
                #break
            #bbb=False
            #if is_include(str(keyx),str(xxx))==True:
                #for w1 in range(len(quiz2)):
                    #if bbb==True:
                        #break
                    #for w2 in range(w1+1,len(quiz2)):
                        #w3=str(quiz2[w1])+str(quiz2[w2])
                        #w4=str(quiz2[w2])+str(quiz2[w1])
                        #d1=Levenshtein.distance(str(keyx).upper(), str(w3).upper())
                        #d2=Levenshtein.distance(str(keyx).upper(), str(w4).upper())
                        #if d1 <=1 or d2 <=1:
                            #include[str(keyx)]=True
                            #bbb=True                
            tmp2=str(keyx)+","+str(xxx)
            if tmp2 not in datakun:
                sum/=1.2
            if tmp2 in datakun and (str(xxx) in NoAns):
                weight=1.0
                if cnt < 5:
                    weight=3.0
                sum*=weight*datakun[tmp2]#float(datakun[tmp2]+cnt)
                if NoAns[xxx] > TABOO:
                    if xxx != "water":
                        sum/=(weight*datakun[tmp2])#float(datakun[tmp2]+cnt)
                if NoAns[xxx] <= RARE and is_english_word(str(xxx)) == 1:
                    sum*=3.0
                #else:
                    #if str(train_num[xx+1])=="Neanderthal":
                        #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))                            
                #if str(keyx)=="4" and (str(keyx) in NoAns) and str(quiz2[0])=="DavidHume":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[str(keyx)])+",NoAns2="+str(NoAns[xxx]))
                #if str(keyx)=="4" and (str(keyx) in NoAns) and str(quiz2[0])=="DavidHume":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[str(keyx)])+",NoAns2="+str(NoAns[xxx]))
                #if str(train_num[xx+1])=="Ones'Complement":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))
                #if str(train_num[xx+1])=="PhilosophyOfLogic":
                    #print(str(tmp2)+",score="+str(sum)+",NoAns1="+str(NoAns[train_num[xx+1]])+",NoAns2="+str(NoAns[xxx]))    
        dic2[str(keyx)]=round(sum,2)
        if sum>maxsum:
            maxsum=sum
            ans=str(keyx)
    tmp_quiz=qn        
    tmp_quiz2=fix_expression(tmp_quiz)
    i1,i2=calculator(tmp_quiz2)
    #print("i1="+str(i1)+",len(quiz)="+str(len(quiz)))
    if i1>=2:
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
    s7=100
    gg = sorted(dic2.items(), key=lambda item: item[1], reverse=True)
    #print("len(gg)="+str(len(gg)))
    if len(gg) < 100:
        s7=len(gg)
    for ab in range(s7):
        keyy,value=gg[ab]
        sd=""
        sd2=""
        try:
            with open("./"+str(keyy)+".txt") as f:
                for line in f:
                    sd=sd+line
            for hj in range(len(sd)):
                if is_sp(sd[hj]) > 0:
                    sd2+=" "+sd[hj]+" "
                else:
                    sd2+=sd[hj]
            nscore=round(value*match_score(quiz,sd2),2)
            dic2[str(keyy)]=nscore
            if nscore>maxsum:
                maxsum=nscore
                ans=str(keyy)
        except FileNotFoundError:
            pass
    ggg = sorted(dic2.items(), key=lambda x: x[1], reverse=True)[:5]
    x_all, y_all = zip(*ggg)
    if (aaa==0 and maxsum >= perfect) or doprint==True:
        print("Top5:",end="")
        print(ggg)
    if y_all[0] < 1.01:
        ans="Unknown"
    if (aaa==0 and maxsum >= perfect) or doprint==True:
        print("Answer:"+str(ans))
    imp=1.0
    for ik in range(len(quiz2)):
        if str(quiz2[ik]) in NoAns:
            if NoAns[str(quiz2[ik])] <= TABOO:
                imp+=1.0
    score=maxsum/imp

    if aaa==0 and maxsum >= perfect:
        return maxsum,str(ans),""
    else:
        return maxsum,str(ans),str(adder)
        
    return 0,"",""

def qs3(loop,add,ttt):

    global ok,ng

    global Topk
    
    quiz=""

    if loop>=0:
        with open('./testcase/quiz'+str(loop+1)+'.txt') as f:
            for line in f:
                quiz=quiz+line
        if len(quiz)==0:
            sys.exit()

    else:
        quiz=ttt

    quiz3=""
    
    for k in range(len(quiz)):   
        if is_sp(quiz[k]) > 0:
            quiz3+=" "+quiz[k]+" "
        else:
            quiz3+=quiz[k]
    quiz=quiz3
    if len(add)>0:
        quiz+=add
    quiz2 = quiz.split()#quiz2ではなく、quizがlist化する前の記号分離文
    
    ddd,uuu,ppp=first_floor(quiz,0,"",False)
    if ddd >= perfect:
        if mode == "3":
            truth=""
            with open('./testcase/ans'+str(loop+1)+'.txt') as f:
                for line in f:
                    truth=truth+line
            if str(truth).lower()==str(uuu).lower():
                ok+=1
                print("State:AC")
            else:
                WA.append(loop+1)
                ng+=1
                print("State:WA")
                print("Truth:"+str(truth))
        return
    maxk=500
    if len(Topk)<500:
        maxk=len(Topk)

    dic1=dict()
    adderk=dict()

    gx=sorted(Topk.items(), key=lambda item: item[1], reverse=True)
    gt=sorted(Topk.items(), key=lambda item: item[1], reverse=True)[:3]
    
    printed = [False] * 15

    print("ReThinking...")
    
    for i in range(maxk):
        per=i/(maxk+1)
        if per < 0.1:
            idx = 0
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.2:
            idx = 1
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.3:
            idx = 2
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.4:
            idx = 3
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.5:
            idx = 4
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.6:
            idx = 5
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.7:
            idx = 6
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.8:
            idx = 7
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        elif per < 0.9:
            idx = 8
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        else:
            idx = 9
            if printed[idx]==False:
                per*=100.0
                per=round(per,1)
                print("thinking..."+str(per)+"%")
                printed[idx]=True
        if i == maxk-1:
            print("complete")
        adder,val=gx[i]
        if str(adder) not in quiz2:
            u1,u2,u3=first_floor(quiz+" "+str(adder),i+1,str(adder),False)
            #dic1[str(u2)]=round(u1,2)
            adderk[str(u3)]=round(u1,2)
        
    s7=100
    maxsum=0
    ans=""
    gy = sorted(adderk.items(), key=lambda item: item[1], reverse=True)
    if len(gy) < 100:
        s7=len(gy)
    for ab in range(s7):
        keyy,value=gy[ab]
        sd=""
        sd2=""
        try:
            with open("./"+str(keyy)+".txt") as f:
                for line in f:
                    sd=sd+line
            for hj in range(len(sd)):
                if is_sp(sd[hj]) > 0:
                    sd2+=" "+sd[hj]+" "
                else:
                    sd2+=sd[hj]
            nscore=round(value*match_score(quiz,sd2),2)
            adderk[str(keyy)]=nscore
            if nscore>maxsum:
                maxsum=nscore
                ans=str(keyy)
        except FileNotFoundError:
            pass
    adder10 = sorted(adderk.items(), key=lambda x: x[1], reverse=True)[:3]

    next_quiz=""
    
    for bc in range(len(adder10)):
        keyy,value=adder10[bc]
        print("adder3,keyword="+str(keyy)+",score="+str(value))
        next_quiz+=str(keyy)+" "

    for i in range(len(gt)):
        no10,value=gt[i]
        print("No3,keyword="+str(no10)+",score="+str(value))
        next_quiz+=str(no10)+" "
    
    i1,i2,i3=first_floor(quiz+" "+next_quiz,0,"",True)#i2がans
    
    if mode == "3":
        truth=""
        with open('./testcase/ans'+str(loop+1)+'.txt') as f:
            for line in f:
                truth=truth+line
        if str(truth).lower()==str(i2).lower():
            ok+=1
            print("State:AC")
        else:
            WA.append(loop+1)
            ng+=1
            print("State:WA")
            print("Truth:"+str(truth))
    
    return

if mode=="2":
    o=True
    add=""
    q=""
    while True:
        a,b=quiz_solve(0,o,add,q)
        if a==0:
            break
        else:
            print("ReThinking...")
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
            a,b=quiz_solve(l,o,add,q)
            if a==0:
                break
            else:
                print("ReThinking...")
                o=False
                add+=" "+str(b)
        qs3(l,add,"")#(loop,add)
elif mode=="1":
    o=True
    add=""
    qz=""
    ttt=""
    while True:
        q=""
        if o==True:
            print("------------------------------------------------------------------")
            print("Input_Quiz:")
            while True:
                q2 = input()
                if "@@@" in q2:
                    q2 = q2.replace("@@@", "")
                    q+=q2+" "
                    break
                q+=q2+" "        
        q = q.replace("@@@", "")
        if o == False:
            q=qz
        qz=q    
        a,b=quiz_solve(0,o,add,q)
        if a!=0:
            print("ReThinking...")
            o=False
            add+=" "+str(b)
        else:
            qs3(-1,add,q)
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
        a,b=quiz_solve(0,o,add,q)
        if a!=0:
            print("ReThinking...")
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
