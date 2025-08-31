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
import googlesearch
import requests
from bs4 import BeautifulSoup

strr=""
meta=""

train = dict()
train_num = dict()
datakun = dict()
NoAns = dict()


with open("./metadata2.txt") as f:
    for line in f:
        meta=meta+line

meta=meta.split()

counter=1
now=[1]
WA=[]

PROBLEM = 116
TABOO = 7000
RARE = 400
docs = 0

translator = Translator()

warnings.simplefilter('ignore')

def get_ansi(tk):

    ret=""
    if len(tk)>0:
        for i in range(len(tk)):
            if tk[len(tk)-i-1]=="/":
                break
            ret+=tk[len(tk)-i-1]
    fret=""
    if len(ret)>0:
        for i in range(len(ret)):
            fret+=ret[len(ret)-i-1]
        
    return fret

def get_wikipedia_intro(url: str) -> str:
    title = url.rsplit('/', 1)[-1]

    endpoint = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "redirects": 1
    }

    response = requests.get(endpoint, params=params)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if "extract" in page:
            return page["extract"].strip()
    return None

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
            datakun[tmp5]+=4#2
        else:
            datakun[tmp5]=4#2

        if tmp6 in datakun:
            datakun[tmp6]+=4#2
        else:
            datakun[tmp6]=4#2

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
                datakun[tmp]+=2#1
            else:
                datakun[tmp]=2#1
            tmp3 = str(train_num[c2+1])+","+str(train_num[c+1])
            if tmp3 in datakun:
                datakun[tmp3]+=2#1
            else:
                datakun[tmp3]=2#1
                

    if (l+1)%100 == 0 or (l+1)==len(meta):
        #print("params="+str(len(datakun))+",train="+str(l+1)+"/"+str(len(meta)))
        b = sys.getsizeof(datakun)
        b += sum(map(sys.getsizeof, datakun.values())) + sum(map(sys.getsizeof, datakun.keys()))
        kb = b / 1024
        mb = kb / 1024
        gb = mb / 1024
        gb2 = format(gb, '.2f')
        print("params="+str(len(datakun))+",mem="+str(gb2)+"GB"+",train="+str(l+1)+"/"+str(len(meta)))
        #if (l+1)%100==0:
            #cot = 0
            #for key, value in datakun.items():
                #if float(value) >=0.9 and float(value) <= 1.1:
                    #cot += 1
            #print("count="+str(cot))


datakun_txt = open("datakun.txt", "wt")

for key, value in datakun.items():
    val=str(value)
    ky=str(key)
    datakun_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del datakun[str(ky)]

train_txt = open("train.txt", "wt")

for key, value in train.items():
    val=str(value)
    ky=str(key)
    train_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del train[str(ky)]

train_num_txt = open("train_num.txt", "wt")

for key, value in train_num.items():
    val=str(value)
    ky=str(key)
    train_num_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del train_num[str(ky)]

NoAns_txt = open("NoAns.txt", "wt")

for key, value in NoAns.items():
    val=str(value)
    ky=str(key)
    NoAns_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del NoAns[str(ky)]

counter_txt = open("counter.txt", "wt")

counter_txt.write(str(counter))
