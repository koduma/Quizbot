import pandas as pd
import re

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
    return sp

datakun=dict()
word=dict()

prob=10000#2000000
counter=0

a = pd.read_csv("Answers.csv",encoding="ISO-8859-1")

q = pd.read_csv("Questions.csv",encoding="ISO-8859-1")

for k in range(prob):
    s=str(q["Body"][k])
    s=s.split()
    question=""
    for i in range(len(s)):
        t=str(s[i])
        to_clean = re.compile("<.*?>")
        c = re.sub(to_clean, "", t)
        question+=c+" "
    if k ==0:
        print(question)
    question=question.split()
        
    s=str(a["Body"][k])
    s=s.split()
    answer=""
    for i in range(len(s)):
        t=str(s[i])
        to_clean = re.compile("<.*?>")
        c = re.sub(to_clean, "", t)
        answer+=c+" "
    if k ==0:
        print("\n"+answer)
    answer=answer.split()

    for c in range(len(answer)):
        if str(answer[c]) not in word:
            word[counter]=answer[c]
            counter+=1        

    for c in range(len(answer)):
        for d in range(len(question)):
            tmp = str(answer[c])+","+str(c)+","+str(question[d])+","+str(d)
            if len(answer[c])==0:
                continue
            if tmp in datakun:
                datakun[tmp]+=2
            else:
                datakun[tmp]=2

    if (k+1)%100 == 0 or (k+1)==prob:
        print("params="+str(len(datakun))+",train="+str(k+1)+"/"+str(prob))

def generate(s,quiz3):
    maxscore=0
    ret=""
    ret2=""
    for d in range(counter):
        a=0
        sum=0
        hit=0
        if len(word[d])>0:
            for c in range(len(quiz3)):
                tmp=str(word[d])+","+str(s)+","+str(quiz3[c])+","+str(c)
                #print(str(tmp))
                if tmp in datakun:
                    a+=datakun[tmp]
                    sum=a
                    hit+=1
                if sum>maxscore and hit>=len(quiz3)/2:
                    maxscore=sum
                    ret=str(word[d])
    return ret,maxscore


quiz=""
with open("./quiz.txt",encoding='UTF-8') as f: 
    for line in f:
        quiz=quiz+line        
        
quiz3=""

for k in range(len(quiz)):
    quiz3+=quiz[k]
        

print("\nQuiz:\n"+str(quiz3))

quiz3=quiz3.split()

pr=""
score=0
for i in range(100):
    print(str(i+1)+"/100")
    c,s=generate(i,quiz3)
    if len(c)==0:
        continue
    pr=pr+c+" "
    score+=s        
print("\nAnswer:\n"+pr)
print("\nScore:\n"+str(score))