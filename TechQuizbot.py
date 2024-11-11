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
NoAns=dict()
nextword=dict()

prob=10000#2000000
counter=0
tolerance=10

a = pd.read_csv("Answers.csv",encoding="ISO-8859-1")

q = pd.read_csv("Questions.csv",encoding="ISO-8859-1")

for k in range(prob):
    s=str(q["Body"][k])
    z=s.encode('utf-8')
    s=z.decode('utf-8')
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
    z=s.encode('utf-8')
    s=z.decode('utf-8')
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
        if str(answer[c]) not in NoAns:
            word[counter]=answer[c]
            NoAns[word[counter]]=1
            counter+=1
        else:
            NoAns[str(answer[c])]+=1


    for c in range(len(answer)):
        if c+1 < len(answer):
            tmp2=str(answer[c])+","+str(answer[c+1])
            if tmp2 not in nextword:
                nextword[tmp2]=1
            else:
                nextword[tmp2]+=1    
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

def generate(s,quiz3,o):
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
                if sum>maxscore and hit>=len(quiz3)/tolerance and NoAns[str(word[d])]<4000:
                    maxscore=sum
                    ret=str(word[d])
                    if NoAns[str(word[d])]<100:
                        tp=str(o)+","+str(word[d])
                        if tp in nextword:
                            if nextword[tp]>0:
                                maxscore+=1000000.0/float(nextword[tp])
                    if ret in NoAns:
                        print(ret+","+str(NoAns[ret])+","+str(maxscore))
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
o=""
for i in range(100):
    print(str(i+1)+"/100")
    c,s=generate(i,quiz3,o)
    if len(c)==0:
        continue
    o=c    
    pr=pr+c+" "
    score+=s        
print("\nAnswer:\n"+pr)
print("\nScore:\n"+str(score))
