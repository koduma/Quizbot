import math
import os
import matplotlib.pyplot as plt
import psutil
import Levenshtein
#from googletrans import Translator
import re
import sys
import random
import warnings
import sympy
from sympy import sympify, Eq, solve
#import googlesearch
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import string
import difflib
from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset

# 警告を無視
warnings.filterwarnings("ignore")

counter=1
now=[]
train = dict()
train_num = dict()
datakun = dict()
NoAns = dict()
looked = dict()

meta=""

with open("./metadata2.txt") as f:
    for line in f:
        meta=meta+line

meta=meta.split()

wrii = ""

for l in range(len(meta)):
    tile = meta[l]
    wrii += str(tile) + " "

for i in range(len(meta)):
    looked[str(meta[i]).lower()] = 1


def get_key(s):
    pos=0
    for i in range(len(s)):
        if s[i]=="@":
            pos=i
    key=""
    for i in range(pos):
        key+=str(s[i])
    return key

def get_val(s):
    pos=0
    for i in range(len(s)):
        if s[i]=="@":
            pos=i
    val=""
    for i in range(pos+1,len(s)):
        val+=str(s[i])
    return val

def clean_filename(title):
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c for c in title if c in valid_chars).strip()

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
    elif s=="—":
        sp=1
    return sp


with open('./datakun2.txt') as f:
    for line in f:
        line=line.replace('\n',"")
        key=get_key(str(line))
        val=get_val(str(line))
        datakun[str(key)]=int(val)
with open('./train2.txt') as f:
    for line in f:
        line=line.replace('\n',"")
        key=get_key(str(line))
        val=get_val(str(line))
        train[str(key)]=int(val)
                
with open('./train_num2.txt') as f:
    for line in f:
        line=line.replace('\n',"")
        key=get_key(str(line))
        val=get_val(str(line))
        if str(key)!="54233@":
            train_num[int(key)]=str(val)
with open('./NoAns2.txt') as f:
    for line in f:
        line=line.replace('\n',"")
        key=get_key(str(line))
        val=get_val(str(line))
        NoAns[str(key)]=int(val)
                
with open('./counter2.txt') as f:
    for line in f:
        line=line.replace('\n',"")
        counter=int(line)
        train_num[54233]="@"

now.append(counter)


def print_simple_wikipedia_variables():
    print("データをダウンロード・読み込み中...（数分かかります）")
    global wrii, counter
    
    # ---------------------------------------------------------
    # 【ここが修正ポイント】
    # 旧: "wikipedia" -> 新: "wikimedia/wikipedia"
    # trust_remote_code=True を追加
    # ---------------------------------------------------------
    dataset = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.simple", 
        split="train", 
        trust_remote_code=True
    )
    
    print(f"読み込み完了！ 総記事数: {len(dataset)}件\n")
    print("="*50)

    # --- 最初の5件を表示 ---
    for i, data in enumerate(dataset):
        if i >= 10000: 
            break
        
        valid_title = data['title']
        strr = data['text'][:1000].replace('\n', ' ')
        title = clean_filename(valid_title)
        title = title.replace(" ", "")
        title = title.replace(".", "")

        if str(title).lower() not in looked:
            looked[str(title).lower()] = 1
        else:
            continue

        wrii += str(title) + " "

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
        
        new_key=False
        nkey=[]
        for x in talk:
            ex=x in train.keys()
            if ex==False:
                train[x]=counter
                train_num[counter]=x
                nkey.append(counter)
                NoAns[x]=1
                counter=counter+1
                n=counter
                new_key=True
            else:
                NoAns[x]+=1
        now.append(n)
        
        if str(title).upper() == "1982IRANIANASSEMBLYOFEXPERTSELECTIONINTEHRANPROVINCE":
            continue
        if str(title).upper() == "1979IRANIANCONSTITUTIONALASSEMBLYELECTIONINGILANPROVINCE":
            continue
        if str(title).upper() == "1979IRANIANCONSTITUTIONALASSEMBLYELECTIONINSISTANANDBALUCHESTANPROVINCE":
            continue
        if str(title).upper() == "1979IRANIANCONSTITUTIONALASSEMBLYELECTIONINMARKAZIPROVINCE":
            continue
        if str(title).upper() == "ME":
            continue

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
        for c in talk:
            tmp5 = str(title)+","+str(c).lower()
            tmp6 = str(c).lower()+","+str(title)
            if str(c).lower()!=str(c):
                if tmp5 in datakun:
                    datakun[tmp5]+=4#2
                else:
                    datakun[tmp5]=4#2
                if tmp6 in datakun:
                    datakun[tmp6]+=4#2
                else:
                    datakun[tmp6]=4#2


        if (i+1)%100 == 0:
            b = sys.getsizeof(datakun)
            b += sum(map(sys.getsizeof, datakun.values())) + sum(map(sys.getsizeof, datakun.keys()))
            kb = b / 1024
            mb = kb / 1024
            gb = mb / 1024
            gb2 = format(gb, '.2f')
            print("params="+str(len(datakun))+",mem="+str(gb2)+"GB"+",train="+str(i+1)+"/"+str(len(dataset)))

print_simple_wikipedia_variables()

with open("metadata2.txt", "w", encoding="utf-8") as f:
    f.write(wrii)    
with open("./getdata2/metadata2.txt", "w", encoding="utf-8") as f:
    f.write(wrii)
with open("./getdata/metadata2.txt", "w", encoding="utf-8") as f:
    f.write(wrii)


datakun_txt = open("datakun2.txt", "wt")

for key, value in datakun.items():
    val=str(value)
    ky=str(key)
    datakun_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del datakun[str(ky)]

train_txt = open("train2.txt", "wt")

for key, value in train.items():
    val=str(value)
    ky=str(key)
    train_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del train[str(ky)]

train_num_txt = open("train_num2.txt", "wt")

for key, value in train_num.items():
    val=str(value)
    ky=str(key)
    train_num_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del train_num[str(ky)]

NoAns_txt = open("NoAns2.txt", "wt")

for key, value in NoAns.items():
    val=str(value)
    ky=str(key)
    NoAns_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del NoAns[str(ky)]

counter_txt = open("counter2.txt", "wt")

counter_txt.write(str(counter))