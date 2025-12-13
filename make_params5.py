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

def calc_vocab(s):
    items = s.split(',')
    if len(items) >= 3 or len(items) < 2:
        return "", ""
    return str(items[0]), str(items[1])

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
        
print("Processing datakun2.txt directly to list...")
datakun3 = []
cck = 0

with open('./datakun2.txt') as f:
    for line in f:
        line=line.replace('\n',"")
        key=get_key(str(line))
        valu=get_val(str(line))
            
        key_str = str(key)
        val_str = str(valu)
        
        vocab = key_str.split(',')
        if len(vocab) != 2:
            continue
            
        a = vocab[0]
        b = vocab[1]
        
        if a in train and b in train:
            a1 = train[a]
            b1 = train[b]
            val = int(val_str)
            
            datakun3.append((int(a1), int(b1), int(val)))
            
        cck += 1
        if cck % 10000 == 0:
            print(f"Reading: {cck}/42000000")

print("Sorting data...")
datakun3.sort()

print("Writing to datakun3.txt...")
cckz = 0
with open("datakun3.txt", "wt") as datakun_txt:
    for item in datakun3:
        datakun_txt.write(f"{item[0]},{item[1]},{item[2]}\n")
        
        cckz += 1
        if cckz % 20000 == 0:
            print(f"Writing: {cckz}/42000000")

print("Complete.")