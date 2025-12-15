import sqlite3
import math
import os
import matplotlib.pyplot as plt
import psutil
import Levenshtein
#from googletrans import Translator
from deep_translator import GoogleTranslator
import re
import sys
import random
import warnings
import sympy
from sympy import sympify, Eq, solve
import googlesearch
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import string
import difflib
from sentence_transformers import SentenceTransformer, util
import torch
import gc
from collections import defaultdict
from bisect import bisect_left
from functools import lru_cache
import nltk
from nltk.corpus import wordnet as wn
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

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

PROBLEM = 116
TABOO = 15000
RARE = 400
LIMIT_P = 100000000

#translator = Translator()

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
    # 1. 入力の正規化
    if "wikipedia.org" in url:
        title = url.rsplit('/', 1)[-1]
    else:
        title = url

    endpoint = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "QuizBot/2.1 (your_email@example.com)" 
    }

    # --- 内部関数: 指定タイトルでテキスト取得 ---
    def fetch_text_by_title(target_title):
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": target_title,
            "redirects": 1
        }
        try:
            # endpointとheadersは外側のget_wikipedia_intro関数のスコープからアクセスされる
            response = requests.get(endpoint, params=params, headers=headers, timeout=5) # タイムアウトを3秒から5秒に変更 (安定性のため)
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            # "-1" はページが存在しない場合のID
            if "-1" in pages:
                return None
                
            for page_id, page in pages.items():
                if "extract" in page and page["extract"]:
                    text = page["extract"].strip()

                    # ======================================================
                    # 曖昧さ回避ページの門番ロジック (Disambiguation Check)
                    # ======================================================
                    # テキストの最初の500文字をチェック（冒頭文の解析）
                    check_text = text[:500].lower()
                    
                    if "may refer to:" in check_text or "may also refer to:" in check_text or "refer to several things" in check_text:
                        # 曖昧さ回避ページ（SNS, Auなど）と判断し、Noneを返す
                        return None
                    # ======================================================
                    
                    return text
        
        except Exception:
            # ネットワークエラー、JSON解析エラーなど
            return None
        
        return None # ページIDはあるが extract が空の場合などに備えて、念のためNoneを返す

    # --- Step 1: そのままトライ ---
    text = fetch_text_by_title(title)
    if text: return text

    # --- Step 2: CamelCase分割 & 一般的な連結語の分割 ---
    # "ElonMusk" -> "Elon Musk"
    split_title = re.sub(r'(?<!^)(?=[A-Z])', ' ', title)
    # "Listofvideogames" -> "List of video games" 対策 (簡易版)
    # 一般的な接続詞の周りに強制的にスペースを入れる
    split_title = re.sub(r'(of|the|vs|and|feat)', r' \1 ', split_title, flags=re.IGNORECASE)
    
    if split_title != title:
        text = fetch_text_by_title(split_title.strip())
        if text: return text

    # --- Step 3: Wikipedia Search API ---
    # Wikipedia内の検索機能を使う
    try:
        search_params = {
            "action": "query", "format": "json", "list": "search",
            "srsearch": title, "srlimit": 1
        }
        search_resp = requests.get(endpoint, params=search_params, headers=headers, timeout=3)
        search_results = search_resp.json().get("query", {}).get("search", [])
        if search_results:
            text = fetch_text_by_title(search_results[0]["title"])
            if text: return text
    except Exception:
        pass

    # --- Step 4: Google Search Fallback (最終手段) ---
    # "Listofvideogames..." のような難物はGoogleに聞くのが一番早い
    try:
        # "site:en.wikipedia.org" をつけて検索
        query = f"site:en.wikipedia.org {title}"
        # google検索は外部ライブラリ依存なのでtryで囲む
        results = googlesearch.search(query, num_results=1)
        
        for result_url in results:
            if "wikipedia.org/wiki/" in result_url:
                # URLからタイトルを抽出 (例: .../wiki/Elon_Musk -> Elon_Musk)
                new_title = result_url.split("/wiki/")[-1]
                # URLデコード (例えば %20 をスペースに戻すなど)
                new_title = requests.utils.unquote(new_title)
                
                print(f"Google Fallback: {title} -> {new_title}") # 動作確認用ログ
                text = fetch_text_by_title(new_title)
                if text: return text
    except Exception as e:
        # Google検索のエラー（レートリミットなど）は無視してNoneを返す
        # print(f"Google Search Error: {e}") 
        pass

    return None

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
        print(str(title)+","+str(l))
    else:
        looked[title]=1
        
#sys.exit()

#print("load?(y/n)=",end="")
#ld=input()

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

def should_register(ak, bk):
    if not ak or not bk:
        return False

    ak_l = ak.lower()
    bk_l = bk.lower()

    ak_is_special = (ak_l == "water" or ak_l == "1")
    bk_is_special = (bk_l == "water" or bk_l == "1")

    if ak_is_special and bk_is_special:
        return True

    if NoAns.get(ak, 0) <= TABOO and NoAns.get(bk, 0) <= TABOO:
        return True
        
    if ak_is_special and NoAns.get(bk, 0) <= TABOO:
        return True
    if bk_is_special and NoAns.get(ak, 0) <= TABOO:
        return True

    return False


cck=0

with open('./datakun2.txt') as f:
    for line in f:
        line=line.replace('\n',"")
        key=get_key(str(line))
        val=get_val(str(line))
        ak,bk=calc_vocab(key)
        if ak != "" and bk != "":
            if should_register(ak, bk): 
                datakun[str(key)] = int(val)
                cck+=1
        if cck % 100000 == 0:
            print("idx="+str(cck)+"/"+str(LIMIT_P))

def fetch_wikipedia_leads_memory(titles):
    """
    Wikipediaの冒頭部を取得し、入力タイトルに対応するテキストのリストを返します。
    入力リストの順序を厳密に維持します。
    """
    # 1. API設定
    url = "https://en.wikipedia.org/w/api.php"
    
    session = requests.Session()
    # リトライ設定: 502/503/504エラーなどは自動で3回まで再試行
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    headers = {
        "User-Agent": "MyWikiBot/1.0 (your_email@example.com)" # 連絡先に変更してください
    }

    BATCH_SIZE = 50 
    
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    all_extracts = []

    # tqdmで進捗を表示
    batches = list(chunked(titles, BATCH_SIZE))
    
    for batch_titles in tqdm(batches, desc="Fetching articles"):
        
        # クエリパラメータ
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": "|".join(batch_titles),
            "exintro": True,
            "explaintext": True,
            "redirects": 1
        }

        try:
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            query_data = data.get("query", {})
            
            # --- マッピングデータの作成 ---
            pages = query_data.get("pages", {})
            title_to_text = {}
            for _, page_data in pages.items():
                if "missing" in page_data:
                    continue
                title_to_text[page_data.get("title")] = page_data.get("extract")

            # 正規化マップ (例: "japan" -> "Japan")
            normalized = query_data.get("normalized", [])
            norm_map = {n["from"]: n["to"] for n in normalized}

            # リダイレクトマップ (例: "USA" -> "United States")
            redirects = query_data.get("redirects", [])
            redir_map = {r["from"]: r["to"] for r in redirects}

            # --- 入力順序通りに結果を取り出す ---
            for original_title in batch_titles:
                current_title = original_title
                
                # 1. 正規化の解決
                if current_title in norm_map:
                    current_title = norm_map[current_title]
                
                # 2. リダイレクトの解決
                if current_title in redir_map:
                    current_title = redir_map[current_title]
                
                # 3. テキスト取得 (なければNone)
                text = title_to_text.get(current_title)
                all_extracts.append(text)

        except Exception as e:
            print(f"Error in batch: {e}")
            # エラー時もNoneで埋めて、入力数と出力数のズレを防ぐ
            all_extracts.extend([None] * len(batch_titles))
        
        # サーバー負荷軽減
        # 30万件 / 50件 = 6000リクエスト
        # 1時間(3600秒)以内に終わらせるには、1リクエストあたり0.6秒以内
        # 通信0.4秒 + sleep 0.05秒 = 0.45秒程度なら余裕で間に合います
        time.sleep(0.05)

    return all_extracts

NNN = len(meta)
print(f"Generating list of {NNN} titles...")
target_titles = meta.copy()
print("Start fetching... (Fetching 300,000 items will take time)")
results = fetch_wikipedia_leads_memory(target_titles)
print("\n--- Verification: Searching for 'Japan' ---")
found_flag = False
li=0
for title, text in zip(target_titles, results):

    if li % 1000 == 0:
        print("size="+str(len(datakun)))
        print(str(li)+"/"+str(len(meta)))
        
    if str(title)=="300000kms":
        title="300000km/s"

    strr=text
    
    strr2=""

    if strr==None:
        continue

    if len(strr)==0:
        continue

    for k in range(len(strr)):   
        if is_sp(strr[k]) > 0:
            strr2+=" "+strr[k]+" "
        else:
            strr2+=strr[k]
    
    talk = strr2.split()

    n=counter
    
    for x in talk:
        ex=x in train.keys()
        if ex==False:
            train[x]=counter
            train_num[counter]=x
            NoAns[x]=1
            counter=counter+1
            n=counter
            
    for y in talk:
        x=str(y).lower()
        ex=x in train.keys()
        if ex==False:
            train[x]=counter
            train_num[counter]=x
            NoAns[x]=1
            counter=counter+1
            n=counter


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
        if str(c) in NoAns:
            if NoAns[str(c)] > TABOO:
                continue
        if tmp5 not in datakun:
            datakun[tmp5]=4
        if tmp6 not in datakun:
            datakun[tmp6]=4
    for c in talk:
        tmp5 = str(title)+","+str(c).lower()
        tmp6 = str(c).lower()+","+str(title)
        if str(c).lower() in NoAns:
            if NoAns[str(c).lower()] > TABOO:
                continue
        if str(c).lower()!=str(c):
            if tmp5 not in datakun:
                datakun[tmp5]=4
            if tmp6 not in datakun:
                datakun[tmp6]=4

    
    if title == "JAPAN" or title =="Japan" or title == "japan":
        print(f"✅ Found target title: {title}")
        if text:
            print(f"Result (First 100 chars): {text[:100]}...")
            if "East Asia" in text or "country" in text:
                print("-> Content check passed: This is likely the article for Japan.")
            else:
                print("-> Content check warning: Text might be different from expected.")
        else:
            print("❌ Result is None (Fetch failed or not found).")
        found_flag = True
    li+=1
        

if not found_flag:
    print("❌ 'Japan' was not found in the target_titles list.")
    


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

datakun_txt = open("datakun2.txt", "wt")

for key, value in datakun.items():
    val=str(value)
    ky=str(key)
    datakun_txt.write(str(ky).rstrip('\n')+"@"+str((val.rstrip('\n'))+"\n"))
    #del datakun[str(ky)]