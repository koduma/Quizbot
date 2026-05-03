#map[過去単語,今単語]学習+ビームサーチ+強化学習
#source .venv/bin/activate
#venv上でpip install -r requirements.txt
import json
from array import array
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
# ==== 変更点: CrossEncoderをインポート ====
from sentence_transformers import CrossEncoder
import torch
import gc
from collections import defaultdict
from bisect import bisect_left
from functools import lru_cache
import nltk
from nltk.corpus import wordnet as wn
import wps
import time
from nltk.tokenize import sent_tokenize
from collections import Counter

strr=""
meta=""

train = dict()
train_num = dict()
datakun = dict()
NoAns = dict()
delta_weights = dict()

x_all_list=[]
ans_type={}

# ==== 変更点: 高速・高精度なCross-encoderモデルをロード ====
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

with open("./metadata2.txt") as f:
    for line in f:
        meta=meta+line

meta=meta.split()

print(len(meta))

counter=1
now=[1]
WA=[]
AC_ex=[]
WA_ex=[]

LIMIT_P = 1000000000
PROBLEM = 214
TABOO = 15000
RARE = 1600
docs = 0
pick = 15

offsets = array('I')
indices = array('I')
w_data = array('H')

current_index = 0 
last_p_id = -1

#translator = Translator()

warnings.simplefilter('ignore')

def load_diff_weights():
    global delta_weights
    if not os.path.exists('datakun3_diff.txt'):
        return
        
    try:
        le=0
        with open('datakun3_diff.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) == 3:
                    p_id = int(parts[0])
                    c_id = int(parts[1])
                    weight_val = int(parts[2].replace('+', ''))
                    delta_weights[(p_id, c_id)] = delta_weights.get((p_id, c_id), 0) + weight_val
                    le+=1
                    if le % 100000 == 0:
                        print("idx="+str(le+len(w_data))+"/"+str(LIMIT_P))
                    
        print(f"datakun3_diff.txt loaded: {len(delta_weights)} diff weights restored.")
    except Exception as e:
        print("datakun3_diff.txt load error:", e)

def wilson_lower(k: float, n: float, z: float = 1.0) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1.0 + (z*z)/n
    center = p + (z*z)/(2.0*n)
    margin = z * math.sqrt((p*(1.0-p))/n + (z*z)/(4.0*n*n))
    return (center - margin) / denom

def is_same_word(s1, s2):
    s1, s2 = s1.lower(), s2.lower()
    
    if s1 == s2:
        return True
        
    len_s1, len_s2 = len(s1), len(s2)
    max_len = max(len_s1, len_s2)
    
    if max_len <= 3:
        return False
        
    if abs(len_s1 - len_s2) > max_len * 0.3:
        return False
        
    ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
    
    if max_len <= 5:
        return ratio >= 0.85
    else:
        return ratio >= 0.80

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
@lru_cache(maxsize=10000)
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
                    # テキストの最初の300文字をチェック（冒頭文の解析）
                    check_text = text[:300].lower()
                    
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
                return len(s), f"{ca:g} m^2"
            elif centimeter==True:
                return len(s), f"{ca*10000.0:g} cm^2"
            else:
                return len(s),f"{ca:g}"
        if xx == 2:
            ca=2.0*(number1+number2)
            if meter==True:
                return len(s), f"{ca:g} m"
            elif centimeter==True:
                return len(s), f"{ca*100.0:g} cm"
            else:
                return len(s),f"{ca:g}"
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
            return len(s), f"{ca:g}π m^2"
        elif centimeter==True:
            return len(s), f"{ca*10000.0:g}π cm^2"
        else:
            return len(s),f"{ca:g}π"

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
            return len(s), f"{ca:g}π m"
        elif centimeter==True:
            return len(s), f"{ca*100.0:g}π cm"
        else:
            return len(s),f"{ca:g}π"
    
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

    if isinstance(ans, float):
        ans = f"{ans:g}"
    else:
        ans = str(ans)
        
    if len(unit)>0:
        ans+=str(unit)
    return ev, ans


def is_ja(s):
    return True if re.search(r'[ぁ-んァ-ン]', s) else False

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
    elif s=="`":
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

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    meaningful_words = []
    for w in words:
        if str(w).lower() == "water" or str(w).lower()=="1":    
            meaningful_words.append(w)
            continue
        if str(w)=="Oconahua":
            continue
        if NoAns.get(w, 0) > TABOO:
            continue
        meaningful_words.append(w)
        
    return meaningful_words

def calculate_jaccard(s, t):
    words_s = set(preprocess_text(s))
    words_t = set(preprocess_text(t))
    
    if len(words_s) == 0 or len(words_t) == 0:
        return 0.0
        
    intersection = words_s & words_t
    union = words_s | words_t
    score = len(intersection) / len(union)
    
    return score

def calculate_order_score(s, t):#s=クイズ文文字列,t=enwiki説明文文字列
    text1 = preprocess_text(s)#リスト化とストップワードカット
    text2 = preprocess_text(t)
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()

def apply_rrf(rankings_lists, weights=None, k=60):
    """
    rankings_lists: [BoWリスト, Jaccardリスト, Orderリスト, BM25リスト, CrossEncoderリスト]
    weights: [BoW重み, Jaccard重み, Order重み, BM25重み,CrossEncoder重み]
    """
    rrf_scores = {}
    
    if weights is None:
        weights = [1.0] * len(rankings_lists)
    
    for rank_list, weight in zip(rankings_lists, weights):
        for i, (word, score) in enumerate(rank_list):
            rank = i + 1
            if word not in rrf_scores:
                rrf_scores[word] = 0.0
            
            # 重みをランクスコアに掛ける
            # weightが高い指標（BoWなど）の上位にある単語ほどスコアが跳ね上がる
            rrf_scores[word] += weight * (1.0 / (k + rank))
            
    final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return final_ranking

for l in range(len(meta)):
    reading=True
    try:
        with open("datakun3.txt", "r") as f:
            for line in f:
                p_id, c_id, weight = map(int, line.strip().split(','))
                if p_id != last_p_id:
                    while len(offsets) <= p_id:
                        offsets.append(current_index)
                    last_p_id = p_id
                indices.append(c_id)
                w_data.append(weight)
                current_index += 1
                if current_index >= LIMIT_P:
                    break
                if current_index % 100000==0:
                    print("idx="+str(current_index)+"/"+str(LIMIT_P))
        offsets.append(current_index)
    except Exception as e:
        print("datakun3 load error:", e)
        reading=False
        # 重大エラーなのでループ抜ける（再読み込みを防ぐ）
        break

    try:
        with open('./train2.txt') as f:
            for line in f:
                line=line.replace('\n',"")
                key=sys.intern(get_key(str(line)))
                val=get_val(str(line))
                train[str(key)]=int(val)
    except Exception as e:
        print("train2 load error:", e)
        reading=False
        break

    try:
        with open('./train_num2.txt') as f:
            for line in f:
                line=line.replace('\n',"")
                key=get_key(str(line))
                val=sys.intern(get_val(line))
                if str(key)!="54233@":
                    train_num[int(key)]=str(val)
    except Exception as e:
        print("train_num2 load error:", e)
        reading=False
        break

    try:
        with open('./NoAns2.txt') as f:
            for line in f:
                line=line.replace('\n',"")
                key=get_key(str(line))
                val_str=get_val(str(line))
                try:
                    val_int = int(val_str)
                except ValueError:
                    # skip malformed line
                    continue
                if val_int <= TABOO or key.lower() in ("water","1"):
                    NoAns[sys.intern(key)] = val_int
    except Exception as e:
        print("NoAns2 load error:", e)
        reading=False
        break

    try:
        with open('./counter2.txt') as f:
            for line in f:
                line=line.replace('\n',"")
                counter=int(line)
    except Exception as e:
        print("counter2 load error:", e)
        reading=False
        break

    train_num[54233]="@"#@が連続しているとget_keyやget_valは間違える
    
    if reading == True:
        load_diff_weights()

    if reading==True:
        print("reading_ok")
        break

@lru_cache(maxsize=100000)
def get_weight_fast(parent_word, child_word, raw=False):
    global train, offsets, indices, w_data, NoAns, delta_weights
    
    p_id = train.get(parent_word)
    c_id = train.get(child_word)
    
    if p_id is None or c_id is None:
        return 0
        
    raw_weight = 0
    # 既存の静的データからの重み取得
    if p_id + 1 < len(offsets):
        start_index = offsets[p_id]
        end_index = offsets[p_id + 1]
        idx = bisect_left(indices, c_id, lo=start_index, hi=end_index)
        
        if idx < end_index and indices[idx] == c_id:
            raw_weight = w_data[idx]
            
    # ==== 追加: 動的に学習した重み（+4など）を加算 ====
    raw_weight += delta_weights.get((p_id, c_id), 0)
    
    if raw_weight == 0:
        return 0
        
    # is_include 等から呼ばれた場合は、生の重みを返す
    if raw:
        return raw_weight
        
    # --- 対数(log)を使った安全なIDF計算 ---
    freq = max(1.0, float(NoAns.get(str(child_word), 1)))
    need = math.log10(float(TABOO) / freq) + 1.0
    need = max(1.0, need)
    
    return raw_weight * need

def is_include(s, t):
    global train, offsets, indices, w_data, NoAns, TABOO

    if s not in NoAns or t not in NoAns:
        return False
    if (NoAns[s] > TABOO and s.lower() not in ("water", "1")) or (NoAns[t] > TABOO and t.lower() not in ("water", "1")):
        return False
        
    w_st = get_weight_fast(s, t,raw=True)
    w_ts = get_weight_fast(t, s,raw=True)

    if w_st == 0 or w_ts == 0:
        return False
        
    if w_st > 5 or w_ts > 5:
        if len(s) >= len(t):
            return (t.upper() in s.upper())
        else:
            return (s.upper() in t.upper())

    return False

def calc_vocab(s):
    items = s.split(',')
    if len(items) >= 3 or len(items) < 2:
        return "", ""
    return str(items[0]), str(items[1])

def collect_children_from_sentence(s):
    global train, train_num, offsets, indices
    
    result_list = []
    words = s.split()
    
    for word in words:
        if word not in train:
            continue
        p_id = train[str(word)]
        
        if p_id + 1 >= len(offsets):
            continue
            
        start = offsets[p_id]
        end = offsets[p_id + 1]
        
        for i in range(start, end):
            c_id = indices[i]
            
            if c_id in train_num:
                child_word = train_num[c_id]
                result_list.append(child_word)
                
    return result_list

def check_exists(x):
    global train, offsets, indices
    
    try:
        parent_str, child_str = x.split(',', 1)
    except ValueError:
        return False
    if parent_str not in train or child_str not in train:
        return False
        
    p_id = train[parent_str]
    c_id = train[child_str]
    
    if p_id + 1 >= len(offsets):
        return False
    start = offsets[p_id]
    end = offsets[p_id + 1]
    
    for i in range(start, end):
        if indices[i] == c_id:
            return True
    return False

def get_children_of_word(s: str) -> list:
    global train, train_num, offsets, indices
    
    result_list = []
    
    if s not in train:
        return result_list
        
    p_id = train[s]
    
    if p_id + 1 >= len(offsets):
        return result_list
        
    start = offsets[p_id]
    end = offsets[p_id + 1]
    
    for i in range(start, end):
        c_id = indices[i]
        if c_id in train_num:
            result_list.append(train_num[c_id])

    sorted_list = sorted(result_list, key=str.lower)
            
    return sorted_list


def wordev(a, b):
    cn_a = get_children_of_word(str(a))
    cn_b = get_children_of_word(str(b))
    
    if not cn_a or not cn_b:
        return 0.0

    score = 0.0
    valid_comparisons = 0

    for child in cn_a:
        if child not in NoAns or NoAns[str(child)] > TABOO:
            continue
            
        weight = get_weight_fast(str(b), str(child))
        
        if weight != 0:
            score += weight
            valid_comparisons += 1
    for child in cn_b:
        if child not in NoAns or NoAns[str(child)] > TABOO:
            continue
            
        weight = get_weight_fast(str(a), str(child))
        
        if weight != 0:
            score += weight
            valid_comparisons += 1

    if valid_comparisons > 0:
        return score / valid_comparisons
    else:
        return 0.0

#print("Apple,Banana:"+str(wordev("Apple","Banana")))
#print("Apple,fruit:"+str(wordev("Apple","fruit")))
#print("Banana,fruit:"+str(wordev("Banana","fruit")))
#print("Apple,car:"+str(wordev("Apple","car")))
#print("Banana,car:"+str(wordev("Banana","car")))
#print(get_children_of_word("Apollo 11"))
#print(get_children_of_word("Apollo11"))
#print(get_children_of_word("Apollo 13"))
#print(get_children_of_word("Apollo13"))

ok=0
ng=0
mode=""

print("mode?(1:keyboard,2:txt,3:testcase,4:generator)=",end="")

mode="3"

if mode=="3" or mode=="4":
    PROBLEM=214
else:
    PROBLEM=1

def remove_duplicates_sorted(lst):
    if not lst:
        return []
    relt = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:
            relt.append(lst[i])
    return relt

def get_top_3_synonyms(word):
    synonyms = []
    seen = set()
    clean_word = word.lower()
    
    seen.add(clean_word)

    synsets = wn.synsets(word)
    
    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().lower().replace('_', ' ')
            
            if synonym not in seen:
                synonyms.append(synonym)
                seen.add(synonym)
                
                if len(synonyms) >= 3:
                    return synonyms

    return synonyms

def get_one_synonym(word,quiz_xxx):
    synsets = wn.synsets(word)
    
    clean_word = word.lower()

    for syn in synsets:
        for lemma in syn.lemmas():
            synonym = lemma.name().lower().replace('_', ' ')
            
            if synonym != clean_word:
                if str(synonym) not in quiz_xxx:
                    return synonym
                
    return None

def is_5W1H(t, s):
    # t: クイズ文の単語リスト (quiz2)
    # s: 正解候補単語 (xxx や train_num[xx+1] など)

    # 1. チェックすべき疑問詞のセットを定義
    q_words_target = {"who", "where", "when", "which", "what", "why", "how"}
    
    # 2. クイズ文(t)の中に、どの疑問詞が含まれているかを抽出する
    found_q_words = set()
    for word in t:
        w_lower = str(word).lower()
        if w_lower in q_words_target:
            found_q_words.add(w_lower)
            
    # 3. 5W1Hがクイズ文になければ即return 1（足切りしない）
    if len(found_q_words) == 0:
        return 1
        
    # 4. クイズ文に「含まれていた疑問詞」に対してのみ、候補sとの重みを確認する
    total_weight = 0
    for qw in found_q_words:
        # 大文字始まり（Who）と小文字（who）の両方のスコアを足す
        total_weight += get_weight_fast(str(s), qw)
        total_weight += get_weight_fast(str(s), qw.capitalize())
        
    # 5. クイズで聞かれている疑問詞との関連度が全く無ければ 0 を返す（足切り）
    if total_weight < 1:
        return 0
        
    return 1

def reinforce_learning(quiz_text, truth_word):
    global counter, train, train_num, NoAns, delta_weights
    
    # 1. 記号の除去（大文字を消さないように a-zA-Z としています）
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', quiz_text)
    raw_words = clean_text.split()
    
    words_to_learn = []
    
    for w in raw_words:
        w_lower = w.lower()
        
        # --- preprocess_text のフィルター機能をここで再現 ---
        if w_lower == "oconahua":
            continue
            
        #if w_lower not in ("water", "1"):
            # 頻度(TABOO)チェック。大文字・小文字どちらかで超えていたら弾く
            #if NoAns.get(w, 0) > TABOO or NoAns.get(w_lower, 0) > TABOO:
                #continue
        # ----------------------------------------------------
        
        # フィルターを生き残った単語だけ、オリジナル(大文字維持)と小文字の両方を登録
        words_to_learn.append(w)
        if w != w_lower:
            words_to_learn.append(w_lower)

    words_to_learn.append(truth_word)
    if truth_word != truth_word.lower():
        words_to_learn.append(truth_word.lower())

    unique_words_to_register = list(set(words_to_learn))

    # --- 以下、辞書登録と重み更新の処理は全く同じです ---
    for w in unique_words_to_register:
        if w not in train:
            train[w] = counter
            train_num[counter] = w
            NoAns[w] = 1 
            
            with open('train2.txt', 'a', encoding='utf-8') as f:
                f.write(f"{w}@{counter}\n")
            with open('train_num2.txt', 'a', encoding='utf-8') as f:
                f.write(f"{counter}@{w}\n")
            with open('NoAns2.txt', 'a', encoding='utf-8') as f:
                f.write(f"{w}@1\n")
            
            counter += 1
            with open('counter2.txt', 'w', encoding='utf-8') as f:
                f.write(str(counter) + "\n")
        else:
            if str(w) in NoAns:
                NoAns[str(w)]+=1

    word_counts = Counter(words_to_learn)
    p_id = train[truth_word]
    diff_lines = []
    
    for w, count in word_counts.items():
        c_id = train[w]
        if p_id == c_id:
            continue
            
        weight_to_add = 4 * count
        
        delta_weights[(p_id, c_id)] = delta_weights.get((p_id, c_id), 0) + weight_to_add
        delta_weights[(c_id, p_id)] = delta_weights.get((c_id, p_id), 0) + weight_to_add
        
        diff_lines.append(f"{p_id},{c_id},+{weight_to_add}\n")
        diff_lines.append(f"{c_id},{p_id},+{weight_to_add}\n")
        
    if diff_lines:
        with open('datakun3_diff.txt', 'a', encoding='utf-8') as f:
            f.writelines(diff_lines)
            
    #print(f">> [Reinforcement] Learnt from WA. Updated weights for '{truth_word}'.")

wrii = ""

for l in range(len(meta)):
    wrii += str(meta[l]) + " "


def clean_filename(title):
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c for c in title if c in valid_chars).strip()

def save_article(title, summary, index):
    global wrii
    valid_title = clean_filename(title)
    valid_title = valid_title.replace(" ", "")
    valid_title = valid_title.replace(".", "")
    if not valid_title:
        valid_title = f"article_{index}"

    if str(valid_title).upper() not in looked:
        quiz_text=""
        for k in range(len(summary)):
            if is_sp(summary[k]) > 0:
                quiz_text+=" "+summary[k]+" "
            else:
                quiz_text+=summary[k]
        reinforce_learning(str(quiz_text), str(valid_title))
        looked[str(valid_title).upper()] = 1
        wrii+=str(valid_title)+" "
        return 1
    return 0
    
maxword=10000000

ctz=1

with open('enwiki_namespace_0_26.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        print("Count:" + str(ctz)+"/"+str(maxword))
        #print("Name:"+str(obj.get("name", "")))

        # 冒頭記事（AbstractかIntroductionの最初の段落があればそれを表示）
        intro_text = ""

        # まずAbstract
        for section in obj.get("sections", []):
            if section.get("name") == "Abstract":
                # Abstractは通常has_partsか直でtext
                if "has_parts" in section:
                    for part in section["has_parts"]:
                        if part.get("type") == "paragraph":
                            intro_text = part.get("value", "")
                            break
                else:
                    intro_text = section.get("value", "")
                break

        # AbstractがなければIntroduction
        if not intro_text:
            for section in obj.get("sections", []):
                if section.get("name", "").lower().startswith("introduction"):
                    if "has_parts" in section:
                        for part in section["has_parts"]:
                            if part.get("type") == "paragraph":
                                intro_text = part.get("value", "")
                                break
                    else:
                        intro_text = section.get("value", "")
                    break

        #print("Intro:"+str(intro_text))

        ctz += save_article(str(obj.get("name","")),str(intro_text),ctz)

        if ctz >= maxword:
            break

with open("metadata2.txt", "w", encoding="utf-8") as f:
    f.write(wrii)

with open('NoAns2.txt', 'w', encoding='utf-8') as f:
    for word, count in NoAns.items():
        f.write(f"{word}@{count}\n")