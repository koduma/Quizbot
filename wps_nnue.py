import ast
import csv
import json
import math
import os
import re
import urllib.request
from collections import defaultdict
from itertools import permutations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. ファイルパス・定数設定
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV_URL = (
    "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/"
    "data/mawps-asdiv-a_svamp/train.csv"
)
TEST_JSON_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"

TRAIN_FILE = os.path.join(BASE_DIR, "mawps_asdiv.csv")
TEST_FILE = os.path.join(BASE_DIR, "SVAMP.json")
WEIGHTS_FILE = os.path.join(BASE_DIR, "math_lr_weights3.json")

NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
VAR_RE = re.compile(r"\bvar(\d+)\b")
# アンダースコア区切りのトークン (例: PREV1_var1_had) 内の var も置換するための正規表現
TOKEN_VAR_RE = re.compile(r"(?<![a-zA-Z0-9])var(\d+)(?![a-zA-Z0-9])")
OPS = {"+", "-", "*", "/"}

# 時間帯分割の設定 (① ざっくり時間帯分割: t0=序盤, t1=中盤, t2=終盤 + all=全体)
N_TIME_BUCKETS = 3
TIME_KEYS = ("all", "t0", "t1", "t2")
TIME_FEATURE_SCALE = 0.35  # 時間帯別特徴の正則化スケール（過学習を抑えつつ位置情報を反映）

# NNUEアーキテクチャ設定:
# 入力層 -> 隠れ層1: weight1[id(formula, word)][HIDDEN1_SIZE] (アキュムレータ加算 + Clipped ReLU)
# 隠れ層1 -> 隠れ層2: W2[HIDDEN1_SIZE][HIDDEN2_SIZE] + Clipped ReLU
# 隠れ層2 -> 出力層 : W3[HIDDEN2_SIZE]
HIDDEN1_SIZE = 16
HIDDEN2_SIZE = 16
TOP_K_CANDIDATES = 8   # 各学習サンプルで競わせる数式候補数 (正解1 + Hard Negative 7)
NNUE_EPOCHS = 18       # NNUE層の学習エポック数
NNUE_LR = 0.015        # Adam学習率
NNUE_L2_DECAY = 1e-4   # L2正則化係数

WORD_TO_NUM = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100",
    "twice": "2 times", "double": "2 times",
}


def download_if_needed(url, path, minimum_size):
    if not os.path.exists(path) or os.path.getsize(path) < minimum_size:
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)


def download_files():
    download_if_needed(TRAIN_CSV_URL, TRAIN_FILE, minimum_size=10_000)
    download_if_needed(TEST_JSON_URL, TEST_FILE, minimum_size=50_000)


# ==========================================
# 2. クイズ vs 算数文章題のルーティング判定器（改善版）
# ==========================================
ENCYCLOPEDIA_PATTERNS = re.compile(
    r"\b("
    r"who is|who was|who were|whom|whose|"
    r"what is the name|what was the name|what is the capital|what country|what city|"
    r"which country|which city|which planet|which element|which president|which king|"
    r"where is|where was|in what year|in which year|what year did|"
    r"known as|famous for|written by|directed by|invented by|discovered by|founded by"
    r")\b",
    re.IGNORECASE,
)

MATH_QUESTION_PATTERNS = re.compile(
    r"\b("
    r"how many|how much|how long|how far|how old|how heavy|how fast|how tall|how deep|"
    r"how big|how wide|how high|how often|"
    r"what is the|what was the|what will be the|what would be the|"
    r"what fraction|what percent|what percentage|what rate|"
    r"calculate|find the|compute"
    r")\b",
    re.IGNORECASE,
)

# 暗黙の数値（硬貨・時間・単数冠詞など）もカウント対象に追加
IMPLICIT_NUM_WORDS = set(WORD_TO_NUM.keys()) | {
    "quarter", "quarters", "dime", "dimes", "nickel", "nickels", "penny", "pennies",
    "dozen", "pair", "pairs", "half", "halves", "twice", "double", "triple",
    "week", "weeks", "year", "years", "hour", "hours", "minute", "minutes",
}

MATH_OPERATION_CUES = {
    "each", "every", "per", "apiece", "total", "altogether", "together", "combined",
    "left", "remain", "remaining", "rest", "away", "lose", "lost", "give", "gave",
    "spend", "spent", "buy", "bought", "cost", "costs", "pay", "paid", "price",
    "discount", "change", "sell", "sold", "earn", "earned", "save", "saved",
    "split", "share", "shared", "divide", "divided", "equally", "equal",
    "more", "less", "fewer", "difference", "add", "added", "plus", "times",
    "half", "twice", "double", "dozen", "pair",
    "dollar", "dollars", "cent", "cents", "yen", "pound", "pounds",
    "pack", "packs", "box", "boxes", "bag", "bags",
    # 追加：算数の基本動詞・状態・単位系
    "have", "has", "had", "make", "made", "take", "took", "get", "got",
    "find", "found", "collect", "collected", "plant", "planted", "bake", "baked",
    "eat", "ate", "drink", "drank", "use", "used", "read", "walk", "walked",
    "run", "ran", "drive", "drove", "fly", "flew", "work", "worked",
    "all", "now", "first", "start", "started", "begin", "began", "finish", "finished",
    "need", "needed", "want", "wanted", "already", "still", "another", "other",
    "group", "groups", "row", "rows", "team", "teams", "page", "pages",
    "hour", "hours", "minute", "minutes", "day", "days", "week", "weeks",
    "mile", "miles", "foot", "feet", "inch", "inches", "meter", "meters",
    "liter", "liters", "gram", "grams", "kilogram", "kilograms",
}


def is_math_word_problem(text):
    """
    入力された問題文が「算数文章題(True)」か「一般の知識クイズ(False)」かを判定する。
    """
    if not text or not isinstance(text, str):
        return False

    text_clean = text.strip()
    text_low = text_clean.lower()

    if ENCYCLOPEDIA_PATTERNS.search(text_low):
        return False

    # 疑問詞パターンのチェック (文末に '?' があり "how" を含む場合も許容)
    if not MATH_QUESTION_PATTERNS.search(text_low):
        if not ("?" in text_low and re.search(r"\bhow\b", text_low)):
            return False

    # number0, number1 や var1, var2 のようなマスク済み変数も数値として扱う
    masked_vars = re.findall(r"\b(?:number|var)\d+\b", text_low)

    # クイズの年号 (in 1999, during 2009 など 1000〜2099年の4桁年号のみ) を除外
    # ※ 以前の \d{1,4} だと "in 3 boxes" や "to 5 friends" の数字まで消えてしまうため修正
    scrubbed = re.sub(
        r"\b(in|from|to|since|during|year|years|released in|born in|founded in)\s+(1[0-9]{3}|20[0-9]{2})\b",
        " ",
        text_low,
    )

    nums = NUMBER_RE.findall(scrubbed)
    total_num_count = len(nums) + len(masked_vars)

    if total_num_count < 2:
        word_nums = [w for w in re.findall(r"[a-z]+", scrubbed) if w in IMPLICIT_NUM_WORDS]
        # "a minute", "an hour", "a pie" などの単数単位表現 (1個分) も補助カウント
        per_a = re.findall(r"\b(?:a|an)\s+(?:minute|hour|day|week|month|year|pie|pack|box|bag|group|shirt|car)\b", scrubbed)
        if total_num_count + len(word_nums) + len(per_a) < 2:
            return False

    words_in_text = set(re.findall(r"[a-z]+", text_low))
    if not (words_in_text & MATH_OPERATION_CUES) and "$" not in text_low and "%" not in text_low:
        return False

    return True


# ==========================================
# 3. 数式の変換・正規化
# ==========================================
def number_to_var(text):
    return re.sub(r"\bnumber(\d+)\b", lambda m: f"var{int(m.group(1)) + 1}", text)


def prefix_to_infix(expression):
    tokens = number_to_var(expression).split()
    index = 0

    def parse():
        nonlocal index
        token = tokens[index]
        index += 1
        if token in OPS:
            left = parse()
            right = parse()
            return f"({left} {token} {right})"
        return token

    return parse()


def canonicalize_ast(expression):
    tree = ast.parse(expression, mode="eval")

    def collect_assoc(node, op_type):
        if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):
            return collect_assoc(node.left, op_type) + collect_assoc(node.right, op_type)
        return [node]

    def to_str(node):
        if isinstance(node, ast.Expression):
            return to_str(node.body)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            val = float(node.value)
            return str(int(val)) if val.is_integer() else str(val)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{to_str(node.operand)})"
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                terms = sorted(to_str(n) for n in collect_assoc(node, ast.Add))
                res = terms[0]
                for t in terms[1:]:
                    res = f"({res} + {t})"
                return res
            if isinstance(node.op, ast.Mult):
                terms = sorted(to_str(n) for n in collect_assoc(node, ast.Mult))
                res = terms[0]
                for t in terms[1:]:
                    res = f"({res} * {t})"
                return res
            left_s, right_s = to_str(node.left), to_str(node.right)
            if isinstance(node.op, ast.Sub):
                return f"({left_s} - {right_s})"
            if isinstance(node.op, ast.Div):
                return f"({left_s} / {right_s})"
        raise ValueError("Unsupported AST node")

    return to_str(tree)


def normalize_equation(expression, prefix=False):
    expression = expression.strip()
    expression = prefix_to_infix(expression) if prefix else number_to_var(expression)
    expression = re.sub(r"\s+", " ", expression).replace("( ", "(").replace(" )", ")")
    try:
        return canonicalize_ast(expression)
    except Exception:
        return expression


def max_variable_index(expression):
    return max((int(idx) for idx in VAR_RE.findall(expression)), default=0)


# ==========================================
# 4. 数値マスクと時間帯別・構造・単位特徴トークン抽出
# ==========================================
def preprocess_implicit_numbers(text):
    s = re.sub(r"\b(a|one)\s+dozen\b", "12", text, flags=re.IGNORECASE)
    return re.sub(r"\ba\s+pair\s+of\b", "2", s, flags=re.IGNORECASE)


def mask_numbers_in_question(text):
    text_norm = preprocess_implicit_numbers(text)
    values = []

    def replace(match):
        values.append(float(match.group()))
        return f"var{len(values)}"

    return NUMBER_RE.sub(replace, text_norm), values


def mask_numbers_in_equation(equation, values):
    used_indices = set()

    def replace(match):
        value = float(match.group())
        candidates = [
            i for i, num in enumerate(values)
            if i not in used_indices and math.isclose(num, value, abs_tol=1e-9)
        ]
        if not candidates:
            candidates = [i for i, num in enumerate(values) if math.isclose(num, value, abs_tol=1e-9)]
        if not candidates:
            raise ValueError("Equation number not in question")
        chosen = candidates[0]
        used_indices.add(chosen)
        return f"var{chosen + 1}"

    return NUMBER_RE.sub(replace, equation)


STOP_WORDS = {
    "a", "an", "the", "he", "her", "hers", "him", "his", "it",
    "she", "their", "them", "they", "there", "am", "are", "be",
    "been", "being", "is", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "some", "that", "this",
    "these", "those", "mr", "mrs", "ms", "you", "your", "i", "we",
    "who", "what", "where",
}

IRREGULAR_LEMMAS = {
    "bought": "buy", "buys": "buy", "buying": "buy",
    "gave": "give", "gives": "give", "given": "give",
    "went": "go", "gone": "go",
    "lost": "lose", "loses": "lose", "losing": "lose",
    "spent": "spend", "spends": "spend",
    "took": "take", "takes": "take", "taken": "take",
    "told": "tell", "made": "make", "makes": "make",
    "ate": "eat", "eaten": "eat", "eating": "eat",
    "paid": "pay", "pays": "pay",
    "sold": "sell", "sells": "sell",
    "found": "find", "finds": "find",
    "caught": "catch", "cut": "cut", "put": "put", "left": "left",
    "flew": "fly", "flies": "fly",
    "children": "child", "men": "man", "women": "woman",
    "feet": "foot", "inches": "inch", "boxes": "box",
    "packs": "pack", "bags": "bag", "dollars": "dollar",
    "cents": "cent", "hours": "hour", "minutes": "minute",
    "days": "day", "weeks": "week", "months": "month",
    "years": "year", "times": "time", "halves": "half",
    "candies": "candy", "berries": "berry", "cookies": "cookie",
    "pennies": "penny", "loaves": "loaf", "leaves": "leaf",
}

ES_SUFFIXES = ("ches", "shes", "sses", "xes", "zes")
MONEY_UNITS = {"dollar", "cent", "money", "price", "cost", "penny", "dime", "quarter", "nickel", "buck", "yen"}
TIME_UNITS = {"hour", "minute", "second", "day", "week", "month", "year"}
DIST_UNITS = {"mile", "foot", "inch", "yard", "meter", "kilometer", "centimeter"}
WEIGHT_UNITS = {"pound", "ounce", "gram", "kilogram", "ton"}

NON_UNIT_WORDS = {
    "of", "in", "on", "at", "to", "for", "from", "with", "by", "into", "among", "between",
    "and", "or", "but", "if", "when", "while", "after", "before", "than", "as", "until",
    "each", "every", "per", "all", "both", "more", "less", "fewer", "other", "another",
    "left", "remain", "remaining", "away", "back", "out", "up", "down", "off", "over",
    "now", "then", "still", "already", "only", "just", "also", "too", "together", "altogether",
    "total", "how", "many", "much", "long", "far", "old", "heavy",
}

SUB_CUES = {
    "give", "lose", "spend", "eat", "take", "drop", "break", "broken", "burn", "melt",
    "throw", "delete", "miss", "sell", "use", "used", "drink", "leave", "left", "quit",
    "fly", "away", "off", "out", "back", "discount", "change", "less", "fewer", "than",
    "difference", "remain", "remaining", "rest", "still", "need", "needed", "want", "wanted",
    "already", "only",
}

ADD_CUES = {
    "add", "added", "plus", "more", "total", "altogether", "together", "combined", "sum",
    "another", "extra", "additional", "new", "receive", "received", "get", "got", "find",
    "pick", "picked", "grow", "collect", "collected", "join", "joined", "came", "come",
    "born", "plant", "planted", "bake", "baked", "earn", "earned", "save", "saved", "win",
}

MATH_CORE_WORDS = NON_UNIT_WORDS | SUB_CUES | ADD_CUES | MONEY_UNITS | TIME_UNITS | DIST_UNITS | WEIGHT_UNITS | {
    "buy", "cost", "pay", "price", "split", "share", "shared", "divide", "divided",
    "equally", "equal", "group", "row", "box", "pack", "bag", "team", "page",
    "piece", "pair", "dozen", "half", "twice", "double", "time", "have", "had", "has",
    "make", "put", "hold", "fill", "filled", "fit",
}


def simple_lemmatize(word):
    if word in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(ES_SUFFIXES):
        return word[:-2]
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def get_time_bucket(index, total_length, n_buckets=N_TIME_BUCKETS):
    """単語の出現位置(index)から、時間帯バケット('t0', 't1', 't2')を計算する"""
    if total_length <= 0:
        return "t0"
    b = int((index / total_length) * n_buckets)
    return f"t{min(b, n_buckets - 1)}"


def get_var_spans(words):
    clauses, curr = [], []
    for w in words:
        if w in (".", ",", "?", ";") or w in ("if", "when", "while", "after", "before", "but", "so", "how", "and"):
            if curr:
                clauses.append(curr)
            curr = [w] if w not in (".", ",", "?", ";", "and") else []
        else:
            curr.append(w)
    if curr:
        clauses.append(curr)

    var_spans = {}
    for c in clauses:
        var_indices = [idx for idx, w in enumerate(c) if w.startswith("var")]
        if len(var_indices) == 1:
            var_spans[c[var_indices[0]]] = c
        elif len(var_indices) > 1:
            for k, v_idx in enumerate(var_indices):
                start = 0 if k == 0 else min(v_idx, var_indices[k - 1] + 2)
                end = len(c) if k == len(var_indices) - 1 else min(var_indices[k + 1], v_idx + 3)
                var_spans[c[v_idx]] = c[start:end]
    return clauses, var_spans


def analyze_question_structure(text):
    text_low = re.sub(r"'s\b", "", text.lower())
    raw_tokens = re.findall(r"var\d+|[a-z]+|[.,?;]", text_low)
    words = []
    for w in raw_tokens:
        if w in (".", ",", "?", ";") or w.startswith("var"):
            words.append(w)
        elif w not in STOP_WORDS:
            words.append(simple_lemmatize(w))

    _, var_spans = get_var_spans(words)
    clean_words = [w for w in words if w not in (".", ",", "?", ";")]
    n_words = len(clean_words)

    q_idx = 0
    for idx in range(n_words - 1, -1, -1):
        if clean_words[idx] in ("how", "what"):
            q_idx = idx
            break
    q_words = clean_words[q_idx:]
    q_words_set = set(q_words)

    var_units, var_time_buckets = {}, {}
    for i, w in enumerate(clean_words):
        if w.startswith("var"):
            var_time_buckets[w] = get_time_bucket(i, n_words)
            u_list = []
            for offset in (1, 2, 3):
                if i + offset < n_words:
                    cand = clean_words[i + offset]
                    if cand.startswith("var"):
                        break
                    if cand not in NON_UNIT_WORDS:
                        u_list.append(cand)
            var_units[w] = (u_list[0] if len(u_list) >= 1 else "NONE", u_list[1] if len(u_list) >= 2 else "NONE")

    each_nouns = set()
    for i, w in enumerate(clean_words):
        if w in ("each", "every", "per", "apiece"):
            if i + 1 < n_words and not clean_words[i + 1].startswith("var"):
                each_nouns.add(clean_words[i + 1])
            if i - 1 >= 0 and not clean_words[i - 1].startswith("var"):
                each_nouns.add(clean_words[i - 1])

    asks_money = bool(q_words_set & {"pay", "spend", "cost", "price", "money", "dollar", "cent", "change", "earn", "save"}) or ("how" in q_words and "much" in q_words and not (q_words_set & WEIGHT_UNITS))
    asks_time = ("how" in q_words and "long" in q_words) or bool(q_words_set & TIME_UNITS)
    asks_dist = ("how" in q_words and "far" in q_words) or bool(q_words_set & DIST_UNITS)
    asks_weight = ("how" in q_words and "heavy" in q_words) or "weigh" in q_words_set or bool(q_words_set & WEIGHT_UNITS)

    q_var_units = {q_words[i + 1] for i, w in enumerate(q_words) if w.startswith("var") and i + 1 < len(q_words)}
    q_target_nouns = {
        w for i, w in enumerate(q_words)
        if w not in NON_UNIT_WORDS and not w.startswith("var") and not (w in q_var_units and "many" not in q_words[:i])
    }

    double_match_vars = {
        v for v, (u1, u2) in var_units.items()
        if u1 != "NONE" and u2 != "NONE" and u1 in q_target_nouns and u2 in q_target_nouns
    }

    q_match_vars = set()
    if double_match_vars:
        q_match_vars = double_match_vars
    else:
        for v, (u1, u2) in var_units.items():
            if (
                (u1 != "NONE" and u1 in q_target_nouns)
                or (u2 != "NONE" and u2 in q_target_nouns)
                or (asks_money and (u1 in MONEY_UNITS or u2 in MONEY_UNITS))
                or (asks_time and (u1 in TIME_UNITS or u2 in TIME_UNITS))
                or (asks_dist and (u1 in DIST_UNITS or u2 in DIST_UNITS))
                or (asks_weight and (u1 in WEIGHT_UNITS or u2 in WEIGHT_UNITS))
            ):
                q_match_vars.add(v)

    same_unit_pairs, diff_unit_pairs = set(), set()
    var_keys = sorted(var_units.keys())
    for i in range(len(var_keys)):
        for j in range(i + 1, len(var_keys)):
            v1, v2 = var_keys[i], var_keys[j]
            u1_a, u1_b = var_units[v1]
            u2_a, u2_b = var_units[v2]
            exact_match = (u1_a != "NONE" and u1_a == u2_a)
            head_match = (u1_b != "NONE" and u1_b == u2_b) or (u1_a != "NONE" and u1_a == u2_b) or (u1_b != "NONE" and u1_b == u2_a)
            cat_match = (
                ({u1_a, u1_b} & MONEY_UNITS and {u2_a, u2_b} & MONEY_UNITS)
                or ({u1_a, u1_b} & TIME_UNITS and {u2_a, u2_b} & TIME_UNITS)
            )
            if exact_match or cat_match:
                same_unit_pairs.add((v1, v2))
            elif head_match and u1_a not in q_target_nouns and u2_a not in q_target_nouns:
                same_unit_pairs.add((v1, v2))
            else:
                diff_unit_pairs.add((v1, v2))

    time_tokens = {k: set() for k in TIME_KEYS}

    def add_tok(t_b, tok):
        time_tokens["all"].add(tok)
        if t_b in time_tokens:
            time_tokens[t_b].add(tok)

    for i, w in enumerate(clean_words):
        if w.startswith("var"):
            t_b = get_time_bucket(i, n_words)
            u1, u2 = var_units[w]
            prev1 = clean_words[i - 1] if i > 0 and clean_words[i - 1] in MATH_CORE_WORDS else "OBJ"
            prev2 = clean_words[i - 2] if i > 1 and clean_words[i - 2] in MATH_CORE_WORDS else "OBJ"
            add_tok(t_b, f"PREV1_{w}_{prev1}")
            add_tok(t_b, f"PREV2_{w}_{prev2}_{prev1}")

            span_set = set(var_spans.get(w, []))
            if span_set & {"each", "every", "per", "apiece"}:
                add_tok(t_b, f"EACH_IN_SPAN_{w}")
            for sh in (span_set & SUB_CUES):
                add_tok(t_b, f"SPAN_SUB_{w}_{sh}")
            if span_set & SUB_CUES:
                add_tok(t_b, f"SPAN_ANY_SUB_{w}")
            for ah in (span_set & ADD_CUES):
                add_tok(t_b, f"SPAN_ADD_{w}_{ah}")
            if span_set & ADD_CUES:
                add_tok(t_b, f"SPAN_ANY_ADD_{w}")

            add_tok(t_b, f"Q_UNIT_MATCH_{w}" if w in q_match_vars else f"Q_UNIT_NOMATCH_{w}")
            if (u1 != "NONE" and u1 in each_nouns) or (u2 != "NONE" and u2 in each_nouns):
                add_tok(t_b, f"ROLE_EACH_NOUN_{w}")
            if w in q_words_set:
                add_tok(t_b, f"IN_Q_CLAUSE_{w}")

    for v1, v2 in same_unit_pairs:
        time_tokens["all"].add(f"SAME_UNIT_EXACT_{v1}_{v2}")
    for v1, v2 in diff_unit_pairs:
        time_tokens["all"].add(f"DIFF_UNIT_{v1}_{v2}")

    abstract_items = []
    for i, w in enumerate(clean_words):
        t_b = get_time_bucket(i, n_words)
        if w.startswith("var") or w in MATH_CORE_WORDS:
            abstract_items.append((w, t_b))
        elif not abstract_items or abstract_items[-1][0] != "OBJ":
            abstract_items.append(("OBJ", t_b))

    for w, t_b in abstract_items:
        if w != "OBJ":
            add_tok(t_b, w)
    for i in range(len(abstract_items) - 1):
        (w1, tb1), (w2, tb2) = abstract_items[i], abstract_items[i + 1]
        if w1 != "OBJ" or w2 != "OBJ":
            add_tok(tb2 if w2.startswith("var") else tb1, f"{w1}_{w2}")

    q_abstract = [
        (clean_words[idx], get_time_bucket(idx, n_words))
        for idx in range(q_idx, n_words)
        if clean_words[idx].startswith("var") or clean_words[idx] in MATH_CORE_WORDS
    ]
    for w, t_b in q_abstract:
        add_tok(t_b, f"Q_{w}")
    for i in range(len(q_abstract) - 1):
        (w1, tb1), (w2, _tb2) = q_abstract[i], q_abstract[i + 1]
        add_tok(tb1, f"QBI_{w1}_{w2}")

    return time_tokens, same_unit_pairs, diff_unit_pairs, q_match_vars, var_time_buckets


def tokenize_by_time(text):
    time_tokens, _, _, _, var_time_buckets = analyze_question_structure(text)
    return time_tokens, var_time_buckets


def permute_time_tokens_and_eq(time_tokens, var_time_buckets, eq, mapping):
    """変数の入れ替え(mapping)に合わせて、トークン内の変数名および時間帯バケットを連動して入れ替える。"""
    def repl(m):
        v = f"var{m.group(1)}"
        return mapping.get(v, v)

    def map_tok(t):
        t_mapped = TOKEN_VAR_RE.sub(repl, t)
        m_pair = re.match(r"^(SAME_UNIT_EXACT|DIFF_UNIT)_(var\d+)_(var\d+)$", t_mapped)
        if m_pair:
            prefix, v_a, v_b = m_pair.groups()
            if v_a > v_b:
                v_a, v_b = v_b, v_a
            t_mapped = f"{prefix}_{v_a}_{v_b}"
        return t_mapped

    new_eq = normalize_equation(TOKEN_VAR_RE.sub(repl, eq), prefix=False)
    orig_vars = sorted([k for k in mapping.keys() if k in var_time_buckets], key=lambda v: int(v[3:]))
    orig_buckets = [var_time_buckets[v] for v in orig_vars]
    vars_by_new_order = sorted(orig_vars, key=lambda v: int(mapping[v][3:]))

    var_to_new_bucket, bucket_map_candidates = {}, defaultdict(set)
    for r, orig_v in enumerate(vars_by_new_order):
        old_b, new_b = var_time_buckets[orig_v], orig_buckets[r]
        var_to_new_bucket[orig_v] = new_b
        bucket_map_candidates[old_b].add(new_b)

    bucket_map = {old_b: next(iter(new_bs)) for old_b, new_bs in bucket_map_candidates.items() if len(new_bs) == 1}
    new_time_tokens = {k: set() for k in TIME_KEYS}
    for t_key, tok_set in time_tokens.items():
        if t_key == "all":
            for tok in tok_set:
                new_time_tokens["all"].add(map_tok(tok))
        else:
            for tok in tok_set:
                vars_in_tok = [f"var{m}" for m in TOKEN_VAR_RE.findall(tok)]
                if len(vars_in_tok) == 1 and vars_in_tok[0] in var_to_new_bucket:
                    target_b = var_to_new_bucket[vars_in_tok[0]]
                else:
                    target_b = bucket_map.get(t_key, t_key)
                new_time_tokens[target_b].add(map_tok(tok))

    return new_time_tokens, new_eq


def time_tokens_to_feature_dict(time_tokens):
    """DictVectorizer用に {'時間::単語': スケール値} のフラット辞書へ変換する。"""
    feat = {}
    for t_key, tok_set in time_tokens.items():
        scale = 1.0 if t_key == "all" else TIME_FEATURE_SCALE
        for w in tok_set:
            feat[f"{t_key}::{w}"] = scale
    return feat


# ==========================================
# 5. AST評価 ＆ 次元解析・非負チェック
# ==========================================
def evaluate_with_eval(expression, values):
    variables = {f"var{i + 1}": v for i, v in enumerate(values)}
    return float(eval(expression, {"__builtins__": {}}, variables))


def check_ast_and_dimensional_bonus(expression, values, same_unit_pairs, diff_unit_pairs, q_match_vars):
    try:
        tree = ast.parse(expression, mode="eval")
    except Exception:
        return False, -1e9, None

    variables = {f"var{i + 1}": v for i, v in enumerate(values)}
    bonus = 0.0
    valid = True

    def eval_node(node):
        nonlocal bonus, valid
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                valid = False
                return 0.0, None
            return variables[node.id], node.id
        if isinstance(node, ast.Constant):
            return float(node.value), None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val, u_var = eval_node(node.operand)
            return -val, u_var
        if isinstance(node, ast.BinOp):
            left_val, left_u = eval_node(node.left)
            right_val, right_u = eval_node(node.right)
            if not valid:
                return 0.0, None

            if isinstance(node.op, (ast.Add, ast.Sub)):
                if left_u is not None and right_u is not None and left_u != right_u:
                    pair = tuple(sorted([left_u, right_u]))
                    if pair in same_unit_pairs:
                        bonus += 1.5
                    elif pair in diff_unit_pairs and len(same_unit_pairs) > 0:
                        bonus -= 4.5
                rep_u = left_u or right_u
                if isinstance(node.op, ast.Add):
                    return left_val + right_val, rep_u
                res = left_val - right_val
                if res <= 0:
                    valid = False
                return res, rep_u

            if isinstance(node.op, ast.Mult):
                return left_val * right_val, None
            if isinstance(node.op, ast.Div):
                if abs(right_val) < 1e-9:
                    valid = False
                    return 0.0, None
                res = left_val / right_val
                if abs(res - round(res)) > 1e-6:
                    if abs(res * 4 - round(res * 4)) > 1e-5 and abs(res * 10 - round(res * 10)) > 1e-5:
                        bonus -= 2.5
                    else:
                        bonus -= 0.5
                return res, None

        valid = False
        return 0.0, None

    final_val, _ = eval_node(tree)
    if not valid or final_val <= 0:
        return False, -1e9, None

    used_vars = {f"var{idx}" for idx in VAR_RE.findall(expression)}
    if q_match_vars:
        for v in q_match_vars:
            bonus += 0.6 if v in used_vars else -0.8

    return True, bonus, final_val


# ==========================================
# 6. NNUE評価関数モデル (weight1[id(formula, word)][H] 型)
# ==========================================
class MathNNUEModel:
    """
    将棋AIのHalfKP型NNUEに倣った2駒関係入力ニューラル評価関数:
      - 入力層 -> 隠れ層1: weight1[id(formula, word)][H1] のアキュムレータ加算 + Clipped ReLU
      - 隠れ層1 -> 隠れ層2: 全結合 (W2, b2) + Clipped ReLU
      - 隠れ層2 -> 出力層 : 全結合 (W3) + 2駒線形ベースライン (base_w, base_b)
    """
    def __init__(self, h1=HIDDEN1_SIZE, h2=HIDDEN2_SIZE):
        self.h1 = h1
        self.h2 = h2
        self.equations = []
        self.eq_to_id = {}
        self.words = []
        self.word_to_id = {}
        self.num_formulas = 0
        self.num_words = 0

        # weight1[id(formula, word)][H1] の2次元テーブル (shape: [F * W, H1])
        self.weight1 = None
        self.b1 = None        # shape: [F, H1]
        self.W2 = None        # shape: [H1, H2]
        self.b2 = None        # shape: [H2]
        self.W3 = None        # shape: [H2]

        # 2駒関係のベース評価値 (Skip Connection)
        self.base_w = None    # shape: [F * W]
        self.base_b = None    # shape: [F]
        self.is_loaded = False

    def get_pair_id(self, f_id, w_id):
        """(formula, word) の2駒を1つの数値IDに変換する"""
        return f_id * self.num_words + w_id

    def fit_and_save(self, dataset, weights_path=WEIGHTS_FILE):
        X_dicts, y_labels = [], []

        for question, eq, values, _ans in dataset:
            base_time_tokens, var_tb = tokenize_by_time(question)
            X_dicts.append(time_tokens_to_feature_dict(base_time_tokens))
            y_labels.append(eq)

            if len(values) == 2 and max_variable_index(eq) == 2:
                sw_tokens, sw_eq = permute_time_tokens_and_eq(
                    base_time_tokens, var_tb, eq, {"var1": "var2", "var2": "var1"}
                )
                X_dicts.append(time_tokens_to_feature_dict(sw_tokens))
                y_labels.append(sw_eq)

                for mapping in (
                    {"var1": "var1", "var2": "var3"},
                    {"var1": "var2", "var2": "var3"},
                    {"var1": "var3", "var2": "var1"},
                    {"var1": "var3", "var2": "var2"},
                ):
                    sh_tokens, sh_eq = permute_time_tokens_and_eq(base_time_tokens, var_tb, eq, mapping)
                    X_dicts.append(time_tokens_to_feature_dict(sh_tokens))
                    y_labels.append(sh_eq)

            elif len(values) == 3 and max_variable_index(eq) == 3:
                vars_3 = ("var1", "var2", "var3")
                for p in permutations(vars_3):
                    if p == vars_3:
                        continue
                    perm_tokens, perm_eq = permute_time_tokens_and_eq(
                        base_time_tokens, var_tb, eq, dict(zip(vars_3, p))
                    )
                    X_dicts.append(time_tokens_to_feature_dict(perm_tokens))
                    y_labels.append(perm_eq)

        vectorizer = DictVectorizer(sparse=True)
        X_csr = vectorizer.fit_transform(X_dicts).astype(np.float32)

        # 1. ベースラインの2駒関係重みと上位候補数式を抽出
        clf = LogisticRegression(C=0.5, max_iter=600)
        clf.fit(X_csr, y_labels)

        self.words = [str(w) for w in vectorizer.get_feature_names_out()]
        self.word_to_id = {w: i for i, w in enumerate(self.words)}
        self.num_words = len(self.words)

        self.equations = [str(eq) for eq in clf.classes_]
        self.eq_to_id = {eq: i for i, eq in enumerate(self.equations)}
        self.num_formulas = len(self.equations)

        F, W, H1, H2 = self.num_formulas, self.num_words, self.h1, self.h2
        N = X_csr.shape[0]

        if F == 2 and clf.coef_.shape[0] == 1:
            coef_mat = np.vstack([-clf.coef_[0], clf.coef_[0]]).astype(np.float32)
            intercept_vec = np.array([-clf.intercept_[0], clf.intercept_[0]], dtype=np.float32)
        else:
            coef_mat = clf.coef_.astype(np.float32)
            intercept_vec = clf.intercept_.astype(np.float32)

        # 単語のスケール(TIME_FEATURE_SCALE)を重み側に焼き込んで推論時は単純加算にする
        word_scales = np.array(
            [1.0 if w.startswith("all::") else TIME_FEATURE_SCALE for w in self.words],
            dtype=np.float32,
        )
        scaled_coef = coef_mat * word_scales[None, :]  # shape: (F, W)

        self.base_w = scaled_coef.reshape(F * W).copy()
        self.base_b = intercept_vec.copy()

        # 2. 各サンプルの正解数式 + 上位ライバル数式(Hard Negatives) のペアID疎行列を構築
        K = min(TOP_K_CANDIDATES, F)
        y_ids = np.array([self.eq_to_id[y] for y in y_labels], dtype=np.int32)

        base_logits = (X_csr @ coef_mat.T) + intercept_vec[None, :]  # (N, F)
        cands = np.zeros((N, K), dtype=np.int32)
        cands[:, 0] = y_ids
        if K > 1:
            base_logits_copy = base_logits.copy()
            base_logits_copy[np.arange(N), y_ids] = -1e9
            top_neg = np.argpartition(-base_logits_copy, kth=K - 1, axis=1)[:, : K - 1]
            cands[:, 1:] = top_neg

        base_cand_scores = base_logits[np.arange(N)[:, None], cands]  # (N, K)

        X_bin = X_csr.copy()
        X_bin.data = np.ones_like(X_bin.data)
        row_lengths = np.diff(X_bin.indptr)
        cand_row_lengths = np.repeat(row_lengths, K)
        M_indptr = np.zeros(N * K + 1, dtype=np.int32)
        np.cumsum(cand_row_lengths, out=M_indptr[1:])

        total_nnz = int(M_indptr[-1])
        base_cols = X_bin.indices.astype(np.int32)
        rep_word_ids = np.concatenate([
            np.tile(base_cols[X_bin.indptr[i] : X_bin.indptr[i + 1]], K)
            for i in range(N)
        ])
        rep_f_ids = np.repeat(cands.reshape(-1), cand_row_lengths)
        pair_indices = rep_f_ids * W + rep_word_ids

        # 実際に出現した有効な (formula, word) 2駒ペアIDのみを抽出して高速学習
        active_pids, compact_col_indices = np.unique(pair_indices, return_inverse=True)
        A = len(active_pids)
        M_active_csr = sparse.csr_matrix(
            (np.ones(total_nnz, dtype=np.float32), compact_col_indices.astype(np.int32), M_indptr),
            shape=(N * K, A),
            dtype=np.float32,
        )

        # 3. Factorizer初期化: 全数式の重み行列をSVD分解して共通の単語表現と数式表現を抽出
        n_comp = min(H1, F - 1, W - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        U_f = svd.fit_transform(scaled_coef).astype(np.float32)   # (F, n_comp)
        V_w = svd.components_.astype(np.float32)                  # (n_comp, W)

        rng = np.random.default_rng(42)
        act_f = active_pids // W
        act_w = active_pids % W
        w1_active = rng.normal(0.0, 0.01, size=(A, H1)).astype(np.float32)
        for h in range(n_comp):
            u_norm = U_f[:, h] / (np.linalg.norm(U_f[:, h]) + 1e-6)
            w1_active[:, h] += 0.15 * (u_norm[act_f] * V_w[h, act_w]) + 0.05 * V_w[h, act_w]

        self.b1 = np.full((F, H1), 0.25, dtype=np.float32)
        self.W2 = (
            np.eye(H1, H2, dtype=np.float32) * 0.5
            + rng.normal(0.0, 0.05, size=(H1, H2)).astype(np.float32)
        )
        self.b2 = np.full((H2,), 0.15, dtype=np.float32)
        self.W3 = rng.normal(0.0, 0.15, size=(H2,)).astype(np.float32)

        params = [w1_active, self.b1, self.W2, self.b2, self.W3]
        m_adam = [np.zeros_like(p) for p in params]
        v_adam = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, NNUE_EPOCHS + 1):
            # --- Forward pass: weight1[id(formula, word)][H1] のアキュムレータ合算 ---
            z1 = (M_active_csr @ w1_active).reshape(N, K, H1) + self.b1[cands]
            a1 = np.clip(z1, 0.0, 1.0)
            z2 = a1 @ self.W2 + self.b2
            a2 = np.clip(z2, 0.0, 1.0)
            nnue_scores = a2 @ self.W3

            total_scores = base_cand_scores + nnue_scores
            s_shift = total_scores - total_scores.max(axis=1, keepdims=True)
            exp_s = np.exp(s_shift)
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)

            dscores = probs
            dscores[:, 0] -= 1.0
            dscores /= N

            # --- Backward pass ---
            dW3 = np.einsum("nk,nkh->h", dscores, a2) + NNUE_L2_DECAY * self.W3
            da2 = dscores[:, :, None] * self.W3[None, None, :]
            dz2 = da2 * ((z2 > 0.0) & (z2 < 1.0))

            dW2 = np.einsum("nki,nkj->ij", a1, dz2) + NNUE_L2_DECAY * self.W2
            db2 = dz2.sum(axis=(0, 1))

            da1 = dz2 @ self.W2.T
            dz1 = da1 * ((z1 > 0.0) & (z1 < 1.0))

            d_w1 = (M_active_csr.T @ dz1.reshape(N * K, H1)) + NNUE_L2_DECAY * w1_active
            db1 = np.zeros_like(self.b1)
            np.add.at(db1, cands, dz1)

            grads = [d_w1, db1, dW2, db2, dW3]
            lr_t = NNUE_LR * (0.92 ** (epoch - 1))
            for idx_p, (p, g) in enumerate(zip(params, grads)):
                m_adam[idx_p] = beta1 * m_adam[idx_p] + (1.0 - beta1) * g
                v_adam[idx_p] = beta2 * v_adam[idx_p] + (1.0 - beta2) * (g * g)
                m_hat = m_adam[idx_p] / (1.0 - beta1 ** epoch)
                v_hat = v_adam[idx_p] / (1.0 - beta2 ** epoch)
                p -= lr_t * m_hat / (np.sqrt(v_hat) + eps)

        # フルサイズの weight1[id(formula, word)][H1] テーブルへ展開
        self.weight1 = np.zeros((F * W, H1), dtype=np.float32)
        self.weight1[active_pids] = w1_active

        sparse_weight1 = {}
        keep_mask = np.max(np.abs(w1_active), axis=1) > 2e-3
        for idx_a in np.where(keep_mask)[0]:
            pid = int(active_pids[idx_a])
            sparse_weight1[str(pid)] = [round(float(x), 5) for x in w1_active[idx_a]]

        sparse_base_w = {
            str(int(pid)): round(float(self.base_w[pid]), 5)
            for pid in np.where(np.abs(self.base_w) > 1e-4)[0]
        }

        save_data = {
            "format_version": "nnue_halfkp_v1",
            "h1": self.h1,
            "h2": self.h2,
            "equations": self.equations,
            "words": self.words,
            "b1": [[round(float(x), 5) for x in row] for row in self.b1],
            "W2": [[round(float(x), 5) for x in row] for row in self.W2],
            "b2": [round(float(x), 5) for x in self.b2],
            "W3": [round(float(x), 5) for x in self.W3],
            "base_b": [round(float(x), 5) for x in self.base_b],
            "base_w": sparse_base_w,
            "weight1": sparse_weight1,
        }

        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False)

        self.is_loaded = True
        print(
            f"NNUE重みファイルを保存しました: {weights_path} "
            f"(数式数: {F}, 語彙数: {W}, 有効2駒ペア数: {len(sparse_weight1)})"
        )

    def load(self, weights_path=WEIGHTS_FILE):
        if not os.path.exists(weights_path):
            return False
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("format_version") != "nnue_halfkp_v1":
                return False

            self.h1 = int(data["h1"])
            self.h2 = int(data["h2"])
            self.equations = data["equations"]
            self.eq_to_id = {eq: i for i, eq in enumerate(self.equations)}
            self.num_formulas = len(self.equations)

            self.words = data["words"]
            self.word_to_id = {w: i for i, w in enumerate(self.words)}
            self.num_words = len(self.words)

            F, W = self.num_formulas, self.num_words
            self.b1 = np.array(data["b1"], dtype=np.float32)
            self.W2 = np.array(data["W2"], dtype=np.float32)
            self.b2 = np.array(data["b2"], dtype=np.float32)
            self.W3 = np.array(data["W3"], dtype=np.float32)
            self.base_b = np.array(data["base_b"], dtype=np.float32)

            self.base_w = np.zeros(F * W, dtype=np.float32)
            for pid_str, val in data["base_w"].items():
                self.base_w[int(pid_str)] = float(val)

            # weight1[id(formula, word)][H1] の2次元配列を復元
            self.weight1 = np.zeros((F * W, self.h1), dtype=np.float32)
            for pid_str, vec in data["weight1"].items():
                self.weight1[int(pid_str)] = vec

            self.is_loaded = True
            return True
        except Exception:
            return False

    def predict(self, masked_question, values):
        time_tokens, same_u, diff_u, q_match, _ = analyze_question_structure(masked_question)
        feat_dict = time_tokens_to_feature_dict(time_tokens)
        num_vars = len(values)

        active_word_ids = [self.word_to_id[w] for w in feat_dict.keys() if w in self.word_to_id]
        active_w_arr = np.array(active_word_ids, dtype=np.int32)

        best_eq = None
        best_score = float("-inf")
        best_ans = None
        W = self.num_words

        for f_id, eq in enumerate(self.equations):
            if max_variable_index(eq) > num_vars:
                continue

            ok, ast_bonus, final_val = check_ast_and_dimensional_bonus(
                eq, values, same_u, diff_u, q_match
            )
            if not ok:
                continue

            if active_w_arr.size > 0:
                # 1. 2駒の数値ID配列: id(formula, word) = f_id * W + w_id
                pair_ids = f_id * W + active_w_arr

                # 2. 入力層 -> 第1隠れ層: weight1[id(formula, word)][H1] のアキュムレータ加算
                z1 = self.weight1[pair_ids].sum(axis=0) + self.b1[f_id]
                base_score = float(self.base_b[f_id] + self.base_w[pair_ids].sum())
            else:
                z1 = self.b1[f_id]
                base_score = float(self.base_b[f_id])

            # 3. Clipped ReLU による多駒関係(AND条件)の非線形合成
            a1 = np.clip(z1, 0.0, 1.0)
            z2 = a1 @ self.W2 + self.b2
            a2 = np.clip(z2, 0.0, 1.0)
            nnue_score = float(a2 @ self.W3)

            total_score = base_score + nnue_score + ast_bonus

            if total_score > best_score:
                best_score = total_score
                best_eq = eq
                best_ans = final_val

        return best_eq, best_score, best_ans


# ==========================================
# 7. データ読み込み関数（初回学習・評価用）
# ==========================================
def load_train_data(csv_path):
    dataset = []
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                question = row["Question"].strip()
                values = [float(v) for v in row["Numbers"].split()]
                answer = float(row["Answer"])
                masked_q = number_to_var(question)
                eq = normalize_equation(row["Equation"], prefix=True)
                if math.isclose(evaluate_with_eval(eq, values), answer, abs_tol=1e-5):
                    dataset.append((masked_q, eq, values, answer))
            except Exception:
                continue
    return dataset


def load_test_data(json_path):
    dataset = []
    with open(json_path, "r", encoding="utf-8") as file:
        for item in json.load(file):
            try:
                full_q = f"{item.get('Body', '').strip()} {item.get('Question', '').strip()}".strip()
                masked_q, values = mask_numbers_in_question(full_q)
                eq = normalize_equation(mask_numbers_in_equation(item["Equation"], values), prefix=False)
                dataset.append((masked_q, eq, values, float(item["Answer"])))
            except Exception:
                continue
    return dataset


# ==========================================
# 8. 外部ファイルから import して使うための公開API
# ==========================================
_GLOBAL_MODEL = MathNNUEModel()


def ensure_model_loaded():
    """
    重みファイル (math_lr_weights3.json) がNNUE形式であれば読み込むだけ。
    まだ存在しないか旧形式の場合は、学習データを読み込んで再学習・保存する。
    """
    if _GLOBAL_MODEL.is_loaded:
        return
    if _GLOBAL_MODEL.load(WEIGHTS_FILE):
        return
    print("NNUEモデルの重みファイルを初回学習して保存します...")
    download_files()
    train_data = load_train_data(TRAIN_FILE)
    _GLOBAL_MODEL.fit_and_save(train_data, WEIGHTS_FILE)


def solve(question_text, check_is_math=False):
    """
    【外部呼び出し用メイン関数】
    引数:
      question_text (str): 問題文のテキストのみ
      check_is_math (bool): True の場合、知識クイズなら None を返す
    戻り値:
      答えの数値のみ (整数なら int, 小数なら float / 解けない場合は None)
    """
    if check_is_math and not is_math_word_problem(question_text):
        return None

    ensure_model_loaded()

    masked_q, values = mask_numbers_in_question(question_text)
    if not values:
        return None

    _pred_eq, _score, pred_ans = _GLOBAL_MODEL.predict(masked_q, values)
    if pred_ans is None:
        return None

    if math.isclose(pred_ans, round(pred_ans), abs_tol=1e-6):
        return int(round(pred_ans))
    return round(pred_ans, 4)


# ==========================================
# 9. 単体実行時の動作確認・精度評価
# ==========================================
def evaluate_dataset(model, dataset, title):
    correct_eq = 0
    correct_ans = 0
    for q, true_eq, values, true_ans in dataset:
        pred_eq, _, pred_ans = model.predict(q, values)
        if pred_eq is None:
            continue
        if pred_eq == true_eq:
            correct_eq += 1
        if pred_ans is not None and math.isclose(pred_ans, true_ans, abs_tol=1e-4):
            correct_ans += 1
    total = len(dataset)
    print(f"[{title}]")
    print(f"  数式完全一致 : {correct_eq:4d} / {total} ({correct_eq / total * 100:.2f}%)")
    print(f"  数値ans正解率: {correct_ans:4d} / {total} ({correct_ans / total * 100:.2f}%)\n")


def main():
    ensure_model_loaded()

    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        train_data = load_train_data(TRAIN_FILE)
        test_data = load_test_data(TEST_FILE)
        evaluate_dataset(_GLOBAL_MODEL, train_data, "HalfKP型NNUE評価関数 - 学習データ")
        evaluate_dataset(_GLOBAL_MODEL, test_data,  "HalfKP型NNUE評価関数 - SVAMPテストデータ")

    sample_math = (
        "Each pack of dvds costs 76 dollars. "
        "If there is a discount of 25 dollars on each pack, "
        "how much do you have to pay to buy 5 packs?"
    )
    sample_quiz = (
        " is an online knowledge engine developed by Research , released in 2009 . [ 1 ] It is offered as an online service that answers queries by computing answers from externally sourced data . [ 2 ][ 3 ]"
    )

    print("=== solve(question_text) ＆ クイズ判定テスト ===")
    print(f"問題1: {sample_math}")
    print(f"  -> 算数判定: {is_math_word_problem(sample_math)}")
    print(f"  -> solve() の返り値: {solve(sample_math)}\n")

    print(f"問題2: {sample_quiz}")
    print(f"  -> 算数判定: {is_math_word_problem(sample_quiz)}")
    print(f"  -> solve(check_is_math=True) の返り値: {solve(sample_quiz, check_is_math=True)}")


if __name__ == "__main__":
    main()
