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
WEIGHTS_FILE = os.path.join(BASE_DIR, "math_lr_weights6.json")

NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
VAR_RE = re.compile(r"\bvar(\d+)\b")
TOKEN_VAR_RE = re.compile(r"(?<![a-zA-Z0-9])var(\d+)(?![a-zA-Z0-9])")
OPS = {"+", "-", "*", "/"}


def is_var_tok(w):
    """トークンが 'var1', 'var2' 等の数値マスク変数であるかを厳密に判定する ('varies' 等を除外)"""
    return len(w) > 3 and w.startswith("var") and w[3:].isdigit()


# 時間帯分割の設定 (t0=序盤, t1=中盤, t2=終盤 + all=全体)
N_TIME_BUCKETS = 3
TIME_KEYS = ("all", "t0", "t1", "t2")
TIME_FEATURE_SCALE = 0.35

# NNUE & ビームサーチ（スタック型段階探索）設定
HIDDEN1_SIZE = 16
HIDDEN2_SIZE = 16
TOP_K_CANDIDATES = 8
NNUE_EPOCHS = 18
NNUE_LR = 0.015
NNUE_L2_DECAY = 1e-4

BEAM_WIDTH = 2               # 各ステップで残す部分盤面のビーム幅
STEP_ACTIONS = ("ADD", "SUB_LR", "SUB_RL", "MUL", "DIV_LR", "DIV_RL", "NONE")
ACTIVE_ACTIONS = ("ADD", "SUB_LR", "SUB_RL", "MUL", "DIV_LR", "DIV_RL")
BEAM_SCORE_WEIGHT = 0.55     # ビームサーチの段階評価スコアの統合重み

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
# 2. クイズ vs 算数文章題のルーティング判定器
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
    if not text or not isinstance(text, str):
        return False

    text_clean = text.strip()
    text_low = text_clean.lower()

    if ENCYCLOPEDIA_PATTERNS.search(text_low):
        return False

    if not MATH_QUESTION_PATTERNS.search(text_low):
        if not ("?" in text_low and re.search(r"\bhow\b", text_low)):
            return False

    masked_vars = re.findall(r"\b(?:number|var)\d+\b", text_low)

    scrubbed = re.sub(
        r"\b(in|from|to|since|during|year|years|released in|born in|founded in)\s+(1[0-9]{3}|20[0-9]{2})\b",
        " ",
        text_low,
    )

    nums = NUMBER_RE.findall(scrubbed)
    total_num_count = len(nums) + len(masked_vars)

    if total_num_count < 2:
        word_nums = [w for w in re.findall(r"[a-z]+", scrubbed) if w in IMPLICIT_NUM_WORDS]
        per_a = re.findall(r"\b(?:a|an)\s+(?:minute|hour|day|week|month|year|pie|pack|box|bag|group|shirt|car)\b", scrubbed)
        if total_num_count + len(word_nums) + len(per_a) < 2:
            return False

    words_in_text = set(re.findall(r"[a-z]+", text_low))
    if not (words_in_text & MATH_OPERATION_CUES) and "$" not in text_low and "%" not in text_low:
        return False

    return True


# ==========================================
# 3. 数式の変換・正規化・ボトムアップAST分解
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


def decompose_ast_to_steps(expression):
    """
    正規化された数式 (例: '((var1 - var2) * var3)') のASTをボトムアップに分解し、
    1手ずつの (left_expr, op_str, right_expr, result_expr) のリストを返す。
    """
    tree = ast.parse(expression, mode="eval")
    steps = []

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            val = float(node.value)
            return str(int(val)) if val.is_integer() else str(val)
        if isinstance(node, ast.BinOp):
            l_expr = visit(node.left)
            r_expr = visit(node.right)
            if isinstance(node.op, ast.Add):
                op_s = "+"
            elif isinstance(node.op, ast.Sub):
                op_s = "-"
            elif isinstance(node.op, ast.Mult):
                op_s = "*"
            elif isinstance(node.op, ast.Div):
                op_s = "/"
            else:
                raise ValueError("Unsupported op")
            res_expr = normalize_equation(f"({l_expr} {op_s} {r_expr})", prefix=False)
            steps.append((l_expr, op_s, r_expr, res_expr))
            return res_expr
        raise ValueError("Unsupported node")

    visit(tree)
    return steps


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
    "each", "every", "per", "apiece", "all", "both", "more", "less", "fewer", "other", "another",
    "left", "remain", "remaining", "away", "back", "out", "up", "down", "off", "over",
    "now", "then", "still", "already", "only", "just", "also", "too", "together", "altogether",
    "total", "how", "many", "much", "long", "far", "old", "heavy",
}

# 単位スキャン時に文境界や前置詞・レート句を越えて無関係な名詞を拾わないための境界語集合
CLAUSE_BREAK_WORDS = {".", ",", "?", ";", "if", "when", "while", "after", "before", "but", "so", "how", "and"}
UNIT_STOP_WORDS = CLAUSE_BREAK_WORDS | {
    "on", "in", "at", "to", "for", "from", "with", "by", "into", "among", "between",
    "each", "every", "per", "apiece", "than", "as", "until", "or",
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
        var_indices = [idx for idx, w in enumerate(c) if is_var_tok(w)]
        if len(var_indices) == 1:
            var_spans[c[var_indices[0]]] = c
        elif len(var_indices) > 1:
            for k, v_idx in enumerate(var_indices):
                start = 0 if k == 0 else min(v_idx, var_indices[k - 1] + 2)
                end = len(c) if k == len(var_indices) - 1 else min(var_indices[k + 1], v_idx + 3)
                var_spans[c[v_idx]] = c[start:end]
    return clauses, var_spans


def analyze_question_structure(text):
    text_low = re.sub(r"[\u2018\u2019`]", "'", text.lower())
    text_low = re.sub(r"'s\b", "", text_low)
    text_low = re.sub(r"\$\s*(var\d+)", r"\1 dollar ", text_low)
    text_low = re.sub(r"[\u00a5\uffe5]\s*(var\d+)", r"\1 yen ", text_low)
    text_low = text_low.replace("%", " percent ")
    text_low = re.sub(r"(?<=[a-z0-9])\s*/\s*(?=[a-z])", " per ", text_low)
    text_low = re.sub(r"[:!]", ".", text_low)
    raw_tokens = re.findall(r"var\d+|[a-z]+|[.,?;]", text_low)
    words = []
    for w in raw_tokens:
        if w in (".", ",", "?", ";") or is_var_tok(w):
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

    # 句読点や前置詞・レート語(on, each等)を越えない厳密な単位抽出
    var_strict_units = {}
    for idx_w, w in enumerate(words):
        if is_var_tok(w):
            u_list = []
            for offset in range(1, 5):
                if idx_w + offset >= len(words):
                    break
                cand = words[idx_w + offset]
                if cand in UNIT_STOP_WORDS or is_var_tok(cand):
                    break
                if cand not in NON_UNIT_WORDS:
                    u_list.append(cand)
                if len(u_list) >= 2:
                    break
            var_strict_units[w] = (
                u_list[0] if len(u_list) >= 1 else "NONE",
                u_list[1] if len(u_list) >= 2 else "NONE",
            )

    var_units, var_time_buckets = {}, {}
    for i, w in enumerate(clean_words):
        if is_var_tok(w):
            var_time_buckets[w] = get_time_bucket(i, n_words)
            var_units[w] = var_strict_units.get(w, ("NONE", "NONE"))

    each_nouns = set()
    for i, w in enumerate(clean_words):
        if w in ("each", "every", "per", "apiece"):
            if i + 1 < n_words and not is_var_tok(clean_words[i + 1]):
                each_nouns.add(clean_words[i + 1])
            if i - 1 >= 0 and not is_var_tok(clean_words[i - 1]):
                each_nouns.add(clean_words[i - 1])

    asks_money = bool(q_words_set & {"pay", "spend", "cost", "price", "money", "dollar", "cent", "change", "earn", "save"}) or (
        "how" in q_words and "much" in q_words and not (q_words_set & WEIGHT_UNITS)
    )
    asks_time = ("how" in q_words and "long" in q_words) or bool(q_words_set & TIME_UNITS)
    asks_dist = ("how" in q_words and "far" in q_words) or bool(q_words_set & DIST_UNITS)
    asks_weight = ("how" in q_words and "heavy" in q_words) or "weigh" in q_words_set or bool(q_words_set & WEIGHT_UNITS)

    q_var_units = {q_words[i + 1] for i, w in enumerate(q_words) if is_var_tok(w) and i + 1 < len(q_words)}
    q_target_nouns = {
        w for i, w in enumerate(q_words)
        if w not in NON_UNIT_WORDS and not is_var_tok(w) and not (w in q_var_units and "many" not in q_words[:i])
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
        if is_var_tok(w):
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
        if is_var_tok(w) or w in MATH_CORE_WORDS:
            abstract_items.append((w, t_b))
        elif not abstract_items or abstract_items[-1][0] != "OBJ":
            abstract_items.append(("OBJ", t_b))

    for w, t_b in abstract_items:
        if w != "OBJ":
            add_tok(t_b, w)
    for i in range(len(abstract_items) - 1):
        (w1, tb1), (w2, tb2) = abstract_items[i], abstract_items[i + 1]
        if w1 != "OBJ" or w2 != "OBJ":
            add_tok(tb2 if is_var_tok(w2) else tb1, f"{w1}_{w2}")

    q_abstract = [
        (clean_words[idx], get_time_bucket(idx, n_words))
        for idx in range(q_idx, n_words)
        if is_var_tok(clean_words[idx]) or clean_words[idx] in MATH_CORE_WORDS
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
    feat = {}
    for t_key, tok_set in time_tokens.items():
        scale = 1.0 if t_key == "all" else TIME_FEATURE_SCALE
        for w in tok_set:
            feat[f"{t_key}::{w}"] = scale
    return feat


# ==========================================
# 5. ビームサーチ（スタック型段階探索）用の状態・単位代数・特徴抽出
# ==========================================
def extract_rich_question_info(text, values=None):
    """
    各変数ノード(VarNode)の単位・レート分母(each/per)・減算/加算ヒント語・節情報と、
    質問文が求めるターゲット単位を抽出する。
    """
    text_low = re.sub(r"[\u2018\u2019`]", "'", text.lower())
    text_low = re.sub(r"'s\b", "", text_low)
    text_low = re.sub(r"\$\s*(var\d+)", r"\1 dollar ", text_low)
    text_low = re.sub(r"[\u00a5\uffe5]\s*(var\d+)", r"\1 yen ", text_low)
    text_low = text_low.replace("%", " percent ")
    text_low = re.sub(r"(?<=[a-z0-9])\s*/\s*(?=[a-z])", " per ", text_low)
    text_low = re.sub(r"[:!]", ".", text_low)
    raw_tokens = re.findall(r"var\d+|[a-z]+|[.,?;]", text_low)

    words = []
    for w in raw_tokens:
        if w in (".", ",", "?", ";") or is_var_tok(w):
            words.append(w)
        elif w not in STOP_WORDS:
            words.append(simple_lemmatize(w))

    clauses, var_spans = get_var_spans(words)
    clean_words = [w for w in words if w not in (".", ",", "?", ";")]
    n_words = len(clean_words)

    q_idx = 0
    for idx in range(n_words - 1, -1, -1):
        if clean_words[idx] in ("how", "what"):
            q_idx = idx
            break
    q_words = clean_words[q_idx:]
    q_words_set = set(q_words)

    var_clause_id = {}
    var_full_clause = {}
    for c_id, c in enumerate(clauses):
        for w in c:
            if is_var_tok(w):
                var_clause_id[w] = c_id
                var_full_clause[w] = c

    var_strict_units = {}
    for i, w in enumerate(words):
        if is_var_tok(w):
            u_list = []
            for offset in range(1, 5):
                if i + offset >= len(words):
                    break
                cand = words[i + offset]
                if cand in UNIT_STOP_WORDS or is_var_tok(cand):
                    break
                if cand not in NON_UNIT_WORDS:
                    u_list.append(cand)
                if len(u_list) >= 2:
                    break
            var_strict_units[w] = (
                u_list[0] if len(u_list) >= 1 else "NONE",
                u_list[1] if len(u_list) >= 2 else "NONE",
            )

    q_asks_rate = bool(q_words_set & {"each", "every", "per", "apiece", "one"})
    q_rate_nouns = set()
    for i, w in enumerate(q_words):
        if w in ("each", "every", "per", "apiece", "one"):
            if i + 1 < len(q_words) and not is_var_tok(q_words[i + 1]):
                q_rate_nouns.add(q_words[i + 1])

    asks_money = bool(q_words_set & {"pay", "spend", "cost", "price", "money", "dollar", "cent", "change", "earn", "save"}) or (
        "how" in q_words and "much" in q_words and not (q_words_set & WEIGHT_UNITS)
    )
    asks_time = ("how" in q_words and "long" in q_words) or bool(q_words_set & TIME_UNITS)
    asks_dist = ("how" in q_words and "far" in q_words) or bool(q_words_set & DIST_UNITS)
    asks_weight = ("how" in q_words and "heavy" in q_words) or "weigh" in q_words_set or bool(q_words_set & WEIGHT_UNITS)

    q_var_units = {q_words[i + 1] for i, w in enumerate(q_words) if is_var_tok(w) and i + 1 < len(q_words)}
    q_target_nouns = {
        w for i, w in enumerate(q_words)
        if w not in NON_UNIT_WORDS and not is_var_tok(w) and w not in q_rate_nouns
        and not (w in q_var_units and "many" not in q_words[:i])
    }

    each_nouns = set()
    for i, w in enumerate(clean_words):
        if w in ("each", "every", "per", "apiece"):
            if i + 1 < n_words and not is_var_tok(clean_words[i + 1]):
                each_nouns.add(clean_words[i + 1])
            if i - 1 >= 0 and not is_var_tok(clean_words[i - 1]):
                each_nouns.add(clean_words[i - 1])

    var_nodes = {}
    for i, w in enumerate(clean_words):
        if is_var_tok(w):
            v_num = int(w[3:])
            val = float(values[v_num - 1]) if values is not None and v_num - 1 < len(values) else float(v_num)
            u1, u2 = var_strict_units.get(w, ("NONE", "NONE"))
            units = {u for u in (u1, u2) if u != "NONE"}

            span = var_spans.get(w, [])
            full_c = var_full_clause.get(w, span)
            span_set = set(span)
            full_c_set = set(full_c)

            rate_units = set()
            for seq in (span, full_c):
                for idx_s, sw in enumerate(seq):
                    if sw in ("each", "every", "per", "apiece"):
                        if idx_s + 1 < len(seq):
                            cand_r = seq[idx_s + 1]
                            if (
                                not is_var_tok(cand_r)
                                and cand_r not in NON_UNIT_WORDS
                                and cand_r not in units
                            ):
                                rate_units.add(cand_r)
                        if idx_s - 1 >= 0 and sw in ("each", "apiece"):
                            cand_r = seq[idx_s - 1]
                            if (
                                not is_var_tok(cand_r)
                                and cand_r not in NON_UNIT_WORDS
                                and cand_r not in units
                            ):
                                rate_units.add(cand_r)

            prev1 = clean_words[i - 1] if i > 0 and clean_words[i - 1] in MATH_CORE_WORDS else "OBJ"
            prev2 = clean_words[i - 2] if i > 1 and clean_words[i - 2] in MATH_CORE_WORDS else "OBJ"

            is_q_match = bool(
                (units & q_target_nouns)
                or (asks_money and (units & MONEY_UNITS))
                or (asks_time and (units & TIME_UNITS))
                or (asks_dist and (units & DIST_UNITS))
                or (asks_weight and (units & WEIGHT_UNITS))
            )

            var_nodes[w] = {
                "name": w,
                "expr": w,
                "val": val,
                "u1": u1,
                "u2": u2,
                "units": units,
                "rate_units": rate_units,
                "cues_sub": span_set & SUB_CUES,
                "cues_add": span_set & ADD_CUES,
                "has_each": bool((span_set | full_c_set) & {"each", "every", "per", "apiece"}),
                "role_each_noun": bool(units & each_nouns),
                "in_q_clause": w in q_words_set,
                "q_match": is_q_match,
                "prev1": prev1,
                "prev2": prev2,
                "pos_idx": i,
                "var_order": v_num,
                "clause_id": var_clause_id.get(w, -1),
                "vars_used": frozenset([w]),
                "depth": 0,
                "last_op": "INIT",
            }

    q_abstract = [
        clean_words[idx]
        for idx in range(q_idx, n_words)
        if is_var_tok(clean_words[idx]) or clean_words[idx] in MATH_CORE_WORDS
    ]

    q_ctx = {
        "clean_words": clean_words,
        "q_words": q_words,
        "q_words_set": q_words_set,
        "q_abstract": q_abstract,
        "q_target_nouns": q_target_nouns,
        "q_asks_rate": q_asks_rate,
        "q_rate_nouns": q_rate_nouns,
        "asks_money": asks_money,
        "asks_time": asks_time,
        "asks_dist": asks_dist,
        "asks_weight": asks_weight,
        "each_nouns": each_nouns,
        "num_vars": len(var_nodes),
    }
    return var_nodes, q_ctx


def apply_step_operation(nodeL, nodeR, action, board_nodes, q_ctx):
    """
    2つの変数ノード (nodeL, nodeR) に1手分の演算 action を適用し、
    新しい合成ノード (new_node) と1手分の次元解析・約分ボーナス (step_dim_bonus) を返す。
    """
    if action == "ADD":
        op_L, op_R, op_sym = nodeL, nodeR, "+"
    elif action == "SUB_LR":
        op_L, op_R, op_sym = nodeL, nodeR, "-"
    elif action == "SUB_RL":
        op_L, op_R, op_sym = nodeR, nodeL, "-"
    elif action == "MUL":
        op_L, op_R, op_sym = nodeL, nodeR, "*"
    elif action == "DIV_LR":
        op_L, op_R, op_sym = nodeL, nodeR, "/"
    elif action == "DIV_RL":
        op_L, op_R, op_sym = nodeR, nodeL, "/"
    else:
        return None

    val_L, val_R = op_L["val"], op_R["val"]
    if op_sym == "+":
        new_val = val_L + val_R
    elif op_sym == "-":
        new_val = val_L - val_R
        if new_val <= 1e-9:
            return None
    elif op_sym == "*":
        new_val = val_L * val_R
        if new_val <= 1e-9:
            return None
    elif op_sym == "/":
        if abs(val_R) < 1e-9:
            return None
        new_val = val_L / val_R
        if new_val <= 1e-9:
            return None
    else:
        return None

    uL, uR = op_L["units"], op_R["units"]
    rL, rR = op_L["rate_units"], op_R["rate_units"]

    same_unit = bool(
        (uL & uR)
        or ((uL & MONEY_UNITS) and (uR & MONEY_UNITS))
        or ((uL & TIME_UNITS) and (uR & TIME_UNITS))
    )
    diff_unit = bool(uL and uR and not same_unit)

    step_dim_bonus = 0.0
    other_nodes = [
        n for n in board_nodes
        if n["expr"] != nodeL["expr"] and n["expr"] != nodeR["expr"]
    ]

    if op_sym in ("+", "-"):
        new_units = uL | uR
        new_rate_units = rL | rR
        if same_unit:
            step_dim_bonus += 1.5
            if rL and rR and (rL & rR):
                step_dim_bonus += 1.8
            elif bool(rL) != bool(rR) and (op_L["last_op"] == "*" or op_R["last_op"] == "*"):
                step_dim_bonus -= 3.5

            if op_sym == "-":
                if op_R["cues_sub"] and not op_L["cues_sub"]:
                    step_dim_bonus += 1.6
                elif op_L["cues_sub"] and not op_R["cues_sub"] and op_L["depth"] == 0:
                    step_dim_bonus -= 1.5
            else:
                if "discount" in (op_L["cues_sub"] | op_R["cues_sub"]):
                    step_dim_bonus -= 2.2
        elif diff_unit:
            step_dim_bonus -= 4.5

    elif op_sym == "*":
        cancel_lr = bool(rL & uR)
        cancel_rl = bool(rR & uL)
        if cancel_lr and not cancel_rl:
            new_units = set(uL)
            new_rate_units = rL - uR
            step_dim_bonus += 2.0
            if op_L["depth"] >= 1:
                step_dim_bonus += 1.5
            elif "discount" in op_L["cues_sub"]:
                if any(bool(n["units"] & uL) and bool(n["rate_units"] & rL) for n in other_nodes):
                    step_dim_bonus -= 2.0
        elif cancel_rl and not cancel_lr:
            new_units = set(uR)
            new_rate_units = rR - uL
            step_dim_bonus += 2.0
            if op_R["depth"] >= 1:
                step_dim_bonus += 1.5
            elif "discount" in op_R["cues_sub"]:
                if any(bool(n["units"] & uR) and bool(n["rate_units"] & rR) for n in other_nodes):
                    step_dim_bonus -= 2.0
        else:
            new_units = uL if op_L["q_match"] else (uR if op_R["q_match"] else (uL | uR))
            new_rate_units = set()
            if same_unit and uL:
                step_dim_bonus -= 1.8

    else:  # "/"
        if abs(new_val - round(new_val)) > 1e-6:
            if abs(new_val * 4 - round(new_val * 4)) > 1e-5 and abs(new_val * 10 - round(new_val * 10)) > 1e-5:
                step_dim_bonus -= 2.5
            else:
                step_dim_bonus -= 0.5
        if same_unit and rR:
            new_units = set(rR)
            new_rate_units = set()
            step_dim_bonus += 1.5
        elif diff_unit:
            new_units = set(uL)
            new_rate_units = set(uR)
            if q_ctx["q_asks_rate"]:
                step_dim_bonus += 1.2
        else:
            new_units = set(uL)
            new_rate_units = set()

    raw_expr = f"({op_L['expr']} {op_sym} {op_R['expr']})"
    norm_expr = normalize_equation(raw_expr, prefix=False)

    is_q_match = bool(
        (new_units & q_ctx["q_target_nouns"])
        or (q_ctx["asks_money"] and (new_units & MONEY_UNITS))
        or (q_ctx["asks_time"] and (new_units & TIME_UNITS))
        or (q_ctx["asks_dist"] and (new_units & DIST_UNITS))
        or (q_ctx["asks_weight"] and (new_units & WEIGHT_UNITS))
    )

    new_node = {
        "name": norm_expr,
        "expr": norm_expr,
        "val": new_val,
        "u1": next(iter(new_units), "NONE"),
        "u2": "NONE",
        "units": new_units,
        "rate_units": new_rate_units,
        "cues_sub": (op_L["cues_sub"] | op_R["cues_sub"]) if op_sym != "-" else set(),
        "cues_add": set(),
        "has_each": bool(new_rate_units),
        "role_each_noun": False,
        "in_q_clause": op_L["in_q_clause"] or op_R["in_q_clause"],
        "q_match": is_q_match,
        "prev1": op_L["prev1"],
        "prev2": op_L["prev2"],
        "pos_idx": min(op_L["pos_idx"], op_R["pos_idx"]),
        "var_order": min(op_L["var_order"], op_R["var_order"]),
        "clause_id": op_L["clause_id"],
        "vars_used": op_L["vars_used"] | op_R["vars_used"],
        "depth": max(op_L["depth"], op_R["depth"]) + 1,
        "last_op": op_sym,
    }
    return new_node, step_dim_bonus


def extract_step_pair_features(nodeL, nodeR, q_ctx, step_num=1):
    """
    盤面上の2つのノード (nodeL, nodeR) に対して、1手読みステップ評価器の特徴辞書を抽出する。
    """
    feat = {}
    uL, uR = nodeL["units"], nodeR["units"]
    rL, rR = nodeL["rate_units"], nodeR["rate_units"]

    same_unit = bool(
        (uL & uR)
        or ((uL & MONEY_UNITS) and (uR & MONEY_UNITS))
        or ((uL & TIME_UNITS) and (uR & TIME_UNITS))
    )
    diff_unit = bool(uL and uR and not same_unit)

    feat[f"S{min(step_num, 2)}::BIAS"] = 1.0
    feat[f"NVARS::{min(q_ctx['num_vars'], 4)}"] = 1.0

    if same_unit:
        feat["PAIR::SAME_UNIT"] = 1.0
    elif diff_unit:
        feat["PAIR::DIFF_UNIT"] = 1.0

    if rL and rR and (rL & rR):
        feat["PAIR::SAME_RATE_DENOM"] = 1.0
    if bool(rL) != bool(rR) and same_unit:
        feat["PAIR::RATE_MISMATCH"] = 1.0

    if rL & uR:
        feat["PAIR::CANCEL_LR"] = 1.0
    if rR & uL:
        feat["PAIR::CANCEL_RL"] = 1.0

    if nodeR["var_order"] == nodeL["var_order"] + 1:
        feat["PAIR::ADJACENT"] = 1.0
    else:
        feat["PAIR::NON_ADJACENT"] = 1.0

    if nodeL["clause_id"] == nodeR["clause_id"] and nodeL["clause_id"] >= 0:
        feat["PAIR::SAME_CLAUSE"] = 1.0

    # L側ノードの特徴
    feat[f"L::LAST_OP_{nodeL['last_op']}"] = 1.0
    feat[f"L::PREV1_{nodeL['prev1']}"] = 1.0
    feat[f"L::PREV2_{nodeL['prev2']}_{nodeL['prev1']}"] = 1.0
    if nodeL["has_each"]:
        feat["L::HAS_EACH"] = 1.0
    if nodeL["role_each_noun"]:
        feat["L::ROLE_EACH_NOUN"] = 1.0
    if nodeL["q_match"]:
        feat["L::Q_MATCH"] = 1.0
    else:
        feat["L::Q_NOMATCH"] = 1.0
    if nodeL["in_q_clause"]:
        feat["L::IN_Q_CLAUSE"] = 1.0
    if nodeL["cues_sub"]:
        feat["L::ANY_SUB"] = 1.0
        for sh in nodeL["cues_sub"]:
            feat[f"L::SUB_{sh}"] = 1.0
    if nodeL["cues_add"]:
        feat["L::ANY_ADD"] = 1.0
        for ah in nodeL["cues_add"]:
            feat[f"L::ADD_{ah}"] = 1.0

    # R側ノードの特徴
    feat[f"R::LAST_OP_{nodeR['last_op']}"] = 1.0
    feat[f"R::PREV1_{nodeR['prev1']}"] = 1.0
    feat[f"R::PREV2_{nodeR['prev2']}_{nodeR['prev1']}"] = 1.0
    if nodeR["has_each"]:
        feat["R::HAS_EACH"] = 1.0
    if nodeR["role_each_noun"]:
        feat["R::ROLE_EACH_NOUN"] = 1.0
    if nodeR["q_match"]:
        feat["R::Q_MATCH"] = 1.0
    else:
        feat["R::Q_NOMATCH"] = 1.0
    if nodeR["in_q_clause"]:
        feat["R::IN_Q_CLAUSE"] = 1.0
    if nodeR["cues_sub"]:
        feat["R::ANY_SUB"] = 1.0
        for sh in nodeR["cues_sub"]:
            feat[f"R::SUB_{sh}"] = 1.0
    if nodeR["cues_add"]:
        feat["R::ANY_ADD"] = 1.0
        for ah in nodeR["cues_add"]:
            feat[f"R::ADD_{ah}"] = 1.0

    # LとRの複合条件特徴
    if same_unit and nodeR["cues_sub"] and not nodeL["cues_sub"]:
        feat["COMBO::SAME_U_R_SUB"] = 1.0
    if same_unit and nodeL["cues_sub"] and not nodeR["cues_sub"]:
        feat["COMBO::SAME_U_L_SUB"] = 1.0
    if same_unit and nodeR["cues_add"]:
        feat["COMBO::SAME_U_R_ADD"] = 1.0
    if (rL & uR) and not nodeL["cues_sub"]:
        feat["COMBO::CLEAN_RATE_MUL_LR"] = 1.0
    if (rR & uL) and not nodeR["cues_sub"]:
        feat["COMBO::CLEAN_RATE_MUL_RL"] = 1.0

    if q_ctx["q_asks_rate"]:
        feat["Q::ASKS_RATE"] = 1.0
    for qw in q_ctx["q_abstract"]:
        if not is_var_tok(qw):
            feat[f"Q::{qw}"] = 0.6
    for idx_q in range(len(q_ctx["q_abstract"]) - 1):
        w1, w2 = q_ctx["q_abstract"][idx_q], q_ctx["q_abstract"][idx_q + 1]
        if not is_var_tok(w1) and not is_var_tok(w2):
            feat[f"QBI::{w1}_{w2}"] = 0.6

    return feat


def permute_question_vars(question, eq, values, mapping):
    def repl(m):
        v = f"var{m.group(1)}"
        return mapping.get(v, v)

    new_q = TOKEN_VAR_RE.sub(repl, question)
    new_eq = normalize_equation(TOKEN_VAR_RE.sub(repl, eq), prefix=False)
    max_v = max(len(values), max((int(v[3:]) for v in mapping.values()), default=0))
    new_values = [1.0] * max_v
    for idx_old, val in enumerate(values):
        v_old = f"var{idx_old + 1}"
        v_new = mapping.get(v_old, v_old)
        new_idx = int(v_new[3:]) - 1
        if 0 <= new_idx < max_v:
            new_values[new_idx] = val
    return new_q, new_eq, new_values


def extract_unified_step_samples(question, eq, values):
    """
    1つの問題 (question, eq, values) からASTをボトムアップに分解し、
    1手目・2手目の (特徴辞書, 正解アクションラベル) リストを生成する。
    """
    try:
        steps = decompose_ast_to_steps(eq)
    except Exception:
        return []
    if not steps:
        return []

    var_nodes, q_ctx = extract_rich_question_info(question, values)
    if len(var_nodes) < 2:
        return []

    samples = []
    leaf_actions = {}
    first_step_node = None
    first_step_pair = None

    for l_expr, op_s, r_expr, _res_expr in steps:
        if l_expr in var_nodes and r_expr in var_nodes and l_expr != r_expr:
            n1, n2 = var_nodes[l_expr], var_nodes[r_expr]
            if n1["var_order"] < n2["var_order"]:
                nodeL, nodeR = n1, n2
                is_lr = True
            else:
                nodeL, nodeR = n2, n1
                is_lr = False

            if op_s == "+":
                act = "ADD"
            elif op_s == "-":
                act = "SUB_LR" if is_lr else "SUB_RL"
            elif op_s == "*":
                act = "MUL"
            elif op_s == "/":
                act = "DIV_LR" if is_lr else "DIV_RL"
            else:
                continue

            pair_key = (nodeL["name"], nodeR["name"])
            leaf_actions[pair_key] = act
            if first_step_node is None:
                res_out = apply_step_operation(nodeL, nodeR, act, list(var_nodes.values()), q_ctx)
                if res_out is not None:
                    first_step_node, _ = res_out
                    first_step_pair = pair_key

    sorted_vars = sorted(var_nodes.keys(), key=lambda v: var_nodes[v]["var_order"])
    for i in range(len(sorted_vars)):
        for j in range(i + 1, len(sorted_vars)):
            v_a, v_b = sorted_vars[i], sorted_vars[j]
            nodeL, nodeR = var_nodes[v_a], var_nodes[v_b]
            f_dict = extract_step_pair_features(nodeL, nodeR, q_ctx, step_num=1)
            label = leaf_actions.get((v_a, v_b), "NONE")
            samples.append((f_dict, label))

    if first_step_node is not None and len(sorted_vars) >= 3:
        rem_vars = [v for v in sorted_vars if v not in first_step_pair]
        if len(steps) == 1:
            for v_rem in rem_vars:
                f_dict = extract_step_pair_features(first_step_node, var_nodes[v_rem], q_ctx, step_num=2)
                samples.append((f_dict, "NONE"))
        elif len(steps) >= 2:
            l2_expr, op2_s, r2_expr, _ = steps[1]
            for v_rem in rem_vars:
                nodeC = var_nodes[v_rem]
                f_dict = extract_step_pair_features(first_step_node, nodeC, q_ctx, step_num=2)
                if l2_expr == v_rem or r2_expr == v_rem:
                    is_res_left = (r2_expr == v_rem)
                    if op2_s == "+":
                        act2 = "ADD"
                    elif op2_s == "-":
                        act2 = "SUB_LR" if is_res_left else "SUB_RL"
                    elif op2_s == "*":
                        act2 = "MUL"
                    elif op2_s == "/":
                        act2 = "DIV_LR" if is_res_left else "DIV_RL"
                    else:
                        act2 = "NONE"
                    samples.append((f_dict, act2))
                else:
                    samples.append((f_dict, "NONE"))

    return samples


# ==========================================
# 6. AST評価 ＆ 次元解析・非負チェック
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
# 7. ビームサーチ（スタック型段階探索）＋ HalfKP型NNUE ハイブリッドモデル
# ==========================================
class MathNNUEModel:
    """
    スタック型ビームサーチ（段階探索）＋ HalfKP型NNUEハイブリッド評価関数:
      1. 1手読みステップ評価器 (step_w, step_b) と単位・レート約分代数により、
         ビーム幅 BEAM_WIDTH で盤面上の変数を1手ずつスタック的に縮約・探索する。
      2. 大局的な HalfKP型NNUE + BoW線形ベースラインで数式全体の整合性を評価し、
         ビームサーチの段階探索スコアと統合して最終的な数式・計算結果を出力する。
    """
    def __init__(self, h1=HIDDEN1_SIZE, h2=HIDDEN2_SIZE, beam_width=BEAM_WIDTH):
        self.h1 = h1
        self.h2 = h2
        self.beam_width = beam_width
        self.equations = []
        self.eq_to_id = {}
        self.words = []
        self.word_to_id = {}
        self.num_formulas = 0
        self.num_words = 0

        # HalfKP型NNUE テーブル
        self.weight1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.base_w = None
        self.base_b = None

        # ビームサーチ用 1手読みステップ評価器テーブル
        self.step_actions = list(STEP_ACTIONS)
        self.step_act_to_id = {a: i for i, a in enumerate(self.step_actions)}
        self.step_words = []
        self.step_word_to_id = {}
        self.step_w = None    # shape: [A_step, W_step]
        self.step_b = None    # shape: [A_step]

        self.is_loaded = False

    def fit_and_save(self, dataset, weights_path=WEIGHTS_FILE):
        X_dicts, y_labels = [], []
        step_X_dicts, step_y_labels = [], []

        for question, eq, values, _ans in dataset:
            base_time_tokens, var_tb = tokenize_by_time(question)
            X_dicts.append(time_tokens_to_feature_dict(base_time_tokens))
            y_labels.append(eq)

            for s_feat, s_lbl in extract_unified_step_samples(question, eq, values):
                step_X_dicts.append(s_feat)
                step_y_labels.append(s_lbl)

            if len(values) == 2 and max_variable_index(eq) == 2:
                sw_map = {"var1": "var2", "var2": "var1"}
                sw_tokens, sw_eq = permute_time_tokens_and_eq(base_time_tokens, var_tb, eq, sw_map)
                X_dicts.append(time_tokens_to_feature_dict(sw_tokens))
                y_labels.append(sw_eq)

                sw_q, sw_eq2, sw_vals = permute_question_vars(question, eq, values, sw_map)
                for s_feat, s_lbl in extract_unified_step_samples(sw_q, sw_eq2, sw_vals):
                    step_X_dicts.append(s_feat)
                    step_y_labels.append(s_lbl)

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
                    p_map = dict(zip(vars_3, p))
                    perm_tokens, perm_eq = permute_time_tokens_and_eq(base_time_tokens, var_tb, eq, p_map)
                    X_dicts.append(time_tokens_to_feature_dict(perm_tokens))
                    y_labels.append(perm_eq)

                    p_q, p_eq2, p_vals = permute_question_vars(question, eq, values, p_map)
                    for s_feat, s_lbl in extract_unified_step_samples(p_q, p_eq2, p_vals):
                        step_X_dicts.append(s_feat)
                        step_y_labels.append(s_lbl)

        # ---------------------------------------------------------
        # 1. 大局的BoW + HalfKP型NNUEの学習
        # ---------------------------------------------------------
        vectorizer = DictVectorizer(sparse=True)
        X_csr = vectorizer.fit_transform(X_dicts).astype(np.float32)

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

        word_scales = np.array(
            [1.0 if w.startswith("all::") else TIME_FEATURE_SCALE for w in self.words],
            dtype=np.float32,
        )
        scaled_coef = coef_mat * word_scales[None, :]

        self.base_w = scaled_coef.reshape(F * W).copy()
        self.base_b = intercept_vec.copy()

        K = min(TOP_K_CANDIDATES, F)
        y_ids = np.array([self.eq_to_id[y] for y in y_labels], dtype=np.int32)

        base_logits = (X_csr @ coef_mat.T) + intercept_vec[None, :]
        cands = np.zeros((N, K), dtype=np.int32)
        cands[:, 0] = y_ids
        if K > 1:
            base_logits_copy = base_logits.copy()
            base_logits_copy[np.arange(N), y_ids] = -1e9
            top_neg = np.argpartition(-base_logits_copy, kth=K - 1, axis=1)[:, : K - 1]
            cands[:, 1:] = top_neg

        base_cand_scores = base_logits[np.arange(N)[:, None], cands]

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

        active_pids, compact_col_indices = np.unique(pair_indices, return_inverse=True)
        A = len(active_pids)
        M_active_csr = sparse.csr_matrix(
            (np.ones(total_nnz, dtype=np.float32), compact_col_indices.astype(np.int32), M_indptr),
            shape=(N * K, A),
            dtype=np.float32,
        )

        n_comp = min(H1, F - 1, W - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        U_f = svd.fit_transform(scaled_coef).astype(np.float32)
        V_w = svd.components_.astype(np.float32)

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

        self.weight1 = np.zeros((F * W, H1), dtype=np.float32)
        self.weight1[active_pids] = w1_active

        # ---------------------------------------------------------
        # 2. ビームサーチ用 1手読みステップ評価器 (step_clf) の学習
        # ---------------------------------------------------------
        A_step = len(self.step_actions)
        if step_X_dicts:
            step_vec = DictVectorizer(sparse=True)
            X_step_csr = step_vec.fit_transform(step_X_dicts).astype(np.float32)
            self.step_words = [str(w) for w in step_vec.get_feature_names_out()]
            self.step_word_to_id = {w: i for i, w in enumerate(self.step_words)}
            W_step = len(self.step_words)

            step_clf = LogisticRegression(C=0.6, max_iter=600)
            step_clf.fit(X_step_csr, step_y_labels)

            self.step_w = np.zeros((A_step, W_step), dtype=np.float32)
            self.step_b = np.full(A_step, -2.0, dtype=np.float32)

            if len(step_clf.classes_) == 2 and step_clf.coef_.shape[0] == 1:
                c0, c1 = str(step_clf.classes_[0]), str(step_clf.classes_[1])
                id0, id1 = self.step_act_to_id[c0], self.step_act_to_id[c1]
                self.step_w[id0] = -step_clf.coef_[0].astype(np.float32)
                self.step_w[id1] = step_clf.coef_[0].astype(np.float32)
                self.step_b[id0] = float(-step_clf.intercept_[0])
                self.step_b[id1] = float(step_clf.intercept_[0])
            else:
                for idx_c, cls_name in enumerate(step_clf.classes_):
                    act_id = self.step_act_to_id[str(cls_name)]
                    self.step_w[act_id] = step_clf.coef_[idx_c].astype(np.float32)
                    self.step_b[act_id] = float(step_clf.intercept_[idx_c])
        else:
            self.step_words = []
            self.step_word_to_id = {}
            self.step_w = np.zeros((A_step, 0), dtype=np.float32)
            self.step_b = np.zeros(A_step, dtype=np.float32)

        # ---------------------------------------------------------
        # 3. 重みファイルへ保存
        # ---------------------------------------------------------
        sparse_weight1 = {}
        keep_mask = np.max(np.abs(w1_active), axis=1) > 2e-3
        for idx_a in np.where(keep_mask)[0]:
            pid = int(active_pids[idx_a])
            sparse_weight1[str(pid)] = [round(float(x), 5) for x in w1_active[idx_a]]

        sparse_base_w = {
            str(int(pid)): round(float(self.base_w[pid]), 5)
            for pid in np.where(np.abs(self.base_w) > 1e-4)[0]
        }

        flat_step_w = self.step_w.reshape(-1)
        sparse_step_w = {
            str(int(pid)): round(float(flat_step_w[pid]), 5)
            for pid in np.where(np.abs(flat_step_w) > 1e-4)[0]
        }

        save_data = {
            "format_version": "beam_stack_nnue_v4",
            "h1": self.h1,
            "h2": self.h2,
            "beam_width": self.beam_width,
            "equations": self.equations,
            "words": self.words,
            "b1": [[round(float(x), 5) for x in row] for row in self.b1],
            "W2": [[round(float(x), 5) for x in row] for row in self.W2],
            "b2": [round(float(x), 5) for x in self.b2],
            "W3": [round(float(x), 5) for x in self.W3],
            "base_b": [round(float(x), 5) for x in self.base_b],
            "base_w": sparse_base_w,
            "weight1": sparse_weight1,
            "step_actions": self.step_actions,
            "step_words": self.step_words,
            "step_b": [round(float(x), 5) for x in self.step_b],
            "step_w": sparse_step_w,
        }

        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False)

        self.is_loaded = True
        print(
            f"ビームサーチ+NNUE重みファイルを保存しました: {weights_path} "
            f"(数式数: {F}, 語彙数: {W}, ステップ特徴数: {len(self.step_words)})"
        )

    def load(self, weights_path=WEIGHTS_FILE):
        if not os.path.exists(weights_path):
            return False
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("format_version") != "beam_stack_nnue_v4":
                return False

            self.h1 = int(data["h1"])
            self.h2 = int(data["h2"])
            self.beam_width = int(data.get("beam_width", BEAM_WIDTH))
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

            self.weight1 = np.zeros((F * W, self.h1), dtype=np.float32)
            for pid_str, vec in data["weight1"].items():
                self.weight1[int(pid_str)] = vec

            self.step_actions = data["step_actions"]
            self.step_act_to_id = {a: i for i, a in enumerate(self.step_actions)}
            self.step_words = data["step_words"]
            self.step_word_to_id = {w: i for i, w in enumerate(self.step_words)}
            A_step, W_step = len(self.step_actions), len(self.step_words)

            self.step_b = np.array(data["step_b"], dtype=np.float32)
            flat_step_w = np.zeros(A_step * W_step, dtype=np.float32)
            for pid_str, val in data["step_w"].items():
                flat_step_w[int(pid_str)] = float(val)
            self.step_w = flat_step_w.reshape(A_step, W_step)

            self.is_loaded = True
            return True
        except Exception:
            return False

    def score_step_pair(self, nodeL, nodeR, q_ctx, step_num=1):
        """
        盤面上の2つのノード (nodeL, nodeR) に対し、step_clf を用いて
        6種類の演算それぞれのステップスコアを計算する。
        """
        if self.step_w is None or self.step_w.shape[1] == 0:
            return {act: 0.0 for act in ACTIVE_ACTIONS}

        f_dict = extract_step_pair_features(nodeL, nodeR, q_ctx, step_num=step_num)
        w_ids = []
        w_vals = []
        for k, v in f_dict.items():
            if k in self.step_word_to_id:
                w_ids.append(self.step_word_to_id[k])
                w_vals.append(v)

        if w_ids:
            w_arr = np.array(w_ids, dtype=np.int32)
            v_arr = np.array(w_vals, dtype=np.float32)
            logits = self.step_b + (self.step_w[:, w_arr] @ v_arr)
        else:
            logits = self.step_b.copy()

        none_id = self.step_act_to_id.get("NONE", -1)
        none_logit = float(logits[none_id]) * 0.35 if none_id >= 0 else 0.0

        scores = {}
        for act in ACTIVE_ACTIONS:
            act_id = self.step_act_to_id[act]
            scores[act] = float(logits[act_id]) - none_logit
        return scores

    def beam_stack_search(self, masked_question, values):
        """
        ビーム幅 self.beam_width で1手ずつ変数を結合・縮約するスタック型段階探索を実行し、
        発見された各数式 expr の (段階探索スコア, 計算値) の辞書を返す。
        """
        var_nodes, q_ctx = extract_rich_question_info(masked_question, values)
        sorted_vars = sorted(var_nodes.keys(), key=lambda v: var_nodes[v]["var_order"])
        initial_board = [var_nodes[v] for v in sorted_vars]

        if len(initial_board) < 2:
            return {}

        # ビームの状態: (累計スコア, 現在の盤面ノードリスト)
        beam = [(0.0, initial_board)]
        discovered_formulas = {}  # expr -> (best_beam_score, final_val)

        max_steps = len(initial_board) - 1
        for step_idx in range(1, max_steps + 1):
            next_candidates = []
            seen_states = set()

            for cum_score, board in beam:
                m = len(board)
                if m < 2:
                    continue

                for i in range(m):
                    for j in range(i + 1, m):
                        n1, n2 = board[i], board[j]
                        # 1手目は出現順、2手目以降は直前に合成された中間ノードを左(L)に配置して評価
                        if step_idx >= 2 and n2["depth"] > n1["depth"]:
                            nodeL, nodeR = n2, n1
                        elif step_idx >= 2 and n1["depth"] > n2["depth"]:
                            nodeL, nodeR = n1, n2
                        elif n1["var_order"] <= n2["var_order"]:
                            nodeL, nodeR = n1, n2
                        else:
                            nodeL, nodeR = n2, n1

                        op_scores = self.score_step_pair(nodeL, nodeR, q_ctx, step_num=step_idx)
                        rem_nodes = [board[k] for k in range(m) if k != i and k != j]

                        for act in ACTIVE_ACTIONS:
                            res_out = apply_step_operation(nodeL, nodeR, act, board, q_ctx)
                            if res_out is None:
                                continue
                            new_node, dim_bonus = res_out
                            step_score = cum_score + op_scores[act] + dim_bonus

                            # 完成した部分式が質問のターゲット単位に合致し、レート分母が解消されていればゴール加点
                            goal_bonus = 0.0
                            if new_node["q_match"]:
                                goal_bonus += 0.8
                            if not new_node["rate_units"] and not q_ctx["q_asks_rate"]:
                                goal_bonus += 0.6
                            elif new_node["rate_units"] and not q_ctx["q_asks_rate"]:
                                goal_bonus -= 1.2

                            # 盤面にまだ同じレート単位の変数やターゲット一致変数が未使用で残っている場合の調整
                            unused_q_match = sum(1 for rn in rem_nodes if rn["q_match"])
                            final_expr_score = (step_score / step_idx) + goal_bonus - (0.5 * unused_q_match)

                            expr_str = new_node["expr"]
                            if (
                                expr_str not in discovered_formulas
                                or final_expr_score > discovered_formulas[expr_str][0]
                            ):
                                discovered_formulas[expr_str] = (final_expr_score, new_node["val"])

                            if rem_nodes:
                                new_board = [new_node] + rem_nodes
                                state_sig = tuple(sorted(n["expr"] for n in new_board))
                                if state_sig not in seen_states:
                                    seen_states.add(state_sig)
                                    next_candidates.append((step_score, new_board))

            if not next_candidates:
                break
            next_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = next_candidates[: self.beam_width]

        return discovered_formulas

    def predict(self, masked_question, values):
        time_tokens, same_u, diff_u, q_match, _ = analyze_question_structure(masked_question)
        feat_dict = time_tokens_to_feature_dict(time_tokens)
        num_vars = len(values)

        active_word_ids = [self.word_to_id[w] for w in feat_dict.keys() if w in self.word_to_id]
        active_w_arr = np.array(active_word_ids, dtype=np.int32)

        # 1. スタック型ビームサーチ（ビーム幅 = self.beam_width）による段階探索を実行
        beam_results = self.beam_stack_search(masked_question, values)

        best_eq = None
        best_score = float("-inf")
        best_ans = None
        W = self.num_words

        # 2. 学習済み数式候補の大局的NNUEスコア + 次元解析ボーナス + ビーム探索スコアの統合評価
        for f_id, eq in enumerate(self.equations):
            if max_variable_index(eq) > num_vars:
                continue

            ok, ast_bonus, final_val = check_ast_and_dimensional_bonus(
                eq, values, same_u, diff_u, q_match
            )
            if not ok:
                continue

            if active_w_arr.size > 0:
                pair_ids = f_id * W + active_w_arr
                z1 = self.weight1[pair_ids].sum(axis=0) + self.b1[f_id]
                base_score = float(self.base_b[f_id] + self.base_w[pair_ids].sum())
            else:
                z1 = self.b1[f_id]
                base_score = float(self.base_b[f_id])

            a1 = np.clip(z1, 0.0, 1.0)
            z2 = a1 @ self.W2 + self.b2
            a2 = np.clip(z2, 0.0, 1.0)
            nnue_score = float(a2 @ self.W3)

            beam_bonus = 0.0
            if eq in beam_results:
                beam_bonus = BEAM_SCORE_WEIGHT * beam_results[eq][0]

            total_score = base_score + nnue_score + ast_bonus + beam_bonus

            if total_score > best_score:
                best_score = total_score
                best_eq = eq
                best_ans = final_val

        # 3. 学習データに存在しなかった未知の複合式がビームサーチで高スコア発見された場合のフォールバック評価
        for expr_str, (b_score, b_val) in beam_results.items():
            if expr_str in self.eq_to_id:
                continue
            ok, ast_bonus, final_val = check_ast_and_dimensional_bonus(
                expr_str, values, same_u, diff_u, q_match
            )
            if not ok:
                continue
            novel_total = b_score + ast_bonus
            if novel_total > best_score:
                best_score = novel_total
                best_eq = expr_str
                best_ans = final_val

        return best_eq, best_score, best_ans


# ==========================================
# 8. データ読み込み関数（初回学習・評価用）
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
# 9. 外部ファイルから import して使うための公開API
# ==========================================
_GLOBAL_MODEL = MathNNUEModel()


def ensure_model_loaded():
    if _GLOBAL_MODEL.is_loaded:
        return
    if _GLOBAL_MODEL.load(WEIGHTS_FILE):
        return
    print("ビームサーチ+NNUEモデルの重みファイルを初回学習して保存します...")
    download_files()
    train_data = load_train_data(TRAIN_FILE)
    _GLOBAL_MODEL.fit_and_save(train_data, WEIGHTS_FILE)


def solve(question_text, check_is_math=False):
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
# 10. 単体実行時の動作確認・精度評価
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
        evaluate_dataset(_GLOBAL_MODEL, train_data, "ビームサーチ+HalfKP型NNUE - 学習データ")
        evaluate_dataset(_GLOBAL_MODEL, test_data,  "ビームサーチ+HalfKP型NNUE - SVAMPテストデータ")

    sample_math = (
        "Each pack of dvds costs 76 dollars. "
        "If there is a discount of 25 dollars on each pack, "
        "how much do you have to pay to buy 5 packs?"
    )
    sample_quiz = (
        "is an online knowledge engine developed by Research , released in 2009 . [ 1 ] It is offered as an online service that answers queries by computing answers from externally sourced data . [ 2 ][ 3 ]"
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
