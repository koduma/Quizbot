import ast
import csv
import gc
import json
import math
import os
import re
import urllib.request
from collections import Counter, defaultdict
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
GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)

# 追加の外部算数文章題データセットURL (MultiArith と GSM8K の両方を使用)
EXTRA_DATASET_URLS = [
    (
        "MultiArith.json",
        "https://raw.githubusercontent.com/wangxr14/Algebraic-Word-Problem-Solver/master/data/MultiArith.json",
        "multiarith_json",
        "MultiArith",
    ),
    (
        "gsm8k_test.jsonl",
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "gsm8k_jsonl",
        "GSM8K",
    ),
]

TRAIN_FILE = os.path.join(BASE_DIR, "mawps_asdiv.csv")
GSM8K_TRAIN_FILE = os.path.join(BASE_DIR, "gsm8k_train.jsonl")
TEST_FILE = os.path.join(BASE_DIR, "SVAMP.json")
WEIGHTS_FILE = os.path.join(BASE_DIR, "math_lr_weights11.json")

# 文末のピリオド (例: "while her sister had 42.") の直前の数値も確実に抽出する正規表現
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w]|\.\d)")
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

# NNUE & ビームサーチ（スタック型段階探索）・省メモリ制限設定
HIDDEN1_SIZE = 16
HIDDEN2_SIZE = 16
TOP_K_CANDIDATES = 8
NNUE_EPOCHS = 18
NNUE_LR = 0.015
NNUE_L2_DECAY = 1e-4

# L-BFGS-B のメモリ爆発 (12GB ArrayMemoryError) を防ぐための大局テンプレート・語彙上限
MAX_GLOBAL_FORMULAS = 400    # 大局NNUEモデルで保持する頻出数式テンプレート数の上限
MAX_GLOBAL_WORDS = 30000     # 大局NNUEモデルで保持する有効語彙数の上限
MIN_WORD_DF = 2              # 大局BoW特徴の最小出現文書数

BEAM_WIDTH = 6               # 各ステップで残す部分盤面のビーム幅
STEP_ACTIONS = ("ADD", "SUB_LR", "SUB_RL", "MUL", "DIV_LR", "DIV_RL", "NONE")
ACTIVE_ACTIONS = ("ADD", "SUB_LR", "SUB_RL", "MUL", "DIV_LR", "DIV_RL")
BEAM_SCORE_WEIGHT = 0.80     # ビームサーチの段階評価スコアの統合重み

TENS_MAP = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
ONES_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
CARDINAL_WORD_TO_NUM = {
    "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90",
}

WORD_TO_NUM = {
    "one": "1",
    **CARDINAL_WORD_TO_NUM,
    "hundred": "100",
    "twice": "2 times",
    "double": "2 times",
    "triple": "3 times",
}


def download_if_needed(url, path, minimum_size):
    if not os.path.exists(path) or os.path.getsize(path) < minimum_size:
        print(f"Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)


def download_files():
    download_if_needed(TRAIN_CSV_URL, TRAIN_FILE, minimum_size=10_000)
    download_if_needed(TEST_JSON_URL, TEST_FILE, minimum_size=50_000)
    try:
        download_if_needed(GSM8K_TRAIN_URL, GSM8K_TRAIN_FILE, minimum_size=100_000)
    except Exception:
        pass


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


def is_self_canceling_or_invalid_duplicate(expression, num_vars_in_question=None):
    """
    1. ((var1 + var2) - var2) や ((var1 * var2) / var2) のように同じ変数を足して引く・掛けて割る自己相殺式を除外する。
    2. (var1 + (var1 * var2)) や (var1 + (var1 / var2)) のような「基準量 + 基準量の倍数/端数」は有効とする。
       ただし問題文中にまだ未使用の変数があるのに変数を重複使用している式は除外する。
    """
    occ = VAR_RE.findall(expression)
    unique_occ = set(occ)
    if len(occ) == len(unique_occ):
        return False

    if num_vars_in_question is not None and len(unique_occ) < num_vars_in_question:
        return True

    test_primes = [13.37, 29.71, 41.19, 53.83, 67.29, 79.43, 89.61, 97.17]
    max_idx = max((int(x) for x in unique_occ), default=1)
    test_vals = {f"var{i + 1}": test_primes[i % len(test_primes)] + i * 100.0 for i in range(max_idx)}
    try:
        val = float(eval(expression, {"__builtins__": {}}, test_vals))
    except Exception:
        return True

    if abs(val) < 1e-6 or abs(val - 1.0) < 1e-6:
        return True
    for v_val in test_vals.values():
        if math.isclose(val, v_val, abs_tol=1e-5):
            return True

    try:
        tree = ast.parse(expression, mode="eval")
    except Exception:
        return True

    invalid_dup = False

    def check_node(node):
        nonlocal invalid_dup
        if isinstance(node, ast.Expression):
            return check_node(node.body)
        if isinstance(node, ast.Name):
            return {node.id}, False
        if isinstance(node, ast.Constant):
            return set(), False
        if isinstance(node, ast.UnaryOp):
            return check_node(node.operand)
        if isinstance(node, ast.BinOp):
            l_vars, l_has_scale = check_node(node.left)
            r_vars, r_has_scale = check_node(node.right)
            overlap = l_vars & r_vars
            is_scale = isinstance(node.op, (ast.Mult, ast.Div))
            if overlap:
                if isinstance(node.op, (ast.Add, ast.Sub)) and len(overlap) == 1:
                    if (isinstance(node.left, ast.Name) and r_has_scale) or (
                        isinstance(node.right, ast.Name) and l_has_scale
                    ):
                        return l_vars | r_vars, True
                invalid_dup = True
            return l_vars | r_vars, (is_scale or l_has_scale or r_has_scale)
        return set(), False

    check_node(tree)
    return invalid_dup


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
    """
    桁区切りカンマ、英語の数詞 (two〜ninety, twenty-five等)、倍数語 (twice, double, triple)、
    分数語 (half, a third, a quarter)、およびダース・ペア表現を数値に正規化する。
    """
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)

    s = re.sub(r"\bhalf\s+a\s+dozen\b", "6", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(a|one)\s+dozen\b", "12", s, flags=re.IGNORECASE)
    s = re.sub(r"\btwo\s+dozen\b", "24", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthree\s+dozen\b", "36", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(a|one)\s+pair\s+of\b", "2", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(a|one)\s+couple\s+(?:of\s+)?", "2 ", s, flags=re.IGNORECASE)

    def repl_compound(m):
        t_w = m.group(1).lower()
        o_w = m.group(2).lower()
        return str(TENS_MAP[t_w] + ONES_MAP[o_w])

    s = re.sub(
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s]+(one|two|three|four|five|six|seven|eight|nine)\b",
        repl_compound,
        s,
        flags=re.IGNORECASE,
    )

    s = re.sub(r"\b(twice|double)\b", "2 times", s, flags=re.IGNORECASE)
    s = re.sub(r"\btriple\b", "3 times", s, flags=re.IGNORECASE)

    def repl_cardinal(m):
        w = m.group(1).lower()
        return CARDINAL_WORD_TO_NUM.get(w, m.group(0))

    card_pat = r"\b(" + "|".join(CARDINAL_WORD_TO_NUM.keys()) + r")\b"
    s = re.sub(card_pat, repl_cardinal, s, flags=re.IGNORECASE)

    s = re.sub(r"\bhalf\b(?!\s+\d)", "half 2", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:a|one)[-\s]+third\b(?!\s+\d)", "third 3", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:a|one)[-\s]+(?:quarter|fourth)\b(?!\s+\d)", "quarter 4", s, flags=re.IGNORECASE)

    return s


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

DET_BOUNDARY_WORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "his", "her", "their", "my", "your", "our", "its",
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
TIME_UNITS = {"hour", "minute", "second", "day", "week", "month", "year", "morn", "afternoon", "night"}
DIST_UNITS = {"mile", "foot", "inch", "yard", "meter", "kilometer", "centimeter"}
WEIGHT_UNITS = {"pound", "ounce", "gram", "kilogram", "ton"}

NON_UNIT_WORDS = {
    "of", "in", "on", "at", "to", "for", "from", "with", "by", "into", "among", "between",
    "and", "or", "but", "if", "when", "while", "after", "before", "than", "as", "until",
    "each", "every", "per", "apiece", "all", "both", "more", "less", "fewer", "other", "another",
    "left", "remain", "remaining", "away", "back", "out", "up", "down", "off", "over",
    "now", "then", "still", "already", "only", "just", "also", "too", "together", "altogether",
    "total", "how", "many", "much", "long", "far", "old", "heavy",
    "first", "second", "third", "fourth", "quarter", "half", "next", "last", "same", "daily",
}

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

ELLIPSIS_BLOCK_PREV = {
    "into", "by", "page", "grade", "level", "step", "chapter", "number", "group", "team", "row",
    "half", "third", "quarter", "fourth",
}
ELLIPSIS_VERB_PREV = {
    "had", "have", "has", "buy", "get", "find", "make", "eat", "give", "take",
    "lose", "spend", "sell", "pick", "collect", "bake", "plant", "earn", "save",
    "receive", "received", "caught", "catch", "read", "drink", "with",
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


def extract_strict_units_with_ellipsis(raw_tokens):
    """
    raw_tokens から各変数の直後単位 (u1, u2) を抽出し、
    単位が省略されている変数 (例: "while her sister had 42." や "She eats 3 for breakfast ... with 4.")
    に対して先行・後続変数の単位を補完する。
    """
    var_strict_units = {}
    var_sent_id = {}
    var_order_list = []
    var_prev_lem = {}
    sent_id = 0

    for idx_r, tok in enumerate(raw_tokens):
        if tok in (".", "?", ";"):
            sent_id += 1
        if is_var_tok(tok):
            var_sent_id[tok] = sent_id
            var_order_list.append(tok)
            prev_w = raw_tokens[idx_r - 1] if idx_r > 0 else "NONE"
            var_prev_lem[tok] = simple_lemmatize(prev_w)

            u_list = []
            prev_was_of = False
            for offset in range(1, 6):
                if idx_r + offset >= len(raw_tokens):
                    break
                cand_raw = raw_tokens[idx_r + offset]
                if cand_raw == "of":
                    prev_was_of = True
                    continue
                if cand_raw in DET_BOUNDARY_WORDS:
                    if prev_was_of:
                        continue
                    break
                cand = simple_lemmatize(cand_raw)
                if cand in UNIT_STOP_WORDS or is_var_tok(cand):
                    break
                if cand not in NON_UNIT_WORDS and cand not in STOP_WORDS:
                    u_list.append(cand)
                    prev_was_of = False
                if len(u_list) >= 2:
                    break

            var_strict_units[tok] = (
                u_list[0] if len(u_list) >= 1 else "NONE",
                u_list[1] if len(u_list) >= 2 else "NONE",
            )

    # 単位省略 (Unit Ellipsis) の補完
    for idx_v, v in enumerate(var_order_list):
        if var_strict_units[v] == ("NONE", "NONE") and var_prev_lem.get(v) not in ELLIPSIS_BLOCK_PREV:
            for prev_v in reversed(var_order_list[:idx_v]):
                if var_sent_id[prev_v] == var_sent_id[v] and var_strict_units[prev_v] != ("NONE", "NONE"):
                    var_strict_units[v] = var_strict_units[prev_v]
                    break
            if var_strict_units[v] == ("NONE", "NONE"):
                for next_v in var_order_list[idx_v + 1 :]:
                    if var_sent_id[next_v] == var_sent_id[v] and var_strict_units[next_v] != ("NONE", "NONE"):
                        var_strict_units[v] = var_strict_units[next_v]
                        break
            if var_strict_units[v] == ("NONE", "NONE") and idx_v > 0:
                if var_prev_lem.get(v) in ELLIPSIS_VERB_PREV:
                    prev_v = var_order_list[idx_v - 1]
                    if var_strict_units[prev_v] != ("NONE", "NONE"):
                        var_strict_units[v] = var_strict_units[prev_v]

    return var_strict_units, var_sent_id


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
    intro_cues = set()
    seen_any_var = False
    for c in clauses:
        var_indices = [idx for idx, w in enumerate(c) if is_var_tok(w)]
        if not var_indices:
            if not seen_any_var:
                intro_cues |= (set(c) & (ADD_CUES | SUB_CUES))
        else:
            seen_any_var = True
            if len(var_indices) == 1:
                var_spans[c[var_indices[0]]] = c
            else:
                for k, v_idx in enumerate(var_indices):
                    start = 0 if k == 0 else min(v_idx, var_indices[k - 1] + 2)
                    end = len(c) if k == len(var_indices) - 1 else min(var_indices[k + 1], v_idx + 3)
                    var_spans[c[v_idx]] = c[start:end]

    if (intro_cues & {"combined", "together", "altogether", "total", "sum"}) and "var2" in var_spans:
        var_spans["var2"] = list(var_spans["var2"]) + list(intro_cues & {"combined", "together", "altogether", "total", "sum"})

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

    var_strict_units, _ = extract_strict_units_with_ellipsis(raw_tokens)

    var_units, var_time_buckets = {}, {}
    for i, w in enumerate(clean_words):
        if is_var_tok(w):
            var_time_buckets[w] = get_time_bucket(i, n_words)
            var_units[w] = var_strict_units.get(w, ("NONE", "NONE"))

    each_nouns = set()
    for i, w in enumerate(clean_words):
        if w in ("each", "every", "per", "apiece"):
            for off in range(1, 4):
                if i + off < n_words and not is_var_tok(clean_words[i + off]):
                    cand_e = clean_words[i + off]
                    if cand_e in UNIT_STOP_WORDS:
                        break
                    if cand_e not in NON_UNIT_WORDS:
                        each_nouns.add(cand_e)
            if i - 1 >= 0 and not is_var_tok(clean_words[i - 1]):
                each_nouns.add(clean_words[i - 1])

    q_rate_nouns = set()
    for i, w in enumerate(q_words):
        if w in ("each", "every", "per", "apiece", "one"):
            if i + 1 < len(q_words) and not is_var_tok(q_words[i + 1]):
                q_rate_nouns.add(q_words[i + 1])
        elif w == "daily":
            q_rate_nouns.add("day")

    asks_money = bool(q_words_set & {"pay", "spend", "cost", "price", "money", "dollar", "cent", "change", "earn", "save"}) or (
        "how" in q_words and "much" in q_words and not (q_words_set & WEIGHT_UNITS)
    )
    asks_time = (
        ("how" in q_words and "long" in q_words)
        or bool((q_words_set - q_rate_nouns) & TIME_UNITS)
    ) and not asks_money
    asks_dist = ("how" in q_words and "far" in q_words) or bool((q_words_set - q_rate_nouns) & DIST_UNITS)
    asks_weight = ("how" in q_words and "heavy" in q_words) or "weigh" in q_words_set or bool((q_words_set - q_rate_nouns) & WEIGHT_UNITS)

    q_var_units = {q_words[i + 1] for i, w in enumerate(q_words) if is_var_tok(w) and i + 1 < len(q_words)}
    q_target_nouns = {
        w for i, w in enumerate(q_words)
        if w not in NON_UNIT_WORDS and not is_var_tok(w) and w not in q_rate_nouns
        and not (w in q_var_units and "many" not in q_words[:i])
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

    var_strict_units, var_sent_id = extract_strict_units_with_ellipsis(raw_tokens)

    q_asks_rate = bool(q_words_set & {"each", "every", "per", "apiece", "one", "daily"})
    q_rate_nouns = set()
    for i, w in enumerate(q_words):
        if w in ("each", "every", "per", "apiece", "one"):
            for off in range(1, 4):
                if i + off < len(q_words) and not is_var_tok(q_words[i + off]):
                    if q_words[i + off] in UNIT_STOP_WORDS:
                        break
                    if q_words[i + off] not in NON_UNIT_WORDS:
                        q_rate_nouns.add(q_words[i + off])
        elif w == "daily":
            q_rate_nouns.add("day")

    if "day" in q_rate_nouns:
        q_rate_nouns.update({"morn", "afternoon", "night"})

    asks_money = bool(q_words_set & {"pay", "spend", "cost", "price", "money", "dollar", "cent", "change", "earn", "save"}) or (
        "how" in q_words and "much" in q_words and not (q_words_set & WEIGHT_UNITS)
    )
    asks_time = (
        ("how" in q_words and "long" in q_words)
        or bool((q_words_set - q_rate_nouns) & TIME_UNITS)
    ) and not asks_money
    asks_dist = ("how" in q_words and "far" in q_words) or bool((q_words_set - q_rate_nouns) & DIST_UNITS)
    asks_weight = ("how" in q_words and "heavy" in q_words) or "weigh" in q_words_set or bool((q_words_set - q_rate_nouns) & WEIGHT_UNITS)

    q_var_units = {q_words[i + 1] for i, w in enumerate(q_words) if is_var_tok(w) and i + 1 < len(q_words)}
    q_target_nouns = {
        w for i, w in enumerate(q_words)
        if w not in NON_UNIT_WORDS and not is_var_tok(w) and w not in q_rate_nouns
        and not (w in q_var_units and "many" not in q_words[:i])
    }

    each_nouns = set()
    for i, w in enumerate(clean_words):
        if w in ("each", "every", "per", "apiece"):
            for off in range(1, 4):
                if i + off < n_words and not is_var_tok(clean_words[i + off]):
                    cand_e = clean_words[i + off]
                    if cand_e in UNIT_STOP_WORDS:
                        break
                    if cand_e not in NON_UNIT_WORDS:
                        each_nouns.add(cand_e)
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
                        for off in range(1, 4):
                            if idx_s + off >= len(seq):
                                break
                            cand_r = seq[idx_s + off]
                            if cand_r in UNIT_STOP_WORDS or is_var_tok(cand_r):
                                break
                            if cand_r not in NON_UNIT_WORDS and cand_r not in units:
                                rate_units.add(cand_r)
                        if idx_s - 1 >= 0 and sw in ("each", "apiece"):
                            cand_r = seq[idx_s - 1]
                            if (
                                not is_var_tok(cand_r)
                                and cand_r not in NON_UNIT_WORDS
                                and cand_r not in units
                            ):
                                rate_units.add(cand_r)

            if q_rate_nouns and (rate_units & q_rate_nouns):
                rate_units = rate_units - q_rate_nouns

            prev1 = clean_words[i - 1] if i > 0 and clean_words[i - 1] in MATH_CORE_WORDS else "OBJ"
            prev2 = clean_words[i - 2] if i > 1 and clean_words[i - 2] in MATH_CORE_WORDS else "OBJ"

            cues_sub = set(span_set & SUB_CUES)
            cues_add = set(span_set & ADD_CUES)
            if "take" in cues_sub and ("take" in q_words_set or bool(q_words_set & {"total", "altogether", "combined", "all"})):
                cues_sub.discard("take")

            if (units & MONEY_UNITS) and rate_units and asks_money and ("sell" in cues_sub):
                cues_sub.discard("sell")

            if prev1 == "with" and "bake" in cues_add:
                cues_add.discard("bake")
                cues_sub.add("use")

            is_frac_div = prev1 in ("half", "third", "quarter", "fourth")
            is_mult_scale = ("time" in units) or (i + 1 < n_words and clean_words[i + 1] == "time")

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
                "cues_sub": cues_sub,
                "cues_add": cues_add,
                "has_each": bool(rate_units) or bool((span_set | full_c_set) & {"each", "every", "per", "apiece"}),
                "role_each_noun": bool(units & each_nouns),
                "in_q_clause": w in q_words_set,
                "q_match": is_q_match,
                "is_frac_div": is_frac_div,
                "is_mult_scale": is_mult_scale,
                "prev1": prev1,
                "prev2": prev2,
                "pos_idx": i,
                "var_order": v_num,
                "clause_id": var_clause_id.get(w, -1),
                "sent_id": var_sent_id.get(w, -1),
                "vars_used": frozenset([w]),
                "depth": 0,
                "last_op": "INIT",
            }

    remainder_positions = [
        idx for idx, w in enumerate(clean_words)
        if w in ("remainder", "remaining", "rest") and idx < q_idx
    ]
    if remainder_positions:
        rem_pos = remainder_positions[0]
        early_vars = [
            vn for vn in sorted(var_nodes.values(), key=lambda x: x["var_order"])
            if vn["pos_idx"] < rem_pos
        ]
        if len(early_vars) >= 2:
            base_u = early_vars[0]["units"]
            for ev in early_vars[1:]:
                if ev["units"] & base_u:
                    ev["cues_sub"].add("leave")
                    ev["cues_add"].clear()

    q_abstract = [
        clean_words[idx]
        for idx in range(q_idx, n_words)
        if is_var_tok(clean_words[idx]) or clean_words[idx] in MATH_CORE_WORDS
    ]

    wants_combine_all = bool(
        set(clean_words) & {"total", "altogether", "together", "combined", "all", "both"}
    )

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
        "wants_combine_all": wants_combine_all,
        "num_vars": len(var_nodes),
    }
    return var_nodes, q_ctx


def apply_step_operation(nodeL, nodeR, action, board_nodes, q_ctx):
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
        if op_L.get("is_frac_div") or op_R.get("is_frac_div") or op_L.get("is_mult_scale") or op_R.get("is_mult_scale"):
            return None
        new_units = uL | uR
        new_rate_units = rL | rR
        if same_unit:
            step_dim_bonus += 1.5
            if rL and rR and (rL & rR):
                step_dim_bonus += 2.2
            elif bool(rL) != bool(rR) and (op_L["last_op"] == "*" or op_R["last_op"] == "*"):
                step_dim_bonus -= 4.0

            if op_sym == "-":
                if op_R["cues_sub"] and not op_L["cues_sub"]:
                    step_dim_bonus += 2.4
                elif op_L["cues_sub"] and not op_R["cues_sub"] and op_L["depth"] == 0:
                    step_dim_bonus -= 2.5
                elif not op_L["cues_sub"] and not op_R["cues_sub"]:
                    if any(bool(n["cues_sub"]) and bool(n["units"] & new_units) for n in other_nodes):
                        step_dim_bonus -= 2.2
                    elif q_ctx.get("wants_combine_all"):
                        step_dim_bonus -= 1.8
            else:
                if "discount" in (op_L["cues_sub"] | op_R["cues_sub"]):
                    step_dim_bonus -= 3.0
                elif op_R["cues_sub"] and not op_R["cues_add"] and "leave" in op_R["cues_sub"]:
                    step_dim_bonus -= 2.5
                elif not op_L["cues_sub"] and not op_R["cues_sub"]:
                    if (
                        (op_L["cues_add"] or op_R["cues_add"])
                        or (op_L.get("sent_id", -1) >= 0 and op_L.get("sent_id") == op_R.get("sent_id"))
                        or any(bool(n["cues_sub"]) and bool(n["units"] & new_units) for n in other_nodes)
                        or q_ctx.get("wants_combine_all")
                    ):
                        step_dim_bonus += 1.6
        elif diff_unit:
            step_dim_bonus -= 4.5

    elif op_sym == "*":
        cancel_lr = bool(rL & uR)
        cancel_rl = bool(rR & uL)
        if cancel_lr and not cancel_rl:
            new_units = set(uL)
            new_rate_units = set()
            step_dim_bonus += 2.2
            if op_L["depth"] >= 1 or op_R["depth"] >= 1:
                step_dim_bonus += 2.5
            else:
                if any(bool(n["units"] & uL) and bool(n["rate_units"] & rL) for n in other_nodes):
                    if op_L["cues_sub"] or any(bool(n["cues_sub"]) for n in other_nodes if bool(n["units"] & uL)):
                        step_dim_bonus -= 4.0
                    else:
                        step_dim_bonus -= 1.5
                if any(bool(n["units"] & uR) and bool(n["cues_sub"] | n["cues_add"]) for n in other_nodes):
                    step_dim_bonus -= 3.5
        elif cancel_rl and not cancel_lr:
            new_units = set(uR)
            new_rate_units = set()
            step_dim_bonus += 2.2
            if op_R["depth"] >= 1 or op_L["depth"] >= 1:
                step_dim_bonus += 2.5
            else:
                if any(bool(n["units"] & uR) and bool(n["rate_units"] & rR) for n in other_nodes):
                    if op_R["cues_sub"] or any(bool(n["cues_sub"]) for n in other_nodes if bool(n["units"] & uR)):
                        step_dim_bonus -= 4.0
                    else:
                        step_dim_bonus -= 1.5
                if any(bool(n["units"] & uL) and bool(n["cues_sub"] | n["cues_add"]) for n in other_nodes):
                    step_dim_bonus -= 3.5
        elif op_R.get("is_mult_scale") and not op_L.get("is_mult_scale"):
            new_units = set(uL)
            new_rate_units = set(rL)
            step_dim_bonus += 2.2
        elif op_L.get("is_mult_scale") and not op_R.get("is_mult_scale"):
            new_units = set(uR)
            new_rate_units = set(rR)
            step_dim_bonus += 2.2
        else:
            new_units = uL if op_L["q_match"] else (uR if op_R["q_match"] else (uL | uR))
            new_rate_units = set()
            if same_unit and uL:
                step_dim_bonus -= 2.0

    else:  # "/"
        if abs(new_val - round(new_val)) > 1e-6:
            if abs(new_val * 4 - round(new_val * 4)) > 1e-5 and abs(new_val * 10 - round(new_val * 10)) > 1e-5:
                step_dim_bonus -= 2.5
            else:
                step_dim_bonus -= 0.5
        if op_R.get("is_frac_div") and not op_L.get("is_frac_div"):
            new_units = set(uL)
            new_rate_units = set(rL)
            step_dim_bonus += 2.6
        elif same_unit and rR:
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
    if is_self_canceling_or_invalid_duplicate(norm_expr):
        return None

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
        "is_frac_div": False,
        "is_mult_scale": False,
        "prev1": op_L["prev1"],
        "prev2": op_L["prev2"],
        "pos_idx": min(op_L["pos_idx"], op_R["pos_idx"]),
        "var_order": max(op_L["var_order"], op_R["var_order"]),
        "clause_id": op_L["clause_id"],
        "sent_id": op_L.get("sent_id", -1),
        "vars_used": op_L["vars_used"] | op_R["vars_used"],
        "depth": max(op_L["depth"], op_R["depth"]) + 1,
        "last_op": op_sym,
    }
    return new_node, step_dim_bonus


def extract_step_pair_features(nodeL, nodeR, q_ctx, step_num=1):
    feat = {}
    uL, uR = nodeL["units"], nodeR["units"]
    rL, rR = nodeL["rate_units"], nodeR["rate_units"]

    same_unit = bool(
        (uL & uR)
        or ((uL & MONEY_UNITS) and (uR & MONEY_UNITS))
        or ((uL & TIME_UNITS) and (uR & TIME_UNITS))
    )
    diff_unit = bool(uL and uR and not same_unit)

    feat[f"S{min(step_num, 3)}::BIAS"] = 1.0
    feat[f"NVARS::{min(q_ctx['num_vars'], 5)}"] = 1.0

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

    if nodeL["clause_id"] == nodeR["clause_id"] and nodeL["clause_id"] >= 0:
        feat["PAIR::SAME_CLAUSE"] = 1.0
    if nodeL.get("sent_id", -1) == nodeR.get("sent_id", -1) and nodeL.get("sent_id", -1) >= 0:
        feat["PAIR::SAME_SENT"] = 1.0

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

    feat[f"R::LAST_OP_{nodeR['last_op']}"] = 1.0
    feat[f"R::PREV1_{nodeR['prev1']}"] = 1.0
    feat[f"R::PREV2_{nodeR['prev2']}_{nodeR['prev1']}"] = 1.0
    if nodeR["has_each"]:
        feat["R::HAS_EACH"] = 1.0
    if nodeR["role_each_noun"]:
        feat["R::ROLE_EACH_NOUN"] = 1.0
    if nodeR.get("is_frac_div"):
        feat["R::IS_FRAC_DIV"] = 1.0
    if nodeR.get("is_mult_scale"):
        feat["R::IS_MULT_SCALE"] = 1.0
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

    if same_unit and nodeR["cues_sub"] and not nodeL["cues_sub"]:
        feat["COMBO::SAME_U_R_SUB"] = 1.0
    if same_unit and nodeL["cues_sub"] and not nodeR["cues_sub"]:
        feat["COMBO::SAME_U_L_SUB"] = 1.0
    if same_unit and (nodeR["cues_add"] or (nodeL.get("sent_id", -1) == nodeR.get("sent_id", -1) and not nodeL["cues_sub"] and not nodeR["cues_sub"])):
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
    任意の深さのAST (2変数〜6変数以上の多段式) をボトムアップに1手ずつシミュレートし、
    各ステップの盤面における正例アクションと負例(NONE)ペアをすべて抽出する。
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
    board = dict(var_nodes)

    for step_idx, (l_expr, op_s, r_expr, _res_expr) in enumerate(steps, start=1):
        if l_expr not in board or r_expr not in board or l_expr == r_expr:
            break

        n1, n2 = board[l_expr], board[r_expr]
        if step_idx >= 2 and n2["depth"] > n1["depth"]:
            target_L, target_R = n2, n1
            is_lr = False
        elif step_idx >= 2 and n1["depth"] > n2["depth"]:
            target_L, target_R = n1, n2
            is_lr = True
        elif n1["var_order"] <= n2["var_order"]:
            target_L, target_R = n1, n2
            is_lr = True
        else:
            target_L, target_R = n2, n1
            is_lr = False

        if op_s == "+":
            true_act = "ADD"
        elif op_s == "-":
            true_act = "SUB_LR" if is_lr else "SUB_RL"
        elif op_s == "*":
            true_act = "MUL"
        elif op_s == "/":
            true_act = "DIV_LR" if is_lr else "DIV_RL"
        else:
            break

        target_pair = (target_L["expr"], target_R["expr"])
        board_list = sorted(board.values(), key=lambda x: (-x["depth"], x["var_order"]))

        for i in range(len(board_list)):
            for j in range(i + 1, len(board_list)):
                b1, b2 = board_list[i], board_list[j]
                if step_idx >= 2 and b2["depth"] > b1["depth"]:
                    cand_L, cand_R = b2, b1
                elif step_idx >= 2 and b1["depth"] > b2["depth"]:
                    cand_L, cand_R = b1, b2
                elif b1["var_order"] <= b2["var_order"]:
                    cand_L, cand_R = b1, b2
                else:
                    cand_L, cand_R = b2, b1

                f_dict = extract_step_pair_features(cand_L, cand_R, q_ctx, step_num=step_idx)
                lbl = true_act if (cand_L["expr"], cand_R["expr"]) == target_pair else "NONE"
                samples.append((f_dict, lbl))

        res_out = apply_step_operation(target_L, target_R, true_act, board_list, q_ctx)
        if res_out is None:
            break
        new_node, _ = res_out
        later_exprs = {x for st in steps[step_idx:] for x in (st[0], st[2])}
        if l_expr not in later_exprs:
            board.pop(l_expr, None)
        if r_expr not in later_exprs:
            board.pop(r_expr, None)
        board[new_node["expr"]] = new_node

    return samples


# ==========================================
# 6. AST評価 ＆ 次元解析・非負・同単位未使用ペナルティチェック
# ==========================================
def evaluate_with_eval(expression, values):
    variables = {f"var{i + 1}": v for i, v in enumerate(values)}
    return float(eval(expression, {"__builtins__": {}}, variables))


def check_ast_and_dimensional_bonus(
    expression,
    values,
    same_unit_pairs,
    diff_unit_pairs,
    q_match_vars,
    var_nodes=None,
):
    if is_self_canceling_or_invalid_duplicate(expression, num_vars_in_question=len(values)):
        return False, -1e9, None

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
                if var_nodes is not None:
                    if (left_u in var_nodes and (var_nodes[left_u].get("is_frac_div") or var_nodes[left_u].get("is_mult_scale"))) or (
                        right_u in var_nodes and (var_nodes[right_u].get("is_frac_div") or var_nodes[right_u].get("is_mult_scale"))
                    ):
                        valid = False
                        return 0.0, None
                if left_u is not None and right_u is not None and left_u != right_u:
                    pair = tuple(sorted([left_u, right_u]))
                    if pair in same_unit_pairs:
                        bonus += 1.5
                        if var_nodes is not None and left_u in var_nodes and right_u in var_nodes:
                            nL, nR = var_nodes[left_u], var_nodes[right_u]
                            if isinstance(node.op, ast.Sub):
                                if nR["cues_sub"] and not nL["cues_sub"]:
                                    bonus += 1.8
                                elif nL["cues_sub"] and not nR["cues_sub"]:
                                    bonus -= 1.8
                            else:
                                if "discount" in (nL["cues_sub"] | nR["cues_sub"]):
                                    bonus -= 2.5
                                elif not nL["cues_sub"] and not nR["cues_sub"]:
                                    if nL["cues_add"] or nR["cues_add"] or (nL.get("sent_id", -1) >= 0 and nL.get("sent_id") == nR.get("sent_id")):
                                        bonus += 1.2
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
                if var_nodes is not None:
                    r_left = var_nodes[left_u]["rate_units"] if left_u in var_nodes else set()
                    u_left = var_nodes[left_u]["units"] if left_u in var_nodes else set()
                    r_right = var_nodes[right_u]["rate_units"] if right_u in var_nodes else set()
                    u_right = var_nodes[right_u]["units"] if right_u in var_nodes else set()
                    if (r_left & u_right) and left_u is not None:
                        bonus += 1.5
                        return left_val * right_val, left_u
                    if (r_right & u_left) and right_u is not None:
                        bonus += 1.5
                        return left_val * right_val, right_u
                    if right_u in var_nodes and var_nodes[right_u].get("is_mult_scale") and left_u is not None:
                        bonus += 1.4
                        return left_val * right_val, left_u
                    if left_u in var_nodes and var_nodes[left_u].get("is_mult_scale") and right_u is not None:
                        bonus += 1.4
                        return left_val * right_val, right_u
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
                if var_nodes is not None and right_u in var_nodes and var_nodes[right_u].get("is_frac_div"):
                    bonus += 1.8
                    return res, left_u
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

    for v_a, v_b in same_unit_pairs:
        if (v_a in used_vars) != (v_b in used_vars):
            v_used = v_a if v_a in used_vars else v_b
            v_unused = v_b if v_a in used_vars else v_a
            if var_nodes is not None and v_used in var_nodes and v_unused in var_nodes:
                nu, nun = var_nodes[v_used], var_nodes[v_unused]
                if nu["rate_units"] and nun["rate_units"] and (nu["rate_units"] & nun["rate_units"]):
                    bonus -= 5.5
                elif "discount" in (nu["cues_sub"] | nun["cues_sub"]):
                    bonus -= 4.5
                elif bool(nun["cues_sub"] | nun["cues_add"]):
                    bonus -= 3.8
                elif v_used in q_match_vars and v_unused in q_match_vars:
                    if nu.get("sent_id", -1) >= 0 and nu.get("sent_id") == nun.get("sent_id"):
                        bonus -= 3.5
                elif v_unused in q_match_vars and any(
                    (v_used, v_other) in diff_unit_pairs or (v_other, v_used) in diff_unit_pairs
                    for v_other in used_vars if v_other != v_used
                ):
                    bonus -= 2.0

    return True, bonus, final_val


# ==========================================
# 7. ビームサーチ（スタック型段階探索）＋ HalfKP型NNUE ハイブリッドモデル
# ==========================================
class MathNNUEModel:
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

        self.weight1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.base_w = None
        self.base_b = None

        self.step_actions = list(STEP_ACTIONS)
        self.step_act_to_id = {a: i for i, a in enumerate(self.step_actions)}
        self.step_words = []
        self.step_word_to_id = {}
        self.step_w = None
        self.step_b = None

        self.is_loaded = False

    def fit_and_save(self, dataset, weights_path=WEIGHTS_FILE):
        raw_X_dicts, raw_y_labels = [], []
        step_X_dicts, step_y_labels = [], []

        for idx_row, item in enumerate(dataset):
            question, eq, values, _ans = item[:4]
            source_tag = item[4] if len(item) >= 5 else "MAWPS"

            if is_self_canceling_or_invalid_duplicate(eq, num_vars_in_question=len(values)):
                continue

            # 1. 7クラス1手読み評価器 (step_clf) は MAWPS と GSM8K の全問題からステップを学習 (メモリ消費極小)
            for s_feat, s_lbl in extract_unified_step_samples(question, eq, values):
                step_X_dicts.append(s_feat)
                step_y_labels.append(s_lbl)

            base_time_tokens, var_tb = tokenize_by_time(question)
            raw_X_dicts.append(time_tokens_to_feature_dict(base_time_tokens))
            raw_y_labels.append((eq, source_tag))

            # 変数の入れ替え水増し (3! = 6倍) は MAWPS/ASDiv に対してのみ実行し、GSM8K長文でのクラス爆発を防ぐ
            if source_tag == "MAWPS":
                if len(values) == 2 and max_variable_index(eq) == 2:
                    sw_map = {"var1": "var2", "var2": "var1"}
                    sw_tokens, sw_eq = permute_time_tokens_and_eq(base_time_tokens, var_tb, eq, sw_map)
                    raw_X_dicts.append(time_tokens_to_feature_dict(sw_tokens))
                    raw_y_labels.append((sw_eq, source_tag))

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
                        raw_X_dicts.append(time_tokens_to_feature_dict(sh_tokens))
                        raw_y_labels.append((sh_eq, source_tag))

                elif len(values) == 3 and max_variable_index(eq) == 3:
                    vars_3 = ("var1", "var2", "var3")
                    for p in permutations(vars_3):
                        if p == vars_3:
                            continue
                        p_map = dict(zip(vars_3, p))
                        perm_tokens, perm_eq = permute_time_tokens_and_eq(base_time_tokens, var_tb, eq, p_map)
                        raw_X_dicts.append(time_tokens_to_feature_dict(perm_tokens))
                        raw_y_labels.append((perm_eq, source_tag))

                        p_q, p_eq2, p_vals = permute_question_vars(question, eq, values, p_map)
                        for s_feat, s_lbl in extract_unified_step_samples(p_q, p_eq2, p_vals):
                            step_X_dicts.append(s_feat)
                            step_y_labels.append(s_lbl)

        # ---------------------------------------------------------
        # 省メモリフィルタリング:
        # 大局NNUEテンプレート分類器 (clf) のクラス数 F <= MAX_GLOBAL_FORMULAS (280)、
        # 語彙数 W <= MAX_GLOBAL_WORDS (10000) に絞り、L-BFGS-B の12GBメモリ確保エラーを完全に防止する
        # （※ 希少な多段式はすべて 7クラスの step_clf + beam_stack_search 側が探索・構築する）
        # ---------------------------------------------------------
        eq_counts = Counter()
        mawps_eq_counts = Counter()
        for eq_str, src_t in raw_y_labels:
            eq_counts[eq_str] += 1
            if src_t == "MAWPS":
                mawps_eq_counts[eq_str] += 1

        # MAWPSの頻出テンプレートと、全体で5回以上出現する汎用テンプレートを優先採用 (最大 MAX_GLOBAL_FORMULAS 個)
        candidate_eqs = sorted(
            eq_counts.keys(),
            key=lambda e: (1 if mawps_eq_counts[e] >= 2 or eq_counts[e] >= 5 else 0, eq_counts[e]),
            reverse=True,
        )
        allowed_eqs = set(candidate_eqs[:MAX_GLOBAL_FORMULAS])

        filtered_X_dicts = []
        y_labels = []
        word_df = Counter()
        for f_dict, (eq_str, _src_t) in zip(raw_X_dicts, raw_y_labels):
            if eq_str in allowed_eqs:
                filtered_X_dicts.append(f_dict)
                y_labels.append(eq_str)
                for w_key in f_dict.keys():
                    word_df[w_key] += 1

        del raw_X_dicts, raw_y_labels

        valid_words_sorted = [
            w for w, cnt in word_df.most_common()
            if cnt >= MIN_WORD_DF
        ][:MAX_GLOBAL_WORDS]
        allowed_words = set(valid_words_sorted)

        X_dicts = [
            {k: v for k, v in fd.items() if k in allowed_words}
            for fd in filtered_X_dicts
        ]
        del filtered_X_dicts, word_df, allowed_words
        gc.collect()

        vectorizer = DictVectorizer(sparse=True)
        X_csr = vectorizer.fit_transform(X_dicts).astype(np.float32)
        del X_dicts
        gc.collect()

        clf = LogisticRegression(C=0.5, max_iter=400)
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

        del clf
        gc.collect()

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
            del base_logits_copy

        base_cand_scores = base_logits[np.arange(N)[:, None], cands]
        del base_logits, coef_mat
        gc.collect()

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
        del rep_word_ids, rep_f_ids, pair_indices, X_bin, X_csr
        gc.collect()

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
        del scaled_coef
        gc.collect()

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

        del M_active_csr, m_adam, v_adam, base_cand_scores, cands
        gc.collect()

        self.weight1 = np.zeros((F * W, H1), dtype=np.float32)
        self.weight1[active_pids] = w1_active

        # ---------------------------------------------------------
        # 2. ビームサーチ用 7クラス1手読み評価器 (step_clf) の学習
        # ---------------------------------------------------------
        A_step = len(self.step_actions)
        if step_X_dicts:
            step_vec = DictVectorizer(sparse=True)
            X_step_csr = step_vec.fit_transform(step_X_dicts).astype(np.float32)
            self.step_words = [str(w) for w in step_vec.get_feature_names_out()]
            self.step_word_to_id = {w: i for i, w in enumerate(self.step_words)}
            W_step = len(self.step_words)

            step_clf = LogisticRegression(C=0.6, max_iter=500)
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
            del X_step_csr, step_X_dicts, step_y_labels, step_clf
            gc.collect()
        else:
            self.step_words = []
            self.step_word_to_id = {}
            self.step_w = np.zeros((A_step, 0), dtype=np.float32)
            self.step_b = np.zeros(A_step, dtype=np.float32)

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
            "format_version": "beam_stack_nnue_v8",
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
            f"(大局数式数: {F}, 大局語彙数: {W}, ステップ特徴数: {len(self.step_words)})"
        )

    def load(self, weights_path=WEIGHTS_FILE):
        if not os.path.exists(weights_path):
            return False
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("format_version") != "beam_stack_nnue_v8":
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
        none_logit = float(logits[none_id]) * 0.25 if none_id >= 0 else 0.0

        scores = {}
        for act in ACTIVE_ACTIONS:
            act_id = self.step_act_to_id[act]
            scores[act] = float(logits[act_id]) - none_logit
        return scores

    def beam_stack_search(self, masked_question, values):
        var_nodes, q_ctx = extract_rich_question_info(masked_question, values)
        sorted_vars = sorted(var_nodes.keys(), key=lambda v: var_nodes[v]["var_order"])
        initial_board = [var_nodes[v] for v in sorted_vars]

        if len(initial_board) < 2:
            return {}, var_nodes

        beam = [(0.0, initial_board, False)]
        discovered_formulas = {}

        max_steps = len(initial_board)
        for step_idx in range(1, max_steps + 1):
            next_candidates = []
            seen_states = set()

            for cum_score, board, used_keep_base in beam:
                m = len(board)
                if m < 2:
                    continue

                for i in range(m):
                    for j in range(i + 1, m):
                        n1, n2 = board[i], board[j]
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

                            goal_bonus = 0.0
                            if new_node["q_match"]:
                                goal_bonus += 1.2
                            if not new_node["rate_units"] and not q_ctx["q_asks_rate"]:
                                goal_bonus += 0.8
                            elif new_node["rate_units"] and not q_ctx["q_asks_rate"]:
                                goal_bonus -= 1.8

                            unused_penalty = 0.0
                            for rn in rem_nodes:
                                if rn["q_match"]:
                                    unused_penalty += 1.5
                                if (rn["units"] & new_node["units"]) and (
                                    rn["rate_units"] or rn["cues_sub"] or rn["cues_add"]
                                ):
                                    unused_penalty += 3.8
                                elif rn["cues_sub"] or rn["cues_add"]:
                                    unused_penalty += 2.2

                            norm_div = 1.0 + 0.30 * (step_idx - 1)
                            final_expr_score = (step_score / norm_div) + goal_bonus - unused_penalty

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
                                    next_candidates.append((step_score, new_board, used_keep_base))

                            if (
                                not used_keep_base
                                and act in ("MUL", "DIV_LR")
                                and nodeL["depth"] == 0
                                and (
                                    nodeR.get("is_frac_div")
                                    or nodeR.get("is_mult_scale")
                                    or q_ctx.get("wants_combine_all")
                                )
                            ):
                                keep_board = [nodeL, new_node] + rem_nodes
                                keep_sig = tuple(sorted(n["expr"] for n in keep_board))
                                if keep_sig not in seen_states:
                                    seen_states.add(keep_sig)
                                    next_candidates.append((step_score - 0.2, keep_board, True))

            if not next_candidates:
                break
            next_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = next_candidates[: self.beam_width]

        return discovered_formulas, var_nodes

    def predict(self, masked_question, values):
        time_tokens, same_u, diff_u, q_match, _ = analyze_question_structure(masked_question)
        feat_dict = time_tokens_to_feature_dict(time_tokens)
        num_vars = len(values)

        active_word_ids = [self.word_to_id[w] for w in feat_dict.keys() if w in self.word_to_id]
        active_w_arr = np.array(active_word_ids, dtype=np.int32)

        beam_results, var_nodes = self.beam_stack_search(masked_question, values)

        best_eq = None
        best_score = float("-inf")
        best_ans = None
        W = self.num_words

        for f_id, eq in enumerate(self.equations):
            if max_variable_index(eq) > num_vars:
                continue

            ok, ast_bonus, final_val = check_ast_and_dimensional_bonus(
                eq, values, same_u, diff_u, q_match, var_nodes=var_nodes
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

        for expr_str, (b_score, _b_val) in beam_results.items():
            if expr_str in self.eq_to_id:
                continue
            ok, ast_bonus, final_val = check_ast_and_dimensional_bonus(
                expr_str, values, same_u, diff_u, q_match, var_nodes=var_nodes
            )
            if not ok:
                continue
            novel_total = (1.15 * b_score) + ast_bonus
            if novel_total > best_score:
                best_score = novel_total
                best_eq = expr_str
                best_ans = final_val

        return best_eq, best_score, best_ans


# ==========================================
# 8. データ読み込み関数 ＆ GSM8K <<expr=res>> 数式木自動抽出パーサ
# ==========================================
def parse_gsm8k_solution_to_equation(masked_q, values, sol_text, true_ans):
    """
    GSM8Kの解答テキスト内の <<expr=res>> アノテーション列をボトムアップにトレースし、
    values に対応する var1, var2, ... の正規化数式木を自動復元する。
    """
    calcs = re.findall(r"<<([^=<>]+)=([^=<>]+)>>", sol_text)
    if not calcs or len(values) < 2:
        return None

    env = [(v, f"var{i + 1}", frozenset([f"var{i + 1}"])) for i, v in enumerate(values)]
    num_tok_re = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")

    for expr_raw, res_raw in calcs:
        expr_clean = expr_raw.strip().replace(",", "")
        res_clean = res_raw.strip().replace(",", "")
        try:
            res_val = float(res_clean)
        except Exception:
            continue

        used_in_step = set()
        step_vars = set()
        failed = False

        def repl_num(m):
            nonlocal failed, step_vars
            s_num = m.group(0)
            try:
                v_num = float(s_num)
            except Exception:
                failed = True
                return s_num

            for idx_e in range(len(env) - 1, len(values) - 1, -1):
                if idx_e not in used_in_step and math.isclose(env[idx_e][0], v_num, abs_tol=1e-5):
                    used_in_step.add(idx_e)
                    step_vars |= env[idx_e][2]
                    return env[idx_e][1]

            already_used_vars = set()
            for idx_e in range(len(values), len(env)):
                already_used_vars |= env[idx_e][2]
            for idx_e in range(len(values)):
                v_name = f"var{idx_e + 1}"
                if (
                    idx_e not in used_in_step
                    and v_name not in already_used_vars
                    and v_name not in step_vars
                    and math.isclose(env[idx_e][0], v_num, abs_tol=1e-5)
                ):
                    used_in_step.add(idx_e)
                    step_vars.add(v_name)
                    return env[idx_e][1]

            for idx_e in range(len(values)):
                if idx_e not in used_in_step and math.isclose(env[idx_e][0], v_num, abs_tol=1e-5):
                    used_in_step.add(idx_e)
                    step_vars.add(env[idx_e][1])
                    return env[idx_e][1]

            failed = True
            return s_num

        mapped_expr = num_tok_re.sub(repl_num, expr_clean)
        if failed:
            continue
        try:
            norm_e = normalize_equation(f"({mapped_expr})", prefix=False)
            eval_v = evaluate_with_eval(norm_e, values)
            if math.isclose(eval_v, res_val, abs_tol=1e-3):
                env.append((res_val, norm_e, frozenset(step_vars)))
        except Exception:
            continue

    for val, expr_s, used_v in reversed(env[len(values) :]):
        if math.isclose(val, true_ans, abs_tol=1e-3) and len(used_v) >= 2:
            if not is_self_canceling_or_invalid_duplicate(expr_s, num_vars_in_question=len(values)):
                return expr_s
    return None


def load_train_data(csv_path, gsm8k_train_path=GSM8K_TRAIN_FILE):
    dataset = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                try:
                    question = row["Question"].strip()
                    values = [float(v) for v in row["Numbers"].split()]
                    answer = float(row["Answer"])
                    masked_q = number_to_var(question)
                    eq = normalize_equation(row["Equation"], prefix=True)
                    if math.isclose(evaluate_with_eval(eq, values), answer, abs_tol=1e-5):
                        dataset.append((masked_q, eq, values, answer, "MAWPS"))
                except Exception:
                    continue

    if gsm8k_train_path and os.path.exists(gsm8k_train_path):
        gsm_added = 0
        with open(gsm8k_train_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    q_text = item.get("question", "").strip()
                    ans_str = item.get("answer", "")
                    if "####" not in ans_str:
                        continue
                    sol_part, num_part = ans_str.split("####", 1)
                    true_ans = float(num_part.strip().replace(",", ""))
                    masked_q, values = mask_numbers_in_question(q_text)
                    if not (2 <= len(values) <= 6):
                        continue
                    eq = parse_gsm8k_solution_to_equation(masked_q, values, sol_part, true_ans)
                    if eq is not None and math.isclose(evaluate_with_eval(eq, values), true_ans, abs_tol=1e-4):
                        dataset.append((masked_q, eq, values, true_ans, "GSM8K"))
                        gsm_added += 1
                except Exception:
                    continue
        if gsm_added > 0:
            print(f"GSM8K学習データから {gsm_added} 問の多段数式木を抽出して統合しました。")

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


def download_and_load_extra_problems():
    """
    MultiArith と GSM8K の両方の外部問題集をダウンロードして読み込み、
    データセット名ごとの問題リスト辞書 {dataset_name: [(q_text, true_eq, true_ans, dataset_name), ...]} を返す。
    """
    datasets_by_name = {}
    for filename, url, fmt, ds_name in EXTRA_DATASET_URLS:
        path = os.path.join(BASE_DIR, filename)
        try:
            if not os.path.exists(path) or os.path.getsize(path) < 1_000:
                print(f"Downloading extra dataset: {filename} ...")
                urllib.request.urlretrieve(url, path)
            problems = []
            if fmt == "multiarith_json":
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    q_text = item.get("sQuestion", "").strip()
                    ans_list = item.get("lSolutions", [])
                    eq_list = item.get("lEquations", [])
                    if q_text and ans_list:
                        true_ans = float(ans_list[0])
                        true_eq = str(eq_list[0]) if eq_list else ""
                        problems.append((q_text, true_eq, true_ans, ds_name))
            elif fmt == "gsm8k_jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        q_text = item.get("question", "").strip()
                        ans_str = item.get("answer", "")
                        if "####" in ans_str:
                            sol_part, num_part = ans_str.split("####", 1)
                            num_clean = num_part.strip().replace(",", "")
                            true_ans = float(num_clean)
                            problems.append((q_text, sol_part.strip(), true_ans, ds_name))
            if problems:
                datasets_by_name[ds_name] = problems
        except Exception as e:
            print(f"Warning: {ds_name} ({filename}) の取得に失敗しました: {e}")
            continue

    return datasets_by_name


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
    train_data = load_train_data(TRAIN_FILE, GSM8K_TRAIN_FILE)
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
# 10. 単体実行時の動作確認・精度評価・MultiArith & GSM8K 両対応出力
# ==========================================
def evaluate_dataset(model, dataset, title):
    correct_eq = 0
    correct_ans = 0
    for item in dataset:
        q, true_eq, values, true_ans = item[:4]
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
        ma_train_data = load_train_data(TRAIN_FILE, gsm8k_train_path=None)
        test_data = load_test_data(TEST_FILE)
        evaluate_dataset(_GLOBAL_MODEL, ma_train_data, "ビームサーチ+HalfKP型NNUE - 学習データ(MAWPS/ASDiv)")
        evaluate_dataset(_GLOBAL_MODEL, test_data,     "ビームサーチ+HalfKP型NNUE - SVAMPテストデータ")

    sample_math = (
        "Each pack of dvds costs 76 dollars. "
        "If there is a discount of 25 dollars on each pack, "
        "how much do you have to pay to buy 5 packs?"
    )
    sample_quiz = (
        "is an online knowledge engine developed by Research , released in 2009 . [ 1 ] It is offered as an online service that answers queries by computing answers from externally sourced data . [ 2 ][ 3 ]"
    )
    sample_math3 = (
        "For Halloween Debby and her sister combined the candy they received. "
        "Debby had 32 pieces of candy while her sister had 42. "
        "If they ate 35 pieces the first night, how many pieces do they have left?"
    )
    sample_math5 = (
        "Janet’s ducks lay 16 eggs per day. "
        "She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
        "How much in dollars does she make every day at the farmers' market?"
    )

    print("=== solve(question_text) ＆ クイズ判定テスト ===")
    masked_q1, vals1 = mask_numbers_in_question(sample_math)
    pred_eq1, pred_score1, _ = _GLOBAL_MODEL.predict(masked_q1, vals1)
    print(f"問題1: {sample_math}")
    print(f"  -> 算数判定: {is_math_word_problem(sample_math)}")
    print(f"  -> 選択された数式: {pred_eq1} (score: {pred_score1:.2f})")
    print(f"  -> solve() の返り値: {solve(sample_math)}\n")

    print(f"問題2: {sample_quiz}")
    print(f"  -> 算数判定: {is_math_word_problem(sample_quiz)}")
    print(f"  -> solve(check_is_math=True) の返り値: {solve(sample_quiz, check_is_math=True)}\n")

    masked_q3, vals3 = mask_numbers_in_question(sample_math3)
    pred_eq3, pred_score3, _ = _GLOBAL_MODEL.predict(masked_q3, vals3)
    print(f"問題3: {sample_math3}")
    print(f"  -> 算数判定: {is_math_word_problem(sample_math3)}")
    print(f"  -> 選択された数式: {pred_eq3} (score: {pred_score3:.2f})")
    print(f"  -> solve() の返り値: {solve(sample_math3)} (正解数値: 39)\n")

    masked_q5, vals5 = mask_numbers_in_question(sample_math5)
    pred_eq5, pred_score5, _ = _GLOBAL_MODEL.predict(masked_q5, vals5)
    print(f"問題5 (GSM8K デバッグ対象): {sample_math5}")
    print(f"  -> 算数判定: {is_math_word_problem(sample_math5)}")
    print(f"  -> 選択された数式: {pred_eq5} (score: {pred_score5:.2f})")
    print(f"  -> solve() の返り値: {solve(sample_math5)} (正解数値: 18)\n")

    # MultiArith と GSM8K の両方をテストデータとして使って正解率と誤答例を表示
    extra_datasets = download_and_load_extra_problems()
    prob_idx = 6
    for ds_name in ("MultiArith", "GSM8K"):
        problems = extra_datasets.get(ds_name, [])
        if not problems:
            continue
        wrong_examples = []
        extra_correct = 0
        eval_count = 0
        for q_text, true_eq, true_ans, _ds in problems:
            pred_val = solve(q_text)
            eval_count += 1
            if pred_val is not None and math.isclose(float(pred_val), true_ans, abs_tol=1e-4):
                extra_correct += 1
            elif len(wrong_examples) < 3:
                mq, vals = mask_numbers_in_question(q_text)
                p_eq, p_sc, _ = _GLOBAL_MODEL.predict(mq, vals) if vals else (None, 0.0, None)
                wrong_examples.append((q_text, true_eq, true_ans, pred_val, p_eq, p_sc))

        print(f"[{ds_name} 外部問題集テスト: {extra_correct} / {eval_count} ({extra_correct / max(eval_count, 1) * 100:.2f}%)]")
        if wrong_examples:
            w_text, w_true_eq, w_true_ans, w_pred_ans, w_pred_eq, w_score = wrong_examples[0]
            ans_disp = int(round(w_true_ans)) if math.isclose(w_true_ans, round(w_true_ans), abs_tol=1e-6) else w_true_ans
            print(f"問題{prob_idx} ({ds_name} より誤答例): {w_text}")
            print(f"  -> 算数判定: {is_math_word_problem(w_text)}")
            print(f"  -> 予測された数式: {w_pred_eq} (score: {w_score:.2f})")
            print(f"  -> solve() の返り値 (誤答): {w_pred_ans}")
            print(f"  -> 本来の正解数値: {ans_disp} (正解式/解法: {w_true_eq})\n")
            prob_idx += 1


if __name__ == "__main__":
    main()
