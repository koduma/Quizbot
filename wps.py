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
from scipy import optimize, sparse

import warnings

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
WEIGHTS_FILE = os.path.join(BASE_DIR, "math_lr_weights19.json")

# 文末のピリオド (例: "while her sister had 42.") の直前の数値も確実に抽出する正規表現
NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w]|\.\d)")
VAR_RE = re.compile(r"\bvar(\d+)\b")
TOKEN_VAR_RE = re.compile(r"(?<![a-zA-Z0-9])var(\d+)(?![a-zA-Z0-9])")
FORMULA_TOKEN_RE = re.compile(r"var\d+|\d+(?:\.\d+)?|[()+\-*/]")
OPS = {"+", "-", "*", "/"}
EOF_TOKEN = "EOF"


def is_var_tok(w):
    """トークンが 'var1', 'var2' 等の数値マスク変数であるかを厳密に判定する ('varies' 等を除外)"""
    return len(w) > 3 and w.startswith("var") and w[3:].isdigit()


def is_num_const_tok(w):
    """トークンが数値定数トークンであるかを判定する"""
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", w))


def is_operand_tok(w):
    """数式トークンがオペランド (var1, var2 または数値定数) であるかを判定する"""
    return is_var_tok(w) or is_num_const_tok(w)


def is_pure_var_formula(expression):
    """数式内の全オペランドが var1..varN であり、生の数値リテラル定数を含まないことを判定する"""
    for tok in formula_to_tokens(expression):
        if tok not in OPS and tok not in ("(", ")", EOF_TOKEN) and not is_var_tok(tok):
            return False
    return True


# NNUE (BoW[next_word][word] / NNUE[id(next_word, word)][H]) ＋ 文順中間変数ステップ生成 (step_k = op(arg1, arg2)) 設定
HIDDEN1_SIZE = 16
HIDDEN2_SIZE = 16
TOP_K_CANDIDATES = 8
NNUE_EPOCHS = 18
NNUE_LR = 0.015
NNUE_L2_DECAY = 1e-4

MAX_GLOBAL_WORDS = 30000     # BoW[next_word][word] で保持する有効語彙数の上限
MIN_WORD_DF = 2              # BoW特徴の最小出現ステップ数
TOKEN_BEAM_WIDTH = 8         # next_word 自己回帰生成時のトークンビーム幅
MAX_FORMULA_TOKENS = 28      # 生成する数式トークン列の最大長

BEAM_WIDTH = 16              # 文順 Scratchpad ステップ探索で残す部分盤面のビーム幅
STEP_ACTIONS = ("ADD", "SUB_LR", "SUB_RL", "MUL", "DIV_LR", "DIV_RL", "NONE")
ACTIVE_ACTIONS = ("ADD", "SUB_LR", "SUB_RL", "MUL", "DIV_LR", "DIV_RL")
BEAM_SCORE_WEIGHT = 1.25     # 文順 Scratchpad ステップ評価スコアの統合重み
AR_SCORE_WEIGHT = 2.25       # 自己回帰 BoW+NNUE ステップ対数確率の統合重み

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
    "quadruple": "4 times",
}


def _dicts_to_csr(feature_dicts):
    """DictVectorizer と同等の疎行列変換 (sklearn がある場合は DictVectorizer を使用)"""
    if HAS_SKLEARN:
        vec = DictVectorizer(sparse=True)
        X_csr = vec.fit_transform(feature_dicts).astype(np.float32)
        vocab = [str(w) for w in vec.get_feature_names_out()]
        return X_csr, vocab

    vocab_set = set()
    for fd in feature_dicts:
        vocab_set.update(fd.keys())
    vocab = sorted(str(w) for w in vocab_set)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    indptr = [0]
    indices = []
    data = []
    for fd in feature_dicts:
        for k, v in sorted(fd.items(), key=lambda x: word_to_idx[str(x[0])]):
            indices.append(word_to_idx[str(k)])
            data.append(float(v))
        indptr.append(len(indices))
    X_csr = sparse.csr_matrix(
        (np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
        shape=(len(feature_dicts), len(vocab)),
        dtype=np.float32,
    )
    return X_csr, vocab


def _fit_multinomial_logreg(X_csr, y_labels, C=0.6, max_iter=1500):
    """
    多クラスロジスティック回帰による重み学習。
    sklearn が存在する場合は LogisticRegression を使用し、無い場合は SciPy L-BFGS-B で最適化する。
    """
    if HAS_SKLEARN:
        clf = LogisticRegression(C=C, max_iter=max_iter, tol=1e-3, random_state=42)
        clf.fit(X_csr, y_labels)
        classes = [str(c) for c in clf.classes_]
        if len(classes) == 2 and clf.coef_.shape[0] == 1:
            coef_mat = np.vstack([-clf.coef_[0], clf.coef_[0]]).astype(np.float32)
            intercept_vec = np.array([-clf.intercept_[0], clf.intercept_[0]], dtype=np.float32)
        else:
            coef_mat = clf.coef_.astype(np.float32)
            intercept_vec = clf.intercept_.astype(np.float32)
        return classes, coef_mat, intercept_vec

    classes = sorted(set(str(y) for y in y_labels))
    cls_to_id = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_to_id[str(y)] for y in y_labels], dtype=np.int32)
    N, W = X_csr.shape
    M = len(classes)
    reg = 1.0 / max(C, 1e-6)

    Y_onehot = np.zeros((N, M), dtype=np.float64)
    Y_onehot[np.arange(N), y_idx] = 1.0

    def loss_and_grad(params):
        W_mat = params[: M * W].reshape(M, W)
        b_vec = params[M * W :]
        logits = (X_csr @ W_mat.T) + b_vec[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        loss = -np.sum(Y_onehot * np.log(np.clip(probs, 1e-12, 1.0))) + 0.5 * reg * np.sum(W_mat * W_mat)
        diff = probs - Y_onehot
        grad_W = (diff.T @ X_csr) + reg * W_mat
        grad_b = diff.sum(axis=0)
        return float(loss), np.concatenate([grad_W.ravel(), grad_b]).astype(np.float64)

    init_p = np.zeros(M * W + M, dtype=np.float64)
    res = optimize.minimize(
        loss_and_grad,
        init_p,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    opt_p = res.x
    coef_mat = opt_p[: M * W].reshape(M, W).astype(np.float32)
    intercept_vec = opt_p[M * W :].astype(np.float32)
    return classes, coef_mat, intercept_vec


def _compute_truncated_svd(mat, n_comp):
    """TruncatedSVD (sklearn が無い場合は numpy.linalg.svd で計算)"""
    if HAS_SKLEARN and min(mat.shape) > n_comp:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        U_mat = svd.fit_transform(mat).astype(np.float32)
        V_mat = svd.components_.astype(np.float32)
        return U_mat, V_mat
    u, s, vt = np.linalg.svd(mat.astype(np.float32), full_matrices=False)
    k = min(n_comp, len(s))
    return (u[:, :k] * s[:k]).astype(np.float32), vt[:k, :].astype(np.float32)


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
    "dozen", "pair", "pairs", "half", "halves", "twice", "double", "triple", "quadruple",
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
# 3. 数式の変換・正規化・Scratchpadステップ分解 (step_k = op(arg1, arg2))
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
    """
    数式ASTを標準形に変換する。
    ただし Scratchpad の中間変数再利用式 (例: step1 + (step1 + var3)) において、
    結合律フラット化を行うと同一部分式 step1 が隣り合って ((step1 + step1) + var3) に変形されてしまうため、
    フラット化後の項がすべて相異なる場合のみ結合律ソートを適用し、重複部分式がある場合は2項構造を保持する。
    """
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
                raw_terms = [to_str(n) for n in collect_assoc(node, ast.Add)]
                if len(raw_terms) == len(set(raw_terms)):
                    terms = sorted(raw_terms)
                    res = terms[0]
                    for t in terms[1:]:
                        res = f"({res} + {t})"
                    return res
                left_s, right_s = sorted([to_str(node.left), to_str(node.right)])
                return f"({left_s} + {right_s})"
            if isinstance(node.op, ast.Mult):
                raw_terms = [to_str(n) for n in collect_assoc(node, ast.Mult)]
                if len(raw_terms) == len(set(raw_terms)):
                    terms = sorted(raw_terms)
                    res = terms[0]
                    for t in terms[1:]:
                        res = f"({res} * {t})"
                    return res
                left_s, right_s = sorted([to_str(node.left), to_str(node.right)])
                return f"({left_s} * {right_s})"
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


def strip_outer_parens(expression):
    s = expression.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        matches_outer = True
        for idx, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx < len(s) - 1:
                    matches_outer = False
                    break
        if matches_outer and depth == 0:
            s = s[1:-1].strip()
        else:
            break
    return s


def formula_to_tokens(expression):
    s = strip_outer_parens(expression)
    tokens = FORMULA_TOKEN_RE.findall(s)
    return tokens + [EOF_TOKEN]


def tokens_to_formula(tokens):
    core = [t for t in tokens if t != EOF_TOKEN]
    if not core:
        return None
    raw_expr = " ".join(core)
    try:
        norm = normalize_equation(raw_expr, prefix=False)
        ast.parse(norm, mode="eval")
        return norm
    except Exception:
        return None


def _analyze_prefix_scope_stack(prefix_tokens):
    stack = [[0, []]]
    for tok in prefix_tokens:
        if not stack:
            return None
        top_state, _top_vars = stack[-1]
        if tok == "(":
            if top_state not in (0, 2):
                return None
            stack.append([0, []])
        elif is_operand_tok(tok):
            if top_state not in (0, 2):
                return None
            stack[-1][0] = top_state + 1
            stack[-1][1].append(tok)
        elif tok in OPS:
            if top_state != 1:
                return None
            stack[-1][0] = 2
        elif tok == ")":
            if len(stack) <= 1 or top_state != 3:
                return None
            closed_frame = stack.pop()
            if stack[-1][0] not in (0, 2):
                return None
            stack[-1][0] += 1
            stack[-1][1].extend(closed_frame[1])
        elif tok == EOF_TOKEN:
            if len(stack) != 1 or top_state != 3:
                return None
            stack[-1][0] = 4
        else:
            return None
    return stack


def get_valid_next_tokens(prefix_tokens, next_words_vocab, num_vars, target_k=None, allow_var_repeat=False):
    if len(prefix_tokens) >= MAX_FORMULA_TOKENS:
        return set()

    stack = _analyze_prefix_scope_stack(prefix_tokens)
    if stack is None or not stack:
        return set()

    depth = len(stack) - 1
    top_state, _ = stack[-1]

    if top_state == 4:
        return set()
    if top_state == 3:
        if depth > 0:
            return {")"} if ")" in next_words_vocab else set()
        return {EOF_TOKEN} if EOF_TOKEN in next_words_vocab else set()
    if top_state == 1:
        return {op for op in OPS if op in next_words_vocab}

    used_var_counts = Counter(t for t in prefix_tokens if is_var_tok(t))
    ops_placed = sum(1 for t in prefix_tokens if t in OPS)
    open_op_slots = sum(1 for frame in stack if frame[0] < 2)

    valid_vars = set()
    has_repeated = sum(max(0, c - 1) for c in used_var_counts.values())
    for nw in next_words_vocab:
        if is_var_tok(nw):
            v_idx = int(nw[3:])
            if 1 <= v_idx <= num_vars:
                c = used_var_counts.get(nw, 0)
                if c == 0 or (allow_var_repeat and has_repeated < 2 and c <= 2):
                    valid_vars.add(nw)

    if target_k is not None:
        target_ops = max(1, target_k - 1)
        ops_still_to_open = target_ops - (ops_placed + open_op_slots)
        valid = set()
        if ops_still_to_open > 0 and "(" in next_words_vocab and len(prefix_tokens) <= MAX_FORMULA_TOKENS - 4:
            if open_op_slots <= 1 and top_state == 2:
                return {"("}
            valid.add("(")
        if ops_still_to_open <= 0:
            return valid_vars
        valid |= valid_vars
        return valid

    pending_right_operands = sum(1 for frame in stack[:-1] if frame[0] == 0)
    max_extra_depth = max(0, num_vars - 1 - pending_right_operands)

    valid = set()
    if depth < max_extra_depth and len(prefix_tokens) <= MAX_FORMULA_TOKENS - 4:
        if "(" in next_words_vocab:
            valid.add("(")
    valid |= valid_vars
    return valid


def max_variable_index(expression):
    return max((int(idx) for idx in VAR_RE.findall(expression)), default=0)


def is_self_canceling_or_invalid_duplicate(expression, num_vars_in_question=None, allow_derived_dup=True):
    """
    1. ((var1 + var2) - var1) や ((var1 * var2) / var1) のように同じ変数を足して引く・掛けて割る自己相殺式を除外する。
    2. Scratchpad方式 (step1 = op(arg1, arg2)) で頻出する:
       - 利益計算: ((var1 + (var1 * var3)) - (var1 + var2))  [step3 - step1]
       - 3連鎖合算: (((var1 * var2) + var1) + ((var1 * var2) + var3)) [A + B + C]
       - 基準量 ± 派生量: (var1 + (var1 * var2)), (var1 - (var1 * var2))
       など、代数的に自己相殺しない再利用式は有効とする。
    """
    occ = VAR_RE.findall(expression)
    unique_occ = set(occ)
    if not unique_occ:
        return True
    if len(occ) == len(unique_occ):
        return False

    if not allow_derived_dup:
        return True
    if any(cnt > 3 for cnt in Counter(occ).values()):
        return True

    test_primes = [13.37, 29.71, 41.19, 53.83, 67.29, 79.43, 89.61, 97.17]
    max_idx = max((int(x) for x in unique_occ), default=1)
    test_vals = {f"var{i + 1}": test_primes[i % len(test_primes)] + i * 100.0 for i in range(max_idx)}

    try:
        tree = ast.parse(expression, mode="eval")
    except Exception:
        return True

    invalid_dup = False

    def get_direct_add_leaves(n):
        if isinstance(n, ast.Name):
            return [n.id]
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return get_direct_add_leaves(n.left) + get_direct_add_leaves(n.right)
        return []

    def check_node(node):
        nonlocal invalid_dup
        if isinstance(node, ast.Expression):
            return check_node(node.body)
        if isinstance(node, ast.Name):
            v = test_vals.get(node.id, 1.0)
            return {node.id}, v, [v]
        if isinstance(node, ast.Constant):
            v = float(node.value)
            return set(), v, [v]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            uv, val, sub_vals = check_node(node.operand)
            return uv, -val, sub_vals + [-val]
        if isinstance(node, ast.BinOp):
            l_vars, l_val, l_subs = check_node(node.left)
            r_vars, r_val, r_subs = check_node(node.right)
            if invalid_dup:
                return set(), 0.0, []

            if isinstance(node.op, ast.Add):
                cur_val = l_val + r_val
                # 同じ単一変数 leaf が加算木の中で複数回裸で足される (((step1 + var1) + var1) 等) のを防止
                add_leaves = get_direct_add_leaves(node)
                if len(add_leaves) > len(set(add_leaves)):
                    # ただし (var1 + (var1 + var2)) の2変数比較合算のみ例外として許可し、3回以上または他複合式がある場合は除外
                    if Counter(add_leaves).most_common(1)[0][1] > 2 or len(add_leaves) > 3:
                        invalid_dup = True
                        return set(), 0.0, []
            elif isinstance(node.op, ast.Sub):
                cur_val = l_val - r_val
            elif isinstance(node.op, ast.Mult):
                cur_val = l_val * r_val
            elif isinstance(node.op, ast.Div):
                if abs(r_val) < 1e-9:
                    invalid_dup = True
                    return set(), 0.0, []
                cur_val = l_val / r_val
            else:
                invalid_dup = True
                return set(), 0.0, []

            if abs(cur_val) < 1e-6 or abs(cur_val - 1.0) < 1e-6:
                invalid_dup = True
                return set(), 0.0, []
            for sv in l_subs + r_subs:
                if math.isclose(cur_val, sv, abs_tol=1e-5):
                    invalid_dup = True
                    return set(), 0.0, []

            overlap = l_vars & r_vars
            if overlap:
                if math.isclose(l_val, r_val, abs_tol=1e-5):
                    invalid_dup = True
                    return set(), 0.0, []
                if not isinstance(node.op, (ast.Add, ast.Sub)):
                    invalid_dup = True
                    return set(), 0.0, []
                if isinstance(node.op, ast.Sub) and isinstance(node.left, ast.Name) and isinstance(node.right, ast.BinOp) and isinstance(node.right.op, (ast.Add, ast.Sub)):
                    invalid_dup = True
                    return set(), 0.0, []

            all_subs = l_subs + r_subs + [cur_val]
            return l_vars | r_vars, cur_val, all_subs

        invalid_dup = True
        return set(), 0.0, []

    check_node(tree)
    return invalid_dup


def decompose_ast_to_steps(expression):
    """
    正規化された数式 (例: '((var1 + (var1 * var3)) - (var1 + var2))') のASTをボトムアップに分解し、
    共通部分式 (中間変数 step_k) の重複計算を排除した上で、
    1手ずつの (left_expr, op_str, right_expr, result_expr) のリストを返す。
    """
    tree = ast.parse(expression, mode="eval")
    steps = []
    seen_res = set()

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
            if res_expr not in seen_res:
                seen_res.add(res_expr)
                steps.append((l_expr, op_s, r_expr, res_expr))
            return res_expr
        raise ValueError("Unsupported node")

    visit(tree)
    return steps


def format_scratchpad_trace(expression, values):
    """
    数式 expression を中間変数ステップ列
      step1 = op(arg1, arg2) (= val1)
      step2 = op(arg3, arg4) (= val2)
    の形式で人間が読みやすいリストに変換する。
    """
    if not expression:
        return []
    try:
        steps = decompose_ast_to_steps(expression)
    except Exception:
        return []

    expr_to_step_name = {f"var{i + 1}": f"var{i + 1}" for i in range(len(values))}
    trace_lines = []
    for idx, (l_e, op_s, r_e, res_e) in enumerate(steps, start=1):
        s_name = f"step{idx}"
        l_disp = expr_to_step_name.get(l_e, l_e)
        r_disp = expr_to_step_name.get(r_e, r_e)
        expr_to_step_name[res_e] = s_name
        try:
            v = evaluate_with_eval(res_e, values)
            v_disp = int(round(v)) if math.isclose(v, round(v), abs_tol=1e-6) else round(v, 4)
            trace_lines.append(f"{s_name} = {l_disp} {op_s} {r_disp} (= {v_disp})")
        except Exception:
            trace_lines.append(f"{s_name} = {l_disp} {op_s} {r_disp}")
    return trace_lines


# ==========================================
# 4. 数値マスクと単語特徴トークン抽出
# ==========================================
def preprocess_implicit_numbers(text):
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    s_low = s.lower()

    # 直後に "One had <num> ... the other had <num>" が続く場合の "two" は内訳の親個数なので保護
    s = re.sub(
        r"\btwo\b(?=[^.?]*\.\s*one\s+(?:of\s+them\s+)?(?:had|has|was|contained|held)\s+\d)",
        "pair_group",
        s,
        flags=re.IGNORECASE,
    )

    s = re.sub(r"\bhalf\s+a\s+dozen\b", "6", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(a|one)\s+dozen\b", "12", s, flags=re.IGNORECASE)
    s = re.sub(r"\btwo\s+dozen\b", "24", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthree\s+dozen\b", "36", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(a|one)\s+pair\s+of\b", "2", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(a|one)\s+couple\s+(?:of\s+)?", "2 ", s, flags=re.IGNORECASE)

    if re.search(r"\b(day|days|daily)\b", s_low) and not re.search(r"\b7\b", s_low):
        s = re.sub(r"\b(?:in|for|per|every|each)\s+(?:a|one)\s+week\b", "for 7 days", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:a|one)\s+week\b", "7 days", s, flags=re.IGNORECASE)
    if re.search(r"\b(month|months|monthly)\b", s_low) and not re.search(r"\b12\b", s_low):
        s = re.sub(r"\b(?:in|for|per|every|each)\s+(?:a|one)\s+year\b", "for 12 months", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:a|one)\s+year\b", "12 months", s, flags=re.IGNORECASE)
    if re.search(r"\b(minute|minutes)\b", s_low) and not re.search(r"\b60\b", s_low):
        s = re.sub(r"\b(?:in|for|per|every|each)\s+(?:an|one)\s+hour\b", "for 60 minutes", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:an|one)\s+hour\b", "60 minutes", s, flags=re.IGNORECASE)

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
    s = re.sub(r"\bquadruple\b", "4 times", s, flags=re.IGNORECASE)

    s = re.sub(r"\bthree[-\s]+(?:quarters|fourths)\b", "0.75 percent", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:a|one)[-\s]+half\b(?!\s+\d)", "half 2", s, flags=re.IGNORECASE)
    s = re.sub(r"\bhalf\b(?!\s+\d)", "half 2", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:a|one)[-\s]+third\b(?!\s+\d)", "third 3", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(?:a|one)[-\s]+(?:quarter|fourth)\b(?!\s+\d)", "quarter 4", s, flags=re.IGNORECASE)

    def repl_cardinal(m):
        w = m.group(1).lower()
        return CARDINAL_WORD_TO_NUM.get(w, m.group(0))

    card_pat = r"\b(" + "|".join(CARDINAL_WORD_TO_NUM.keys()) + r")\b"
    s = re.sub(card_pat, repl_cardinal, s, flags=re.IGNORECASE)

    if not re.search(r"\bwhat\s+percent", s_low) and not re.search(r"\bhow\s+much\s+percent", s_low):
        def repl_percent(m):
            val = float(m.group(1)) / 100.0
            return f"{val:g} percent"
        s = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%", repl_percent, s)

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
    "gave": "give", "gives": "give", "given": "give", "giving": "give",
    "went": "go", "gone": "go", "goes": "go", "going": "go",
    "lost": "lose", "loses": "lose", "losing": "lose",
    "spent": "spend", "spends": "spend", "spending": "spend",
    "took": "take", "takes": "take", "taken": "take", "taking": "take",
    "told": "tell", "made": "make", "makes": "make", "making": "make",
    "ate": "eat", "eaten": "eat", "eating": "eat", "eats": "eat",
    "paid": "pay", "pays": "pay", "paying": "pay",
    "sold": "sell", "sells": "sell", "selling": "sell",
    "found": "find", "finds": "find", "finding": "find",
    "caught": "catch", "cut": "cut", "put": "put", "left": "left",
    "flew": "fly", "flies": "fly", "flying": "fly",
    "drank": "drink", "drinks": "drink", "drinking": "drink",
    "broke": "break", "broken": "break", "breaks": "break",
    "threw": "throw", "thrown": "throw", "throws": "throw",
    "kept": "keep", "keeps": "keep", "keeping": "keep",
    "got": "get", "gets": "get", "getting": "get", "gotten": "get",
    "came": "come", "comes": "come", "coming": "come",
    "ran": "run", "runs": "run", "running": "run",
    "drove": "drive", "drives": "drive", "driving": "drive",
    "wrote": "write", "written": "write", "writes": "write",
    "read": "read", "reads": "read", "reading": "read",
    "deleted": "delete", "deletes": "delete", "deleting": "delete",
    "baked": "bake", "bakes": "bake", "baking": "bake",
    "used": "use", "uses": "use", "using": "use",
    "shared": "share", "shares": "share", "sharing": "share",
    "divided": "divide", "divides": "divide", "dividing": "divide",
    "combined": "combined", "combines": "combined",
    "saved": "save", "saves": "save", "saving": "save",
    "received": "receive", "receives": "receive", "receiving": "receive",
    "earned": "earn", "earns": "earn", "earning": "earn",
    "planted": "plant", "plants": "plant", "planting": "plant",
    "collected": "collect", "collects": "collect", "collecting": "collect",
    "picked": "pick", "picks": "pick", "picking": "pick",
    "added": "add", "adds": "add", "adding": "add",
    "needed": "need", "needs": "need", "needing": "need",
    "wanted": "want", "wants": "want", "wanting": "want",
    "dropped": "drop", "drops": "drop", "dropping": "drop",
    "melted": "melt", "melts": "melt", "melting": "melt",
    "burned": "burn", "burnt": "burn", "burns": "burn",
    "missed": "miss", "misses": "miss", "missing": "miss",
    "ordered": "order", "orders": "order", "ordering": "order",
    "filled": "fill", "fills": "fill", "filling": "fill",
    "packed": "pack", "packs": "pack", "packing": "pack",
    "colored": "color", "colors": "color", "coloring": "coloring",
    "increased": "increase", "increases": "increase", "increasing": "increase",
    "decreased": "decrease", "decreases": "decrease", "decreasing": "decrease",
    "repaired": "repair", "repairs": "repair", "repairing": "repair",
    "children": "child", "men": "man", "women": "woman",
    "feet": "foot", "inches": "inch", "boxes": "box",
    "bags": "bag", "dollars": "dollar",
    "cents": "cent", "hours": "hour", "minutes": "minute",
    "days": "day", "weeks": "week", "months": "month",
    "years": "year", "times": "time", "halves": "half",
    "candies": "candy", "berries": "berry", "cookies": "cookie",
    "pennies": "penny", "loaves": "loaf", "leaves": "leaf",
    "shelves": "shelf", "knives": "knife", "wolves": "wolf",
    "lives": "life", "wives": "wife", "scarves": "scarf",
    "tomatoes": "tomato", "potatoes": "potato", "heroes": "hero",
}

ES_SUFFIXES = ("ches", "shes", "sses", "xes", "zes")
MONEY_UNITS = {"dollar", "cent", "money", "price", "cost", "penny", "dime", "quarter", "nickel", "buck", "yen", "profit", "value", "bill"}
TIME_UNITS = {"hour", "minute", "second", "day", "week", "month", "year", "morn", "afternoon", "night"}
DIST_UNITS = {"mile", "foot", "inch", "yard", "meter", "kilometer", "centimeter"}
WEIGHT_UNITS = {"pound", "ounce", "gram", "kilogram", "ton"}
HUMAN_UNITS = {
    "person", "people", "customer", "student", "child", "kid", "boy", "girl",
    "man", "woman", "friend", "guest", "passenger", "player", "worker",
    "member", "adult", "pupil", "teammate", "classmate", "visitor", "client",
}

ADJECTIVE_NON_UNITS = {
    "new", "old", "big", "small", "large", "little", "short", "long", "tall", "high",
    "low", "fresh", "raw", "clean", "dirty", "full", "empty", "extra", "additional",
    "good", "bad", "nice", "free", "busy", "leftover", "unused", "broken", "damaged",
    "rotten", "ripe", "unripe", "warm", "cold", "hot", "sweet", "sour", "heavy", "light",
    "fast", "slow", "early", "late", "young", "whole", "entire", "single", "double", "triple",
    "coloring",
}

NON_UNIT_WORDS = ADJECTIVE_NON_UNITS | {
    "of", "in", "on", "at", "to", "for", "from", "with", "by", "into", "among", "between",
    "and", "or", "but", "if", "when", "while", "after", "before", "than", "as", "until",
    "each", "every", "per", "apiece", "all", "both", "more", "less", "fewer", "other", "another",
    "left", "remain", "remaining", "away", "back", "out", "up", "down", "off", "over",
    "now", "then", "still", "already", "only", "just", "also", "too", "together", "altogether",
    "total", "how", "many", "much", "long", "far", "old", "heavy",
    "first", "second", "third", "fourth", "quarter", "half", "next", "last", "same", "daily",
    "time", "percent", "pair",
}

CLAUSE_BREAK_WORDS = {".", ",", "?", ";", "if", "when", "while", "after", "before", "but", "so", "how", "and"}
UNIT_STOP_WORDS = CLAUSE_BREAK_WORDS | {
    "on", "in", "at", "to", "for", "from", "with", "by", "into", "among", "between",
    "each", "every", "per", "apiece", "than", "as", "until", "or",
}

SUB_CUES = {
    "give", "lose", "spend", "eat", "take", "drop", "break", "burn", "melt",
    "throw", "delete", "miss", "sell", "use", "drink", "leave", "left", "quit",
    "fly", "away", "off", "out", "back", "discount", "change", "less", "fewer", "than",
    "difference", "remain", "remaining", "rest", "still", "need", "want",
    "already", "only", "extra", "end", "decrease", "repair",
}

ADD_CUES = {
    "add", "plus", "more", "total", "altogether", "together", "combined", "sum",
    "another", "additional", "new", "receive", "get", "find",
    "pick", "grow", "collect", "join", "come",
    "born", "plant", "bake", "earn", "save", "win", "increase",
}

DIV_CUES = {"split", "share", "divide", "equally", "equal", "into", "among", "between", "half", "third", "quarter", "fourth"}

MATH_CORE_WORDS = NON_UNIT_WORDS | SUB_CUES | ADD_CUES | DIV_CUES | MONEY_UNITS | TIME_UNITS | DIST_UNITS | WEIGHT_UNITS | HUMAN_UNITS | {
    "buy", "cost", "pay", "price", "group", "row", "box", "pack", "bag", "team", "page",
    "piece", "pair", "dozen", "twice", "double", "have", "had", "has",
    "make", "put", "hold", "fill", "fit", "order", "profit", "value", "bill",
}

ELLIPSIS_BLOCK_PREV = {
    "into", "by", "page", "grade", "level", "step", "chapter", "number", "group", "team", "row",
    "half", "third", "quarter", "fourth",
}
ELLIPSIS_VERB_PREV = {
    "had", "have", "has", "buy", "get", "find", "make", "eat", "give", "take",
    "lose", "spend", "sell", "pick", "collect", "bake", "plant", "earn", "save",
    "receive", "catch", "read", "drink", "with", "delete", "add", "drop", "throw",
    "burn", "melt", "use", "order", "color",
}

Q_VERB_BOUNDARY = {
    "do", "does", "did", "is", "are", "was", "were", "will", "would", "can", "could", "should",
    "have", "has", "had", "go", "went", "come", "came", "leave", "left", "remain", "cost",
    "take", "make", "buy", "sell", "eat", "get", "earn", "spend", "pay", "need", "want", "end",
}


def simple_lemmatize(word):
    if word in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ves"):
        return word[:-3] + "f"
    if len(word) > 4 and word.endswith(ES_SUFFIXES):
        return word[:-2]
    if len(word) > 4 and word.endswith("ed"):
        if word.endswith("ied"):
            return word[:-3] + "y"
        if len(word) > 5 and word[-3] == word[-4] and word[-3] not in "aeiou":
            return word[:-3]
        return word[:-2]
    if len(word) > 4 and word.endswith("ing"):
        if len(word) > 6 and word[-4] == word[-5] and word[-4] not in "aeiou":
            return word[:-4]
        return word[:-3]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def extract_target_nouns_from_q_words(q_words, q_rate_nouns):
    q_var_units = {q_words[i + 1] for i, w in enumerate(q_words) if is_var_tok(w) and i + 1 < len(q_words)}
    head_nouns = set()
    found_how = False
    for i, w in enumerate(q_words):
        if w in ("how", "what"):
            found_how = True
            continue
        if not found_how:
            continue
        if w in Q_VERB_BOUNDARY and head_nouns:
            break
        if (
            w not in NON_UNIT_WORDS
            and w not in STOP_WORDS
            and w not in Q_VERB_BOUNDARY
            and not is_var_tok(w)
            and w not in q_rate_nouns
            and not (w in q_var_units and "many" not in q_words[:i])
        ):
            head_nouns.add(w)

    if head_nouns:
        return head_nouns

    return {
        w for i, w in enumerate(q_words)
        if w not in NON_UNIT_WORDS
        and w not in Q_VERB_BOUNDARY
        and not is_var_tok(w)
        and w not in q_rate_nouns
        and not (w in q_var_units and "many" not in q_words[:i])
    }


def extract_strict_units_with_ellipsis(raw_tokens):
    var_strict_units = {}
    var_sent_id = {}
    var_order_list = []
    var_prev_lem = {}
    var_followed_by_of_them = {}
    sent_id = 0

    for idx_r, tok in enumerate(raw_tokens):
        if tok in (".", "?", ";"):
            sent_id += 1
        if is_var_tok(tok):
            var_sent_id[tok] = sent_id
            var_order_list.append(tok)
            prev_w = raw_tokens[idx_r - 1] if idx_r > 0 else "NONE"
            var_prev_lem[tok] = simple_lemmatize(prev_w)

            next1_raw = raw_tokens[idx_r + 1] if idx_r + 1 < len(raw_tokens) else "NONE"
            if simple_lemmatize(next1_raw) in ("time", "percent") or var_prev_lem[tok] in ("half", "third", "quarter", "fourth"):
                var_strict_units[tok] = ("NONE", "NONE")
                var_followed_by_of_them[tok] = False
                continue

            next2 = raw_tokens[idx_r + 1 : idx_r + 3]
            var_followed_by_of_them[tok] = (
                len(next2) >= 2 and next2[0] == "of" and next2[1] in ("them", "it", "those", "these", "the")
            )

            u_list = []
            prev_was_of = False
            for offset in range(1, 6):
                if idx_r + offset >= len(raw_tokens):
                    break
                cand_raw = raw_tokens[idx_r + offset]
                if cand_raw == "of":
                    prev_was_of = True
                    continue
                if cand_raw in DET_BOUNDARY_WORDS or cand_raw in ("them", "it"):
                    if prev_was_of:
                        continue
                    break
                cand = simple_lemmatize(cand_raw)
                if cand in UNIT_STOP_WORDS or is_var_tok(cand):
                    break
                if cand not in NON_UNIT_WORDS and cand not in STOP_WORDS and cand not in SUB_CUES and cand not in ADD_CUES and cand not in DIV_CUES:
                    u_list.append(cand)
                    prev_was_of = False
                if len(u_list) >= 2:
                    break

            var_strict_units[tok] = (
                u_list[0] if len(u_list) >= 1 else "NONE",
                u_list[1] if len(u_list) >= 2 else "NONE",
            )

    for idx_v, v in enumerate(var_order_list):
        if var_strict_units[v] == ("NONE", "NONE") and var_prev_lem.get(v) not in ELLIPSIS_BLOCK_PREV:
            if var_followed_by_of_them.get(v) and idx_v > 0:
                for prev_v in reversed(var_order_list[:idx_v]):
                    if var_strict_units[prev_v] != ("NONE", "NONE"):
                        var_strict_units[v] = var_strict_units[prev_v]
                        break
            if var_strict_units[v] == ("NONE", "NONE"):
                for prev_v in reversed(var_order_list[:idx_v]):
                    if var_sent_id[prev_v] == var_sent_id[v] and var_strict_units[prev_v] != ("NONE", "NONE"):
                        if var_prev_lem.get(v) in ELLIPSIS_VERB_PREV or var_prev_lem.get(v) in ("and", "or", "other", "another"):
                            var_strict_units[v] = var_strict_units[prev_v]
                            break
            if var_strict_units[v] == ("NONE", "NONE"):
                for next_v in var_order_list[idx_v + 1 :]:
                    if var_sent_id[next_v] == var_sent_id[v] and var_strict_units[next_v] != ("NONE", "NONE"):
                        if var_prev_lem.get(v) in ELLIPSIS_VERB_PREV:
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


def _are_units_compatible(u1_set, u2_set, q_target_nouns):
    if not u1_set or not u2_set:
        return False
    common = u1_set & u2_set
    if common:
        u1_only = u1_set - u2_set
        u2_only = u2_set - u1_set
        if (u1_only & q_target_nouns) and (u2_only) and not (u2_only & q_target_nouns):
            return False
        if (u2_only & q_target_nouns) and (u1_only) and not (u1_only & q_target_nouns):
            return False
        return True
    if (u1_set & MONEY_UNITS) and (u2_set & MONEY_UNITS):
        return True
    if (u1_set & TIME_UNITS) and (u2_set & TIME_UNITS):
        return True
    if (u1_set & HUMAN_UNITS) and (u2_set & HUMAN_UNITS):
        if not ((u1_set ^ u2_set) & q_target_nouns):
            return True
    return False


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

    var_units = {}
    for w in clean_words:
        if is_var_tok(w):
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

    asks_money = bool(q_words_set & {"pay", "spend", "cost", "price", "money", "dollar", "cent", "change", "earn", "save", "profit"}) or (
        "how" in q_words and "much" in q_words and not (q_words_set & WEIGHT_UNITS)
    )
    asks_time = (
        ("how" in q_words and "long" in q_words)
        or bool((q_words_set - q_rate_nouns) & TIME_UNITS)
    ) and not asks_money
    asks_dist = ("how" in q_words and "far" in q_words) or bool((q_words_set - q_rate_nouns) & DIST_UNITS)
    asks_weight = ("how" in q_words and "heavy" in q_words) or "weigh" in q_words_set or bool((q_words_set - q_rate_nouns) & WEIGHT_UNITS)
    asks_human = bool((q_words_set - q_rate_nouns) & HUMAN_UNITS)

    q_target_nouns = extract_target_nouns_from_q_words(q_words, q_rate_nouns)

    double_match_vars = {
        v for v, (u1, u2) in var_units.items()
        if u1 != "NONE" and u2 != "NONE" and u1 in q_target_nouns and u2 in q_target_nouns
    }

    q_match_vars = set()
    if double_match_vars:
        q_match_vars = double_match_vars
    else:
        for v, (u1, u2) in var_units.items():
            u_set = {u for u in (u1, u2) if u != "NONE"}
            if (
                bool(u_set & q_target_nouns)
                or (asks_money and bool(u_set & MONEY_UNITS))
                or (asks_time and bool(u_set & TIME_UNITS))
                or (asks_dist and bool(u_set & DIST_UNITS))
                or (asks_weight and bool(u_set & WEIGHT_UNITS))
                or (asks_human and bool(u_set & HUMAN_UNITS))
            ):
                q_match_vars.add(v)

    same_unit_pairs, diff_unit_pairs = set(), set()
    var_keys = sorted(var_units.keys())
    for i in range(len(var_keys)):
        for j in range(i + 1, len(var_keys)):
            v1, v2 = var_keys[i], var_keys[j]
            u1_set = {u for u in var_units[v1] if u != "NONE"}
            u2_set = {u for u in var_units[v2] if u != "NONE"}
            if _are_units_compatible(u1_set, u2_set, q_target_nouns):
                same_unit_pairs.add((v1, v2))
            else:
                diff_unit_pairs.add((v1, v2))

    question_words = set()

    for i, w in enumerate(clean_words):
        if is_var_tok(w):
            u1, u2 = var_units[w]
            prev1 = clean_words[i - 1] if i > 0 and clean_words[i - 1] in MATH_CORE_WORDS else "OBJ"
            prev2 = clean_words[i - 2] if i > 1 and clean_words[i - 2] in MATH_CORE_WORDS else "OBJ"
            question_words.add(f"PREV1_{w}_{prev1}")
            question_words.add(f"PREV2_{w}_{prev2}_{prev1}")

            span_set = set(var_spans.get(w, []))
            if span_set & {"each", "every", "per", "apiece"}:
                question_words.add(f"EACH_IN_SPAN_{w}")
            for sh in (span_set & SUB_CUES):
                question_words.add(f"SPAN_SUB_{w}_{sh}")
            if span_set & SUB_CUES:
                question_words.add(f"SPAN_ANY_SUB_{w}")
            for ah in (span_set & ADD_CUES):
                question_words.add(f"SPAN_ADD_{w}_{ah}")
            if span_set & ADD_CUES:
                question_words.add(f"SPAN_ANY_ADD_{w}")
            for dh in (span_set & DIV_CUES):
                question_words.add(f"SPAN_DIV_{w}_{dh}")
            if span_set & DIV_CUES:
                question_words.add(f"SPAN_ANY_DIV_{w}")

            question_words.add(f"Q_UNIT_MATCH_{w}" if w in q_match_vars else f"Q_UNIT_NOMATCH_{w}")
            if (u1 != "NONE" and u1 in each_nouns) or (u2 != "NONE" and u2 in each_nouns):
                question_words.add(f"ROLE_EACH_NOUN_{w}")
            if w in q_words_set:
                question_words.add(f"IN_Q_CLAUSE_{w}")

    for v1, v2 in same_unit_pairs:
        question_words.add(f"SAME_UNIT_EXACT_{v1}_{v2}")
    for v1, v2 in diff_unit_pairs:
        question_words.add(f"DIFF_UNIT_{v1}_{v2}")

    abstract_items = []
    for w in clean_words:
        if is_var_tok(w) or w in MATH_CORE_WORDS:
            abstract_items.append(w)
        elif not abstract_items or abstract_items[-1] != "OBJ":
            abstract_items.append("OBJ")

    for w in abstract_items:
        if w != "OBJ":
            question_words.add(w)
    for i in range(len(abstract_items) - 1):
        w1, w2 = abstract_items[i], abstract_items[i + 1]
        if w1 != "OBJ" or w2 != "OBJ":
            question_words.add(f"{w1}_{w2}")

    q_abstract = [
        clean_words[idx]
        for idx in range(q_idx, n_words)
        if is_var_tok(clean_words[idx]) or clean_words[idx] in MATH_CORE_WORDS
    ]
    for w in q_abstract:
        question_words.add(f"Q_{w}")
    for i in range(len(q_abstract) - 1):
        w1, w2 = q_abstract[i], q_abstract[i + 1]
        question_words.add(f"QBI_{w1}_{w2}")

    return question_words, same_unit_pairs, diff_unit_pairs, q_match_vars


def tokenize_question(text):
    question_words, _, _, _ = analyze_question_structure(text)
    return question_words


def permute_question_words_and_eq(question_words, eq, mapping):
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
    new_words = {map_tok(tok) for tok in question_words}
    return new_words, new_eq


def build_autoregressive_feature_dict(question_words, prefix_tokens, num_vars=None):
    feat = {w: 1.0 for w in question_words}

    step_idx = len(prefix_tokens)
    stack = _analyze_prefix_scope_stack(prefix_tokens)
    depth = (len(stack) - 1) if stack else 0
    top_state = stack[-1][0] if stack else 0
    top_scope_vars = stack[-1][1] if stack else []

    used_vars = []
    used_vars_set = set()
    last_var = None
    last_op = None

    for idx, tok in enumerate(prefix_tokens):
        feat[tok] = 1.0
        feat[f"HIST_TOK::{tok}"] = 1.0
        feat[f"HIST_POS_{min(idx, 12)}::{tok}"] = 1.0
        if is_var_tok(tok):
            if tok not in used_vars_set:
                used_vars.append(tok)
                used_vars_set.add(tok)
            last_var = tok
        elif tok in OPS:
            last_op = tok

    n_v = num_vars if num_vars is not None else max((int(v[3:]) for v in used_vars_set), default=2)
    rem_vars = max(0, n_v - len(used_vars_set))
    unused_vars_ordered = [f"var{i}" for i in range(1, n_v + 1) if f"var{i}" not in used_vars_set]
    first_unused = unused_vars_ordered[0] if unused_vars_ordered else None

    feat[f"HIST_STEP::{min(step_idx, 14)}"] = 1.0
    feat[f"HIST_DEPTH::{min(depth, 5)}"] = 1.0
    feat[f"HIST_STATE::{top_state}"] = 1.0
    feat[f"HIST_NUM_USED::{min(len(used_vars_set), 6)}"] = 1.0
    feat[f"HIST_NVARS::{min(n_v, 6)}"] = 1.0
    feat[f"HIST_REM_VARS::{min(rem_vars, 6)}"] = 1.0

    last1 = prefix_tokens[-1] if step_idx >= 1 else "BOS"
    last2 = f"{prefix_tokens[-2]}_{prefix_tokens[-1]}" if step_idx >= 2 else f"BOS_{last1}"
    last3 = (
        f"{prefix_tokens[-3]}_{prefix_tokens[-2]}_{prefix_tokens[-1]}"
        if step_idx >= 3
        else f"BOS_{last2}"
    )

    feat[f"HIST_LAST1::{last1}"] = 1.0
    feat[f"HIST_LAST1_DEPTH::{last1}_{min(depth, 4)}"] = 1.0
    feat[f"HIST_LAST2::{last2}"] = 1.0
    feat[f"HIST_LAST3::{last3}"] = 1.0

    for uv in used_vars_set:
        feat[f"HIST_USED::{uv}"] = 1.0
    for unv in unused_vars_ordered:
        feat[f"HIST_UNUSED::{unv}"] = 1.0

    if top_state == 3:
        if depth > 0:
            feat["PHASE::CLOSE_PAREN_NOW"] = 1.0
        else:
            feat["PHASE::EOF_NOW"] = 1.0

    elif top_state in (0, 2):
        feat[f"PHASE::SELECT_OPERAND_STATE_{top_state}"] = 1.0
        pending_right = sum(1 for frame in (stack[:-1] if stack else []) if frame[0] == 0)
        ops_still_needed = max(0, rem_vars - 1 - pending_right - depth)
        feat[f"OPEN_DECISION::NEEDED_DEPTH_{min(ops_still_needed, 4)}"] = 1.0
        if ops_still_needed > 0 and top_state == 0:
            feat["OPEN_DECISION::SHOULD_OPEN_PAREN"] = 1.0
        else:
            feat["OPEN_DECISION::NO_PAREN_NEEDED"] = 1.0

        if first_unused is not None:
            feat[f"NEXT_ORDERED_VAR::{first_unused}"] = 1.0
            if top_state == 2 and last_op is not None:
                feat[f"OP_NEXT_ORDERED::{last_op}_{first_unused}"] = 1.0

        if top_state == 2 and top_scope_vars:
            v_left = top_scope_vars[0]
            for unv in unused_vars_ordered:
                va, vb = sorted([v_left, unv])
                if f"SAME_UNIT_EXACT_{va}_{vb}" in question_words:
                    feat[f"CAND_SAME_UNIT_AS_LEFT::{unv}"] = 1.0
                    if last_op in ("+", "-"):
                        feat[f"CAND_ADD_SUB_MATCH::{unv}"] = 1.0
                if f"DIFF_UNIT_{va}_{vb}" in question_words:
                    feat[f"CAND_DIFF_UNIT_AS_LEFT::{unv}"] = 1.0
                    if last_op in ("*", "/") and (
                        f"EACH_IN_SPAN_{unv}" in question_words or f"EACH_IN_SPAN_{v_left}" in question_words
                    ):
                        feat[f"CAND_RATE_MATCH::{unv}"] = 1.0

    elif top_state == 1:
        feat["PHASE::SELECT_OPERATOR"] = 1.0
        v_left = top_scope_vars[0] if top_scope_vars else last_var

        v_right = first_unused
        if v_left is not None and len(unused_vars_ordered) >= 2:
            for cand_v in unused_vars_ordered:
                va, vb = sorted([v_left, cand_v])
                if f"SAME_UNIT_EXACT_{va}_{vb}" in question_words and (
                    f"SPAN_ANY_SUB_{cand_v}" in question_words or f"SPAN_ANY_ADD_{cand_v}" in question_words
                ):
                    v_right = cand_v
                    break

        is_final_op = (depth == 0 and rem_vars <= 1)
        if is_final_op:
            feat["OP_STAGE::FINAL"] = 1.0
            for qw in question_words:
                if qw.startswith("Q_") or qw.startswith("QBI_"):
                    feat[f"FINAL_OP::{qw}"] = 1.0
        else:
            feat["OP_STAGE::INNER"] = 1.0
            for qw in question_words:
                if qw.startswith("Q_") or qw.startswith("QBI_"):
                    feat[f"INNER_OP::{qw}"] = 1.0

        if v_left is not None:
            for qw in question_words:
                if v_left in qw:
                    norm_qw = qw.replace(v_left, "VL")
                    feat[f"OP_VL::{norm_qw}"] = 1.0
        if v_right is not None:
            for qw in question_words:
                if v_right in qw:
                    norm_qw = qw.replace(v_right, "VR")
                    feat[f"OP_VR::{norm_qw}"] = 1.0

        if v_left is not None and v_right is not None and v_left != v_right:
            va, vb = sorted([v_left, v_right])
            same_u = f"SAME_UNIT_EXACT_{va}_{vb}" in question_words
            diff_u = f"DIFF_UNIT_{va}_{vb}" in question_words
            if same_u:
                feat["OP_PAIR::SAME_UNIT"] = 1.0
                if f"SPAN_ANY_SUB_{v_right}" in question_words:
                    feat["OP_PAIR::SAME_UNIT_VR_SUB"] = 1.0
                elif f"SPAN_ANY_ADD_{v_right}" in question_words:
                    feat["OP_PAIR::SAME_UNIT_VR_ADD"] = 1.0
                elif not (f"SPAN_ANY_SUB_{v_left}" in question_words):
                    feat["OP_PAIR::SAME_UNIT_NO_SUB"] = 1.0
            if diff_u:
                feat["OP_PAIR::DIFF_UNIT"] = 1.0
                if f"EACH_IN_SPAN_{v_left}" in question_words or f"EACH_IN_SPAN_{v_right}" in question_words:
                    feat["OP_PAIR::DIFF_UNIT_WITH_EACH"] = 1.0
                if f"SPAN_ANY_DIV_{v_right}" in question_words or f"SPAN_ANY_DIV_{v_left}" in question_words:
                    feat["OP_PAIR::DIFF_UNIT_WITH_DIV"] = 1.0
                if f"SPAN_ANY_SUB_{v_right}" in question_words and not (
                    f"EACH_IN_SPAN_{v_left}" in question_words or f"EACH_IN_SPAN_{v_right}" in question_words
                ):
                    feat["OP_PAIR::DIFF_UNIT_1TO1_SUB"] = 1.0

    return feat


# ==========================================
# 5. 文順・中間変数ステップ生成 (step1 = op(arg1, arg2)) エンジン ＆ 局所文脈ポインタ
# ==========================================
ENTITY_STOP_WORDS = (
    NON_UNIT_WORDS
    | STOP_WORDS
    | MATH_CORE_WORDS
    | MONEY_UNITS
    | TIME_UNITS
    | DIST_UNITS
    | WEIGHT_UNITS
    | {"value", "amount", "number", "total", "rest", "remainder", "purpose", "try", "decide"}
)


def extract_rich_question_info(text, values=None):
    """
    問題文をピリオド '.' 単位の文 (Sentence) と節 (Clause) に分割し、
    各数値 (var1..varN) の局所文脈特徴・単位・意味ロール・照応関係 (remainder, house等のエンティティ参照) を抽出する。
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

    sentences = []
    curr_s = []
    for w in words:
        if w in (".", "?", ";"):
            if curr_s:
                sentences.append(curr_s)
                curr_s = []
        elif w != ",":
            curr_s.append(w)
    if curr_s:
        sentences.append(curr_s)

    var_clause_id = {}
    var_full_clause = {}
    clause_var_counts = {}
    for c_id, c in enumerate(clauses):
        v_in_c = [w for w in c if is_var_tok(w)]
        clause_var_counts[c_id] = len(v_in_c)
        for w in v_in_c:
            var_clause_id[w] = c_id
            var_full_clause[w] = c

    var_strict_units, var_sent_id = extract_strict_units_with_ellipsis(raw_tokens)

    q_asks_rate = bool(q_words_set & {"each", "every", "per", "apiece", "one", "daily"})
    asks_profit = bool(q_words_set & {"profit", "net"})
    asks_change = bool(q_words_set & {"change"}) and bool(q_words_set & {"back", "get", "receive", "left"})
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

    asks_money = bool(q_words_set & {"pay", "spend", "cost", "price", "money", "dollar", "cent", "change", "earn", "save", "profit"}) or (
        "how" in q_words and "much" in q_words and not (q_words_set & WEIGHT_UNITS)
    )
    asks_time = (
        ("how" in q_words and "long" in q_words)
        or bool((q_words_set - q_rate_nouns) & TIME_UNITS)
    ) and not asks_money
    asks_dist = ("how" in q_words and "far" in q_words) or bool((q_words_set - q_rate_nouns) & DIST_UNITS)
    asks_weight = ("how" in q_words and "heavy" in q_words) or "weigh" in q_words_set or bool((q_words_set - q_rate_nouns) & WEIGHT_UNITS)
    asks_human = bool((q_words_set - q_rate_nouns) & HUMAN_UNITS)

    q_target_nouns = extract_target_nouns_from_q_words(q_words, q_rate_nouns)

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
            c_id = var_clause_id.get(w, -1)
            full_c = var_full_clause.get(w, span)
            s_id = var_sent_id.get(w, -1)
            sent_tokens = sentences[s_id] if 0 <= s_id < len(sentences) else full_c
            span_set = set(span)
            full_c_set = set(full_c)
            sent_set = set(sent_tokens)

            prev1 = clean_words[i - 1] if i > 0 and clean_words[i - 1] in MATH_CORE_WORDS else "OBJ"
            prev2 = clean_words[i - 2] if i > 1 and clean_words[i - 2] in MATH_CORE_WORDS else "OBJ"
            next1 = clean_words[i + 1] if i + 1 < n_words else "NONE"

            is_frac_div = prev1 in ("half", "third", "quarter", "fourth")
            is_percent = (next1 == "percent") or ("percent" in span_set) or ("percent" in sent_set and val <= 5.0)
            is_mult_scale = is_percent or ("time" in units) or (next1 == "time")

            if is_mult_scale or is_frac_div:
                units = set()
                u1, u2 = "NONE", "NONE"

            local_window = set(clean_words[max(0, i - 4) : min(n_words, i + 5)]) - {w}

            entity_nouns = {
                tok for tok in full_c
                if not is_var_tok(tok) and tok not in ENTITY_STOP_WORDS and len(tok) > 2
            }
            ref_entities = {
                tok for tok in sent_tokens
                if not is_var_tok(tok) and tok not in ENTITY_STOP_WORDS and len(tok) > 2
            }

            rate_seqs = (span,) if clause_var_counts.get(c_id, 1) > 1 else (span, full_c)
            rate_units = set()
            for seq in rate_seqs:
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

            cues_sub = set((span_set | full_c_set) & SUB_CUES)
            cues_add = set((span_set | full_c_set) & ADD_CUES)
            cues_div = set(span_set & DIV_CUES)

            # 肯定比較表現 ("Spike has 5 more apples than Jerry") の判定 (文内の "than" も確認)
            is_comp_more = False
            if ("than" in cues_sub or "than" in sent_set) and w not in q_words_set:
                has_pos_comp = bool((span_set | full_c_set) & {"more", "older", "taller", "longer", "heavier", "faster", "higher", "extra", "additional"})
                has_neg_comp = bool((span_set | full_c_set) & {"less", "fewer", "smaller", "shorter", "younger", "cheaper", "lighter", "slower", "lower"})
                if has_pos_comp and not has_neg_comp:
                    cues_sub.discard("than")
                    cues_add.add("more")
                    is_comp_more = True

            if "take" in cues_sub and ("take" in q_words_set or bool(q_words_set & {"total", "altogether", "combined", "all"})):
                cues_sub.discard("take")
            if not (q_words_set & {"left", "remain", "remaining", "rest", "difference", "change", "extra", "more", "less", "fewer", "still", "need", "profit"}):
                cues_sub -= (q_words_set & {"sell", "eat", "spend", "give", "use", "drink", "lose", "burn", "drop", "break"})

            if (units & MONEY_UNITS) and rate_units and asks_money and ("sell" in cues_sub):
                cues_sub.discard("sell")

            if rate_units:
                cues_sub -= {"rest", "remainder", "remaining", "sell"}

            if prev1 == "with" and "bake" in cues_add:
                cues_add.discard("bake")
                cues_sub.add("use")

            if asks_profit and v_num >= 2 and (units & MONEY_UNITS) and not rate_units:
                if full_c_set & {"repair", "put", "spend", "cost", "pay", "buy", "expense", "fee"}:
                    cues_sub.add("spend")

            refers_to_remainder = bool(sent_set & {"remainder", "remaining", "rest"})

            if is_percent and (sent_set & {"increase", "more", "up", "gain", "raise"}):
                cues_add.add("increase")
                cues_sub.discard("decrease")

            has_each_in_span = bool(rate_units) or bool(span_set & {"each", "every", "per", "apiece"})

            if is_mult_scale:
                sem_role = "MULT_SCALE"
            elif is_frac_div:
                sem_role = "FRAC_DIV"
            elif has_each_in_span:
                sem_role = "RATE"
            elif asks_change and (units & MONEY_UNITS) and (full_c_set & {"pay", "give", "bill", "hand"}):
                sem_role = "PAID_AMOUNT"
                cues_sub.discard("pay")
            elif asks_profit and (units & MONEY_UNITS) and (full_c_set & {"buy", "purchase", "repair", "put", "spend", "cost"}):
                sem_role = "BASE_COST" if v_num == 1 else "EXTRA_COST"
            elif is_comp_more:
                sem_role = "COMP_MORE"
            elif cues_sub:
                sem_role = "DELTA_SUB"
            elif cues_add:
                sem_role = "DELTA_ADD"
            else:
                sem_role = "QTY"

            is_q_match = bool(
                (units & q_target_nouns)
                or (asks_money and (units & MONEY_UNITS))
                or (asks_time and (units & TIME_UNITS))
                or (asks_dist and (units & DIST_UNITS))
                or (asks_weight and (units & WEIGHT_UNITS))
                or (asks_human and (units & HUMAN_UNITS))
            )

            var_nodes[w] = {
                "name": w,
                "step_label": w,
                "expr": w,
                "val": val,
                "u1": u1,
                "u2": u2,
                "units": units,
                "rate_units": rate_units,
                "entity_nouns": entity_nouns,
                "ref_entities": ref_entities,
                "local_window": local_window,
                "sem_role": sem_role,
                "is_comp_more": is_comp_more,
                "refers_to_remainder": refers_to_remainder,
                "cues_sub": cues_sub,
                "cues_add": cues_add,
                "cues_div": cues_div,
                "has_each": has_each_in_span,
                "role_each_noun": bool(units & each_nouns),
                "in_q_clause": w in q_words_set,
                "q_match": is_q_match,
                "is_frac_div": is_frac_div,
                "is_percent": is_percent,
                "is_mult_scale": is_mult_scale,
                "is_distractor_cand": False,
                "prev1": prev1,
                "prev2": prev2,
                "pos_idx": i,
                "var_order": v_num,
                "clause_id": c_id,
                "sent_id": s_id,
                "vars_used": frozenset([w]),
                "depth": 0,
                "last_op": "INIT",
            }

    for vn in var_nodes.values():
        if vn["has_each"] and not vn["rate_units"]:
            c_peers = [
                other for other in var_nodes.values()
                if other["name"] != vn["name"]
                and other["clause_id"] == vn["clause_id"]
                and other["clause_id"] >= 0
                and other["units"]
                and not _are_units_compatible(other["units"], vn["units"], q_target_nouns)
            ]
            if not c_peers:
                c_peers = [
                    other for other in var_nodes.values()
                    if other["name"] != vn["name"]
                    and other["sent_id"] == vn["sent_id"]
                    and other["sent_id"] >= 0
                    and not other["has_each"]
                    and other["units"]
                    and not _are_units_compatible(other["units"], vn["units"], q_target_nouns)
                ]
            if c_peers:
                vn["rate_units"] = set(c_peers[0]["units"])
                c_peers[0]["role_each_noun"] = True

    q_matched_nodes = [vn for vn in var_nodes.values() if vn["q_match"]]
    if len(q_matched_nodes) >= 2 and len(var_nodes) >= 3:
        for vn in var_nodes.values():
            if (
                not vn["q_match"]
                and not vn["rate_units"]
                and not vn["has_each"]
                and not vn["role_each_noun"]
                and not vn["is_mult_scale"]
                and not vn["is_frac_div"]
                and not vn["in_q_clause"]
                and not any(_are_units_compatible(vn["units"], qn["units"], q_target_nouns) for qn in q_matched_nodes)
            ):
                vn["is_distractor_cand"] = True

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
                if _are_units_compatible(ev["units"], base_u, q_target_nouns) and not ev["rate_units"]:
                    ev["cues_sub"].add("leave")
                    ev["cues_add"].clear()
                    ev["sem_role"] = "DELTA_SUB"

    q_abstract = [
        clean_words[idx]
        for idx in range(q_idx, n_words)
        if is_var_tok(clean_words[idx]) or clean_words[idx] in MATH_CORE_WORDS
    ]

    wants_combine_all = bool(
        (q_words_set & {"total", "altogether", "together", "combined", "all", "both"})
        or ("and" in q_words_set and not (q_words_set & {"difference", "more", "less", "fewer", "extra", "left", "remain", "profit", "change"}))
    )
    asks_comparison_or_rem = bool(
        q_words_set & {"extra", "more", "less", "fewer", "left", "remain", "remaining", "rest", "difference", "still", "end", "profit", "change"}
    )

    q_ctx = {
        "clean_words": clean_words,
        "sentences": sentences,
        "num_sentences": len(sentences),
        "q_words": q_words,
        "q_words_set": q_words_set,
        "q_abstract": q_abstract,
        "q_target_nouns": q_target_nouns,
        "q_asks_rate": q_asks_rate,
        "q_rate_nouns": q_rate_nouns,
        "asks_money": asks_money,
        "asks_profit": asks_profit,
        "asks_change": asks_change,
        "asks_time": asks_time,
        "asks_dist": asks_dist,
        "asks_weight": asks_weight,
        "asks_human": asks_human,
        "each_nouns": each_nouns,
        "wants_combine_all": wants_combine_all,
        "asks_comparison_or_rem": asks_comparison_or_rem,
        "num_vars": len(var_nodes),
    }
    return var_nodes, q_ctx


def compute_local_context_pointer_bonus(nodeL, nodeR, action, q_ctx):
    ptr_bonus = 0.0
    sL, sR = nodeL.get("sent_id", -1), nodeR.get("sent_id", -1)
    cL, cR = nodeL.get("clause_id", -1), nodeR.get("clause_id", -1)

    if cL >= 0 and cL == cR:
        ptr_bonus += 0.65
    elif sL >= 0 and sR >= 0:
        if sL == sR:
            ptr_bonus += 0.45
        elif sR == sL + 1 and nodeL["depth"] >= 1:
            ptr_bonus += 0.65

    if (nodeR.get("refers_to_remainder") or nodeL.get("refers_to_remainder")) and action == "MUL":
        if nodeL.get("sem_role") == "REMAINDER" or nodeR.get("sem_role") == "REMAINDER":
            ptr_bonus += 2.2

    ref_R = nodeR.get("ref_entities", set())
    ent_L = nodeL.get("entity_nouns", set())
    if ref_R and ent_L and (ref_R & ent_L):
        if action in ("MUL", "DIV_LR") and (nodeR.get("is_mult_scale") or nodeR.get("is_frac_div")):
            if nodeL["depth"] == 0:
                ptr_bonus += 2.4
        elif action == "ADD" and nodeR.get("is_comp_more"):
            ptr_bonus += 3.2

    roleL, roleR = nodeL.get("sem_role", "QTY"), nodeR.get("sem_role", "QTY")
    if q_ctx.get("asks_profit"):
        if action == "ADD" and roleL == "BASE_COST" and roleR == "EXTRA_COST":
            ptr_bonus += 2.2
        elif action == "ADD" and roleL == "BASE_COST" and roleR == "INCREASE_VAL":
            ptr_bonus += 2.8
        elif action == "SUB_LR" and roleL == "NEW_VALUE" and roleR in ("TOTAL_COST", "EXTRA_COST"):
            ptr_bonus += 3.6
        elif action == "SUB_RL" and roleR == "NEW_VALUE" and roleL in ("TOTAL_COST", "EXTRA_COST"):
            ptr_bonus += 3.6
    if q_ctx.get("asks_change"):
        if action == "ADD" and roleL == "SUBTOTAL" and roleR == "SUBTOTAL":
            ptr_bonus += 2.6
        elif action == "SUB_LR" and roleL == "PAID_AMOUNT" and roleR in ("SUBTOTAL", "TOTAL_COST", "QTY"):
            ptr_bonus += 3.2
        elif action == "SUB_RL" and roleR == "PAID_AMOUNT" and roleL in ("SUBTOTAL", "TOTAL_COST", "QTY"):
            ptr_bonus += 3.2

    return ptr_bonus


def apply_step_operation(nodeL, nodeR, action, board_nodes, q_ctx, step_idx=1):
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
    roleL, roleR = op_L.get("sem_role", "QTY"), op_R.get("sem_role", "QTY")
    q_target = q_ctx["q_target_nouns"]

    same_unit = _are_units_compatible(uL, uR, q_target)
    diff_unit = bool(uL and uR and not same_unit)

    step_dim_bonus = compute_local_context_pointer_bonus(nodeL, nodeR, action, q_ctx)
    other_nodes = [
        n for n in board_nodes
        if n["expr"] != nodeL["expr"] and n["expr"] != nodeR["expr"]
    ]

    new_sem_role = "STEP_VAL"

    if op_sym in ("+", "-"):
        if op_L.get("is_frac_div") or op_R.get("is_frac_div") or op_L.get("is_mult_scale") or op_R.get("is_mult_scale"):
            return None
        new_units = uL | uR
        new_rate_units = rL | rR

        if op_sym == "+":
            if roleL == "PAID_AMOUNT" or roleR == "PAID_AMOUNT":
                step_dim_bonus -= 4.0
            if roleL in ("BASE_COST", "TOTAL_COST") and roleR in ("EXTRA_COST", "BASE_COST"):
                new_sem_role = "TOTAL_COST"
            elif (roleL == "BASE_COST" and roleR == "INCREASE_VAL") or (roleR == "BASE_COST" and roleL == "INCREASE_VAL"):
                new_sem_role = "NEW_VALUE"
            elif roleR == "COMP_MORE" or roleL == "COMP_MORE":
                new_sem_role = "DERIVED_QTY"
            else:
                new_sem_role = "SUBTOTAL"
                # まだ未適用の比較増分ノード (COMP_MORE: "5 more than Jerry") が盤面に残っているのに先に親同士を合算するのを防ぐ
                if any(n.get("is_comp_more") and bool(n.get("ref_entities", set()) & (op_L.get("entity_nouns", set()) | op_R.get("entity_nouns", set()))) for n in other_nodes):
                    step_dim_bonus -= 3.2
        else:
            if roleL == "NEW_VALUE" and roleR in ("TOTAL_COST", "EXTRA_COST", "BASE_COST"):
                new_sem_role = "PROFIT"
            elif roleL == "PAID_AMOUNT":
                new_sem_role = "REMAINDER"
                if any(bool(n["rate_units"]) for n in other_nodes):
                    step_dim_bonus -= 4.5
            else:
                new_sem_role = "REMAINDER"

        if op_L["vars_used"] & op_R["vars_used"]:
            if op_sym == "+" and (q_ctx.get("wants_combine_all") or "increase" in (op_L["cues_add"] | op_R["cues_add"]) or new_sem_role == "NEW_VALUE"):
                step_dim_bonus += 3.5
            elif op_sym == "-" and (op_L["cues_sub"] or op_R["cues_sub"] or q_ctx.get("asks_comparison_or_rem") or new_sem_role == "PROFIT"):
                step_dim_bonus += 3.5
        if same_unit:
            step_dim_bonus += 1.5
            if rL and rR:
                if _are_units_compatible(rL, rR, q_target):
                    has_separate_qty_L = any(
                        n["clause_id"] == op_L["clause_id"] and _are_units_compatible(n["units"], rL, q_target)
                        for n in other_nodes
                    )
                    has_separate_qty_R = any(
                        n["clause_id"] == op_R["clause_id"] and _are_units_compatible(n["units"], rR, q_target)
                        for n in other_nodes
                    )
                    if has_separate_qty_L and has_separate_qty_R and op_L["clause_id"] != op_R["clause_id"]:
                        step_dim_bonus -= 4.0
                    else:
                        step_dim_bonus += 2.2
                else:
                    step_dim_bonus -= 4.5
            elif bool(rL) != bool(rR) and (op_L["last_op"] == "*" or op_R["last_op"] == "*" or rL or rR):
                step_dim_bonus -= 4.5

            if op_sym == "-":
                if op_R["cues_sub"] and not op_L["cues_sub"]:
                    step_dim_bonus += 2.4
                elif new_sem_role == "PROFIT" or roleL == "PAID_AMOUNT":
                    step_dim_bonus += 3.0
                elif op_L["cues_sub"] and not op_R["cues_sub"] and op_L["depth"] == 0:
                    step_dim_bonus -= 2.5
                elif not op_L["cues_sub"] and not op_R["cues_sub"]:
                    if any(bool(n["cues_sub"]) and _are_units_compatible(n["units"], new_units, q_target) for n in other_nodes):
                        step_dim_bonus -= 2.2
                    elif q_ctx.get("wants_combine_all"):
                        step_dim_bonus -= 2.5
            else:
                if "discount" in (op_L["cues_sub"] | op_R["cues_sub"]):
                    step_dim_bonus -= 3.0
                elif op_R["cues_sub"] and not op_R["cues_add"] and new_sem_role != "TOTAL_COST" and (
                    "leave" in op_R["cues_sub"]
                    or "delete" in op_R["cues_sub"]
                    or "eat" in op_R["cues_sub"]
                    or "already" in op_R["cues_sub"]
                ):
                    step_dim_bonus -= 2.5
                elif not op_L["cues_sub"] and not op_R["cues_sub"]:
                    if (
                        (op_L["cues_add"] or op_R["cues_add"])
                        or (op_L.get("sent_id", -1) >= 0 and op_L.get("sent_id") == op_R.get("sent_id"))
                        or any(bool(n["cues_sub"]) and _are_units_compatible(n["units"], new_units, q_target) for n in other_nodes)
                        or q_ctx.get("wants_combine_all")
                    ):
                        step_dim_bonus += 1.6
        elif diff_unit:
            if op_L.get("is_distractor_cand") or op_R.get("is_distractor_cand"):
                step_dim_bonus -= 6.0
            else:
                has_same_unit_peer_left = any(
                    _are_units_compatible(n["units"], uL, q_target) or _are_units_compatible(n["units"], uR, q_target)
                    for n in other_nodes
                )
                incompat_phys = (
                    bool((uL | uR) & MONEY_UNITS)
                    or bool((uL | uR) & TIME_UNITS)
                    or bool((uL | uR) & DIST_UNITS)
                    or bool((uL | uR) & WEIGHT_UNITS)
                )
                if (
                    op_sym == "-"
                    and not rL
                    and not rR
                    and not op_L["has_each"]
                    and not op_R["has_each"]
                    and not incompat_phys
                    and not has_same_unit_peer_left
                    and (op_R["cues_sub"] or q_ctx.get("asks_comparison_or_rem"))
                ):
                    new_units = set(uL)
                    step_dim_bonus += 2.0
                else:
                    step_dim_bonus -= 4.5

    elif op_sym == "*":
        cancel_lr = _are_units_compatible(rL, uR, q_target)
        cancel_rl = _are_units_compatible(rR, uL, q_target)
        if cancel_lr and not cancel_rl:
            new_units = set(uL)
            new_rate_units = set()
            new_sem_role = "SUBTOTAL"
            step_dim_bonus += 2.5
            if op_L["clause_id"] == op_R["clause_id"] and op_L["clause_id"] >= 0:
                step_dim_bonus += 1.8
            premature_qty = any(
                _are_units_compatible(n["units"], uR, q_target)
                and not n["rate_units"]
                and bool(n["cues_sub"] | n["cues_add"])
                and n["clause_id"] == op_R["clause_id"]
                for n in other_nodes
            )
            if premature_qty:
                step_dim_bonus -= 4.5
            elif op_L["depth"] >= 1 or op_R["depth"] >= 1:
                step_dim_bonus += 2.5
            elif any(_are_units_compatible(n["units"], uL, q_target) and _are_units_compatible(n["rate_units"], rL, q_target) for n in other_nodes):
                if op_L["cues_sub"] or any(bool(n["cues_sub"]) for n in other_nodes if _are_units_compatible(n["units"], uL, q_target)):
                    step_dim_bonus -= 4.0
                else:
                    step_dim_bonus -= 1.5
        elif cancel_rl and not cancel_lr:
            new_units = set(uR)
            new_rate_units = set()
            new_sem_role = "SUBTOTAL"
            step_dim_bonus += 2.5
            if op_L["clause_id"] == op_R["clause_id"] and op_L["clause_id"] >= 0:
                step_dim_bonus += 1.8
            premature_qty = any(
                _are_units_compatible(n["units"], uL, q_target)
                and not n["rate_units"]
                and bool(n["cues_sub"] | n["cues_add"])
                and n["clause_id"] == op_L["clause_id"]
                for n in other_nodes
            )
            if premature_qty:
                step_dim_bonus -= 4.5
            elif op_R["depth"] >= 1 or op_L["depth"] >= 1:
                step_dim_bonus += 2.5
            elif any(_are_units_compatible(n["units"], uR, q_target) and _are_units_compatible(n["rate_units"], rR, q_target) for n in other_nodes):
                if op_R["cues_sub"] or any(bool(n["cues_sub"]) for n in other_nodes if _are_units_compatible(n["units"], uR, q_target)):
                    step_dim_bonus -= 4.0
                else:
                    step_dim_bonus -= 1.5
        elif op_R.get("is_mult_scale") and not op_L.get("is_mult_scale"):
            new_units = set(uL)
            new_rate_units = set(rL)
            new_sem_role = "INCREASE_VAL" if ("increase" in op_R["cues_add"]) else "DERIVED_QTY"
            step_dim_bonus += 2.2
            ref_ents = op_R.get("ref_entities", set())
            l_ents = op_L.get("entity_nouns", set())
            if ref_ents and l_ents and (ref_ents & l_ents):
                if op_L["depth"] == 0:
                    step_dim_bonus += 3.8
                elif (l_ents - ref_ents):
                    step_dim_bonus -= 3.5
            elif ref_ents and any(bool(ref_ents & n.get("entity_nouns", set())) for n in other_nodes):
                step_dim_bonus -= 4.0
        elif op_L.get("is_mult_scale") and not op_R.get("is_mult_scale"):
            new_units = set(uR)
            new_rate_units = set(rR)
            new_sem_role = "INCREASE_VAL" if ("increase" in op_L["cues_add"]) else "DERIVED_QTY"
            step_dim_bonus += 2.2
            ref_ents = op_L.get("ref_entities", set())
            r_ents = op_R.get("entity_nouns", set())
            if ref_ents and r_ents and (ref_ents & r_ents):
                step_dim_bonus += 3.2
            elif ref_ents and any(bool(ref_ents & n.get("entity_nouns", set())) for n in other_nodes):
                step_dim_bonus -= 4.0
        else:
            new_units = uL if op_L["q_match"] else (uR if op_R["q_match"] else (uL | uR))
            new_rate_units = set()
            if same_unit and uL:
                step_dim_bonus -= 3.5
            elif not op_L["has_each"] and not op_R["has_each"] and not op_L["role_each_noun"] and not op_R["role_each_noun"]:
                if op_L["cues_sub"] or op_R["cues_sub"] or q_ctx.get("asks_comparison_or_rem"):
                    step_dim_bonus -= 3.5

    else:  # "/"
        if abs(new_val - round(new_val)) > 1e-6:
            if abs(new_val * 4 - round(new_val * 4)) > 1e-5 and abs(new_val * 10 - round(new_val * 10)) > 1e-5:
                step_dim_bonus -= 2.5
            else:
                step_dim_bonus -= 0.5
        if op_R.get("is_frac_div") and not op_L.get("is_frac_div"):
            new_units = set(uL)
            new_rate_units = set(rL)
            new_sem_role = "DERIVED_QTY"
            step_dim_bonus += 2.6
        elif same_unit and rR:
            new_units = set(rR)
            new_rate_units = set()
            step_dim_bonus += 2.4
            if any(
                _are_units_compatible(n["units"], uL, q_target)
                and not n["rate_units"]
                and bool(n["cues_sub"] | n["cues_add"])
                for n in other_nodes
            ):
                step_dim_bonus -= 4.0
        elif diff_unit:
            new_units = set(uL)
            new_rate_units = set(uR)
            if q_ctx["q_asks_rate"] or op_R.get("cues_div") or op_L.get("cues_div"):
                step_dim_bonus += 1.2
            elif op_R["cues_sub"] and not op_R.get("cues_div"):
                step_dim_bonus -= 3.8
        else:
            new_units = set(uL)
            new_rate_units = set()
            if op_R["cues_sub"] and not op_R.get("cues_div"):
                step_dim_bonus -= 3.0

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
        or (q_ctx["asks_human"] and (new_units & HUMAN_UNITS))
    )

    inherited_cues_sub = (op_L["cues_sub"] | op_R["cues_sub"]) if op_sym != "-" else set()
    if new_sem_role == "TOTAL_COST" and q_ctx.get("asks_profit"):
        inherited_cues_sub = set(inherited_cues_sub) | {"spend"}

    new_node = {
        "name": norm_expr,
        "step_label": f"step{step_idx}",
        "expr": norm_expr,
        "val": new_val,
        "u1": sorted(new_units)[0] if new_units else "NONE",
        "u2": "NONE",
        "units": new_units,
        "rate_units": new_rate_units,
        "entity_nouns": op_L.get("entity_nouns", set()) | op_R.get("entity_nouns", set()),
        "ref_entities": set(),
        "local_window": op_L.get("local_window", set()) | op_R.get("local_window", set()),
        "sem_role": new_sem_role,
        "is_comp_more": False,
        "refers_to_remainder": False,
        "cues_sub": inherited_cues_sub,
        "cues_add": set(),
        "cues_div": set(),
        "has_each": bool(new_rate_units),
        "role_each_noun": False,
        "in_q_clause": op_L["in_q_clause"] or op_R["in_q_clause"],
        "q_match": is_q_match,
        "is_frac_div": False,
        "is_percent": False,
        "is_mult_scale": False,
        "is_distractor_cand": False,
        "prev1": op_L["prev1"],
        "prev2": op_L["prev2"],
        "pos_idx": min(op_L["pos_idx"], op_R["pos_idx"]),
        "var_order": max(op_L["var_order"], op_R["var_order"]),
        "clause_id": max(op_L["clause_id"], op_R["clause_id"]),
        "sent_id": max(op_L.get("sent_id", -1), op_R.get("sent_id", -1)),
        "vars_used": op_L["vars_used"] | op_R["vars_used"],
        "depth": max(op_L["depth"], op_R["depth"]) + 1,
        "last_op": op_sym,
    }
    return new_node, step_dim_bonus


def extract_step_pair_features(nodeL, nodeR, q_ctx, step_num=1):
    feat = {}
    uL, uR = nodeL["units"], nodeR["units"]
    rL, rR = nodeL["rate_units"], nodeR["rate_units"]
    q_target = q_ctx["q_target_nouns"]

    same_unit = _are_units_compatible(uL, uR, q_target)
    diff_unit = bool(uL and uR and not same_unit)

    feat[f"S{min(step_num, 4)}::BIAS"] = 1.0
    feat[f"NVARS::{min(q_ctx['num_vars'], 5)}"] = 1.0
    feat[f"ROLE_PAIR::{nodeL.get('sem_role', 'QTY')}_{nodeR.get('sem_role', 'QTY')}"] = 1.0

    if nodeL["depth"] > 0:
        feat["L::IS_INTERMEDIATE_STEP"] = 1.0
    if nodeR["depth"] > 0:
        feat["R::IS_INTERMEDIATE_STEP"] = 1.0

    if same_unit:
        feat["PAIR::SAME_UNIT"] = 1.0
    elif diff_unit:
        feat["PAIR::DIFF_UNIT"] = 1.0

    if rL and rR and _are_units_compatible(rL, rR, q_target):
        feat["PAIR::SAME_RATE_DENOM"] = 1.0
    if bool(rL) != bool(rR) and same_unit:
        feat["PAIR::RATE_MISMATCH"] = 1.0

    if _are_units_compatible(rL, uR, q_target):
        feat["PAIR::CANCEL_LR"] = 1.0
    if _are_units_compatible(rR, uL, q_target):
        feat["PAIR::CANCEL_RL"] = 1.0

    if nodeL["clause_id"] == nodeR["clause_id"] and nodeL["clause_id"] >= 0:
        feat["PAIR::SAME_CLAUSE"] = 1.0
    sL, sR = nodeL.get("sent_id", -1), nodeR.get("sent_id", -1)
    if sL == sR and sL >= 0:
        feat["PAIR::SAME_SENT"] = 1.0
    elif sL >= 0 and sR == sL + 1:
        feat["PAIR::NEXT_SENT_STEP"] = 1.0

    if nodeL.get("entity_nouns", set()) & nodeR.get("ref_entities", set()):
        feat["PAIR::ENTITY_REF_MATCH_LR"] = 1.0
    if nodeR.get("refers_to_remainder") and nodeL.get("sem_role") == "REMAINDER":
        feat["PAIR::REMAINDER_ANAPHORA_LR"] = 1.0

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
    if same_unit and (nodeR["cues_add"] or (sL == sR and sL >= 0 and not nodeL["cues_sub"] and not nodeR["cues_sub"])):
        feat["COMBO::SAME_U_R_ADD"] = 1.0
    if _are_units_compatible(rL, uR, q_target) and not nodeL["cues_sub"]:
        feat["COMBO::CLEAN_RATE_MUL_LR"] = 1.0
    if _are_units_compatible(rR, uL, q_target) and not nodeR["cues_sub"]:
        feat["COMBO::CLEAN_RATE_MUL_RL"] = 1.0

    if q_ctx["q_asks_rate"]:
        feat["Q::ASKS_RATE"] = 1.0
    if q_ctx.get("asks_profit"):
        feat["Q::ASKS_PROFIT"] = 1.0
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

        res_out = apply_step_operation(target_L, target_R, true_act, board_list, q_ctx, step_idx=step_idx)
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
# 6. AST評価 ＆ 次元解析 (中間ステップの単位・レート解消状態を正確に伝播)
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
    q_ctx=None,
):
    allow_dup = True
    if var_nodes is not None and q_ctx is not None:
        has_scale_or_comp = any(
            vn.get("is_mult_scale")
            or vn.get("is_frac_div")
            or bool((vn["cues_add"] | vn["cues_sub"]) & {"more", "less", "fewer", "than", "increase", "decrease"})
            for vn in var_nodes.values()
        ) or q_ctx.get("asks_profit", False)
        allow_dup = has_scale_or_comp

    if is_self_canceling_or_invalid_duplicate(
        expression, num_vars_in_question=len(values), allow_derived_dup=allow_dup
    ):
        return False, -1e9, None

    try:
        tree = ast.parse(expression, mode="eval")
    except Exception:
        return False, -1e9, None

    variables = {f"var{i + 1}": v for i, v in enumerate(values)}
    all_used_vars = {f"var{idx}" for idx in VAR_RE.findall(expression)}
    q_target = q_ctx["q_target_nouns"] if q_ctx is not None else set()
    bonus = 0.0
    valid = True

    def eval_node(node):
        nonlocal bonus, valid
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                valid = False
                return 0.0, None, set(), set(), frozenset()
            u_set = set(var_nodes[node.id]["units"]) if (var_nodes and node.id in var_nodes) else set()
            r_set = set(var_nodes[node.id]["rate_units"]) if (var_nodes and node.id in var_nodes) else set()
            return variables[node.id], node.id, u_set, r_set, frozenset([node.id])
        if isinstance(node, ast.Constant):
            return float(node.value), None, set(), set(), frozenset()
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val, u_var, u_set, r_set, sub_v = eval_node(node.operand)
            return -val, u_var, u_set, r_set, sub_v
        if isinstance(node, ast.BinOp):
            left_val, left_u, left_units, left_rate, left_vars = eval_node(node.left)
            right_val, right_u, right_units, right_rate, right_vars = eval_node(node.right)
            if not valid:
                return 0.0, None, set(), set(), frozenset()
            cur_vars = left_vars | right_vars

            if isinstance(node.op, (ast.Add, ast.Sub)):
                if var_nodes is not None:
                    if (left_u in var_nodes and isinstance(node.left, ast.Name) and (var_nodes[left_u].get("is_frac_div") or var_nodes[left_u].get("is_mult_scale"))) or (
                        right_u in var_nodes and isinstance(node.right, ast.Name) and (var_nodes[right_u].get("is_frac_div") or var_nodes[right_u].get("is_mult_scale"))
                    ):
                        valid = False
                        return 0.0, None, set(), set(), frozenset()

                same_sub_u = _are_units_compatible(left_units, right_units, q_target)
                diff_sub_u = bool(left_units and right_units and not same_sub_u)

                if left_u is not None and right_u is not None and left_u != right_u:
                    pair = tuple(sorted([left_u, right_u]))
                    if same_sub_u or pair in same_unit_pairs:
                        bonus += 1.5
                        if bool(left_rate) != bool(right_rate):
                            bonus -= 4.5
                        elif left_rate and right_rate and not _are_units_compatible(left_rate, right_rate, q_target):
                            bonus -= 4.5
                        if var_nodes is not None and left_u in var_nodes and right_u in var_nodes:
                            nL, nR = var_nodes[left_u], var_nodes[right_u]
                            if isinstance(node.op, ast.Sub):
                                if nR["cues_sub"] and not nL["cues_sub"]:
                                    bonus += 1.8
                                elif nL.get("sem_role") == "PAID_AMOUNT":
                                    bonus += 2.6
                                elif nL["cues_sub"] and not nR["cues_sub"] and isinstance(node.left, ast.Name):
                                    bonus -= 1.8
                            else:
                                if nL.get("sem_role") == "PAID_AMOUNT" or nR.get("sem_role") == "PAID_AMOUNT":
                                    bonus -= 3.5
                                elif "discount" in (nL["cues_sub"] | nR["cues_sub"]):
                                    bonus -= 2.5
                                elif (
                                    q_ctx
                                    and q_ctx.get("asks_profit")
                                    and ("spend" in nR["cues_sub"] or "spend" in nL["cues_sub"])
                                    and not (isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name))
                                ):
                                    bonus -= 2.5
                                elif not nL["cues_sub"] and not nR["cues_sub"]:
                                    if nL["cues_add"] or nR["cues_add"] or (nL.get("sent_id", -1) >= 0 and nL.get("sent_id") == nR.get("sent_id")):
                                        bonus += 1.2
                    elif diff_sub_u or pair in diff_unit_pairs:
                        if var_nodes is not None and (
                            (left_u in var_nodes and var_nodes[left_u].get("is_distractor_cand"))
                            or (right_u in var_nodes and var_nodes[right_u].get("is_distractor_cand"))
                        ):
                            bonus -= 6.0
                        elif (
                            isinstance(node.op, ast.Sub)
                            and var_nodes is not None
                            and left_u in var_nodes
                            and right_u in var_nodes
                            and not left_rate
                            and not right_rate
                            and not var_nodes[left_u]["has_each"]
                            and not var_nodes[right_u]["has_each"]
                            and (var_nodes[right_u]["cues_sub"] or (q_ctx and q_ctx.get("asks_comparison_or_rem")))
                        ):
                            bonus += 1.5
                        elif len(same_unit_pairs) > 0:
                            bonus -= 4.5
                elif left_u is not None and left_u == right_u:
                    if isinstance(node.left, ast.Name) or isinstance(node.right, ast.Name):
                        if isinstance(node.op, ast.Add) and q_ctx is not None and (
                            q_ctx.get("wants_combine_all") or q_ctx.get("asks_profit")
                        ):
                            bonus += 4.2
                        elif isinstance(node.op, ast.Sub):
                            bonus += 3.8
                    elif isinstance(node.op, ast.Sub) and q_ctx is not None and q_ctx.get("asks_profit"):
                        bonus += 4.5
                    elif isinstance(node.op, ast.Add) and q_ctx is not None and q_ctx.get("wants_combine_all"):
                        bonus += 4.2

                rep_u = left_u or right_u
                comb_units = left_units | right_units
                comb_rate = left_rate | right_rate
                if isinstance(node.op, ast.Add):
                    return left_val + right_val, rep_u, comb_units, comb_rate, cur_vars
                res = left_val - right_val
                if res <= 0:
                    valid = False
                return res, rep_u, comb_units, comb_rate, cur_vars

            if isinstance(node.op, ast.Mult):
                if var_nodes is not None:
                    if _are_units_compatible(left_rate, right_units, q_target) and left_u is not None:
                        outside_vars = all_used_vars - cur_vars
                        if any(
                            ov in var_nodes
                            and _are_units_compatible(var_nodes[ov]["units"], right_units, q_target)
                            and not var_nodes[ov]["rate_units"]
                            and bool(var_nodes[ov]["cues_sub"] | var_nodes[ov]["cues_add"])
                            and var_nodes[ov]["clause_id"] == (var_nodes[right_u]["clause_id"] if right_u in var_nodes else -2)
                            for ov in outside_vars
                        ):
                            bonus -= 4.0
                        else:
                            bonus += 2.0
                        return left_val * right_val, left_u, set(left_units), set(), cur_vars
                    if _are_units_compatible(right_rate, left_units, q_target) and right_u is not None:
                        outside_vars = all_used_vars - cur_vars
                        if any(
                            ov in var_nodes
                            and _are_units_compatible(var_nodes[ov]["units"], left_units, q_target)
                            and not var_nodes[ov]["rate_units"]
                            and bool(var_nodes[ov]["cues_sub"] | var_nodes[ov]["cues_add"])
                            and var_nodes[ov]["clause_id"] == (var_nodes[left_u]["clause_id"] if left_u in var_nodes else -2)
                            for ov in outside_vars
                        ):
                            bonus -= 4.0
                        else:
                            bonus += 2.0
                        return left_val * right_val, right_u, set(right_units), set(), cur_vars
                    if right_u in var_nodes and var_nodes[right_u].get("is_mult_scale") and left_u is not None:
                        bonus += 1.8
                        ref_e = var_nodes[right_u].get("ref_entities", set())
                        l_all_ents = set()
                        for lv in left_vars:
                            if lv in var_nodes:
                                l_all_ents |= var_nodes[lv].get("entity_nouns", set())
                        if ref_e and (ref_e & l_all_ents):
                            if len(left_vars) == 1:
                                bonus += 3.8
                            elif any(
                                lv in var_nodes and not bool(ref_e & var_nodes[lv].get("entity_nouns", set()))
                                for lv in left_vars
                            ):
                                bonus -= 3.8
                        elif ref_e and any(
                            ov in var_nodes and bool(ref_e & var_nodes[ov].get("entity_nouns", set()))
                            for ov in (set(var_nodes.keys()) - left_vars)
                        ):
                            bonus -= 4.0
                        return left_val * right_val, left_u, set(left_units), set(left_rate), cur_vars
                    if left_u in var_nodes and var_nodes[left_u].get("is_mult_scale") and right_u is not None:
                        bonus += 1.8
                        ref_e = var_nodes[left_u].get("ref_entities", set())
                        r_e = var_nodes[right_u].get("entity_nouns", set()) if right_u in var_nodes else set()
                        if ref_e and r_e and (ref_e & r_e):
                            bonus += 3.2
                        elif ref_e and any(
                            ov in var_nodes and bool(ref_e & var_nodes[ov].get("entity_nouns", set()))
                            for ov in (set(var_nodes.keys()) - right_vars)
                        ):
                            bonus -= 4.0
                        return left_val * right_val, right_u, set(right_units), set(right_rate), cur_vars
                    if left_u in var_nodes and right_u in var_nodes:
                        nL, nR = var_nodes[left_u], var_nodes[right_u]
                        if _are_units_compatible(left_units, right_units, q_target) and left_units:
                            bonus -= 3.5
                        elif not nL["has_each"] and not nR["has_each"] and not nL["role_each_noun"] and not nR["role_each_noun"]:
                            if nL["cues_sub"] or nR["cues_sub"] or (q_ctx and q_ctx.get("asks_comparison_or_rem")):
                                bonus -= 3.5
                return left_val * right_val, (left_u or right_u), (left_units | right_units), set(), cur_vars

            if isinstance(node.op, ast.Div):
                if abs(right_val) < 1e-9:
                    valid = False
                    return 0.0, None, set(), set(), frozenset()
                res = left_val / right_val
                if abs(res - round(res)) > 1e-6:
                    if abs(res * 4 - round(res * 4)) > 1e-5 and abs(res * 10 - round(res * 10)) > 1e-5:
                        bonus -= 2.5
                    else:
                        bonus -= 0.5
                if var_nodes is not None:
                    if right_u in var_nodes and var_nodes[right_u].get("is_frac_div"):
                        bonus += 1.8
                        return res, left_u, set(left_units), set(left_rate), cur_vars
                    if right_u in var_nodes and var_nodes[right_u]["cues_sub"] and not var_nodes[right_u].get("cues_div"):
                        bonus -= 3.8
                    if _are_units_compatible(left_units, right_units, q_target) and right_rate:
                        bonus += 2.0
                        return res, left_u, set(right_rate), set(), cur_vars
                return res, left_u, set(left_units), set(right_units), cur_vars

        valid = False
        return 0.0, None, set(), set(), frozenset()

    final_val, _, _, _, _ = eval_node(tree)
    if not valid or final_val <= 0:
        return False, -1e9, None

    if q_match_vars:
        for v in q_match_vars:
            bonus += 0.6 if v in all_used_vars else -0.8
        if not (all_used_vars & q_match_vars):
            bonus -= 5.0

    if var_nodes is not None and q_ctx is not None and isinstance(tree.body, ast.BinOp) and isinstance(tree.body.op, (ast.Mult, ast.Div)):
        if isinstance(tree.body.left, ast.Name) and isinstance(tree.body.right, ast.Name):
            n_l = var_nodes.get(tree.body.left.id)
            n_r = var_nodes.get(tree.body.right.id)
            if (n_l and (n_l.get("is_mult_scale") or n_l.get("is_frac_div"))) or (n_r and (n_r.get("is_mult_scale") or n_r.get("is_frac_div"))):
                scale_node = n_r if (n_r and (n_r.get("is_mult_scale") or n_r.get("is_frac_div"))) else n_l
                if q_ctx.get("wants_combine_all"):
                    bonus -= 3.5
                elif scale_node.get("is_percent") and (scale_node["cues_sub"] or q_ctx.get("asks_comparison_or_rem")):
                    bonus -= 3.5

    if var_nodes is not None and len(all_used_vars) < len(var_nodes):
        for v_name, vn in var_nodes.items():
            if v_name not in all_used_vars:
                if vn.get("is_distractor_cand"):
                    continue
                if vn["cues_sub"] or vn["cues_add"] or vn["rate_units"] or vn["q_match"] or vn["is_mult_scale"] or vn["is_frac_div"]:
                    bonus -= 2.8
                else:
                    bonus -= 0.8

    for v_a, v_b in same_unit_pairs:
        if (v_a in all_used_vars) != (v_b in all_used_vars):
            v_used = v_a if v_a in all_used_vars else v_b
            v_unused = v_b if v_a in all_used_vars else v_a
            if var_nodes is not None and v_used in var_nodes and v_unused in var_nodes:
                if var_nodes[v_unused].get("is_distractor_cand"):
                    continue
                nu, nun = var_nodes[v_used], var_nodes[v_unused]
                if nu["rate_units"] and nun["rate_units"] and _are_units_compatible(nu["rate_units"], nun["rate_units"], q_target):
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
                    for v_other in all_used_vars if v_other != v_used
                ):
                    bonus -= 2.0

    return True, bonus, final_val


# ==========================================
# 7. 文順・中間変数ステップ生成 (step1 = op(arg1, arg2)) + BoW[next_word][word] + NNUE モデル
# ==========================================
class MathNNUEModel:
    """
    1. 文順・中間変数ステップ生成 (`step1 = op(arg1, arg2)` Scratchpad 方式 + 局所文脈ポインタ)
    2. 自己回帰型 `BoW[next_word][word]` + `NNUE[id(next_word, word)][H]`
    のハイブリッド推論モデル。
    """

    def __init__(self, h1=HIDDEN1_SIZE, h2=HIDDEN2_SIZE, beam_width=BEAM_WIDTH):
        self.h1 = h1
        self.h2 = h2
        self.beam_width = beam_width

        self.next_words = []
        self.next_word_to_id = {}
        self.num_next_words = 0

        self.words = []
        self.word_to_id = {}
        self.num_words = 0

        self.equations = []
        self.eq_to_id = {}

        self.weight1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.W3 = None
        self.bow_w = None
        self.bow_b = None

        self.step_actions = list(STEP_ACTIONS)
        self.step_act_to_id = {a: i for i, a in enumerate(self.step_actions)}
        self.step_words = []
        self.step_word_to_id = {}
        self.step_w = None
        self.step_b = None

        self.is_loaded = False

    def _collect_formula_step_samples(self, question_words, eq, num_vars):
        if not is_pure_var_formula(eq):
            return []
        f_tokens = formula_to_tokens(eq)
        m = len(f_tokens)
        step_samples = []
        for z in range(m):
            next_word_z = f_tokens[z]
            prefix = f_tokens[:z]
            feat_dict = build_autoregressive_feature_dict(question_words, prefix, num_vars=num_vars)
            step_samples.append((feat_dict, next_word_z))
        return step_samples

    def fit_and_save(self, dataset, weights_path=WEIGHTS_FILE):
        raw_X_dicts = []
        raw_y_next_words = []
        step_X_dicts = []
        step_y_labels = []
        eq_counts = Counter()

        for item in dataset:
            question, eq, values, _ans = item[:4]
            source_tag = item[4] if len(item) >= 5 else "MAWPS"

            if not is_pure_var_formula(eq):
                continue
            if is_self_canceling_or_invalid_duplicate(eq, num_vars_in_question=len(values), allow_derived_dup=True):
                continue

            eq_counts[eq] += 1

            for s_feat, s_lbl in extract_unified_step_samples(question, eq, values):
                step_X_dicts.append(s_feat)
                step_y_labels.append(s_lbl)

            base_q_words = tokenize_question(question)
            for feat_d, nw_tok in self._collect_formula_step_samples(base_q_words, eq, num_vars=len(values)):
                raw_X_dicts.append(feat_d)
                raw_y_next_words.append(nw_tok)

            if source_tag == "MAWPS":
                if len(values) == 2 and max_variable_index(eq) == 2:
                    sw_map = {"var1": "var2", "var2": "var1"}
                    sw_words, sw_eq = permute_question_words_and_eq(base_q_words, eq, sw_map)
                    eq_counts[sw_eq] += 1
                    for feat_d, nw_tok in self._collect_formula_step_samples(sw_words, sw_eq, num_vars=2):
                        raw_X_dicts.append(feat_d)
                        raw_y_next_words.append(nw_tok)

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
                        sh_words, sh_eq = permute_question_words_and_eq(base_q_words, eq, mapping)
                        eq_counts[sh_eq] += 1
                        for feat_d, nw_tok in self._collect_formula_step_samples(sh_words, sh_eq, num_vars=3):
                            raw_X_dicts.append(feat_d)
                            raw_y_next_words.append(nw_tok)

                elif len(values) == 3 and max_variable_index(eq) == 3:
                    vars_3 = ("var1", "var2", "var3")
                    for p in permutations(vars_3):
                        if p == vars_3:
                            continue
                        p_map = dict(zip(vars_3, p))
                        perm_words, perm_eq = permute_question_words_and_eq(base_q_words, eq, p_map)
                        eq_counts[perm_eq] += 1
                        for feat_d, nw_tok in self._collect_formula_step_samples(perm_words, perm_eq, num_vars=3):
                            raw_X_dicts.append(feat_d)
                            raw_y_next_words.append(nw_tok)

                        p_q, p_eq2, p_vals = permute_question_vars(question, eq, values, p_map)
                        for s_feat, s_lbl in extract_unified_step_samples(p_q, p_eq2, p_vals):
                            step_X_dicts.append(s_feat)
                            step_y_labels.append(s_lbl)

        self.equations = [eq for eq, _cnt in sorted(eq_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:400]]
        self.eq_to_id = {eq: i for i, eq in enumerate(self.equations)}

        word_df = Counter()
        for f_dict in raw_X_dicts:
            for w_key in f_dict.keys():
                word_df[w_key] += 1

        valid_words_sorted = [
            w for w, cnt in sorted(word_df.items(), key=lambda kv: (-kv[1], kv[0]))
            if cnt >= MIN_WORD_DF
        ][:MAX_GLOBAL_WORDS]
        allowed_words = set(valid_words_sorted)

        X_dicts = [
            {k: v for k, v in fd.items() if k in allowed_words}
            for fd in raw_X_dicts
        ]
        y_labels = raw_y_next_words
        del raw_X_dicts, word_df, allowed_words
        gc.collect()

        X_csr, vocab_words = _dicts_to_csr(X_dicts)
        del X_dicts
        gc.collect()

        classes, coef_mat, intercept_vec = _fit_multinomial_logreg(X_csr, y_labels, C=0.6, max_iter=1500)

        self.words = vocab_words
        self.word_to_id = {w: i for i, w in enumerate(self.words)}
        self.num_words = len(self.words)

        self.next_words = classes
        self.next_word_to_id = {nw: i for i, nw in enumerate(self.next_words)}
        self.num_next_words = len(self.next_words)

        M, W, H1, H2 = self.num_next_words, self.num_words, self.h1, self.h2
        N = X_csr.shape[0]

        self.bow_w = coef_mat.reshape(M * W).copy()
        self.bow_b = intercept_vec.copy()

        K = min(TOP_K_CANDIDATES, M)
        y_ids = np.array([self.next_word_to_id[y] for y in y_labels], dtype=np.int32)

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
        del base_logits
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
        rep_nw_ids = np.repeat(cands.reshape(-1), cand_row_lengths)
        pair_indices = rep_nw_ids * W + rep_word_ids

        active_pids, compact_col_indices = np.unique(pair_indices, return_inverse=True)
        del rep_word_ids, rep_nw_ids, pair_indices, X_bin, X_csr
        gc.collect()

        A = len(active_pids)
        M_active_csr = sparse.csr_matrix(
            (np.ones(total_nnz, dtype=np.float32), compact_col_indices.astype(np.int32), M_indptr),
            shape=(N * K, A),
            dtype=np.float32,
        )

        n_comp = max(1, min(H1, M - 1, W - 1))
        U_nw, V_w = _compute_truncated_svd(coef_mat, n_comp)
        del coef_mat
        gc.collect()

        rng = np.random.default_rng(42)
        act_nw = active_pids // W
        act_w = active_pids % W
        w1_active = rng.normal(0.0, 0.01, size=(A, H1)).astype(np.float32)
        for h in range(U_nw.shape[1]):
            u_norm = U_nw[:, h] / (np.linalg.norm(U_nw[:, h]) + 1e-6)
            w1_active[:, h] += 0.15 * (u_norm[act_nw] * V_w[h, act_w]) + 0.05 * V_w[h, act_w]

        self.b1 = np.full((M, H1), 0.25, dtype=np.float32)
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

        self.weight1 = np.zeros((M * W, H1), dtype=np.float32)
        self.weight1[active_pids] = w1_active

        A_step = len(self.step_actions)
        if step_X_dicts:
            X_step_csr, step_vocab = _dicts_to_csr(step_X_dicts)
            self.step_words = step_vocab
            self.step_word_to_id = {w: i for i, w in enumerate(self.step_words)}
            W_step = len(self.step_words)

            step_classes, step_coef, step_inter = _fit_multinomial_logreg(X_step_csr, step_y_labels, C=0.6, max_iter=1500)

            self.step_w = np.zeros((A_step, W_step), dtype=np.float32)
            self.step_b = np.full(A_step, -2.0, dtype=np.float32)

            for idx_c, cls_name in enumerate(step_classes):
                act_id = self.step_act_to_id[str(cls_name)]
                self.step_w[act_id] = step_coef[idx_c].astype(np.float32)
                self.step_b[act_id] = float(step_inter[idx_c])

            del X_step_csr, step_X_dicts, step_y_labels
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

        sparse_bow_w = {
            str(int(pid)): round(float(self.bow_w[pid]), 5)
            for pid in np.where(np.abs(self.bow_w) > 1e-4)[0]
        }

        flat_step_w = self.step_w.reshape(-1)
        sparse_step_w = {
            str(int(pid)): round(float(flat_step_w[pid]), 5)
            for pid in np.where(np.abs(flat_step_w) > 1e-4)[0]
        }

        save_data = {
            "format_version": "step_scratchpad_nnue_v13",
            "h1": self.h1,
            "h2": self.h2,
            "beam_width": self.beam_width,
            "next_words": self.next_words,
            "words": self.words,
            "equations": self.equations,
            "b1": [[round(float(x), 5) for x in row] for row in self.b1],
            "W2": [[round(float(x), 5) for x in row] for row in self.W2],
            "b2": [round(float(x), 5) for x in self.b2],
            "W3": [round(float(x), 5) for x in self.W3],
            "bow_b": [round(float(x), 5) for x in self.bow_b],
            "bow_w": sparse_bow_w,
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
            f"文順・中間変数ステップ生成 (step1 = op(arg1, arg2)) + NNUE 重みファイルを保存しました: {weights_path} "
            f"(next_word数: {M}, 語彙数: {W}, ステップ特徴数: {len(self.step_words)})"
        )

    def load(self, weights_path=WEIGHTS_FILE):
        if not os.path.exists(weights_path):
            return False
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("format_version") != "step_scratchpad_nnue_v13":
                return False

            self.h1 = int(data["h1"])
            self.h2 = int(data["h2"])
            self.beam_width = int(data.get("beam_width", BEAM_WIDTH))

            self.next_words = data["next_words"]
            self.next_word_to_id = {nw: i for i, nw in enumerate(self.next_words)}
            self.num_next_words = len(self.next_words)

            self.words = data["words"]
            self.word_to_id = {w: i for i, w in enumerate(self.words)}
            self.num_words = len(self.words)

            self.equations = data.get("equations", [])
            self.eq_to_id = {eq: i for i, eq in enumerate(self.equations)}

            M, W = self.num_next_words, self.num_words
            self.b1 = np.array(data["b1"], dtype=np.float32)
            self.W2 = np.array(data["W2"], dtype=np.float32)
            self.b2 = np.array(data["b2"], dtype=np.float32)
            self.W3 = np.array(data["W3"], dtype=np.float32)
            self.bow_b = np.array(data["bow_b"], dtype=np.float32)

            self.bow_w = np.zeros(M * W, dtype=np.float32)
            for pid_str, val in data["bow_w"].items():
                self.bow_w[int(pid_str)] = float(val)

            self.weight1 = np.zeros((M * W, self.h1), dtype=np.float32)
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

    def score_all_next_words(self, question_words, prefix_tokens, num_vars=None):
        M, W = self.num_next_words, self.num_words
        feat_dict = build_autoregressive_feature_dict(question_words, prefix_tokens, num_vars=num_vars)
        active_word_ids = [self.word_to_id[w] for w in feat_dict.keys() if w in self.word_to_id]

        if active_word_ids:
            active_w_arr = np.array(active_word_ids, dtype=np.int32)
            pair_ids_mat = (np.arange(M, dtype=np.int32)[:, None] * W) + active_w_arr[None, :]
            bow_scores = self.bow_b + self.bow_w[pair_ids_mat].sum(axis=1)
            z1 = self.weight1[pair_ids_mat].sum(axis=1) + self.b1
        else:
            bow_scores = self.bow_b.copy()
            z1 = self.b1.copy()

        a1 = np.clip(z1, 0.0, 1.0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.clip(z2, 0.0, 1.0)
        nnue_scores = a2 @ self.W3
        return bow_scores + nnue_scores

    def _step_log_probs(self, question_words, prefix_tokens, valid_next, num_vars=None):
        if len(valid_next) == 1:
            only_nw = next(iter(valid_next))
            return {only_nw: 0.0}
        raw_scores = self.score_all_next_words(question_words, prefix_tokens, num_vars=num_vars)
        valid_list = sorted(nw for nw in valid_next if nw in self.next_word_to_id)
        if not valid_list:
            return {}
        logits = np.array([float(raw_scores[self.next_word_to_id[nw]]) for nw in valid_list], dtype=np.float64)
        max_l = float(np.max(logits))
        log_denom = max_l + float(np.log(np.sum(np.exp(logits - max_l))))
        return {nw: float(logits[i] - log_denom) for i, nw in enumerate(valid_list)}

    def generate_formula_tokens_autoregressive(self, question_words, num_vars):
        next_words_set = set(self.next_words)
        candidates = {}

        target_ks = [num_vars]
        if num_vars >= 3:
            target_ks.append(num_vars - 1)
        if num_vars <= 5:
            target_ks.append(num_vars + 1)

        for t_k in target_ks:
            allow_rep = (t_k > num_vars)
            beam = [(0.0, 0, [])]
            for _step in range(MAX_FORMULA_TOKENS):
                next_beam = []
                for cum_lp, dec_cnt, prefix in beam:
                    if prefix and prefix[-1] == EOF_TOKEN:
                        continue
                    valid_next = get_valid_next_tokens(
                        prefix, next_words_set, num_vars, target_k=t_k, allow_var_repeat=allow_rep
                    )
                    if not valid_next:
                        continue
                    logp_map = self._step_log_probs(question_words, prefix, valid_next, num_vars=num_vars)
                    is_dec = 1 if len(valid_next) > 1 else 0
                    for nw, lp in logp_map.items():
                        new_prefix = prefix + [nw]
                        new_lp = cum_lp + (lp if is_dec else 0.0)
                        new_dec = dec_cnt + is_dec
                        if nw == EOF_TOKEN:
                            norm_eq = tokens_to_formula(new_prefix)
                            if norm_eq is not None:
                                cal_sc = new_lp / max(new_dec, 1)
                                if norm_eq not in candidates or cal_sc > candidates[norm_eq]:
                                    candidates[norm_eq] = cal_sc
                        else:
                            next_beam.append((new_lp, new_dec, new_prefix))
                if not next_beam:
                    break
                next_beam.sort(key=lambda item: (item[0] / max(item[1], 1), item[2]), reverse=True)
                beam = next_beam[:TOKEN_BEAM_WIDTH]

        return candidates

    def score_formula_by_next_words(self, question_words, eq, num_vars):
        f_tokens = formula_to_tokens(eq)
        if not f_tokens:
            return -10.0
        next_words_set = set(self.next_words)
        op_tokens = [t for t in f_tokens if is_operand_tok(t)]
        t_k = len(op_tokens)
        allow_rep = len(op_tokens) > len(set(op_tokens))

        total_lp = 0.0
        dec_cnt = 0
        for z, nw in enumerate(f_tokens):
            if nw not in self.next_word_to_id:
                if is_var_tok(nw):
                    continue
                return -10.0
            prefix = f_tokens[:z]
            valid_next = get_valid_next_tokens(
                prefix, next_words_set, num_vars, target_k=t_k, allow_var_repeat=allow_rep
            )
            if nw not in valid_next:
                valid_next = set(valid_next) | {nw}
            if len(valid_next) <= 1:
                continue
            logp_map = self._step_log_probs(question_words, prefix, valid_next, num_vars=num_vars)
            total_lp += logp_map.get(nw, -6.0)
            dec_cnt += 1
        return total_lp / max(dec_cnt, 1)

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
        """
        文順・中間変数ステップ生成 (`step1 = op(arg1, arg2)` Scratchpad 方式):
        1. 問題文の文順 (sent_id = 0, 1, ...) に沿って各ステップの中間変数 `step1, step2, ...` を生成する。
        2. 基準変数 (例: 家 $80,000 の `var1` や、比較連鎖の基準エンティティ `Tom = 10`, `Jerry = step1 = 20`) を
           Scratchpad 上に保持して次のステップでも再利用可能にする。
        """
        var_nodes, q_ctx = extract_rich_question_info(masked_question, values)
        sorted_vars = sorted(var_nodes.keys(), key=lambda v: var_nodes[v]["var_order"])
        initial_board = [var_nodes[v] for v in sorted_vars]

        if len(initial_board) < 2:
            return {}, var_nodes, q_ctx

        beam = [(0.0, initial_board, 0)]
        discovered_formulas = {}
        q_target = q_ctx["q_target_nouns"]

        max_steps = len(initial_board) + 2
        for step_idx in range(1, max_steps + 1):
            next_candidates = []
            seen_states = set()

            for cum_score, board, keep_count in beam:
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

                        prox_bonus = 0.0
                        if nodeL.get("clause_id", -1) >= 0 and nodeL.get("clause_id") == nodeR.get("clause_id"):
                            prox_bonus += 0.55
                        elif nodeL.get("sent_id", -1) >= 0 and nodeL.get("sent_id") == nodeR.get("sent_id"):
                            prox_bonus += 0.35
                        elif abs(nodeL["var_order"] - nodeR["var_order"]) == 1:
                            prox_bonus += 0.25

                        for act in ACTIVE_ACTIONS:
                            res_out = apply_step_operation(nodeL, nodeR, act, board, q_ctx, step_idx=step_idx)
                            if res_out is None:
                                continue
                            new_node, dim_bonus = res_out
                            step_score = cum_score + op_scores[act] + dim_bonus + prox_bonus

                            goal_bonus = 0.0
                            if new_node["q_match"]:
                                goal_bonus += 1.2
                            if not new_node["rate_units"] and not q_ctx["q_asks_rate"]:
                                goal_bonus += 0.8
                            elif new_node["rate_units"] and not q_ctx["q_asks_rate"]:
                                goal_bonus -= 1.8

                            unused_penalty = 0.0
                            for rn in rem_nodes:
                                if rn.get("is_distractor_cand"):
                                    continue
                                if rn["expr"] in new_node["vars_used"]:
                                    continue
                                if rn["q_match"]:
                                    unused_penalty += 1.5
                                if _are_units_compatible(rn["units"], new_node["units"], q_target) and (
                                    rn["rate_units"] or rn["cues_sub"] or rn["cues_add"]
                                ):
                                    unused_penalty += 3.8
                                elif rn["cues_sub"] or rn["cues_add"] or rn["rate_units"] or rn["is_mult_scale"] or rn["is_frac_div"]:
                                    unused_penalty += 2.4
                                else:
                                    unused_penalty += 0.8

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
                                    next_candidates.append((step_score, new_board, keep_count))

                            # Scratchpad 参照保持遷移 (基準量 nodeL を Scratchpad 上に残して次のステップでも再利用可能にする)
                            has_entity_link = bool(
                                nodeL.get("entity_nouns", set()) & nodeR.get("ref_entities", set())
                            )
                            later_node_refers_to_L = any(
                                (rn.get("is_mult_scale") or rn.get("is_frac_div"))
                                and bool(nodeL.get("entity_nouns", set()) & rn.get("ref_entities", set()))
                                for rn in rem_nodes
                            )
                            is_valid_base_pair = (
                                (nodeL["depth"] == 0 and nodeR["depth"] == 0)
                                or (
                                    nodeL["depth"] <= 1
                                    and nodeR["depth"] == 0
                                    and q_ctx.get("wants_combine_all")
                                    and (
                                        nodeR.get("is_mult_scale")
                                        or nodeR.get("is_frac_div")
                                        or nodeR.get("is_comp_more")
                                        or bool((nodeR["cues_add"] | nodeR["cues_sub"]) & {"more", "less", "fewer", "than"})
                                    )
                                )
                            ) and (
                                nodeR["var_order"] == nodeL["var_order"] + 1
                                or (has_entity_link and nodeR["var_order"] > nodeL["var_order"])
                            )

                            can_keep_scale = (
                                is_valid_base_pair
                                and act in ("MUL", "DIV_LR")
                                and (
                                    nodeR.get("is_frac_div")
                                    or nodeR.get("is_mult_scale")
                                    or q_ctx.get("wants_combine_all")
                                )
                            )
                            can_keep_comp = (
                                is_valid_base_pair
                                and act in ("ADD", "SUB_LR")
                                and q_ctx.get("wants_combine_all")
                                and (
                                    nodeR.get("is_comp_more")
                                    or bool((nodeR["cues_add"] | nodeR["cues_sub"]) & {"more", "less", "fewer", "than"})
                                )
                            )
                            can_keep_entity_base_for_later = (
                                nodeL["depth"] == 0
                                and nodeR["depth"] == 0
                                and act == "ADD"
                                and later_node_refers_to_L
                            )

                            if keep_count < 2 and (can_keep_scale or can_keep_comp or can_keep_entity_base_for_later):
                                keep_board = [nodeL, new_node] + rem_nodes
                                keep_sig = tuple(sorted(n["expr"] for n in keep_board))
                                if keep_sig not in seen_states:
                                    seen_states.add(keep_sig)
                                    next_candidates.append((step_score - 0.10, keep_board, keep_count + 1))

            if not next_candidates:
                break
            next_candidates.sort(key=lambda x: x[0], reverse=True)
            beam = next_candidates[: self.beam_width]

        return discovered_formulas, var_nodes, q_ctx

    def predict(self, masked_question, values):
        question_words, same_u, diff_u, q_match = analyze_question_structure(masked_question)
        num_vars = len(values)

        beam_results, var_nodes, q_ctx = self.beam_stack_search(masked_question, values)
        ar_candidates = self.generate_formula_tokens_autoregressive(question_words, num_vars)

        all_candidate_eqs = sorted(
            set(ar_candidates.keys())
            | set(beam_results.keys())
            | {eq for eq in self.equations[:80] if max_variable_index(eq) <= num_vars}
        )

        best_eq = None
        best_score = float("-inf")
        best_ans = None

        for eq in all_candidate_eqs:
            if max_variable_index(eq) > num_vars:
                continue

            ok, ast_bonus, final_val = check_ast_and_dimensional_bonus(
                eq, values, same_u, diff_u, q_match, var_nodes=var_nodes, q_ctx=q_ctx
            )
            if not ok:
                continue

            if eq in ar_candidates:
                ar_logp = ar_candidates[eq]
            else:
                ar_logp = self.score_formula_by_next_words(question_words, eq, num_vars=num_vars)

            beam_bonus = 0.0
            if eq in beam_results:
                beam_bonus = BEAM_SCORE_WEIGHT * beam_results[eq][0]

            total_score = (AR_SCORE_WEIGHT * ar_logp) + ast_bonus + beam_bonus

            if total_score > best_score:
                best_score = total_score
                best_eq = eq
                best_ans = final_val

        return best_eq, best_score, best_ans


# ==========================================
# 8. データ読み込み関数 ＆ GSM8K <<expr=res>> Scratchpad ステップ抽出パーサ
# ==========================================
def parse_gsm8k_solution_to_equation(masked_q, values, sol_text, true_ans):
    calcs = re.findall(r"<<([^=<>]+)=([^=<>]+)>>", sol_text)
    if not calcs or len(values) < 2:
        return None

    env = [(v, f"var{i + 1}", frozenset([f"var{i + 1}"])) for i, v in enumerate(values)]
    num_tok_re = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")

    for expr_raw, res_raw in calcs:
        expr_clean = expr_raw.strip().replace(",", "")
        res_clean = res_raw.strip().replace(",", "")
        if any(math.isclose(v, 2.0, abs_tol=1e-6) for v in values) and not any(math.isclose(v, 0.5, abs_tol=1e-6) for v in values):
            expr_clean = re.sub(r"\*\s*0?\.5\b", "/2", expr_clean)
        expr_clean = re.sub(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*/\s*100\b",
            lambda m: f"{float(m.group(1)) / 100.0:g}"
            if any(math.isclose(v, float(m.group(1)) / 100.0, abs_tol=1e-6) for v in values)
            else m.group(0),
            expr_clean,
        )
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
            if is_pure_var_formula(expr_s) and not is_self_canceling_or_invalid_duplicate(
                expr_s, num_vars_in_question=len(values), allow_derived_dup=True
            ):
                return expr_s

    for i in range(len(env) - 1, -1, -1):
        for j in range(len(env) - 1, -1, -1):
            if i == j:
                continue
            v_i, e_i, u_i = env[i]
            v_j, e_j, u_j = env[j]
            if len(u_i | u_j) < 2:
                continue
            for op_sym, cand_v in (
                ("+", v_i + v_j),
                ("-", v_i - v_j),
                ("*", v_i * v_j),
                ("/", v_i / v_j if abs(v_j) > 1e-9 else -1e9),
            ):
                if math.isclose(cand_v, true_ans, abs_tol=1e-3):
                    try:
                        norm_e = normalize_equation(f"({e_i} {op_sym} {e_j})", prefix=False)
                        if is_pure_var_formula(norm_e) and not is_self_canceling_or_invalid_duplicate(
                            norm_e, num_vars_in_question=len(values), allow_derived_dup=True
                        ):
                            return norm_e
                    except Exception:
                        continue
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
                    if is_pure_var_formula(eq) and math.isclose(evaluate_with_eval(eq, values), answer, abs_tol=1e-5):
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
                    if not (2 <= len(values) <= 8):
                        continue
                    eq = parse_gsm8k_solution_to_equation(masked_q, values, sol_part, true_ans)
                    if eq is not None and math.isclose(evaluate_with_eval(eq, values), true_ans, abs_tol=1e-4):
                        dataset.append((masked_q, eq, values, true_ans, "GSM8K"))
                        gsm_added += 1
                except Exception:
                    continue
        if gsm_added > 0:
            print(f"GSM8K学習データから {gsm_added} 問のScratchpadステップ数式を抽出して統合しました。")

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
    print("文順・中間変数ステップ生成 (step1 = op(arg1, arg2)) + NNUE モデルの重みファイルを初回学習して保存します...")
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
        evaluate_dataset(_GLOBAL_MODEL, ma_train_data, "文順Scratchpad(step=op(a,b))+NNUE - 学習データ(MAWPS/ASDiv)")
        evaluate_dataset(_GLOBAL_MODEL, test_data,     "文順Scratchpad(step=op(a,b))+NNUE - SVAMPテストデータ")

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
    sample_josh = (
        "Josh decides to try flipping a house. "
        "He buys a house for $80,000 and then puts in $50,000 in repairs. "
        "This increased the value of the house by 150%. "
        "How much profit did he make?"
    )

    print("=== solve(question_text) ＆ Scratchpad ステップ生成テスト ===")
    for label, q_sample, expected in (
        ("問題1", sample_math, 255),
        ("問題3", sample_math3, 39),
        ("問題5 (GSM8K Janet)", sample_math5, 18),
        ("問題5b (GSM8K Josh 住宅転売)", sample_josh, 70000),
    ):
        mq, vals = mask_numbers_in_question(q_sample)
        p_eq, p_sc, _ = _GLOBAL_MODEL.predict(mq, vals)
        steps_trace = format_scratchpad_trace(p_eq, vals)
        print(f"{label}: {q_sample}")
        print(f"  -> 算数判定: {is_math_word_problem(q_sample)}")
        print(f"  -> 抽出数値: { {f'var{i+1}': v for i, v in enumerate(vals)} }")
        print(f"  -> 中間変数ステップ列: {steps_trace}")
        print(f"  -> 選択された数式: {p_eq} (score: {p_sc:.2f})")
        print(f"  -> solve() の返り値: {solve(q_sample)} (正解数値: {expected})\n")

    print(f"問題2: {sample_quiz}")
    print(f"  -> 算数判定: {is_math_word_problem(sample_quiz)}")
    print(f"  -> solve(check_is_math=True) の返り値: {solve(sample_quiz, check_is_math=True)}\n")

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
                wrong_examples.append((q_text, true_eq, true_ans, pred_val, p_eq, p_sc, vals))

        print(f"[{ds_name} 外部問題集テスト: {extra_correct} / {eval_count} ({extra_correct / max(eval_count, 1) * 100:.2f}%)]")
        if wrong_examples:
            w_text, w_true_eq, w_true_ans, w_pred_ans, w_pred_eq, w_score, w_vals = wrong_examples[0]
            ans_disp = int(round(w_true_ans)) if math.isclose(w_true_ans, round(w_true_ans), abs_tol=1e-6) else w_true_ans
            print(f"問題{prob_idx} ({ds_name} より誤答例): {w_text}")
            print(f"  -> 算数判定: {is_math_word_problem(w_text)}")
            print(f"  -> 中間変数ステップ列: {format_scratchpad_trace(w_pred_eq, w_vals) if w_pred_eq else []}")
            print(f"  -> 予測された数式: {w_pred_eq} (score: {w_score:.2f})")
            print(f"  -> solve() の返り値 (誤答): {w_pred_ans}")
            print(f"  -> 本来の正解数値: {ans_disp} (正解式/解法: {w_true_eq})\n")
            prob_idx += 1


if __name__ == "__main__":
    main()
