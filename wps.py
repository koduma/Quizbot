import re
from dataclasses import dataclass, field
from typing import List, Set

# ==========================================
# 1. キーワードとデータ構造
# ==========================================
# 符号を推測するためのキーワード群
KEYWORDS = {
    '+': ["profit", "earned", "total", "sum", "more", "add", "increase"],
    '-': ["bought", "spent", "left", "cost", "pay", "paid", "remain", "sold", "reduce", "decrease"]
}

@dataclass
class ExtractedNum:
    value: float
    start_idx: int
    pref_op: str = None  # 距離から推測された「期待される符号」 (+ または -)

@dataclass
class ExprNode:
    value: float
    expr: str
    ops_used: Set[str] = field(default_factory=set)

# ==========================================
# 2. 距離ベースの数値抽出モジュール
# ==========================================
def extract_and_analyze_numbers(text: str) -> List[ExtractedNum]:
    text_lower = text.lower()
    
    # 文章中のキーワードの位置をすべてマッピング
    found_keywords = []
    for op, words in KEYWORDS.items():
        for w in words:
            for m in re.finditer(r'\b' + w + r'\b', text_lower):
                found_keywords.append((m.start(), op))
                
    # 数値の抽出と、最短距離のキーワードによる符号推測
    matches = re.finditer(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
    nums = []
    
    for m in matches:
        val = float(m.group(0).replace(',', ''))
        start_idx = m.start()
        
        pref_op = None
        min_dist = float('inf')
        
        # 最も距離の近いキーワードを探す
        for kw_idx, op in found_keywords:
            dist = abs(kw_idx - start_idx)
            if dist < min_dist:
                min_dist = dist
                pref_op = op
                
        nums.append(ExtractedNum(value=val, start_idx=start_idx, pref_op=pref_op))
        
    return nums

# ==========================================
# 3. 数式木の全探索ジェネレータ
# ==========================================
def generate_all_expressions(nodes: List[ExprNode]) -> List[ExprNode]:
    n = len(nodes)
    if n == 1:
        return [nodes[0]]
    
    results = []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            rem = [nodes[k] for k in range(n) if k != i and k != j]
            left, right = nodes[i], nodes[j]
            
            # 足し算 (+)
            res_add = ExprNode(left.value + right.value, f"({left.expr} + {right.expr})", left.ops_used | right.ops_used | {'+'})
            results.extend(generate_all_expressions(rem + [res_add]))
            
            # 引き算 (-)
            res_sub = ExprNode(left.value - right.value, f"({left.expr} - {right.expr})", left.ops_used | right.ops_used | {'-'})
            results.extend(generate_all_expressions(rem + [res_sub]))
            
            # 掛け算 (*)
            res_mul = ExprNode(left.value * right.value, f"({left.expr} * {right.expr})", left.ops_used | right.ops_used | {'*'})
            results.extend(generate_all_expressions(rem + [res_mul]))
            
            # 割り算 (/)
            if abs(right.value) > 1e-9:
                res_div = ExprNode(left.value / right.value, f"({left.expr} / {right.expr})", left.ops_used | right.ops_used | {'/'})
                results.extend(generate_all_expressions(rem + [res_div]))
                
    return results

# ==========================================
# 4. 微分を用いた AST 評価関数
# ==========================================
def evaluate_candidates(candidates: List[ExprNode], original_nums: List[ExtractedNum]) -> ExprNode:
    best_cand = None
    best_score = -float('inf')
    seen_expr = set()

    for cand in candidates:
        if cand.expr in seen_expr:
            continue
        seen_expr.add(cand.expr)
        
        score = 0.0
        
        # --- 基本ルールの評価 (算数としての美しさ) ---
        if cand.value < 0:
            score -= 1000
        if not cand.value.is_integer():
            score -= 500
        else:
            score += 100
        if cand.value == 0:
            score -= 200
        if '*' in cand.ops_used or '/' in cand.ops_used:
            score -= 10 

        # --- 距離ベース＋微分の評価 (強力なヒューリスティック) ---
        for num in original_nums:
            if num.pref_op:
                target_str = str(num.value)
                # 対象の数字を +1 動かした検証用の式を作る
                test_expr = cand.expr.replace(target_str, str(num.value + 1.0), 1)
                
                try:
                    new_val = eval(test_expr)
                    # 全体の値が増えれば「+」、減れば「-」として機能している
                    effective_op = '+' if new_val > cand.value else '-'
                    
                    # 文章から推測された符号と、数式での実際の振る舞いが一致すれば大ボーナス
                    if effective_op == num.pref_op:
                        score += 300
                    else:
                        score -= 300
                except:
                    pass

        # ベストスコアの更新
        if score > best_score:
            best_score = score
            best_cand = cand

    return best_cand, best_score

#算数文章題ならTrue、それ以外ならFalse
def check_wp(t):
    return False

# ==========================================
# 5. 実行パイプライン
# ==========================================
def solve_math_problem(text: str):
    #print(f"【問題文】\n{text}\n")
    
    extracted_nums = extract_and_analyze_numbers(text)
    
    # フィルタリング (128GBの除外など)
    #extracted_nums = [n for n in extracted_nums] 
    
    #print("抽出された数値と推測された符号:")
    #for n in extracted_nums:
        #print(f" - {n.value} (期待される演算: {n.pref_op})")
        
    initial_nodes = [ExprNode(value=n.value, expr=str(n.value)) for n in extracted_nums]
    candidates = generate_all_expressions(initial_nodes)
    
    best_cand, best_score = evaluate_candidates(candidates, extracted_nums)
    
    #print("\n=== 【推論結果】 ===")
    if best_cand:
        return best_cand.value
    else:
        return None
        #print(f"最も確からしい数式: {best_cand.expr}")
        #print(f"スコア: {best_score}")
        #print(f"★ 答え: {best_cand.value}")