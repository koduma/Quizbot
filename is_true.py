import csv
import json
import os
import re
import wps_nnue as m


def analyze_false_reason(text):
    """is_math_word_problem(text) が False になった具体的な原因を特定する"""
    t_low = text.strip().lower()
    if m.ENCYCLOPEDIA_PATTERNS.search(t_low):
        return "1_encyclopedia_pattern (知識クイズ語句にヒット)"
    if not m.MATH_QUESTION_PATTERNS.search(t_low):
        return "2_no_question_pattern (疑問詞パターン不一致)"

    scrubbed = re.sub(
        r"\b(in|from|to|since|during|year|years|war|apollo|century)\s+\d{1,4}\b",
        " ",
        t_low,
    )
    scrubbed = re.sub(r"\b(a|one)\s+dozen\b", " 12 ", scrubbed)
    scrubbed = re.sub(r"\b(half)\b", " 0.5 ", scrubbed)
    scrubbed = re.sub(r"\b(twice|double)\b", " 2 ", scrubbed)

    nums = m.NUMBER_RE.findall(scrubbed)
    w_nums = [w for w in re.findall(r"[a-z]+", scrubbed) if w in m.WORD_TO_NUM]
    if len(nums) + len(w_nums) < 2:
        return "3_fewer_than_2_nums (数字が2個未満と判定)"

    return "4_no_operation_cue (演算ヒント語なし)"


def evaluate_mawps(csv_path):
    if not os.path.exists(csv_path):
        print(f"ファイルが見つかりません: {csv_path}")
        return

    raw_ok = 0
    restored_ok = 0
    total = 0
    reasons_restored = {
        "1_encyclopedia_pattern (知識クイズ語句にヒット)": 0,
        "2_no_question_pattern (疑問詞パターン不一致)": 0,
        "3_fewer_than_2_nums (数字が2個未満と判定)": 0,
        "4_no_operation_cue (演算ヒント語なし)": 0,
    }
    false_samples = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            q = row["Question"].strip()
            nums = row["Numbers"].split()
            total += 1

            # 1) number0, number1 のまま渡した場合
            if m.is_math_word_problem(q):
                raw_ok += 1

            # 2) Numbers列の数値を埋め戻した本来の英文で渡した場合
            def repl(match):
                idx = int(match.group(1))
                return nums[idx] if idx < len(nums) else match.group(0)

            q_restored = re.sub(r"\bnumber(\d+)\b", repl, q)
            if m.is_math_word_problem(q_restored):
                restored_ok += 1
            else:
                r = analyze_false_reason(q_restored)
                reasons_restored[r] += 1
                if len(false_samples) < 5:
                    false_samples.append((r, q_restored))

    print("==========================================")
    print(" 1. mawps_asdiv.csv の算数判定 (True) 正解率")
    print("==========================================")
    print(f"  そのまま (number0形式) : {raw_ok:4d} / {total} ({raw_ok / total * 100:.2f}%)")
    print(f"  数値復元 (本来の問題文): {restored_ok:4d} / {total} ({restored_ok / total * 100:.2f}%)")
    print("\n  [数値復元後に False になった原因の内訳]")
    for k, v in reasons_restored.items():
        print(f"    - {k}: {v} 問")
    if false_samples:
        print("\n  [False と誤判定された問題例 (最大5件)]")
        for r, s in false_samples:
            print(f"    ({r}) {s}")
    print()


def evaluate_svamp(json_path):
    if not os.path.exists(json_path):
        print(f"ファイルが見つかりません: {json_path}")
        return

    svamp_ok = 0
    total = 0
    reasons = {
        "1_encyclopedia_pattern (知識クイズ語句にヒット)": 0,
        "2_no_question_pattern (疑問詞パターン不一致)": 0,
        "3_fewer_than_2_nums (数字が2個未満と判定)": 0,
        "4_no_operation_cue (演算ヒント語なし)": 0,
    }
    false_samples = []

    with open(json_path, "r", encoding="utf-8") as f:
        for item in json.load(f):
            full_q = f"{item.get('Body', '').strip()} {item.get('Question', '').strip()}".strip()
            total += 1
            if m.is_math_word_problem(full_q):
                svamp_ok += 1
            else:
                r = analyze_false_reason(full_q)
                reasons[r] += 1
                if len(false_samples) < 5:
                    false_samples.append((r, full_q))

    print("==========================================")
    print(" 2. SVAMP.json の算数判定 (True) 正解率")
    print("==========================================")
    print(f"  算数判定正解率 : {svamp_ok:4d} / {total} ({svamp_ok / total * 100:.2f}%)")
    print("\n  [SVAMPで False になった原因の内訳]")
    for k, v in reasons.items():
        print(f"    - {k}: {v} 問")
    if false_samples:
        print("\n  [False と誤判定された問題例 (最大5件)]")
        for r, s in false_samples:
            print(f"    ({r}) {s}")
    print()


def main():
    m.download_files()
    evaluate_mawps(m.TRAIN_FILE)
    evaluate_svamp(m.TEST_FILE)


if __name__ == "__main__":
    main()
