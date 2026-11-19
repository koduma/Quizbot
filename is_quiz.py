import os
import wps_nnue as m

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
    elif s=="@":
        sp=1
    elif s=="=":
        sp=1
    elif s=="^":
        sp=1
    elif s=="?":
        sp=1
    elif s=="`":
        sp=1
    elif s=="?":
        sp=1
    elif s=="¥":
        sp=1
    return sp

def evaluate_testcases():
    testcase_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testcase")
    
    if not os.path.exists(testcase_dir):
        print(f"ディレクトリが見つかりません: {testcase_dir}")
        return

    print("==========================================")
    print(" testcase 内の 217個のクイズ文 算数判定テスト")
    print("==========================================")

    true_count = 0
    false_count = 0

    # quiz1.txt から quiz217.txt までを順番にループ
    for i in range(1, 218):
        filename = f"quiz{i}.txt"
        filepath = os.path.join(testcase_dir, filename)

        if not os.path.exists(filepath):
            print(f"{i}問目 ({filename}): ファイルが存在しません")
            continue

        # テキストファイルを読み込む (文字コードはUTF-8を想定)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                text_tmp=""
                for k in range(len(text)):
                    if is_sp(text[k]) > 0:
                        text_tmp+=" "+text[k]+" "
                    else:
                        text_tmp+=text[k]
                text=text_tmp
        except Exception as e:
            print(f"{i}問目 ({filename}): 読み込みエラー ({e})")
            continue

        # 算数判定を実行
        #is_math = m.is_math_word_problem(text)
        is_math = m.solve(text, check_is_math=True)

        if is_math is not None:
            true_count += 1
        else:
            false_count += 1

        # 217個すべて出力
        if is_math is not None:
            print(f"{i}問目: {is_math}")

    print("==========================================")
    print(f" 判定結果まとめ: True = {true_count}問 / False = {false_count}問 (合計: {true_count + false_count}問)")
    print("==========================================")

def main():
    evaluate_testcases()

if __name__ == "__main__":
    main()
