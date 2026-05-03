import os

def get_key(s):
    pos = s.rfind("@")
    if pos == -1: return ""
    return s[:pos]

def get_val(s):
    pos = s.rfind("@")
    if pos == -1: return ""
    return s[pos+1:]

def merge_dictionaries():
    # ファイルパスの定義
    new_train_path = 'train2.txt'
    new_train_num_path = 'train_num2.txt'
    new_noans_path = 'NoAns2.txt'
    new_counter_path = 'counter2.txt'

    old_train_path = 'train2_old.txt'
    old_train_num_path = 'train_num2_old.txt'
    old_noans_path = 'NoAns2_old.txt'

    # データを格納する辞書
    new_train = {}
    new_train_num = {}
    new_noans = {}
    new_counter = 1

    print("=== 辞書データの復元・マージを開始します ===")

    # 1. 現在(New)のデータを読み込む
    if os.path.exists(new_train_path):
        with open(new_train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                k = get_key(line)
                try:
                    v = int(get_val(line))
                    new_train[k] = v
                except ValueError:
                    continue # 破損行はスキップ

    if os.path.exists(new_train_num_path):
        with open(new_train_num_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    k = int(get_key(line))
                    v = get_val(line)
                    new_train_num[k] = v
                except ValueError:
                    continue

    if os.path.exists(new_noans_path):
        with open(new_noans_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                k = get_key(line)
                try:
                    v = int(get_val(line))
                    new_noans[k] = v
                except ValueError:
                    continue

    if os.path.exists(new_counter_path):
        with open(new_counter_path, 'r', encoding='utf-8') as f:
            try:
                new_counter = int(f.read().strip())
            except ValueError:
                pass

    print(f"-> 現在のデータを読み込みました (train2: {len(new_train)}件, NoAns2: {len(new_noans)}件)")

    # 2. 過去(Old)のデータから足りないものをマージする
    merged_train_count = 0
    merged_noans_count = 0

    # train2 のマージ
    if os.path.exists(old_train_path):
        with open(old_train_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                word = get_key(line)
                try:
                    old_id = int(get_val(line))
                except ValueError:
                    continue
                
                # 新しい辞書にその単語が存在しない場合のみ復元
                if word not in new_train:
                    # IDの衝突チェック
                    if old_id in new_train_num and new_train_num[old_id] != word:
                        # print(f"  [注意] ID衝突回避: '{word}' に新規ID({new_counter})を割り当てます。")
                        old_id = new_counter
                        new_counter += 1
                    
                    new_train[word] = old_id
                    new_train_num[old_id] = word
                    merged_train_count += 1
    else:
        print(f"[スキップ] {old_train_path} が見つかりません。")

    # NoAns2 のマージ
    if os.path.exists(old_noans_path):
        with open(old_noans_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                word = get_key(line)
                try:
                    count = int(get_val(line))
                except ValueError:
                    continue
                
                if word not in new_noans:
                    new_noans[word] = count
                    merged_noans_count += 1
    else:
        print(f"[スキップ] {old_noans_path} が見つかりません。")

    print(f"\n-> マージ結果: train2に {merged_train_count}件 追加, NoAns2に {merged_noans_count}件 追加しました。")

    # 3. 新しいファイルとして上書き保存する
    print("新しいデータとしてファイルを上書き保存しています...")
    
    with open(new_train_path, 'w', encoding='utf-8') as f:
        for k, v in new_train.items():
            f.write(f"{k}@{v}\n")
            
    with open(new_train_num_path, 'w', encoding='utf-8') as f:
        for k, v in new_train_num.items():
            f.write(f"{k}@{v}\n")
            
    with open(new_noans_path, 'w', encoding='utf-8') as f:
        for k, v in new_noans.items():
            f.write(f"{k}@{v}\n")
            
    # counter2の整合性を保つ
    max_id = max(new_train_num.keys()) if new_train_num else 0
    final_counter = max(new_counter, max_id + 1)
    
    with open(new_counter_path, 'w', encoding='utf-8') as f:
        f.write(f"{final_counter}\n")

    print("=== 復元・マージ処理が正常に完了しました！ ===")

if __name__ == '__main__':
    merge_dictionaries()