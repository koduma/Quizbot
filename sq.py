import sqlite3
import os
import time

def sqlite_merge_weights():
    main_file = 'datakun3.txt'
    diff_file = 'datakun3_diff.txt'
    output_file = 'datakun3_new.txt'
    db_file = 'temp_merge.db'

    # 古い一時DBが残っていれば削除
    if os.path.exists(db_file):
        os.remove(db_file)

    print("=== SQLite 省メモリ・マージ処理開始 ===")
    start_time = time.time()

    # DB接続とパフォーマンスチューニング
    conn = sqlite3.connect(db_file)
    # ディスクI/Oを犠牲にしてでもRAM消費を抑え、最速で書き込む設定
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA temp_store = FILE") # RAMではなくディスクを計算領域に使う（超重要）
    conn.execute("PRAGMA cache_size = -1000000") # キャッシュを最大約1GBに制限

    c = conn.cursor()
    # テーブル作成（インデックスは挿入後に自動でかかるため、ここでは作らない）
    c.execute("CREATE TABLE w_data (p_id INTEGER, c_id INTEGER, w INTEGER)")

    def load_to_db(filepath):
        if not os.path.exists(filepath): return
        print(f"[{filepath}] をDBに取り込んでいます...")
        batch = []
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                parts = line.split(',')
                # +記号を外して数値化
                batch.append((int(parts[0]), int(parts[1]), int(parts[2].replace('+', ''))))
                
                # 100万件ごとにDBへ流し込み、Python側のメモリを解放
                if len(batch) >= 1000000:
                    c.executemany("INSERT INTO w_data VALUES (?,?,?)", batch)
                    count += len(batch)
                    print(f"  ... {count} 件挿入")
                    batch.clear()
        
        # 端数の処理
        if batch:
            c.executemany("INSERT INTO w_data VALUES (?,?,?)", batch)
            count += len(batch)
            print(f"  ... {count} 件完了")
        conn.commit()

    # 1. データの流し込み（RAMはほぼ消費しない）
    load_to_db(main_file)
    load_to_db(diff_file)

    print("\nデータをマージ(GROUP BY)してソート(ORDER BY)し、ファイルへ書き出します...")
    print("※数分〜十数分かかりますが、ディスク上で処理されるためRAMは安全です。")
    
    # 2. SQLiteに集計とソートを丸投げ
    query = """
        SELECT p_id, c_id, SUM(w) 
        FROM w_data 
        GROUP BY p_id, c_id 
        ORDER BY p_id, c_id
    """
    
    written_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        # fetchoneずつ書き出すので、ここでもRAMは消費しない
        for row in c.execute(query):
            f.write(f"{row[0]},{row[1]},{row[2]}\n")
            written_count += 1
            if written_count % 10000000 == 0:
                print(f"  ... {written_count} 件書き込み完了")

    conn.close()
    
    # 使い終わったDBを削除
    if os.path.exists(db_file):
        os.remove(db_file)

    print(f"-> 書き出し完了: 計 {written_count} 件のユニークな重みペア\n")

    # 3. ファイルのバックアップとリネーム
    print("ファイルのリネームとクリーンアップを実行します...")
    if os.path.exists(main_file):
        backup_file = main_file + '.bak'
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(main_file, backup_file)
        print(f"  -> 旧 {main_file} を {backup_file} にバックアップ。")
        
    os.rename(output_file, main_file)
    print(f"  -> 新データを {main_file} に配置。")
    
    open(diff_file, 'w').close()
    print(f"  -> {diff_file} をリセット(空)しました。")

    elapsed_time = time.time() - start_time
    print(f"\n=== マージ処理が正常に完了しました ({elapsed_time:.1f} 秒) ===")

if __name__ == '__main__':
    sqlite_merge_weights()