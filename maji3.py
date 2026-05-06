import sqlite3
import os
import time

def merge_datakun3_files():
    file1 = 'datakun3.txt'
    file2 = 'datakun3_old.txt'
    output_file = 'datakun3_new.txt'
    db_file = 'temp_merge_datakun.db'

    # 古い一時DBが残っていれば削除
    if os.path.exists(db_file):
        os.remove(db_file)

    print("=== datakun3.txt を優先し、datakun3_old.txt の差分を追記します ===")
    start_time = time.time()

    # SQLiteへの接続（省メモリ・高速書き込み設定）
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA temp_store = FILE") # メモリではなくディスクでソートする
    conn.execute("PRAGMA cache_size = -1000000")

    c = conn.cursor()
    # PRIMARY KEY を設定し、p_idとc_idのペアを一意にする
    c.execute("CREATE TABLE w_data (p_id INTEGER, c_id INTEGER, w INTEGER, PRIMARY KEY(p_id, c_id))")

    # ファイルからDBへ流し込む関数
    def load_to_db(filepath, insert_mode="IGNORE"):
        if not os.path.exists(filepath):
            print(f"[スキップ] {filepath} が見つかりません。")
            return
        print(f"[{filepath}] を読み込んでいます... (モード: {insert_mode})")
        batch = []
        count = 0
        
        # 挿入モード（REPLACE か IGNORE）を動的に切り替え
        sql = f"INSERT OR {insert_mode} INTO w_data VALUES (?,?,?)"
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) == 3:
                    try:
                        p_id = int(parts[0])
                        c_id = int(parts[1])
                        w = int(parts[2].replace('+', ''))
                        batch.append((p_id, c_id, w))
                    except ValueError:
                        continue # 破損行はスキップ
                
                # 100万件ごとにDBへ挿入
                if len(batch) >= 1000000:
                    c.executemany(sql, batch)
                    count += len(batch)
                    print(f"  ... {count} 件処理")
                    batch.clear()
        
        # 端数の処理
        if batch:
            c.executemany(sql, batch)
            count += len(batch)
            print(f"  ... {count} 件完了")
        conn.commit()

    # 1. まず最新の file1 (datakun3.txt) をDBに登録
    # ※万が一ファイル内に重複があった場合、最新の値を優先するため REPLACE を指定
    print("\n--- 最新データの取り込み ---")
    load_to_db(file1, "REPLACE")

    # 2. 次に古い file2 (datakun3_old.txt) をDBに登録
    # ※IGNORE により、既に file1 で登録された (p_id, c_id) ペアは無視され、無いものだけが追記される
    print("\n--- 旧データからの差分追記 ---")
    load_to_db(file2, "IGNORE")

    print("\np_id, c_idの昇順でソートして書き出します...")
    print("※データが巨大なため、ディスク上で安全に処理します。少しお待ちください。")
    
    # 3. 抽出とソート（合算のSUMやGROUP BYは不要）
    query = """
        SELECT p_id, c_id, w 
        FROM w_data 
        ORDER BY p_id, c_id
    """
    
    written_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        # fetchoneずつ書き出すのでRAM消費を抑制
        for row in c.execute(query):
            f.write(f"{row[0]},{row[1]},{row[2]}\n")
            written_count += 1
            if written_count % 10000000 == 0:
                print(f"  ... {written_count} 件書き出し完了")

    conn.close()
    
    # 一時DBを削除
    if os.path.exists(db_file):
        os.remove(db_file)

    print(f"\n-> 書き出し完了: 計 {written_count} 件のユニークな重みペア")

    # 4. ファイルのバックアップとリネーム
    print("ファイルのバックアップとリネームを実行します...")
    if os.path.exists(file1):
        backup_file = file1 + '.bak'
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(file1, backup_file)
        print(f"  -> 旧 {file1} を {backup_file} にバックアップしました。")
        
    os.rename(output_file, file1)
    print(f"  -> 補完済みの新データを {file1} に配置しました。")
    
    elapsed_time = time.time() - start_time
    print(f"\n=== 処理が正常に完了しました ({elapsed_time:.1f} 秒) ===")

if __name__ == '__main__':
    merge_datakun3_files()
