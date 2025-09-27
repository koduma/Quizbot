import os

meta = ""
with open("./metadata2.txt", encoding="utf-8") as f:
    for line in f:
        meta += line
meta = meta.split()

wrii = ""

for l in range(len(meta)):
    title = meta[l]
    if l < 288245:
        wrii += str(title) + " "

folder = "./getdata"
for p in os.listdir(folder):
    path = os.path.join(folder, p)
    if os.path.isfile(path):
        p2 = os.path.splitext(p)[0]  # 拡張子除く
        ext = os.path.splitext(p)[1]
        if (p2 in ["sc", "check", "Genmetadata", "metadata", "metadata2"]) or ("enwiki-latest-pages-articles-multistream" in p2):
            continue
        if str(ext)!=".txt":
            continue
        try:
            with open(os.path.join(folder, p), encoding="utf-8") as f:
                pass  # ファイルの中身を見るならここで読む
            wrii += str(p2) + " "
        except Exception as e:
            print(f"Error: {e}")

print("書き込む内容:", wrii)
with open("metadata2.txt", "w", encoding="utf-8") as f:
    f.write(wrii)    
with open("./getdata2/metadata2.txt", "w", encoding="utf-8") as f:
    f.write(wrii)
with open("./getdata/metadata2.txt", "w", encoding="utf-8") as f:
    f.write(wrii)
