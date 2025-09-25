import json

counter = 0

looked = dict()
meta = ""

with open("./metadata2.txt") as f:#getdata(≠Quizbot/quiz)のほうのmetadata2.txt
    for line in f:
        meta = meta + line
meta = meta.split()


maxword=40000

for i in range(len(meta)):
    looked[str(meta[i]).lower()] = 1


def clean_filename(title):
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c for c in title if c in valid_chars).strip()

def save_article(title, summary, index):
    valid_title = clean_filename(title)
    valid_title = valid_title.replace(" ", "")
    valid_title = valid_title.replace(".", "")
    if not valid_title:
        valid_title = f"article_{index}"
    filename = f"{valid_title}.txt"

    if str(valid_title).lower() not in looked:
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"{summary}")
            looked[str(valid_title).lower()] = 1
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving file {filename}: {e}")

with open('enwiki_namespace_0_13.jsonl') as f:
    for line in f:
        counter += 1
        obj = json.loads(line)
        print("Count:" + str(counter)+"/"+str(maxword))
        #print("Name:"+str(obj.get("name", "")))

        # 冒頭記事（AbstractかIntroductionの最初の段落があればそれを表示）
        intro_text = ""

        # まずAbstract
        for section in obj.get("sections", []):
            if section.get("name") == "Abstract":
                # Abstractは通常has_partsか直でtext
                if "has_parts" in section:
                    for part in section["has_parts"]:
                        if part.get("type") == "paragraph":
                            intro_text = part.get("value", "")
                            break
                else:
                    intro_text = section.get("value", "")
                break

        # AbstractがなければIntroduction
        if not intro_text:
            for section in obj.get("sections", []):
                if section.get("name", "").lower().startswith("introduction"):
                    if "has_parts" in section:
                        for part in section["has_parts"]:
                            if part.get("type") == "paragraph":
                                intro_text = part.get("value", "")
                                break
                    else:
                        intro_text = section.get("value", "")
                    break

        #print("Intro:"+str(intro_text))

        save_article(str(obj.get("name","")),str(intro_text),counter)

        if counter >= maxword:
            break
