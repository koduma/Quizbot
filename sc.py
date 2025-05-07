import requests
import os
import time

looked = dict()
meta = ""

with open("./metadata2.txt") as f:#getdata(≠Quizbot/quiz)のほうのmetadata2.txt
    for line in f:
        meta = meta + line
meta = meta.split()

for i in range(len(meta)):
    looked[str(meta[i]).lower()] = 1

def is_likely_full_article(summary_text):
    # 改行で段落数を数える
    paragraphs = summary_text.strip().split('\n')
    # 例えば3段落以上あれば「本体らしい」と仮判定
    if len(paragraphs) >= 3 or len(summary_text) > 500:
        return True
    else:
        return False

def is_entertainment_article(title):
    """
    Check if the article belongs to entertainment categories like singer, movie, etc.
    Returns True if it is entertainment-related, False otherwise.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "categories",
        "titles": title,
        "format": "json",
        "cllimit": 500
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        page = next(iter(data["query"]["pages"].values()))
        if "categories" not in page:
            return False
        ng_keywords = [
            "singer", "musician", "film", "movie", "novel", "music", "album",
            "band", "actor", "entertainment", "drama", "tv", "theatre", "series",
            "song", "composer", "artist", "media", "idol", "comedian", "anime", "manga"
        ]
        cat_names = [c["title"].lower() for c in page["categories"]]
        return any(any(ng in cat for ng in ng_keywords) for cat in cat_names)
    except Exception as e:
        print(f"Error checking category for '{title}': {e}")
        return False  # 万一エラー時は除外しない方針

def fetch_random_article(max_retries=3):
    """
    Fetches a random article summary and title from Wikipedia using its REST API.
    Retries up to max_retries times on failure.
    Returns a tuple (title, summary). If the request fails, returns (None, None).
    """
    url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    headers = {
        "User-Agent": "YourAppName/1.0 (your.email@example.com)"
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            title = data.get("title", "")
            summary = data.get("extract", "")
            return title, summary
        except Exception as e:
            print(f"Error fetching article (try {attempt+1}/{max_retries}): {e}")
            time.sleep(2)  # Wait before retrying
    return None, None

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

def main():
    number_of_articles = 1000
    count = 0
    tries = 0
    max_tries = number_of_articles * 10  # 無限ループ防止

    while count < number_of_articles and tries < max_tries:
        tries += 1
        print(f"{count+1}/{number_of_articles} (try {tries})")
        title, summary = fetch_random_article()
        if not title or not summary:
            print("error! (fetch failed)")
            continue
        if not is_likely_full_article(summary):
            print("error! (article too short)")
            print(str(title))
            continue
        if is_entertainment_article(title):
            print(f"skip: entertainment [{title}]")
            continue
        save_article(title, summary, count+1)
        count += 1
        time.sleep(3)  # 3秒待つことでAPIの負荷を下げる

    if count < number_of_articles:
        print(f"終了: {count}記事のみ取得")

if __name__ == "__main__":
    main()
