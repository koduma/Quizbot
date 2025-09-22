import requests
import os
import time
import random

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
    if len(paragraphs) >= 1 or len(summary_text) > 100:
        return True
    else:
        return False

def is_science_article(title):
    """
    Returns True if the article belongs to scientific categories.
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
        ok_keywords = [
            "science", "scientific", "scientist", "research", "laboratory", "technology", "engineering",
            "mathematics", "math", "statistics", "data science", "machine learning", "artificial intelligence", "ai",
            "physics", "physicist", "mechanics", "quantum", "theoretical physics", "applied mathematics",
            "chemistry", "chemist", "biochemistry", "chemical", "biology", "biologist", "bioinformatics",
            "anatomy", "physiology", "neuroscience", "cognitive science", "psychology", "sociology",
            "ecology", "environment", "ecological", "environmental", "geoscience",
            "astronomy", "astronomer", "astrophysics", "cosmology", "planetary science",
            "geology", "geologist", "geography", "geographical", "earth science",
            "oceanography", "hydrology", "meteorology", "climatology", "paleontology",
            "botany", "zoology", "genetics", "genome", "genomic", "evolutionary biology",
            "veterinary", "medicine", "medical", "doctor", "dentistry", "nursing", "pharmacy",
            "clinical", "health", "epidemiology", "immunology", "pharmacology", "toxicology",
            "agriculture", "horticulture", "forestry", "soil science", "nutrition",
            "information science", "computer science", "computing", "informatics", "cybernetics",
            "engineering", "mechanical engineering", "electrical engineering", "civil engineering", 
            "chemical engineering", "biomedical engineering", "nanotechnology", "robotics", "automation",
            "materials science", "applied science", "applied physics", "applied chemistry", "applied biology",
            "forensic science","fact", "facts", "encyclopedia", "history", "historical", "timeline", "chronology", "documentary","atlas", 
            "natural history", "scientific method", "statistics", "survey"
        ]
        cat_names = [c["title"].lower() for c in page["categories"]]
        return any(any(kw in cat for kw in ok_keywords) for cat in cat_names)
    except Exception as e:
        print(f"Error checking science category for '{title}': {e}")
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
         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.117 Safari/537.36"
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

science_categories = [
    "science", "scientific", "scientist", "research", "laboratory", "technology", "engineering",
    "mathematics", "math", "statistics", "data science", "machine learning", "artificial intelligence", "ai",
    "physics", "physicist", "mechanics", "quantum", "theoretical physics", "applied mathematics",
    "chemistry", "chemist", "biochemistry", "chemical", "biology", "biologist", "bioinformatics",
    "anatomy", "physiology", "neuroscience", "cognitive science", "psychology", "sociology",
    "ecology", "environment", "ecological", "environmental", "geoscience",
    "astronomy", "astronomer", "astrophysics", "cosmology", "planetary science",
    "geology", "geologist", "geography", "geographical", "earth science",
    "oceanography", "hydrology", "meteorology", "climatology", "paleontology",
    "botany", "zoology", "genetics", "genome", "genomic", "evolutionary biology",
    "veterinary", "medicine", "medical", "doctor", "dentistry", "nursing", "pharmacy",
    "clinical", "health", "epidemiology", "immunology", "pharmacology", "toxicology",
    "agriculture", "horticulture", "forestry", "soil science", "nutrition",
    "information science", "computer science", "computing", "informatics", "cybernetics",
    "engineering", "mechanical engineering", "electrical engineering", "civil engineering", 
    "chemical engineering", "biomedical engineering", "nanotechnology", "robotics", "automation",
    "materials science", "applied science", "applied physics", "applied chemistry", "applied biology",
    "forensic science","fact", "facts", "encyclopedia", "history", "historical", "timeline", "chronology", "documentary","atlas", 
    "natural history", "scientific method", "statistics", "survey"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.117 Safari/537.36"
}

def fetch_titles_from_category(category, limit=500):
    url = "https://en.wikipedia.org/w/api.php"
    titles = []
    cmcontinue = ""
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(500, limit - len(titles)),
            "format": "json"
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                print(f"HTTP error code: {r.status_code} for category {category}")
                break
            result = r.json()
            members = result.get("query", {}).get("categorymembers", [])
            titles.extend([m["title"] for m in members if m["ns"] == 0])
            cmcontinue = result.get("continue", {}).get("cmcontinue", "")
            if not cmcontinue:
                break
        except requests.RequestException as e:
            print(f"Request error: {e}")
            time.sleep(5)
            continue
    return titles


all_titles = []
for cat in science_categories:
    all_titles.extend(fetch_titles_from_category(cat))

def main():
    number_of_articles = 1000
    count = 0
    tries = 0
    max_tries = number_of_articles * 10  # 無限ループ防止

    while count < number_of_articles and tries < max_tries:
        tries += 1
        print(f"{count+1}/{number_of_articles} (try {tries})")

        # science_categoriesから集めた全記事タイトルリストからランダムに1つ選択
        if not all_titles:
            print("No titles loaded from categories")
            break
        title = random.choice(all_titles)

        # 選ばれたタイトルのsummaryをWikipedia REST APIで取得
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        try:
            r = requests.get(summary_url, headers={ "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.117 Safari/537.36"}, timeout=10)
            if r.status_code != 200:
                print(f"error! (summary fetch failed) [{title}]")
                continue
            data = r.json()
            summary = data.get("extract", "")
        except Exception as e:
            print(f"Exception during summary fetch: {e}")
            continue

        if not is_likely_full_article(summary):
            print(f"error! (article too short) [{title}]")
            continue

        if is_entertainment_article(title):
            print(f"skip: entertainment [{title}]")
            continue

        #if not is_science_article(title):
            #print(f"skip: not science [{title}]")
            #continue

        save_article(title, summary, count + 1)
        count += 1
        time.sleep(3)

    if count < number_of_articles:
        print(f"終了: {count}記事のみ取得")

if __name__ == "__main__":
    main()
