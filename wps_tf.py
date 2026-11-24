#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wps_nnue.py — ESR-Transformer v22.0
(Soft-Typed Entity-State Register Machine + ALiBi Cross-Attention Transformer + Relational Highway + Beam Stack Search)

外部LLM APIおよび事前学習済みモデルを一切使用せず、単一ファイル内で完結する算数文章題ソルバ:
  - 厳守インターフェース・ガワ完全維持 (is_true.py / is_quiz.py 完全互換)
  - 重みファイル: wps_nnue_ver22_weights.npz
  - Windows 11 実行保護 (UTF-8 安全出力・数値オーバーフロー保護・ベンチマーク結果ファイル保存・終了時コンソール待機)

厳守インターフェース:
  - solve(text, check_is_math=False)
  - is_math_word_problem(text)
  - download_files()
  - TRAIN_FILE, TEST_FILE
  - ENCYCLOPEDIA_PATTERNS, MATH_QUESTION_PATTERNS, NUMBER_RE, WORD_TO_NUM
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ==============================================================================
# 0. パス定数・公開グローバルオブジェクト（is_true.py / is_quiz.py 完全互換）
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_data_path(filename: str) -> str:
    if os.path.exists(filename):
        return filename
    return os.path.join(BASE_DIR, filename)


TRAIN_FILE: str = _default_data_path("mawps_asdiv.csv")
TEST_FILE: str = _default_data_path("SVAMP.json")
GSM8K_TRAIN_FILE: str = _default_data_path("gsm8k_train.jsonl")
GSM8K_TEST_FILE: str = _default_data_path("gsm8k_test.jsonl")
MODEL_WEIGHTS_FILE: str = os.path.join(BASE_DIR, "wps_nnue_ver22_weights.npz")
BENCHMARK_REPORT_FILE: str = os.path.join(BASE_DIR, "wps_nnue_ver22_benchmark.txt")

DATASET_URLS: Dict[str, str] = {
    "mawps_asdiv.csv": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/data/mawps-asdiv-a_svamp/train.csv",
    "SVAMP.json": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json",
    "gsm8k_train.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}

WORD_TO_NUM: Dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
    "twice": 2, "thrice": 3, "double": 2, "triple": 3,
    "dozen": 12, "half": 0.5, "quarter": 0.25,
}

NUMBER_RE: re.Pattern = re.compile(
    r"\b(?:number\d+|\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?)\b",
    re.IGNORECASE,
)

ENCYCLOPEDIA_PATTERNS: re.Pattern = re.compile(
    r"\b(?:"
    r"for\s+(?:10|ten)\s+points|"
    r"ftp\b|"
    r"name\s+this\s+|"
    r"identify\s+this\s+|"
    r"give\s+the\s+name\s+of|"
    r"who\s+(?:was|is|were|wrote|invented|discovered|painted|composed|directed|founded|led|assassinated|coined|formulated|governed|ruled)|"
    r"whom\s+did|"
    r"which\s+(?:country|nation|city|state|planet|moon|star|element|compound|president|monarch|king|queen|emperor|war|battle|treaty|novel|poem|play|opera|symphony|author|poet|painter|composer|scientist|philosopher|river|ocean|sea|mountain|continent|dynasty|empire|organ|disease|hormone|religion|deity|god|goddess)|"
    r"what\s+(?:country|nation|city|planet|element|president|dynasty|empire|war|battle|treaty|religion|language|instrument|philosopher|author|scientist)|"
    r"what\s+(?:is|was|are|were)\s+the\s+(?:name|title|capital|nick\s*name|motto|official\s+language|chemical\s+symbol|atomic\s+number|scientific\s+name|genus|largest\s+city|longest\s+river|highest\s+mountain|largest\s+organ)|"
    r"capital\s+(?:city\s+)?of\s+|"
    r"(?:world|civil|revolutionary|peloponnesian|punic|crimean|vietnam|korean|boer|thirty\s+years|hundred\s+years)\s+war|"
    r"nobel\s+prize|"
    r"pulitzer\s+prize|"
    r"academy\s+award|"
    r"periodic\s+table|"
    r"atomic\s+(?:number|mass|weight)|"
    r"mythology|"
    r"constellation|"
    r"photosynthesis|"
    r"mitochondri|"
    r"chromosome|"
    r"roman\s+empire|"
    r"british\s+empire|"
    r"ottoman\s+empire|"
    r"byzantine|"
    r"shakespeare|"
    r"beethoven|"
    r"mozart|"
    r"supreme\s+court|"
    r"constitutional\s+amendment|"
    r"(?:novel|poem|play|opera|symphony|concerto|sonata|painting|treatise|manifesto)\s+by\s+|"
    r"(?:written|directed|composed|painted|sculpted|invented|discovered|formulated)\s+by\s+[a-z]+|"
    r"(?:born|died|assassinated|coronated|canonized)\s+in\s+\d{3,4}|"
    r"(?:treaty|battle|siege|peace|edict|diet|council)\s+of\s+[a-z]+"
    r")\b",
    re.IGNORECASE,
)

MATH_QUESTION_PATTERNS: re.Pattern = re.compile(
    r"(?:"
    r"\bhow\s+(?:many|much|long|far|old|fast|tall|heavy|deep|wide|high|big|large|often|more|fewer|less)\b|"
    r"\bwhat\s+(?:is|was|are|were|will(?:\s+be)?|would(?:\s+be)?)\s+"
    r"(?:the\s+|his\s+|her\s+|their\s+|its\s+|a\s+|an\s+|final\s+|new\s+|current\s+|[a-z]+(?:'s|\s+'\s*s)?\s+){0,4}"
    r"(?:total|sum|difference|product|quotient|average|mean|number|amount|value|cost|price|change|profit|loss|balance|"
    r"remainder|remaining|rest|distance|length|width|height|depth|area|volume|perimeter|weight|mass|speed|rate|time|"
    r"duration|age|score|grade|ratio|percent|percentage|fraction|portion|share|part|fee|tax|tip|discount|commission|"
    r"interest|salary|wage|income|earnings|revenue|rent|bill|saving|savings|capacity|population|count)\b|"
    r"\bwhat\s+(?:fraction|percent|percentage|portion|share|part|rate|speed|price|cost|amount|number|age|time|distance)\b|"
    r"\bwhat\s+(?:did|does|do|will|would|can|could|should)\s+(?:[a-z]+(?:'s|\s+'\s*s)?\s+){1,5}"
    r"(?:pay|cost|earn|spend|make|get|receive|have|owe|need|save|take|weigh|measure|score|charge|collect|sell|buy)\b|"
    r"\b(?:find|calculate|compute|determine|work\s+out)\s+(?:the\s+|his\s+|her\s+|their\s+|how\s+|what\s+)|"
    r"\bat\s+what\s+(?:rate|speed|price|cost|age|time)\b"
    r")",
    re.IGNORECASE,
)


# ==============================================================================
# 1. 正規化・語彙辞書・算数文章題ゲートキーパー
# ==============================================================================

TENS_MAP = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
ONES_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

WORD_FRACTIONS: List[Tuple[re.Pattern, str]] = [
    # "an hour and a half" -> "1.5 hours", "two and a half" -> "2.5"
    (re.compile(r"\b(?:an|one|1)\s+([a-z]+)\s+and\s+a\s+half\b", re.I), r"1.5 \1"),
    (re.compile(r"\b(\d+)\s+and\s+a\s+half\b", re.I), r"\1.5"),
    (re.compile(r"\ba\s+half\s*-\s*(hour|day|mile|pound|cup|gallon|liter|year)\b", re.I), r"0.5 \1"),
    (re.compile(r"\b(?:one|a)\s*-\s*half\b|\bone\s+half\b", re.I), " 0.5 "),
    (re.compile(r"\b(?:one|a)\s*-\s*third\b|\bone\s+third\b|\ba\s+third\s+of\b|\ba\s+third\b", re.I), " 1/3 of "),
    (re.compile(r"\btwo\s*-\s*thirds?\b|\btwo\s+thirds?\b", re.I), " 2/3 "),
    (re.compile(r"\b(?:one|a)\s*-\s*fourth\b|\bone\s+fourth\b|\b(?:one|a)\s*-\s*quarter\b|\ba\s+quarter\s+of\b|\bone\s+quarter\s+of\b", re.I), " 0.25 of "),
    (re.compile(r"\bthree\s*-\s*fourths?\b|\bthree\s+fourths?\b|\bthree\s*-\s*quarters?\b|\bthree\s+quarters?\b", re.I), " 0.75 "),
    (re.compile(r"\b(?:one|a)\s*-\s*fifth\b|\bone\s+fifth\b|\ba\s+fifth\s+of\b", re.I), " 0.2 of "),
    (re.compile(r"\btwo\s*-\s*fifths?\b|\btwo\s+fifths?\b", re.I), " 0.4 "),
    (re.compile(r"\bthree\s*-\s*fifths?\b|\bthree\s+fifths?\b", re.I), " 0.6 "),
    (re.compile(r"\bfour\s*-\s*fifths?\b|\bfour\s+fifths?\b", re.I), " 0.8 "),
]

# 同義・上位語の部分正規化（"seashells" と "shells"、"rosebushes" と "bushes" 等の不一致を解消）
IRREGULAR_SINGULAR: Dict[str, str] = {
    "children": "child", "kids": "child", "kid": "child",
    "boys": "boy", "girls": "girl", "men": "man", "women": "woman",
    "people": "person", "persons": "person", "friends": "friend",
    "students": "student", "pupils": "student", "pupil": "student",
    "mice": "mouse", "geese": "goose", "teeth": "tooth", "feet": "foot",
    "leaves": "leaf", "halves": "half", "loaves": "loaf", "knives": "knife",
    "wives": "wife", "wolves": "wolf", "shelves": "shelf",
    "berries": "berry", "strawberries": "strawberry", "blueberries": "blueberry",
    "raspberries": "raspberry", "cherries": "cherry", "candies": "candy",
    "cookies": "cookie", "brownies": "brownie", "pies": "pie", "donuts": "donut",
    "doughnuts": "donut", "doughnut": "donut", "cupcakes": "cupcake",
    "seashells": "shell", "seashell": "shell", "shells": "shell",
    "rosebushes": "bush", "rosebush": "bush", "bushes": "bush",
    "pennies": "cent", "penny": "cent", "cents": "cent",
    "dollars": "dollar", "bucks": "dollar", "buck": "dollar",
    "quarters": "quarter", "dimes": "dime", "nickels": "nickel",
    "puppies": "puppy", "kitties": "kitty", "butterflies": "butterfly", "flies": "fly",
    "boxes": "box", "peaches": "peach", "dishes": "dish",
    "brushes": "brush", "watches": "watch", "sandwiches": "sandwich", "batches": "batch",
    "inches": "inch", "branches": "branch", "bunches": "bunch", "benches": "bench",
    "buses": "bus", "glasses": "glass", "classes": "class", "dresses": "dress",
    "tomatoes": "tomato", "potatoes": "potato", "mangoes": "mango", "mangos": "mango",
    "heroes": "hero", "fish": "fish", "fishes": "fish", "sheep": "sheep", "deer": "deer",
    "series": "series", "species": "species", "clothes": "clothing", "dozens": "dozen",
    "crunches": "crunch", "push-ups": "pushup", "pushups": "pushup", "sit-ups": "situp",
    "dvds": "dvd", "cds": "cd", "erasers": "eraser", "customers": "customer",
    "hrs": "hour", "hr": "hour", "mins": "minute", "min": "minute",
    "secs": "second", "sec": "second", "lbs": "pound", "lb": "pound",
    "oz": "ounce", "km": "kilometer", "kg": "kilogram",
}

UNIT_HYPERNYM: Dict[str, str] = {
    "boy": "person", "girl": "person", "man": "person", "woman": "person",
    "child": "person", "adult": "person", "student": "person", "teacher": "person",
    "person": "person", "friend": "person", "member": "person", "worker": "person",
    "player": "person", "guest": "person", "passenger": "person", "customer": "person",
    "apple": "fruit", "banana": "fruit", "orange": "fruit", "peach": "fruit",
    "pear": "fruit", "berry": "fruit", "strawberry": "fruit", "blueberry": "fruit",
    "cherry": "fruit", "mango": "fruit", "plum": "fruit", "grape": "fruit",
    "lemon": "fruit", "lime": "fruit", "melon": "fruit", "watermelon": "fruit", "fruit": "fruit",
    "rose": "flower", "tulip": "flower", "daisy": "flower", "lily": "flower",
    "sunflower": "flower", "orchid": "flower", "carnation": "flower", "carnations": "flower", "flower": "flower", "plant": "flower",
    "oak": "tree", "pine": "tree", "maple": "tree", "walnut": "tree", "tree": "tree",
    "dog": "animal", "cat": "animal", "puppy": "animal", "kitty": "animal", "bird": "animal",
    "stork": "animal", "sparrow": "animal", "crow": "animal", "sparrow": "animal", "robin": "animal", "bird_family": "animal", "family": "animal", "duck": "animal",
    "hen": "animal", "chicken": "animal", "rooster": "animal", "fish": "animal", "rabbit": "animal",
    "cow": "animal", "horse": "animal", "sheep": "animal", "pig": "animal", "goat": "animal", "animal": "animal",
    "shirt": "clothing", "dress": "clothing", "jacket": "clothing", "coat": "clothing",
    "sweater": "clothing", "hat": "clothing", "pant": "clothing", "sock": "clothing", "clothing": "clothing",
    "quarter": "coin", "dime": "coin", "nickel": "coin", "cent": "coin", "coin": "coin",
    "cake": "pastry", "pie": "pastry", "cookie": "pastry", "cupcake": "pastry", "muffin": "pastry", "donut": "pastry",
    "slice": "piece", "piece": "piece",
}

CATEGORY_TARGET_UNITS: Set[str] = {
    "person", "student", "child", "member", "player", "guest", "passenger", "worker",
    "fruit", "flower", "tree", "animal", "bird", "clothing", "coin", "pastry", "piece", "item",
}

TIME_UNITS: Set[str] = {"second", "minute", "hour", "day", "week", "month", "year", "morning"}

MULTIPLIER_CONTAINER_UNITS: Set[str] = {
    "box", "bag", "pack", "package", "packet", "carton", "crate", "basket", "jar",
    "bottle", "can", "tray", "row", "column", "shelf", "stack", "pile", "bunch",
    "bundle", "group", "team", "class", "bus", "car", "van", "truck", "boat", "table",
    "page", "letter", "book", "album", "folder", "day", "week", "month", "year",
    "hour", "minute", "second", "morning", "afternoon", "evening", "night",
    "time", "trip", "round", "lap", "batch", "set", "pair", "dozen", "bill",
    "roll", "sheet", "bowl", "plate", "cup", "omelet", "pie", "pizza", "cake",
    "necklace", "bracelet", "friend", "child", "boy", "girl", "person", "student", "worker",
    "chair", "spider", "dog", "cat", "bird", "cow", "chicken", "garden", "bed", "vase", "pot",
}

# 単位探索を直ちに打ち切る前置詞境界（"eats 3 for breakfast", "6 during dinner", "5 in the basket" での誤単位化を根絶！）
BOUNDARY_PREPOSITIONS: Set[str] = {
    "in", "on", "at", "for", "from", "to", "with", "by", "into", "onto",
    "during", "after", "before", "until", "under", "over", "between", "among",
    "across", "through", "without", "behind", "near", "around", "inside", "outside",
}

GROSS_EVENT_VERBS: Set[str] = {
    "bake", "cook", "buy", "sell", "eat", "drink", "give", "pick", "collect",
    "lose", "plant", "cut", "read", "write", "catch", "throw", "break", "find",
    "pack", "print", "serve", "lay",
}

VERB_LEMMA: Dict[str, str] = {
    "has": "have", "had": "have", "having": "have",
    "bought": "buy", "buys": "buy", "buying": "buy", "purchased": "buy", "purchases": "buy",
    "sold": "sell", "sells": "sell", "selling": "sell",
    "gave": "give", "gives": "give", "given": "give", "giving": "give", "handed": "give", "passed": "give", "donated": "give",
    "got": "get", "gets": "get", "gotten": "get", "getting": "get", "received": "get", "receives": "get",
    "ate": "eat", "eats": "eat", "eaten": "eat", "eating": "eat",
    "drank": "drink", "drinks": "drink", "drinking": "drink",
    "spent": "spend", "spends": "spend", "spending": "spend",
    "paid": "pay", "pays": "pay", "paying": "pay",
    "cost": "cost", "costs": "cost", "costing": "cost",
    "earned": "earn", "earns": "earn", "earning": "earn",
    "made": "make", "makes": "make", "making": "make",
    "baked": "bake", "bakes": "bake", "baking": "bake",
    "cooked": "cook", "cooks": "cook",
    "picked": "pick", "picks": "pick", "picking": "pick",
    "found": "find", "finds": "find", "finding": "find",
    "lost": "lose", "loses": "lose", "losing": "lose",
    "broke": "break", "breaks": "break", "broken": "break",
    "used": "use", "uses": "use", "using": "use",
    "took": "take", "takes": "take", "taken": "take", "taking": "take",
    "brought": "bring", "brings": "bring", "bringing": "bring",
    "caught": "catch", "catches": "catch",
    "collected": "collect", "collects": "collect",
    "planted": "plant", "plants": "plant",
    "read": "read", "reads": "read", "reading": "read",
    "wrote": "write", "writes": "write", "written": "write",
    "walked": "walk", "walks": "walk",
    "ran": "run", "runs": "run",
    "drove": "drive", "drives": "drive",
    "saved": "save", "saves": "save", "saving": "save",
    "left": "leave", "leaves": "leave", "leaving": "leave",
    "packed": "pack", "packs": "pack",
    "produced": "produce", "produces": "produce",
    "printed": "print", "prints": "print",
    "cut": "cut", "cuts": "cut", "cutting": "cut",
    "threw": "throw", "throws": "throw", "thrown": "throw",
    "added": "add", "adds": "add", "adding": "add",
    "removed": "remove", "removes": "remove", "removing": "remove",
    "organized": "organize", "organizes": "organize",
    "played": "play", "plays": "play", "playing": "play",
    "jumped": "jump", "jumps": "jump",
    "laid": "lay", "lays": "lay", "served": "serve", "serves": "serve",
    "mowed": "mow", "mows": "mow", "shoveled": "shovel", "shovels": "shovel",
}

MATH_OPERATION_CUES: re.Pattern = re.compile(
    r"\b(?:"
    r"there\s+(?:are|is|were|was)|are\s+in\s+the|is\s+in\s+the|"
    r"have|has|had|start|started|begin|began|left|remain|remaining|rest|"
    r"more|less|lesser|fewer|farther|longer|bigger|smaller|empty|total|altogether|all|combined|together|sum|difference|"
    r"each|every|per|apiece|average|equally|equal|split|share|shared|divide|divided|organized|groups|"
    r"times|twice|double|doubled|triple|tripled|thrice|half|halved|quarter|dozen|pair|"
    r"percent|percentage|discount|off|tax|tip|interest|profit|change|"
    r"bought|buy|sold|sell|gave|give|got|get|received|ate|eat|spent|spend|paid|pay|cost|costs|"
    r"earned|earn|made|make|baked|picked|found|lost|broke|used|uses|took|taken|brought|collected|planted|saved|packed|wrote|read|cut|threw|added|removed|played|jumped|served|laid"
    r")\b",
    re.IGNORECASE,
)

QUIZBOWL_CLUE_PATTERNS: re.Pattern = re.compile(
    r"\b(?:"
    r"this\s+(?:country|nation|city|island|river|author|poet|novel|play|film|painter|scientist|king|emperor|war|battle|treaty|theory|effect|law|element|compound|process|organ|disease|conflict|tragedy|comedy)|"
    r"its\s+(?:capital|author|protagonist|narrator|founder|discoverer|atomic|stable\s+isotope)|"
    r"his\s+(?:most\s+famous\s+novel|symphony|painting|treatise|reign|presidency)|"
    r"her\s+(?:most\s+famous\s+novel|reign)|"
    r"discovered\s+by\s+[a-z]+\s+[a-z]+"
    r")\b",
    re.IGNORECASE,
)


def lemmatize_noun(word: str) -> str:
    w = word.lower().strip(".,?!:;\"'()[]{}")
    if not w:
        return ""
    if w in IRREGULAR_SINGULAR:
        return IRREGULAR_SINGULAR[w]
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
    if re.search(r"(?:ch|sh|x|z|ss)es$", w):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "chaos", "cosmos", "ics", "news", "less")):
        return w[:-1]
    return w


def lemmatize_verb(word: str) -> str:
    w = word.lower().strip(".,?!:;\"'()[]{}")
    if w in VERB_LEMMA:
        return VERB_LEMMA[w]
    if len(w) > 4 and w.endswith("ed"):
        if w[-3] == w[-4]:
            return w[:-3]
        return w[:-2]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


class TextNormalizer:
    _SPACED_DEC_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")
    _SPACED_COMMA_NUM_RE = re.compile(r"(?<=\d)\s*,\s*(?=\d{3}\b)")
    _SPACED_FRAC_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
    _SPACED_APOS_RE = re.compile(r"\b([A-Za-z]+)\s*'\s*(s|t|re|ve|ll|d|m)\b", re.I)
    _SPACED_DOLLAR_RE = re.compile(r"\$\s*(number\d+|\d+(?:\.\d+)?)", re.I)
    _SPACED_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
    _COMPOUND_NOUN_RE = re.compile(
        r"\b(action\s+figures?|push\s*-\s*ups?|sit\s*-\s*ups?|jumping\s+jacks?|ice\s+creams?|video\s+games?|baseball\s+cards?|apple\s+pies?|school\s+buses|candy\s+bars?)\b",
        re.I,
    )
    _HYPHEN_NUM_RE = re.compile(
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*-\s*"
        r"(one|two|three|four|five|six|seven|eight|nine)\b",
        re.I,
    )
    _ADJ_TENS_ONES_RE = re.compile(
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+"
        r"(one|two|three|four|five|six|seven|eight|nine)\b",
        re.I,
    )
    _HYPHEN_UNIT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*-\s*([A-Za-z]+)\b")

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""
        s = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        s = s.replace("—", " - ").replace("–", "-")
        s = cls._SPACED_DEC_RE.sub(".", s)
        for _ in range(2):
            s = cls._SPACED_COMMA_NUM_RE.sub("", s)
        s = cls._SPACED_FRAC_RE.sub(r"\1/\2", s)
        s = cls._SPACED_APOS_RE.sub(r"\1'\2", s)
        s = cls._SPACED_DOLLAR_RE.sub(r"$\1", s)
        s = cls._SPACED_PERCENT_RE.sub(r"\1%", s)

        def _repl_tens(m: re.Match) -> str:
            return str(TENS_MAP[m.group(1).lower()] + ONES_MAP[m.group(2).lower()])

        s = cls._HYPHEN_NUM_RE.sub(_repl_tens, s)
        s = cls._ADJ_TENS_ONES_RE.sub(_repl_tens, s)

        def _repl_comp(m: re.Match) -> str:
            raw_c = re.sub(r"[\s-]+", "_", m.group(1).lower())
            if raw_c.startswith("push_up"):
                return "pushups" if raw_c.endswith("s") else "pushup"
            if raw_c.startswith("sit_up"):
                return "situps" if raw_c.endswith("s") else "situp"
            if raw_c.startswith("action_figure"):
                return "figures" if raw_c.endswith("s") else "figure"
            if raw_c.startswith("school_bus"):
                return "buses"
            if raw_c.startswith("apple_pie"):
                return "apple pies"
            return m.group(0)

        s = cls._COMPOUND_NOUN_RE.sub(_repl_comp, s)

        # Glued digit+word typos in MAWPS/ASDiv ("43Pokemon" -> "43 Pokemon", "Achef" kept safe, ordinals 1st/2nd/3rd/4th preserved)
        s = re.sub(r"\b(?!number\d+\b)(\d+(?:\.\d+)?)(?!(?:st|nd|rd|th)\b)([A-Za-z]{2,})\b", r"\1 \2", s)
        s = re.sub(r"\bA(chef|baker|farmer|teacher|store|waiter)\b", r"A \1", s)
        s = re.sub(r"\bhalf\s+(?:of\s+)?a\s+dozen\b", " 6 ", s, flags=re.I)

        # Clock interval normalization: "from 1:00 PM to 5:00 PM" -> "for 4 hours"
        def _repl_clock_span(m: re.Match) -> str:
            h1, m1, ap1, h2, m2, ap2 = (
                int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower(),
                int(m.group(4)), int(m.group(5) or 0), (m.group(6) or "").lower(),
            )
            if ap1 == "pm" and h1 < 12:
                h1 += 12
            elif ap1 == "am" and h1 == 12:
                h1 = 0
            if ap2 == "pm" and h2 < 12:
                h2 += 12
            elif ap2 == "am" and h2 == 12:
                h2 = 0
            t1 = h1 + m1 / 60.0
            t2 = h2 + m2 / 60.0
            diff = t2 - t1 if t2 >= t1 else (t2 + 24.0 - t1)
            if 0 < diff <= 24:
                sv = str(int(diff)) if abs(diff - round(diff)) < 1e-9 else str(round(diff, 4))
                return f"for {sv} hours"
            return m.group(0)

        s = re.sub(
            r"\bfrom\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s+to\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
            _repl_clock_span,
            s,
            flags=re.I,
        )

        for pat, rep in WORD_FRACTIONS:
            s = pat.sub(rep, s)

        s = cls._HYPHEN_UNIT_RE.sub(r"\1 \2", s)
        s = re.sub(r"\s+([.,?!:;])", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def convert_word_numbers_in_clause(cls, clause: str) -> str:
        tokens = clause.split()
        out: List[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            raw = tokens[i]
            clean = raw.lower().strip(".,?!:;\"'()[]{}")
            punct_suffix = raw[len(raw.rstrip(".,?!:;\"')]}")) :]
            punct_prefix = raw[: len(raw) - len(raw.lstrip("(\"'[{"))]

            if clean in ("a", "one") and i + 1 < n:
                nxt = tokens[i + 1].lower().strip(".,?!:;\"'()[]{}")
                if nxt == "dozen":
                    nxt_suf = tokens[i + 1][len(tokens[i + 1].rstrip(".,?!:;\"')]}")) :]
                    out.append(f"{punct_prefix}12{nxt_suf}")
                    i += 2
                    continue
                if nxt in ("pair", "couple") and not (i + 2 < n and tokens[i + 2].lower() == "of"):
                    nxt_suf = tokens[i + 1][len(tokens[i + 1].rstrip(".,?!:;\"')]}")) :]
                    out.append(f"{punct_prefix}2{nxt_suf}")
                    i += 2
                    continue

            if clean in ("a", "one") and i + 1 < n and tokens[i + 1].lower().strip(".,?!:;\"'()[]{}") in ("hundred", "thousand"):
                nxt_w = tokens[i + 1].lower().strip(".,?!:;\"'()[]{}")
                nxt_suf = tokens[i + 1][len(tokens[i + 1].rstrip(".,?!:;\"')]}")) :]
                out.append(f"{punct_prefix}{100 if nxt_w == 'hundred' else 1000}{nxt_suf}")
                i += 2
                continue

            if re.fullmatch(r"\d+(?:\.\d+)?", clean) and i + 1 < n:
                nxt = tokens[i + 1].lower().strip(".,?!:;\"'()[]{}")
                if nxt == "dozen" and not re.search(r"\b(?:per|each|every|a)\s+dozen\b", clause, re.I):
                    nxt_suf = tokens[i + 1][len(tokens[i + 1].rstrip(".,?!:;\"')]}")) :]
                    val = float(clean) * 12.0
                    sval = str(int(val)) if abs(val - round(val)) < 1e-9 else str(val)
                    out.append(f"{punct_prefix}{sval}{nxt_suf}")
                    i += 2
                    continue

            if clean in ("one", "ones"):
                prev_w = out[-1].lower().strip(".,?!") if out else ""
                next_w = tokens[i + 1].lower().strip(".,?!") if i + 1 < n else ""
                if clean == "ones" or prev_w in ("each", "every", "this", "that", "another", "which", "any", "only", "the", "then", "when", "if", "old", "new", "red", "green", "blue", "big", "small", "gets", "got") or next_w == "again":
                    out.append(raw)
                    i += 1
                    continue
                if not next_w:
                    out.append(raw)
                    i += 1
                    continue

            if clean in ONES_MAP or clean in TENS_MAP:
                total = 0
                current = ONES_MAP.get(clean, TENS_MAP.get(clean, 0))
                j = i + 1
                last_suf = punct_suffix
                while j < n:
                    w_j = tokens[j].lower().strip(".,?!:;\"'()[]{}")
                    suf_j = tokens[j][len(tokens[j].rstrip(".,?!:;\"')]}")) :]
                    if w_j == "and" and j + 1 < n:
                        w_k = tokens[j + 1].lower().strip(".,?!:;\"'()[]{}")
                        if w_k in ONES_MAP or w_k in TENS_MAP:
                            j += 1
                            continue
                        break
                    if w_j == "hundred":
                        current = max(1, current) * 100
                        last_suf = suf_j
                        j += 1
                    elif w_j == "thousand":
                        total += max(1, current) * 1000
                        current = 0
                        last_suf = suf_j
                        j += 1
                    elif w_j == "dozen":
                        current = max(1, current) * 12
                        last_suf = suf_j
                        j += 1
                    elif w_j in TENS_MAP:
                        current += TENS_MAP[w_j]
                        last_suf = suf_j
                        j += 1
                    elif w_j in ONES_MAP:
                        current += ONES_MAP[w_j]
                        last_suf = suf_j
                        j += 1
                    else:
                        break
                val_num = total + current
                out.append(f"{punct_prefix}{val_num}{last_suf}")
                i = j
                continue

            out.append(raw)
            i += 1
        return " ".join(out)


def is_math_word_problem(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False

    norm = TextNormalizer.normalize(text)
    t_low = norm.lower()

    if ENCYCLOPEDIA_PATTERNS.search(t_low) or QUIZBOWL_CLUE_PATTERNS.search(t_low):
        return False

    if not MATH_QUESTION_PATTERNS.search(t_low):
        return False

    scrubbed = re.sub(
        r"\b(?:in|from|to|since|during|year|years|war|apollo|century)\s+\d{1,4}\b",
        " ",
        t_low,
    )
    scrubbed = re.sub(r"\b(?:a|one)\s+dozen\b", " 12 ", scrubbed)
    scrubbed = re.sub(r"\b(?:a|one)\s+(?:pair|couple)\b", " 2 ", scrubbed)
    scrubbed = re.sub(r"\b(?:half|halved)\b", " 0.5 ", scrubbed)
    scrubbed = re.sub(r"\b(?:twice|double|doubled)\b", " 2 ", scrubbed)
    scrubbed = re.sub(r"\b(?:thrice|triple|tripled)\b", " 3 ", scrubbed)
    implicit_time_hits = len(
        re.findall(r"\b(?:a\s+week|per\s+day|an\s+hour|per\s+minute|a\s+year|per\s+month|quarters|dimes|nickels)\b", scrubbed)
    )

    nums = NUMBER_RE.findall(scrubbed)
    w_nums = [w for w in re.findall(r"[a-z]+", scrubbed) if w in WORD_TO_NUM]
    if len(nums) + len(w_nums) + (1 if implicit_time_hits >= 2 else 0) < 2:
        return False

    if not MATH_OPERATION_CUES.search(t_low):
        return False

    return True


# ==============================================================================
# 2. 構文・節・前置詞境界・省略単位抽出と型付きレジスタ状態モデル
# ==============================================================================

STOP_CAPITAL_WORDS: Set[str] = {
    "the", "a", "an", "and", "but", "or", "so", "yet", "nor", "there", "here",
    "if", "when", "while", "after", "before", "since", "because", "although",
    "though", "in", "on", "at", "for", "from", "to", "with", "by", "during",
    "over", "under", "between", "among", "each", "every", "some", "many", "most",
    "all", "both", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "how", "what", "which", "who", "where", "why", "find",
    "calculate", "compute", "determine", "suppose", "assume", "today", "yesterday",
    "tomorrow", "now", "then", "next", "later", "finally", "first", "second",
    "third", "last", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december", "spring",
    "summer", "autumn", "fall", "winter", "his", "her", "their", "its", "he",
    "she", "they", "it", "we", "you", "i", "this", "that", "these", "those",
    "grade", "class", "room", "chapter", "mr", "mrs", "ms", "dr", "once", "twice",
    "together", "altogether", "grandma", "grandpa", "park", "city", "town",
    "half", "quarter", "double", "triple", "thrice", "few", "several",
    "rest", "remainder", "another", "other", "others", "each", "neither", "either",
    "however", "therefore", "meanwhile", "furthermore", "moreover", "otherwise",
    "instead", "besides", "also", "plus", "normally", "usually", "actually",
    "currently", "originally", "initially", "recently", "everybody", "everyone",
    "someone", "nobody", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
}

FEMALE_NAMES_HINT: Set[str] = {
    "mary", "jessica", "jennifer", "linda", "patricia", "elizabeth", "barbara",
    "susan", "sarah", "karen", "nancy", "lisa", "betty", "margaret", "sandra",
    "ashley", "kimberly", "emily", "donna", "michelle", "dorothy", "carol", "amanda",
    "melissa", "deborah", "stephanie", "rebecca", "sharon", "laura", "cynthia",
    "kathleen", "amy", "angela", "shirley", "anna", "brenda", "pamela", "emma",
    "nicole", "helen", "samantha", "katherine", "christine", "debra", "rachel",
    "carolyn", "janet", "catherine", "maria", "heather", "diane", "ruth", "julie",
    "julia", "olivia", "joyce", "virginia", "victoria", "kelly", "lauren", "christina",
    "joan", "evelyn", "judith", "megan", "cheryl", "andrea", "hannah", "jacqueline",
    "martha", "gloria", "teresa", "ann", "sara", "madison", "frances", "kathryn",
    "janice", "jean", "abigail", "alice", "judy", "sophia", "grace", "denise",
    "amber", "doris", "marilyn", "danielle", "beverly", "isabella", "theresa",
    "diana", "natalie", "natalia", "brittany", "charlotte", "marie", "kayla", "alexis",
    "lori", "tara", "lucy", "claire", "chloe", "lily", "zoe", "mia", "ella", "jane",
    "sally", "molly", "holly", "jenny", "kate", "katie", "beth", "ellen", "rosa",
    "carmen", "wendy", "cindy", "becky", "tina", "gina", "nina", "rita", "eva",
    "weng", "nell", "suzy", "harriet",
}

ROLE_OWNERS: Set[str] = {
    "farmer", "baker", "teacher", "chef", "cook", "grocer", "florist", "carpenter",
    "painter", "tailor", "driver", "pilot", "doctor", "nurse", "librarian", "cashier",
    "waiter", "waitress", "clerk", "manager", "worker", "builder", "gardener",
    "parents", "grandparents", "mother", "father", "mom", "dad",
    "store", "shop", "bakery", "restaurant", "cafe", "booth", "factory", "company",
    "farm", "library", "school", "club", "team", "theater", "cinema", "zoo",
    "museum", "hotel", "hospital", "printer", "basket", "barn", "vase",
}

ADJECTIVE_MODIFIERS: Set[str] = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white",
    "brown", "gray", "grey", "golden", "silver", "big", "large", "huge", "giant",
    "small", "little", "tiny", "medium", "short", "long", "tall", "high", "low",
    "heavy", "light", "new", "old", "young", "fresh", "rotten", "ripe", "unripe",
    "sweet", "salty", "sour", "bitter", "spicy", "crispy", "hot", "cold", "diet",
    "damaged", "unopened", "opened", "unread", "used", "unused", "filled", "partially",
    "siamese", "house", "violet", "mini", "mixed", "separate", "standstill", "warm", "chocolate", "vanilla", "strawberry",
    "lemon", "mango", "apple", "cherry", "cheese", "pepperoni", "wooden", "plastic",
    "metal", "paper", "glass", "clean", "dirty", "broken", "whole", "full", "empty",
    "extra", "adult", "male", "female", "hardcover", "paperback", "fiction",
    "nonfiction", "regular", "special", "different", "morning", "afternoon", "evening",
    "yesterday", "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "april", "may", "june", "july", "january", "february", "march", "august",
    "september", "october", "november", "december",
}

NON_UNIT_WORDS: Set[str] = {
    "of", "the", "a", "an", "in", "on", "at", "to", "for", "from", "with", "by",
    "and", "or", "but", "if", "than", "as", "is", "are", "was", "were", "be",
    "been", "have", "has", "had", "do", "does", "did", "he", "she", "it", "they",
    "we", "you", "his", "her", "their", "its", "him", "them", "who", "which",
    "that", "there", "here", "now", "then", "left", "more", "less", "lesser", "fewer",
    "away", "out", "up", "down", "back", "off", "over", "each", "every", "per",
    "altogether", "together", "total", "combined", "already", "still", "also",
    "too", "only", "just", "about", "around", "nearly", "almost", "enough", "ones", "one",
    "currently", "today", "yesterday", "tomorrow", "daily", "weekly", "monthly",
}

CONTENT_STOPWORDS: Set[str] = NON_UNIT_WORDS | {
    "how", "many", "much", "what", "when", "where", "did", "does", "do", "will", "would",
    "can", "could", "should", "got", "get", "gets", "had", "has", "have",
}

UNIT_CONVERSIONS: Dict[Tuple[str, str], float] = {
    ("cent", "dollar"): 0.01,
    ("dollar", "cent"): 100.0,
    ("nickel", "dollar"): 0.05,
    ("dime", "dollar"): 0.10,
    ("quarter", "dollar"): 0.25,
    ("dollar", "quarter"): 4.0,
    ("dime", "dollar"): 0.10,
    ("dollar", "dime"): 10.0,
    ("minute", "hour"): 1.0 / 60.0,
    ("hour", "minute"): 60.0,
    ("second", "minute"): 1.0 / 60.0,
    ("minute", "second"): 60.0,
    ("hour", "day"): 1.0 / 24.0,
    ("day", "hour"): 24.0,
    ("day", "week"): 1.0 / 7.0,
    ("week", "day"): 7.0,
    ("month", "year"): 1.0 / 12.0,
    ("year", "month"): 12.0,
    ("week", "year"): 1.0 / 52.0,
    ("year", "week"): 52.0,
    ("inch", "foot"): 1.0 / 12.0,
    ("foot", "inch"): 12.0,
    ("foot", "yard"): 1.0 / 3.0,
    ("yard", "foot"): 3.0,
    ("gram", "kilogram"): 0.001,
    ("kilogram", "gram"): 1000.0,
    ("milliliter", "liter"): 0.001,
    ("liter", "milliliter"): 1000.0,
    ("ounce", "pound"): 1.0 / 16.0,
    ("pound", "ounce"): 16.0,
    ("dozen", "item"): 12.0,
    ("item", "dozen"): 1.0 / 12.0,
    ("day", "night"): 1.0,
    ("night", "day"): 1.0,
}


@dataclass(frozen=True)
class QuantReg:
    reg_id: int
    value: float
    owner: str = "global"
    unit: str = "item"
    per_unit: Optional[str] = None
    attr: str = ""
    stage: str = "INIT"
    action: str = "have"
    clause_idx: int = 0
    ref_owner: str = ""
    is_implicit: bool = False
    is_remaining_frac: bool = False
    depth: int = 0
    origin_mask: int = 0
    left_win: Tuple[str, ...] = ()
    right_win: Tuple[str, ...] = ()
    clause_words: Tuple[str, ...] = ()


@dataclass
class GoalSpec:
    target_owner: str = "global"
    second_owner: str = ""
    target_unit: str = "item"
    second_unit: str = ""
    per_unit: Optional[str] = None
    target_attr: str = ""
    second_attr: str = ""
    query_type: str = "COUNT"
    action_lemma: str = "have"
    question_tokens: Set[str] = field(default_factory=set)
    question_seq: Tuple[str, ...] = ()
    question_bigrams: Set[str] = field(default_factory=set)
    q_focus_modifiers: Set[str] = field(default_factory=set)
    known_entities: Set[str] = field(default_factory=set)
    has_the_rest_cue: bool = False
    unknown_some_stage: str = ""
    unknown_some_clause: int = -1


@dataclass
class ClauseInfo:
    idx: int
    raw_text: str
    norm_text: str
    tokens: List[str]
    owner: str
    verb_lemma: str
    stage_hint: str
    transfer_to: str = ""
    transfer_from: str = ""
    has_remaining_cue: bool = False
    has_unknown_some: bool = False
    cue_tokens: Set[str] = field(default_factory=set)


class ClauseRoleParser:
    _PRE_QUESTION_SPLIT_RE = re.compile(
        r",\s+(?=(?:how|what|find|calculate|determine|at\s+what|in\s+how)\b)"
        r"|(?<=[a-z0-9])\s+(?=(?:How|What)\s+(?:many|much|long|far|old|fast|big|large|tall|heavy|deep|often|is|was|are|were|did|does|do|will|would|fraction|percent)\b)"
    )

    _CLAUSE_SPLIT_RE = re.compile(
        r"(?<=[.?!;])\s+"
        r"|,\s+(?=(?:and|but|then|so|while|whereas|after|before|if|when|who|which|he|she|they|we|you)\b)"
        r"|(?<=[a-zA-Z0-9])\s+(?=(?:and\s+then|and\s+now|and\s+finally|and\s+today|and\s+yesterday|and\s+tomorrow|after\s+that|but\s+\d+)\b)"
        r"|\s+and\s+(?=(?:a|an|one|each)\s+(?:[a-z]+\s+)?[a-z]+\s+(?:has|have|had|holds?|contains?|costs?|takes?|requires?)\b)"
        r"|\s+(?=(?:and|but)\s+(?:had|has|have|now\s+has|now\s+have|now\s+(?:he|she|it|they|[a-z]+)\s+(?:has|have|had)|there\s+are|there\s+were|deleted|erased|missed|sawed|put|puts|placed|distributed|arranged|"
        r"gave|give|ate|eat|bought|buy|sold|sell|spent|spend|lost|lose|got|get|received|earned|earn|made|baked|bakes|picked|threw|cut|removed|packed|shared|divided|added|poured|paid|served|serves|"
        r"\d+\s+(?:less|lesser|fewer|more)\b|\d+\s+(?:[a-z]+\s+){1,4}(?:for|at|which\s+cost|that\s+cost))\b)",
        re.IGNORECASE,
    )

    _FRONTED_SUBORD_SPLIT_RE = re.compile(
        r"^((?:after|before|when|while|if|since)\s+[^,]+),\s+(.+)$",
        re.IGNORECASE,
    )

    _LABEL_NOISE_RE = re.compile(
        r"\b(?:grade|room|chapter|section|part|type|model|version|bus|route|highway|apollo|war)\s+\d+\b"
        r"|\b\d+(?:st|nd|rd|th)\s+(?:grade|floor|street|avenue|century|birthday)\b",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, text: str, placeholder_map: Optional[Dict[str, float]] = None) -> Tuple[List[QuantReg], List[ClauseInfo], GoalSpec]:
        norm_full = TextNormalizer.normalize(text)
        if not norm_full:
            return [], [], GoalSpec()

        norm_q_sep = cls._PRE_QUESTION_SPLIT_RE.sub(". ", norm_full)
        prelim_clauses = [c.strip() for c in cls._CLAUSE_SPLIT_RE.split(norm_q_sep) if c.strip()]

        raw_clauses: List[str] = []
        for pc in prelim_clauses:
            m_fr = cls._FRONTED_SUBORD_SPLIT_RE.match(pc)
            if m_fr:
                raw_clauses.append(m_fr.group(1).strip())
                raw_clauses.append(m_fr.group(2).strip())
            else:
                raw_clauses.append(pc)

        expanded_clauses: List[str] = []
        for rc in raw_clauses:
            m_nocomma = re.match(
                r"^((?:after|before|when)\s+(?:some|[a-z]+)\s+(?:left|ate|came|went|broke|died|joined|fell))\s+((?:he|she|it|they|there|[A-Z][a-z]+)\s+.+)$",
                rc,
                re.I,
            )
            if m_nocomma:
                expanded_clauses.append(m_nocomma.group(1).strip())
                expanded_clauses.append(m_nocomma.group(2).strip())
            else:
                expanded_clauses.append(rc)
        raw_clauses = expanded_clauses or [norm_full]

        q_idx = len(raw_clauses) - 1
        for idx in range(len(raw_clauses) - 1, -1, -1):
            if "?" in raw_clauses[idx] or MATH_QUESTION_PATTERNS.search(raw_clauses[idx]):
                q_idx = idx
                break

        body_raw = raw_clauses[:q_idx]
        question_raw = " ".join(raw_clauses[q_idx:])

        known_entities: Set[str] = set()
        for seg in raw_clauses:
            for w in seg.split():
                clean = w.strip(".,?!:;\"'()[]{}")
                if clean.endswith("'s"):
                    clean = clean[:-2]
                if len(clean) >= 2 and clean[0].isupper() and clean.lower() not in STOP_CAPITAL_WORDS:
                    lw = clean.lower()
                    if lw not in ADJECTIVE_MODIFIERS and lw not in ("dollars", "cents", "pounds", "hours", "minutes", "days", "weeks"):
                        known_entities.add(lw)

        entity_stack: List[Tuple[str, str]] = []
        clauses_info: List[ClauseInfo] = []
        registers: List[QuantReg] = []
        next_reg_id = 0

        for c_i, c_raw in enumerate(body_raw):
            c_clean_labels = cls._LABEL_NOISE_RE.sub(" ", c_raw)
            c_conv = TextNormalizer.convert_word_numbers_in_clause(c_clean_labels)
            c_info, c_regs, next_reg_id = cls._parse_body_clause(
                c_i, c_raw, c_conv, entity_stack, known_entities, next_reg_id, placeholder_map
            )
            if c_i > 0 and c_raw.lower().startswith(("and ", "but ")) and c_info.verb_lemma == "have":
                if not (set(c_info.tokens) & {"have", "has", "had", "is", "are", "was", "were"}):
                    prev_ci = clauses_info[-1]
                    c_info.verb_lemma = prev_ci.verb_lemma
                    c_info.stage_hint = prev_ci.stage_hint
                    c_info.owner = prev_ci.owner
                    for r_k in range(len(c_regs)):
                        rk = c_regs[r_k]
                        inh_st = ("COST_RATE" if rk.per_unit else "COST_OUT") if (rk.unit == "dollar" and prev_ci.verb_lemma in ("buy", "spend", "pay")) else prev_ci.stage_hint
                        if rk.stage in ("INIT", "RATE"):
                            c_regs[r_k] = QuantReg(**{**rk.__dict__, "owner": prev_ci.owner, "action": prev_ci.verb_lemma, "stage": inh_st})
            clauses_info.append(c_info)
            registers.extend(c_regs)

        for idx in range(len(registers)):
            r = registers[idx]
            if r.owner == "global" and r.clause_idx + 1 < len(clauses_info):
                c_cur = clauses_info[r.clause_idx].raw_text.lower()
                nxt_owner = clauses_info[r.clause_idx + 1].owner
                if c_cur.startswith(("after ", "before ", "when ", "while ")) and nxt_owner not in ("global", "GROUP"):
                    registers[idx] = QuantReg(**{**r.__dict__, "owner": nxt_owner})

        goal, q_regs, next_reg_id = cls._parse_question_clause(
            len(clauses_info), question_raw, entity_stack, known_entities, registers, clauses_info, next_reg_id, placeholder_map
        )
        registers.extend(q_regs)

        registers = cls._inject_implicit_domain_constants(registers, norm_full, goal, next_reg_id)
        registers = cls._propagate_missing_units(registers, clauses_info, goal)
        registers = cls._link_indefinite_unit_rates(registers, clauses_info, goal)
        registers = cls._backfill_coordinated_comparative_refs(registers)

        return registers, clauses_info, goal

    @classmethod
    def _link_indefinite_unit_rates(
        cls, regs: List[QuantReg], clauses: Sequence[ClauseInfo], goal: GoalSpec
    ) -> List[QuantReg]:
        """'A large pizza has 16 slices' や 'charges $15 to mow a lawn' のような不定冠詞レート定義を既出単位と結合"""
        problem_units = {r.unit for r in regs if r.unit not in ("item", "scalar", "percent", "dollar")}
        if goal.target_unit not in ("item", "scalar", "percent", "dollar"):
            problem_units.add(goal.target_unit)
        if not problem_units:
            return regs

        out = list(regs)
        for idx, r in enumerate(out):
            if r.per_unit is not None or r.is_implicit or r.clause_idx >= len(clauses):
                continue
            if r.stage not in ("INIT", "GAIN", "COST_OUT"):
                continue
            c_low = " ".join(clauses[r.clause_idx].tokens)
            for m_ind in re.finditer(r"\b(?:a|an|one|each|every)\s+(?:([a-z]+)\s+)?([a-z]+)\b", c_low):
                adj_w = m_ind.group(1) or ""
                u_cand = lemmatize_noun(m_ind.group(2))
                if u_cand in problem_units and u_cand != r.unit:
                    has_other_qty = (u_cand == goal.target_unit and goal.query_type == "COUNT") or any(
                        k != idx and out[k].unit == u_cand for k in range(len(out))
                    )
                    if has_other_qty:
                        new_attr = adj_w if (adj_w in ADJECTIVE_MODIFIERS and not r.attr) else r.attr
                        new_st = "COST_RATE" if (r.unit == "dollar" and r.action in ("buy", "spend", "pay")) else "RATE"
                        out[idx] = QuantReg(**{**r.__dict__, "per_unit": u_cand, "attr": new_attr, "stage": new_st})
                        break
        return out

    @classmethod
    def _backfill_coordinated_comparative_refs(cls, regs: List[QuantReg]) -> List[QuantReg]:
        out = list(regs)
        for idx in range(len(out) - 1):
            r_cur = out[idx]
            r_nxt = out[idx + 1]
            if r_cur.stage == "INIT" and r_nxt.stage in ("COMP_MORE", "COMP_LESS") and r_nxt.ref_owner and abs(r_nxt.clause_idx - r_cur.clause_idx) <= 1:
                if "more" in r_cur.right_win or "more" in r_cur.left_win:
                    out[idx] = QuantReg(**{**r_cur.__dict__, "stage": "COMP_MORE", "ref_owner": r_nxt.ref_owner, "owner": r_nxt.owner if r_cur.owner == "global" else r_cur.owner})
                elif any(w in ("less", "lesser", "fewer") for w in r_cur.right_win + r_cur.left_win):
                    out[idx] = QuantReg(**{**r_cur.__dict__, "stage": "COMP_LESS", "ref_owner": r_nxt.ref_owner, "owner": r_nxt.owner if r_cur.owner == "global" else r_cur.owner})
        return out

    @classmethod
    def _resolve_pronoun_entity(cls, pron: str, stack: Sequence[Tuple[str, str]]) -> str:
        if pron in ("her", "she"):
            for ent, g in reversed(stack):
                if g == "F":
                    return ent
        if pron in ("him", "he", "his"):
            for ent, g in reversed(stack):
                if g == "M":
                    return ent
        if pron in ("it", "its"):
            for ent, g in reversed(stack):
                if g == "N":
                    return ent
        return stack[-1][0] if stack else ""

    @classmethod
    def _detect_entities_in_tokens(
        cls, raw_tokens: List[str], known_entities: Set[str], entity_stack: List[Tuple[str, str]]
    ) -> Tuple[str, List[str]]:
        prev_stack = list(entity_stack)
        first_words = [t.lower().strip(".,?!:;\"'()[]{}") for t in raw_tokens[:6]]

        if first_words and first_words[0] in ("who", "which") and prev_stack:
            rel_subj = prev_stack[-1][0]
            return rel_subj, [rel_subj]

        if len(first_words) >= 2 and first_words[0] in ("one", "another") and lemmatize_noun(first_words[1]) in ("boy", "girl", "child", "student", "friend", "person", "man", "woman"):
            return "one_individual", []
        if len(first_words) >= 3 and first_words[0] == "then" and first_words[1] == "one":
            return "one_individual", []

        pron_subj = ""
        for fw in first_words[:4]:
            fw_base = fw[:-2] if fw.endswith("'s") else fw
            if fw_base in known_entities or fw_base in ROLE_OWNERS:
                break
            if fw in ("she", "he", "it"):
                pron_subj = cls._resolve_pronoun_entity(fw, prev_stack)
                break
            if fw in ("you", "your"):
                pron_subj = "you"
                break
            if fw in ("they", "them"):
                uniq_ents = list(dict.fromkeys(e for e, _ in prev_stack))
                if len(uniq_ents) == 1 and uniq_ents[0] in {"store", "shop", "bakery", "restaurant", "cafe", "company", "factory", "school", "club", "team", "farm", "library", "hotel", "hospital"}:
                    pron_subj = uniq_ents[0]
                else:
                    pron_subj = "GROUP"
                break

        mentioned: List[str] = []
        for idx_t, tok in enumerate(raw_tokens):
            clean = tok.strip(".,?!:;\"'()[]{}")
            has_pos = clean.endswith("'s")
            if has_pos:
                clean = clean[:-2]
            low = clean.lower()
            if not low:
                continue
            if has_pos and idx_t + 1 < len(raw_tokens):
                nxt_low = raw_tokens[idx_t + 1].lower().strip(".,?!")
                if nxt_low in ("mother", "father", "mom", "dad", "brother", "sister", "friend"):
                    comp_ent = f"{low}_{nxt_low}"
                    if comp_ent not in mentioned:
                        mentioned.append(comp_ent)
                        entity_stack.append((comp_ent, "F" if nxt_low in ("mother", "mom", "sister") else "M"))
                    continue
            prev_tok_low = raw_tokens[idx_t - 1].lower().strip(".,?!") if idx_t > 0 else ""
            if low in known_entities:
                if prev_tok_low in ("than", "to", "from", "for", "with"):
                    gender = "F" if low in FEMALE_NAMES_HINT else "M"
                    entity_stack.append((low, gender))
                    continue
                if low not in mentioned:
                    mentioned.append(low)
                    gender = "F" if low in FEMALE_NAMES_HINT else "M"
                    entity_stack.append((low, gender))
            elif low in ROLE_OWNERS and low not in STOP_CAPITAL_WORDS:
                prev2_low = raw_tokens[idx_t - 2].lower().strip(".,?!") if idx_t > 1 else ""
                if prev_tok_low in ("than", "to", "from", "in", "on", "at", "into", "of", "for") or (
                    prev_tok_low in ("his", "her", "their", "the", "a", "an") and prev2_low in ("in", "on", "at", "into", "to", "from", "for")
                ):
                    continue
                if low not in mentioned:
                    mentioned.append(low)
                    g = "N" if low in {"store", "shop", "bakery", "booth", "club", "team", "school", "basket", "barn", "factory", "printer", "vase", "restaurant", "theater"} else "M"
                    entity_stack.append((low, g))

        if pron_subj:
            return pron_subj, mentioned

        if not mentioned:
            for fw in first_words:
                if fw in ("his", "her") and prev_stack:
                    resolved = cls._resolve_pronoun_entity(fw, prev_stack)
                    if resolved:
                        return resolved, mentioned

        if len(mentioned) >= 2 and "and" in first_words[:4]:
            subj = "GROUP"
        elif mentioned:
            subj = mentioned[0]
        elif prev_stack:
            subj = prev_stack[-1][0]
        else:
            subj = "global"

        return subj, mentioned

    @classmethod
    def _classify_clause_verb_and_transfer(
        cls, low_text: str, low_tokens: List[str], subj: str, mentioned: List[str], known_entities: Set[str]
    ) -> Tuple[str, str, str, str]:
        transfer_to = ""
        transfer_from = ""

        m_to = re.search(
            r"\b(?:gave|give|gives|handed|passed|lent|donated)\b.*?\bto\s+(?:(?:her|his|their|the|a|an)\s+)?([a-z]+)\b",
            low_text,
        )
        if m_to:
            cand = m_to.group(1)
            if cand in known_entities or cand in ROLE_OWNERS or cand in ("her", "him", "student", "students", "friend", "friends", "child", "children", "kid", "kids", "boy", "boys", "girl", "girls", "brother", "sister"):
                transfer_to = cand

        if not transfer_to:
            m_ditrans = re.search(r"\b(?:gave|give|gives|handed|lent)\s+(her|him|[a-z]+)\s+(?:\$?\d+|some\b)", low_text)
            if m_ditrans:
                cand = m_ditrans.group(1)
                if cand in known_entities or cand in ("her", "him"):
                    transfer_to = cand

        m_from = re.search(
            r"\b(?:got|get|gets|received|receives|borrowed|took|take|bought|buy)\b.*?\bfrom\s+([a-z]+)\b",
            low_text,
        )
        if m_from:
            cand = m_from.group(1)
            if cand in known_entities or cand in ROLE_OWNERS:
                transfer_from = cand

        verb_lemma = "have"
        for tok in low_tokens:
            if tok in ("decided", "wanted", "tried", "went", "started", "began", "caused", "cause", "bring", "poured", "created", "placed"):
                continue
            lem = lemmatize_verb(tok)
            if lem in {
                "buy", "sell", "give", "get", "eat", "drink", "spend", "pay", "cost",
                "earn", "make", "bake", "cook", "pick", "find", "lose", "break", "use",
                "take", "bring", "catch", "collect", "plant", "read", "write", "walk",
                "run", "drive", "save", "leave", "pack", "produce", "print", "cut",
                "throw", "add", "remove", "organize", "play", "jump", "lay", "serve", "have",
            }:
                if lem != "have" or verb_lemma == "have":
                    verb_lemma = lem
                    if lem != "have":
                        break

        if re.search(r"\b(?:spent|paid|for)\s+a\s+total\s+of\s+\$?\d+\b", low_text):
            stage = "TOTAL_GIVEN"
        elif re.search(r"\b(?:saving\s+money\s+for|wants\s+to\s+buy|needs\s+\$\d+|needs\s+\d+(?!\s+[a-z]+\s+to\s+(?:make|bake|build|cook|sew|create))|which\s+costs\s+\$?\d+|recipe\s+wants\s+\d+)\b", low_text):
            stage = "TARGET_GOAL"
        elif re.search(
            r"\b(?:left\s+over|had\s+\$?\d+\s+.*?left|has\s+\$?\d+\s+.*?left|now\s+has|now\s+have|there\s+are\s+now|now\s+there\s+(?:are|is|were|was)|\bin\s+total\b|\bin\s+all\b|are\s+\d+\s+.*?now\b|were\s+\d+\s+.*?remaining|are\s+\d+\s+.*?remaining|now\s+(?:he|she|it|they|we|you|[a-z]+)\s+(?:has|have|had)|still\s+had|still\s+has|still\s+have|ended\s+up\s+with|remained)\b",
            low_text,
        ):
            stage = "FINAL"
        elif re.search(r"\b(?:paid\s+with|gave\s+(?:the\s+)?cashier|handed\s+(?:the\s+)?cashier)\b", low_text):
            stage = "PAID_BILL"
        elif transfer_to:
            stage = "LOSE"
        elif transfer_from:
            stage = "GAIN"
        elif re.search(
            r"\b(?:discount\s+of|ate|eats|eaten|lost|lose|spent|spend|sold|sell|broke|broken|dropped|threw\s+away|threw|flew\s+away|ran\s+away|went\s+away|moved\s+away|gave\s+away|drank|melted|spilled|used(?!\s+to)|uses|wasted|rotten|defective|died|fell\s+off|cut\s+off|taken\s+away|taken|removed|deleted|erased|sawed|destroyed|spilled|spill|melted|melts|missed|didn't\s+show|did\s+not\s+show|got\s+off|subtracted|deducted|has\s+to\s+pay|had\s+to\s+pay|owes|owed|some\s+left|bakes\s+[a-z]+\s+.*?with\s+\d+)\b",
            low_text,
        ):
            stage = "LOSE"
        elif re.search(
            r"\b(?:bought|buy|purchased|found|find|got|received|earned|earn|collected|picked|baked|made|added|poured|grew|caught|won|saved|saves|came\s+to\s+join|joined|cut\s+some\s+more|more\s+[a-z]+|another\s+\d+)\b",
            low_text,
        ):
            stage = "GAIN"
        else:
            stage = "INIT"

        if low_text.startswith(("after eating", "after giving", "after losing", "after spending", "after using", "after selling")):
            stage = "LOSE"
        elif low_text.startswith(("after buying", "after finding", "after getting", "after receiving", "after earning", "after baking", "after making")):
            stage = "GAIN"

        return verb_lemma, stage, transfer_to, transfer_from

    @classmethod
    def _extract_unit_and_rate_around(
        cls, tokens: List[str], pos: int, raw_num_str: str, clause_low: str
    ) -> Tuple[str, Optional[str], str]:
        unit = "item"
        per_unit: Optional[str] = None
        attrs: List[str] = []

        if raw_num_str.startswith("$"):
            unit = "dollar"
        elif raw_num_str.endswith("%"):
            unit = "percent"

        n = len(tokens)
        if pos >= 2 and tokens[pos - 1].lower().strip(".,?!") in ("of", "with", "holding", "containing"):
            prev_container = lemmatize_noun(tokens[pos - 2])
            if prev_container in MULTIPLIER_CONTAINER_UNITS:
                per_unit = prev_container

        m_cut_into = re.search(r"\b(?:each|every)\s+([a-z]+)\s+into\s+\d+", clause_low)
        if m_cut_into:
            cp = lemmatize_noun(m_cut_into.group(1))
            if cp and cp not in NON_UNIT_WORDS:
                per_unit = cp

        direct_obj_before_to = ""
        if pos >= 2 and tokens[pos - 1].lower().strip(".,?!") == "to":
            cand_obj = lemmatize_noun(tokens[pos - 2])
            if cand_obj and cand_obj not in NON_UNIT_WORDS and cand_obj not in VERB_LEMMA:
                direct_obj_before_to = cand_obj

        j = pos + 1
        if j < n and tokens[j].lower().strip(".,?!") == "of":
            j += 1
            while j < n and tokens[j].lower().strip(".,?!") in ("the", "his", "her", "their", "its", "these", "those", "remaining", "rest"):
                j += 1

        end_scan = min(n, pos + 8)
        for scan_k in range(pos + 1, end_scan):
            tk_clean = tokens[scan_k].strip(".,?!:;\"'()[]{}")
            if re.fullmatch(r"\$?\d+(?:\.\d+)?%?|\d+/\d+|number\d+", tk_clean, re.I):
                end_scan = scan_k
                break
            if tk_clean.lower() in ("and", "or", "but", "who", "which", "while", "if", "when", "before", "after", "than"):
                end_scan = scan_k
                break

        # 単位名詞の探索: 前置詞境界 (BOUNDARY_PREPOSITIONS: in/on/for/during/with/at 等) に到達した時点で打ち切る！
        # これにより "eats 3 for breakfast" や "6 during dinner" や "5 in the basket" で誤って前置詞目的語を単位にせず unit="item" (先行単位継承) になる！
        for k in range(j, end_scan):
            w = tokens[k].lower().strip(".,?!:;\"'()[]{}")
            if not w:
                continue
            if w in ("per", "each", "every", "a", "an") and unit != "item":
                if w in ("per", "every", "each") and k + 1 < end_scan:
                    pw = lemmatize_noun(tokens[k + 1])
                    if pw == "morning":
                        pw = "day"
                    if pw and pw not in NON_UNIT_WORDS:
                        per_unit = pw
                        break
                if w == "each":
                    nxt_after_each = tokens[k + 1].lower().strip(".,?!") if k + 1 < len(tokens) else ""
                    if nxt_after_each and (nxt_after_each in VERB_LEMMA or nxt_after_each in ("have", "has", "had", "get", "got", "need", "needs", "hold", "holds", "cost", "costs", "weigh", "weighs")):
                        break
                    if per_unit is None:
                        found_prev_u = ""
                        for back_k in range(pos - 1, -1, -1):
                            bw_raw = tokens[back_k].lower().strip(".,?!")
                            if bw_raw in ("and", "but", "then", ";"):
                                break
                            if re.fullmatch(r"\$?\d+(?:\.\d+)?%?|number\d+", bw_raw):
                                for u_scan in range(back_k + 1, min(pos, back_k + 4)):
                                    wu = tokens[u_scan].lower().strip(".,?!")
                                    if wu in BOUNDARY_PREPOSITIONS or wu in NON_UNIT_WORDS:
                                        break
                                    if wu in ADJECTIVE_MODIFIERS:
                                        continue
                                    cand_u = lemmatize_noun(wu)
                                    if cand_u and cand_u not in NON_UNIT_WORDS:
                                        found_prev_u = cand_u
                                        break
                                if found_prev_u:
                                    break
                        per_unit = found_prev_u or "group"
                    break
                if w in ("a", "an") and k + 1 < end_scan:
                    prev_prep = tokens[k - 1].lower().strip(".,?!") if k > 0 else ""
                    pw = lemmatize_noun(tokens[k + 1])
                    if prev_prep not in BOUNDARY_PREPOSITIONS and pw in TIME_UNITS:
                        per_unit = "day" if pw == "morning" else pw
                        break
                continue
            # 前置詞境界にぶつかったら単位名詞探索は終了！
            if w in BOUNDARY_PREPOSITIONS:
                break
            if w in NON_UNIT_WORDS:
                continue
            if w in ADJECTIVE_MODIFIERS:
                attrs.append(w)
                continue
            # Morphological modifier detection: "<num != 1> <singular_mod> <plural_head_noun>" (e.g. "26 salty cookies", "18 siamese cats")
            if unit == "item" and k + 1 < end_scan and not w.endswith("s") and w not in ("dozen", "pair", "hundred", "thousand", "foot", "inch", "mile", "hour", "minute", "second", "day", "week", "month", "year"):
                w_nxt = tokens[k + 1].lower().strip(".,?!:;\"'()[]{}")
                if (
                    len(w_nxt) >= 3
                    and w_nxt not in BOUNDARY_PREPOSITIONS
                    and w_nxt not in NON_UNIT_WORDS
                    and w_nxt not in VERB_LEMMA
                    and (w_nxt.endswith("s") or w_nxt in IRREGULAR_SINGULAR)
                    and not w_nxt.endswith(("ss", "is", "us"))
                ):
                    attrs.append(w)
                    continue
            lem = lemmatize_noun(w)
            if lem and lem not in NON_UNIT_WORDS:
                if unit == "dollar" and lem not in ("dollar", "cent", "bill"):
                    if lem not in ("selling", "buying", "paying", "making", "discount"):
                        attrs.append(lem)
                elif unit == "item":
                    unit = lem
                break

        if direct_obj_before_to and unit in ("friend", "item"):
            unit = direct_obj_before_to

        local_after_tokens = [tokens[k].lower().strip(".,?!") for k in range(pos + 1, end_scan)]
        window_after = " ".join(local_after_tokens)
        if per_unit is None:
            m_pe = re.search(r"\b(?:per|every|each)\s+(?:(?:fresh|new|single|whole|full|small|large|big|mini| duck)\s+){0,2}([a-z]+)\b", window_after)
            if m_pe:
                cp = lemmatize_noun(m_pe.group(1))
                if cp in ("morning", "night", "evening", "afternoon"):
                    cp = "day"
                if cp and cp != unit and cp not in NON_UNIT_WORDS and cp not in BOUNDARY_PREPOSITIONS:
                    per_unit = cp
            elif re.search(r"(?<!\bon\s)(?<!\bin\s)(?<!\bfor\s)(?<!\bat\s)\b(?:an|a)\s+(hour|minute|second|day|week|month|year|morning)\b", window_after):
                m_tm = re.search(r"\b(?:an|a)\s+(hour|minute|second|day|week|month|year|morning)\b", window_after)
                if m_tm:
                    cp = lemmatize_noun(m_tm.group(1))
                    per_unit = "day" if cp == "morning" else cp
            elif re.search(r"\b(?:each\s*$|apiece\b)", window_after):
                found_prev_u = ""
                for back_k in range(pos - 1, -1, -1):
                    bw_raw = tokens[back_k].lower().strip(".,?!")
                    if bw_raw in ("and", "but", "then", ";"):
                        break
                    if re.fullmatch(r"\$?\d+(?:\.\d+)?%?|number\d+", bw_raw):
                        if back_k + 1 < pos:
                            cand_u = lemmatize_noun(tokens[back_k + 1])
                            if cand_u and cand_u not in NON_UNIT_WORDS:
                                found_prev_u = cand_u
                                break
                per_unit = found_prev_u or "group"

        if per_unit is None and pos > 0:
            before_str = " ".join(t.lower().strip(".,?!") for t in tokens[:pos])
            m_subj_each = re.search(r"\b\d+(?:\.\d+)?\s+([a-z]+)\s+each\b", before_str)
            if m_subj_each:
                cand_s = lemmatize_noun(m_subj_each.group(1))
                if cand_s and cand_s != unit and cand_s not in NON_UNIT_WORDS:
                    per_unit = cand_s

        if per_unit is None:
            m_each_of_n = re.search(r"\b(?:each|every)\s+of\s+(?:his|her|their|the)\s+\d+\s+([a-z]+)\b", clause_low)
            if m_each_of_n and pos < clause_low.find("each of"):
                cand_eo = lemmatize_noun(m_each_of_n.group(1))
                if cand_eo and cand_eo != unit and cand_eo not in NON_UNIT_WORDS:
                    per_unit = cand_eo
        if per_unit is None:
            m_each_any = re.search(r"\b(?:each|every)\s+([a-z]+)\b", clause_low)
            if m_each_any:
                raw_after_e = m_each_any.group(1)
                if raw_after_e not in VERB_LEMMA and raw_after_e not in ("have", "has", "had", "get", "got", "need", "hold", "cost", "other", "time"):
                    cand_p = lemmatize_noun(raw_after_e)
                    if cand_p == "morning":
                        cand_p = "day"
                    if cand_p and cand_p != unit and cand_p not in NON_UNIT_WORDS:
                        per_unit = cand_p

        m_build_for = re.search(r"\b(?:to\s+build|for)\s+(?:a|an|the)\s+([a-z]+)\b", window_after)
        if m_build_for:
            bf = lemmatize_noun(m_build_for.group(1))
            if bf and bf not in NON_UNIT_WORDS and bf not in TIME_UNITS and bf not in attrs:
                attrs.append(bf)

        m_time_attr = re.search(
            r"\b(?:in|on|during)\s+(?:the\s+)?(morning|afternoon|evening|lunch|dinner|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b",
            window_after if window_after else clause_low,
        )
        if m_time_attr and m_time_attr.group(1) not in attrs:
            attrs.append(m_time_attr.group(1))
        else:
            m_day_adv = re.search(r"\b(yesterday|today|tomorrow)\b", clause_low)
            if m_day_adv and m_day_adv.group(1) not in attrs:
                attrs.append(m_day_adv.group(1))

        if pos + 3 < n and tokens[pos + 2].lower().strip(".,?!") == "of":
            w3 = tokens[pos + 3].lower().strip(".,?!")
            if w3 in ADJECTIVE_MODIFIERS and pos + 4 < n:
                if w3 not in attrs:
                    attrs.append(w3)
                w4 = lemmatize_noun(tokens[pos + 4])
                if w4 and w4 not in NON_UNIT_WORDS and w4 not in attrs:
                    attrs.append(w4)
            else:
                item_of = lemmatize_noun(w3)
                if item_of and item_of not in NON_UNIT_WORDS and item_of not in attrs:
                    attrs.append(item_of)

        return unit, per_unit, "_".join(attrs[:2])

    @classmethod
    def _parse_numeric_token(cls, tok: str, placeholder_map: Optional[Dict[str, float]]) -> Optional[Tuple[float, str]]:
        clean = tok.strip(".,?!:;\"'()[]{}")
        if not clean:
            return None
        low = clean.lower()
        if low.startswith("number") and low[6:].isdigit():
            idx = int(low[6:])
            if placeholder_map and low in placeholder_map:
                return float(placeholder_map[low]), "NUM"
            return float(17 + idx * 5), "NUM"
        if clean.startswith("$") and len(clean) > 1:
            inner = clean[1:].replace(",", "")
            if inner.lower().startswith("number") and inner[6:].isdigit():
                idx = int(inner[6:])
                return float(placeholder_map.get(inner.lower(), 17 + idx * 5)) if placeholder_map else float(17 + idx * 5), "DOLLAR"
            if re.fullmatch(r"\d+(?:\.\d+)?", inner):
                return float(inner), "DOLLAR"
        if clean.endswith("%") and len(clean) > 1:
            inner = clean[:-1].replace(",", "")
            if re.fullmatch(r"\d+(?:\.\d+)?", inner):
                return float(inner), "PCT"
        if "/" in clean:
            parts = clean.split("/")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) != 0:
                return float(int(parts[0])) / float(int(parts[1])), "FRAC"
        plain = clean.replace(",", "")
        if re.fullmatch(r"\d+(?:\.\d+)?", plain):
            return float(plain), "NUM"
        return None

    @classmethod
    def _parse_body_clause(
        cls,
        c_idx: int,
        c_raw: str,
        c_conv: str,
        entity_stack: List[Tuple[str, str]],
        known_entities: Set[str],
        next_reg_id: int,
        placeholder_map: Optional[Dict[str, float]],
    ) -> Tuple[ClauseInfo, List[QuantReg], int]:
        raw_tokens = c_conv.split()
        low_tokens = [t.lower().strip(".,?!:;\"'()[]{}") for t in raw_tokens if t.strip(".,?!:;\"'()[]{}")]
        low_text = " ".join(low_tokens)

        prev_stack_snapshot = list(entity_stack)
        subj, mentioned = cls._detect_entities_in_tokens(raw_tokens, known_entities, entity_stack)
        verb_lemma, stage_hint, transfer_to, transfer_from = cls._classify_clause_verb_and_transfer(
            low_text, low_tokens, subj, mentioned, known_entities
        )
        if transfer_to in ("her", "him"):
            transfer_to = cls._resolve_pronoun_entity(transfer_to, prev_stack_snapshot)

        has_rem = bool(re.search(r"\b(?:remaining|remainder|the\s+rest|left\s+over|what\s+was\s+left)\b", low_text))
        has_some = bool(re.search(r"\b(?:some\s+(?:more|new|of|left)|(?:added|cut|gave|sold|ate|took|lost)\s+some)\b", low_text))
        cues = {w for w in low_tokens if MATH_OPERATION_CUES.fullmatch(w)}
        clause_content_words = tuple(
            lemmatize_noun(w) for w in low_tokens if w not in CONTENT_STOPWORDS and not re.fullmatch(r"\$?\d+.*", w)
        )

        c_info = ClauseInfo(
            idx=c_idx, raw_text=c_raw, norm_text=c_conv, tokens=low_tokens,
            owner=subj, verb_lemma=verb_lemma, stage_hint=stage_hint,
            transfer_to=transfer_to, transfer_from=transfer_from,
            has_remaining_cue=has_rem, has_unknown_some=has_some, cue_tokens=cues,
        )

        comp_ref = ""
        comp_mode = ""
        m_comp = re.search(
            r"\b(more|fewer|less|lesser|older|younger|taller|shorter|heavier|lighter|longer|farther|faster|higher|lower)\b.*?\bthan\s+(?:on\s+|in\s+)?([a-z]+)\b",
            low_text,
        )
        if m_comp:
            cw, ref_w = m_comp.group(1), m_comp.group(2)
            if ref_w in known_entities or ref_w in ROLE_OWNERS or ref_w in ADJECTIVE_MODIFIERS:
                comp_ref = ref_w
                comp_mode = "COMP_MORE" if cw in ("more", "older", "taller", "heavier", "longer", "farther", "faster", "higher") else "COMP_LESS"

        post_mult_offsets: Dict[float, str] = {}
        for m_pm in re.finditer(
            r"\b(\d+(?:\.\d+)?)\s+(more|less|fewer|older|younger|taller|shorter|longer)\s+than\s+(?:twice|thrice|double|triple|half|\d+(?:\.\d+)?\s+times|\d+/\d+|0\.\d+)",
            low_text,
        ):
            pm_val = float(m_pm.group(1))
            pm_dir = "POST_MULT_MORE" if m_pm.group(2) in ("more", "older", "taller", "longer") else "POST_MULT_LESS"
            post_mult_offsets[pm_val] = pm_dir

        # GSM8K 頻出の暗黙倍率・半減表現を網羅 ("half the ...", "in half", "half are ...", "twice Mary's ...")
        mult_anaphors: List[Tuple[float, str, Optional[str]]] = []
        used_token_positions: Set[int] = set()

        m_twice_rate = re.search(r"\btwice\s+a\s+(week|day|month|year|hour)\b", low_text)
        if m_twice_rate:
            p_u = lemmatize_noun(m_twice_rate.group(1))
            mult_anaphors.append((2.0, "", p_u))
        elif re.search(r"\b(?:twice|double|doubled|doubles)\b", low_text):
            m_as = re.search(
                r"\b(?:twice|double)\s+(?:as\s+(?:many|much|long|far|fast|old)(?:\s+[a-z]+)?\s+as\s+(?:on\s+|in\s+|her\s+|his\s+)?|the\s+(?:number|amount|age|size|cost|price)\s+of\s+)?([a-z]+)(?:'s)?\b",
                low_text,
            )
            ref_t = ""
            if m_as:
                cand_r = m_as.group(1)
                if cand_r in known_entities or cand_r in ROLE_OWNERS or cand_r in ADJECTIVE_MODIFIERS:
                    ref_t = cand_r
            mult_anaphors.append((2.0, ref_t, None))

        if re.search(r"\b(?:thrice|triple|tripled|triples)\b", low_text):
            mult_anaphors.append((3.0, "", None))

        # "half" 全般 ("half as many", "half of ...", "half the ...", "in half", "half are/were", "halved")
        if re.search(r"\b(?:half|halved)\b", low_text):
            mult_anaphors.append((0.5, "", None))

        # Ratio detection ("ratio ... 7:11" or "ratio ... 3 to 5")
        ratio_pair_vals: Optional[Tuple[float, float]] = None
        if "ratio" in low_text:
            m_rat_col = re.search(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", c_conv)
            m_rat_to = re.search(r"\bratio\b.*?\b(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\b", low_text)
            m_rat = m_rat_col or m_rat_to
            if m_rat:
                ratio_pair_vals = (float(m_rat.group(1)), float(m_rat.group(2)))

        # "One <unit> (of <attr>) costs/has/holds $X" unit-rate pattern ("One pair of shorts costs $16.50")
        m_one_unit_rate = re.match(
            r"^(?:one|1)\s+([a-z]+)(?:\s+of\s+([a-z]+))?\s+costs?\s+\$?\d+",
            low_text,
        )
        one_rate_unit = lemmatize_noun(m_one_unit_rate.group(1)) if m_one_unit_rate else ""
        one_rate_attr = lemmatize_noun(m_one_unit_rate.group(2)) if (m_one_unit_rate and m_one_unit_rate.group(2)) else ""
        if m_one_unit_rate and low_tokens and low_tokens[0] == "1":
            used_token_positions.add(0)

        m_times_fewer = re.search(
            r"\b(\d+(?:\.\d+)?)\s+times\s+(?:fewer|less|smaller|shorter|lighter)\b(?:.*?\bthan\s+([a-z]+)\b)?",
            low_text,
        )
        if m_times_fewer:
            tf_val = float(m_times_fewer.group(1))
            tf_ref = m_times_fewer.group(2) or ""
            if tf_ref not in known_entities and tf_ref not in ROLE_OWNERS:
                tf_ref = ""
            mult_anaphors.append((1.0 / max(tf_val, 1e-6), tf_ref, None))
            for ti, tok in enumerate(low_tokens):
                if tok == m_times_fewer.group(1) and ti + 1 < len(low_tokens) and low_tokens[ti + 1] == "times":
                    used_token_positions.add(ti)
                    break

        m_times_as = re.search(
            r"\b(\d+(?:\.\d+)?)\s+times\s+(?:as\s+(?:many|much|long|far|old)|more|the\b|her\b|his\b)\b(?:\s+.*?\b(?:as|than)\s+(?:on\s+|in\s+)?([a-z]+)\b)?",
            low_text,
        )
        if m_times_as:
            t_val = float(m_times_as.group(1))
            t_ref = m_times_as.group(2) or ""
            if t_ref not in known_entities and t_ref not in ROLE_OWNERS and t_ref not in ADJECTIVE_MODIFIERS:
                t_ref = ""
            mult_anaphors.append((t_val, t_ref, None))
            for ti, tok in enumerate(low_tokens):
                if tok == m_times_as.group(1) and ti + 1 < len(low_tokens) and low_tokens[ti + 1] == "times":
                    used_token_positions.add(ti)
                    break

        regs: List[QuantReg] = []
        local_owner = subj

        for pos, tok in enumerate(raw_tokens):
            clean = tok.strip(".,?!:;\"'()[]{}")
            if clean.endswith("'s"):
                cand_o = clean[:-2].lower()
                prev_of = raw_tokens[pos - 1].lower().strip(".,?!") if pos > 0 else ""
                if pos + 1 < len(raw_tokens) and raw_tokens[pos + 1].lower().strip(".,?!") in ("mother", "father", "mom", "dad", "brother", "sister", "friend"):
                    local_owner = f"{cand_o}_{raw_tokens[pos + 1].lower().strip('.,?!')}"
                elif cand_o in known_entities or cand_o in ROLE_OWNERS:
                    if prev_of == "of" and subj != cand_o and verb_lemma in ("buy", "take", "eat", "steal", "borrow", "use"):
                        local_owner = cand_o
                        stage_hint = "LOSE"
                    else:
                        local_owner = cand_o
            elif clean.lower() in known_entities and subj != "GROUP":
                prev_w = raw_tokens[pos - 1].lower().strip(".,?!") if pos > 0 else ""
                if prev_w not in ("than", "to", "from", "with", "for", "as", "of") and lemmatize_verb(prev_w) not in ("give", "sell", "pay", "lend", "hand", "pass"):
                    local_owner = clean.lower()

            if pos in used_token_positions:
                continue

            parsed_num = cls._parse_numeric_token(tok, placeholder_map)
            if parsed_num is None:
                continue
            val, num_kind = parsed_num

            prev_w = raw_tokens[pos - 1].lower().strip(".,?!") if pos > 0 else ""
            if prev_w in ("in", "since", "year", "during", "from") and 1700 <= val <= 2035 and abs(val - round(val)) < 1e-9:
                continue

            unit, per_unit, attr = cls._extract_unit_and_rate_around(raw_tokens, pos, clean, low_text)

            if pos + 1 < len(raw_tokens):
                nxt_num = cls._parse_numeric_token(raw_tokens[pos + 1], placeholder_map)
                if nxt_num is not None and nxt_num[1] == "DOLLAR":
                    unit = "bill"
            if prev_w and cls._parse_numeric_token(raw_tokens[pos - 1], placeholder_map) is not None and num_kind == "DOLLAR":
                per_unit = "bill"

            left_w = tuple(
                lemmatize_verb(raw_tokens[k])
                for k in range(max(0, pos - 4), pos)
                if raw_tokens[k].lower().strip(".,?!") not in ("the", "a", "an")
            )
            right_w = tuple(
                lemmatize_noun(raw_tokens[k])
                for k in range(pos + 1, min(len(raw_tokens), pos + 5))
                if raw_tokens[k].lower().strip(".,?!") not in ("the", "a", "an")
            )

            reg_stage = stage_hint
            ref_ent = ""
            is_rem_frac = False

            loc_win = " ".join(t.lower().strip(".,?!") for t in raw_tokens[max(0, pos - 3) : min(len(raw_tokens), pos + 5)])

            if val in post_mult_offsets and any(w in ("more", "less", "fewer", "older", "younger", "taller", "shorter", "longer") for w in right_w):
                reg_stage = post_mult_offsets[val]
            elif unit in TIME_UNITS and per_unit is None and re.search(r"\bfor\s+(?:a\s+)?\d+", loc_win):
                reg_stage = "TIME_PARAM"
            elif stage_hint == "PAID_BILL":
                reg_stage = "PAID_BILL"
            elif num_kind == "PCT":
                if re.search(r"\b(?:off|discount|less|decrease|decreased|reduced|reduction|sale)\b", low_text):
                    reg_stage = "PCT_LESS"
                elif re.search(r"\b(?:tip|tax|more|increase|increased|increases|extra|interest|markup|bonus)\b", low_text):
                    reg_stage = "PCT_MORE"
                else:
                    reg_stage = "PCT_OF"
            elif num_kind == "FRAC" or (val in (0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8) and pos + 1 < len(raw_tokens) and raw_tokens[pos + 1].lower().strip(".,?!") == "of"):
                reg_stage = "FRAC_LOSE" if stage_hint == "LOSE" else "FRAC"
                if has_rem or "remaining" in loc_win or "rest" in loc_win:
                    is_rem_frac = True
            elif comp_mode and comp_ref:
                if any(w in ("less", "lesser", "fewer", "shorter", "smaller", "younger") for w in right_w):
                    reg_stage = "COMP_LESS"
                elif any(w in ("more", "farther", "longer", "older", "taller", "bigger") for w in right_w):
                    reg_stage = "COMP_MORE"
                else:
                    reg_stage = comp_mode
                ref_ent = comp_ref
            elif transfer_to and local_owner == subj:
                reg_stage = "TRANSFER_OUT"
                ref_ent = transfer_to
            elif transfer_from and local_owner == subj:
                reg_stage = "TRANSFER_IN"
                ref_ent = transfer_from
            elif re.search(r"\bdiscount\b", loc_win):
                reg_stage = "LOSE"
            elif per_unit is not None and stage_hint != "LOSE":
                if unit == "dollar" and verb_lemma in ("buy", "spend", "pay"):
                    reg_stage = "COST_RATE"
                else:
                    reg_stage = "RATE"
            elif unit == "dollar" and verb_lemma in ("buy", "spend", "pay"):
                reg_stage = "COST_OUT"
            elif unit == "dollar" and verb_lemma == "sell":
                reg_stage = "GAIN"

            if one_rate_unit and per_unit is None:
                per_unit = one_rate_unit
                if one_rate_attr and not attr:
                    attr = one_rate_attr
                reg_stage = "COST_RATE" if unit == "dollar" else "RATE"

            if stage_hint == "TOTAL_GIVEN" and unit == "dollar" and per_unit is None:
                reg_stage = "TOTAL_GIVEN"
            elif unit == "dollar" and re.search(r"\b(?:rent|cost|fee|expense|tuition|repair|repairs|puts\s+in)\b", low_text) and reg_stage in ("INIT", "LOSE") and per_unit is None:
                reg_stage = "COST_OUT"

            # Total book/trip capacity ("reading a 120-page book", "during his 60-mile bike trip")
            if per_unit is None and re.search(rf"\b{int(val) if abs(val - round(val)) < 1e-9 else val}\s+{re.escape(unit)}s?\s+(?:book|novel|trip|trail|journey|drive|race|course|file)\b", low_text):
                reg_stage = "POOL_TOTAL"

            reg = QuantReg(
                reg_id=next_reg_id, value=val, owner=local_owner, unit=unit, per_unit=per_unit,
                attr=attr, stage=reg_stage, action=verb_lemma, clause_idx=c_idx,
                ref_owner=ref_ent, is_implicit=False, is_remaining_frac=is_rem_frac,
                depth=0, origin_mask=(1 << min(next_reg_id, 28)),
                left_win=left_w, right_win=right_w, clause_words=clause_content_words,
            )
            regs.append(reg)
            next_reg_id += 1

        for m_val, m_ref, m_per_u in mult_anaphors:
            _, _, m_attr = cls._extract_unit_and_rate_around(raw_tokens, 0, "", low_text)
            if m_per_u is not None:
                reg = QuantReg(
                    reg_id=next_reg_id, value=m_val, owner=subj, unit="time", per_unit=m_per_u,
                    attr=m_attr, stage="RATE", action=verb_lemma, clause_idx=c_idx,
                    is_implicit=True, depth=0, origin_mask=(1 << min(next_reg_id, 28)),
                    clause_words=clause_content_words,
                )
            else:
                is_verb_scale = bool(
                    re.search(r"\b(?:to\s+(?:double|triple|halve)|(?:weight|size|amount|money|value|height|speed|population)\s+(?:doubled|tripled|halved)|(?:double|triple|halve)\s+the\s+[a-z]+)", low_text)
                    and "as many" not in low_text
                    and "as much" not in low_text
                )
                anaphor_st = "FRAC_LOSE" if (stage_hint == "LOSE" and "as many" not in low_text and "as much" not in low_text and not has_rem) else "MULT_ANAPHOR"
                eff_o = prev_stack_snapshot[-1][0] if (subj == "GROUP" and prev_stack_snapshot) else subj
                eff_attr = "inplace_scale" if is_verb_scale else m_attr
                reg = QuantReg(
                    reg_id=next_reg_id, value=m_val, owner=eff_o, unit="scalar", per_unit="ratio",
                    attr=eff_attr, stage=anaphor_st, action=verb_lemma, clause_idx=c_idx,
                    ref_owner=m_ref, is_implicit=True, is_remaining_frac=has_rem,
                    depth=0, origin_mask=(1 << min(next_reg_id, 28)),
                    clause_words=clause_content_words,
                )
            regs.append(reg)
            next_reg_id += 1

        if ratio_pair_vals is not None and not any(r.stage == "RATIO_PART" for r in regs):
            regs = [r for r in regs if not (r.value in ratio_pair_vals and r.per_unit is None)]
            ra_v, rb_v = ratio_pair_vals
            ent_a = mentioned[0] if len(mentioned) >= 1 else subj
            ent_b = mentioned[1] if len(mentioned) >= 2 else ""
            regs.append(QuantReg(reg_id=next_reg_id, value=ra_v, owner=ent_a, unit="scalar", attr="ratio_a", stage="RATIO_PART", action="have", clause_idx=c_idx, ref_owner=ent_b, is_implicit=True, depth=0, origin_mask=(1 << min(next_reg_id, 28)), clause_words=clause_content_words))
            next_reg_id += 1
            regs.append(QuantReg(reg_id=next_reg_id, value=rb_v, owner=ent_b or subj, unit="scalar", attr="ratio_b", stage="RATIO_PART", action="have", clause_idx=c_idx, ref_owner=ent_a, is_implicit=True, depth=0, origin_mask=(1 << min(next_reg_id, 28)), clause_words=clause_content_words))
            next_reg_id += 1

        return c_info, regs, next_reg_id

    @classmethod
    def _parse_question_clause(
        cls,
        q_clause_idx: int,
        question_raw: str,
        entity_stack: List[Tuple[str, str]],
        known_entities: Set[str],
        body_regs: List[QuantReg],
        clauses_info: List[ClauseInfo],
        next_reg_id: int,
        placeholder_map: Optional[Dict[str, float]],
    ) -> Tuple[GoalSpec, List[QuantReg], int]:
        q_clean = cls._LABEL_NOISE_RE.sub(" ", question_raw)
        q_conv = TextNormalizer.convert_word_numbers_in_clause(q_clean)
        raw_tokens = q_conv.split()
        low_tokens = [t.lower().strip(".,?!:;\"'()[]{}") for t in raw_tokens if t.strip(".,?!:;\"'()[]{}")]
        low_q = " ".join(low_tokens)

        mentioned_q: List[str] = []
        for tok in raw_tokens:
            cl = tok.strip(".,?!:;\"'()[]{}")
            if cl.endswith("'s"):
                cl = cl[:-2]
            lw = cl.lower()
            if lw in known_entities or lw in ROLE_OWNERS:
                if lw not in mentioned_q:
                    mentioned_q.append(lw)

        action_lemma = "have"
        for tok in low_tokens:
            if tok in ("do", "does", "did", "is", "are", "was", "were", "will", "would", "can", "could", "have", "to"):
                continue
            lem = lemmatize_verb(tok)
            if lem in {
                "buy", "sell", "give", "get", "eat", "drink", "spend", "pay", "cost",
                "earn", "make", "bake", "cook", "pick", "find", "lose", "break", "use",
                "take", "bring", "catch", "collect", "plant", "read", "write", "walk",
                "run", "drive", "save", "pack", "produce", "print", "cut", "throw",
                "add", "remove", "leave", "play", "jump", "serve", "lay",
            }:
                action_lemma = lem
                break

        main_q_head = re.split(r"\b(?:if|when|assuming)\b", low_q)[0]
        query_type = "COUNT"
        if re.search(r"\b(?:how\s+many\s+(?:more|fewer|less)|how\s+much\s+(?:more|less|longer|faster|heavier|older|taller|farther|shorter)|difference\s+between|more\s+.*?\s+than|fewer\s+.*?\s+than|less\s+.*?\s+than)\b", main_q_head):
            query_type = "DIFF"
        elif re.search(r"\b(?:at\s+first|start\s+with|started\s+with|did\s+[a-z]+\s+start|begin\s+with|began\s+with|originally|original\s+price|initially|before\s+(?:giving|eating|losing|spending|selling|buying|adding|the\s+discount))\b", low_q):
            query_type = "INITIAL"
        elif re.search(r"\bchange\b.*?\b(?:get|got|receive|received|back|from|his|her)\b|\bwhat\s+is\s+(?:his|her|the)\s+change\b", low_q):
            query_type = "CHANGE"
        elif re.search(r"\bprofit\b", low_q):
            query_type = "PROFIT"
        elif re.search(r"\b(?:average|on\s+average)\b", main_q_head) or (re.search(r"\b(?:each|apiece|per\s+[a-z]+|cost\s+of\s+(?:one|1|a)\b|how\s+big\s+is\s+each)\b", main_q_head) and (not re.search(r"\b(?:every|each)\s+(?:day|week|month|year|morning|night)\b", main_q_head) or not any(r.per_unit in TIME_UNITS for r in body_regs))):
            query_type = "RATE_EACH"
        elif re.search(r"\b(?:how\s+many\s+times|what\s+fraction|what\s+percent|what\s+percentage|what\s+ratio)\b", main_q_head):
            query_type = "RATIO"
        elif re.search(r"\b(?:have\s+left|has\s+left|are\s+left|were\s+left|left\s+over|remain|remaining|still\s+have|still\s+has|empty\s+[a-z]+|miss|missed|stayed\s+behind|partially\s+filled|between\s+his\s+first)\b", low_q):
            query_type = "REMAIN"
        elif re.search(r"\b(?:more\s+.*?\s+(?:need|needed)|need\s+to\s+(?:buy|save|earn|add))\b", main_q_head) and any(r.stage in ("TARGET_GOAL",) for r in body_regs):
            query_type = "DIFF"
        elif re.search(r"\b(?:altogether|in\s+all|in\s+total|total|combined|together|sum)\b", main_q_head):
            query_type = "TOTAL"

        unknown_some_stage = ""
        unknown_some_clause = -1
        for ci in clauses_info:
            if ci.has_unknown_some:
                if ci.verb_lemma == action_lemma or (action_lemma in ("cut", "add", "get") and ci.stage_hint == "GAIN") or (action_lemma in ("leave", "give", "lose", "take", "sell", "eat") and ci.stage_hint == "LOSE"):
                    unknown_some_stage = ci.stage_hint
                    unknown_some_clause = ci.idx
                    query_type = "SOLVE_SOME"
                    break

        target_owner = "global"
        second_owner = ""

        if any(w in low_tokens for w in ("they", "them", "their", "both")):
            uniq_b_owners = list(dict.fromkeys(r.owner for r in body_regs if r.owner not in ("global", "GROUP", "one_individual")))
            if len(uniq_b_owners) == 1 and uniq_b_owners[0] in {"store", "shop", "bakery", "restaurant", "cafe", "company", "factory", "school", "club", "team", "farm"}:
                target_owner = uniq_b_owners[0]
            else:
                target_owner = "GROUP"
        elif len(mentioned_q) >= 2 and "and" in low_tokens and "than" not in low_tokens:
            target_owner = "GROUP"
        elif len(mentioned_q) >= 2 and "than" in low_tokens:
            target_owner = mentioned_q[0]
            second_owner = mentioned_q[1]
        elif len(mentioned_q) == 1:
            target_owner = mentioned_q[0]
        else:
            if any(w in low_tokens for w in ("she", "her")):
                target_owner = cls._resolve_pronoun_entity("she", entity_stack) or "global"
            elif any(w in low_tokens for w in ("he", "him", "his")):
                target_owner = cls._resolve_pronoun_entity("he", entity_stack) or "global"
            elif query_type == "TOTAL" and len({r.owner for r in body_regs if r.owner not in ("global", "one_individual")}) >= 2:
                target_owner = "GROUP"
            elif entity_stack and len({e for e, _ in entity_stack}) == 1:
                target_owner = entity_stack[-1][0]

        target_unit = "item"
        second_unit = ""
        target_attr = ""
        second_attr = ""
        per_unit: Optional[str] = None

        if re.search(r"\b(?:how\s+much\s+(?:money|more\s+money|did|does|do|will|would|is|was)|cost|price|pay|paid|earn|earned|spend|spent|profit|change|bill|salary|wage|save|saved|make\s+every\s+day)\b", low_q):
            target_unit = "dollar"
        if re.search(r"\bcents\b", low_q):
            target_unit = "cent"
        m_in_unit = re.search(r",\s*in\s+([a-z]+)\s*\??$", low_q)
        if m_in_unit:
            target_unit = lemmatize_noun(m_in_unit.group(1))

        m_cost_item = re.search(r"\bhow\s+much\s+(?:did|do|does)\s+the\s+([a-z]+)\s+cost\b", low_q)
        if m_cost_item:
            c_it = lemmatize_noun(m_cost_item.group(1))
            if c_it and c_it not in NON_UNIT_WORDS:
                target_attr = c_it

        if "you" in low_tokens and target_owner == "global" and any(r.owner == "you" for r in body_regs):
            target_owner = "you"

        m_hm = re.search(r"\bhow\s+many\s+(?:more\s+|fewer\s+|less\s+|empty\s+|full\s+)?([a-z]+(?:\s+[a-z]+){0,3})", low_q)
        if m_hm:
            phrase_words = m_hm.group(1).split()
            if len(phrase_words) >= 2 and lemmatize_noun(phrase_words[0]) == "dozen" and phrase_words[1] not in ("do", "does", "did", "are", "is", "were", "was"):
                target_unit = "dozen"
                if phrase_words[1] != "of":
                    target_attr = lemmatize_noun(phrase_words[1])
            else:
                for idx_pw, pw in enumerate(phrase_words):
                    if pw in ("do", "does", "did", "are", "is", "were", "was", "will", "would", "can", "could", "should", "have", "has", "had", "in", "on", "at", "than", "left", "came"):
                        break
                    if pw == "of":
                        if idx_pw + 1 < len(phrase_words) and phrase_words[idx_pw + 1] in ADJECTIVE_MODIFIERS:
                            target_attr = phrase_words[idx_pw + 1]
                        break
                    if (pw in ADJECTIVE_MODIFIERS and pw != "empty") or (
                        idx_pw + 1 < len(phrase_words)
                        and not pw.endswith("s")
                        and phrase_words[idx_pw + 1] not in ("do", "does", "did", "are", "is", "were", "was", "in", "on", "at", "of", "than")
                        and (phrase_words[idx_pw + 1].endswith("s") or phrase_words[idx_pw + 1] in IRREGULAR_SINGULAR)
                    ):
                        if not target_attr and pw != "and":
                            target_attr = pw
                        continue
                    lem = lemmatize_noun(pw)
                    if lem and lem not in NON_UNIT_WORDS:
                        target_unit = lem
                        break

        if query_type == "DIFF" and "than" in low_tokens:
            than_pos = low_tokens.index("than")
            for pw in low_tokens[than_pos + 1 : than_pos + 7]:
                if pw in ADJECTIVE_MODIFIERS:
                    second_attr = pw
                    break
                lem = lemmatize_noun(pw)
                if lem and lem not in CONTENT_STOPWORDS and lem not in ("build", "make", "do", "did") and lem not in known_entities:
                    if lem != target_unit and not second_unit:
                        second_unit = lem
                    break
            m_obj_comp = re.search(r"\b(?:build|make|for|in|on)\s+(?:a|an|the)\s+([a-z]+)\s+than\b.*?\b(?:build|make|for|in|on)\s+(?:a|an|the)\s+([a-z]+)\b", low_q)
            if m_obj_comp:
                target_attr = lemmatize_noun(m_obj_comp.group(1))
                second_attr = lemmatize_noun(m_obj_comp.group(2))

        if query_type == "RATE_EACH":
            m_each = re.search(r"\b(?:each|per|every|one)\s+([a-z]+)\b", low_q)
            if m_each:
                cand_p = lemmatize_noun(m_each.group(1))
                if cand_p and cand_p != target_unit and cand_p not in NON_UNIT_WORDS:
                    per_unit = cand_p
            if per_unit is None:
                per_unit = "group"
        else:
            # "How much does she make every day?" のように日次レート自体を尋ねている場合 per_unit="day" も整合許容できるよう記録
            m_q_rate = re.search(r"\b(?:every|each|per|a|an)\s+(day|week|month|year|hour)\b", low_q)
            if m_q_rate and any(r.per_unit == m_q_rate.group(1) for r in body_regs):
                per_unit = m_q_rate.group(1)

        if action_lemma == "give" and len(mentioned_q) >= 2 and "to" in low_tokens:
            target_owner = mentioned_q[0]

        if query_type == "DIFF" and per_unit is None:
            m_diff_each = re.search(r"\b(?:in|on|per)\s+each\s+([a-z]+)\b", main_q_head)
            if m_diff_each:
                per_unit = lemmatize_noun(m_diff_each.group(1))

        q_focus_mods: Set[str] = set()
        for w in low_tokens:
            if w in {"morning", "afternoon", "evening", "lunch", "dinner", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}:
                q_focus_mods.add(w)
        m_two_adj = re.search(r"\bhow\s+many\s+([a-z]+)\s+and\s+([a-z]+)\s+[a-z]+", low_q)
        if m_two_adj and m_two_adj.group(1) in ADJECTIVE_MODIFIERS and m_two_adj.group(2) in ADJECTIVE_MODIFIERS:
            q_focus_mods.add(m_two_adj.group(1))
            q_focus_mods.add(m_two_adj.group(2))
            target_attr = "" 

        q_regs: List[QuantReg] = []
        for pos, tok in enumerate(raw_tokens):
            parsed = cls._parse_numeric_token(tok, placeholder_map)
            if parsed is None:
                continue
            val, num_kind = parsed
            if val == 1.0 and query_type == "RATE_EACH":
                continue
            u, pu, at = cls._extract_unit_and_rate_around(raw_tokens, pos, tok.strip(".,?!"), low_q)
            if pos >= 2 and raw_tokens[pos - 1].lower().strip(".,?!") == "the" and raw_tokens[pos - 2].lower().strip(".,?!") in ("in", "over", "during", "for") and u in TIME_UNITS and not any(r.per_unit == u for r in body_regs):
                continue
            if any(abs(r.value - val) < 1e-9 and r.unit == u and r.stage not in ("LOSE", "GAIN", "COST_OUT") for r in body_regs):
                continue
            st = "PAID_BILL" if (query_type == "CHANGE" and u == "dollar") else "Q_PARAM"
            q_regs.append(
                QuantReg(
                    reg_id=next_reg_id, value=val, owner=target_owner, unit=u, per_unit=pu,
                    attr=at, stage=st, action=action_lemma, clause_idx=q_clause_idx,
                    is_implicit=False, depth=0, origin_mask=(1 << min(next_reg_id, 28)),
                )
            )
            next_reg_id += 1

        has_rest_q = bool(re.search(r"\b(?:the\s+rest|remaining|remainder|other|not\b|don't|didn't|empty)\b", low_q)) or any(ci.has_remaining_cue for ci in clauses_info)
        if target_unit == "item" and body_regs:
            unit_counts: Dict[str, int] = {}
            for r in body_regs:
                if r.unit not in ("item", "scalar", "percent") and r.unit != per_unit:
                    unit_counts[r.unit] = unit_counts.get(r.unit, 0) + 1
            if unit_counts:
                target_unit = max(unit_counts.items(), key=lambda kv: kv[1])[0]

        q_bigrams = {f"{low_tokens[i]}_{low_tokens[i+1]}" for i in range(len(low_tokens) - 1)}

        goal = GoalSpec(
            target_owner=target_owner, second_owner=second_owner,
            target_unit=target_unit, second_unit=second_unit,
            per_unit=per_unit, target_attr=target_attr, second_attr=second_attr,
            query_type=query_type, action_lemma=action_lemma,
            question_tokens=set(low_tokens), question_seq=tuple(low_tokens), question_bigrams=q_bigrams,
            q_focus_modifiers=q_focus_mods, known_entities=set(known_entities),
            has_the_rest_cue=has_rest_q,
            unknown_some_stage=unknown_some_stage, unknown_some_clause=unknown_some_clause,
        )
        return goal, q_regs, next_reg_id

    @classmethod
    def _inject_implicit_domain_constants(
        cls, regs: List[QuantReg], norm_full: str, goal: GoalSpec, next_reg_id: int
    ) -> List[QuantReg]:
        out = list(regs)
        low = norm_full.lower()
        existing_units = {r.unit for r in out} | {r.per_unit for r in out if r.per_unit}

        def _add_const(val: float, unit: str, per_u: Optional[str]) -> None:
            nonlocal next_reg_id
            if any(abs(r.value - val) < 1e-9 and r.unit == unit and r.per_unit == per_u for r in out):
                return
            out.append(
                QuantReg(
                    reg_id=next_reg_id, value=val, owner="global", unit=unit, per_unit=per_u,
                    stage="Q_PARAM", clause_idx=99, is_implicit=True, depth=0,
                    origin_mask=(1 << min(next_reg_id, 28)),
                )
            )
            next_reg_id += 1

        has_explicit_week_qty = any(r.unit == "week" for r in out)
        has_explicit_hour_qty = any(r.unit == "hour" for r in out)
        has_explicit_year_qty = any(r.unit == "year" for r in out)
        has_explicit_day_qty = any(r.unit == "day" for r in out)

        if ("day" in existing_units or goal.per_unit == "day" or "day" in goal.question_tokens) and not has_explicit_week_qty and re.search(r"\b(?:a|one|per|every|each|last|this|next)\s+week\b|\bin\s+a\s+week\b", low):
            if goal.per_unit != "week":
                _add_const(7.0, "day", None)
        if has_explicit_week_qty and ("day" in existing_units or "morning" in existing_units or "night" in existing_units) and not has_explicit_day_qty:
            pass
        if "day" in existing_units and re.search(r"\bweekdays\b", low):
            _add_const(5.0, "day", None)
        if "day" in existing_units and re.search(r"\bweekend\b", low):
            _add_const(2.0, "day", None)
        if "hour" in existing_units and not has_explicit_day_qty and re.search(r"\bin\s+(?:a|one|1)\s+day\b", low) and goal.per_unit != "day":
            _add_const(24.0, "hour", None)
        if "minute" in existing_units and not has_explicit_hour_qty and re.search(r"\b(?:an|one|per|every|each)\s+hour\b", low) and not any(r.per_unit == "hour" for r in out):
            _add_const(60.0, "minute", None)
        if "month" in existing_units and not has_explicit_year_qty and re.search(r"\b(?:a|one|per|every|each)\s+year\b", low):
            _add_const(12.0, "month", None)
        if "week" in existing_units and not has_explicit_year_qty and re.search(r"\b(?:a|one|per|every|each)\s+year\b", low):
            _add_const(52.0, "week", None)

        if goal.target_unit == "dollar":
            for r_idx, r in enumerate(out):
                if r.unit in ("quarter", "dime", "nickel", "cent"):
                    scale = UNIT_CONVERSIONS[(r.unit, "dollar")]
                    out[r_idx] = QuantReg(**{**r.__dict__, "value": r.value * scale, "unit": "dollar"})

        return out

    @classmethod
    def _propagate_missing_units(
        cls, regs: List[QuantReg], clauses: Sequence[ClauseInfo], goal: GoalSpec
    ) -> List[QuantReg]:
        """省略された単位 ("eats 3 for breakfast", "6 during dinner") に先行する具体単位と必要に応じた per_unit を継承"""
        out: List[QuantReg] = []
        last_concrete_unit = goal.target_unit if goal.target_unit not in ("item", "scalar", "dozen") else "item"
        last_per_unit: Optional[str] = None
        for r in regs:
            if r.unit in ("color", "type", "kind", "flavor", "category", "variety") and goal.target_unit not in ("item", r.unit):
                out.append(r)
                continue
            if r.unit not in ("item", "scalar", "percent"):
                last_concrete_unit = r.unit
                if r.unit != "dollar":
                    last_per_unit = r.per_unit
                out.append(r)
            elif r.unit == "item" and r.stage not in ("FRAC", "FRAC_LOSE", "PCT_OF", "PCT_MORE", "PCT_LESS", "MULT_ANAPHOR"):
                # "16 eggs per day. She eats 3 for breakfast" -> 3 も同スケールの egg (/day) を引き継ぐ！
                inh_per = r.per_unit if r.per_unit is not None else last_per_unit
                out.append(QuantReg(**{**r.__dict__, "unit": last_concrete_unit, "per_unit": inh_per}))
            else:
                out.append(r)
        return out


# ==============================================================================
# 3. 高到達率 Soft-Typed 遷移生成・役割抽象化 SCReLU NNUE・Beam Stack Search
# ==============================================================================

STEP_COST: float = 4.25


@dataclass(frozen=True)
class TransitionMove:
    op: str
    idx_a: int
    idx_b: int
    new_reg: QuantReg
    prior_score: float
    rule_tag: str = "GENERIC"
    keep_a_active: bool = False


@dataclass
class SearchNode:
    regs: Tuple[QuantReg, ...]
    consumed_mask: int
    cum_score: float
    depth: int
    history: Tuple[Tuple[str, float, float, float], ...] = ()
    step_feats: Tuple[Tuple[object, ...], ...] = ()


def _fnv_hash(key: str, mod: int) -> int:
    h = 2166136261
    for ch in key:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return 1 + (h % (mod - 1))


class LegalMoveGenerator:
    """
    ハードな `return None` 壁を取り除いた Soft-Typed 合法手生成器。
    異単位減算 (`200 seats - 145 people`)、同単位包含除 (`24 cups / 3 cups`)、
    非コンテナ乗除算 (`6 chairs * 4 legs`) もソフトな事前スコア付きで生成し、
    Oracle 到達率を最大化しつつ NNUE が文脈特徴から最適手を選べるようにする。
    """

    @classmethod
    def generate(
        cls,
        regs: Sequence[QuantReg],
        consumed_mask: int,
        clauses: Sequence[ClauseInfo],
        goal: GoalSpec,
        next_reg_id: int,
        current_depth: int,
    ) -> List[TransitionMove]:
        moves: List[TransitionMove] = []
        active_indices = [i for i in range(len(regs)) if not ((consumed_mask >> i) & 1)]
        if len(active_indices) < 2:
            return moves

        any_clause_needs_rem = goal.has_the_rest_cue or goal.query_type == "REMAIN" or any(c.has_remaining_cue for c in clauses)

        # Algebraic Ratio Split: Total * r_a / (r_a + r_b)
        ratio_idxs = [idx for idx in active_indices if regs[idx].stage == "RATIO_PART"]
        if len(ratio_idxs) == 2:
            ri_a, ri_b = ratio_idxs[0], ratio_idxs[1]
            ra_r, rb_r = regs[ri_a], regs[ri_b]
            r_sum = ra_r.value + rb_r.value
            if r_sum > 1e-6:
                for ti in active_indices:
                    if ti in (ri_a, ri_b):
                        continue
                    rt = regs[ti]
                    if rt.stage in ("INIT", "GAIN", "TOTAL_GIVEN") and rt.per_unit is None:
                        mask_r = rt.origin_mask | ra_r.origin_mask | rb_r.origin_mask
                        for part_r, tag_s in ((ra_r, "RATIO_SPLIT_A"), (rb_r, "RATIO_SPLIT_B")):
                            v_sp = rt.value * part_r.value / r_sum
                            if math.isfinite(v_sp):
                                o_sp = part_r.owner if part_r.owner not in ("global", "") else (goal.target_owner if tag_s == "RATIO_SPLIT_B" else rt.owner)
                                nr_sp = QuantReg(reg_id=next_reg_id, value=v_sp, owner=o_sp, unit=rt.unit if rt.unit != "item" else goal.target_unit, stage="GAIN", action=rt.action, clause_idx=max(rt.clause_idx, part_r.clause_idx), depth=current_depth + 1, origin_mask=mask_r)
                                pr_sp = 7.9 if o_sp == goal.target_owner else 7.2
                                moves.append(TransitionMove("MUL", ti, ri_a if part_r is ra_r else ri_b, nr_sp, pr_sp, tag_s, keep_a_active=False))

        for pos_i, i in enumerate(active_indices):
            ra = regs[i]
            if ra.owner == "one_individual" or ra.stage == "RATIO_PART":
                continue
            if ra.unit in ("color", "type", "kind", "flavor", "category", "variety", "stop") and goal.target_unit != ra.unit:
                continue
            for pos_j, j in enumerate(active_indices):
                if i == j:
                    continue
                rb = regs[j]
                if rb.owner == "one_individual" or rb.stage == "RATIO_PART":
                    continue
                if rb.unit in ("color", "type", "kind", "flavor", "category", "variety", "stop") and goal.target_unit != rb.unit:
                    continue

                if rb.stage in ("PCT_LESS", "PCT_MORE", "PCT_OF") and ra.stage not in ("PCT_LESS", "PCT_MORE", "PCT_OF", "FRAC", "FRAC_LOSE", "MULT_ANAPHOR"):
                    if ra.clause_idx <= rb.clause_idx:
                        moves.extend(cls._make_pct_moves(i, j, ra, rb, any_clause_needs_rem, next_reg_id, current_depth))
                    continue
                if ra.stage in ("PCT_LESS", "PCT_MORE", "PCT_OF"):
                    continue

                if rb.stage == "MULT_ANAPHOR" and ra.stage not in ("MULT_ANAPHOR", "FRAC", "FRAC_LOSE", "POST_MULT_MORE", "POST_MULT_LESS"):
                    if ra.clause_idx <= rb.clause_idx or (rb.ref_owner and rb.ref_owner == ra.owner):
                        moves.extend(cls._make_mult_anaphor_moves(i, j, ra, rb, regs, active_indices, goal, next_reg_id, current_depth))
                    continue
                if ra.stage == "MULT_ANAPHOR":
                    continue

                if rb.stage in ("FRAC", "FRAC_LOSE") and ra.stage not in ("FRAC", "FRAC_LOSE", "MULT_ANAPHOR"):
                    moves.extend(cls._make_frac_moves(i, j, ra, rb, any_clause_needs_rem, next_reg_id, current_depth))
                    continue
                if ra.stage in ("FRAC", "FRAC_LOSE"):
                    continue

                disjoint_origins = (ra.origin_mask & rb.origin_mask) == 0
                if i < j:
                    m_add = cls._try_add(i, j, ra, rb, goal, next_reg_id, current_depth)
                    if m_add is not None:
                        moves.extend(m_add)
                    if disjoint_origins:
                        m_mul = cls._try_mul(i, j, ra, rb, goal, next_reg_id, current_depth)
                        if m_mul is not None:
                            moves.extend(m_mul)

                m_sub = cls._try_sub(i, j, ra, rb, regs, active_indices, any_clause_needs_rem, goal, next_reg_id, current_depth)
                if m_sub is not None:
                    moves.extend(m_sub)

                if disjoint_origins:
                    m_div = cls._try_div(i, j, ra, rb, goal, next_reg_id, current_depth, regs, active_indices)
                    if m_div is not None:
                        moves.extend(m_div)

        moves = [mv for mv in moves if math.isfinite(mv.new_reg.value) and abs(mv.new_reg.value) <= 1e14]
        moves.sort(key=lambda mv: mv.prior_score, reverse=True)
        return moves[:48]

    @classmethod
    def _units_compatible_add(
        cls, ra: QuantReg, rb: QuantReg, goal: GoalSpec, is_sub: bool = False
    ) -> Tuple[bool, float, Optional[QuantReg], Optional[QuantReg]]:
        # 片方だけ per_unit がある場合でも同単位なら省略表現 ("16 eggs/day ... eats 3") として整合許容！
        eff_per = ra.per_unit if ra.per_unit == rb.per_unit else (ra.per_unit or rb.per_unit if ra.unit == rb.unit else None)
        if ra.per_unit != rb.per_unit and ra.unit != rb.unit:
            return False, 0.0, None, None

        if ra.unit == rb.unit:
            b_factor = 1.0 if ra.per_unit == rb.per_unit else 0.90
            return True, b_factor, QuantReg(**{**ra.__dict__, "per_unit": eff_per}), QuantReg(**{**rb.__dict__, "per_unit": eff_per})

        if goal.query_type == "DIFF" and goal.second_unit and {ra.unit, rb.unit} == {goal.target_unit, goal.second_unit}:
            return True, 1.0, QuantReg(**{**ra.__dict__, "unit": goal.target_unit}), QuantReg(**{**rb.__dict__, "unit": goal.target_unit})

        ha = UNIT_HYPERNYM.get(ra.unit, "")
        hb = UNIT_HYPERNYM.get(rb.unit, "")
        hg = UNIT_HYPERNYM.get(goal.target_unit, goal.target_unit)
        if ha and ha == hb and goal.target_unit in CATEGORY_TARGET_UNITS and hg == ha:
            return True, 0.95, QuantReg(**{**ra.__dict__, "unit": goal.target_unit}), QuantReg(**{**rb.__dict__, "unit": goal.target_unit})
        if is_sub and ha and ha == hb and (ra.unit in CATEGORY_TARGET_UNITS or ra.unit == ha) and rb.unit != goal.target_unit and hg == ha:
            return True, 1.0, QuantReg(**{**ra.__dict__, "unit": goal.target_unit, "stage": "POOL_TOTAL"}), QuantReg(**{**rb.__dict__, "unit": goal.target_unit, "stage": "PART"})

        if (rb.unit, ra.unit) in UNIT_CONVERSIONS:
            scale = UNIT_CONVERSIONS[(rb.unit, ra.unit)]
            rb_conv = QuantReg(**{**rb.__dict__, "value": rb.value * scale, "unit": ra.unit})
            return True, 0.85, ra, rb_conv

        if ra.unit == "item" or rb.unit == "item":
            u = ra.unit if ra.unit != "item" else rb.unit
            return True, 0.60, QuantReg(**{**ra.__dict__, "unit": u}), QuantReg(**{**rb.__dict__, "unit": u})

        # 減算時の1対1対応差 ("200 seats - 145 people = 55 empty seats", "15 children - 12 cookies") はソフト許容！
        if is_sub and "dollar" not in (ra.unit, rb.unit) and (ra.unit in TIME_UNITS) == (rb.unit in TIME_UNITS):
            u_res = ra.unit if ra.unit == goal.target_unit else (rb.unit if rb.unit == goal.target_unit else ra.unit)
            return True, 0.55, QuantReg(**{**ra.__dict__, "unit": u_res}), QuantReg(**{**rb.__dict__, "unit": u_res})

        return False, 0.0, None, None

    @classmethod
    def _make_pct_moves(
        cls, i: int, j: int, ra: QuantReg, rb: QuantReg, needs_rem: bool, next_id: int, depth: int
    ) -> List[TransitionMove]:
        out: List[TransitionMove] = []
        locality = 1.2 if abs(rb.clause_idx - ra.clause_idx) <= 1 else 0.2
        mask = ra.origin_mask | rb.origin_mask
        if rb.stage == "PCT_LESS":
            val = ra.value * (1.0 - rb.value / 100.0)
            nr = QuantReg(
                reg_id=next_id, value=val, owner=ra.owner, unit=ra.unit, per_unit=ra.per_unit,
                attr=ra.attr, stage="GAIN", action=ra.action, clause_idx=rb.clause_idx,
                depth=depth + 1, origin_mask=mask,
            )
            out.append(TransitionMove("PCT_SUB", i, j, nr, 6.4 + locality, "PCT_DISCOUNT", keep_a_active=False))
            out.append(TransitionMove("PCT_SUB", i, j, nr, 6.2 + locality, "PCT_DISCOUNT", keep_a_active=True))
            if 0.0 < rb.value < 100.0:
                inv_v = ra.value / (1.0 - rb.value / 100.0)
                nr_inv = QuantReg(
                    reg_id=next_id, value=inv_v, owner=ra.owner, unit=ra.unit, per_unit=ra.per_unit,
                    attr=ra.attr, stage="INITIAL_SOLVED", action="have", clause_idx=rb.clause_idx,
                    depth=depth + 1, origin_mask=mask,
                )
                out.append(TransitionMove("DIV", i, j, nr_inv, 7.6 + locality, "PCT_INV_DIV"))
        elif rb.stage == "PCT_MORE":
            val = ra.value * (1.0 + rb.value / 100.0)
            is_distinct_subset = bool(rb.attr and rb.attr != ra.attr and "tax" not in rb.clause_words and "tip" not in rb.clause_words)
            nr = QuantReg(
                reg_id=next_id, value=val, owner=ra.owner, unit=ra.unit, per_unit=ra.per_unit,
                attr=rb.attr or ra.attr, stage="GAIN", action="sell" if ra.stage == "COST_OUT" else ra.action, clause_idx=rb.clause_idx,
                depth=depth + 1, origin_mask=mask,
            )
            if is_distinct_subset:
                out.append(TransitionMove("PCT_ADD", i, j, nr, 6.6 + locality, "PCT_MARKUP_KEEP", keep_a_active=True))
            out.append(TransitionMove("PCT_ADD", i, j, nr, 6.4 + locality, "PCT_MARKUP", keep_a_active=False))
            if ra.stage == "COST_OUT":
                out.append(TransitionMove("PCT_ADD", i, j, nr, 6.5 + locality, "PCT_MARKUP_KEEP_COST", keep_a_active=True))
        else:
            val = ra.value * (rb.value / 100.0)
            is_comp_pct = "as" in rb.left_win or "as" in rb.right_win
            st_p = "GAIN" if is_comp_pct else "PART"
            nr = QuantReg(
                reg_id=next_id, value=val, owner=rb.owner if rb.owner not in ("global", "GROUP") else ra.owner,
                unit=ra.unit, per_unit=ra.per_unit, attr=rb.attr or ra.attr,
                stage=st_p, action=rb.action, clause_idx=rb.clause_idx,
                depth=depth + 1, origin_mask=mask,
            )
            if is_comp_pct:
                out.append(TransitionMove("PCT_OF", i, j, nr, 6.6 + locality, "PCT_COMP_KEEP", keep_a_active=True))
            out.append(TransitionMove("PCT_OF", i, j, nr, 6.2 + locality, "PCT_DIRECT", keep_a_active=False))
            if needs_rem:
                out.append(TransitionMove("PCT_OF", i, j, nr, 6.1 + locality, "PCT_PART", keep_a_active=True))
        return out

    @classmethod
    def _make_mult_anaphor_moves(
        cls,
        i: int,
        j: int,
        ra: QuantReg,
        rb: QuantReg,
        regs: Sequence[QuantReg],
        active_indices: Sequence[int],
        goal: GoalSpec,
        next_id: int,
        depth: int,
    ) -> List[TransitionMove]:
        # Algebraic Sum-Ratio: "Alice and Bob have 60 stickers together. Alice has twice as many as Bob." -> 60 / (2 + 1) = 20
        if ra.owner in ("GROUP", "global") and rb.value > 1.0 and ("together" in ra.clause_words or "total" in ra.clause_words or ra.owner == "GROUP") and goal.target_owner not in ("GROUP", "global"):
            v_small = ra.value / (rb.value + 1.0)
            if abs(v_small - round(v_small)) < 1e-6:
                o_sm = rb.ref_owner if rb.ref_owner else goal.target_owner
                v_final = v_small * rb.value if goal.target_owner == rb.owner and rb.owner != o_sm else v_small
                nr_sr = QuantReg(reg_id=next_id, value=v_final, owner=goal.target_owner, unit=ra.unit, per_unit=ra.per_unit, stage="FINAL", action="have", clause_idx=rb.clause_idx, depth=depth + 1, origin_mask=ra.origin_mask | rb.origin_mask)
                return [TransitionMove("DIV", i, j, nr_sr, 8.1, "SUM_RATIO_SPLIT", keep_a_active=False)]

        # Inverse Multiplicative Anaphora: "Tom has 24 apples, which is twice as many as Mary" / "Sam is 36. He is 3 times as old as Mia"
        if rb.value > 1.0 and rb.owner == ra.owner and rb.ref_owner and rb.ref_owner != ra.owner and ra.owner != goal.target_owner:
            v_inv = ra.value / rb.value
            if math.isfinite(v_inv) and (abs(v_inv - round(v_inv)) < 1e-6 or abs(v_inv * 4 - round(v_inv * 4)) < 1e-6):
                nr_inv = QuantReg(reg_id=next_id, value=v_inv, owner=rb.ref_owner, unit=ra.unit, per_unit=ra.per_unit, stage="INIT", action=ra.action, clause_idx=rb.clause_idx, depth=depth + 1, origin_mask=ra.origin_mask | rb.origin_mask)
                return [
                    TransitionMove("DIV", i, j, nr_inv, 8.0, "ANAPHOR_DIV_INV", keep_a_active=False),
                    TransitionMove("DIV", i, j, nr_inv, 7.6, "ANAPHOR_DIV_INV_KEEP", keep_a_active=True),
                ]

        if rb.ref_owner and rb.ref_owner not in (ra.owner, ra.attr):
            return []
        dist = rb.clause_idx - ra.clause_idx
        if dist < 0 and not (rb.ref_owner and rb.ref_owner == ra.owner):
            return []

        if rb.is_remaining_frac:
            if ra.stage == "REMAIN_POOL":
                val = ra.value * rb.value
                nr = QuantReg(
                    reg_id=next_id, value=val, owner=rb.owner if rb.owner not in ("global", "GROUP") else ra.owner,
                    unit=ra.unit, per_unit=ra.per_unit, stage="FINAL", action=rb.action,
                    clause_idx=rb.clause_idx, depth=depth + 1, origin_mask=ra.origin_mask | rb.origin_mask,
                )
                return [
                    TransitionMove("MUL", i, j, nr, 7.8, "REM_ANAPHOR_DIRECT", keep_a_active=False),
                    TransitionMove("MUL", i, j, nr, 7.2, "REM_ANAPHOR_KEEP", keep_a_active=True),
                ]
            return []

        if not rb.ref_owner and ra.stage not in ("TARGET_GOAL", "RATE") and not (ra.attr and any(ap in rb.clause_words for ap in ra.attr.split("_") if ap not in ("pound", "pack", "pair"))):
            has_more_recent_same = any(
                k not in (i, j)
                and regs[k].unit == ra.unit
                and regs[k].owner == ra.owner
                and ra.clause_idx < regs[k].clause_idx <= rb.clause_idx
                and regs[k].reg_id < rb.reg_id
                and regs[k].stage in ("INIT", "GAIN")
                for k in active_indices
            )
            if has_more_recent_same:
                return []

        needs_rem_q = goal.has_the_rest_cue or goal.query_type == "REMAIN"
        unit_named_in_rb = ra.unit != "item" and ra.unit in rb.clause_words
        if dist == 0 and rb.value < 1.0 and not unit_named_in_rb:
            if any(regs[k].clause_idx < rb.clause_idx and regs[k].unit in rb.clause_words for k in active_indices if k not in (i, j)):
                return []

        if rb.attr == "inplace_scale":
            # 累積重量等を倍増する場合は先行する同単位レジスタが合算済みであることを要求 ("2 + 6 = 8 -> double -> 16")
            if any(k not in (i, j) and regs[k].unit == ra.unit and regs[k].clause_idx < rb.clause_idx for k in active_indices):
                return []
            val = ra.value * rb.value
            nr = QuantReg(
                reg_id=next_id, value=val, owner=ra.owner, unit=ra.unit, per_unit=ra.per_unit,
                attr="", stage="GAIN", action=ra.action, clause_idx=rb.clause_idx,
                depth=depth + 1, origin_mask=ra.origin_mask | rb.origin_mask,
            )
            return [TransitionMove("MUL", i, j, nr, 7.4, "ANAPHOR_INPLACE_ONLY", keep_a_active=False)]

        is_in_place_mutation = (
            rb.ref_owner == ""
            and rb.action in ("have", "add")
            and ra.owner == rb.owner
            and ra.stage != "TARGET_GOAL"
            and goal.query_type != "TOTAL"
            and "altogether" not in goal.question_tokens
            and not (rb.value < 1.0 and needs_rem_q)
            and not any(regs[k].stage in ("POST_MULT_MORE", "POST_MULT_LESS") and regs[k].clause_idx == rb.clause_idx for k in active_indices)
        )
        score = 6.7 if dist == 1 else (6.1 if dist == 0 else max(4.5, 5.8 - 0.5 * dist))
        if unit_named_in_rb:
            score += 0.8
        if rb.ref_owner and rb.ref_owner in (ra.owner, ra.attr):
            score += 1.4
        val = ra.value * rb.value

        if ra.stage == "TARGET_GOAL":
            out_stage = "INIT"
        elif ra.stage == "TRANSFER_OUT" and ra.ref_owner:
            out_stage = "TRANSFER_OUT"
        elif ra.stage == "RATE":
            out_stage = "RATE"
        elif rb.value < 1.0 and needs_rem_q:
            out_stage = "PART"
        else:
            out_stage = "GAIN"

        out_owner = rb.owner if rb.owner not in ("global", "GROUP") else ra.owner
        nr = QuantReg(
            reg_id=next_id, value=val, owner=out_owner, unit=ra.unit, per_unit=ra.per_unit,
            attr=rb.attr or f"{ra.attr}_scaled", stage=out_stage,
            action=rb.action if rb.action != "have" else ra.action,
            clause_idx=rb.clause_idx,
            ref_owner=ra.ref_owner if ra.stage == "TRANSFER_OUT" else "",
            depth=depth + 1, origin_mask=ra.origin_mask | rb.origin_mask,
        )
        if ra.stage == "TARGET_GOAL":
            return [TransitionMove("MUL", i, j, nr, score + 0.8, "ANAPHOR_FROM_TARGET", keep_a_active=True)]
        if is_in_place_mutation:
            return [
                TransitionMove("MUL", i, j, nr, score + 0.6, "ANAPHOR_SCALE_REPLACE", keep_a_active=False),
                TransitionMove("MUL", i, j, nr, score - 0.3, "ANAPHOR_SCALE_KEEP", keep_a_active=True),
            ]
        return [
            TransitionMove("MUL", i, j, nr, score, "ANAPHOR_SCALE_KEEP", keep_a_active=True),
            TransitionMove("MUL", i, j, nr, score - 0.1, "ANAPHOR_SCALE_REPLACE", keep_a_active=False),
        ]

    @classmethod
    def _make_frac_moves(
        cls, i: int, j: int, ra: QuantReg, rb: QuantReg, needs_rem: bool, next_id: int, depth: int
    ) -> List[TransitionMove]:
        if rb.clause_idx < ra.clause_idx:
            return []
        if rb.is_remaining_frac and ra.stage != "REMAIN_POOL" and depth == 0:
            score = 3.0
        elif rb.is_remaining_frac and ra.stage == "REMAIN_POOL":
            score = 7.3
        else:
            dist = abs(rb.clause_idx - ra.clause_idx)
            score = 6.4 if dist <= 1 else 4.6
        val = ra.value * rb.value
        st_out = "LOSE" if rb.stage == "FRAC_LOSE" else "PART"
        nr = QuantReg(
            reg_id=next_id, value=val, owner=rb.owner if rb.owner not in ("global", "GROUP") else ra.owner,
            unit=ra.unit, per_unit=ra.per_unit, attr=rb.attr, stage=st_out,
            action=rb.action, clause_idx=rb.clause_idx, depth=depth + 1,
            origin_mask=ra.origin_mask | rb.origin_mask,
        )
        if st_out == "LOSE":
            return [TransitionMove("MUL", i, j, nr, score + 0.4, "FRAC_LOSE_APPLY", keep_a_active=True)]
        moves = [TransitionMove("MUL", i, j, nr, score - 0.1, "FRAC_DIRECT", keep_a_active=False)]
        if needs_rem:
            moves.insert(0, TransitionMove("MUL", i, j, nr, score + 0.2, "FRAC_APPLY", keep_a_active=True))
        return moves

    @classmethod
    def _try_add(
        cls, i: int, j: int, ra: QuantReg, rb: QuantReg, goal: GoalSpec, next_id: int, depth: int
    ) -> Optional[List[TransitionMove]]:
        ok, u_bonus, ca, cb = cls._units_compatible_add(ra, rb, goal, is_sub=False)
        if not ok or ca is None or cb is None:
            return None
        if any(st in ("PAID_BILL", "TARGET_GOAL", "TOTAL_GIVEN", "POOL_TOTAL", "TIME_PARAM", "DIFF_SOLVED", "POST_MULT_LESS") for st in (ca.stage, cb.stage)):
            return None
        if "POST_MULT_MORE" in (ca.stage, cb.stage):
            base_r, off_r = (ca, cb) if cb.stage == "POST_MULT_MORE" else (cb, ca)
            if base_r.depth >= 1 and base_r.clause_idx == off_r.clause_idx:
                nr = QuantReg(
                    reg_id=next_id, value=ca.value + cb.value, owner=base_r.owner, unit=base_r.unit,
                    per_unit=base_r.per_unit, attr=base_r.attr, stage="GAIN", action=base_r.action,
                    clause_idx=base_r.clause_idx, depth=depth + 1, origin_mask=ca.origin_mask | cb.origin_mask,
                )
                return [TransitionMove("ADD", i, j, nr, 8.2, "POST_MULT_ADD")]
            return None

        val = ca.value + cb.value
        same_owner = ca.owner == cb.owner
        new_owner = ca.owner if same_owner else ("GROUP" if "global" not in (ca.owner, cb.owner) else (ca.owner if ca.owner != "global" else cb.owner))
        new_attr = ca.attr if ca.attr == cb.attr else "GROUP"
        mask = ca.origin_mask | cb.origin_mask

        if cb.stage == "COMP_MORE" and cb.ref_owner in (ca.owner, ca.attr) and ra.unit == rb.unit:
            nr = QuantReg(reg_id=next_id, value=val, owner=cb.owner, unit=ca.unit, per_unit=ca.per_unit, attr=cb.attr, stage="INIT", action=cb.action, clause_idx=cb.clause_idx, depth=depth + 1, origin_mask=mask)
            return [
                TransitionMove("ADD", i, j, nr, 7.6, "COMP_MORE_FWD", keep_a_active=False),
                TransitionMove("ADD", i, j, nr, 7.4, "COMP_MORE_FWD_KEEP", keep_a_active=True),
            ]
        if ca.stage == "COMP_MORE" and ca.ref_owner in (cb.owner, cb.attr) and ra.unit == rb.unit:
            nr = QuantReg(reg_id=next_id, value=val, owner=ca.owner, unit=cb.unit, per_unit=cb.per_unit, attr=ca.attr, stage="INIT", action=ca.action, clause_idx=ca.clause_idx, depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 7.6, "COMP_MORE_FWD", keep_a_active=False)]

        if cb.stage == "COMP_LESS" and cb.owner == ca.owner and cb.ref_owner and cb.ref_owner != ca.owner and ra.unit == rb.unit:
            nr = QuantReg(reg_id=next_id, value=val, owner=cb.ref_owner, unit=ca.unit, per_unit=ca.per_unit, attr="", stage="INIT", action=ca.action, clause_idx=cb.clause_idx, depth=depth + 1, origin_mask=mask)
            return [
                TransitionMove("ADD", i, j, nr, 7.8, "COMP_LESS_INV", keep_a_active=False),
                TransitionMove("ADD", i, j, nr, 7.5, "COMP_LESS_INV_KEEP", keep_a_active=True),
            ]
        if ca.stage == "COMP_LESS" and ca.owner == cb.owner and ca.ref_owner and ca.ref_owner != cb.owner and ra.unit == rb.unit:
            nr = QuantReg(reg_id=next_id, value=val, owner=ca.ref_owner, unit=cb.unit, per_unit=cb.per_unit, attr="", stage="INIT", action="have", clause_idx=ca.clause_idx, depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 7.8, "COMP_LESS_INV", keep_a_active=False)]

        if "POST_MULT_MORE" in (ca.stage, cb.stage) or ca.stage == "POST_MULT_LESS":
            return None
        if cb.stage == "POST_MULT_LESS":
            if ca.depth >= 1 and ca.clause_idx == cb.clause_idx:
                nr = QuantReg(
                    reg_id=next_id, value=val, owner=ca.owner, unit=ca.unit, per_unit=ca.per_unit,
                    attr=ca.attr, stage="GAIN", action=ca.action, clause_idx=ca.clause_idx,
                    depth=depth + 1, origin_mask=mask,
                )
                return [TransitionMove("SUB", i, j, nr, 8.2, "POST_MULT_SUB")]
            return None
        if any(st in ("COMP_MORE", "COMP_LESS", "DIFF_SOLVED") for st in (ca.stage, cb.stage)):
            return None

        if cb.stage == "TRANSFER_OUT" and cb.ref_owner == ca.owner and ca.stage != "COST_OUT":
            nr = QuantReg(reg_id=next_id, value=val, owner=ca.owner, unit=ca.unit, per_unit=ca.per_unit, attr=ca.attr, stage="GAIN", action="have", clause_idx=cb.clause_idx, depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 7.4, "TRANSFER_RECV")]
        if ca.stage == "TRANSFER_OUT" and ca.ref_owner == cb.owner and cb.stage != "COST_OUT":
            nr = QuantReg(reg_id=next_id, value=val, owner=cb.owner, unit=cb.unit, per_unit=cb.per_unit, attr=cb.attr, stage="GAIN", action="have", clause_idx=ca.clause_idx, depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 7.4, "TRANSFER_RECV")]
        if ca.stage == "TRANSFER_OUT" and cb.stage == "TRANSFER_OUT" and ca.ref_owner and ca.ref_owner == cb.ref_owner:
            nr = QuantReg(reg_id=next_id, value=val, owner=ca.ref_owner, unit=ca.unit, per_unit=ca.per_unit, stage="GAIN", action="have", clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 7.4, "TRANSFER_RECV_SUM")]

        if goal.query_type in ("INITIAL", "SOLVE_SOME") and (same_owner or "global" in (ca.owner, cb.owner) or "vase" in (ca.owner, cb.owner)):
            if {ca.stage, cb.stage} in ({"LOSE", "FINAL"}, {"TRANSFER_OUT", "FINAL"}, {"COST_OUT", "FINAL"}, {"LOSE", "INITIAL_SOLVED"}, {"TRANSFER_OUT", "INITIAL_SOLVED"}):
                eff_o = goal.target_owner if goal.target_owner != "global" else (ca.owner if ca.owner != "global" else cb.owner)
                st_tag = "INITIAL_SOLVED" if goal.query_type == "INITIAL" else "FINAL_PLUS_LOSE"
                nr = QuantReg(reg_id=next_id, value=val, owner=eff_o, unit=ca.unit, per_unit=ca.per_unit, attr=ca.attr, stage=st_tag, action="have", clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
                return [TransitionMove("ADD", i, j, nr, 7.9, "CONSERVATION_INIT")]

        if goal.query_type == "DIFF":
            if goal.second_unit and {ra.unit, rb.unit} == {goal.target_unit, goal.second_unit}:
                return None
            if goal.target_attr and goal.second_attr and {ca.attr, cb.attr} == {goal.target_attr, goal.second_attr}:
                return None

        if goal.target_attr and ca.attr and cb.attr and ca.attr != cb.attr and goal.query_type == "COUNT":
            return None
        if ca.attr != cb.attr and (not ca.attr or not cb.attr):
            new_attr = ca.attr or cb.attr

        q_mod_penalty = 0.0
        if goal.q_focus_modifiers:
            for r_chk in (ca, cb):
                if r_chk.attr and r_chk.attr not in goal.q_focus_modifiers and r_chk.attr != "GROUP":
                    q_mod_penalty -= 3.5

        both_match_q_action = (
            goal.action_lemma not in ("have", "leave", "cost")
            and ca.action == goal.action_lemma
            and cb.action == goal.action_lemma
        )
        if both_match_q_action and (same_owner or goal.target_owner in ("GROUP", "global")):
            nr = QuantReg(reg_id=next_id, value=val, owner=new_owner, unit=ca.unit, per_unit=ca.per_unit, attr=new_attr, stage="GAIN", action=goal.action_lemma, clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 6.6 * u_bonus + q_mod_penalty, "ACTION_ACCUM")]

        if ca.stage in ("LOSE", "COST_OUT") and cb.stage in ("LOSE", "COST_OUT"):
            st_comb = "COST_OUT" if "COST_OUT" in (ca.stage, cb.stage) else "LOSE"
            nr = QuantReg(reg_id=next_id, value=val, owner=new_owner, unit=ca.unit, per_unit=ca.per_unit, attr=new_attr, stage=st_comb, action=ca.action, clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 6.4 * u_bonus + q_mod_penalty, "COST_SUM")]

        if goal.query_type == "DIFF" and "FINAL" in (ca.stage, cb.stage):
            return None
        if (ca.origin_mask & cb.origin_mask) != 0 and ("PART" in (ca.stage, cb.stage) or "COST_OUT" in (ca.stage, cb.stage)):
            return None
        if any(r.stage in ("LOSE", "TRANSFER_OUT", "COST_OUT") or (r.stage == "FINAL" and r.depth == 0) for r in (ca, cb)):
            if goal.action_lemma in ("cost", "spend", "pay") and ("COST_OUT" in (ca.stage, cb.stage)):
                return None
            nr = QuantReg(reg_id=next_id, value=val, owner=new_owner, unit=ca.unit, per_unit=ca.per_unit, attr=new_attr, stage="DERIVED", action="have", clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 2.2 * u_bonus + q_mod_penalty, "LOW_ADD")]

        if ca.per_unit is not None and ca.per_unit == cb.per_unit and same_owner:
            nr = QuantReg(reg_id=next_id, value=val, owner=ca.owner, unit=ca.unit, per_unit=ca.per_unit, attr="GROUP", stage="RATE", action=ca.action, clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("ADD", i, j, nr, 6.9 * u_bonus + q_mod_penalty, "RATE_SUM")]

        if goal.target_attr and goal.query_type in ("REMAIN", "COUNT", "DIFF") and ca.attr and cb.attr and ca.attr != cb.attr:
            return None
        score = 3.0 * u_bonus + q_mod_penalty
        ca_hyp = UNIT_HYPERNYM.get(ca.unit, ca.unit)
        g_hyp = UNIT_HYPERNYM.get(goal.target_unit, goal.target_unit)
        unit_relevant_to_goal = (
            goal.target_unit == "item"
            or ca.unit in (goal.target_unit, goal.second_unit, goal.per_unit)
            or ca_hyp == g_hyp
            or (ca.unit, goal.target_unit) in UNIT_CONVERSIONS
            or ca.unit == "dollar"
            or ca.per_unit is not None
        )
        if same_owner or "global" in (ca.owner, cb.owner):
            eff_o = ca.owner if ca.owner != "global" else cb.owner
            new_owner = eff_o
            if not unit_relevant_to_goal:
                score -= 0.5
            elif goal.target_owner not in ("global", "GROUP") and eff_o not in ("global", goal.target_owner, goal.second_owner):
                score += 1.0
            else:
                score += 3.2
        else:
            if goal.target_owner in ("GROUP", "global") or goal.query_type == "TOTAL":
                score += 3.0
            else:
                score -= 1.8

        nr = QuantReg(
            reg_id=next_id, value=val, owner=new_owner, unit=ca.unit, per_unit=ca.per_unit,
            attr=new_attr, stage="GAIN", action=ca.action if ca.action == cb.action else "have",
            clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask,
        )
        return [TransitionMove("ADD", i, j, nr, score, "STD_ADD")]

    @classmethod
    def _try_sub(
        cls,
        i: int,
        j: int,
        ra: QuantReg,
        rb: QuantReg,
        regs: Sequence[QuantReg],
        active_indices: Sequence[int],
        needs_rem: bool,
        goal: GoalSpec,
        next_id: int,
        depth: int,
    ) -> Optional[List[TransitionMove]]:
        ok, u_bonus, ca, cb = cls._units_compatible_add(ra, rb, goal, is_sub=True)
        if not ok or ca is None or cb is None:
            return None
        val = ca.value - cb.value
        if val < -1e-9:
            return None

        same_owner = ca.owner == cb.owner
        eff_owner = ca.owner if ca.owner != "global" else cb.owner
        mask = ca.origin_mask | cb.origin_mask

        if cb.stage == "COMP_LESS" and cb.ref_owner in (ca.owner, ca.attr) and ra.unit == rb.unit:
            nr = QuantReg(reg_id=next_id, value=val, owner=cb.owner, unit=ca.unit, per_unit=ca.per_unit, attr=cb.attr, stage="INIT", action=cb.action, clause_idx=cb.clause_idx, depth=depth + 1, origin_mask=mask)
            return [
                TransitionMove("SUB", i, j, nr, 7.6, "COMP_LESS_FWD", keep_a_active=False),
                TransitionMove("SUB", i, j, nr, 7.4, "COMP_LESS_FWD_KEEP", keep_a_active=True),
            ]
        if cb.stage == "POST_MULT_LESS" and ca.depth >= 1 and ca.clause_idx == cb.clause_idx:
            nr = QuantReg(reg_id=next_id, value=val, owner=ca.owner, unit=ca.unit, per_unit=ca.per_unit, attr=ca.attr, stage="GAIN", action=ca.action, clause_idx=ca.clause_idx, depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr, 8.3, "POST_MULT_SUB")]
        if "POST_MULT_LESS" in (ca.stage, cb.stage) or "POST_MULT_MORE" in (ca.stage, cb.stage):
            return None

        # Algebraic Sum-Diff: "A bat and a ball cost $110 in total. The bat costs $100 more than the ball." -> (110 - 100) / 2 = 5
        if ca.stage in ("TOTAL_GIVEN", "COST_OUT", "INIT") and cb.stage in ("COMP_MORE", "COMP_LESS") and ("total" in ca.clause_words or "together" in ca.clause_words or ca.owner == "GROUP"):
            half_sm = val / 2.0
            nr_sd = QuantReg(reg_id=next_id, value=half_sm, owner=goal.target_owner, unit=ca.unit, attr=goal.target_attr, stage="FINAL", action=goal.action_lemma, clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr_sd, 7.9, "SUM_DIFF_HALF")]

        if cb.stage == "COMP_MORE" and cb.owner == ca.owner and cb.ref_owner and cb.ref_owner != ca.owner and ra.unit == rb.unit:
            nr = QuantReg(reg_id=next_id, value=val, owner=cb.ref_owner, unit=ca.unit, per_unit=ca.per_unit, attr="", stage="INIT", action=ca.action, clause_idx=cb.clause_idx, depth=depth + 1, origin_mask=mask)
            return [
                TransitionMove("SUB", i, j, nr, 7.8, "COMP_MORE_INV", keep_a_active=False),
                TransitionMove("SUB", i, j, nr, 7.5, "COMP_MORE_INV_KEEP", keep_a_active=True),
            ]
        if any(st in ("COMP_MORE", "COMP_LESS", "DIFF_SOLVED") for st in (ca.stage, cb.stage)):
            return None

        if ca.stage in ("PAID_BILL", "TOTAL_GIVEN") and cb.stage not in ("PAID_BILL", "TOTAL_GIVEN"):
            has_pending_item_cost = any(
                k not in (i, j) and (regs[k].stage == "COST_RATE" or (regs[k].stage == "COST_OUT" and regs[k].unit == "dollar"))
                for k in active_indices
            )
            if has_pending_item_cost:
                return None
            st_b = "FINAL" if ca.stage == "PAID_BILL" else "REMAIN_POOL"
            pr_b = 7.9 if (ca.stage == "PAID_BILL" and goal.query_type == "CHANGE") or ca.stage == "TOTAL_GIVEN" else 5.5
            nr = QuantReg(reg_id=next_id, value=val, owner=goal.target_owner if goal.target_owner != "global" else eff_owner, unit=ca.unit, stage=st_b, action="get" if ca.stage == "PAID_BILL" else goal.action_lemma, clause_idx=ca.clause_idx, depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr, pr_b, "BILL_CHANGE")]

        if ca.stage == "POOL_TOTAL" and cb.stage not in ("POOL_TOTAL",):
            nr = QuantReg(reg_id=next_id, value=val, owner=eff_owner, unit=ca.unit, per_unit=ca.per_unit, attr="remaining", stage="REMAIN_POOL", action="have", clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr, 7.5, "REMAIN_AFTER_PART")]

        if ca.stage == "TARGET_GOAL" and cb.stage != "TARGET_GOAL":
            for k in active_indices:
                if k in (i, j):
                    continue
                rk = regs[k]
                if rk.unit == ca.unit and (rk.owner == cb.owner or rk.ref_owner == cb.owner or rk.stage == "MULT_ANAPHOR"):
                    return None
            nr = QuantReg(reg_id=next_id, value=val, owner=cb.owner, unit=ca.unit, stage="FINAL", action="need", clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr, 7.8 if goal.query_type == "DIFF" else 5.8, "GOAL_SHORTFALL")]

        if ca.stage == "FINAL_PLUS_LOSE" and cb.stage == "INIT":
            nr = QuantReg(reg_id=next_id, value=val, owner=goal.target_owner if goal.target_owner != "global" else eff_owner, unit=ca.unit, stage="SOME_SOLVED", action=goal.action_lemma, clause_idx=ca.clause_idx, depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr, 8.2, "SOLVE_SOME_GAIN")]

        if goal.query_type == "DIFF":
            is_fewer_q = bool(goal.question_tokens & {"fewer", "less", "lesser", "shorter", "smaller"})
            owner_pair_ok = bool(goal.second_owner and ((ca.owner == goal.second_owner and cb.owner == goal.target_owner) if is_fewer_q else (ca.owner == goal.target_owner and cb.owner == goal.second_owner)))
            attr_pair_ok = bool(goal.target_attr and goal.second_attr and ((ca.attr == goal.second_attr and cb.attr == goal.target_attr) if is_fewer_q else (ca.attr == goal.target_attr and cb.attr == goal.second_attr)))
            unit_pair_ok = bool(goal.second_unit and ((ra.unit == goal.second_unit and rb.unit == goal.target_unit) if is_fewer_q else (ra.unit == goal.target_unit and rb.unit == goal.second_unit)))
            premature = any(
                k not in (i, j) and regs[k].unit == ra.unit and regs[k].owner == ra.owner and regs[k].stage == "GAIN"
                for k in active_indices
            )
            if not premature and (owner_pair_ok or attr_pair_ok or unit_pair_ok) and (goal.target_owner == "global" or ca.owner in (goal.target_owner, "global")):
                nr = QuantReg(reg_id=next_id, value=val, owner=goal.target_owner, unit=goal.target_unit, attr=goal.target_attr, stage="DIFF_SOLVED", action=goal.action_lemma, clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
                return [TransitionMove("SUB", i, j, nr, 8.4, "GOAL_DIFF_PAIR")]

        has_later_rem_anaphor = any(
            k not in (i, j) and regs[k].is_remaining_frac and regs[k].clause_idx >= cb.clause_idx
            for k in active_indices
        )
        if needs_rem and (
            cb.stage == "PART"
            or (ca.stage == "REMAIN_POOL" and goal.has_the_rest_cue and cb.stage == "INIT")
            or (has_later_rem_anaphor and ca.depth == 0 and ca.clause_idx == 0 and cb.clause_idx > 0 and ca.unit == cb.unit)
        ) and ca.stage in ("INIT", "GAIN", "REMAIN_POOL", "DERIVED") and (ca.stage == "REMAIN_POOL" or ca.clause_idx <= cb.clause_idx):
            out_u = goal.target_unit if (ca.stage == "REMAIN_POOL" and goal.has_the_rest_cue and goal.target_unit != "item") else ca.unit
            nr = QuantReg(reg_id=next_id, value=val, owner=eff_owner, unit=out_u, per_unit=ca.per_unit, attr="remaining", stage="REMAIN_POOL", action="have", clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr, 7.1, "REMAIN_AFTER_PART")]

        if (same_owner or "global" in (ca.owner, cb.owner)) and cb.stage in ("LOSE", "TRANSFER_OUT", "COST_OUT") and ca.stage not in ("LOSE", "TRANSFER_OUT", "COST_OUT", "PAID_BILL", "TARGET_GOAL") and not (ca.attr and cb.attr and ca.attr != "GROUP" and cb.attr != "GROUP" and ca.attr != cb.attr and cb.attr not in ca.attr and ca.attr not in cb.attr):
            # 先行する割合減少 ("sold 1/4 of them ... and 15 shirts") が未適用のうちは元在庫から固定数を先に引かない
            has_earlier_frac_lose = ca.depth == 0 and any(
                k not in (i, j)
                and regs[k].stage in ("FRAC_LOSE", "PCT_LESS")
                and not regs[k].is_remaining_frac
                and ca.clause_idx <= regs[k].clause_idx <= cb.clause_idx
                and regs[k].reg_id < cb.reg_id
                for k in active_indices
            )
            if has_earlier_frac_lose:
                return None
            if not (cb.stage == "COST_OUT" and goal.action_lemma in ("spend", "pay", "cost") and goal.query_type not in ("REMAIN", "CHANGE", "PROFIT")):
                gross_act_focus = (
                    goal.query_type in ("COUNT", "TOTAL")
                    and goal.action_lemma in GROSS_EVENT_VERBS
                    and not goal.has_the_rest_cue
                    and not (goal.question_tokens & {"left", "remain", "remaining", "now", "still"})
                    and ca.unit == goal.target_unit
                    and any(r.depth == 0 and r.action == goal.action_lemma for r in regs)
                )
                if goal.query_type not in ("INITIAL", "SOLVE_SOME") and not (goal.action_lemma == cb.action and goal.action_lemma not in ("have", "leave", "earn", "pay", "buy", "make")):
                    st_res = "RATE" if ca.per_unit is not None else "GAIN"
                    dec_prior = 3.2 * u_bonus if gross_act_focus else 6.9 * u_bonus
                    nr = QuantReg(reg_id=next_id, value=val, owner=eff_owner, unit=ca.unit, per_unit=ca.per_unit, attr=ca.attr, stage=st_res, action="have", clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
                    return [TransitionMove("SUB", i, j, nr, dec_prior, "STATE_DECREASE")]

        if goal.query_type == "INITIAL" and (same_owner or "global" in (ca.owner, cb.owner)) and ca.stage in ("FINAL", "INITIAL_SOLVED") and cb.stage in ("GAIN", "TRANSFER_IN"):
            nr = QuantReg(reg_id=next_id, value=val, owner=eff_owner, unit=ca.unit, stage="INITIAL_SOLVED", action="have", clause_idx=ca.clause_idx, depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr, 7.7, "CONSERVATION_INIT_SUB")]

        if cb.depth == 0 and cb.stage == "FINAL" and ca.stage in ("INIT", "GAIN") and ca.clause_idx <= cb.clause_idx and goal.query_type not in ("INITIAL", "REMAIN"):
            if not (goal.query_type == "SOLVE_SOME" and goal.unknown_some_stage == "GAIN"):
                st_out = "SOME_SOLVED" if goal.query_type == "SOLVE_SOME" else "FINAL"
                nr = QuantReg(reg_id=next_id, value=val, owner=eff_owner, unit=ca.unit, stage=st_out, action=goal.action_lemma, clause_idx=cb.clause_idx, depth=depth + 1, origin_mask=mask)
                return [TransitionMove("SUB", i, j, nr, 7.4 * u_bonus, "DELTA_FROM_TOTAL_REMAIN")]

        if goal.query_type == "PROFIT" and ca.unit == "dollar" and cb.stage in ("LOSE", "COST_OUT"):
            nr = QuantReg(reg_id=next_id, value=val, owner=eff_owner, unit="dollar", stage="FINAL", action="make", clause_idx=max(ca.clause_idx, cb.clause_idx), depth=depth + 1, origin_mask=mask)
            return [TransitionMove("SUB", i, j, nr, 7.3, "PROFIT_SUB")]

        # 1対1不足・空席問題 ("200 seats - 145 people = 55 empty seats")
        fb_score = 4.2 * u_bonus if goal.query_type in ("DIFF", "REMAIN") else 3.0 * u_bonus
        if ra.unit != rb.unit and goal.has_the_rest_cue:
            fb_score = 5.8
        has_explicit_q_act = any(r.depth == 0 and r.action == goal.action_lemma for r in regs)
        if not has_explicit_q_act and ca.depth == 0 and cb.depth == 0 and ca.clause_idx < cb.clause_idx and goal.action_lemma in ("give", "sell", "eat", "lose", "spend", "take", "leave") and goal.query_type != "REMAIN":
            fb_score = 6.2 * u_bonus
        if not same_owner and "global" not in (ca.owner, cb.owner) and ra.unit == rb.unit:
            fb_score -= 2.5
        nr = QuantReg(
            reg_id=next_id, value=val, owner=eff_owner, unit=ca.unit, per_unit=ca.per_unit,
            stage="DERIVED", action=goal.action_lemma, clause_idx=max(ca.clause_idx, cb.clause_idx),
            depth=depth + 1, origin_mask=mask,
        )
        return [TransitionMove("SUB", i, j, nr, fb_score, "GENERIC_SUB")]

    @classmethod
    def _try_mul(
        cls, i: int, j: int, ra: QuantReg, rb: QuantReg, goal: GoalSpec, next_id: int, depth: int
    ) -> List[TransitionMove]:
        if any(st in ("COMP_MORE", "COMP_LESS", "DIFF_SOLVED") for st in (ra.stage, rb.stage)):
            return []
        if ra.unit == "dollar" and rb.unit == "dollar":
            return []

        mask = ra.origin_mask | rb.origin_mask

        for rate_r, qty_r in ((ra, rb), (rb, ra)):
            if rate_r.per_unit is not None:
                # "$2 per egg * 9 eggs/day -> $18/day" のように qty_r 自体が per_unit="day" を持っていても rate_r.per_unit == qty_r.unit なら単価キャンセル成立！
                if rate_r.per_unit == qty_r.unit or rate_r.per_unit == "group":
                    if goal.query_type == "RATE_EACH" and qty_r.unit == goal.per_unit:
                        return []
                    if rate_r.per_unit == "group" and rate_r.clause_idx != qty_r.clause_idx and qty_r.stage not in ("Q_PARAM", "TIME_PARAM"):
                        continue
                    val = rate_r.value * qty_r.value
                    exact_cancel = rate_r.per_unit == qty_r.unit
                    owner_out = rate_r.owner if rate_r.owner != "global" else qty_r.owner
                    if "PAID_BILL" in (rate_r.stage, qty_r.stage) or (rate_r.per_unit == "bill" and goal.query_type == "CHANGE"):
                        st_out = "PAID_BILL"
                    elif "COST_RATE" in (rate_r.stage, qty_r.stage) or "COST_OUT" in (rate_r.stage, qty_r.stage):
                        st_out = "COST_OUT"
                    elif "LOSE" in (rate_r.stage, qty_r.stage):
                        st_out = "LOSE"
                    else:
                        st_out = "GAIN"
                    act_out = "spend" if st_out == "COST_OUT" else rate_r.action
                    attr_out = rate_r.attr or qty_r.attr
                    # 合成された PARTITIVE_RATE_DIV 由来のレートは単価キャンセルボーナスを控えめにして偽レート発明ループを防止
                    if rate_r.depth > 0 and rate_r.per_unit not in TIME_UNITS and rate_r.per_unit not in MULTIPLIER_CONTAINER_UNITS:
                        continue
                    if rate_r.attr in ADJECTIVE_MODIFIERS and qty_r.attr in ADJECTIVE_MODIFIERS and rate_r.attr != qty_r.attr:
                        continue
                    bonus = (7.2 if (rate_r.depth == 0 or rate_r.stage == 'RATE') else 5.6) if exact_cancel else 6.2
                    if rate_r.attr and rate_r.attr == qty_r.attr:
                        bonus += 0.8
                    if rate_r.clause_idx == qty_r.clause_idx and rate_r.depth == 0:
                        bonus += 0.5
                    nr = QuantReg(
                        reg_id=next_id, value=val, owner=owner_out, unit=rate_r.unit,
                        per_unit=qty_r.per_unit, attr=attr_out, stage=st_out,
                        action=act_out, clause_idx=max(rate_r.clause_idx, qty_r.clause_idx),
                        depth=depth + 1, origin_mask=mask,
                    )
                    return [TransitionMove("MUL", i, j, nr, bonus, "RATE_CANCEL")]

                if (qty_r.unit, rate_r.per_unit) in UNIT_CONVERSIONS:
                    scale = UNIT_CONVERSIONS[(qty_r.unit, rate_r.per_unit)]
                    val = rate_r.value * (qty_r.value * scale)
                    nr = QuantReg(
                        reg_id=next_id, value=val, owner=rate_r.owner if rate_r.owner != "global" else qty_r.owner,
                        unit=rate_r.unit, per_unit=qty_r.per_unit, attr=rate_r.attr or qty_r.attr, stage="GAIN",
                        action=rate_r.action, clause_idx=max(rate_r.clause_idx, qty_r.clause_idx),
                        depth=depth + 1, origin_mask=mask,
                    )
                    return [TransitionMove("MUL", i, j, nr, 6.8, "RATE_CONV_CANCEL")]

        if "PAID_BILL" in (ra.stage, rb.stage) and "bill" not in (ra.unit, rb.unit):
            return []
        if "COST_OUT" in (ra.stage, rb.stage) and "TIME_PARAM" not in (ra.stage, rb.stage):
            return []
        if (ra.unit in TIME_UNITS or rb.unit in TIME_UNITS) and ra.per_unit is None and rb.per_unit is None:
            t_r = ra if ra.unit in TIME_UNITS else rb
            if t_r.stage == "LOSE" and t_r.action == "spend":
                return []

        a_is_mult = ra.unit in MULTIPLIER_CONTAINER_UNITS
        b_is_mult = rb.unit in MULTIPLIER_CONTAINER_UNITS

        if ra.unit == "scalar" and rb.unit != "scalar":
            content_r, mult_r = rb, ra
        elif rb.unit == "scalar" and ra.unit != "scalar":
            content_r, mult_r = ra, rb
        elif b_is_mult and not a_is_mult:
            content_r, mult_r = ra, rb
        elif a_is_mult and not b_is_mult:
            content_r, mult_r = rb, ra
        else:
            content_r, mult_r = (ra, rb) if ra.unit == goal.target_unit else (rb, ra)

        if goal.query_type == "RATE_EACH" and mult_r.unit == goal.per_unit:
            return []

        val = content_r.value * mult_r.value
        dist = abs(ra.clause_idx - rb.clause_idx)
        st_out = "COST_OUT" if "COST_OUT" in (ra.stage, rb.stage) else "GAIN"

        # 別人物の独立所持品同士 ("Allan 5 balloons * Jake 4 balloons") の直接乗算を防止
        if ra.owner not in ("global", "GROUP") and rb.owner not in ("global", "GROUP") and ra.owner != rb.owner:
            if not (a_is_mult or b_is_mult):
                return []

        same_clause_coord = (
            dist == 0
            and "and" in ra.clause_words
            and not (set(ra.clause_words) & {"each", "every", "per", "in", "on", "into", "for", "with", "of"})
        )
        if same_clause_coord:
            return []
        if (a_is_mult or b_is_mult) and ra.unit != rb.unit:
            score = 5.6 if dist == 0 else (4.9 if dist == 1 else 3.2)
            tag = "CONTAINER_TIME_MUL"
        elif ra.unit != rb.unit:
            # 非登録コンテナ ("6 monsters with 3 eyes") でもソフト許容し NNUE の語彙特徴で判断可能にする（未学習時は STEP_COST 未満で安全）
            score = 1.8 if dist <= 1 else 1.2
            tag = "SOFT_CROSS_MUL"
        else:
            if not (goal.question_tokens & {"area", "square"}) and ra.unit not in ("foot", "meter", "inch", "yard", "mile", "centimeter", "kilometer", "item"):
                return []
            score = 5.2 if (goal.question_tokens & {"area", "square"}) else 2.0
            tag = "AREA_OR_SAME_MUL"

        if goal.query_type == "RATE_EACH":
            score -= 2.5

        nr = QuantReg(
            reg_id=next_id, value=val,
            owner=content_r.owner if content_r.owner != "global" else mult_r.owner,
            unit=content_r.unit, per_unit=mult_r.per_unit, attr=content_r.attr,
            stage=st_out, action=content_r.action,
            clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask,
        )
        return [TransitionMove("MUL", i, j, nr, score, tag)]

    @classmethod
    def _try_div(
        cls, i: int, j: int, ra: QuantReg, rb: QuantReg, goal: GoalSpec, next_id: int, depth: int,
        regs: Sequence[QuantReg] = (), active_indices: Sequence[int] = (),
    ) -> Optional[List[TransitionMove]]:
        if abs(rb.value) < 1e-9 or rb.value == 1.0 or not math.isfinite(ra.value) or not math.isfinite(rb.value):
            return None
        if any(s in ("PAID_BILL", "COMP_MORE", "COMP_LESS", "DIFF_SOLVED", "TARGET_GOAL") for s in (ra.stage, rb.stage)):
            if not (ra.unit == rb.unit and rb.per_unit is not None):
                return None

        val = ra.value / rb.value
        if not math.isfinite(val) or abs(val) > 1e14:
            return None
        is_int = abs(val - round(val)) < 1e-6
        clean_q = is_int or abs(val * 4 - round(val * 4)) < 1e-6 or abs(val * 8 - round(val * 8)) < 1e-6 or abs(val * 10 - round(val * 10)) < 1e-6
        mask = ra.origin_mask | rb.origin_mask
        out_moves: List[TransitionMove] = []

        # 1) Explicit Quotitive Division: 48 cookies / (6 cookies/box) -> 8 boxes, or $17 / ($8.50/box) -> 2 boxes
        if ra.unit == rb.unit and rb.per_unit is not None:
            out_u = rb.per_unit if rb.per_unit != "group" else goal.target_unit
            if rb.stage == "COST_RATE" and goal.target_unit == "dollar" and out_u == "dollar":
                return None
            needed_downstream = any(
                k not in (i, j) and (regs[k].per_unit == out_u or regs[k].unit == out_u)
                for k in active_indices
            )
            u_hyp_match = UNIT_HYPERNYM.get(out_u, out_u) == UNIT_HYPERNYM.get(goal.target_unit, goal.target_unit)
            if clean_q and (out_u == goal.target_unit or goal.target_unit == "item" or needed_downstream or u_hyp_match):
                st_q = "GAIN" if needed_downstream and out_u != goal.target_unit else "FINAL"
                score = (7.4 if out_u == goal.target_unit else 6.8) + (0.4 if is_int else 0.0)
                nr = QuantReg(
                    reg_id=next_id, value=val, owner=ra.owner if ra.owner != "global" else goal.target_owner,
                    unit=out_u, per_unit=None, stage=st_q, action=ra.action,
                    clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask,
                )
                out_moves.append(TransitionMove("DIV", i, j, nr, score, "QUOTITIVE_DIV"))

            # Discrete Remainder (MOD), Ceiling ("needed"), and Floor ("full" / "can buy") when non-integer!
            if not is_int and ra.value > rb.value > 1.0 and abs(ra.value - round(ra.value)) < 1e-6 and abs(rb.value - round(rb.value)) < 1e-6:
                rem_v = float(int(round(ra.value)) % int(round(rb.value)))
                ceil_v = float(math.ceil(ra.value / rb.value - 1e-9))
                floor_v = float(math.floor(ra.value / rb.value + 1e-9))
                if goal.query_type == "REMAIN" and (ra.unit == goal.target_unit or goal.target_unit == "item"):
                    nr_mod = QuantReg(reg_id=next_id, value=rem_v, owner=ra.owner, unit=ra.unit, stage="FINAL", action="have", clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask)
                    out_moves.append(TransitionMove("DIV", i, j, nr_mod, 8.0, "MOD_REMAINDER"))
                if ("needed" in goal.question_tokens or "need" in goal.question_tokens or "require" in goal.question_tokens) and (out_u == goal.target_unit or goal.target_unit == "item"):
                    nr_ceil = QuantReg(reg_id=next_id, value=ceil_v, owner=ra.owner, unit=out_u, stage="FINAL", action="need", clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask)
                    out_moves.append(TransitionMove("DIV", i, j, nr_ceil, 8.0, "CEIL_DIV"))
                if ("full" in goal.question_tokens or "can" in goal.question_tokens or rb.stage in ("COST_RATE", "RATE") and ra.unit == "dollar") and (out_u == goal.target_unit or goal.target_unit == "item"):
                    nr_flr = QuantReg(reg_id=next_id, value=floor_v, owner=ra.owner if ra.owner != "global" else goal.target_owner, unit=out_u, stage="FINAL", action=goal.action_lemma, clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask)
                    out_moves.append(TransitionMove("DIV", i, j, nr_flr, 7.6, "FLOOR_DIV"))
            return out_moves or None

        if not clean_q and goal.query_type != "RATIO":
            return None

        # 2) Implicit Quotitive / Ratio Division (ra.unit == rb.unit)
        if ra.unit == rb.unit and rb.per_unit is None:
            if goal.query_type == "RATIO" and ra.value <= rb.value * 100.0:
                is_pct_q = bool(goal.question_tokens & {"percent", "percentage"})
                r_val = val * 100.0 if (is_pct_q and val <= 1.0) else val
                if is_pct_q and ra.value > rb.value:
                    return None
                nr = QuantReg(reg_id=next_id, value=r_val, owner=ra.owner, unit="scalar", stage="FINAL", clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask)
                return [TransitionMove("DIV", i, j, nr, 7.2, "RATIO_DIV")]
            if ra.value >= rb.value:
                ra_hyp = UNIT_HYPERNYM.get(ra.unit, ra.unit)
                g_hyp = UNIT_HYPERNYM.get(goal.target_unit, goal.target_unit)
                if goal.target_unit not in (ra.unit, "item") and g_hyp != ra_hyp and is_int:
                    nr = QuantReg(reg_id=next_id, value=val, owner=ra.owner, unit=goal.target_unit, stage="FINAL", clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask)
                    out_moves.append(TransitionMove("DIV", i, j, nr, 6.5, "IMPLICIT_QUOTITIVE_DIV"))
                if is_int:
                    # Proportional scaling factor ("120 pages / 8 pages = 15x -> * 20 minutes = 300 minutes")
                    has_other_unit = any(k not in (i, j) and regs[k].unit != ra.unit for k in active_indices)
                    sc_pr = 5.6 if has_other_unit else 2.2
                    nr = QuantReg(reg_id=next_id, value=val, owner=ra.owner, unit="scalar", stage="GAIN" if has_other_unit else "DERIVED", clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask)
                    out_moves.append(TransitionMove("DIV", i, j, nr, sc_pr, "SOFT_SAME_DIV"))
            return out_moves or None

        # 3) Partitive Rate Division (ra.unit != rb.unit)
        if ra.unit != rb.unit and ra.per_unit is None and rb.stage != "COST_RATE":
            if "COST_OUT" in (ra.stage, rb.stage) and goal.query_type != "RATE_EACH":
                return None
            in_mult = rb.unit in MULTIPLIER_CONTAINER_UNITS or rb.unit in TIME_UNITS
            if goal.query_type == "RATE_EACH" or (goal.query_type == "DIFF" and goal.per_unit == rb.unit):
                score = (7.5 if in_mult else 6.5) + (0.4 if is_int else 0.0)
            elif in_mult and abs(ra.clause_idx - rb.clause_idx) <= 1:
                score = 5.4 + (0.3 if is_int else 0.0)
            else:
                score = 2.4 if is_int else 1.6
            nr = QuantReg(
                reg_id=next_id, value=val, owner=ra.owner if ra.owner != "global" else rb.owner,
                unit=ra.unit, per_unit=rb.unit, stage="RATE", action=ra.action,
                clause_idx=max(ra.clause_idx, rb.clause_idx), depth=depth + 1, origin_mask=mask,
            )
            return [TransitionMove("DIV", i, j, nr, score, "PARTITIVE_RATE_DIV")]

        return None


# ==============================================================================
# 3b. ハイブリッド ALiBi-Transformer + 関係ハイウェイ評価関数 (ESRTransformerEvaluator v22)
# ==============================================================================

_SQRT_2_OVER_PI: float = math.sqrt(2.0 / math.pi)
_ALIBI_SLOPES: np.ndarray = np.array([0.50, 0.15, 0.04, 0.0], dtype=np.float32)


def _safe_is_int(v: float) -> float:
    if not math.isfinite(v) or abs(v) > 1e14:
        return 0.0
    return 1.0 if abs(v - round(v)) < 1e-6 else 0.0


def _safe_log_mag(v: float) -> float:
    if not math.isfinite(v) or abs(v) <= 0.0:
        return 0.0
    return min(1.6, math.log1p(min(abs(v), 1e14)) / 10.0)


def _gelu_fwd(x: np.ndarray) -> np.ndarray:
    x3 = x * x * x
    return (0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x3)))).astype(np.float32)


def _gelu_bwd(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    x2 = x * x
    u = _SQRT_2_OVER_PI * (x + 0.044715 * x * x2)
    t = np.tanh(u)
    dt = (1.0 - t * t) * _SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x2)
    return (dy * (0.5 * (1.0 + t) + 0.5 * x * dt)).astype(np.float32)


def _layernorm_fwd(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    mean = x.mean(axis=-1, keepdims=True)
    diff = x - mean
    var = (diff * diff).mean(axis=-1, keepdims=True)
    inv_std = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    x_hat = (diff * inv_std).astype(np.float32)
    out = (x_hat * gamma + beta).astype(np.float32)
    return out, (x_hat, inv_std, gamma)


def _layernorm_bwd(
    dy: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_hat, inv_std, gamma = cache
    d_dim = gamma.shape[0]
    dgamma = (dy * x_hat).reshape(-1, d_dim).sum(axis=0).astype(np.float32)
    dbeta = dy.reshape(-1, d_dim).sum(axis=0).astype(np.float32)
    dx_hat = dy * gamma
    dx = inv_std * (
        dx_hat
        - dx_hat.mean(axis=-1, keepdims=True)
        - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)
    )
    return dx.astype(np.float32), dgamma, dbeta


def _build_sinusoidal_pos(max_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / float(d_model)))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe * 0.25


def _mha_self_fwd(
    x: np.ndarray,
    mask: np.ndarray,
    Wqkv: np.ndarray,
    bqkv: np.ndarray,
    Wo: np.ndarray,
    bo: np.ndarray,
    n_heads: int,
    alibi_bias: Optional[np.ndarray] = None,
    drop_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, tuple]:
    B, L, D = x.shape
    Dh = D // n_heads
    scale = 1.0 / math.sqrt(float(Dh))

    qkv = x @ Wqkv + bqkv
    q, k, v = np.split(qkv, 3, axis=-1)
    qh = q.reshape(B, L, n_heads, Dh).transpose(0, 2, 1, 3)
    kh = k.reshape(B, L, n_heads, Dh).transpose(0, 2, 1, 3)
    vh = v.reshape(B, L, n_heads, Dh).transpose(0, 2, 1, 3)

    scores = (qh @ kh.transpose(0, 1, 3, 2)) * scale
    if alibi_bias is not None:
        scores = scores + alibi_bias[None, :, :L, :L]
    key_mask = mask[:, None, None, :]
    scores_masked = np.where(key_mask, scores, -1e9)
    s_max = np.max(scores_masked, axis=-1, keepdims=True)
    exp_s = np.exp(scores_masked - s_max) * key_mask
    attn_raw = (exp_s / np.maximum(exp_s.sum(axis=-1, keepdims=True), 1e-12)).astype(np.float32)

    if drop_rate > 0.0 and rng is not None:
        drop_mask = (rng.random(attn_raw.shape, dtype=np.float32) >= drop_rate).astype(np.float32) / (1.0 - drop_rate)
        attn = attn_raw * drop_mask
    else:
        drop_mask = None
        attn = attn_raw

    ctx_h = attn @ vh
    ctx = ctx_h.transpose(0, 2, 1, 3).reshape(B, L, D)
    out = (ctx @ Wo + bo).astype(np.float32)
    cache = (x, mask, qh, kh, vh, attn_raw, attn, drop_mask, ctx, Wqkv, Wo, n_heads, scale)
    return out, cache


def _mha_self_bwd(dy: np.ndarray, cache: tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, mask, qh, kh, vh, attn_raw, attn, drop_mask, ctx, Wqkv, Wo, n_heads, scale = cache
    B, L, D = x.shape
    Dh = D // n_heads

    dWo = (ctx.reshape(-1, D).T @ dy.reshape(-1, D)).astype(np.float32)
    dbo = dy.reshape(-1, D).sum(axis=0).astype(np.float32)

    dctx = (dy @ Wo.T).reshape(B, L, n_heads, Dh).transpose(0, 2, 1, 3)
    d_attn = dctx @ vh.transpose(0, 1, 3, 2)
    if drop_mask is not None:
        d_attn = d_attn * drop_mask
    dvh = attn.transpose(0, 1, 3, 2) @ dctx

    d_scores = attn_raw * (d_attn - (d_attn * attn_raw).sum(axis=-1, keepdims=True))
    d_scores = np.where(mask[:, None, None, :], d_scores, 0.0).astype(np.float32) * scale

    dqh = d_scores @ kh
    dkh = d_scores.transpose(0, 1, 3, 2) @ qh

    dq = dqh.transpose(0, 2, 1, 3).reshape(B, L, D)
    dk = dkh.transpose(0, 2, 1, 3).reshape(B, L, D)
    dv = dvh.transpose(0, 2, 1, 3).reshape(B, L, D)
    dqkv = np.concatenate([dq, dk, dv], axis=-1)

    dWqkv = (x.reshape(-1, D).T @ dqkv.reshape(-1, 3 * D)).astype(np.float32)
    dbqkv = dqkv.reshape(-1, 3 * D).sum(axis=0).astype(np.float32)
    dx = (dqkv @ Wqkv.T).astype(np.float32)
    return dx, dWqkv, dbqkv, dWo, dbo


def _mha_cross_fwd(
    q_in: np.ndarray,
    kv_mem: np.ndarray,
    kv_mask: np.ndarray,
    prob_idx: np.ndarray,
    Wq: np.ndarray,
    bq: np.ndarray,
    Wkv: np.ndarray,
    bkv: np.ndarray,
    Wo: np.ndarray,
    bo: np.ndarray,
    n_heads: int,
) -> Tuple[np.ndarray, tuple]:
    M, D = q_in.shape
    B, L, _ = kv_mem.shape
    Dh = D // n_heads
    scale = 1.0 / math.sqrt(float(Dh))

    q_proj = (q_in @ Wq + bq).reshape(M, n_heads, 1, Dh)
    kv_proj = kv_mem @ Wkv + bkv
    k_all, v_all = np.split(kv_proj, 2, axis=-1)
    kh_all = k_all.reshape(B, L, n_heads, Dh).transpose(0, 2, 1, 3)
    vh_all = v_all.reshape(B, L, n_heads, Dh).transpose(0, 2, 1, 3)

    if B == 1:
        scores = np.einsum("mhid,hld->mhil", q_proj, kh_all[0]) * scale
        k_mask = kv_mask[0][None, None, None, :]
        scores_masked = np.where(k_mask, scores, -1e9)
        s_max = np.max(scores_masked, axis=-1, keepdims=True)
        exp_s = np.exp(scores_masked - s_max) * k_mask
        attn = (exp_s / np.maximum(exp_s.sum(axis=-1, keepdims=True), 1e-12)).astype(np.float32)
        ctx_h = np.einsum("mhil,hld->mhid", attn, vh_all[0])
    else:
        kh_sel = kh_all[prob_idx]
        vh_sel = vh_all[prob_idx]
        scores = (q_proj @ kh_sel.transpose(0, 1, 3, 2)) * scale
        k_mask = kv_mask[prob_idx][:, None, None, :]
        scores_masked = np.where(k_mask, scores, -1e9)
        s_max = np.max(scores_masked, axis=-1, keepdims=True)
        exp_s = np.exp(scores_masked - s_max) * k_mask
        attn = (exp_s / np.maximum(exp_s.sum(axis=-1, keepdims=True), 1e-12)).astype(np.float32)
        ctx_h = attn @ vh_sel

    ctx = ctx_h.reshape(M, D)
    out = (ctx @ Wo + bo).astype(np.float32)
    cache = (q_in, kv_mem, kv_mask, prob_idx, q_proj, kh_all, vh_all, attn, ctx, Wq, Wkv, Wo, n_heads, scale)
    return out, cache


def _mha_cross_bwd(
    dy: np.ndarray, cache: tuple
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_in, kv_mem, kv_mask, prob_idx, q_proj, kh_all, vh_all, attn, ctx, Wq, Wkv, Wo, n_heads, scale = cache
    M, D = q_in.shape
    B, L, _ = kv_mem.shape
    Dh = D // n_heads

    dWo = (ctx.T @ dy).astype(np.float32)
    dbo = dy.sum(axis=0).astype(np.float32)

    dctx_h = (dy @ Wo.T).reshape(M, n_heads, 1, Dh)
    kh_sel = kh_all[prob_idx]
    vh_sel = vh_all[prob_idx]

    d_attn = dctx_h @ vh_sel.transpose(0, 1, 3, 2)
    dvh_sel = attn.transpose(0, 1, 3, 2) @ dctx_h

    d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
    d_scores = np.where(kv_mask[prob_idx][:, None, None, :], d_scores, 0.0).astype(np.float32) * scale

    dq_proj = (d_scores @ kh_sel).reshape(M, D)
    dkh_sel = d_scores.transpose(0, 1, 3, 2) @ q_proj

    dWq = (q_in.T @ dq_proj).astype(np.float32)
    dbq = dq_proj.sum(axis=0).astype(np.float32)
    dq_in = (dq_proj @ Wq.T).astype(np.float32)

    dkh_all = np.zeros_like(kh_all)
    dvh_all = np.zeros_like(vh_all)
    np.add.at(dkh_all, prob_idx, dkh_sel)
    np.add.at(dvh_all, prob_idx, dvh_sel)

    dk_all = dkh_all.transpose(0, 2, 1, 3).reshape(B, L, D)
    dv_all = dvh_all.transpose(0, 2, 1, 3).reshape(B, L, D)
    dkv_proj = np.concatenate([dk_all, dv_all], axis=-1)

    dWkv = (kv_mem.reshape(-1, D).T @ dkv_proj.reshape(-1, 2 * D)).astype(np.float32)
    dbkv = dkv_proj.reshape(-1, 2 * D).sum(axis=0).astype(np.float32)
    dkv_mem = (dkv_proj @ Wkv.T).astype(np.float32)
    return dq_in, dkv_mem, dWq, dbq, dWkv, dbkv, dWo, dbo


def _conv1d_k3_fwd(
    x: np.ndarray, mask: np.ndarray, Wc: np.ndarray, bc: np.ndarray
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    B, L, D = x.shape
    xm = x * mask[:, :, None]
    pad_z = np.zeros((B, 1, D), dtype=np.float32)
    xl = np.concatenate([pad_z, xm[:, :-1, :]], axis=1)
    xr = np.concatenate([xm[:, 1:, :], pad_z], axis=1)
    cat = np.concatenate([xl, xm, xr], axis=-1)
    pre = cat @ Wc + bc
    act = _gelu_fwd(pre) * mask[:, :, None]
    out = (xm + act).astype(np.float32)
    return out, (cat, pre, mask, Wc)


def _conv1d_k3_bwd(
    dy: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cat, pre, mask, Wc = cache
    B, L, _ = pre.shape
    D = pre.shape[-1]
    dym = dy * mask[:, :, None]
    dpre = _gelu_bwd(pre, dym)
    dWc = (cat.reshape(-1, 3 * D).T @ dpre.reshape(-1, D)).astype(np.float32)
    dbc = dpre.reshape(-1, D).sum(axis=0).astype(np.float32)
    dcat = dpre @ Wc.T
    dxl, dxm_c, dxr = np.split(dcat, 3, axis=-1)
    pad_z = np.zeros((B, 1, D), dtype=np.float32)
    dx_from_l = np.concatenate([dxl[:, 1:, :], pad_z], axis=1)
    dx_from_r = np.concatenate([pad_z, dxr[:, :-1, :]], axis=1)
    dx = ((dym + dxm_c + dx_from_l + dx_from_r) * mask[:, :, None]).astype(np.float32)
    return dx, dWc, dbc


def _transformer_block_fwd(
    x: np.ndarray,
    mask: np.ndarray,
    params: Dict[str, np.ndarray],
    prefix: str,
    n_heads: int,
    alibi_bias: Optional[np.ndarray] = None,
    drop_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, tuple]:
    m3 = mask[:, :, None]
    xm = x * m3
    ln1_out, ln1_cache = _layernorm_fwd(xm, params[f"{prefix}_ln1_g"], params[f"{prefix}_ln1_b"])
    attn_out, attn_cache = _mha_self_fwd(
        ln1_out, mask,
        params[f"{prefix}_Wqkv"], params[f"{prefix}_bqkv"],
        params[f"{prefix}_Wo"], params[f"{prefix}_bo"],
        n_heads=n_heads, alibi_bias=alibi_bias, drop_rate=drop_rate * 0.5, rng=rng,
    )
    if drop_rate > 0.0 and rng is not None:
        dm1 = (rng.random(attn_out.shape, dtype=np.float32) >= drop_rate).astype(np.float32) / (1.0 - drop_rate)
        attn_out = attn_out * dm1
    else:
        dm1 = None
    h1 = (xm + attn_out) * m3

    ln2_out, ln2_cache = _layernorm_fwd(h1, params[f"{prefix}_ln2_g"], params[f"{prefix}_ln2_b"])
    ff_pre = ln2_out @ params[f"{prefix}_Wff1"] + params[f"{prefix}_bff1"]
    ff_act = _gelu_fwd(ff_pre)
    if drop_rate > 0.0 and rng is not None:
        dm2 = (rng.random(ff_act.shape, dtype=np.float32) >= drop_rate).astype(np.float32) / (1.0 - drop_rate)
        ff_act_d = ff_act * dm2
    else:
        dm2 = None
        ff_act_d = ff_act
    ff_out = ff_act_d @ params[f"{prefix}_Wff2"] + params[f"{prefix}_bff2"]
    out = ((h1 + ff_out) * m3).astype(np.float32)
    cache = (mask, ln1_cache, attn_cache, dm1, ln2_out, ln2_cache, ff_pre, ff_act_d, dm2, prefix)
    return out, cache


def _transformer_block_bwd(
    dy: np.ndarray, cache: tuple, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]
) -> np.ndarray:
    mask, ln1_cache, attn_cache, dm1, ln2_out, ln2_cache, ff_pre, ff_act_d, dm2, prefix = cache
    m3 = mask[:, :, None]
    D = dy.shape[-1]
    FF = ff_pre.shape[-1]

    dym = dy * m3
    grads[f"{prefix}_Wff2"] += (ff_act_d.reshape(-1, FF).T @ dym.reshape(-1, D)).astype(np.float32)
    grads[f"{prefix}_bff2"] += dym.reshape(-1, D).sum(axis=0).astype(np.float32)

    dff_act = dym @ params[f"{prefix}_Wff2"].T
    if dm2 is not None:
        dff_act = dff_act * dm2
    dff_pre = _gelu_bwd(ff_pre, dff_act)
    grads[f"{prefix}_Wff1"] += (ln2_out.reshape(-1, D).T @ dff_pre.reshape(-1, FF)).astype(np.float32)
    grads[f"{prefix}_bff1"] += dff_pre.reshape(-1, FF).sum(axis=0).astype(np.float32)

    dln2_out = dff_pre @ params[f"{prefix}_Wff1"].T
    dh1_from_ln2, dg2, db2 = _layernorm_bwd(dln2_out, ln2_cache)
    grads[f"{prefix}_ln2_g"] += dg2
    grads[f"{prefix}_ln2_b"] += db2

    dh1 = (dym + dh1_from_ln2) * m3
    dattn_out = dh1 * dm1 if dm1 is not None else dh1
    dln1_out, dWqkv, dbqkv, dWo, dbo = _mha_self_bwd(dattn_out, attn_cache)
    grads[f"{prefix}_Wqkv"] += dWqkv
    grads[f"{prefix}_bqkv"] += dbqkv
    grads[f"{prefix}_Wo"] += dWo
    grads[f"{prefix}_bo"] += dbo

    dx_from_ln1, dg1, db1 = _layernorm_bwd(dln1_out, ln1_cache)
    grads[f"{prefix}_ln1_g"] += dg1
    grads[f"{prefix}_ln1_b"] += db1
    return ((dh1 + dx_from_ln1) * m3).astype(np.float32)


_CORE_VOCAB_WORDS: Tuple[str, ...] = (
    "how", "many", "much", "what", "which", "who", "when", "where", "find", "calculate",
    "is", "are", "was", "were", "be", "do", "does", "did", "will", "would", "can", "could",
    "have", "has", "had", "there", "in", "on", "at", "for", "from", "to", "with", "by",
    "into", "onto", "during", "after", "before", "until", "under", "over", "between", "among",
    "of", "the", "a", "an", "and", "but", "or", "if", "while", "than", "as", "that", "not",
    "more", "less", "lesser", "fewer", "older", "younger", "taller", "shorter", "longer", "farther",
    "faster", "heavier", "lighter", "bigger", "smaller", "left", "remain", "remaining", "remainder",
    "rest", "other", "others", "another", "each", "every", "per", "apiece", "average", "equally",
    "equal", "split", "share", "shared", "divide", "divided", "groups", "group", "total",
    "altogether", "all", "combined", "together", "sum", "difference", "times", "twice", "double",
    "doubled", "triple", "tripled", "thrice", "half", "halved", "quarter", "dozen", "pair",
    "percent", "percentage", "discount", "off", "tax", "tip", "interest", "profit", "change",
    "first", "start", "started", "begin", "began", "originally", "original", "initially", "now", "still",
    "already", "also", "too", "only", "enough", "empty", "full", "whole", "broken", "extra",
    "new", "old", "big", "large", "small", "morning", "afternoon", "evening", "night",
    "yesterday", "today", "tomorrow", "daily", "weekly", "monthly", "some", "both", "needed",
    "buy", "sell", "give", "get", "eat", "drink", "spend", "pay", "cost", "earn",
    "make", "bake", "cook", "pick", "lose", "break", "use", "take", "bring", "catch",
    "collect", "plant", "read", "write", "walk", "run", "drive", "save", "leave", "pack",
    "produce", "print", "cut", "throw", "add", "remove", "organize", "play", "jump", "lay",
    "serve", "need", "want", "owe", "charge", "hold", "contain", "require", "weigh", "miss",
)
_CORE_WORD_TO_ID: Dict[str, int] = {w: idx + 32 for idx, w in enumerate(_CORE_VOCAB_WORDS)}

_STAGE_LIST: Tuple[str, ...] = (
    "INIT", "GAIN", "LOSE", "RATE", "COST_RATE", "COST_OUT", "TRANSFER_OUT", "TRANSFER_IN",
    "COMP_MORE", "COMP_LESS", "MULT_ANAPHOR", "FRAC", "FRAC_LOSE", "PCT_LESS", "PCT_MORE", "PCT_OF",
    "PAID_BILL", "TARGET_GOAL", "TOTAL_GIVEN", "POOL_TOTAL", "RATIO_PART", "TIME_PARAM",
    "REMAIN_POOL", "DIFF_SOLVED", "INITIAL_SOLVED", "SOME_SOLVED", "FINAL", "PART", "DERIVED",
    "Q_PARAM", "POST_MULT_MORE", "POST_MULT_LESS", "FINAL_PLUS_LOSE",
)
_STAGE_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(_STAGE_LIST)}

_OP_LIST: Tuple[str, ...] = ("NONE", "ADD", "SUB", "MUL", "DIV", "PCT_SUB", "PCT_ADD", "PCT_OF")
_OP_TO_ID: Dict[str, int] = {op: i for i, op in enumerate(_OP_LIST)}

_QTYPE_LIST: Tuple[str, ...] = (
    "COUNT", "TOTAL", "REMAIN", "DIFF", "RATE_EACH", "INITIAL", "SOLVE_SOME", "CHANGE", "PROFIT", "RATIO"
)
_QTYPE_TO_ID: Dict[str, int] = {q: i for i, q in enumerate(_QTYPE_LIST)}

_RULE_LIST: Tuple[str, ...] = (
    "GENERIC", "STD_ADD", "LOW_ADD", "ACTION_ACCUM", "COST_SUM", "RATE_SUM",
    "COMP_MORE_FWD", "COMP_MORE_FWD_KEEP", "COMP_LESS_INV", "COMP_LESS_INV_KEEP",
    "COMP_LESS_FWD", "COMP_LESS_FWD_KEEP", "COMP_MORE_INV", "COMP_MORE_INV_KEEP",
    "TRANSFER_RECV", "TRANSFER_RECV_SUM", "CONSERVATION_INIT", "CONSERVATION_INIT_SUB",
    "POST_MULT_ADD", "POST_MULT_SUB", "STATE_DECREASE", "REMAIN_AFTER_PART",
    "GOAL_DIFF_PAIR", "BILL_CHANGE", "GOAL_SHORTFALL", "SOLVE_SOME_GAIN",
    "DELTA_FROM_TOTAL_REMAIN", "PROFIT_SUB", "GENERIC_SUB", "SUM_DIFF_HALF",
    "RATE_CANCEL", "RATE_CONV_CANCEL", "CONTAINER_TIME_MUL", "SOFT_CROSS_MUL", "AREA_OR_SAME_MUL",
    "ANAPHOR_SCALE_KEEP", "ANAPHOR_SCALE_REPLACE", "ANAPHOR_INPLACE_ONLY", "ANAPHOR_FROM_TARGET",
    "ANAPHOR_DIV_INV", "ANAPHOR_DIV_INV_KEEP", "SUM_RATIO_SPLIT", "RATIO_SPLIT_A", "RATIO_SPLIT_B",
    "REM_ANAPHOR_DIRECT", "REM_ANAPHOR_KEEP", "FRAC_LOSE_APPLY", "FRAC_APPLY", "FRAC_DIRECT",
    "PCT_DISCOUNT", "PCT_INV_DIV", "PCT_MARKUP", "PCT_MARKUP_KEEP", "PCT_MARKUP_KEEP_COST", "PCT_DIRECT", "PCT_PART", "PCT_COMP_KEEP",
    "QUOTITIVE_DIV", "IMPLICIT_QUOTITIVE_DIV", "PARTITIVE_RATE_DIV", "RATIO_DIV", "SOFT_SAME_DIV",
    "MOD_REMAINDER", "CEIL_DIV", "FLOOR_DIV",
)
_RULE_TO_ID: Dict[str, int] = {r: i for i, r in enumerate(_RULE_LIST)}


@dataclass
class ProblemContext:
    tok_ids: np.ndarray
    seg_ids: np.ndarray
    tok_mask: np.ndarray
    q_mask: np.ndarray
    leaf_positions: List[int]
    H_text: Optional[np.ndarray] = None
    h_question: Optional[np.ndarray] = None


class ESRTransformerEvaluator:
    """
    v22 ハイブリッド評価関数:
      1) ALiBi Multi-Scale Text Transformer (Sinusoidal Pos + 4-scale ALiBi + k=3 Local Conv1D + 2x Pre-LN MHSA)
      2) Directed 3-Token Span Anchor + Register-Set Self-Attention (未消費レジスタ競合状態の順列不変把握)
      3) Multi-Head Cross-Attention + Bilinear Residual MLP (Transition & Terminal Dual Heads)
      4) Delexicalized Relational Highway (65,536バケットの役割抽象化共起特徴による高速凸学習スキップ結合)
      5) Tanh-Bounded Residual Calibration (未知語に対するロジット暴走を抑え Prior との協調を保証)
    """

    ARCH_VERSION = "esr_transformer_v22_0"

    VOCAB_SIZE = 4096
    MAX_SEQ_LEN = 144
    MAX_ACTIVE_REGS = 20
    MAX_HIST_LEN = 8

    REL_BUCKETS = 65536
    TRANS_REL_K = 44
    TERM_REL_K = 32

    TRANS_DENSE = 20
    TERM_DENSE = 12
    REG_DENSE = 14
    DELTA_BOUND = 6.5

    NUM_STAGES = len(_STAGE_LIST) + 2
    NUM_OPS = len(_OP_LIST) + 2
    NUM_QTYPES = len(_QTYPE_LIST) + 2
    NUM_RULES = len(_RULE_LIST) + 2

    def __init__(
        self,
        seed: int = 42,
        d_model: int = 64,
        n_heads: int = 4,
        n_text_layers: int = 2,
        d_ff: Optional[int] = None,
    ) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_text_layers = n_text_layers
        self.d_ff = d_ff if d_ff is not None else (d_model * 2)
        self.is_trained: bool = False

        # Deterministic Sinusoidal PosEnc & Multi-Scale ALiBi Distance Bias
        self.fixed_pos_enc = _build_sinusoidal_pos(self.MAX_SEQ_LEN, self.d_model)
        pos_idx = np.arange(self.MAX_SEQ_LEN, dtype=np.float32)
        dist_mat = np.abs(pos_idx[:, None] - pos_idx[None, :])
        slopes = np.resize(_ALIBI_SLOPES, (self.n_heads,)).astype(np.float32)
        self.alibi_bias = (-slopes[:, None, None] * dist_mat[None, :, :]).astype(np.float32)

        rng = np.random.default_rng(seed)
        D = self.d_model
        FF = self.d_ff
        s_emb = 0.04
        s_lin = 1.0 / math.sqrt(float(D))

        p: Dict[str, np.ndarray] = {}

        # 1. Text Transformer パラメータ
        p["E_tok"] = (rng.normal(0.0, s_emb, (self.VOCAB_SIZE, D))).astype(np.float32)
        p["E_tok"][0] = 0.0
        p["E_seg"] = (rng.normal(0.0, 0.02, (4, D))).astype(np.float32)
        p["Wc_tok"] = (rng.normal(0.0, 1.0 / math.sqrt(3.0 * D), (3 * D, D))).astype(np.float32)
        p["bc_tok"] = np.zeros(D, dtype=np.float32)

        for l_idx in range(self.n_text_layers):
            px = f"txt_b{l_idx}"
            p[f"{px}_ln1_g"] = np.ones(D, dtype=np.float32)
            p[f"{px}_ln1_b"] = np.zeros(D, dtype=np.float32)
            p[f"{px}_Wqkv"] = (rng.normal(0.0, s_lin, (D, 3 * D))).astype(np.float32)
            p[f"{px}_bqkv"] = np.zeros(3 * D, dtype=np.float32)
            p[f"{px}_Wo"] = (rng.normal(0.0, s_lin * 0.5, (D, D))).astype(np.float32)
            p[f"{px}_bo"] = np.zeros(D, dtype=np.float32)
            p[f"{px}_ln2_g"] = np.ones(D, dtype=np.float32)
            p[f"{px}_ln2_b"] = np.zeros(D, dtype=np.float32)
            p[f"{px}_Wff1"] = (rng.normal(0.0, s_lin, (D, FF))).astype(np.float32)
            p[f"{px}_bff1"] = np.zeros(FF, dtype=np.float32)
            p[f"{px}_Wff2"] = (rng.normal(0.0, 0.5 / math.sqrt(float(FF)), (FF, D))).astype(np.float32)
            p[f"{px}_bff2"] = np.zeros(D, dtype=np.float32)

        p["txt_ln_f_g"] = np.ones(D, dtype=np.float32)
        p["txt_ln_f_b"] = np.zeros(D, dtype=np.float32)
        p["w_qpool"] = (rng.normal(0.0, s_lin, (D,))).astype(np.float32)

        # 2. Register-Set Self-Attention パラメータ
        p["E_stage"] = (rng.normal(0.0, s_emb, (self.NUM_STAGES, D))).astype(np.float32)
        p["E_urel"] = (rng.normal(0.0, s_emb, (8, D))).astype(np.float32)
        p["E_orel"] = (rng.normal(0.0, s_emb, (6, D))).astype(np.float32)
        p["E_prel"] = (rng.normal(0.0, s_emb, (6, D))).astype(np.float32)
        p["E_arel"] = (rng.normal(0.0, s_emb, (4, D))).astype(np.float32)
        p["W_reg_dn"] = (rng.normal(0.0, 0.08, (self.REG_DENSE, D))).astype(np.float32)
        p["b_reg_dn"] = np.zeros(D, dtype=np.float32)
        p["W_span"] = (rng.normal(0.0, s_lin, (D, D))).astype(np.float32)
        p["b_span"] = np.zeros(D, dtype=np.float32)

        px = "reg_b0"
        p[f"{px}_ln1_g"] = np.ones(D, dtype=np.float32)
        p[f"{px}_ln1_b"] = np.zeros(D, dtype=np.float32)
        p[f"{px}_Wqkv"] = (rng.normal(0.0, s_lin, (D, 3 * D))).astype(np.float32)
        p[f"{px}_bqkv"] = np.zeros(3 * D, dtype=np.float32)
        p[f"{px}_Wo"] = (rng.normal(0.0, s_lin * 0.5, (D, D))).astype(np.float32)
        p[f"{px}_bo"] = np.zeros(D, dtype=np.float32)
        p[f"{px}_ln2_g"] = np.ones(D, dtype=np.float32)
        p[f"{px}_ln2_b"] = np.zeros(D, dtype=np.float32)
        p[f"{px}_Wff1"] = (rng.normal(0.0, s_lin, (D, FF))).astype(np.float32)
        p[f"{px}_bff1"] = np.zeros(FF, dtype=np.float32)
        p[f"{px}_Wff2"] = (rng.normal(0.0, 0.5 / math.sqrt(float(FF)), (FF, D))).astype(np.float32)
        p[f"{px}_bff2"] = np.zeros(D, dtype=np.float32)
        p["reg_ln_f_g"] = np.ones(D, dtype=np.float32)
        p["reg_ln_f_b"] = np.zeros(D, dtype=np.float32)

        # 3. Transition Head + Delexicalized Relational Highway
        p["E_op"] = (rng.normal(0.0, s_emb, (self.NUM_OPS, D))).astype(np.float32)
        p["E_rule"] = (rng.normal(0.0, s_emb, (self.NUM_RULES, D))).astype(np.float32)
        p["E_qtype_tr"] = (rng.normal(0.0, s_emb, (self.NUM_QTYPES, D))).astype(np.float32)
        p["E_newstage"] = (rng.normal(0.0, s_emb, (self.NUM_STAGES, D))).astype(np.float32)

        tr_in_dim = 5 * D + self.TRANS_DENSE
        p["W_tr_qproj"] = (rng.normal(0.0, 1.0 / math.sqrt(float(tr_in_dim)), (tr_in_dim, D))).astype(np.float32)
        p["b_tr_qproj"] = np.zeros(D, dtype=np.float32)
        p["tr_lnq_g"] = np.ones(D, dtype=np.float32)
        p["tr_lnq_b"] = np.zeros(D, dtype=np.float32)

        p["tr_ca_Wq"] = (rng.normal(0.0, s_lin, (D, D))).astype(np.float32)
        p["tr_ca_bq"] = np.zeros(D, dtype=np.float32)
        p["tr_ca_Wkv"] = (rng.normal(0.0, s_lin, (D, 2 * D))).astype(np.float32)
        p["tr_ca_bkv"] = np.zeros(2 * D, dtype=np.float32)
        p["tr_ca_Wo"] = (rng.normal(0.0, s_lin * 0.5, (D, D))).astype(np.float32)
        p["tr_ca_bo"] = np.zeros(D, dtype=np.float32)

        p["W_tr_m1"] = (rng.normal(0.0, 1.0 / math.sqrt(float(4 * D)), (4 * D, FF))).astype(np.float32)
        p["b_tr_m1"] = np.zeros(FF, dtype=np.float32)
        p["tr_lnm1_g"] = np.ones(FF, dtype=np.float32)
        p["tr_lnm1_b"] = np.zeros(FF, dtype=np.float32)
        p["W_tr_m2"] = (rng.normal(0.0, 1.0 / math.sqrt(float(FF)), (FF, D))).astype(np.float32)
        p["b_tr_m2"] = np.zeros(D, dtype=np.float32)
        p["w_tr_out"] = np.zeros(D, dtype=np.float32)
        p["b_tr_out"] = np.zeros(1, dtype=np.float32)
        p["W_rel_tr"] = np.zeros(self.REL_BUCKETS, dtype=np.float32)

        # 4. Terminal Head + Delexicalized Relational Highway
        p["E_hist_op"] = (rng.normal(0.0, s_emb, (self.NUM_OPS, D))).astype(np.float32)
        p["E_hist_op"][0] = 0.0
        p["E_hist_pos"] = (rng.normal(0.0, 0.02, (self.MAX_HIST_LEN, D))).astype(np.float32)
        p["W_hist"] = (rng.normal(0.0, 1.0 / math.sqrt(float(self.MAX_HIST_LEN * D)), (self.MAX_HIST_LEN * D, D))).astype(np.float32)
        p["b_hist"] = np.zeros(D, dtype=np.float32)
        p["E_qtype_tm"] = (rng.normal(0.0, s_emb, (self.NUM_QTYPES, D))).astype(np.float32)

        tm_in_dim = 5 * D + self.TERM_DENSE
        p["W_tm_qproj"] = (rng.normal(0.0, 1.0 / math.sqrt(float(tm_in_dim)), (tm_in_dim, D))).astype(np.float32)
        p["b_tm_qproj"] = np.zeros(D, dtype=np.float32)
        p["tm_lnq_g"] = np.ones(D, dtype=np.float32)
        p["tm_lnq_b"] = np.zeros(D, dtype=np.float32)

        p["tm_ca_Wq"] = (rng.normal(0.0, s_lin, (D, D))).astype(np.float32)
        p["tm_ca_bq"] = np.zeros(D, dtype=np.float32)
        p["tm_ca_Wkv"] = (rng.normal(0.0, s_lin, (D, 2 * D))).astype(np.float32)
        p["tm_ca_bkv"] = np.zeros(2 * D, dtype=np.float32)
        p["tm_ca_Wo"] = (rng.normal(0.0, s_lin * 0.5, (D, D))).astype(np.float32)
        p["tm_ca_bo"] = np.zeros(D, dtype=np.float32)

        p["W_tm_m1"] = (rng.normal(0.0, 1.0 / math.sqrt(float(3 * D)), (3 * D, FF))).astype(np.float32)
        p["b_tm_m1"] = np.zeros(FF, dtype=np.float32)
        p["tm_lnm1_g"] = np.ones(FF, dtype=np.float32)
        p["tm_lnm1_b"] = np.zeros(FF, dtype=np.float32)
        p["W_tm_m2"] = (rng.normal(0.0, 1.0 / math.sqrt(float(FF)), (FF, D))).astype(np.float32)
        p["b_tm_m2"] = np.zeros(D, dtype=np.float32)
        p["w_tm_out"] = np.zeros(D, dtype=np.float32)
        p["b_tm_out"] = np.zeros(1, dtype=np.float32)
        p["W_rel_tm"] = np.zeros(self.REL_BUCKETS, dtype=np.float32)

        self.params: Dict[str, np.ndarray] = p

    def parameter_count(self) -> int:
        return int(sum(arr.size for arr in self.params.values()))

    def save(self, path: str = MODEL_WEIGHTS_FILE) -> None:
        meta = np.array([self.d_model, self.n_heads, self.n_text_layers, self.d_ff], dtype=np.int32)
        arch = np.array([self.ARCH_VERSION])
        np.savez_compressed(path, __arch__=arch, __meta__=meta, **self.params)
        self.is_trained = True

    def load(self, path: str = MODEL_WEIGHTS_FILE) -> bool:
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=False)
            if "__arch__" not in data or str(data["__arch__"][0]) != self.ARCH_VERSION:
                return False
            meta = data["__meta__"]
            if int(meta[0]) != self.d_model or int(meta[1]) != self.n_heads or int(meta[2]) != self.n_text_layers:
                self.__init__(
                    seed=42,
                    d_model=int(meta[0]),
                    n_heads=int(meta[1]),
                    n_text_layers=int(meta[2]),
                    d_ff=int(meta[3]),
                )
            for k in self.params:
                if k not in data or data[k].shape != self.params[k].shape:
                    return False
                self.params[k] = data[k].astype(np.float32)
            self.params["E_tok"][0] = 0.0
            self.params["E_hist_op"][0] = 0.0
            self.params["W_rel_tr"][0] = 0.0
            self.params["W_rel_tm"][0] = 0.0
            self.is_trained = True
            return True
        except Exception:
            return False

    @staticmethod
    def _delex_token(w: str, ra: QuantReg, rb: QuantReg, goal: GoalSpec) -> str:
        if w == goal.target_unit:
            return "<UQ>"
        if goal.second_unit and w == goal.second_unit:
            return "<UQ2>"
        if w == ra.unit:
            return "<UA>"
        if w == rb.unit:
            return "<UB>"
        if w in goal.known_entities:
            return "<PQ>" if w == goal.target_owner else "<PO>"
        return w

    @classmethod
    def _word_to_token_id(cls, raw_w: str, goal: GoalSpec, is_question: bool = False) -> int:
        clean = raw_w.strip(".,?!:;\"'()[]{}")
        if not clean:
            return 0
        low = clean.lower()
        if low.endswith("'s"):
            low = low[:-2]
        if re.fullmatch(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+/\d+|number\d+", low):
            if clean.startswith("$"):
                return 10
            if clean.endswith("%"):
                return 11
            if "/" in clean:
                return 12
            return 13 if is_question else 9

        if low in goal.known_entities or low in ROLE_OWNERS:
            return 7 if low == goal.target_owner else 8

        lem_n = lemmatize_noun(low)
        if lem_n == goal.target_unit and goal.target_unit != "item":
            return 5
        if goal.second_unit and lem_n == goal.second_unit:
            return 6
        g_hyp = UNIT_HYPERNYM.get(goal.target_unit, "")
        if g_hyp and goal.target_unit in CATEGORY_TARGET_UNITS and UNIT_HYPERNYM.get(lem_n, "") == g_hyp:
            return 16

        if low in _CORE_WORD_TO_ID:
            return _CORE_WORD_TO_ID[low]
        lem_v = lemmatize_verb(low)
        if lem_v in _CORE_WORD_TO_ID:
            return _CORE_WORD_TO_ID[lem_v]
        if lem_n in _CORE_WORD_TO_ID:
            return _CORE_WORD_TO_ID[lem_n]
        if lem_n in TIME_UNITS:
            return 14
        if lem_n in MULTIPLIER_CONTAINER_UNITS:
            return 15
        return 512 + (_fnv_hash(f"VOC|{lem_n or low}", cls.VOCAB_SIZE - 512))

    @classmethod
    def tokenize_problem(
        cls,
        regs: Sequence[QuantReg],
        clauses: Sequence[ClauseInfo],
        goal: GoalSpec,
    ) -> ProblemContext:
        L = cls.MAX_SEQ_LEN
        max_body_len = L - 28

        # 全ての節トークンを構築し、長文 GSM8K で万一上限を超える場合は質問直前の後半節を優先保持
        clause_tok_blocks: List[Tuple[int, List[int], List[int]]] = []
        for ci in clauses:
            c_ids: List[int] = []
            c_num_offs: List[int] = []
            for raw_w in ci.norm_text.split():
                tid = cls._word_to_token_id(raw_w, goal, is_question=False)
                if tid == 0:
                    continue
                off = len(c_ids)
                if tid in (9, 10, 11, 12) or raw_w.lower().strip(".,?!") in ("twice", "double", "doubled", "triple", "thrice", "half", "halved"):
                    c_num_offs.append(off)
                c_ids.append(tid)
            c_ids.append(2)  # <SEP>
            clause_tok_blocks.append((ci.idx, c_ids, c_num_offs))

        total_body_toks = sum(len(b[1]) for b in clause_tok_blocks)
        body_toks: List[int] = [1]  # <CLS>
        clause_num_positions: Dict[int, List[int]] = {}
        clause_span_mid: Dict[int, int] = {}

        skip_budget = max(0, total_body_toks - (max_body_len - 1))
        for c_idx, c_ids, c_offs in clause_tok_blocks:
            if skip_budget > 0 and len(c_ids) > 6:
                trim = min(skip_budget, len(c_ids) - 6)
                c_ids = c_ids[trim:]
                c_offs = [max(0, o - trim) for o in c_offs]
                skip_budget -= trim
            c_start = len(body_toks)
            avail = max_body_len - c_start
            if avail <= 0:
                clause_num_positions[c_idx] = [max(0, len(body_toks) - 1)]
                clause_span_mid[c_idx] = max(0, len(body_toks) - 1)
                continue
            kept = c_ids[:avail]
            body_toks.extend(kept)
            clause_num_positions[c_idx] = [min(c_start + o, len(body_toks) - 1) for o in c_offs]
            clause_span_mid[c_idx] = min(max(0, (c_start + len(body_toks) - 1) // 2), len(body_toks) - 1)

        q_seq_ids: List[int] = [3]  # <Q_START>
        q_words = list(goal.question_seq) if getattr(goal, "question_seq", ()) else sorted(goal.question_tokens)
        for qw in q_words[:25]:
            tid = cls._word_to_token_id(qw, goal, is_question=True)
            if tid > 0:
                q_seq_ids.append(tid)
        q_seq_ids.append(4)  # <EOS>

        q_start_idx = len(body_toks)
        full_ids = (body_toks + q_seq_ids)[:L]
        full_segs = ([0] * len(body_toks) + [1] * len(q_seq_ids))[:L]
        actual_len = len(full_ids)

        tok_ids = np.zeros(L, dtype=np.int32)
        seg_ids = np.zeros(L, dtype=np.int32)
        tok_mask = np.zeros(L, dtype=bool)
        q_mask = np.zeros(L, dtype=bool)

        tok_ids[:actual_len] = full_ids
        seg_ids[:actual_len] = full_segs
        tok_mask[:actual_len] = True
        q_mask[min(q_start_idx, actual_len - 1) : actual_len] = True

        clause_reg_counters: Dict[int, int] = {}
        leaf_positions: List[int] = []
        for r in regs:
            c_idx = r.clause_idx
            k_ord = clause_reg_counters.get(c_idx, 0)
            clause_reg_counters[c_idx] = k_ord + 1
            pos_list = clause_num_positions.get(c_idx, [])
            if k_ord < len(pos_list):
                p_tok = pos_list[k_ord]
            elif pos_list:
                p_tok = pos_list[-1]
            elif c_idx in clause_span_mid:
                p_tok = clause_span_mid[c_idx]
            elif r.stage in ("Q_PARAM", "PAID_BILL") and c_idx >= len(clauses):
                p_tok = min(q_start_idx + 1, actual_len - 1)
            else:
                p_tok = 0
            leaf_positions.append(int(min(max(0, p_tok), actual_len - 1)))

        return ProblemContext(
            tok_ids=tok_ids,
            seg_ids=seg_ids,
            tok_mask=tok_mask,
            q_mask=q_mask,
            leaf_positions=leaf_positions,
        )

    def forward_text_batch(
        self,
        tok_ids: np.ndarray,
        seg_ids: np.ndarray,
        tok_mask: np.ndarray,
        q_mask: np.ndarray,
        drop_rate: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, tuple]:
        B, L = tok_ids.shape
        p = self.params
        x0 = (p["E_tok"][tok_ids] + self.fixed_pos_enc[:L][None, :, :] + p["E_seg"][seg_ids]) * tok_mask[:, :, None]
        h_conv, conv_cache = _conv1d_k3_fwd(x0, tok_mask, p["Wc_tok"], p["bc_tok"])

        h = h_conv
        blk_caches = []
        for l_idx in range(self.n_text_layers):
            h, bc = _transformer_block_fwd(
                h, tok_mask, p, f"txt_b{l_idx}", self.n_heads,
                alibi_bias=self.alibi_bias, drop_rate=drop_rate, rng=rng,
            )
            blk_caches.append(bc)

        H_text_raw, lnf_cache = _layernorm_fwd(h, p["txt_ln_f_g"], p["txt_ln_f_b"])
        H_text = (H_text_raw * tok_mask[:, :, None]).astype(np.float32)

        q_logits = H_text @ p["w_qpool"]
        q_logits_masked = np.where(q_mask, q_logits, -1e9)
        q_max = np.max(q_logits_masked, axis=-1, keepdims=True)
        q_exp = np.exp(q_logits_masked - q_max) * q_mask
        q_weights = (q_exp / np.maximum(q_exp.sum(axis=-1, keepdims=True), 1e-12)).astype(np.float32)
        h_question = (q_weights[:, :, None] * H_text).sum(axis=1).astype(np.float32)

        cache = (tok_ids, seg_ids, tok_mask, q_mask, conv_cache, blk_caches, lnf_cache, H_text, q_weights)
        return H_text, h_question, cache

    def backward_text_batch(
        self,
        dH_text_in: np.ndarray,
        dh_question: np.ndarray,
        cache: tuple,
        grads: Dict[str, np.ndarray],
    ) -> None:
        tok_ids, seg_ids, tok_mask, q_mask, conv_cache, blk_caches, lnf_cache, H_text, q_weights = cache
        B, L, D = H_text.shape
        p = self.params

        dH = dH_text_in + q_weights[:, :, None] * dh_question[:, None, :]
        dq_weights = (H_text * dh_question[:, None, :]).sum(axis=-1)
        dq_logits = q_weights * (dq_weights - (dq_weights * q_weights).sum(axis=-1, keepdims=True))
        dq_logits = np.where(q_mask, dq_logits, 0.0).astype(np.float32)

        grads["w_qpool"] += (H_text.reshape(-1, D).T @ dq_logits.reshape(-1)).astype(np.float32)
        dH = (dH + dq_logits[:, :, None] * p["w_qpool"][None, None, :]) * tok_mask[:, :, None]

        dh, dg_f, db_f = _layernorm_bwd(dH, lnf_cache)
        grads["txt_ln_f_g"] += dg_f
        grads["txt_ln_f_b"] += db_f
        dh = dh * tok_mask[:, :, None]

        for l_idx in range(self.n_text_layers - 1, -1, -1):
            dh = _transformer_block_bwd(dh, blk_caches[l_idx], p, grads)

        dx0, dWc, dbc = _conv1d_k3_bwd(dh, conv_cache)
        grads["Wc_tok"] += dWc
        grads["bc_tok"] += dbc

        dx0 = dx0 * tok_mask[:, :, None]
        np.add.at(grads["E_seg"], seg_ids.ravel(), dx0.reshape(-1, D))

        flat_tok = tok_ids.ravel()
        valid = flat_tok > 0
        if np.any(valid):
            np.add.at(grads["E_tok"], flat_tok[valid], dx0.reshape(-1, D)[valid])
        grads["E_tok"][0] = 0.0

    def encode_problem_context(
        self, regs: Sequence[QuantReg], clauses: Sequence[ClauseInfo], goal: GoalSpec
    ) -> ProblemContext:
        ctx = self.tokenize_problem(regs, clauses, goal)
        H_text, h_q, _ = self.forward_text_batch(
            ctx.tok_ids[None, :], ctx.seg_ids[None, :], ctx.tok_mask[None, :], ctx.q_mask[None, :], drop_rate=0.0
        )
        ctx.H_text = H_text
        ctx.h_question = h_q
        return ctx

    @classmethod
    def featurize_board_state(
        cls,
        regs: Sequence[QuantReg],
        consumed_mask: int,
        goal: GoalSpec,
        ctx: ProblemContext,
    ) -> Tuple[Dict[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R = cls.MAX_ACTIVE_REGS
        L = cls.MAX_SEQ_LEN
        active_indices = [
            i for i in range(len(regs))
            if not ((consumed_mask >> i) & 1) and math.isfinite(regs[i].value)
        ][:R]
        if not active_indices:
            active_indices = [len(regs) - 1]

        slot_map: Dict[int, int] = {r_idx: slot for slot, r_idx in enumerate(active_indices)}
        reg_cat = np.zeros((R, 5), dtype=np.int32)
        reg_dense = np.zeros((R, cls.REG_DENSE), dtype=np.float32)
        reg_anchor = np.zeros((R, L), dtype=np.float32)
        reg_mask = np.zeros(R, dtype=bool)

        vals = [regs[i].value for i in active_indices if math.isfinite(regs[i].value)]
        max_v = max(vals) if vals else 1.0
        sorted_vals = sorted(vals) if vals else [0.0]
        n_act = max(1, len(sorted_vals))
        g_hyp = UNIT_HYPERNYM.get(goal.target_unit, goal.target_unit)
        n_leaves = len(ctx.leaf_positions)
        max_valid_tok = int(np.sum(ctx.tok_mask)) - 1

        for slot, r_idx in enumerate(active_indices):
            r = regs[r_idx]
            reg_mask[slot] = True

            st_id = _STAGE_TO_ID.get(r.stage, len(_STAGE_LIST))
            r_hyp = UNIT_HYPERNYM.get(r.unit, r.unit)
            if r.unit == goal.target_unit and goal.target_unit != "item":
                u_rel = 1
            elif goal.second_unit and r.unit == goal.second_unit:
                u_rel = 2
            elif goal.target_unit in CATEGORY_TARGET_UNITS and r_hyp == g_hyp:
                u_rel = 3
            elif (r.unit, goal.target_unit) in UNIT_CONVERSIONS:
                u_rel = 4
            elif r.unit == "dollar":
                u_rel = 5
            elif r.unit in TIME_UNITS:
                u_rel = 6
            elif r.unit in MULTIPLIER_CONTAINER_UNITS:
                u_rel = 7
            else:
                u_rel = 0

            if r.owner == goal.target_owner and goal.target_owner != "global":
                o_rel = 1
            elif goal.second_owner and r.owner == goal.second_owner:
                o_rel = 2
            elif r.owner == "GROUP":
                o_rel = 3
            elif r.owner in ("global", "vase", "basket", "barn", "restaurant", "theater", "school"):
                o_rel = 4
            elif r.owner == "one_individual":
                o_rel = 5
            else:
                o_rel = 0

            if r.per_unit is None:
                p_rel = 0
            elif goal.per_unit and r.per_unit == goal.per_unit:
                p_rel = 1
            elif r.per_unit == "group":
                p_rel = 2
            elif r.per_unit in TIME_UNITS:
                p_rel = 3
            elif r.per_unit in MULTIPLIER_CONTAINER_UNITS:
                p_rel = 4
            else:
                p_rel = 5

            if r.action == goal.action_lemma and goal.action_lemma != "have":
                a_rel = 1
            elif r.action == "have":
                a_rel = 2
            elif r.action in GROSS_EVENT_VERBS:
                a_rel = 3
            else:
                a_rel = 0

            reg_cat[slot] = (st_id, u_rel, o_rel, p_rel, a_rel)

            is_int = _safe_is_int(r.value)
            is_ufrac = 1.0 if (math.isfinite(r.value) and 0.0 < abs(r.value) < 1.0) else 0.0
            pop_cnt = float(bin(r.origin_mask).count("1"))
            q_ov = len(set(r.clause_words) & goal.question_tokens) / max(1.0, float(len(r.clause_words) or 1))
            rank_idx = np.searchsorted(sorted_vals, r.value)
            rank_v = float(min(rank_idx, n_act - 1)) / float(max(1, n_act - 1)) if n_act > 1 else 0.5
            is_max = 1.0 if (math.isfinite(r.value) and abs(r.value - max_v) < 1e-6) else 0.0

            reg_dense[slot] = (
                _safe_log_mag(r.value),
                1.0 if r.value >= 0.0 else -1.0,
                is_int,
                is_ufrac,
                min(1.0, r.depth / 6.0),
                min(1.0, max(0, r.clause_idx) / 10.0),
                min(1.0, pop_cnt / 6.0),
                1.0 if r.per_unit is not None else 0.0,
                1.0 if r.is_remaining_frac else 0.0,
                1.0 if r.is_implicit else 0.0,
                q_ov,
                rank_v,
                is_max,
                1.0,
            )

            hit_positions: List[int] = []
            mask_bits = r.origin_mask
            for bit_k in range(min(n_leaves, 29)):
                if (mask_bits >> bit_k) & 1:
                    hit_positions.append(ctx.leaf_positions[bit_k])
            if not hit_positions:
                hit_positions = [0]
            # 3-token local span weighting (0.25 left + 0.50 center + 0.25 right)
            inv_n = 1.0 / float(len(hit_positions))
            for hp in hit_positions:
                hp_l = max(0, hp - 1)
                hp_r = min(max(0, max_valid_tok), hp + 1)
                reg_anchor[slot, hp_l] += 0.25 * inv_n
                reg_anchor[slot, hp] += 0.50 * inv_n
                reg_anchor[slot, hp_r] += 0.25 * inv_n

        return slot_map, reg_cat, reg_dense, reg_anchor, reg_mask

    def forward_board_states_batch(
        self,
        reg_cat: np.ndarray,
        reg_dense: np.ndarray,
        reg_anchor: np.ndarray,
        reg_mask: np.ndarray,
        H_text: np.ndarray,
        prob_idx: np.ndarray,
        drop_rate: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, tuple]:
        p = self.params
        m3 = reg_mask[:, :, None]

        if H_text.shape[0] == 1:
            h_span = np.einsum("url,ld->urd", reg_anchor, H_text[0])
        else:
            h_span = reg_anchor @ H_text[prob_idx]

        x_cat = (
            p["E_stage"][reg_cat[:, :, 0]]
            + p["E_urel"][reg_cat[:, :, 1]]
            + p["E_orel"][reg_cat[:, :, 2]]
            + p["E_prel"][reg_cat[:, :, 3]]
            + p["E_arel"][reg_cat[:, :, 4]]
        )
        x0 = (x_cat + (reg_dense @ p["W_reg_dn"] + p["b_reg_dn"]) + (h_span @ p["W_span"] + p["b_span"])) * m3

        h1, blk_cache = _transformer_block_fwd(
            x0, reg_mask, p, "reg_b0", self.n_heads, alibi_bias=None, drop_rate=drop_rate, rng=rng
        )
        H_reg_raw, lnf_cache = _layernorm_fwd(h1, p["reg_ln_f_g"], p["reg_ln_f_b"])
        H_reg = (H_reg_raw * m3).astype(np.float32)

        cnt = np.maximum(1.0, reg_mask.sum(axis=1, keepdims=True)).astype(np.float32)
        h_board = (H_reg.sum(axis=1) / cnt).astype(np.float32)

        cache = (reg_cat, reg_dense, reg_anchor, reg_mask, prob_idx, h_span, blk_cache, lnf_cache, cnt)
        return H_reg, h_board, cache

    def backward_board_states_batch(
        self,
        dH_reg_in: np.ndarray,
        dh_board: np.ndarray,
        cache: tuple,
        dH_text_accum: np.ndarray,
        grads: Dict[str, np.ndarray],
    ) -> None:
        reg_cat, reg_dense, reg_anchor, reg_mask, prob_idx, h_span, blk_cache, lnf_cache, cnt = cache
        p = self.params
        D = self.d_model
        m3 = reg_mask[:, :, None]

        dH_reg = (dH_reg_in + (dh_board / cnt)[:, None, :]) * m3
        dh1, dg_f, db_f = _layernorm_bwd(dH_reg, lnf_cache)
        grads["reg_ln_f_g"] += dg_f
        grads["reg_ln_f_b"] += db_f
        dh1 = dh1 * m3

        dx0 = _transformer_block_bwd(dh1, blk_cache, p, grads) * m3

        grads["W_reg_dn"] += (reg_dense.reshape(-1, self.REG_DENSE).T @ dx0.reshape(-1, D)).astype(np.float32)
        grads["b_reg_dn"] += dx0.reshape(-1, D).sum(axis=0).astype(np.float32)

        grads["W_span"] += (h_span.reshape(-1, D).T @ dx0.reshape(-1, D)).astype(np.float32)
        grads["b_span"] += dx0.reshape(-1, D).sum(axis=0).astype(np.float32)

        dh_span = dx0 @ p["W_span"].T
        dH_p = reg_anchor.transpose(0, 2, 1) @ dh_span
        np.add.at(dH_text_accum, prob_idx, dH_p)

        flat_valid = reg_mask.ravel()
        dx_valid = dx0.reshape(-1, D)[flat_valid]
        cat_flat = reg_cat.reshape(-1, 5)[flat_valid]
        np.add.at(grads["E_stage"], cat_flat[:, 0], dx_valid)
        np.add.at(grads["E_urel"], cat_flat[:, 1], dx_valid)
        np.add.at(grads["E_orel"], cat_flat[:, 2], dx_valid)
        np.add.at(grads["E_prel"], cat_flat[:, 3], dx_valid)
        np.add.at(grads["E_arel"], cat_flat[:, 4], dx_valid)

    @classmethod
    def extract_transition_features(
        cls,
        regs: Sequence[QuantReg],
        move: TransitionMove,
        clauses: Sequence[ClauseInfo],
        goal: GoalSpec,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """構造メタID (4), 役割抽象化リレーショナルID (TRANS_REL_K), 連続特徴 (20) を抽出"""
        ra = regs[move.idx_a]
        rb = regs[move.idx_b]
        rn = move.new_reg
        op = move.op
        dist = rb.clause_idx - ra.clause_idx
        B = cls.REL_BUCKETS

        meta_ids = np.array(
            [
                _OP_TO_ID.get(op, len(_OP_LIST)),
                _RULE_TO_ID.get(move.rule_tag, len(_RULE_LIST)),
                _QTYPE_TO_ID.get(goal.query_type, len(_QTYPE_LIST)),
                _STAGE_TO_ID.get(rn.stage, len(_STAGE_LIST)),
            ],
            dtype=np.int32,
        )

        owner_rel = (
            "SAME_GOAL" if (ra.owner == rb.owner == goal.target_owner)
            else "SAME_OTHER" if (ra.owner == rb.owner)
            else "DIFF_HAS_GOAL" if (goal.target_owner in (ra.owner, rb.owner))
            else "DIFF_OTHER"
        )
        unit_rel = (
            "RATE_CANCEL" if move.rule_tag in ("RATE_CANCEL", "RATE_CONV_CANCEL")
            else "SAME_GOAL_U" if (ra.unit == rb.unit == goal.target_unit)
            else "COMP_PAIR_U" if (goal.second_unit and {ra.unit, rb.unit} == {goal.target_unit, goal.second_unit})
            else "SAME_U" if (ra.unit == rb.unit)
            else "DIFF_U"
        )
        dist_b = "0" if dist == 0 else ("1" if dist == 1 else ("2+" if dist > 1 else "REV"))

        rel_feats: List[int] = [
            _fnv_hash(f"T1|{op}|{move.rule_tag}|{goal.query_type}", B),
            _fnv_hash(f"T2|{op}|{ra.stage}|{rb.stage}", B),
            _fnv_hash(f"T3|{op}|{ra.stage}|{rb.stage}|{goal.query_type}", B),
            _fnv_hash(f"T4|{op}|{owner_rel}|{goal.query_type}", B),
            _fnv_hash(f"T5|{op}|{unit_rel}|{goal.query_type}", B),
            _fnv_hash(f"T6|{op}|{dist_b}|keep={move.keep_a_active}", B),
            _fnv_hash(f"T7|{op}|actMatchA={ra.action == goal.action_lemma}|actMatchB={rb.action == goal.action_lemma}", B),
            _fnv_hash(f"VA|{op}|{ra.action}|{rb.action}|{goal.action_lemma}", B),
        ]
        for w in ra.left_win[:3]:
            rel_feats.append(_fnv_hash(f"AL|{op}|{cls._delex_token(w, ra, rb, goal)}", B))
        for w in ra.right_win[:3]:
            rel_feats.append(_fnv_hash(f"AR|{op}|{cls._delex_token(w, ra, rb, goal)}", B))
        for w in rb.left_win[:3]:
            rel_feats.append(_fnv_hash(f"BL|{op}|{cls._delex_token(w, ra, rb, goal)}", B))
        for w in rb.right_win[:3]:
            rel_feats.append(_fnv_hash(f"BR|{op}|{cls._delex_token(w, ra, rb, goal)}", B))
        if len(rb.left_win) >= 2:
            rel_feats.append(_fnv_hash(f"BL2|{op}|{cls._delex_token(rb.left_win[-2], ra, rb, goal)}_{cls._delex_token(rb.left_win[-1], ra, rb, goal)}", B))
        if len(rb.right_win) >= 2:
            rel_feats.append(_fnv_hash(f"BR2|{op}|{cls._delex_token(rb.right_win[0], ra, rb, goal)}_{cls._delex_token(rb.right_win[1], ra, rb, goal)}", B))

        b_cues = list(clauses[rb.clause_idx].cue_tokens)[:4] if 0 <= rb.clause_idx < len(clauses) else []
        q_cues = [
            qw for qw in goal.question_tokens
            if qw in {"more", "less", "fewer", "left", "each", "every", "per", "total", "altogether", "first", "start", "original", "profit", "change", "average", "rest", "remaining", "pay", "buy", "sell", "empty", "needed", "full"}
        ][:4]
        for qb in list(goal.question_bigrams)[:4]:
            rel_feats.append(_fnv_hash(f"QB|{op}|{qb}", B))
        for bc in b_cues:
            rel_feats.append(_fnv_hash(f"BC|{op}|{bc}", B))
            for qc in q_cues[:2]:
                rel_feats.append(_fnv_hash(f"X|{op}|{bc}|{qc}", B))
        for qc in q_cues:
            rel_feats.append(_fnv_hash(f"QC|{op}|{qc}", B))

        rel_arr = np.zeros(cls.TRANS_REL_K, dtype=np.int32)
        Lr = min(len(rel_feats), cls.TRANS_REL_K)
        rel_arr[:Lr] = rel_feats[:Lr]

        q_overlap_a = len(set(ra.clause_words) & goal.question_tokens) / max(1.0, float(len(ra.clause_words) or 1))
        q_overlap_b = len(set(rb.clause_words) & goal.question_tokens) / max(1.0, float(len(rb.clause_words) or 1))
        is_int = _safe_is_int(rn.value)
        clean_div = 1.0 if (op == "DIV" and is_int > 0.5) else 0.0

        dense = np.array(
            [
                (move.prior_score - STEP_COST) / 4.0,
                1.0 if rn.unit == goal.target_unit else 0.0,
                1.0 if (rn.owner == goal.target_owner or goal.target_owner == "global") else 0.0,
                1.0 if move.rule_tag in ("RATE_CANCEL", "RATE_CONV_CANCEL", "QUOTITIVE_DIV", "IMPLICIT_QUOTITIVE_DIV", "PARTITIVE_RATE_DIV") else 0.0,
                q_overlap_a,
                q_overlap_b,
                is_int,
                clean_div,
                1.0 / (1.0 + abs(dist)),
                1.0 if dist >= 0 else 0.0,
                1.0 if ("CONSERVATION" in move.rule_tag or "SOLVE_SOME" in move.rule_tag or move.rule_tag == "PCT_INV_DIV") else 0.0,
                1.0 if "COMP_" in move.rule_tag else 0.0,
                1.0 if "TRANSFER" in move.rule_tag else 0.0,
                _safe_log_mag(rn.value),
                rn.depth / 6.0,
                1.0 if op == "ADD" else 0.0,
                1.0 if op == "SUB" else 0.0,
                1.0 if op == "MUL" else 0.0,
                1.0 if op == "DIV" else 0.0,
                1.0 if move.keep_a_active else 0.0,
            ],
            dtype=np.float32,
        )
        return meta_ids, rel_arr, dense

    @classmethod
    def compute_terminal_prior_and_features(
        cls,
        cand_idx: int,
        regs: Sequence[QuantReg],
        consumed_mask: int,
        goal: GoalSpec,
        clauses: Sequence[ClauseInfo] = (),
        history_ops: Tuple[str, ...] = (),
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        rc = regs[cand_idx]
        prior = 0.0
        z_hist = np.zeros(cls.MAX_HIST_LEN + 1, dtype=np.int32)
        z_rel = np.zeros(cls.TERM_REL_K, dtype=np.int32)
        z_dn = np.zeros(cls.TERM_DENSE, dtype=np.float32)

        if not math.isfinite(rc.value) or abs(rc.value) > 1e14:
            return -25.0, z_hist, z_rel, z_dn
        if rc.owner == "one_individual":
            return -25.0, z_hist, z_rel, z_dn
        if rc.stage in ("MULT_ANAPHOR", "FRAC", "FRAC_LOSE", "PCT_LESS", "PCT_MORE", "PCT_OF", "PAID_BILL", "TARGET_GOAL", "TOTAL_GIVEN", "RATIO_PART", "COMP_MORE", "COMP_LESS", "POST_MULT_MORE", "POST_MULT_LESS", "COST_RATE", "TIME_PARAM", "FINAL_PLUS_LOSE"):
            return -25.0, z_hist, z_rel, z_dn
        if rc.stage == "COST_OUT" and goal.action_lemma not in ("spend", "pay", "cost"):
            return -25.0, z_hist, z_rel, z_dn
        if rc.stage in ("LOSE", "TRANSFER_OUT") and goal.query_type in ("REMAIN", "CHANGE", "INITIAL") and goal.action_lemma not in ("give", "lose", "eat", "sell", "spend", "drop", "throw", "miss"):
            prior -= 6.5

        unit_ok = 0.0
        rc_hyp = UNIT_HYPERNYM.get(rc.unit, rc.unit)
        g_hyp = UNIT_HYPERNYM.get(goal.target_unit, goal.target_unit)
        if rc.unit == goal.target_unit or goal.target_unit == "item":
            prior += 3.6
            unit_ok = 1.0
        elif goal.target_unit in CATEGORY_TARGET_UNITS and rc_hyp == g_hyp:
            prior += 3.5
            unit_ok = 1.0
        elif goal.target_unit == "dozen" and rc.per_unit is None:
            prior += 3.0
            unit_ok = 1.0
        elif (rc.unit, goal.target_unit) in UNIT_CONVERSIONS:
            prior += 2.6
            unit_ok = 1.0
        elif goal.query_type == "RATIO" and rc.unit == "scalar":
            prior += 3.4
            unit_ok = 1.0
        else:
            prior -= 4.8

        if goal.query_type == "RATE_EACH":
            if rc.per_unit is not None:
                prior += 3.8
            else:
                prior -= 2.8
        elif goal.per_unit is not None and rc.per_unit == goal.per_unit:
            prior += 3.5
        else:
            if rc.per_unit is not None:
                prior -= 4.8

        owner_ok = 0.0
        if goal.target_owner == "global":
            owner_ok = 1.0
            prior += 1.2
        elif rc.owner == goal.target_owner:
            owner_ok = 1.0
            prior += 3.8
        elif goal.target_owner == "GROUP" and rc.owner == "GROUP":
            owner_ok = 1.0
            prior += 4.0
        elif rc.owner in ("global", "vase", "basket", "barn", "restaurant", "theater", "school"):
            prior += 1.2
        else:
            prior -= 5.2

        if goal.q_focus_modifiers:
            if rc.attr == "GROUP" or any(m in rc.attr for m in goal.q_focus_modifiers):
                prior += 2.6
            elif rc.attr:
                prior -= 3.5
        elif goal.target_attr and goal.query_type != "DIFF":
            if goal.target_attr in rc.attr or goal.target_unit == "dozen":
                prior += 3.0
            elif rc.attr == "GROUP":
                prior -= 2.5
            elif rc.attr:
                prior -= 3.0

        if goal.query_type == "INITIAL":
            prior += 4.5 if rc.stage == "INITIAL_SOLVED" else -4.0
        elif goal.query_type == "SOLVE_SOME":
            prior += 4.5 if rc.stage == "SOME_SOLVED" else -4.0
        elif goal.query_type == "DIFF":
            if rc.stage == "DIFF_SOLVED":
                prior += 3.6
            elif rc.depth == 0:
                prior -= 3.6
        elif goal.query_type in ("REMAIN", "CHANGE") and (rc.stage in ("FINAL", "REMAIN_POOL") or rc.depth > 0):
            prior += 2.0
        if goal.has_the_rest_cue and rc.stage == "REMAIN_POOL":
            prior += 2.6
        if rc.stage == "POOL_TOTAL" and rc.depth == 0 and goal.query_type in ("REMAIN", "COUNT", "DIFF"):
            prior -= 3.5

        gross_event_mode = (
            goal.query_type in ("COUNT", "TOTAL")
            and goal.action_lemma in GROSS_EVENT_VERBS
            and not goal.has_the_rest_cue
            and not (goal.question_tokens & {"left", "remain", "remaining", "now", "still"})
            and any(
                r.depth == 0
                and r.action == goal.action_lemma
                and (r.unit == goal.target_unit or goal.target_unit == "item" or UNIT_HYPERNYM.get(r.unit, r.unit) == g_hyp)
                for r in regs
            )
        )
        if gross_event_mode:
            if rc.action == goal.action_lemma:
                prior += 3.0
            elif rc.per_unit is None and rc.stage != "FINAL":
                prior -= 2.2

        direct_target_units = {
            r.unit for r in regs
            if r.depth == 0 and r.owner == goal.target_owner and r.stage not in ("COMP_MORE", "COMP_LESS")
        }

        unconsumed_relevant = 0
        unconsumed_operator = 0
        for idx, r in enumerate(regs):
            if idx == cand_idx or ((consumed_mask >> idx) & 1):
                continue
            if r.owner == "one_individual":
                continue
            if r.unit in ("color", "type", "kind", "flavor", "category", "variety", "stop") and goal.target_unit != r.unit:
                continue
            if r.depth == 0 and (rc.origin_mask & r.origin_mask):
                continue
            if goal.action_lemma == "cost" and r.stage == "TRANSFER_OUT":
                continue
            if gross_event_mode and r.depth == 0 and r.per_unit is None and r.stage in ("INIT", "GAIN", "LOSE", "TRANSFER_OUT", "TRANSFER_IN") and r.action != goal.action_lemma:
                continue
            if goal.query_type == "SOLVE_SOME" and goal.unknown_some_clause >= 0 and r.clause_idx > goal.unknown_some_clause + 1:
                continue
            if goal.q_focus_modifiers and r.attr and not any(m in r.attr for m in goal.q_focus_modifiers):
                continue
            if goal.query_type == "DIFF" and goal.target_attr and goal.second_attr and r.attr not in (goal.target_attr, goal.second_attr):
                continue
            if r.stage in ("COMP_MORE", "COMP_LESS") and r.owner != goal.target_owner and r.unit in direct_target_units:
                continue

            if r.stage in ("MULT_ANAPHOR", "FRAC", "FRAC_LOSE", "PCT_LESS", "PCT_MORE", "PCT_OF", "PAID_BILL", "TARGET_GOAL", "TOTAL_GIVEN", "RATIO_PART", "TIME_PARAM", "POST_MULT_MORE", "POST_MULT_LESS"):
                unconsumed_relevant += 1
                if r.is_remaining_frac or r.stage in ("PCT_LESS", "PCT_MORE", "PAID_BILL", "TOTAL_GIVEN", "RATIO_PART", "POST_MULT_MORE", "POST_MULT_LESS"):
                    unconsumed_operator += 1
                continue
            if r.stage == "Q_PARAM" and (r.unit in (goal.target_unit, goal.per_unit) or r.per_unit == goal.target_unit or r.is_implicit):
                unconsumed_relevant += 1
                continue

            r_hyp = UNIT_HYPERNYM.get(r.unit, r.unit)
            owner_rel_match = (
                goal.target_owner in ("global", "GROUP")
                or r.owner in (goal.target_owner, "global", "vase", "basket", "barn", "restaurant", "theater", "school")
                or (r.stage == "TRANSFER_OUT" and r.ref_owner == goal.target_owner)
                or (r.stage in ("COMP_MORE", "COMP_LESS") and goal.target_owner in (r.owner, r.ref_owner))
            )
            unit_rel_match = (
                r.unit in (goal.target_unit, goal.second_unit, rc.unit)
                or (goal.target_unit in CATEGORY_TARGET_UNITS and r_hyp == g_hyp)
                or (r.per_unit is not None and (r.unit == goal.target_unit or r.per_unit in {x.unit for x in regs}))
            )
            if owner_rel_match and unit_rel_match:
                if goal.target_attr and goal.query_type in ("COUNT", "REMAIN") and r.attr and goal.target_attr not in r.attr:
                    continue
                if goal.query_type == "RATE_EACH" and goal.question_tokens & {"first", "initially"} and r.stage == "LOSE":
                    continue
                unconsumed_relevant += 1

        prior -= 2.8 * unconsumed_relevant + 2.4 * unconsumed_operator

        meta_hist = np.zeros(cls.MAX_HIST_LEN + 1, dtype=np.int32)
        meta_hist[0] = _QTYPE_TO_ID.get(goal.query_type, len(_QTYPE_LIST))
        recent_ops = history_ops[-cls.MAX_HIST_LEN :]
        for h_i, op_name in enumerate(recent_ops):
            meta_hist[1 + h_i] = _OP_TO_ID.get(op_name, len(_OP_LIST))

        B = cls.REL_BUCKETS
        op_seq = ">".join(history_ops[-3:]) if history_ops else "NONE"
        last_op = history_ops[-1] if history_ops else "NONE"
        init_reg_cnt = sum(1 for r in regs if r.depth == 0 and not r.is_implicit)

        rel_list = [
            _fnv_hash(f"G1|{goal.query_type}|{rc.stage}|u={int(unit_ok)}|o={int(owner_ok)}", B),
            _fnv_hash(f"G2|{goal.query_type}|unc={min(unconsumed_relevant, 4)}", B),
            _fnv_hash(f"G3|d={min(rc.depth, 6)}|n={min(init_reg_cnt, 6)}|{goal.query_type}", B),
            _fnv_hash(f"G4|per={rc.per_unit == goal.per_unit}|{goal.query_type}", B),
            _fnv_hash(f"G5|actMatch={rc.action == goal.action_lemma}|gross={gross_event_mode}", B),
            _fnv_hash(f"G6|seq={op_seq}|{goal.query_type}", B),
            _fnv_hash(f"G7|last={last_op}|unc={min(unconsumed_relevant, 3)}|{goal.query_type}", B),
        ]
        for qb in list(goal.question_bigrams)[:6]:
            q_delex = qb.replace(goal.target_unit, "<UQ>").replace(goal.target_owner, "<PQ>")
            rel_list.append(_fnv_hash(f"GQ|{op_seq}|{q_delex}", B))
            rel_list.append(_fnv_hash(f"GL|{last_op}|{q_delex}", B))
        body_cues_all: List[str] = []
        for ci in clauses:
            for ct in ci.cue_tokens:
                if ct not in body_cues_all:
                    body_cues_all.append(ct)
        for bc in body_cues_all[:5]:
            rel_list.append(_fnv_hash(f"GB|{op_seq}|{bc}|{goal.query_type}", B))
            rel_list.append(_fnv_hash(f"GBL|{last_op}|{bc}", B))

        rel_arr = np.zeros(cls.TERM_REL_K, dtype=np.int32)
        Lr = min(len(rel_list), cls.TERM_REL_K)
        rel_arr[:Lr] = rel_list[:Lr]

        is_int = _safe_is_int(rc.value)
        dense = np.array(
            [
                prior / 8.0,
                unit_ok,
                owner_ok,
                1.0 if rc.per_unit == goal.per_unit else 0.0,
                float(min(unconsumed_relevant, 5)) / 5.0,
                rc.depth / 6.0,
                float(min(init_reg_cnt, 6)) / 6.0,
                1.0 if gross_event_mode and rc.action == goal.action_lemma else 0.0,
                is_int,
                1.0 if rc.value >= 0.0 else -1.0,
                _safe_log_mag(rc.value),
                1.0,
            ],
            dtype=np.float32,
        )
        return prior, meta_hist, rel_arr, dense

    def forward_transitions_head(
        self,
        H_reg: np.ndarray,
        h_board: np.ndarray,
        H_text: np.ndarray,
        tok_mask: np.ndarray,
        h_question: np.ndarray,
        state_uids: np.ndarray,
        slot_a: np.ndarray,
        slot_b: np.ndarray,
        meta_ids: np.ndarray,
        rel_ids: np.ndarray,
        dense_tr: np.ndarray,
        priors: np.ndarray,
        prob_idx: np.ndarray,
        drop_rate: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, tuple]:
        p = self.params
        h_a = H_reg[state_uids, slot_a]
        h_b = H_reg[state_uids, slot_b]
        h_prod = h_a * h_b
        h_diff = h_a - h_b

        e_mv = (
            p["E_op"][meta_ids[:, 0]]
            + p["E_rule"][meta_ids[:, 1]]
            + p["E_qtype_tr"][meta_ids[:, 2]]
            + p["E_newstage"][meta_ids[:, 3]]
        )
        feat_cat = np.concatenate([h_a, h_b, h_prod, h_diff, e_mv, dense_tr], axis=-1)
        q_pre = feat_cat @ p["W_tr_qproj"] + p["b_tr_qproj"]
        q_act = _gelu_fwd(q_pre)
        q_tr, lnq_cache = _layernorm_fwd(q_act, p["tr_lnq_g"], p["tr_lnq_b"])

        ctx_tr, ca_cache = _mha_cross_fwd(
            q_tr, H_text, tok_mask, prob_idx,
            p["tr_ca_Wq"], p["tr_ca_bq"], p["tr_ca_Wkv"], p["tr_ca_bkv"], p["tr_ca_Wo"], p["tr_ca_bo"],
            n_heads=self.n_heads,
        )

        hb_sel = h_board[state_uids]
        hq_sel = h_question[0:1].repeat(len(state_uids), axis=0) if h_question.shape[0] == 1 else h_question[prob_idx]
        u_tr = np.concatenate([q_tr, ctx_tr, hb_sel, hq_sel], axis=-1)

        m1_lin = u_tr @ p["W_tr_m1"] + p["b_tr_m1"]
        m1_ln, lnm1_cache = _layernorm_fwd(m1_lin, p["tr_lnm1_g"], p["tr_lnm1_b"])
        m1_act = _gelu_fwd(m1_ln)
        if drop_rate > 0.0 and rng is not None:
            dm = (rng.random(m1_act.shape, dtype=np.float32) >= drop_rate).astype(np.float32) / (1.0 - drop_rate)
            m1_d = m1_act * dm
            rel_dm = (rng.random(rel_ids.shape, dtype=np.float32) >= (drop_rate * 0.5)).astype(np.float32) / (1.0 - drop_rate * 0.5)
        else:
            dm = None
            m1_d = m1_act
            rel_dm = None

        m2_pre = m1_d @ p["W_tr_m2"] + p["b_tr_m2"]
        m2 = _gelu_fwd(m2_pre) + q_tr
        rel_w = p["W_rel_tr"][rel_ids]
        if rel_dm is not None:
            rel_w = rel_w * rel_dm
        raw_delta = (m2 @ p["w_tr_out"] + p["b_tr_out"][0]) + 0.25 * rel_w.sum(axis=1)
        th = np.tanh(raw_delta / self.DELTA_BOUND).astype(np.float32)
        delta = self.DELTA_BOUND * th
        scores = ((priors - STEP_COST) + delta).astype(np.float32)

        cache = (
            state_uids, slot_a, slot_b, meta_ids, rel_ids, rel_dm, prob_idx,
            h_a, h_b, feat_cat, q_pre, lnq_cache, q_tr, ca_cache,
            u_tr, m1_ln, lnm1_cache, m1_d, dm, m2_pre, m2, th,
        )
        return scores, cache

    def backward_transitions_head(
        self,
        d_scores: np.ndarray,
        cache: tuple,
        dH_reg: np.ndarray,
        dh_board: np.ndarray,
        dH_text: np.ndarray,
        dh_question: np.ndarray,
        grads: Dict[str, np.ndarray],
    ) -> None:
        (
            state_uids, slot_a, slot_b, meta_ids, rel_ids, rel_dm, prob_idx,
            h_a, h_b, feat_cat, q_pre, lnq_cache, q_tr, ca_cache,
            u_tr, m1_ln, lnm1_cache, m1_d, dm, m2_pre, m2, th,
        ) = cache
        p = self.params
        D = self.d_model
        g = (d_scores * (1.0 - th * th)).astype(np.float32)

        # Relational Highway gradient
        flat_rel = rel_ids.ravel()
        valid_r = flat_rel > 0
        if np.any(valid_r):
            rep_g = np.repeat((0.25 * g)[:, None], rel_ids.shape[1], axis=1)
            if rel_dm is not None:
                rep_g = rep_g * rel_dm
            np.add.at(grads["W_rel_tr"], flat_rel[valid_r], rep_g.ravel()[valid_r])
        grads["W_rel_tr"][0] = 0.0

        grads["w_tr_out"] += (m2.T @ g).astype(np.float32)
        grads["b_tr_out"] += np.array([g.sum()], dtype=np.float32)

        dm2 = g[:, None] * p["w_tr_out"][None, :]
        dq_tr_total = dm2.copy()
        dm2_pre = _gelu_bwd(m2_pre, dm2)

        grads["W_tr_m2"] += (m1_d.T @ dm2_pre).astype(np.float32)
        grads["b_tr_m2"] += dm2_pre.sum(axis=0).astype(np.float32)

        dm1_act = dm2_pre @ p["W_tr_m2"].T
        if dm is not None:
            dm1_act = dm1_act * dm
        dm1_ln = _gelu_bwd(m1_ln, dm1_act)
        dm1_lin, dg_m1, db_m1 = _layernorm_bwd(dm1_ln, lnm1_cache)
        grads["tr_lnm1_g"] += dg_m1
        grads["tr_lnm1_b"] += db_m1

        grads["W_tr_m1"] += (u_tr.T @ dm1_lin).astype(np.float32)
        grads["b_tr_m1"] += dm1_lin.sum(axis=0).astype(np.float32)

        du_tr = dm1_lin @ p["W_tr_m1"].T
        dq_from_u, dctx_tr, dhb_sel, dhq_sel = np.split(du_tr, 4, axis=-1)
        dq_tr_total += dq_from_u
        np.add.at(dh_board, state_uids, dhb_sel)
        np.add.at(dh_question, prob_idx, dhq_sel)

        dq_from_ca, dkv_mem, dWq, dbq, dWkv, dbkv, dWo, dbo = _mha_cross_bwd(dctx_tr, ca_cache)
        dq_tr_total += dq_from_ca
        dH_text += dkv_mem
        grads["tr_ca_Wq"] += dWq
        grads["tr_ca_bq"] += dbq
        grads["tr_ca_Wkv"] += dWkv
        grads["tr_ca_bkv"] += dbkv
        grads["tr_ca_Wo"] += dWo
        grads["tr_ca_bo"] += dbo

        dq_act, dg_q, db_q = _layernorm_bwd(dq_tr_total, lnq_cache)
        grads["tr_lnq_g"] += dg_q
        grads["tr_lnq_b"] += db_q
        dq_pre = _gelu_bwd(q_pre, dq_act)

        grads["W_tr_qproj"] += (feat_cat.T @ dq_pre).astype(np.float32)
        grads["b_tr_qproj"] += dq_pre.sum(axis=0).astype(np.float32)

        dfeat = dq_pre @ p["W_tr_qproj"].T
        dh_a0 = dfeat[:, 0:D]
        dh_b0 = dfeat[:, D : 2 * D]
        dh_prod = dfeat[:, 2 * D : 3 * D]
        dh_diff = dfeat[:, 3 * D : 4 * D]
        de_mv = dfeat[:, 4 * D : 5 * D]

        dh_a = dh_a0 + dh_prod * h_b + dh_diff
        dh_b = dh_b0 + dh_prod * h_a - dh_diff

        np.add.at(dH_reg, (state_uids, slot_a), dh_a)
        np.add.at(dH_reg, (state_uids, slot_b), dh_b)

        np.add.at(grads["E_op"], meta_ids[:, 0], de_mv)
        np.add.at(grads["E_rule"], meta_ids[:, 1], de_mv)
        np.add.at(grads["E_qtype_tr"], meta_ids[:, 2], de_mv)
        np.add.at(grads["E_newstage"], meta_ids[:, 3], de_mv)

    def forward_terminals_head(
        self,
        H_reg: np.ndarray,
        reg_mask: np.ndarray,
        H_text: np.ndarray,
        tok_mask: np.ndarray,
        h_question: np.ndarray,
        state_uids: np.ndarray,
        slot_cand: np.ndarray,
        meta_hist: np.ndarray,
        rel_ids: np.ndarray,
        dense_tm: np.ndarray,
        priors: np.ndarray,
        prob_idx: np.ndarray,
        drop_rate: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, tuple]:
        p = self.params
        D = self.d_model
        K_tm = len(state_uids)

        h_cand = H_reg[state_uids, slot_cand]
        board_sum = H_reg[state_uids].sum(axis=1)
        other_cnt = np.maximum(1.0, reg_mask[state_uids].sum(axis=1, keepdims=True) - 1.0).astype(np.float32)
        h_other = (board_sum - h_cand) / other_cnt

        qtype_ids = meta_hist[:, 0]
        hist_ops = meta_hist[:, 1:]
        h_mask = (hist_ops > 0).astype(np.float32)[:, :, None]
        hist_seq = (p["E_hist_op"][hist_ops] + p["E_hist_pos"][None, :, :]) * h_mask
        hist_flat = hist_seq.reshape(K_tm, self.MAX_HIST_LEN * D)
        hist_pre = hist_flat @ p["W_hist"] + p["b_hist"]
        h_hist = _gelu_fwd(hist_pre)

        e_qt = p["E_qtype_tm"][qtype_ids]
        feat_cat = np.concatenate([h_cand, h_other, h_cand - h_other, h_hist, e_qt, dense_tm], axis=-1)
        q_pre = feat_cat @ p["W_tm_qproj"] + p["b_tm_qproj"]
        q_act = _gelu_fwd(q_pre)
        q_tm, lnq_cache = _layernorm_fwd(q_act, p["tm_lnq_g"], p["tm_lnq_b"])

        ctx_tm, ca_cache = _mha_cross_fwd(
            q_tm, H_text, tok_mask, prob_idx,
            p["tm_ca_Wq"], p["tm_ca_bq"], p["tm_ca_Wkv"], p["tm_ca_bkv"], p["tm_ca_Wo"], p["tm_ca_bo"],
            n_heads=self.n_heads,
        )

        hq_sel = h_question[0:1].repeat(K_tm, axis=0) if h_question.shape[0] == 1 else h_question[prob_idx]
        u_tm = np.concatenate([q_tm, ctx_tm, hq_sel], axis=-1)

        m1_lin = u_tm @ p["W_tm_m1"] + p["b_tm_m1"]
        m1_ln, lnm1_cache = _layernorm_fwd(m1_lin, p["tm_lnm1_g"], p["tm_lnm1_b"])
        m1_act = _gelu_fwd(m1_ln)
        if drop_rate > 0.0 and rng is not None:
            dm = (rng.random(m1_act.shape, dtype=np.float32) >= drop_rate).astype(np.float32) / (1.0 - drop_rate)
            m1_d = m1_act * dm
            rel_dm = (rng.random(rel_ids.shape, dtype=np.float32) >= (drop_rate * 0.5)).astype(np.float32) / (1.0 - drop_rate * 0.5)
        else:
            dm = None
            m1_d = m1_act
            rel_dm = None

        m2_pre = m1_d @ p["W_tm_m2"] + p["b_tm_m2"]
        m2 = _gelu_fwd(m2_pre) + q_tm
        rel_w = p["W_rel_tm"][rel_ids]
        if rel_dm is not None:
            rel_w = rel_w * rel_dm
        raw_delta = (m2 @ p["w_tm_out"] + p["b_tm_out"][0]) + 0.25 * rel_w.sum(axis=1)
        th = np.tanh(raw_delta / self.DELTA_BOUND).astype(np.float32)
        delta = self.DELTA_BOUND * th
        scores = (priors + delta).astype(np.float32)

        cache = (
            state_uids, slot_cand, reg_mask, other_cnt, qtype_ids, hist_ops, h_mask,
            rel_ids, rel_dm, hist_flat, hist_pre, feat_cat, q_pre, lnq_cache, q_tm, ca_cache,
            u_tm, m1_ln, lnm1_cache, m1_d, dm, m2_pre, m2, th, prob_idx,
        )
        return scores, cache

    def backward_terminals_head(
        self,
        d_scores: np.ndarray,
        cache: tuple,
        dH_reg: np.ndarray,
        dH_text: np.ndarray,
        dh_question: np.ndarray,
        grads: Dict[str, np.ndarray],
    ) -> None:
        (
            state_uids, slot_cand, reg_mask, other_cnt, qtype_ids, hist_ops, h_mask,
            rel_ids, rel_dm, hist_flat, hist_pre, feat_cat, q_pre, lnq_cache, q_tm, ca_cache,
            u_tm, m1_ln, lnm1_cache, m1_d, dm, m2_pre, m2, th, prob_idx,
        ) = cache
        p = self.params
        D = self.d_model
        K_tm = len(state_uids)
        g = (d_scores * (1.0 - th * th)).astype(np.float32)

        flat_rel = rel_ids.ravel()
        valid_r = flat_rel > 0
        if np.any(valid_r):
            rep_g = np.repeat((0.25 * g)[:, None], rel_ids.shape[1], axis=1)
            if rel_dm is not None:
                rep_g = rep_g * rel_dm
            np.add.at(grads["W_rel_tm"], flat_rel[valid_r], rep_g.ravel()[valid_r])
        grads["W_rel_tm"][0] = 0.0

        grads["w_tm_out"] += (m2.T @ g).astype(np.float32)
        grads["b_tm_out"] += np.array([g.sum()], dtype=np.float32)

        dm2 = g[:, None] * p["w_tm_out"][None, :]
        dq_tm_total = dm2.copy()
        dm2_pre = _gelu_bwd(m2_pre, dm2)

        grads["W_tm_m2"] += (m1_d.T @ dm2_pre).astype(np.float32)
        grads["b_tm_m2"] += dm2_pre.sum(axis=0).astype(np.float32)

        dm1_act = dm2_pre @ p["W_tm_m2"].T
        if dm is not None:
            dm1_act = dm1_act * dm
        dm1_ln = _gelu_bwd(m1_ln, dm1_act)
        dm1_lin, dg_m1, db_m1 = _layernorm_bwd(dm1_ln, lnm1_cache)
        grads["tm_lnm1_g"] += dg_m1
        grads["tm_lnm1_b"] += db_m1

        grads["W_tm_m1"] += (u_tm.T @ dm1_lin).astype(np.float32)
        grads["b_tm_m1"] += dm1_lin.sum(axis=0).astype(np.float32)

        du_tm = dm1_lin @ p["W_tm_m1"].T
        dq_from_u, dctx_tm, dhq_sel = np.split(du_tm, 3, axis=-1)
        dq_tm_total += dq_from_u
        np.add.at(dh_question, prob_idx, dhq_sel)

        dq_from_ca, dkv_mem, dWq, dbq, dWkv, dbkv, dWo, dbo = _mha_cross_bwd(dctx_tm, ca_cache)
        dq_tm_total += dq_from_ca
        dH_text += dkv_mem
        grads["tm_ca_Wq"] += dWq
        grads["tm_ca_bq"] += dbq
        grads["tm_ca_Wkv"] += dWkv
        grads["tm_ca_bkv"] += dbkv
        grads["tm_ca_Wo"] += dWo
        grads["tm_ca_bo"] += dbo

        dq_act, dg_q, db_q = _layernorm_bwd(dq_tm_total, lnq_cache)
        grads["tm_lnq_g"] += dg_q
        grads["tm_lnq_b"] += db_q
        dq_pre = _gelu_bwd(q_pre, dq_act)

        grads["W_tm_qproj"] += (feat_cat.T @ dq_pre).astype(np.float32)
        grads["b_tm_qproj"] += dq_pre.sum(axis=0).astype(np.float32)

        dfeat = dq_pre @ p["W_tm_qproj"].T
        dh_cand0 = dfeat[:, 0:D]
        dh_other0 = dfeat[:, D : 2 * D]
        dh_cdiff = dfeat[:, 2 * D : 3 * D]
        dh_hist = dfeat[:, 3 * D : 4 * D]
        de_qt = dfeat[:, 4 * D : 5 * D]

        np.add.at(grads["E_qtype_tm"], qtype_ids, de_qt)

        dhist_pre = _gelu_bwd(hist_pre, dh_hist)
        grads["W_hist"] += (hist_flat.T @ dhist_pre).astype(np.float32)
        grads["b_hist"] += dhist_pre.sum(axis=0).astype(np.float32)

        dhist_seq = (dhist_pre @ p["W_hist"].T).reshape(K_tm, self.MAX_HIST_LEN, D) * h_mask
        grads["E_hist_pos"] += dhist_seq.sum(axis=0).astype(np.float32)
        flat_hops = hist_ops.ravel()
        valid_h = flat_hops > 0
        if np.any(valid_h):
            np.add.at(grads["E_hist_op"], flat_hops[valid_h], dhist_seq.reshape(-1, D)[valid_h])
        grads["E_hist_op"][0] = 0.0

        dh_cand = dh_cand0 + dh_cdiff
        dh_other = dh_other0 - dh_cdiff
        dh_board_each = (dh_other / other_cnt)[:, None, :] * reg_mask[state_uids][:, :, None]
        np.add.at(dH_reg, state_uids, dh_board_each)
        np.add.at(dH_reg, (state_uids, slot_cand), dh_cand - (dh_other / other_cnt))


FactoredNNUE = ESRTransformerEvaluator


# ==============================================================================
# 3c. 置換表付き Beam Stack Search (ガワ・探索パイプライン維持)
# ==============================================================================


class BeamStackSolver:
    def __init__(self, nnue: ESRTransformerEvaluator, beam_width: int = 26, max_depth: int = 6) -> None:
        self.nnue = nnue
        self.beam_width = beam_width
        self.max_depth = max_depth

    @staticmethod
    def _state_signature(regs: Sequence[QuantReg], consumed_mask: int) -> Tuple[Tuple[float, str, str, str, str], ...]:
        active = [
            (round(regs[i].value, 4) if math.isfinite(regs[i].value) else 0.0, regs[i].owner, regs[i].unit, regs[i].per_unit or "", regs[i].stage)
            for i in range(len(regs))
            if not ((consumed_mask >> i) & 1)
        ]
        active.sort()
        return tuple(active)

    def search_candidates(
        self,
        regs: List[QuantReg],
        clauses: List[ClauseInfo],
        goal: GoalSpec,
        record_feats: bool = False,
        oracle_milestones: Optional[Sequence[float]] = None,
        oracle_gold: Optional[float] = None,
    ) -> List[Tuple[float, float, SearchNode, int, np.ndarray, np.ndarray, float]]:
        ctx = self.nnue.encode_problem_context(regs, clauses, goal)
        assert ctx.H_text is not None and ctx.h_question is not None

        root = SearchNode(regs=tuple(regs), consumed_mask=0, cum_score=0.0, depth=0)
        layers: List[List[SearchNode]] = [[root]]
        transposition_table: Dict[Tuple[Tuple[float, str, str, str, str], ...], float] = {
            self._state_signature(root.regs, 0): 0.0
        }
        all_explored_nodes: List[SearchNode] = [root]

        eff_max_depth = max(self.max_depth, min(8, len(regs))) if len(regs) >= 6 else self.max_depth
        for depth in range(eff_max_depth):
            current_beam = layers[-1]
            if not current_beam:
                break

            n_beam = len(current_beam)
            b_slot_maps: List[Dict[int, int]] = []
            b_cat = np.zeros((n_beam, ESRTransformerEvaluator.MAX_ACTIVE_REGS, 5), dtype=np.int32)
            b_dn = np.zeros((n_beam, ESRTransformerEvaluator.MAX_ACTIVE_REGS, ESRTransformerEvaluator.REG_DENSE), dtype=np.float32)
            b_anc = np.zeros((n_beam, ESRTransformerEvaluator.MAX_ACTIVE_REGS, ESRTransformerEvaluator.MAX_SEQ_LEN), dtype=np.float32)
            b_msk = np.zeros((n_beam, ESRTransformerEvaluator.MAX_ACTIVE_REGS), dtype=bool)

            cand_entries: List[Tuple[int, SearchNode, TransitionMove]] = []
            for b_idx, node in enumerate(current_beam):
                s_map, rc, rd, ra_w, rm = ESRTransformerEvaluator.featurize_board_state(
                    node.regs, node.consumed_mask, goal, ctx
                )
                b_slot_maps.append(s_map)
                b_cat[b_idx] = rc
                b_dn[b_idx] = rd
                b_anc[b_idx] = ra_w
                b_msk[b_idx] = rm

                next_id = len(node.regs)
                moves = LegalMoveGenerator.generate(
                    node.regs, node.consumed_mask, clauses, goal, next_id, depth
                )
                for mv in moves:
                    if mv.idx_a in s_map and mv.idx_b in s_map:
                        cand_entries.append((b_idx, node, mv))

            if not cand_entries:
                break

            H_reg_beam, h_board_beam, _ = self.nnue.forward_board_states_batch(
                b_cat, b_dn, b_anc, b_msk, ctx.H_text, np.zeros(n_beam, dtype=np.int32), drop_rate=0.0
            )

            n_cands = len(cand_entries)
            st_uids = np.zeros(n_cands, dtype=np.int32)
            sl_a = np.zeros(n_cands, dtype=np.int32)
            sl_b = np.zeros(n_cands, dtype=np.int32)
            meta_arr = np.zeros((n_cands, 4), dtype=np.int32)
            rel_arr = np.zeros((n_cands, ESRTransformerEvaluator.TRANS_REL_K), dtype=np.int32)
            dn_arr = np.zeros((n_cands, ESRTransformerEvaluator.TRANS_DENSE), dtype=np.float32)
            pr_arr = np.zeros(n_cands, dtype=np.float32)

            for idx, (b_idx, node, mv) in enumerate(cand_entries):
                s_map = b_slot_maps[b_idx]
                st_uids[idx] = b_idx
                sl_a[idx] = s_map[mv.idx_a]
                sl_b[idx] = s_map[mv.idx_b]
                m_ids, r_ids, dn = ESRTransformerEvaluator.extract_transition_features(node.regs, mv, clauses, goal)
                meta_arr[idx] = m_ids
                rel_arr[idx] = r_ids
                dn_arr[idx] = dn
                pr_arr[idx] = mv.prior_score

            step_scores, _ = self.nnue.forward_transitions_head(
                H_reg_beam, h_board_beam, ctx.H_text, ctx.tok_mask[None, :], ctx.h_question,
                st_uids, sl_a, sl_b, meta_arr, rel_arr, dn_arr, pr_arr, np.zeros(n_cands, dtype=np.int32), drop_rate=0.0
            )

            next_candidates: List[Tuple[float, SearchNode]] = []
            for idx, (b_idx, node, mv) in enumerate(cand_entries):
                new_regs = node.regs + (mv.new_reg,)
                full_mask = node.consumed_mask | (1 << mv.idx_b)
                if not mv.keep_a_active:
                    full_mask |= (1 << mv.idx_a)

                new_cum = node.cum_score + float(step_scores[idx])
                sig = self._state_signature(new_regs, full_mask)
                if sig in transposition_table and transposition_table[sig] >= new_cum - 1e-5:
                    continue
                transposition_table[sig] = new_cum
                ra_v = node.regs[mv.idx_a].value
                rb_v = node.regs[mv.idx_b].value

                s_feats = (
                    node.step_feats
                    + ((node.regs, node.consumed_mask, int(sl_a[idx]), int(sl_b[idx]), meta_arr[idx].copy(), rel_arr[idx].copy(), dn_arr[idx].copy(), float(pr_arr[idx])),)
                    if record_feats
                    else ()
                )
                child = SearchNode(
                    regs=new_regs,
                    consumed_mask=full_mask,
                    cum_score=new_cum,
                    depth=depth + 1,
                    history=node.history + ((mv.op, ra_v, rb_v, mv.new_reg.value),),
                    step_feats=s_feats,
                )
                sort_key = new_cum
                if oracle_milestones is not None and any(abs(mv.new_reg.value - ms) <= 1e-2 for ms in oracle_milestones):
                    sort_key += 3.0
                if oracle_gold is not None and abs(mv.new_reg.value - oracle_gold) <= max(1e-3, 1e-4 * abs(oracle_gold)):
                    sort_key += 3.0
                next_candidates.append((sort_key, child))

            if not next_candidates:
                break

            next_candidates.sort(key=lambda item: item[0], reverse=True)
            pruned_beam = [nd for _, nd in next_candidates[: self.beam_width]]
            layers.append(pruned_beam)
            all_explored_nodes.extend(pruned_beam)

        term_meta: List[Tuple[int, SearchNode, int]] = []
        term_hist_list: List[np.ndarray] = []
        term_rel_list: List[np.ndarray] = []
        term_dn_list: List[np.ndarray] = []
        term_pr_list: List[float] = []

        unique_term_nodes: List[SearchNode] = []
        node_id_map: Dict[int, int] = {}
        node_slot_maps: List[Dict[int, int]] = []

        for node in all_explored_nodes:
            for r_idx in range(len(node.regs)):
                if (node.consumed_mask >> r_idx) & 1:
                    continue
                if node.depth > 0 and r_idx != len(node.regs) - 1:
                    continue
                h_ops = tuple(h[0] for h in node.history)
                t_prior, t_hist, t_rel, t_dn = ESRTransformerEvaluator.compute_terminal_prior_and_features(
                    r_idx, node.regs, node.consumed_mask, goal, clauses, h_ops
                )
                if t_prior <= -20.0:
                    continue
                obj_key = id(node)
                if obj_key not in node_id_map:
                    s_map, _, _, _, _ = ESRTransformerEvaluator.featurize_board_state(
                        node.regs, node.consumed_mask, goal, ctx
                    )
                    if r_idx not in s_map:
                        continue
                    node_id_map[obj_key] = len(unique_term_nodes)
                    unique_term_nodes.append(node)
                    node_slot_maps.append(s_map)
                u_idx = node_id_map[obj_key]
                if r_idx not in node_slot_maps[u_idx]:
                    continue
                term_meta.append((u_idx, node, r_idx))
                term_hist_list.append(t_hist)
                term_rel_list.append(t_rel)
                term_dn_list.append(t_dn)
                term_pr_list.append(t_prior)

        if not term_meta:
            return []

        n_tnodes = len(unique_term_nodes)
        t_cat = np.zeros((n_tnodes, ESRTransformerEvaluator.MAX_ACTIVE_REGS, 5), dtype=np.int32)
        t_bdn = np.zeros((n_tnodes, ESRTransformerEvaluator.MAX_ACTIVE_REGS, ESRTransformerEvaluator.REG_DENSE), dtype=np.float32)
        t_anc = np.zeros((n_tnodes, ESRTransformerEvaluator.MAX_ACTIVE_REGS, ESRTransformerEvaluator.MAX_SEQ_LEN), dtype=np.float32)
        t_msk = np.zeros((n_tnodes, ESRTransformerEvaluator.MAX_ACTIVE_REGS), dtype=bool)

        for u_idx, nd in enumerate(unique_term_nodes):
            _, rc, rd, ra_w, rm = ESRTransformerEvaluator.featurize_board_state(nd.regs, nd.consumed_mask, goal, ctx)
            t_cat[u_idx] = rc
            t_bdn[u_idx] = rd
            t_anc[u_idx] = ra_w
            t_msk[u_idx] = rm

        H_reg_tm, _, _ = self.nnue.forward_board_states_batch(
            t_cat, t_bdn, t_anc, t_msk, ctx.H_text, np.zeros(n_tnodes, dtype=np.int32), drop_rate=0.0
        )

        n_tm = len(term_meta)
        tm_uids = np.array([u_idx for u_idx, _, _ in term_meta], dtype=np.int32)
        tm_slots = np.array([node_slot_maps[u_idx][r_idx] for u_idx, _, r_idx in term_meta], dtype=np.int32)
        t_hist_arr = np.stack(term_hist_list, axis=0)
        t_rel_arr = np.stack(term_rel_list, axis=0)
        t_dn_arr = np.stack(term_dn_list, axis=0)
        t_pr_arr = np.array(term_pr_list, dtype=np.float32)

        term_scores, _ = self.nnue.forward_terminals_head(
            H_reg_tm, t_msk, ctx.H_text, ctx.tok_mask[None, :], ctx.h_question,
            tm_uids, tm_slots, t_hist_arr, t_rel_arr, t_dn_arr, t_pr_arr,
            np.zeros(n_tm, dtype=np.int32), drop_rate=0.0
        )

        results: List[Tuple[float, float, SearchNode, int, Tuple[np.ndarray, np.ndarray], np.ndarray, float]] = []
        for idx, (u_idx, node, r_idx) in enumerate(term_meta):
            rc = node.regs[r_idx]
            val = rc.value
            if goal.target_unit == "dozen" and rc.unit != "dozen":
                if goal.target_attr and (rc.unit, goal.target_attr) in UNIT_CONVERSIONS:
                    val = val * UNIT_CONVERSIONS[(rc.unit, goal.target_attr)]
                val = val / 12.0
            elif rc.unit != goal.target_unit and (rc.unit, goal.target_unit) in UNIT_CONVERSIONS:
                val = val * UNIT_CONVERSIONS[(rc.unit, goal.target_unit)]
            if not math.isfinite(val):
                continue
            tot = node.cum_score + float(term_scores[idx])
            results.append((val, tot, node, r_idx, (t_hist_arr[idx], t_rel_arr[idx]), t_dn_arr[idx], float(t_pr_arr[idx])))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def solve_problem(
        self,
        regs: List[QuantReg],
        clauses: List[ClauseInfo],
        goal: GoalSpec,
    ) -> Optional[Tuple[float, SearchNode, int]]:
        if not regs:
            return None
        if len(regs) == 1:
            v = regs[0].value
            if goal.target_unit == "dozen" and regs[0].unit != "dozen":
                v /= 12.0
            elif regs[0].unit != goal.target_unit and (regs[0].unit, goal.target_unit) in UNIT_CONVERSIONS:
                v *= UNIT_CONVERSIONS[(regs[0].unit, goal.target_unit)]
            return v, SearchNode(tuple(regs), 0, 0.0, 0), 0

        cands = self.search_candidates(regs, clauses, goal, record_feats=False)
        if not cands:
            return regs[-1].value, SearchNode(tuple(regs), 0, 0.0, 0), len(regs) - 1
        best_val, _, best_node, best_r_idx, _, _, _ = cands[0]
        return best_val, best_node, best_r_idx

# ==============================================================================
# 4. データ取得・トレース解析・正則化 (Dropout + WeightDecay + CosineLR) 付き AdamW 学習器
# ==============================================================================


def download_files() -> None:
    targets = [
        (TRAIN_FILE, DATASET_URLS["mawps_asdiv.csv"]),
        (TEST_FILE, DATASET_URLS["SVAMP.json"]),
        (GSM8K_TRAIN_FILE, DATASET_URLS["gsm8k_train.jsonl"]),
        (GSM8K_TEST_FILE, DATASET_URLS["gsm8k_test.jsonl"]),
    ]
    for local_path, url in targets:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            continue
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        try:
            print(f"[download_files] Downloading {os.path.basename(local_path)} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(local_path, "wb") as out_f:
                out_f.write(resp.read())
        except Exception as exc:
            print(f"[download_files] スキップ ({os.path.basename(local_path)}): {exc}")


class TraceCompiler:
    _CALC_TAG_RE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
    _FINAL_HASH_RE = re.compile(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)")

    @classmethod
    def parse_equation_milestones(cls, eq_str: str, num_list: List[float]) -> Tuple[List[float], Optional[float]]:
        tokens = eq_str.replace("(", " ( ").replace(")", " ) ").split()
        if not tokens:
            return [], None

        def _tok_val(t: str) -> Optional[float]:
            low = t.lower()
            if low.startswith("number") and low[6:].isdigit():
                idx = int(low[6:])
                return num_list[idx] if idx < len(num_list) else None
            try:
                return float(t.replace(",", ""))
            except ValueError:
                return None

        milestones: List[float] = []

        if tokens[0] in {"+", "-", "*", "/"} and "(" not in tokens:
            pos = 0

            def _eval_prefix() -> Optional[float]:
                nonlocal pos
                if pos >= len(tokens):
                    return None
                tk = tokens[pos]
                pos += 1
                if tk in {"+", "-", "*", "/"}:
                    va = _eval_prefix()
                    vb = _eval_prefix()
                    if va is None or vb is None:
                        return None
                    if tk == "+":
                        res = va + vb
                    elif tk == "-":
                        res = va - vb
                    elif tk == "*":
                        res = va * vb
                    else:
                        if abs(vb) < 1e-9:
                            return None
                        res = va / vb
                    milestones.append(res)
                    return res
                return _tok_val(tk)

            final_v = _eval_prefix()
            return milestones, final_v

        expr = eq_str
        for idx, nv in enumerate(num_list):
            expr = re.sub(rf"\bnumber{idx}\b", str(nv), expr, flags=re.I)
        return cls._eval_infix_ast(expr)

    @classmethod
    def _eval_infix_ast(cls, expr: str) -> Tuple[List[float], Optional[float]]:
        milestones: List[float] = []
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError:
            return [], None

        def _walk(node: ast.AST) -> Optional[float]:
            if isinstance(node, ast.Expression):
                return _walk(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                v = _walk(node.operand)
                return -v if v is not None else None
            if isinstance(node, ast.BinOp):
                va = _walk(node.left)
                vb = _walk(node.right)
                if va is None or vb is None:
                    return None
                if isinstance(node.op, ast.Add):
                    r = va + vb
                elif isinstance(node.op, ast.Sub):
                    r = va - vb
                elif isinstance(node.op, ast.Mult):
                    r = va * vb
                elif isinstance(node.op, ast.Div):
                    if abs(vb) < 1e-9:
                        return None
                    r = va / vb
                else:
                    return None
                milestones.append(r)
                return r
            return None

        fv = _walk(tree)
        return milestones, fv

    @classmethod
    def parse_gsm8k_answer(cls, ans_text: str) -> Tuple[List[float], Optional[float]]:
        milestones: List[float] = []
        for m in cls._CALC_TAG_RE.finditer(ans_text):
            sub_ms, _ = cls._eval_infix_ast(m.group(1))
            if sub_ms:
                milestones.extend(sub_ms)
            else:
                try:
                    milestones.append(float(m.group(2).replace(",", "").strip()))
                except ValueError:
                    pass
        m_fin = cls._FINAL_HASH_RE.search(ans_text)
        if not m_fin:
            return milestones, (milestones[-1] if milestones else None)
        final_v = float(m_fin.group(1).replace(",", ""))
        return milestones, final_v



@dataclass
class PathCandidateRecord:
    is_gold: bool
    milestone_hits: int
    trans_local_ids: List[int]
    term_local_id: int


@dataclass
class ProblemTrainingGraph:
    tok_ids: np.ndarray
    seg_ids: np.ndarray
    tok_mask: np.ndarray
    q_mask: np.ndarray
    state_cat: np.ndarray
    state_dn: np.ndarray
    state_anc: np.ndarray
    state_msk: np.ndarray
    tr_state_uid: np.ndarray
    tr_slot_a: np.ndarray
    tr_slot_b: np.ndarray
    tr_meta: np.ndarray
    tr_rel: np.ndarray
    tr_dn: np.ndarray
    tr_pr: np.ndarray
    tm_state_uid: np.ndarray
    tm_slot_cand: np.ndarray
    tm_hist: np.ndarray
    tm_rel: np.ndarray
    tm_dn: np.ndarray
    tm_pr: np.ndarray
    paths: List[PathCandidateRecord]


class _TransformerAdamW:
    """疎な関係ハイウェイ更新 (Sparse AdamW) と密な Transformer 更新 (Clipped AdamW) の統合オプティマイザ"""

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 2.2e-3,
        weight_decay: float = 8.0e-3,
        max_grad_norm: float = 1.2,
    ) -> None:
        self.base_lr = lr
        self.lr = lr
        self.wd = weight_decay
        self.max_grad_norm = max_grad_norm
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        active_rel: Dict[str, np.ndarray],
    ) -> None:
        self.t += 1
        dense_sq = sum(float(np.sum(g * g)) for k, g in grads.items() if k not in active_rel)
        grad_norm = math.sqrt(max(dense_sq, 1e-16))
        clip_coef = min(1.0, self.max_grad_norm / grad_norm)

        b1, b2, eps = 0.9, 0.999, 1e-8
        lr_t = self.lr * math.sqrt(1.0 - b2 ** self.t) / (1.0 - b1 ** self.t)
        rel_lr_t = lr_t * 0.6

        for k, p in params.items():
            if k in active_rel:
                idx = active_rel[k]
                if len(idx) == 0:
                    continue
                g_sp = np.clip(grads[k][idx], -1.0, 1.0)
                p[idx] *= (1.0 - self.lr * self.wd * 0.5)
                self.m[k][idx] = b1 * self.m[k][idx] + (1.0 - b1) * g_sp
                self.v[k][idx] = b2 * self.v[k][idx] + (1.0 - b2) * (g_sp * g_sp)
                p[idx] -= rel_lr_t * self.m[k][idx] / (np.sqrt(self.v[k][idx]) + eps)
                p[0] = 0.0
            else:
                g = grads[k] * clip_coef
                if p.ndim >= 2 and not k.endswith(("_g", "_b")):
                    p *= (1.0 - self.lr * self.wd)
                self.m[k] = b1 * self.m[k] + (1.0 - b1) * g
                self.v[k] = b2 * self.v[k] + (1.0 - b2) * (g * g)
                p -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + eps)

        params["E_tok"][0] = 0.0
        params["E_hist_op"][0] = 0.0


class TransformerTrainer:
    """
    ALiBi-Transformer + Register-Set Self-Attention + Relational Highway を
    EMA (指数移動平均重み) + Hard-Negative 再マイニング付きで一気通貫学習する学習器。
    """

    def __init__(
        self,
        nnue: ESRTransformerEvaluator,
        lr: float = 2.2e-3,
        weight_decay: float = 8.0e-3,
        dropout: float = 0.12,
    ) -> None:
        self.nnue = nnue
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

    @staticmethod
    def _is_close_ans(pred: float, gold: float) -> bool:
        if not math.isfinite(pred) or not math.isfinite(gold):
            return False
        return abs(pred - gold) <= max(1e-3, 1e-4 * abs(gold))

    def build_problem_candidate_graphs(
        self, problems: Sequence[Tuple[str, List[float], float]], max_depth: int = 7
    ) -> Tuple[List[ProblemTrainingGraph], int]:
        graphs: List[ProblemTrainingGraph] = []
        solver = BeamStackSolver(self.nnue, beam_width=26, max_depth=max_depth)
        reachable_count = 0

        for q_text, milestones, gold_ans in problems:
            regs, clauses, goal = ClauseRoleParser.parse(q_text)
            if not regs:
                continue
            ms_round = [round(v, 3) for v in milestones if math.isfinite(v)]
            eff_depth = min(max_depth, max(2, len(ms_round) + 1)) if ms_round else 4
            solver.max_depth = eff_depth
            cands = solver.search_candidates(
                regs,
                clauses,
                goal,
                record_feats=True,
                oracle_milestones=ms_round,
                oracle_gold=gold_ans,
            )
            if not cands:
                continue

            gold_raw = []
            neg_raw = []
            for item in cands:
                val, _, node, r_idx, (t_hist, t_rel), t_dn, t_pr = item
                is_g = self._is_close_ans(val, gold_ans)
                ms_hits = sum(1 for _, _, _, step_v in node.history if any(abs(step_v - ms) <= 1e-2 for ms in ms_round))
                if is_g:
                    gold_raw.append((ms_hits, node, r_idx, t_hist, t_rel, t_dn, t_pr))
                else:
                    neg_raw.append((ms_hits, node, r_idx, t_hist, t_rel, t_dn, t_pr))

            if not gold_raw:
                continue
            reachable_count += 1
            if not neg_raw:
                continue

            gold_raw.sort(key=lambda x: x[0], reverse=True)
            selected = [(True, x) for x in gold_raw[:4]] + [(False, x) for x in neg_raw[:16]]

            ctx = self.nnue.tokenize_problem(regs, clauses, goal)

            state_map: Dict[Tuple[int, int, float], int] = {}
            state_slot_maps: List[Dict[int, int]] = []
            s_cats, s_dns, s_ancs, s_msks = [], [], [], []

            def _intern_state(s_regs: Tuple[QuantReg, ...], c_mask: int) -> int:
                last_v = round(s_regs[-1].value, 4) if (s_regs and math.isfinite(s_regs[-1].value)) else 0.0
                full_key = (len(s_regs), c_mask, last_v)
                if full_key not in state_map:
                    sm, rc, rd, ra_w, rm = ESRTransformerEvaluator.featurize_board_state(s_regs, c_mask, goal, ctx)
                    state_map[full_key] = len(s_cats)
                    state_slot_maps.append(sm)
                    s_cats.append(rc)
                    s_dns.append(rd)
                    s_ancs.append(ra_w)
                    s_msks.append(rm)
                return state_map[full_key]

            tr_map: Dict[Tuple[int, int, int, int, int], int] = {}
            tr_suids, tr_sa, tr_sb, tr_metas, tr_rels, tr_dns, tr_prs = [], [], [], [], [], [], []
            tm_suids, tm_sc, tm_hists, tm_rels, tm_dns, tm_prs = [], [], [], [], [], []
            path_recs: List[PathCandidateRecord] = []

            for is_g, (ms_hits, node, r_idx, t_hist, t_rel, t_dn, t_pr) in selected:
                tr_ids: List[int] = []
                for sf in node.step_feats:
                    p_regs, p_mask, sl_a, sl_b, m_ids, r_ids, dn_v, pr_v = sf
                    suid = _intern_state(p_regs, p_mask)
                    t_key = (suid, sl_a, sl_b, int(m_ids[0]), int(m_ids[1]))
                    if t_key not in tr_map:
                        tr_map[t_key] = len(tr_suids)
                        tr_suids.append(suid)
                        tr_sa.append(sl_a)
                        tr_sb.append(sl_b)
                        tr_metas.append(m_ids)
                        tr_rels.append(r_ids)
                        tr_dns.append(dn_v)
                        tr_prs.append(pr_v)
                    tr_ids.append(tr_map[t_key])

                tm_uid = _intern_state(node.regs, node.consumed_mask)
                if r_idx not in state_slot_maps[tm_uid]:
                    continue
                tm_local = len(tm_suids)
                tm_suids.append(tm_uid)
                tm_sc.append(state_slot_maps[tm_uid][r_idx])
                tm_hists.append(t_hist)
                tm_rels.append(t_rel)
                tm_dns.append(t_dn)
                tm_prs.append(t_pr)
                path_recs.append(PathCandidateRecord(is_g, ms_hits, tr_ids, tm_local))

            if any(r.is_gold for r in path_recs) and any(not r.is_gold for r in path_recs):
                graphs.append(
                    ProblemTrainingGraph(
                        tok_ids=ctx.tok_ids,
                        seg_ids=ctx.seg_ids,
                        tok_mask=ctx.tok_mask,
                        q_mask=ctx.q_mask,
                        state_cat=np.stack(s_cats, axis=0),
                        state_dn=np.stack(s_dns, axis=0),
                        state_anc=np.stack(s_ancs, axis=0),
                        state_msk=np.stack(s_msks, axis=0),
                        tr_state_uid=np.array(tr_suids, dtype=np.int32),
                        tr_slot_a=np.array(tr_sa, dtype=np.int32),
                        tr_slot_b=np.array(tr_sb, dtype=np.int32),
                        tr_meta=np.stack(tr_metas, axis=0) if tr_metas else np.zeros((0, 4), dtype=np.int32),
                        tr_rel=np.stack(tr_rels, axis=0) if tr_rels else np.zeros((0, ESRTransformerEvaluator.TRANS_REL_K), dtype=np.int32),
                        tr_dn=np.stack(tr_dns, axis=0) if tr_dns else np.zeros((0, ESRTransformerEvaluator.TRANS_DENSE), dtype=np.float32),
                        tr_pr=np.array(tr_prs, dtype=np.float32),
                        tm_state_uid=np.array(tm_suids, dtype=np.int32),
                        tm_slot_cand=np.array(tm_sc, dtype=np.int32),
                        tm_hist=np.stack(tm_hists, axis=0),
                        tm_rel=np.stack(tm_rels, axis=0),
                        tm_dn=np.stack(tm_dns, axis=0),
                        tm_pr=np.array(tm_prs, dtype=np.float32),
                        paths=path_recs,
                    )
                )

        return graphs, reachable_count

    def compute_loss_and_grads(
        self,
        batch_graphs: Sequence[ProblemTrainingGraph],
        drop_rate: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        nn = self.nnue
        grads: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in nn.params.items()}
        B = len(batch_graphs)
        if B == 0:
            return 0.0, grads, {}

        tok_ids = np.stack([g.tok_ids for g in batch_graphs], axis=0)
        seg_ids = np.stack([g.seg_ids for g in batch_graphs], axis=0)
        tok_mask = np.stack([g.tok_mask for g in batch_graphs], axis=0)
        q_mask = np.stack([g.q_mask for g in batch_graphs], axis=0)

        H_text, h_question, txt_cache = nn.forward_text_batch(
            tok_ids, seg_ids, tok_mask, q_mask, drop_rate=drop_rate, rng=rng
        )

        state_offsets: List[int] = []
        s_cursor = 0
        state_prob_list: List[np.ndarray] = []
        for p_idx, g in enumerate(batch_graphs):
            state_offsets.append(s_cursor)
            u_cnt = len(g.state_cat)
            s_cursor += u_cnt
            state_prob_list.append(np.full(u_cnt, p_idx, dtype=np.int32))

        all_s_cat = np.concatenate([g.state_cat for g in batch_graphs], axis=0)
        all_s_dn = np.concatenate([g.state_dn for g in batch_graphs], axis=0)
        all_s_anc = np.concatenate([g.state_anc for g in batch_graphs], axis=0)
        all_s_msk = np.concatenate([g.state_msk for g in batch_graphs], axis=0)
        all_s_prob = np.concatenate(state_prob_list, axis=0)

        H_reg, h_board, reg_cache = nn.forward_board_states_batch(
            all_s_cat, all_s_dn, all_s_anc, all_s_msk, H_text, all_s_prob, drop_rate=drop_rate, rng=rng
        )

        tr_offsets: List[int] = []
        tm_offsets: List[int] = []
        tr_cursor = 0
        tm_cursor = 0

        tr_suid_l, tr_sa_l, tr_sb_l, tr_meta_l, tr_rel_l, tr_dn_l, tr_pr_l, tr_prob_l = [], [], [], [], [], [], [], []
        tm_suid_l, tm_sc_l, tm_hist_l, tm_rel_l, tm_dn_l, tm_pr_l, tm_prob_l = [], [], [], [], [], [], []

        for p_idx, g in enumerate(batch_graphs):
            s_off = state_offsets[p_idx]
            tr_offsets.append(tr_cursor)
            m_cnt = len(g.tr_pr)
            if m_cnt > 0:
                tr_suid_l.append(g.tr_state_uid + s_off)
                tr_sa_l.append(g.tr_slot_a)
                tr_sb_l.append(g.tr_slot_b)
                tr_meta_l.append(g.tr_meta)
                tr_rel_l.append(g.tr_rel)
                tr_dn_l.append(g.tr_dn)
                tr_pr_l.append(g.tr_pr)
                tr_prob_l.append(np.full(m_cnt, p_idx, dtype=np.int32))
                tr_cursor += m_cnt

            tm_offsets.append(tm_cursor)
            k_cnt = len(g.tm_pr)
            tm_suid_l.append(g.tm_state_uid + s_off)
            tm_sc_l.append(g.tm_slot_cand)
            tm_hist_l.append(g.tm_hist)
            tm_rel_l.append(g.tm_rel)
            tm_dn_l.append(g.tm_dn)
            tm_pr_l.append(g.tm_pr)
            tm_prob_l.append(np.full(k_cnt, p_idx, dtype=np.int32))
            tm_cursor += k_cnt

        has_trans = tr_cursor > 0
        if has_trans:
            cat_tr_rel = np.concatenate(tr_rel_l, axis=0)
            s_trans, tr_cache = nn.forward_transitions_head(
                H_reg, h_board, H_text, tok_mask, h_question,
                np.concatenate(tr_suid_l), np.concatenate(tr_sa_l), np.concatenate(tr_sb_l),
                np.concatenate(tr_meta_l, axis=0), cat_tr_rel, np.concatenate(tr_dn_l, axis=0),
                np.concatenate(tr_pr_l), np.concatenate(tr_prob_l),
                drop_rate=drop_rate, rng=rng,
            )
            act_tr_rel = np.unique(cat_tr_rel[cat_tr_rel > 0])
        else:
            s_trans = np.zeros(0, dtype=np.float32)
            tr_cache = ()
            act_tr_rel = np.array([], dtype=np.int32)

        cat_tm_rel = np.concatenate(tm_rel_l, axis=0)
        s_term, tm_cache = nn.forward_terminals_head(
            H_reg, all_s_msk, H_text, tok_mask, h_question,
            np.concatenate(tm_suid_l), np.concatenate(tm_sc_l),
            np.concatenate(tm_hist_l, axis=0), cat_tm_rel, np.concatenate(tm_dn_l, axis=0),
            np.concatenate(tm_pr_l), np.concatenate(tm_prob_l),
            drop_rate=drop_rate, rng=rng,
        )
        act_tm_rel = np.unique(cat_tm_rel[cat_tm_rel > 0])

        d_trans = np.zeros(tr_cursor, dtype=np.float32)
        d_term = np.zeros(tm_cursor, dtype=np.float32)
        total_loss = 0.0
        m_probs = float(B)
        margin = 1.8

        for p_idx, g in enumerate(batch_graphs):
            tr_off = tr_offsets[p_idx]
            tm_off = tm_offsets[p_idx]
            n_paths = len(g.paths)
            path_logits = np.empty(n_paths, dtype=np.float64)
            g_mask = np.zeros(n_paths, dtype=bool)
            ms_b = np.zeros(n_paths, dtype=np.float64)

            for c_i, rec in enumerate(g.paths):
                step_sum = sum(float(s_trans[tr_off + tid]) for tid in rec.trans_local_ids)
                path_logits[c_i] = step_sum + float(s_term[tm_off + rec.term_local_id])
                g_mask[c_i] = rec.is_gold
                ms_b[c_i] = 0.8 * rec.milestone_hits

            neg_mask = ~g_mask
            # 最良の正解トレース (マイルストーン一致 + スコア最大) 1本を基準にし、誤答パス群に対してのみマージン付き Listwise 損失を課す
            # (既に十分マージンがある正解問題の負例を持ち上げず、誤答・僅差問題に勾配を100%集中させる)
            gold_indices = np.where(g_mask)[0]
            best_g_idx = int(gold_indices[int(np.argmax(path_logits[gold_indices] + ms_b[gold_indices]))])
            s_star = path_logits[best_g_idx]

            neg_indices = np.where(neg_mask)[0]
            u_neg = path_logits[neg_indices] + margin - s_star
            u_max = max(0.0, float(np.max(u_neg)))
            exp_zero = math.exp(-u_max)
            exp_neg = np.exp(np.clip(u_neg - u_max, -50.0, 50.0))
            denom = exp_zero + float(np.sum(exp_neg))

            loss_p = u_max + math.log(denom)
            total_loss += loss_p

            d_path = np.zeros(n_paths, dtype=np.float32)
            probs_neg = exp_neg / denom
            d_path[neg_indices] = (probs_neg / m_probs).astype(np.float32)
            d_path[best_g_idx] = (-float(np.sum(probs_neg)) / m_probs)
            for c_i, rec in enumerate(g.paths):
                dp_val = d_path[c_i]
                d_term[tm_off + rec.term_local_id] += dp_val
                for tid in rec.trans_local_ids:
                    d_trans[tr_off + tid] += dp_val

        dH_reg = np.zeros_like(H_reg)
        dh_board = np.zeros_like(h_board)
        dH_text = np.zeros_like(H_text)
        dh_question = np.zeros_like(h_question)

        nn.backward_terminals_head(d_term, tm_cache, dH_reg, dH_text, dh_question, grads)
        if has_trans:
            nn.backward_transitions_head(d_trans, tr_cache, dH_reg, dh_board, dH_text, dh_question, grads)

        nn.backward_board_states_batch(dH_reg, dh_board, reg_cache, dH_text, grads)
        nn.backward_text_batch(dH_text, dh_question, txt_cache, grads)

        return total_loss / m_probs, grads, {"W_rel_tr": act_tr_rel, "W_rel_tm": act_tm_rel}

    def fit(
        self,
        train_problems: Sequence[Tuple[str, List[float], float]],
        val_problems: Optional[Sequence[Tuple[str, float]]] = None,
        epochs: int = 8,
        batch_size: int = 32,
    ) -> float:
        rng = np.random.default_rng(123)
        graphs, reachable = self.build_problem_candidate_graphs(train_problems)
        if train_problems:
            print(
                f"[TransformerTrainer] Oracle到達可能問題数: {reachable} / {len(train_problems)} "
                f"({reachable / len(train_problems) * 100:.1f}%), 有効学習グラフ数: {len(graphs)}, "
                f"総パラメータ数: {self.nnue.parameter_count():,}",
                flush=True,
            )
        if not graphs:
            return 0.0

        solver = BeamStackSolver(self.nnue, beam_width=24, max_depth=6)
        init_val_acc = self._eval_quick(solver, val_problems) if val_problems else -1.0
        best_val_acc = init_val_acc
        best_state = self._snapshot_weights()

        if not self.nnue.is_trained:
            self.nnue.params["w_tr_out"] = rng.normal(0.0, 0.004, self.nnue.params["w_tr_out"].shape).astype(np.float32)
            self.nnue.params["w_tm_out"] = rng.normal(0.0, 0.004, self.nnue.params["w_tm_out"].shape).astype(np.float32)

        if val_problems:
            print(f"[TransformerTrainer] Epoch 0 (Priorのみ初期精度): Val Acc = {init_val_acc * 100:.2f}%", flush=True)

        opt = _TransformerAdamW(self.nnue.params, lr=self.lr, weight_decay=self.weight_decay, max_grad_norm=1.2)
        ema_state = self._snapshot_weights()
        total_steps = max(1, epochs * ((len(graphs) + batch_size - 1) // batch_size))
        warmup_steps = max(1, total_steps // 12)
        global_step = 0

        for ep in range(1, epochs + 1):
            if ep == (epochs // 2) + 1 and len(train_problems) > 0:
                fresh_graphs, _ = self.build_problem_candidate_graphs(train_problems)
                if fresh_graphs:
                    # 初期グラフと現方策での Hard-Negative グラフから均等サンプリングしてエポック計算量と Cosine LR 整合を維持
                    combined_g = list(graphs) + fresh_graphs
                    keep_idx = rng.choice(len(combined_g), size=len(graphs), replace=False)
                    graphs = [combined_g[int(i)] for i in keep_idx]

            order = rng.permutation(len(graphs))
            ep_loss = 0.0
            n_batches = 0
            ema_decay = 0.92 if ep > max(1, epochs // 3) else 0.75
            for start in range(0, len(graphs), batch_size):
                global_step += 1
                if global_step <= warmup_steps:
                    opt.lr = self.lr * (float(global_step) / float(warmup_steps))
                else:
                    prog = float(global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    opt.lr = self.lr * (0.12 + 0.88 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

                batch = [graphs[int(i)] for i in order[start : start + batch_size]]
                b_loss, grads, act_rel = self.compute_loss_and_grads(batch, drop_rate=self.dropout, rng=rng)
                opt.step(self.nnue.params, grads, act_rel)
                for k, arr in self.nnue.params.items():
                    ema_state[k] = ema_decay * ema_state[k] + (1.0 - ema_decay) * arr
                ep_loss += b_loss
                n_batches += 1

            avg_loss = ep_loss / max(1, n_batches)
            if val_problems:
                live_snap = self._snapshot_weights()
                live_acc = self._eval_quick(solver, val_problems)
                self._restore_weights(ema_state)
                ema_acc = self._eval_quick(solver, val_problems)
                val_acc = max(live_acc, ema_acc)
                print(
                    f"[TransformerTrainer] Epoch {ep}/{epochs} | lr={opt.lr:.5f} | "
                    f"Loss = {avg_loss:.4f} | Val Acc = {val_acc * 100:.2f}% (EMA={ema_acc * 100:.2f}%)",
                    flush=True,
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.copy() for k, v in (ema_state if ema_acc >= live_acc else live_snap).items()}
                self._restore_weights(live_snap)

        self._restore_weights(best_state)
        self.nnue.is_trained = True
        return best_val_acc

    def _snapshot_weights(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.nnue.params.items()}

    def _restore_weights(self, snap: Dict[str, np.ndarray]) -> None:
        for k, v in snap.items():
            self.nnue.params[k] = v.copy()

    def _eval_quick(self, solver: BeamStackSolver, pairs: Sequence[Tuple[str, float]]) -> float:
        if not pairs:
            return 0.0
        ok = 0
        for q, gold in pairs:
            try:
                regs, cls_info, goal = ClauseRoleParser.parse(q)
                res = solver.solve_problem(regs, cls_info, goal)
                if res is not None and self._is_close_ans(res[0], gold):
                    ok += 1
            except Exception:
                pass
        return ok / len(pairs)


NNUETrainer = TransformerTrainer


# ==============================================================================
# 5. シングルトンエンジン管理と公開メイン関数 solve(text, check_is_math=False)
# ==============================================================================

_GLOBAL_NNUE: Optional[ESRTransformerEvaluator] = None
_GLOBAL_SOLVER: Optional[BeamStackSolver] = None


def get_engine() -> BeamStackSolver:
    global _GLOBAL_NNUE, _GLOBAL_SOLVER
    if _GLOBAL_SOLVER is None:
        _GLOBAL_NNUE = ESRTransformerEvaluator(seed=42)
        _GLOBAL_NNUE.load(MODEL_WEIGHTS_FILE)
        _GLOBAL_SOLVER = BeamStackSolver(_GLOBAL_NNUE, beam_width=26, max_depth=6)
    return _GLOBAL_SOLVER


def _format_answer(val: float) -> int | float:
    if not math.isfinite(val) or abs(val) > 1e15:
        return 0
    if abs(val - round(val)) < 1e-5:
        return int(round(val))
    return round(float(val), 4)


def solve(text: str, check_is_math: bool = False) -> Optional[int | float]:
    """
    算数文章題の判定および解答を行うメインインターフェース。
    - check_is_math=True: 算数文章題なら計算結果（非Noneの数値）、一般知識クイズ等なら None を返す。
    - check_is_math=False: 算数文章題として推論を実行し最終的な数値解答を返す（対象外・数値なしは None）。
    """
    if not text or not isinstance(text, str):
        return None

    if check_is_math:
        if not is_math_word_problem(text):
            return None

    try:
        regs, clauses, goal = ClauseRoleParser.parse(text)
        if not regs:
            return 0 if check_is_math else None

        solver = get_engine()
        res = solver.solve_problem(regs, clauses, goal)
        if res is None:
            return _format_answer(regs[-1].value)

        ans_val, _, _ = res
        return _format_answer(ans_val)
    except Exception:
        return 0 if check_is_math else None


# ==============================================================================
# 6. CLI 学習・ベンチマーク評価ユーティリティ (Windows 11 コンソール保護対応)
# ==============================================================================


def _configure_console_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        st = getattr(sys, stream_name, None)
        if st is not None and hasattr(st, "reconfigure"):
            try:
                st.reconfigure(errors="replace", line_buffering=True)
            except Exception:
                pass


def load_mawps_dataset(csv_path: str) -> List[Tuple[str, List[float], float]]:
    if not os.path.exists(csv_path):
        return []
    out: List[Tuple[str, List[float], float]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            q_raw = row.get("Question", "").strip()
            if "Body" in row and row["Body"].strip() and not q_raw.startswith(row["Body"].strip()[:10]):
                q_raw = f"{row['Body'].strip()} {q_raw}".strip()
            nums_str = row.get("Numbers", "").split()
            nums = []
            for ns in nums_str:
                try:
                    nums.append(float(ns))
                except ValueError:
                    pass

            def _repl(m: re.Match) -> str:
                idx = int(m.group(1))
                if idx < len(nums):
                    v = nums[idx]
                    return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)
                return m.group(0)

            q_restored = re.sub(r"\bnumber(\d+)\b", _repl, q_raw, flags=re.I)
            eq_str = row.get("Equation", "").strip()
            ans_str = row.get("Answer", "").strip()
            ms, eq_val = TraceCompiler.parse_equation_milestones(eq_str, nums)
            try:
                gold = float(ans_str.replace(",", ""))
            except ValueError:
                gold = eq_val if eq_val is not None else 0.0
            out.append((q_restored, ms, gold))
    return out


def load_svamp_dataset(json_path: str) -> List[Tuple[str, float]]:
    if not os.path.exists(json_path):
        return []
    out: List[Tuple[str, float]] = []
    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        for item in json.load(f):
            full_q = f"{item.get('Body', '').strip()} {item.get('Question', '').strip()}".strip()
            gold = float(item["Answer"])
            out.append((full_q, gold))
    return out


def load_gsm8k_dataset(jsonl_path: str, limit: Optional[int] = None) -> List[Tuple[str, List[float], float]]:
    if not os.path.exists(jsonl_path):
        return []
    out: List[Tuple[str, List[float], float]] = []
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = obj["question"].strip()
                ms, gold = TraceCompiler.parse_gsm8k_answer(obj["answer"])
                if gold is not None and math.isfinite(gold):
                    out.append((q, ms, gold))
            except Exception:
                continue
            if limit and len(out) >= limit:
                break
    return out


def evaluate_accuracy(name: str, items: Sequence[Tuple[str, float]]) -> float:
    if not items:
        print(f"[{name}] データセットが見つかりません（スキップ）", flush=True)
        return 0.0
    correct = 0
    total = len(items)
    t0 = time.perf_counter()
    for idx, (q, gold) in enumerate(items, 1):
        pred = solve(q, check_is_math=False)
        if pred is not None and math.isfinite(float(pred)) and abs(float(pred) - gold) <= max(1e-3, 1e-4 * abs(gold)):
            correct += 1
        if total >= 500 and idx % 500 == 0:
            print(f"  ... [{name}] 評価進行中: {idx}/{total} ({correct / idx * 100:.2f}%)", flush=True)
    dt = time.perf_counter() - t0
    acc = correct / total * 100.0
    print(f"[{name}] 正解率: {correct:4d} / {total:4d} ({acc:.2f}%)  [経過: {dt:.1f}s]", flush=True)
    return acc


def main() -> None:
    _configure_console_streams()
    parser = argparse.ArgumentParser(description="ESR-Transformer v22.0 Math Word Problem Solver & Trainer")
    parser.add_argument("--train", action="store_true", help="訓練データ (mawps_asdiv + gsm8k_train) のみから評価関数重みを学習・保存する")
    parser.add_argument("--epochs", type=int, default=8, help="学習エポック数")
    parser.add_argument("--gsm8k-limit", type=int, default=7500, help="gsm8k_train.jsonl の最大使用件数 (0でMAWPSのみ学習)")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer 隠れ層次元数 (n_heads の倍数)")
    parser.add_argument("--heads", type=int, default=4, help="Multi-Head Attention ヘッド数")
    parser.add_argument("--layers", type=int, default=2, help="問題文 Text Transformer 層数")
    parser.add_argument("--batch-size", type=int, default=32, help="ミニバッチ問題数")
    parser.add_argument("--lr", type=float, default=2.2e-3, help="ピーク学習率 (AdamW + Warmup Cosine)")
    parser.add_argument("--no-pause", action="store_true", help="Windows 実行完了時の Enter 入力待機をスキップする")
    args = parser.parse_args()

    try:
        download_files()
        mawps_data = load_mawps_dataset(TRAIN_FILE)
        svamp_data = load_svamp_dataset(TEST_FILE)
        gsm8k_train = load_gsm8k_dataset(GSM8K_TRAIN_FILE, limit=args.gsm8k_limit) if args.gsm8k_limit > 0 else []
        gsm8k_test = load_gsm8k_dataset(GSM8K_TEST_FILE)

        if args.train and (mawps_data or gsm8k_train):
            print(
                f"=== ESR-Transformer v22.0 (ALiBi Cross-Attention + Relational Highway: "
                f"d={args.d_model}, H={args.heads}, L={args.layers}) 学習開始 ===",
                flush=True,
            )
            global _GLOBAL_NNUE, _GLOBAL_SOLVER
            _GLOBAL_NNUE = ESRTransformerEvaluator(
                seed=42, d_model=args.d_model, n_heads=args.heads, n_text_layers=args.layers
            )
            _GLOBAL_SOLVER = BeamStackSolver(_GLOBAL_NNUE, beam_width=28, max_depth=6)

            rng = np.random.default_rng(42)
            # 層化検証分割: MAWPS と GSM8K の双方をバランス良く開発検証 (Val) に含め SVAMP/GSM8K 双方の汎化を担保
            val_set: List[Tuple[str, float]] = []
            trn_set: List[Tuple[str, List[float], float]] = []
            if mawps_data:
                p_m = rng.permutation(len(mawps_data))
                nv_m = max(1, min(200, len(mawps_data) // 10))
                val_set.extend((mawps_data[i][0], mawps_data[i][2]) for i in p_m[:nv_m])
                trn_set.extend(mawps_data[i] for i in p_m[nv_m:])
            if gsm8k_train:
                p_g = rng.permutation(len(gsm8k_train))
                nv_g = max(1, min(200, len(gsm8k_train) // 10))
                val_set.extend((gsm8k_train[i][0], gsm8k_train[i][2]) for i in p_g[:nv_g])
                trn_set.extend(gsm8k_train[i] for i in p_g[nv_g:])

            print(
                f"[Data Split] 訓練元: MAWPS/ASDiv={len(mawps_data)}問 + GSM8K(train)={len(gsm8k_train)}問 "
                f"-> Train={len(trn_set)}問, Balanced Val={len(val_set)}問 "
                f"(※SVAMP {len(svamp_data)}問・GSM8K(test) {len(gsm8k_test)}問は学習・Valに一切未使用)",
                flush=True,
            )

            trainer = TransformerTrainer(_GLOBAL_NNUE, lr=args.lr, weight_decay=8.0e-3, dropout=0.12)
            val_acc = trainer.fit(trn_set, val_problems=val_set, epochs=args.epochs, batch_size=args.batch_size)
            _GLOBAL_NNUE.save(MODEL_WEIGHTS_FILE)
            print(f"=== 学習完了 (Best Balanced Val Acc: {val_acc * 100:.2f}%) -> {MODEL_WEIGHTS_FILE} に保存 ===", flush=True)

        print("=== ベンチマーク評価 ===", flush=True)
        report_lines: List[str] = [
            f"=== ESR-Transformer v22.0 Benchmark Summary ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===",
            f"Weights File: {MODEL_WEIGHTS_FILE}",
        ]
        if mawps_data:
            acc_m = evaluate_accuracy("MAWPS/ASDiv", [(q, g) for q, _, g in mawps_data])
            report_lines.append(f"MAWPS/ASDiv : {acc_m:.2f}% ({len(mawps_data)} items)")
        if svamp_data:
            acc_s = evaluate_accuracy("SVAMP", svamp_data)
            report_lines.append(f"SVAMP       : {acc_s:.2f}% ({len(svamp_data)} items)")
        if gsm8k_test:
            acc_g = evaluate_accuracy("GSM8K (Test)", [(q, g) for q, _, g in gsm8k_test])
            report_lines.append(f"GSM8K (Test): {acc_g:.2f}% ({len(gsm8k_test)} items)")

        if len(report_lines) > 2:
            try:
                with open(BENCHMARK_REPORT_FILE, "w", encoding="utf-8") as rf:
                    rf.write("\n".join(report_lines) + "\n")
                print(f"[結果保存] ベンチマーク結果サマリを {BENCHMARK_REPORT_FILE} にも保存しました。", flush=True)
            except Exception:
                pass

    except Exception as exc:
        print(f"\n[エラー発生] 実行中に予期しない例外が発生しました: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        # Windows 11 でダブルクリック実行等された際に GSM8K 結果を確認する前にウィンドウが閉じるのを防止
        if os.name == "nt" and not args.no_pause:
            try:
                if sys.stdin is not None and sys.stdin.isatty():
                    input("\n[完了] 全ての評価が終了しました。Enterキーを押すとウィンドウを閉じます...")
            except Exception:
                pass


if __name__ == "__main__":
    main()