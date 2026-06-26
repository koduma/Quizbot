# Quizbotです。
## 日本語と英語に対応してます。
## 単語とその説明文を687万enwiki冒頭記事収録
# Linux:
# Quizbot動かす前に、source .venv/bin/activate
# 新pcに引っ越しするとき
### ステップ0:旧pcのvenv上で、pip freeze > requirements.txt
### ステップ1:mkdir Quizbot
### ステップ2:cd Quizbot
### ステップ3:requirements.txtをQuizbotに置く
### ステップ4:python3 -m venv.venv
### ステップ5:source .venv/bin/activate
### 最終ステップ:venv上でpip install -r requirements.txt
# Windows:
# Quizbot動かす前に、`.\.venv\Scripts\activate`
# 新pcに引っ越しするとき
### ステップ0:旧pcのコマンドプロンプトのvenv上で、pip freeze > requirements_windows.txt
### ステップ1:mkdir Quizbot
### ステップ2:cd Quizbot
### ステップ3:requirements_windows.txtをQuizbotに置く
### ステップ4:python -m venv .venv
### ステップ5:`.\.venv\Scripts\activate`
### 最終ステップ:venv上でpip install -r requirements_windows.txt
