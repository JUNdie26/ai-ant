import os
import re
import collections
import pandas as pd

csv_path   = "csv/JT_SNS_KWRD_STATS_LIST_202312.csv"
outdir     = "outputs"
text_col   = None         # None이면 자동 감지
topn       = 50
analyzer   = "okt"        # "okt" | "kkma" | "regex"
min_len    = 2
stopwords_file = None     # 불용어 파일 경로 (.txt, 줄 단위)
place_suffix_file = None  # 장소 접미사 확장 파일
font_path  = r"C:\Windows\Fonts\malgun.ttf"  # 맑은 고딕 TTF


# ====== Tokenizer 준비 ======
def get_tokenizer(analyzer: str = "okt"):
    analyzer = (analyzer or "okt").lower()
    if analyzer == "regex":
        return ("regex", None)
    try:
        if analyzer == "kkma":
            from konlpy.tag import Kkma
            return ("kkma", Kkma())
        else:
            from konlpy.tag import Okt
            return ("okt", Okt())
    except Exception as e:
        print(f"[WARN] KoNLPy 불러오기 실패({e}), regex로 대체")
        return ("regex", None)

def extract_nouns(text, kind, obj, min_len=2):
    if not isinstance(text, str):
        text = str(text)
    if kind == "okt" and obj:
        try:
            return [w for w in obj.nouns(text) if len(w) >= min_len]
        except:
            pass
    elif kind == "kkma" and obj:
        try:
            return [w for w in obj.nouns(text) if len(w) >= min_len]
        except:
            pass
    return re.findall(r"[가-힣]{%d,}" % min_len, text)

# ====== 장소 추출 ======
DEFAULT_PLACE_SUFFIXES = [
    "카페","커피","식당","분식","한식","중식","일식","양식","뷔페","족발","보쌈","곱창","횟집","치킨",
    "베이커리","빵집","제과점","피자","버거","포차","주점","술집","바","펍",
    "호텔","리조트","펜션","게스트하우스","모텔",
    "공원","해변","해수욕장","산","호수","강","섬","계곡","폭포","사찰","성당","교회","전망대",
    "미술관","박물관","수목원","식물원","동물원","수족관",
    "시장","전통시장","프리미엄아울렛","아울렛","백화점","몰","쇼핑몰",
    "역","터미널","정거장"
]
GENERIC_PLACE_BLACKLIST = {"호텔","식당","카페","공원","시장","역","바","펍"}

def compile_place_pattern(suffixes):
    suffix_alt = "|".join(map(re.escape, suffixes))
    pattern = rf"([가-힣A-Za-z0-9][가-힣A-Za-z0-9\s\-·&()']{{0,20}}?(?:{suffix_alt}))"
    return re.compile(pattern)

def extract_places(text, regex, blacklist=GENERIC_PLACE_BLACKLIST):
    if not isinstance(text, str):
        text = str(text)
    candidates = regex.findall(text)
    cleaned = []
    for c in candidates:
        name = re.sub(r"\s{2,}", " ", c).strip()
        name = re.sub(r"^[\"'“”‘’\(\)\[\]]+|[\"'“”‘’\(\)\[\]]+$", "", name)
        if name in blacklist:
            continue
        if len(name) < 2:
            continue
        cleaned.append(name)
    return cleaned

# ====== 텍스트 컬럼 자동 감지 ======
PRIORITY_KEYWORDS = ["text","content","review","body","message","comment","본문","내용","게시","글","후기"]

def autodetect_text_column(df: pd.DataFrame):
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return df.columns[0]
    def score_col(c: str) -> float:
        name = c.lower()
        score = sum(5 for kw in PRIORITY_KEYWORDS if kw in name)
        try:
            avg_len = df[c].dropna().astype(str).str.len().mean() or 0.0
        except:
            avg_len = 0.0
        return score + avg_len / 100.0
    obj_cols.sort(key=score_col, reverse=True)
    return obj_cols[0]

# ====== WordCloud ======
def try_wordcloud(freqs, out_png, font_path=None):
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=1200, height=800, font_path=font_path, background_color="white")
        img = wc.generate_from_frequencies(freqs).to_image()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        img.save(out_png)
        return out_png
    except Exception as e:
        return f"[WARN] WordCloud 실패: {e}"

# ================== 실행부 ==================
if __name__ == "__main__":
    # Load
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    # Text column
    if text_col is None:
        text_col = autodetect_text_column(df)
    print(f"[INFO] 텍스트 컬럼: {text_col}")

    texts = df[text_col].fillna("").astype(str).tolist()

    # Tokenizer
    kind, obj = get_tokenizer(analyzer)

    # Stopwords
    stopwords = set()
    if stopwords_file and os.path.exists(stopwords_file):
        with open(stopwords_file, "r", encoding="utf-8") as f:
            stopwords = {ln.strip() for ln in f if ln.strip()}

    # Nouns
    noun_counter = collections.Counter()
    for t in texts:
        nouns = extract_nouns(t, kind, obj, min_len=min_len)
        nouns = [w for w in nouns if w not in stopwords]
        noun_counter.update(nouns)

    # Places
    place_suffixes = DEFAULT_PLACE_SUFFIXES[:]
    if place_suffix_file and os.path.exists(place_suffix_file):
        with open(place_suffix_file, "r", encoding="utf-8") as f:
            place_suffixes.extend([ln.strip() for ln in f if ln.strip()])
    place_regex = compile_place_pattern(place_suffixes)

    place_counter = collections.Counter()
    for t in texts:
        place_counter.update(extract_places(t, place_regex))

    # Save outputs
    os.makedirs(outdir, exist_ok=True)

    kw_df = pd.DataFrame(noun_counter.most_common(topn), columns=["keyword","count"])
    kw_df.to_csv(os.path.join(outdir,"keyword_frequencies.csv"), index=False, encoding="utf-8")

    place_df = pd.DataFrame(place_counter.most_common(topn), columns=["place","count"])
    place_df.to_csv(os.path.join(outdir,"place_frequencies.csv"), index=False, encoding="utf-8")

    preview = df[[text_col]].copy()
    preview["extracted_nouns_top20"] = df[text_col].astype(str).apply(
        lambda t: " ".join(extract_nouns(t, kind, obj, min_len=min_len)[:20])
    )
    preview["extracted_places_top10"] = df[text_col].astype(str).apply(
        lambda t: " | ".join(extract_places(t, place_regex)[:10])
    )

    wc_png = os.path.join(outdir,"wordcloud.png")
    msg = try_wordcloud(dict(noun_counter.most_common(200)), wc_png, font_path=font_path)
    print("[WordCloud]", msg)

    print("=== Done ===")
    print("Outputs in:", outdir)
