import os
import re
import collections
import pandas as pd

# 설정값
csv_dir   = "csv"       # <- 여기에 CSV 파일이 모여 있는 폴더
outdir    = "outputs"   # 결과 저장 폴더
text_col  = None        # None이면 자동 감지
topn      = 50
analyzer  = "okt"       # "okt" | "kkma" | "regex"
min_len   = 2
stopwords = {"제주","제주도"}  # 기본 불용어 (원하면 확장 가능)
font_path = r"C:\Windows\Fonts\malgun.ttf"   # 워드클라우드 폰트

# 토크나이저

def get_tokenizer(analyzer="okt"):
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
        print(f"[WARN] KoNLPy 로드 실패({e}), regex로 대체")
        return ("regex", None)

def extract_tokens(text, kind, obj, min_len=2):
    s = str(text) if text is not None else ""
    if kind == "okt" and obj:
        try:
            return [w for w in obj.nouns(s) if len(w) >= min_len]
        except: pass
    elif kind == "kkma" and obj:
        try:
            return [w for w in obj.nouns(s) if len(w) >= min_len]
        except: pass
    return re.findall(rf"[가-힣]{{{min_len},}}", s)

# 장소 추출
PLACE_SUFFIXES = [
    "카페","커피","식당","분식","한식","중식","일식","양식","뷔페","족발","보쌈","곱창","횟집","치킨",
    "베이커리","빵집","제과점","피자","버거","포차","주점","술집","바","펍",
    "호텔","리조트","펜션","게스트하우스","모텔",
    "공원","해변","해수욕장","산","호수","강","섬","계곡","폭포","사찰","성당","교회","전망대",
    "미술관","박물관","수목원","식물원","동물원","수족관",
    "시장","전통시장","프리미엄아울렛","아울렛","백화점","몰","쇼핑몰",
    "역","터미널","정거장"
]  # 고유 장소 이름에서 자주 끝나는 접미어들을 모아놓음. 정규 표현식으로 리스트에서 하나로 끝나는 문자열을 잡아서 장소명 후보로 인식


GENERIC_PLACE_BLACKLIST = {"호텔","식당","카페","공원","시장","역","바","펍"}   # 접미어로써 "호텔에서 잤다" 와 같은 상황에서는 무슨 호텔인지도 모호하며 정보의 불확실성으로 인해 단독적 사용 시 BAN

def compile_place_pattern(suffixes):
    suffix_alt = "|".join(map(re.escape, suffixes))
    return re.compile(rf"([가-힣A-Za-z0-9][가-힣A-Za-z0-9\s\-·&()']{{0,20}}?(?:{suffix_alt}))")

def extract_places(text, regex):
    s = str(text) if text is not None else ""
    cands = regex.findall(s)
    out = []
    for c in cands:
        name = re.sub(r"\s{2,}", " ", c).strip()
        if name in GENERIC_PLACE_BLACKLIST:
            continue
        out.append(name)
    return out


# 텍스트 컬럼 자동 감지
PRIORITY_KEYWORDS = ["text","content","review","body","message","comment","본문","내용","게시","글","후기"]

def autodetect_text_column(df: pd.DataFrame):
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return df.columns[0]
    def score_col(c: str):
        score = sum(5 for kw in PRIORITY_KEYWORDS if kw in c.lower())
        try:
            score += (df[c].dropna().astype(str).str.len().mean() or 0)/100
        except: pass
        return score
    obj_cols.sort(key=score_col, reverse=True)
    return obj_cols[0]


# WordCloud
def try_wordcloud(freqs, out_png, font_path=None):
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=1200, height=800, font_path=font_path, background_color="white")
        img = wc.generate_from_frequencies(freqs).to_image()
        img.save(out_png)
        return out_png
    except Exception as e:
        return f"[WARN] WordCloud 실패: {e}"


# 실행부
if __name__ == "__main__":
    os.makedirs(outdir, exist_ok=True)

    # 1) 디렉토리 내 CSV 로드
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        raise SystemExit(f"[ERR] {csv_dir} 내 CSV 파일이 없습니다.")
    dfs = []
    for f in csv_files:
        df_i = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df_i["SOURCE"] = os.path.basename(f)
        dfs.append(df_i)
    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 합쳐진 데이터: {df.shape}")

    # 2) 텍스트 컬럼
    if text_col is None:
        text_col = autodetect_text_column(df)
    print(f"[INFO] 텍스트 컬럼: {text_col}")

    # 3) 토큰화
    kind, obj = get_tokenizer(analyzer)

    # 4) 키워드 빈도
    noun_counter = collections.Counter()
    for t in df[text_col].fillna(""):
        toks = extract_tokens(t, kind, obj, min_len=min_len)
        toks = [w for w in toks if w not in stopwords]
        noun_counter.update(toks)

    # 5) 장소 빈도
    place_regex = compile_place_pattern(PLACE_SUFFIXES)
    place_counter = collections.Counter()
    for t in df[text_col].fillna(""):
        place_counter.update(extract_places(t, place_regex))

    # 6) 결과 저장
    pd.DataFrame(noun_counter.most_common(topn), columns=["keyword","count"]) \
        .to_csv(os.path.join(outdir,"keyword_frequencies.csv"), index=False, encoding="utf-8")
    pd.DataFrame(place_counter.most_common(topn), columns=["place","count"]) \
        .to_csv(os.path.join(outdir,"place_frequencies.csv"), index=False, encoding="utf-8")

    # 7) 워드클라우드
    wc_png = os.path.join(outdir,"wordcloud.png")
    msg = try_wordcloud(dict(noun_counter.most_common(200)), wc_png, font_path=font_path)
    print("[WordCloud]", msg)

    print("=== Done ===")
    print("Outputs in:", os.path.abspath(outdir))
