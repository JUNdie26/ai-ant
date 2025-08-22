# -*- coding: utf-8 -*-
"""
통합 분석 스크립트
- 여러 CSV 병합(pd.concat)
- 텍스트 컬럼 자동 감지(없으면 지정)
- 키워드(명사) 빈도 / 장소 추출
- 워드클라우드 생성 (Windows: malgun.ttf 사용)
- 요구사항: 키워드 빈도에서 '제주', '제주도' 제외

필요 패키지:
    pip install pandas konlpy wordcloud matplotlib jpype1
"""

import os
import re
import collections
import pandas as pd

# ====== 입력 CSV들 ======
csv_paths = [
    "csv/JT_SNS_KWRD_STATS_LIST_202206.csv",
    "csv/JT_SNS_KWRD_STATS_LIST_202212.csv",
    "csv/JT_SNS_KWRD_STATS_LIST_202306.csv",
    "csv/JT_SNS_KWRD_STATS_LIST_202312.csv",
]

# ====== 설정 ======
outdir     = "outputs_all"
text_col   = None          # None이면 자동 감지
topn       = 50
analyzer   = "okt"         # "okt" | "kkma" | "regex"
min_len    = 2
stopwords_file = None      # 외부 불용어 파일(.txt, 줄 단위) — 선택
place_suffix_file = None   # 장소 접미사 확장 파일 — 선택
font_path  = r"C:\Windows\Fonts\malgun.ttf"  # 워드클라우드 폰트(Windows TTF)

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
    # Fallback: 한글 2자 이상
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
    os.makedirs(outdir, exist_ok=True)

    # 1) 여러 CSV 로드 + 합치기
    dfs = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"[WARN] 파일 없음: {p} (건너뜀)")
            continue
        df_i = pd.read_csv(p, encoding="utf-8", low_memory=False)
        df_i["SOURCE"] = os.path.basename(p).replace(".csv","")
        dfs.append(df_i)

    if not dfs:
        raise SystemExit("[ERR] 읽을 수 있는 CSV가 없습니다.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 합쳐진 데이터: {df.shape}")

    # 2) 텍스트 컬럼 결정
    if text_col is None:
        text_col = autodetect_text_column(df)
    print(f"[INFO] 텍스트 컬럼: {text_col}")

    texts = df[text_col].fillna("").astype(str).tolist()

    # 3) 토크나이저
    kind, obj = get_tokenizer(analyzer)

    # 4) 불용어
    stopwords = set()
    if stopwords_file and os.path.exists(stopwords_file):
        with open(stopwords_file, "r", encoding="utf-8") as f:
            stopwords = {ln.strip() for ln in f if ln.strip()}

    # ✅ 키워드 빈도에서 '제주', '제주도' 제외
    stopwords.update({"제주", "제주도"})

    # 5) 명사 카운트 (키워드)
    noun_counter = collections.Counter()
    for t in texts:
        nouns = extract_nouns(t, kind, obj, min_len=min_len)
        nouns = [w for w in nouns if w not in stopwords]
        noun_counter.update(nouns)

    # 6) 장소 카운트 (장소에서는 '제주/제주도'를 제외하지 않음)
    place_suffixes = DEFAULT_PLACE_SUFFIXES[:]
    if place_suffix_file and os.path.exists(place_suffix_file):
        with open(place_suffix_file, "r", encoding="utf-8") as f:
            place_suffixes.extend([ln.strip() for ln in f if ln.strip()])
    place_regex = compile_place_pattern(place_suffixes)

    place_counter = collections.Counter()
    for t in texts:
        place_counter.update(extract_places(t, place_regex))

    # 7) 전체 결과 저장
    kw_df = pd.DataFrame(noun_counter.most_common(topn), columns=["keyword","count"])
    kw_df.to_csv(os.path.join(outdir,"keyword_frequencies_all.csv"), index=False, encoding="utf-8")

    place_df = pd.DataFrame(place_counter.most_common(topn), columns=["place","count"])
    place_df.to_csv(os.path.join(outdir,"place_frequencies_all.csv"), index=False, encoding="utf-8")

    # 8) 행별 미리보기 (상위 20/10)
    preview = df[[text_col, "SOURCE"]].copy()
    preview["extracted_nouns_top20"] = df[text_col].astype(str).apply(
        lambda t: " ".join([w for w in extract_nouns(t, kind, obj, min_len=min_len) if w not in stopwords][:20])
    )
    preview["extracted_places_top10"] = df[text_col].astype(str).apply(
        lambda t: " | ".join(extract_places(t, place_regex)[:10])
    )
    preview.to_csv(os.path.join(outdir,"extracted_per_row_all.csv"), index=False, encoding="utf-8")

    # 9) SOURCE(파일별) TopN도 추가 저장
    # SOURCE별 키워드
    source_kw_rows = []
    for src, sub in df.groupby("SOURCE"):
        sub_nouns = []
        for t in sub[text_col].fillna(""):
            ns = extract_nouns(t, kind, obj, min_len=min_len)
            ns = [w for w in ns if w not in stopwords]  # 제주/제주도 제외 유지
            sub_nouns.extend(ns)
        c = collections.Counter(sub_nouns).most_common(topn)
        for k, v in c:
            source_kw_rows.append({"SOURCE": src, "keyword": k, "count": v})
    pd.DataFrame(source_kw_rows).to_csv(os.path.join(outdir,"keyword_frequencies_by_source.csv"),
                                        index=False, encoding="utf-8")

    # SOURCE별 장소
    source_place_rows = []
    for src, sub in df.groupby("SOURCE"):
        sub_places = []
        for t in sub[text_col].fillna(""):
            sub_places.extend(extract_places(t, place_regex))
        c = collections.Counter(sub_places).most_common(topn)
        for k, v in c:
            source_place_rows.append({"SOURCE": src, "place": k, "count": v})
    pd.DataFrame(source_place_rows).to_csv(os.path.join(outdir,"place_frequencies_by_source.csv"),
                                           index=False, encoding="utf-8")

    # 10) 워드클라우드 (전체 키워드)
    wc_png = os.path.join(outdir,"wordcloud_all.png")
    msg = try_wordcloud(dict(noun_counter.most_common(200)), wc_png, font_path=font_path)
    print("[WordCloud]", msg)

    print("=== Done ===")
    print("Outputs in:", os.path.abspath(outdir))