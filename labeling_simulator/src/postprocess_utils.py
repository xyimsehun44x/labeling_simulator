import pandas as pd
import json
import os
import re
import numpy as np

# A constant list of features to ensure consistent column ordering and naming.
FEATURE_LIST = [
    "Design", "Price", "Battery", "Camera", "Display", "AP/Memory", 
    "AI", "S-Pen", "UI/UX", "Connected_Exp", "Durability", "Game", 
    "Audio", "Security"
]

BRAND_LIST = [
    "Samsung", "Co.A", "Google", "Huawei/Honor", "Infinix", "Itel", 
    "Oneplus", "Oppo", "Realme", "Tecno", "Vivo", "Xiaomi"
]

# --- LLM Response Schema (Defaults) ---
# 작성자: 강보선
# 날짜: 1월 19일 
# 목적: 제미나이한테 물어볼 때, 어떤 형식으로 답변을 받을지 정의하는 코드 

SENTIMENT_ALLOWED = {-1, 0, 1}
STATUS_ALLOWED = {0, 1}
SENTIMENT_LABELS = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}
SENTIMENT_KEYWORDS_PATH = os.environ.get(
    "SENTIMENT_KEYWORDS_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "sentiment_keywords.json")),
)
SENTIMENT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_SIM_THRESHOLD = 0.30
SENTIMENT_MARGIN = 0.05
_SENTIMENT_MODEL = None
_SENTIMENT_KEYWORD_EMBEDS = None

def get_flattened_output_columns():
    columns = ['id', 'consumer_buzz', 'overall_sentiment']
    columns += [f'feature_sentiments_{feat}' for feat in FEATURE_LIST]
    columns += ['comparison_superiority_status']
    columns += [f'comparison_superiority_{feat}' for feat in FEATURE_LIST]
    columns += ['switching_intent_status']
    columns += [f'switching_intent_{feat}' for feat in FEATURE_LIST]
    return columns

# --- 들어오는 값 파싱  ---

def _coerce_int(value, allowed=None, default=0):
    if value is None:
        return default
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return default
    if allowed is not None and value_int not in allowed:
        return default
    return value_int

def _coerce_nullable_int(value, allowed=None):
    if value is None:
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    if allowed is not None and value_int not in allowed:
        return None
    return value_int

def _strip_code_fences(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z]*\n', '', text)
        if text.endswith('```'):
            text = text[:-3]
    return text.strip()

def _find_first_json_object(text):
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def _extract_json_from_text(text):
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    candidate = _find_first_json_object(cleaned)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def flatten_entry(key, parts):
    row = {'key': key}
    row['consumer_buzz'] = None
    row['overall_sentiment'] = None
    for feat in FEATURE_LIST:
        row[f'feature_sentiments_{feat}'] = 0
    row['comparison_superiority_status'] = 0
    for feat in FEATURE_LIST:
        row[f'comparison_superiority_{feat}'] = 0
    row['switching_intent_status'] = 0
    for feat in FEATURE_LIST:
        row[f'switching_intent_{feat}'] = 0

    if not parts or not isinstance(parts, dict):
        return row

    row['consumer_buzz'] = _coerce_nullable_int(parts.get('consumer_buzz'))
    row['overall_sentiment'] = _coerce_nullable_int(
        parts.get('overall_sentiment'),
        allowed=SENTIMENT_ALLOWED,
    )
    
    fs = parts.get('feature_sentiments', {})
    if not isinstance(fs, dict):
        fs = {}
    if fs:
        for feat in FEATURE_LIST:
            row[f'feature_sentiments_{feat}'] = _coerce_int(
                fs.get(feat, 0),
                allowed=SENTIMENT_ALLOWED,
                default=0,
            )

    comp = parts.get('comparison_superiority', {})
    if not isinstance(comp, dict):
        comp = {}
    if comp:
        row['comparison_superiority_status'] = _coerce_int(
            comp.get('status', 0),
            allowed=STATUS_ALLOWED,
            default=0,
        )
        comp_feats = comp.get('features', {})
        if not isinstance(comp_feats, dict):
            comp_feats = {}
        if comp_feats:
            for feat in FEATURE_LIST:
                row[f'comparison_superiority_{feat}'] = _coerce_int(
                    comp_feats.get(feat, 0),
                    allowed=SENTIMENT_ALLOWED,
                    default=0,
                )

    switch = parts.get('switching_intent', {})
    if not isinstance(switch, dict):
        switch = {}
    if switch:
        row['switching_intent_status'] = _coerce_int(
            switch.get('status', 0),
            allowed=STATUS_ALLOWED,
            default=0,
        )
        switch_feats = switch.get('features', {})
        if not isinstance(switch_feats, dict):
            switch_feats = {}
        if switch_feats:
            for feat in FEATURE_LIST:
                row[f'switching_intent_{feat}'] = _coerce_int(
                    switch_feats.get(feat, 0),
                    allowed=SENTIMENT_ALLOWED,
                    default=0,
                )
        
    return row

def process_jsonl_to_df(input_jsonl_paths: list):
    all_rows = []
    print(f"📄 Parsing {len(input_jsonl_paths)} JSONL file(s)...")
    
    for file_path in input_jsonl_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(f, desc=f"Parsing {os.path.basename(file_path)}", unit="lines")
                except Exception:
                    iterator = f
                for i, line in enumerate(iterator):
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        key = data.get('key')
                        if key is None: continue
                        
                        try:
                            prediction_text = data['response']['candidates'][0]['content']['parts'][0]['text']
                        except (KeyError, IndexError, TypeError) as e:
                            print(f"   Skipping line #{i+1}: Could not extract prediction text. {e}")
                            continue

                        parts_dict = _extract_json_from_text(prediction_text)
                        if not parts_dict or not isinstance(parts_dict, dict):
                            print(f"   Skipping line #{i+1}: Could not parse JSON object.")
                            continue

                        flat_row = flatten_entry(key, parts_dict)
                        all_rows.append(flat_row)

                    except (json.JSONDecodeError, AttributeError) as e:
                        print(f"   Skipping line #{i+1} parsing error: {e}")
        except Exception as e:
            print(f" Error reading file {file_path}: {e}") 
            continue
        # 요 부분 강보선 테스팅 땜에 있음. 나중에 삭제할 예정 

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['id'] = df['key']
    df = df.drop(columns=['key'])
    return df

def merge_data(preprocessed_df, output_df):
    if 'id' not in preprocessed_df.columns or 'id' not in output_df.columns:
        raise ValueError("Both DataFrames must contain an 'id' column for merging.")
    merged_df = pd.merge(preprocessed_df, output_df, on='id', how='left')
    return merged_df

def apply_bs_logic(df):
    print("  Applying post-processing business logic rules...")
    
    comp_status_col = 'comparison_superiority_status'
    comp_feat_cols = [f'comparison_superiority_{feat}' for feat in FEATURE_LIST]
    switch_status_col = 'switching_intent_status'
    switch_feat_cols = [f'switching_intent_{feat}' for feat in FEATURE_LIST]
    sentiment_feat_cols = [f'feature_sentiments_{feat}' for feat in FEATURE_LIST]

    # Rule 1: Zero out unmentioned features
    if 'Features_Mentioned' in df.columns:
        for feat in FEATURE_LIST:
            unmentioned_mask = df['Features_Mentioned'].fillna('').str.contains(feat) == False
            df.loc[unmentioned_mask, [f'feature_sentiments_{feat}', f'comparison_superiority_{feat}', f'switching_intent_{feat}']] = 0

    # Rule 2: Single Brand Mention
    if 'Brands_Mentioned' in df.columns:
        single_brand_mask = (df['Brands_Mentioned'].fillna('').str.contains(',') == False) & \
                            (df['Brands_Mentioned'].fillna('') != '')
        df.loc[single_brand_mask, [comp_status_col] + comp_feat_cols + [switch_status_col] + switch_feat_cols] = 0

    # Rule 3: News Source Sentiment
    if 'source' in df.columns:
        news_sources = ["Radio", "TV", "News", "Print"]
        news_mask = df['source'].isin(news_sources)
        df.loc[news_mask, ['overall_sentiment'] + sentiment_feat_cols] = 0

    add_keyword_sentiment(df)
    add_sentiment_labels(df)

    return df

def load_sentiment_keywords(path=SENTIMENT_KEYWORDS_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

def get_sentiment_model():
    global _SENTIMENT_MODEL
    if _SENTIMENT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTIMENT_MODEL = SentenceTransformer(SENTIMENT_EMBEDDING_MODEL)
    return _SENTIMENT_MODEL

def get_keyword_embeddings(model, keywords):
    if not keywords:
        return None
    return model.encode(keywords, normalize_embeddings=True)

def add_keyword_sentiment(df):
    if 'text' not in df.columns:
        return

    keywords = load_sentiment_keywords()
    pos_keywords = keywords.get("positive", [])
    neg_keywords = keywords.get("negative", [])
    if not pos_keywords and not neg_keywords:
        return

    model = get_sentiment_model()
    pos_embeds = get_keyword_embeddings(model, pos_keywords)
    neg_embeds = get_keyword_embeddings(model, neg_keywords)

    texts = df['text'].fillna('').astype(str).tolist()
    if not texts:
        return
    text_embeds = model.encode(texts, normalize_embeddings=True)

    if pos_embeds is None:
        pos_max = np.zeros(len(texts), dtype=float)
    else:
        pos_max = (text_embeds @ pos_embeds.T).max(axis=1)

    if neg_embeds is None:
        neg_max = np.zeros(len(texts), dtype=float)
    else:
        neg_max = (text_embeds @ neg_embeds.T).max(axis=1)

    labels = []
    for pos_sim, neg_sim in zip(pos_max, neg_max):
        best = max(pos_sim, neg_sim)
        if best < SENTIMENT_SIM_THRESHOLD:
            labels.append(0)
        elif pos_sim >= neg_sim + SENTIMENT_MARGIN:
            labels.append(1)
        elif neg_sim >= pos_sim + SENTIMENT_MARGIN:
            labels.append(-1)
        else:
            labels.append(0)

    df['overall_sentiment_kw'] = labels
    df['overall_sentiment_kw_label'] = pd.Series(labels).map(SENTIMENT_LABELS).fillna('')

    if os.environ.get("USE_KEYWORD_SENTIMENT", "0") == "1":
        df['overall_sentiment'] = df['overall_sentiment_kw']

def add_sentiment_labels(df):
    sentiment_cols = ['overall_sentiment'] + [f'feature_sentiments_{feat}' for feat in FEATURE_LIST]
    for col in sentiment_cols:
        if col in df.columns:
            label_col = f"{col}_label"
            df[label_col] = df[col].map(SENTIMENT_LABELS).fillna('')
