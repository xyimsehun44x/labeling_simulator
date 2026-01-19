import pandas as pd
import yaml
import json
from jinja2 import Template
import os
import math
import numpy as np
import re

HTR_PHRASES = [
    "Want to turn bad weather into a great weekend? Pivot your plans easily with a true AI companion",
    "Steal the spotlight in any light with AI #portrait on the #GalaxyS25Ultra. #GalaxyAI #GalaxyS25",
    "Want to capture the night in the best light? Discover Nightography and Audio Eraser on the #GalaxyS25Ultra. #GalaxyAI #GalaxyS25",
    "boosted by a customized processor that's our most powerful ever. #GalaxyAI #PlayGalaxy",
    "boosted by a customized processor that's our most powerful ever.",
    "There's a lot more that happened after the #GalaxyUnpacked presentation!",
    "follow this thread and get live updates on all the epic events as they happen with #GalaxyS25 and #GalaxyAI",
    "2022 champs Tribe face runners-up Luminosity again at #PlayGalaxy Cup 2025!",
    "Follow the thread for more #GalaxyAI and #GalaxyS25Ultra action!",
    "#GalaxyUnpacked was such a blast! From zooming in on special guests to capturing epic moments",
    "Seize the day the easy way with your very own Now Brief and Now Bar on the #GalaxyS25Ultra.",
    "No time to make dinner plans? A true AI companion",
    "follow this thread and get live updates on all the epic events as they happen with #GalaxyS25",
    "This is a whole new way to experience #GalaxyAI",
    "It's time! Join us LIVE at #GalaxyUnpacked. A true AI companion has arrived.",
    "New ways to get things done are almost here. #GalaxyAI",
    "You're now unsubscribed from #GalaxyUnpacked updates. #GalaxyAI",
    "We spy with our lil' eye",
    "The #GalaxyAI stars illuminating the night are just a taste of what's coming at",
    "All set to unpack the latest #GalaxyAI news! We'll ping you before #GalaxyUnpacked begins."
]

# --- From preprocessing.py ---

def concatenate_csvs(input_paths, output_path):
    """
    Concatenates multiple CSV files into a single CSV file.
    """
    if not input_paths:
        print("⚠️ No CSV files to concatenate.")
        return

    print(f"⚙️ Concatenating {len(input_paths)} CSV file(s) into {output_path}...")
    
    dfs = []
    for file_path in input_paths:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue 
    
    if not dfs:
        print("❌ No dataframes were created. Concatenation failed.")
        return

    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df.to_csv(output_path, index=False)
    print(f"✅ Concatenated file saved to {output_path}")

def preprocess_data(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # 1. Transform column names
    def transform_col_name(col):
        if col == 'Message Content':
            return 'text'
        if col == 'Source':
            return 'source'
        if col == 'Mentions (SUM)':
            return 'mentions'
        
        # Dynamic Product Column Mapping
        # Matches prefixes like "260107_PRODUCT_..." or just "PRODUCT_..." 
        # Maps to "Product_..."
        match_product = re.match(r'(?:\d+_)?PRODUCT_(.+)', col)
        if match_product:
            # Preserve the suffix (e.g., S25_Family, S26_Ultra) verbatim
            return f"Product_{match_product.group(1)}"

        # Dynamic Galaxy AI Mapping
        # Matches "260107_Galaxy_AI_Total" or similar
        if 'Galaxy_AI_Total' in col:
            return 'GalaxyAI'
        
        match_brand = re.match(r'\d+_BRAND_(.+)_Mentions \(SUM\)', col)
        if match_brand:
            return f"Brand_{match_brand.group(1)}"
        
        match_feature = re.match(r'\d+_FEATURE_(.+)_Mentions \(SUM\)', col)
        if match_feature:
            return f"Feature_{match_feature.group(1)}"
        
        return col

    df.columns = [transform_col_name(col) for col in df.columns]
    
    print(f"🔎 DEBUG: Columns after renaming: {list(df.columns)}")
    product_cols_debug = [col for col in df.columns if col.startswith('Product_') or col == 'GalaxyAI']
    print(f"🔎 DEBUG: Detected Product Cols: {product_cols_debug}")

    # 2. Add 'id' column
    df.insert(0, 'id', range(len(df)))

    # 3. Clean 'text' column for CSV compatibility
    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).fillna('')
        df['text'] = df['text'].str.replace(r'[\n\r\t]+', ' ', regex=True)
        df['text'] = df['text'].str.replace('"', '', regex=False)

    # 4. Process Brand, Feature, and Product columns
    brand_cols = [col for col in df.columns if col.startswith('Brand_')]
    feature_cols = [col for col in df.columns if col.startswith('Feature_')]
    
    # Dynamic detection of product columns (GalaxyAI and any Product_*)
    product_cols = [col for col in df.columns if col.startswith('Product_') or col == 'GalaxyAI']
    
    target_cols = brand_cols + feature_cols + product_cols
    
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = np.where(
            (df['mentions'] != 0) & (df[col] / df['mentions'] == 1), 
            1, 
            0
        )

    # 5. Get all mentioned Brands and Features
    def get_brands_mentioned(row):
        brands = [col.replace('Brand_', '') for col in brand_cols if row[col] == 1]
        return ', '.join(brands)

    def get_features_mentioned(row):
        features = [col.replace('Feature_', '') for col in feature_cols if row[col] == 1]
        return ', '.join(features)

    df['Brands_Mentioned'] = df.apply(get_brands_mentioned, axis=1)
    df['Features_Mentioned'] = df.apply(get_features_mentioned, axis=1)

    # 6. Create HTR Column
    print("⚙️ Generating HTR column based on phrase matching...")
    # Escape special regex characters in phrases and join with OR operator
    htr_pattern = '|'.join(map(re.escape, HTR_PHRASES))
    # Case-insensitive matching
    df['HTR'] = df['text'].str.contains(htr_pattern, case=False, regex=True).astype(int)

    # Save the updated dataframe
    if output_file.endswith('.xlsx'):
        try:
            with pd.ExcelWriter(output_file, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_urls': False}}) as writer:
                df.to_excel(writer, index=False)
            print(f"Processed data saved to {output_file} (Excel)")
        except Exception as e:
            print(f"❌ Error saving to Excel: {e}")
            raise e
    else:
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file} (CSV)")


# --- From formatter.py ---

def parse_list_col(val):
    if pd.isna(val) or val == '' or str(val).strip() == '0':
        return []
    return [x.strip() for x in str(val).split(',')]

def load_prompt(prompt_file):
    print(f"📖 Loading prompt from {prompt_file}...")
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
            raw_template_string = prompts.get('inference_prompt')
            if not raw_template_string:
                raise ValueError(f"Key 'inference_prompt' not found in {prompt_file}")
            return Template(raw_template_string)
    except Exception as e:
        print(f"❌ Error reading YAML: {e}")
        raise

def transform_data(df, jinja_template):
    jsonl_data = []
    print(f"⚙️  Processing {len(df)} rows for transformation...")
    
    for index, row in df.iterrows():
        text_content = row.get('text', '')
        if pd.isna(text_content) or str(text_content).strip() == "":
            continue

        brands = parse_list_col(row.get('Brands_Mentioned', ''))
        features = parse_list_col(row.get('Features_Mentioned', ''))
        
        row_key = str(row.get('id', index))

        rendered_prompt = jinja_template.render(
            id=row_key,
            input_text=text_content,
            features_mentioned=str(features),
            brands_mentioned=str(brands)
        )

        request_body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": rendered_prompt}]
                }
            ],
            "generation_config": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        }

        final_line_object = {
            "key": row_key,
            "request": request_body
        }

        jsonl_data.append(final_line_object)
    
    return jsonl_data

def save_jsonl(data, output_path, shard_size=None): 
    """
    Saves data to JSONL format, always sharding the output into 10 files.
    Returns a list of file paths that were created.
    """
    if not data:
        print("⚠️ No data to save.")
        return []

    num_rows = len(data)
    num_shards = 10
    shard_size = math.ceil(num_rows / num_shards) if num_rows > 0 else 0

    print(f"💾 Saving {num_rows} items into {num_shards} shard files...")
    
    created_files = []
    base_name, ext = os.path.splitext(output_path)

    for i in range(num_shards):
        # Slice the data for the current shard
        start_index = i * shard_size
        end_index = start_index + shard_size
        shard_data = data[start_index:end_index]
        
        shard_filename = f"{base_name}_part_{i+1:02d}{ext}"
        
        print(f"   Saving shard {i+1}/{num_shards} with {len(shard_data)} rows to {shard_filename}...")
        try:
            with open(shard_filename, 'w', encoding='utf-8') as f:
                for line in shard_data:
                    f.write(json.dumps(line) + '\n')
            created_files.append(shard_filename)
        except Exception as e:
            print(f"❌ Error saving shard {shard_filename}: {e}")
            raise
            
    print(f"✅ All {num_shards} shards saved successfully.")
    return created_files

