import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.cluster import KMeans
from clean_data import clean_price

def parse_bath_val(text_val):
    """helper to parse bathroom text"""
    if pd.isna(text_val):
        return np.nan
    text_str = str(text_val).lower()
    if 'half-bath' in text_str or 'half bath' in text_str:
        return 0.5
    match_val = re.search(r'(\d+(\.\d+)?)', text_str)
    if match_val:
        return float(match_val.group(1))
    return np.nan

def generate_artifacts():
    """generate and save artifacts needed for inference"""
    print("generating artifacts...")
    os.makedirs('models', exist_ok=True)
    
    # --- part 1: imputation statistics from raw data ---
    listings_path = 'data/raw/listings.csv'
    if not os.path.exists(listings_path):
        print("error: data/raw/listings.csv not found")
        return

    print(f"loading {listings_path} for imputation stats...")
    raw_df = pd.read_csv(listings_path)
    
    # clean price to filter invalid rows (consistent with training)
    raw_df = clean_price(raw_df)
    
    # calculate group medians for bedrooms/beds
    print("calculating group medians...")
    group_medians = raw_df.groupby(['neighbourhood_cleansed', 'room_type'])[['bedrooms', 'beds']].median()
    
    # calculate global medians
    raw_df['bathrooms_cleaned'] = raw_df['bathrooms_text'].apply(parse_bath_val)
    global_medians = {
        'bedrooms': raw_df['bedrooms'].median(),
        'beds': raw_df['beds'].median(),
        'bathrooms_cleaned': raw_df['bathrooms_cleaned'].median()
    }

    # --- part 2: model assets from cleaned data ---
    clean_path = 'data/cleaned_listings.csv'
    if not os.path.exists(clean_path):
        print("error: data/cleaned_listings.csv not found. run clean_data.py first.")
        return

    print(f"loading {clean_path} for model assets...")
    clean_df = pd.read_csv(clean_path)
    
    # filter outliers (consistent with utils.load_enrich)
    limit_val = clean_df['price'].quantile(0.99)
    clean_df = clean_df[clean_df['price'] <= limit_val]
    
    # fit kmeans
    if 'latitude' in clean_df.columns and 'longitude' in clean_df.columns:
        print("fitting kmeans spatial clusters...")
        coords = clean_df[['latitude', 'longitude']]
        kmeans_mod = KMeans(n_clusters=20, random_state=42, n_init=10)
        kmeans_mod.fit(coords)
    else:
        kmeans_mod = None
        print("warning: lat/lon not found, skipping kmeans")

    # get final feature columns
    # reproduce utils.load_enrich feature engineering steps that add columns
    # note: cleaned_listings.csv already has most features, 
    # but utils.py adds 'name_length', 'name_has_X', 'spatial_cluster', 'dist_to_center'
    
    print("generating final feature list...")
    # apply dynamic features to get the full column list
    if 'name' in clean_df.columns:
        clean_df['name'] = clean_df['name'].fillna('')
        clean_df['name_length'] = clean_df['name'].str.len()
        key_words = ['luxury', 'view', 'pool', 'gym', 'parking', 'free', 'private', 'historic', 'modern']
        for word_val in key_words:
            clean_df[f'name_has_{word_val}'] = clean_df['name'].str.contains(word_val, case=False, regex=False).astype(int)
            
    if kmeans_mod is not None:
        clean_df['spatial_cluster'] = kmeans_mod.predict(clean_df[['latitude', 'longitude']])
        dc_center = np.array([38.8977, -77.0365])
        clean_df['dist_to_center'] = np.sqrt(
            (clean_df['latitude'] - dc_center[0])**2 + (clean_df['longitude'] - dc_center[1])**2
        )

    # drop columns to get X
    drop_list = ['id', 'scrape_id', 'name', 'description', 'picture_url', 'listing_url', 
                 'host_id', 'host_url', 'host_name', 'host_location', 'host_about', 
                 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 
                 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 
                 'calendar_updated', 'has_availability', 'calendar_last_scraped', 
                 'first_review', 'last_review', 'license', 'instant_bookable',
                 'bathrooms', 'amenities', 'bathrooms_text', 'neighbourhood_cleansed', 
                 'room_type', 'host_since', 'host_response_time', 'host_is_superhost',
                 'price'] # also drop target
                 
    drop_cols = [col for col in drop_list if col in clean_df.columns]
    X = clean_df.drop(columns=drop_cols)
    
    # ensure boolean columns (from get_dummies) are int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"converting {len(bool_cols)} boolean columns to int...")
        X[bool_cols] = X[bool_cols].astype(int)
    
    X = X.select_dtypes(include=[np.number])
    feature_columns = X.columns.tolist()
    
    print(f"captured {len(feature_columns)} feature columns")

    # --- save artifacts ---
    artifacts = {
        'group_medians': group_medians,
        'global_medians': global_medians,
        'kmeans_model': kmeans_mod,
        'feature_columns': feature_columns
    }
    
    out_path = 'models/artifacts.pkl'
    joblib.dump(artifacts, out_path)
    print(f"saved artifacts to {out_path}")

if __name__ == "__main__":
    generate_artifacts()

