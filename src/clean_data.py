import pandas as pd
import numpy as np
import os
import re
import ast

def load_data():
    """load listings and reviews data"""
    listings_path = 'data/raw/listings.csv'
    reviews_path = 'data/raw/reviews.csv'
    
    if not os.path.exists(listings_path) or not os.path.exists(reviews_path):
        print("error: data files not found")
        return None, None

    print(f"loading {listings_path}...")
    listings_df = pd.read_csv(listings_path)
    
    print(f"loading {reviews_path}...")
    reviews_df = pd.read_csv(reviews_path)
    
    return listings_df, reviews_df

def clean_price(data_frame):
    """clean price column by removing '$' and ',' and converting to float"""
    print("cleaning price column...")
    
    data_frame['price'] = data_frame['price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    data_frame['price'] = pd.to_numeric(data_frame['price'], errors='coerce')
    
    initial_rows = len(data_frame)
    data_frame = data_frame.dropna(subset=['price'])
    dropped_rows = initial_rows - len(data_frame)
    print(f"dropped {dropped_rows} rows with missing or invalid prices")
    
    return data_frame

def fill_missing(data_frame):
    """impute missing values for bedrooms, beds, bathrooms, and review scores"""
    print("imputing missing values...")
    
    # group by neighbourhood and room type to calculate median bedrooms and beds
    group_medians = data_frame.groupby(['neighbourhood_cleansed', 'room_type'])[['bedrooms', 'beds']].transform('median')
    
    data_frame['bedrooms'] = data_frame['bedrooms'].fillna(group_medians['bedrooms'])
    data_frame['beds'] = data_frame['beds'].fillna(group_medians['beds'])
    
    # fallback to global median
    data_frame['bedrooms'] = data_frame['bedrooms'].fillna(data_frame['bedrooms'].median())
    data_frame['beds'] = data_frame['beds'].fillna(data_frame['beds'].median())

    def parse_bath(text_val):
        if pd.isna(text_val):
            return np.nan
        text_str = str(text_val).lower()
        if 'half-bath' in text_str or 'half bath' in text_str:
            return 0.5
        match_val = re.search(r'(\d+(\.\d+)?)', text_str)
        if match_val:
            return float(match_val.group(1))
        return np.nan

    data_frame['bathrooms_cleaned'] = data_frame['bathrooms_text'].apply(parse_bath)
    data_frame['bathrooms_cleaned'] = data_frame['bathrooms_cleaned'].fillna(data_frame['bathrooms_cleaned'].median())

    data_frame['has_reviews'] = np.where(data_frame['number_of_reviews'] > 0, 1, 0)
    
    score_cols = [col_name for col_name in data_frame.columns if 'review_scores' in col_name]
    for col_name in score_cols:
        data_frame[col_name] = data_frame[col_name].fillna(0)
        
    return data_frame

def add_features(data_frame):
    """perform feature engineering on amenities, host, and categorical variables"""
    print("performing feature engineering...")
    
    def count_items(item_str):
        try:
            item_list = ast.literal_eval(item_str)
            return len(item_list)
        except (ValueError, SyntaxError):
            return 0

    data_frame['amenities_count'] = data_frame['amenities'].apply(count_items)
    
    common_items = ['Wifi', 'Air conditioning', 'Kitchen', 'Washer', 'Dryer', 'Heating']
    for item_val in common_items:
        col_name = f'has_{item_val.lower().replace(" ", "_")}'
        data_frame[col_name] = data_frame['amenities'].str.contains(item_val, case=False, regex=False).astype(int)

    time_map = {
        'within an hour': 4,
        'within a few hours': 3,
        'within a day': 2,
        'a few days or more': 1,
        pd.NA: 0,
        np.nan: 0
    }
    data_frame['host_response_time_encoded'] = data_frame['host_response_time'].map(time_map).fillna(0)
    data_frame['host_is_superhost_encoded'] = data_frame['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)

    # one-hot encode categorical variables
    data_frame = pd.get_dummies(data_frame, columns=['room_type', 'neighbourhood_cleansed'], prefix=['room', 'nb'], dummy_na=False)

    return data_frame

def agg_reviews(reviews_df):
    """aggregate reviews by listing and month"""
    print("aggregating reviews...")
    
    reviews_df['date'] = pd.to_datetime(reviews_df['date'])
    reviews_df['month'] = reviews_df['date'].dt.to_period('M')
    
    monthly_revs = reviews_df.groupby(['listing_id', 'month']).size().reset_index(name='monthly_review_count')
    monthly_revs['month'] = monthly_revs['month'].astype(str)
    
    return monthly_revs

def main():
    listings_df, reviews_df = load_data()
    
    if listings_df is None:
        return

    print(f"original listings shape: {listings_df.shape}")

    keep_cols = [
        'id', 'name', 'host_id', 'host_since', 'host_response_time', 'host_is_superhost',
        'neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds',
        'amenities', 'price', 'minimum_nights', 'maximum_nights', 'number_of_reviews',
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'reviews_per_month', 'latitude', 'longitude' 
    ]
    
    exist_cols = [col_name for col_name in keep_cols if col_name in listings_df.columns]
    listings_df = listings_df[exist_cols]
    
    listings_df = clean_price(listings_df)
    listings_df = fill_missing(listings_df)
    listings_df = add_features(listings_df)
    
    print(f"cleaned listings shape: {listings_df.shape}")
    
    crit_cols = ['price', 'bedrooms', 'beds', 'bathrooms_cleaned']
    miss_vals = listings_df[crit_cols].isnull().sum()
    print("missing values in critical columns after cleaning:")
    print(miss_vals)

    if reviews_df is not None:
        proc_reviews = agg_reviews(reviews_df)
        proc_path = 'data/processed_reviews.csv'
        proc_reviews.to_csv(proc_path, index=False)
        print(f"saved processed reviews to {proc_path}")

    clean_path = 'data/cleaned_listings.csv'
    listings_df.to_csv(clean_path, index=False)
    print(f"saved cleaned listings to {clean_path}")
    
    print("data cleaning and preprocessing complete")

if __name__ == "__main__":
    main()
