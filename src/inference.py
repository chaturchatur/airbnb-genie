import pandas as pd
import numpy as np
import joblib
import re
import os

def load_model_and_artifacts():
    """load the trained model and feature engineering artifacts"""
    try:
        model_path = 'models/baseline_model_v2.pkl'
        artifacts_path = 'models/artifacts.pkl'
        
        if not os.path.exists(model_path):
            print(f"error: {model_path} not found")
            return None, None
            
        if not os.path.exists(artifacts_path):
            print(f"error: {artifacts_path} not found")
            return None, None
            
        model = joblib.load(model_path)
        artifacts = joblib.load(artifacts_path)
        return model, artifacts
    except Exception as e:
        print(f"error loading files: {e}")
        return None, None

def parse_bath_val(text_val):
    """parse bathroom text to float"""
    if pd.isna(text_val):
        return np.nan
    text_str = str(text_val).lower()
    if 'half-bath' in text_str or 'half bath' in text_str:
        return 0.5
    match_val = re.search(r'(\d+(\.\d+)?)', text_str)
    if match_val:
        return float(match_val.group(1))
    return np.nan

def predict_price(input_data, model, artifacts):
    """
    predict price for a single listing
    input_data: dict containing listing details
    """
    # convert input dict to dataframe
    df = pd.DataFrame([input_data])
    
    group_medians = artifacts['group_medians']
    global_medians = artifacts['global_medians']
    
    # --- fill missing values ---
    
    # helper to look up median from multi-index series
    def get_median(row, col):
        val = row.get(col)
        if pd.notna(val) and val != '':
            return float(val)
        try:
            # lookup by (neighbourhood, room_type)
            nb = row.get('neighbourhood_cleansed')
            rt = row.get('room_type')
            return group_medians.loc[(nb, rt), col]
        except KeyError:
            return global_medians[col]

    # ensure columns exist
    for col in ['bedrooms', 'beds']:
        if col not in df.columns:
            df[col] = np.nan

    df['bedrooms'] = df.apply(lambda r: get_median(r, 'bedrooms'), axis=1)
    df['beds'] = df.apply(lambda r: get_median(r, 'beds'), axis=1)
    
    # bathrooms
    if 'bathrooms_text' in df.columns:
        df['bathrooms_cleaned'] = df['bathrooms_text'].apply(parse_bath_val)
        df['bathrooms_cleaned'] = df['bathrooms_cleaned'].fillna(global_medians['bathrooms_cleaned'])
    else:
        df['bathrooms_cleaned'] = global_medians['bathrooms_cleaned']

    # --- feature engineering ---
    
    # amenities (expecting list or None)
    amenities = input_data.get('amenities', [])
    if not isinstance(amenities, list):
        amenities = []
        
    df['amenities_count'] = len(amenities)
    
    common_items = ['Wifi', 'Air conditioning', 'Kitchen', 'Washer', 'Dryer', 'Heating']
    for item in common_items:
        col_name = f'has_{item.lower().replace(" ", "_")}'
        df[col_name] = 1 if item in amenities else 0

    # host response
    time_map = {
        'within an hour': 4, 'within a few hours': 3,
        'within a day': 2, 'a few days or more': 1
    }
    resp_time = input_data.get('host_response_time')
    df['host_response_time_encoded'] = time_map.get(resp_time, 0)
    
    # superhost
    is_superhost = input_data.get('host_is_superhost')
    df['host_is_superhost_encoded'] = 1 if is_superhost in [True, 't', 'True'] else 0

    # name features
    name_val = str(input_data.get('name', ''))
    df['name'] = name_val
    df['name_length'] = len(name_val)
    
    key_words = ['luxury', 'view', 'pool', 'gym', 'parking', 'free', 'private', 'historic', 'modern']
    for word in key_words:
        df[f'name_has_{word}'] = 1 if word.lower() in name_val.lower() else 0

    # spatial clusters
    kmeans_mod = artifacts.get('kmeans_model')
    if kmeans_mod and 'latitude' in df.columns and 'longitude' in df.columns:
        try:
            lat = float(df['latitude'].iloc[0])
            lon = float(df['longitude'].iloc[0])
            df['spatial_cluster'] = kmeans_mod.predict([[lat, lon]])[0]
            
            dc_center = np.array([38.8977, -77.0365])
            df['dist_to_center'] = np.sqrt((lat - dc_center[0])**2 + (lon - dc_center[1])**2)
        except (ValueError, TypeError):
             df['spatial_cluster'] = 0
             df['dist_to_center'] = 0
    else:
        df['spatial_cluster'] = 0
        df['dist_to_center'] = 0

    # --- encoding and alignment ---
    
    # one-hot encode
    df_encoded = pd.get_dummies(df, columns=['room_type', 'neighbourhood_cleansed'], prefix=['room', 'nb'])
    
    # align to training columns
    feature_columns = artifacts['feature_columns']
    
    # create empty dataframe with all feature columns
    X_final = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # update with matching columns from our processed input
    common_cols = [c for c in df_encoded.columns if c in feature_columns]
    X_final[common_cols] = df_encoded[common_cols]
    
    # --- predict ---
    # model was trained on log(price + 1)
    log_pred = model.predict(X_final)
    pred_price = np.expm1(log_pred)
    
    return max(0, float(pred_price[0]))

