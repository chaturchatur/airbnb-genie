import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import joblib

def load_enrich(file_path='data/cleaned_listings.csv', rm_outliers=True):
    """load data, perform advanced feature engineering, and prepare for modeling"""
    if not os.path.exists(file_path):
        print(f"error: {file_path} not found")
        return None, None

    print(f"loading data from {file_path}...")
    data_df = pd.read_csv(file_path)
    
    target_col = 'price'
    
    if rm_outliers:
        limit_val = data_df[target_col].quantile(0.99)
        data_df = data_df[data_df[target_col] <= limit_val]
        print(f"removed outliers > ${limit_val:.2f} (99th percentile)")

    print("engineering features...")
    
    if 'name' in data_df.columns:
        data_df['name'] = data_df['name'].fillna('')
        data_df['name_length'] = data_df['name'].str.len()
        
        key_words = ['luxury', 'view', 'pool', 'gym', 'parking', 'free', 'private', 'historic', 'modern']
        for word_val in key_words:
            # create binary feature if keyword present in name
            data_df[f'name_has_{word_val}'] = data_df['name'].str.contains(word_val, case=False, regex=False).astype(int)
    
    if 'latitude' in data_df.columns and 'longitude' in data_df.columns:
        print("creating spatial clusters...")
        coords = data_df[['latitude', 'longitude']]
        kmeans_mod = KMeans(n_clusters=20, random_state=42, n_init=10)
        data_df['spatial_cluster'] = kmeans_mod.fit_predict(coords)
        
        # calculate distance to approx dc center
        dc_center = np.array([38.8977, -77.0365])
        data_df['dist_to_center'] = np.sqrt(
            (data_df['latitude'] - dc_center[0])**2 + (data_df['longitude'] - dc_center[1])**2
        )

    drop_list = ['id', 'scrape_id', 'name', 'description', 'picture_url', 'listing_url', 
                 'host_id', 'host_url', 'host_name', 'host_location', 'host_about', 
                 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 
                 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 
                 'calendar_updated', 'has_availability', 'calendar_last_scraped', 
                 'first_review', 'last_review', 'license', 'instant_bookable',
                 'bathrooms', 'amenities', 'bathrooms_text', 'neighbourhood_cleansed', 
                 'room_type', 'host_since', 'host_response_time', 'host_is_superhost']

    drop_cols = [col for col in drop_list if col in data_df.columns]
    model_df = data_df.drop(columns=drop_cols)
    
    model_df = model_df.select_dtypes(include=[np.number])
    
    model_df = model_df.dropna(subset=[target_col])
    model_df = model_df.fillna(0)
    
    print(f"data shape for modeling: {model_df.shape}")
    
    X_vals = model_df.drop(columns=[target_col])
    y_vals = model_df[target_col]
    
    return X_vals, y_vals

def train_tuned(X_vals, y_vals):
    """train a histgradientboostingregressor with randomizedsearchcv tuning"""
    X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)
    
    y_log = np.log1p(y_train)
    
    gb_model = HistGradientBoostingRegressor(random_state=42)
    
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [100, 200, 300, 500],
        'max_leaf_nodes': [31, 50, 100],
        'min_samples_leaf': [20, 50, 100],
        'l2_regularization': [0, 0.1, 1.0]
    }
    
    print("starting hyperparameter tuning (randomizedsearchcv)...")
    search_cv = RandomizedSearchCV(
        gb_model, 
        param_distributions=param_grid, 
        n_iter=20, 
        scoring='neg_mean_squared_error', 
        cv=5, 
        n_jobs=-1, 
        random_state=42,
        verbose=1
    )
    
    search_cv.fit(X_train, y_log)
    
    best_est = search_cv.best_estimator_
    print(f"best parameters: {search_cv.best_params_}")
    
    print("evaluating best model...")
    pred_log = best_est.predict(X_test)
    y_pred = np.expm1(pred_log)
    y_pred = np.maximum(y_pred, 0)
    
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_val = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    
    res_dict = {'rmse': rmse_val, 'mae': mae_val, 'r2': r2_val}
    
    return res_dict, best_est, X_test, y_test, y_pred

def save_plots(y_test, y_pred, model_name):
    """plot actual vs predicted"""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='perfect prediction')
    
    plt.xlabel('actual price')
    plt.ylabel('predicted price')
    plt.title(f'actual vs predicted prices ({model_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/actual_vs_predicted_{model_name.lower().replace(" ", "_")}.png')
    print(f"saved actual_vs_predicted_{model_name.lower().replace(' ', '_')}.png")
    plt.close()

def main():
    X_vals, y_vals = load_enrich()
    if X_vals is None:
        return

    res_dict, model_obj, X_test, y_test, y_pred = train_tuned(X_vals, y_vals)
    
    print("\nimproved model evaluation results (gradient boosting + tuning):")
    print("-" * 60)
    print(f"  rmse: ${res_dict['rmse']:.2f}")
    print(f"  mae:  ${res_dict['mae']:.2f}")
    print(f"  r2:   {res_dict['r2']:.4f}")
    print("-" * 60)
    
    save_plots(y_test, y_pred, 'gradient boosting tuned')
    
    joblib.dump(model_obj, 'models/baseline_model_v2.pkl')
    print("saved model to models/baseline_model_v2.pkl")

if __name__ == "__main__":
    main()
