import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def load_prep(file_path='data/cleaned_listings.csv', rm_outliers=True):
    """load data and prepare features for modeling"""
    if not os.path.exists(file_path):
        print(f"error: {file_path} not found")
        return None, None

    print(f"loading data from {file_path}...")
    data_df = pd.read_csv(file_path)
    
    target_col = 'price'
    
    if rm_outliers:
        start_rows = len(data_df)
        limit_val = data_df[target_col].quantile(0.99)
        data_df = data_df[data_df[target_col] <= limit_val]
        print(f"removed outliers > ${limit_val:.2f} (99th percentile)")
        print(f"rows dropped: {start_rows - len(data_df)}")

    drop_list = ['id', 'scrape_id', 'name', 'description', 'picture_url', 'listing_url', 
                 'host_id', 'host_url', 'host_name', 'host_location', 'host_about', 
                 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 
                 'host_verifications', 'neighbourhood', 'neighbourhood_group_cleansed', 
                 'calendar_updated', 'has_availability', 'calendar_last_scraped', 
                 'first_review', 'last_review', 'license', 'instant_bookable',
                 'bathrooms', 
                 'amenities', 'bathrooms_text', 'neighbourhood_cleansed', 'room_type', 'host_since', 'host_response_time', 'host_is_superhost']

    drop_cols = [col for col in drop_list if col in data_df.columns]
    model_df = data_df.drop(columns=drop_cols)
    
    model_df = model_df.select_dtypes(include=[np.number])
    
    model_df = model_df.dropna(subset=[target_col])
    model_df = model_df.fillna(0)
    
    print(f"data shape for modeling: {model_df.shape}")
    
    X_vals = model_df.drop(columns=[target_col])
    y_vals = model_df[target_col]
    
    return X_vals, y_vals

def train_eval(X_vals, y_vals, log_target=True):
    """train linear regression and random forest models and evaluate them"""
    print("splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)
    
    if log_target:
        print("applying log-transformation to target variable...")
        y_train_tf = np.log1p(y_train)
    else:
        y_train_tf = y_train
        
    res_dict = {}
    mod_dict = {}
    
    print("training linear regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_tf)
    
    lr_raw = lr_model.predict(X_test)
    
    if log_target:
        lr_pred = np.expm1(lr_raw)
        # handle negative predictions
        lr_pred = np.maximum(lr_pred, 0)
    else:
        lr_pred = lr_raw
        
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr_pred))
    mae_lr = mean_absolute_error(y_test, lr_pred)
    r2_lr = r2_score(y_test, lr_pred)
    
    res_dict['linear regression'] = {'rmse': rmse_lr, 'mae': mae_lr, 'r2': r2_lr}
    mod_dict['linear regression'] = lr_model
    
    print("training random forest regressor (this may take a moment)...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train_tf)
    
    rf_raw = rf_model.predict(X_test)
    
    if log_target:
        rf_pred = np.expm1(rf_raw)
    else:
        rf_pred = rf_raw
        
    rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
    mae_rf = mean_absolute_error(y_test, rf_pred)
    r2_rf = r2_score(y_test, rf_pred)
    
    res_dict['random forest'] = {'rmse': rmse_rf, 'mae': mae_rf, 'r2': r2_rf}
    mod_dict['random forest'] = rf_model
    
    return res_dict, mod_dict, X_test, y_test, lr_pred, rf_pred

def save_plots(y_test, y_pred, model_name, feat_names=None, model_obj=None):
    """generate and save evaluation plots"""
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
    
    if hasattr(model_obj, 'feature_importances_') and feat_names is not None:
        imp_vals = model_obj.feature_importances_
        idx_sort = np.argsort(imp_vals)[::-1]
        top_n = 20
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=imp_vals[idx_sort[:top_n]], y=[feat_names[i] for i in idx_sort[:top_n]])
        plt.title(f'top 20 feature importances ({model_name})')
        plt.xlabel('importance')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        print("saved feature_importance.png")
        plt.close()

def main():
    X_vals, y_vals = load_prep(rm_outliers=True)
    if X_vals is None:
        return

    res_dict, mod_dict, X_test, y_test, lr_pred, rf_pred = train_eval(X_vals, y_vals, log_target=True)
    
    print("\nmodel evaluation results (after outlier removal & log transform):")
    print("-" * 60)
    for name_val, metrics in res_dict.items():
        print(f"{name_val}:")
        print(f"  rmse: ${metrics['rmse']:.2f}")
        print(f"  mae:  ${metrics['mae']:.2f}")
        print(f"  r2:   {metrics['r2']:.4f}")
        print("-" * 60)
        
    save_plots(y_test, rf_pred, 'random forest', X_vals.columns, mod_dict['random forest'])
    save_plots(y_test, lr_pred, 'linear regression')
    
    print("\nbaseline modeling complete")

if __name__ == "__main__":
    main()
