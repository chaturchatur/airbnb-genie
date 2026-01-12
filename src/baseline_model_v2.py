import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from utils import load_enrich, save_plots

def train_tuned(X_vals, y_vals):
    """train a gradientboostingregressor with randomizedsearchcv tuning"""
    X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)
    
    y_log = np.log1p(y_train)
    
    gb_model = GradientBoostingRegressor(random_state=42)
    
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6],
        'min_samples_leaf': [1, 3, 5],
        'min_samples_split': [2, 5, 10]
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
