import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from autosklearn.regression import AutoSklearnRegressor
from utils import load_enrich, save_plots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from autosklearn.regression import AutoSklearnRegressor
from utils import load_enrich, save_plots

def train_autosklearn(X_vals, y_vals):
    """train model using auto-sklearn"""
    print("splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)
    
    # log transform target
    y_log_train = np.log1p(y_train)
    
    print("configuring auto-sklearn...")
    automl = AutoSklearnRegressor(
        time_left_for_this_task=600,
        per_run_time_limit=60,
        n_jobs=-1,
        seed=42,
        memory_limit=None
    )
    
    print("training auto-sklearn model (this may take up to 10 minutes)...")
    automl.fit(X_train, y_log_train)
    
    print("evaluating model...")
    pred_log = automl.predict(X_test)
    y_pred = np.expm1(pred_log)
    y_pred = np.maximum(y_pred, 0)
    
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_val = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    
    res_dict = {'rmse': rmse_val, 'mae': mae_val, 'r2': r2_val}
    
    # print leaderboard
    print("\nauto-sklearn leaderboard:")
    print(automl.leaderboard())
    
    # print statistics
    print("\nauto-sklearn sprint statistics:")
    print(automl.sprint_statistics())
    
    return res_dict, automl, X_test, y_test, y_pred

def main():
    X_vals, y_vals = load_enrich()
    if X_vals is None:
        return

    res_dict, model_obj, X_test, y_test, y_pred = train_autosklearn(X_vals, y_vals)
    
    print("\nauto-sklearn model evaluation results:")
    print("-" * 60)
    print(f"  rmse: ${res_dict['rmse']:.2f}")
    print(f"  mae:  ${res_dict['mae']:.2f}")
    print(f"  r2:   {res_dict['r2']:.4f}")
    print("-" * 60)
    
    save_plots(y_test, y_pred, 'autosklearn')
    
    joblib.dump(model_obj, 'models/autosklearn_model.pkl')
    print("saved model to models/autosklearn_model.pkl")

if __name__ == "__main__":
    main()

