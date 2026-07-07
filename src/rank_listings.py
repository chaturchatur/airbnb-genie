"""
rank_listings.py

turns the existing price *predictor* into a fair-price *ranking engine*.

big picture (the analogy this mirrors):
    the trained model predicts an "expected" (fair) price for a listing given
    its features. we then score every real listing on actual-vs-predicted.
    that is exactly a "performance vs expectation" metric: predict the expected
    value, then measure how far reality sits above or below it. a listing priced
    well above its predicted fair value is "overpriced"; well below is
    "underpriced". sorting by that gap gives a ranking of best/worst value.

two things this script produces:
    1. over/underpriced flagging + a ranked value table (csv).
    2. sensitivity analysis:
        a) permutation feature importance -- how much each feature actually
           moves price (acts like implicit feature weights).
        b) threshold sensitivity -- does the over/underpriced flagging stay
           stable as we move the cutoff? (defensibility check.)

it reuses the exact same cleaning + feature pipeline and the saved model, so
the "fair price" here is the same number the rest of the repo would predict.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# we reuse the repo's own cleaning functions so features are built the SAME way
# they were built at training time. no new feature logic invented here.
from clean_data import clean_price, fill_missing, add_features


# --- named constants (easy to point at and defend in an interview) ---

# a listing is flagged once its actual price is more than this fraction above
# (overpriced) or below (underpriced) the model's predicted fair price.
# 0.15 = 15%. chosen as a round, defensible "meaningfully different" cutoff.
PRICE_DIFF_THRESHOLD = 0.15

# grid of thresholds we sweep in the sensitivity analysis to check stability.
THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25]

# same columns the training pipeline (clean_data.main) kept from the raw file.
KEEP_COLS = [
    'id', 'name', 'host_id', 'host_since', 'host_response_time', 'host_is_superhost',
    'neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms_text', 'bedrooms', 'beds',
    'amenities', 'price', 'minimum_nights', 'maximum_nights', 'number_of_reviews',
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
    'review_scores_value', 'reviews_per_month', 'latitude', 'longitude'
]

# approximate lat/lon of central washington dc (same value used in utils.py).
DC_CENTER = np.array([38.8977, -77.0365])

# keyword flags pulled from the listing name (same list as utils.py).
NAME_KEYWORDS = ['luxury', 'view', 'pool', 'gym', 'parking', 'free', 'private',
                 'historic', 'modern']


def load_model_and_artifacts(model_path='models/baseline_model_v2.pkl',
                             artifacts_path='models/artifacts.pkl'):
    """load the trained gradient boosting model and the feature artifacts."""
    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        print("error: model or artifacts not found. train the model first.")
        return None, None
    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    return model, artifacts


def build_features(raw_df, artifacts):
    """
    take the raw listings dataframe and produce:
      - meta_df: the human-readable columns we want in the final report
                 (id, neighbourhood, actual price).
      - X: a numeric feature matrix aligned EXACTLY to the columns the model
           was trained on (artifacts['feature_columns']).

    this mirrors clean_data.py + utils.py feature engineering, then aligns the
    result to the saved training column list the same way inference.py does.
    """
    # keep only the columns the training pipeline used (if present).
    exist_cols = [c for c in KEEP_COLS if c in raw_df.columns]
    df = raw_df[exist_cols].copy()

    # --- step 1: reuse the repo's cleaning + feature engineering functions ---
    # clean_price drops rows with no valid price. we need a real actual price
    # to compare against, so dropping those rows is correct here.
    df = clean_price(df)
    df = fill_missing(df)      # impute bedrooms/beds/bathrooms, review scores
    df = add_features(df)      # amenities count, has_wifi..., host encodings,
                               # and one-hot room_type / neighbourhood dummies

    # --- step 2: add the extra features utils.load_enrich creates ---
    # (name-based flags, spatial cluster, distance to city center)
    # note: add_features one-hot-encoded and dropped 'neighbourhood_cleansed',
    # so we recover the neighbourhood name from the raw frame by row index.
    df['name'] = raw_df.loc[df.index, 'name'].fillna('') if 'name' in raw_df.columns else ''
    df['name_length'] = df['name'].str.len()
    for word in NAME_KEYWORDS:
        df[f'name_has_{word}'] = df['name'].str.contains(word, case=False, regex=False).astype(int)

    # spatial cluster comes from the SAME kmeans that was fit at training time
    # (loaded from artifacts), so cluster ids line up with what the model saw.
    kmeans = artifacts.get('kmeans_model')
    if kmeans is not None and 'latitude' in df.columns and 'longitude' in df.columns:
        coords = df[['latitude', 'longitude']]
        df['spatial_cluster'] = kmeans.predict(coords)
        df['dist_to_center'] = np.sqrt(
            (df['latitude'] - DC_CENTER[0]) ** 2 + (df['longitude'] - DC_CENTER[1]) ** 2
        )
    else:
        df['spatial_cluster'] = 0
        df['dist_to_center'] = 0

    # --- step 3: pull out the human-readable info before we go numeric-only ---
    # recover the neighbourhood name from the raw frame (add_features dropped it).
    neighbourhood = raw_df.loc[df.index, 'neighbourhood_cleansed'] \
        if 'neighbourhood_cleansed' in raw_df.columns else pd.Series('', index=df.index)
    meta_df = pd.DataFrame({
        'id': raw_df.loc[df.index, 'id'].values if 'id' in raw_df.columns else df.index,
        'neighbourhood': neighbourhood.values,
        'actual': df['price'].values,
    }, index=df.index)

    # boolean dummy columns -> int, to match how training features were stored.
    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # --- step 4: align to the exact training feature columns ---
    # this is the same trick inference.py uses: start from an all-zero frame
    # with the training columns, then copy over whatever columns we actually
    # have. any training column we don't have stays 0; any extra column we have
    # (e.g. a neighbourhood that wasn't in training) is simply ignored.
    feature_columns = artifacts['feature_columns']
    X = pd.DataFrame(0, index=df.index, columns=feature_columns)
    common = [c for c in df.columns if c in feature_columns]
    X[common] = df[common]

    # match training: utils.load_enrich did fillna(0) on the feature matrix, so
    # any leftover missing values (e.g. reviews_per_month with no reviews) become
    # 0 here too. GradientBoostingRegressor cannot accept NaN.
    X = X.fillna(0)

    return meta_df, X


def predict_fair_price(model, X):
    """
    predict the fair price for each listing.
    the model was trained on log1p(price), so we invert with expm1 to get back
    to real dollars, and clip at 0 (a price can't be negative).
    """
    log_pred = model.predict(X)
    pred = np.expm1(log_pred)
    return np.maximum(pred, 0)


def flag_listings(meta_df, predicted, threshold=PRICE_DIFF_THRESHOLD):
    """
    build the value table: actual vs predicted, the percentage gap, and a flag.

    residual  = actual - predicted        (dollars over/under fair value)
    pct_diff  = residual / predicted       (that gap as a fraction of fair value)

    flag:
      'overpriced'  if pct_diff >  +threshold  (charging notably above fair)
      'underpriced' if pct_diff <  -threshold  (a notably good deal)
      'fair'        otherwise

    this is the "performance vs expectation" score: predicted is the expectation,
    actual is the performance, pct_diff is how far off expectation reality is.
    """
    out = meta_df.copy()
    out['predicted'] = predicted
    out['residual'] = out['actual'] - out['predicted']

    # guard against divide-by-zero if a predicted price is 0.
    safe_pred = out['predicted'].replace(0, np.nan)
    out['pct_diff'] = out['residual'] / safe_pred
    out = out.dropna(subset=['pct_diff'])

    def label(p):
        if p > threshold:
            return 'overpriced'
        if p < -threshold:
            return 'underpriced'
        return 'fair'

    out['flag'] = out['pct_diff'].apply(label)

    # rank by pct_diff: most overpriced at the top, best deals at the bottom.
    out = out.sort_values('pct_diff', ascending=False).reset_index(drop=True)
    return out[['id', 'neighbourhood', 'actual', 'predicted', 'pct_diff', 'flag']]


def run_permutation_importance(model, X, actual_prices):
    """
    sensitivity part (a): permutation feature importance.

    idea in plain english: shuffle one feature's values across rows so it becomes
    noise, then see how much worse the model scores. a big drop means that
    feature really mattered for price; a tiny drop means it barely did. this is
    an honest, model-agnostic way to read "how amenities / location / size relate
    to price" -- effectively the model's implicit feature weights.

    we score against log1p(actual) because the model predicts in log space
    (that's how it was trained and evaluated), so this measures it fairly.
    we run it on a held-out slice (same 80/20 split + seed as training) so the
    importances aren't just memorized from data the model already fit.
    """
    y_log = np.log1p(actual_prices)
    # recreate the training split so we evaluate on unseen rows.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print("running permutation importance on held-out set (this takes a bit)...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,        # shuffle each feature 10x to average out luck
        random_state=42,
        n_jobs=-1,
    )

    imp = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return imp


def run_threshold_sensitivity(meta_df, predicted, grid=THRESHOLD_GRID):
    """
    sensitivity part (b): threshold sensitivity.

    we vary the over/underpriced cutoff across a small grid and report:
      - how many listings get flagged over/under at each cutoff, and
      - how stable the *top 20* over/underpriced listings are as the cutoff moves
        (overlap vs the 0.15 baseline).

    if the same listings keep showing up as the cutoff wiggles, the flagging is
    robust and defensible -- it isn't an artifact of one arbitrary number.
    returns (summary_df, printable_notes).
    """
    # baseline ranking (independent of threshold; sort order never changes,
    # only the labels do). compute once.
    base = flag_listings(meta_df, predicted, threshold=PRICE_DIFF_THRESHOLD)
    base_top_over = set(base.head(20)['id'])
    base_top_under = set(base.tail(20)['id'])

    rows = []
    for t in grid:
        tbl = flag_listings(meta_df, predicted, threshold=t)
        n_over = int((tbl['flag'] == 'overpriced').sum())
        n_under = int((tbl['flag'] == 'underpriced').sum())
        n_fair = int((tbl['flag'] == 'fair').sum())

        # top-20 sets don't depend on the threshold (ranking is by pct_diff),
        # so overlap with baseline top-20 is always 20/20. we still report the
        # count overlap to make the stability explicit and honest.
        top_over = set(tbl.head(20)['id'])
        top_under = set(tbl.tail(20)['id'])
        over_overlap = len(top_over & base_top_over)
        under_overlap = len(top_under & base_top_under)

        rows.append({
            'threshold': t,
            'n_overpriced': n_over,
            'n_underpriced': n_under,
            'n_fair': n_fair,
            'top20_over_overlap_vs_0.15': over_overlap,
            'top20_under_overlap_vs_0.15': under_overlap,
        })

    summary = pd.DataFrame(rows)
    return summary


def main():
    print("=" * 60)
    print("fair-price ranking engine")
    print("=" * 60)

    # 1. load model + artifacts and the raw listings.
    model, artifacts = load_model_and_artifacts()
    if model is None:
        return

    raw_path = 'data/raw/listings.csv'
    if not os.path.exists(raw_path):
        print(f"error: {raw_path} not found.")
        return
    print(f"loading {raw_path}...")
    raw_df = pd.read_csv(raw_path)
    print(f"raw listings: {len(raw_df)} rows")

    # 2. build features exactly like training, then predict the fair price.
    meta_df, X = build_features(raw_df, artifacts)
    print(f"scoring {len(X)} listings with {X.shape[1]} features...")
    predicted = predict_fair_price(model, X)

    # 3. flag + rank, and save the value ranking.
    ranking = flag_listings(meta_df, predicted, threshold=PRICE_DIFF_THRESHOLD)

    os.makedirs('data/processed', exist_ok=True)
    ranking_path = 'data/processed/listing_value_ranking.csv'
    ranking.to_csv(ranking_path, index=False)
    print(f"saved value ranking to {ranking_path}")

    counts = ranking['flag'].value_counts()
    print("\nflag counts (threshold = {:.0%}):".format(PRICE_DIFF_THRESHOLD))
    for flag_name in ['overpriced', 'fair', 'underpriced']:
        print(f"  {flag_name:12s}: {int(counts.get(flag_name, 0))}")

    print("\ntop 5 most OVERPRICED (charging above fair value):")
    print(ranking.head(5).to_string(index=False))
    print("\ntop 5 most UNDERPRICED (best deals vs fair value):")
    print(ranking.tail(5).to_string(index=False))

    # 4. sensitivity (a): permutation feature importance.
    imp = run_permutation_importance(model, X, meta_df['actual'].values)
    imp_path = 'data/processed/feature_sensitivity.csv'
    imp.to_csv(imp_path, index=False)
    print(f"\nsaved feature sensitivity to {imp_path}")
    print("top 10 features by how much they move price:")
    print(imp.head(10).to_string(index=False))

    # 5. sensitivity (b): threshold sensitivity.
    print("\nthreshold sensitivity (does flagging stay stable?):")
    summary = run_threshold_sensitivity(meta_df, predicted)
    print(summary.to_string(index=False))
    print(
        "\nreading it: the top-20 over/underpriced sets are identical across all\n"
        "thresholds (overlap 20/20) because ranking is by pct_diff, not the cutoff.\n"
        "only the fair/over/under *counts* shift as the cutoff moves. that means\n"
        "the 'who is most over/underpriced' answer is robust; the threshold only\n"
        "controls how many borderline listings get a label."
    )

    print("\ndone.")


if __name__ == "__main__":
    main()
