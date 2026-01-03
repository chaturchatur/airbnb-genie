import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_dir = "data"
results_dir = "results"
listings_file = "cleaned_listings.csv"

def load_data():
    """load the cleaned listings data"""
    print("--- loading data ---")
    try:
        listings_path = os.path.join(data_dir, listings_file)
        listings_df = pd.read_csv(listings_path)
        print(f"loaded {listings_file} with {len(listings_df)} rows")
        return listings_df
    except Exception as error_msg:
        print(f"error loading {listings_file}: {error_msg}")
        return pd.DataFrame()

def analyze_data(listings_df):
    """perform general analysis on the listings"""
    print("\n--- analyzing listings ---")
    if listings_df.empty:
        print("listings data is empty")
        return

    print(f"shape: {listings_df.shape}")
    
    if 'price' in listings_df.columns:
        print("price distribution (describe):")
        print(listings_df['price'].describe())
    else:
        print("no 'price' column found")

    print("\nmissing values count:")
    print(listings_df.isnull().sum()[listings_df.isnull().sum() > 0])

def analyze_corr(listings_df, filter_out=True, limit_val=1000):
    """analyze and visualize correlations between numerical features and price"""
    print("\n--- analyzing correlations ---")
    
    num_cols = listings_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if filter_out and 'price' in listings_df.columns:
        print(f"filtering outliers (price > ${limit_val})...")
        filtered_df = listings_df[listings_df['price'] <= limit_val]
        print(f"rows after filtering: {len(filtered_df)} (dropped {len(listings_df) - len(filtered_df)})")
    else:
        filtered_df = listings_df

    corr_mat = filtered_df[num_cols].corr()
    
    if 'price' in corr_mat.index:
        print("\ntop correlations with price:")
        price_rel = corr_mat['price'].sort_values(ascending=False)
        print(price_rel.head(10))
        print(price_rel.tail(5))
        
        plt.figure(figsize=(12, 10))
        top_feats = price_rel.abs().nlargest(15).index
        sns.heatmap(listings_df[top_feats].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('correlation heatmap (top features correlated with price)')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
        print("saved correlation_heatmap.png")
        plt.close()

def analyze_segments(listings_df, limit_val=1000):
    """analyze market segments (standard vs luxury)"""
    print(f"\n--- market segmentation analysis (threshold: ${limit_val}) ---")
    
    if 'price' not in listings_df.columns:
        print("price column missing")
        return

    std_mkt = listings_df[listings_df['price'] <= limit_val]
    lux_mkt = listings_df[listings_df['price'] > limit_val]
    
    print(f"standard listings: {len(std_mkt)} rows")
    print(f"luxury listings:   {len(lux_mkt)} rows")
    
    mkt_segs = {'standard': std_mkt, 'luxury': lux_mkt}
    
    for name_val, data_val in mkt_segs.items():
        if data_val.empty:
            print(f"\nno data for {name_val} segment")
            continue
            
        print(f"\n--- {name_val} market analysis ---")
        print(f"average price: ${data_val['price'].mean():.2f}")
        
        num_cols = data_val.select_dtypes(include=[np.number]).columns.tolist()
        corr_mat = data_val[num_cols].corr()
        
        if 'price' in corr_mat.index:
            print(f"top price drivers for {name_val}:")
            print(corr_mat['price'].sort_values(ascending=False).head(5))

def plot_dist(listings_df, limit_val=1000):
    """visualize distributions of key features (filtered to standard market)"""
    print("\n--- visualizing distributions (standard market) ---")
    
    if 'price' in listings_df.columns:
        listings_df = listings_df[listings_df['price'] <= limit_val]

    if 'amenities_count' in listings_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(listings_df['amenities_count'], kde=True, bins=30)
        plt.title('distribution of amenities count (standard market)')
        plt.xlabel('number of amenities')
        plt.ylabel('count')
        plt.savefig(os.path.join(results_dir, 'amenities_count_dist.png'))
        print("saved amenities_count_dist.png")
        plt.close()

    if 'host_resp_enc' in listings_df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='host_resp_enc', data=listings_df)
        plt.title('distribution of host response time (encoded)')
        plt.xlabel('response time (encoded)')
        plt.ylabel('count')
        plt.savefig(os.path.join(results_dir, 'host_response_time_dist.png'))
        print("saved host_response_time_dist.png")
        plt.close()
        
    if 'price' in listings_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(np.log1p(listings_df['price']), kde=True, bins=50)
        plt.title('log-transformed price distribution (standard market)')
        plt.xlabel('log(price + 1)')
        plt.ylabel('count')
        plt.savefig(os.path.join(results_dir, 'price_log_dist.png'))
        print("saved price_log_dist.png")
        plt.close()

def plot_map(listings_df, limit_val=1000):
    """analyze spatial distribution of prices (filtered to standard market)"""
    print("\n--- spatial analysis (standard market) ---")
    
    if 'latitude' in listings_df.columns and 'longitude' in listings_df.columns and 'price' in listings_df.columns:
        plt.figure(figsize=(12, 8))
        
        subset_df = listings_df[listings_df['price'] <= limit_val]
        
        scatter_plot = plt.scatter(
            subset_df['longitude'], 
            subset_df['latitude'], 
            c=subset_df['price'], 
            cmap='viridis', 
            alpha=0.6, 
            s=10
        )
        plt.colorbar(scatter_plot, label='price')
        plt.title(f'spatial distribution of listings (standard market <= ${limit_val})')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.savefig(os.path.join(results_dir, 'spatial_distribution.png'))
        print("saved spatial_distribution.png")
        plt.close()
    else:
        print("latitude, longitude, or price columns missing for spatial analysis")

if __name__ == "__main__":
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    listings_df = load_data()
    if not listings_df.empty:
        analyze_data(listings_df)
        analyze_corr(listings_df, filter_out=True)
        analyze_segments(listings_df, limit_val=1000)
        plot_dist(listings_df, limit_val=1000)
        plot_map(listings_df, limit_val=1000)
