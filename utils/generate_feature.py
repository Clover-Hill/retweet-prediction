import os
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from scipy.stats import norm, zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import datetime
from datetime import timezone
from math import ceil
import warnings
warnings.filterwarnings('ignore')

# Libraries needed:
# pip install datasets pandas numpy scipy scikit-learn tqdm

# Constants
RANDOM_STATE = 42
USER_CLUSTER_NUM = 1000
N_SPLITS = 5

def extract_time_features(df):
    """Extract time-related features from timestamp"""
    timestamps = pd.to_datetime(df['timestamp'], unit='ms')
    
    df['weekday'] = timestamps.dt.weekday
    df['hour'] = timestamps.dt.hour
    df['day'] = timestamps.dt.day
    df['month'] = timestamps.dt.month
    df['week_of_month'] = timestamps.apply(lambda x: ceil(x.day / 7.0))
    
    # Time since a reference point (latest time in dataset)
    latest = timestamps.max()
    df['hours_from_latest'] = (latest - timestamps).dt.total_seconds() / 3600
    
    return df

def create_tweet_metrics_features(df):
    """Create various transformations of tweet metrics"""
    # Base metrics
    base_metrics = ['user_followers_count', 'user_friends_count', 'user_statuses_count']
    
    # Create interaction features
    df['followers_friends'] = df['user_followers_count'] * df['user_friends_count']
    df['followers_statuses'] = df['user_followers_count'] * df['user_statuses_count']
    df['friends_statuses'] = df['user_friends_count'] * df['user_statuses_count']
    df['followers_friends_statuses'] = df['user_followers_count'] * df['user_friends_count'] * df['user_statuses_count']
    
    # Extended base metrics
    extended_metrics = base_metrics + ['followers_friends', 'followers_statuses', 
                                      'friends_statuses', 'followers_friends_statuses']
    
    # Apply transformations
    for metric in extended_metrics:
        # Log transformation
        df[f'{metric}_log'] = np.log1p(df[metric])
        
        # Z-score normalization
        df[f'{metric}_z'] = zscore(df[metric].fillna(0))
        
        # Rank transformation
        df[f'{metric}_rank'] = df[metric].rank(method='min')
        
        # CDF transformation
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        if std_val > 0:
            df[f'{metric}_cdf'] = norm.cdf(df[metric].values, loc=mean_val, scale=std_val)
        else:
            df[f'{metric}_cdf'] = 0.5
        
        # Quantile binning (10 bins)
        df[f'{metric}_qbin_10'] = pd.qcut(df[metric], q=10, labels=False, duplicates='drop')
    
    return df

def process_varlen_features(df):
    """Process variable length features like mentions, hashtags, urls"""
    # Count features
    df['mentions_count'] = df['user_mentions'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['hashtags_count'] = df['hashtags'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    df['urls_count'] = df['urls'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    
    # Binary features
    df['has_mentions'] = (df['mentions_count'] > 0).astype(int)
    df['has_hashtags'] = (df['hashtags_count'] > 0).astype(int)
    df['has_urls'] = (df['urls_count'] > 0).astype(int)
    
    # URL domain features
    def extract_url_features(urls_str):
        if pd.isna(urls_str) or not urls_str:
            return {'n_domains': 0, 'has_https': 0, 'has_shortener': 0}
        
        urls = urls_str.split(',')
        domains = set()
        has_https = 0
        has_shortener = 0
        
        shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly']
        
        for url in urls:
            if 'https://' in url:
                has_https = 1
            
            # Extract domain
            if '://' in url:
                domain = url.split('://')[1].split('/')[0]
            else:
                domain = url.split('/')[0]
            domains.add(domain)
            
            # Check for URL shorteners
            for shortener in shorteners:
                if shortener in domain:
                    has_shortener = 1
                    break
        
        return {'n_domains': len(domains), 'has_https': has_https, 'has_shortener': has_shortener}
    
    url_features = df['urls'].apply(extract_url_features)
    df['n_unique_domains'] = [f['n_domains'] for f in url_features]
    df['has_https'] = [f['has_https'] for f in url_features]
    df['has_url_shortener'] = [f['has_shortener'] for f in url_features]
    
    return df

def create_text_features(df):
    """Create features from tweet text"""
    # Basic text statistics
    df['text_length'] = df['text'].fillna('').apply(len)
    df['word_count'] = df['text'].fillna('').apply(lambda x: len(x.split()))
    
    # Character-level features
    df['n_capital_letters'] = df['text'].fillna('').apply(lambda x: sum(1 for c in x if c.isupper()))
    df['n_exclamation'] = df['text'].fillna('').apply(lambda x: x.count('!'))
    df['n_question'] = df['text'].fillna('').apply(lambda x: x.count('?'))
    df['n_punctuation'] = df['text'].fillna('').apply(lambda x: sum(1 for c in x if c in '.,;:!?'))
    
    # Ratios
    df['capital_ratio'] = df['n_capital_letters'] / (df['text_length'] + 1)
    df['exclamation_ratio'] = df['n_exclamation'] / (df['word_count'] + 1)
    df['punctuation_ratio'] = df['n_punctuation'] / (df['text_length'] + 1)
    
    return df

def create_tfidf_features(df, n_components=5):
    """Create TF-IDF features from combined text elements"""
    # Combine text elements
    texts = []
    for idx, row in df.iterrows():
        elements = []
        
        # Add hashtags
        if pd.notna(row['hashtags']):
            elements.extend(row['hashtags'].split(','))
        
        # Add mentions
        if pd.notna(row['user_mentions']):
            elements.extend(['@' + m for m in row['user_mentions'].split(',')])
        
        # Add some words from text
        if pd.notna(row['text']):
            words = row['text'].lower().split()[:10]  # First 10 words
            elements.extend(words)
        
        texts.append(' '.join(elements))
    
    # TF-IDF with SVD
    try:
        tfv = TfidfVectorizer(min_df=5, max_features=1000, 
                              strip_accents='unicode', analyzer='word',
                              ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)
        
        tfidf_matrix = tfv.fit_transform(texts)
        
        # Reduce dimensions
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        
        # Add to dataframe
        for i in range(n_components):
            df[f'tfidf_svd_{i}'] = tfidf_reduced[:, i]
    except:
        # If TF-IDF fails, create dummy features
        for i in range(n_components):
            df[f'tfidf_svd_{i}'] = 0
    
    return df

def create_user_clusters(df):
    """Create user clusters based on user statistics"""
    # For user clustering, we'll use the 'id' as a proxy for user
    # Group by user to get user-level statistics
    user_stats = df.groupby('id').agg({
        'user_followers_count': ['mean', 'std', 'max'],
        'user_friends_count': ['mean', 'std', 'max'],
        'user_statuses_count': ['mean', 'std', 'max'],
        'mentions_count': 'mean',
        'hashtags_count': 'mean',
        'urls_count': 'mean',
        'text_length': 'mean',
        'word_count': 'mean'
    }).fillna(0)
    
    # Flatten column names
    user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
    
    # K-means clustering
    try:
        kmeans = KMeans(n_clusters=min(USER_CLUSTER_NUM, len(user_stats) // 10), 
                        random_state=RANDOM_STATE)
        user_stats['cluster'] = kmeans.fit_predict(user_stats)
        
        # Map back to original dataframe
        user_to_cluster = user_stats['cluster'].to_dict()
        df['user_cluster'] = df['id'].map(user_to_cluster).fillna(0)
    except:
        df['user_cluster'] = 0
    
    return df

def create_count_encoding(df, columns):
    """Create count encoding for categorical columns"""
    for col in columns:
        counts = df[col].value_counts().to_dict()
        df[f'{col}_count_enc'] = df[col].map(counts)
    
    return df

def create_target_encoding(df, columns, target_col='retweet_count', n_splits=5):
    """Create target encoding with K-fold to prevent overfitting"""
    # Only apply to training data (where target is available)
    train_mask = df[target_col].notna()
    
    if not train_mask.any():
        # No training data, return dummy features
        for col in columns:
            df[f'{col}_target_enc'] = 0
        return df
    
    # Log transform target
    y_train = np.log1p(df.loc[train_mask, target_col].values)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    for col in columns:
        # Initialize target encoding column
        df[f'{col}_target_enc'] = 0.0
        
        # Only process training data
        X_train = df.loc[train_mask, col].values
        te_values = np.zeros(len(X_train))
        
        # K-fold target encoding
        for train_idx, val_idx in kf.split(X_train):
            # Calculate mean target for each category in training fold
            temp_df = pd.DataFrame({
                'feature': X_train[train_idx],
                'target': y_train[train_idx]
            })
            means = temp_df.groupby('feature')['target'].mean().to_dict()
            
            # Apply to validation fold
            te_values[val_idx] = [means.get(x, y_train.mean()) for x in X_train[val_idx]]
        
        # Set values for training data
        df.loc[train_mask, f'{col}_target_enc'] = te_values
        
        # For test data, use overall means from all training data
        temp_df = pd.DataFrame({
            'feature': X_train,
            'target': y_train
        })
        overall_means = temp_df.groupby('feature')['target'].mean().to_dict()
        global_mean = y_train.mean()
        
        test_mask = ~train_mask
        if test_mask.any():
            df.loc[test_mask, f'{col}_target_enc'] = df.loc[test_mask, col].map(overall_means).fillna(global_mean)
    
    return df

def main():
    print("Loading dataset...")
    dataset = datasets.load_from_disk("/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/raw_dataset")
    
    # Convert to pandas for easier manipulation
    train_df = dataset['train'].to_pandas()
    eval_df = dataset['eval'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # Add a flag to distinguish train/test
    train_df['is_train'] = 1
    eval_df['is_train'] = 0
    test_df['is_train'] = 0

    eval_df['is_eval'] = 1
    test_df['is_eval'] = 0
    
    # Combine for feature engineering
    df = pd.concat([train_df, eval_df,  test_df], ignore_index=True)
    print(f"Total samples: {len(df)}")
    
    # Feature engineering
    print("Extracting time features...")
    df = extract_time_features(df)
    
    print("Creating tweet metrics features...")
    df = create_tweet_metrics_features(df)
    
    print("Processing variable length features...")
    df = process_varlen_features(df)
    
    print("Creating text features...")
    df = create_text_features(df)
    
    print("Creating TF-IDF features...")
    df = create_tfidf_features(df)
    
    print("Creating user clusters...")
    df = create_user_clusters(df)
    
    # Count encoding for categorical features
    print("Creating count encodings...")
    cat_features = ['weekday', 'hour', 'day', 'month', 'user_verified', 'user_cluster']
    df = create_count_encoding(df, cat_features)
    
    # Target encoding (only for features with reasonable cardinality)
    print("Creating target encodings...")
    target_enc_features = ['weekday', 'hour', 'user_cluster', 
                          'user_followers_count_qbin_10', 'user_friends_count_qbin_10']
    df = create_target_encoding(df, target_enc_features)
    
    # Handle missing values
    print("Handling missing values...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Split back to train/test
    train_df = df[df['is_train'] == 1].drop('is_train', axis=1)
    test_eval_df = df[df['is_train'] == 0].drop('is_train', axis=1)
    eval_df = test_eval_df[test_eval_df['is_eval'] == 1].drop('is_eval', axis=1)
    test_df = test_eval_df[test_eval_df['is_eval'] == 0].drop('is_eval', axis=1)
    
    # Create new datasets
    print("Creating new datasets...")
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create DatasetDict
    new_dataset = DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset,
        'test': test_dataset
    })
    
    # Save
    print("Saving dataset...")
    new_dataset.save_to_disk("/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/feature_dataset")
    
    # Print feature summary
    original_features = ['id', 'timestamp', 'retweet_count', 'user_verified', 'user_statuses_count',
                        'user_followers_count', 'user_friends_count', 'user_mentions', 'urls', 
                        'hashtags', 'text']
    new_features = [col for col in train_df.columns if col not in original_features]
    
    print(f"\nFeature extraction complete!")
    print(f"Original features: {len(original_features)}")
    print(f"New features added: {len(new_features)}")
    print(f"Total features: {len(train_df.columns)}")
    print(f"\nNew features: {new_features[:20]}...")  # Show first 20 new features

if __name__ == "__main__":
    main()