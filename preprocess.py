import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def preprocess_and_partition(filepath):
    """
    Reads the loan default data, processes categorical variables,
    and returns partitioned sets: train, val, test.
    """
    df = pd.read_csv(filepath)
    
    # Drop "id" as it is just an identifier, not a feature
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # Convert string columns to numeric
    # term: " 36 months" -> 36
    if df['term'].dtype == object:
        df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
        
    # emp_length: "10+ years" -> 10, "< 1 year" -> 0
    if df['emp_length'].dtype == object:
        df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
        df['emp_length'] = df['emp_length'].fillna(0) # assume missing implies 0 or impute with 0
        
    # Extract year from earliest_cr_line instead of keeping the string (e.g. Mar-00)
    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y').dt.year
        
    # For other object columns, use LabelEncoder
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    # Handle any remaining missing values (if any)
    df = df.fillna(df.median())
    
    X = df.drop(columns=['class']).values
    y = df['class'].values
    
    # Partition data
    # We will use 70% Train, 15% Validation, 15% Test
    # First split into 70% Train, 30% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    # Then split Temp into 50% Validation, 50% Test (which means 15% / 15% of total)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    feature_names = df.drop(columns=['class']).columns.tolist()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def feature_selection_pearson(x, y):
    """
    Calculates Pearson correlation of the features with the target variable, y.
    Returns the list of feature column indices ranked in descending order of correlation magnitude.
    """
    n_features = x.shape[1]
    correlations = []
    for i in range(n_features):
        corr, _ = pearsonr(x[:, i], y)
        if np.isnan(corr):
            corr = 0
        correlations.append(np.abs(corr))
    
    # Rank in descending order
    ranked_indices = np.argsort(correlations)[::-1]
    return ranked_indices.tolist()


def feature_selection_spearman(x, y):
    """
    Calculates Spearman correlation of the features with the target variable, y.
    Returns the list of feature column indices ranked in descending order of correlation magnitude.
    """
    n_features = x.shape[1]
    correlations = []
    for i in range(n_features):
        corr, _ = spearmanr(x[:, i], y)
        if np.isnan(corr):
            corr = 0
        correlations.append(np.abs(corr))
        
    # Rank in descending order
    ranked_indices = np.argsort(correlations)[::-1]
    return ranked_indices.tolist()


def feature_selection_mi(x, y):
    """
    Calculates mutual information of the features with the target variable, y.
    Returns the list of feature column indices ranked in descending order of MI.
    """
    mi_scores = mutual_info_classif(x, y, random_state=42)
    
    # Rank in descending order
    ranked_indices = np.argsort(mi_scores)[::-1]
    return ranked_indices.tolist()

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, feats = preprocess_and_partition('loan_default.csv')
    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    
    pearson_idx = feature_selection_pearson(X_train, y_train)
    print("Top 5 Pearson features:", [feats[i] for i in pearson_idx[:5]])
    
    spearman_idx = feature_selection_spearman(X_train, y_train)
    print("Top 5 Spearman features:", [feats[i] for i in spearman_idx[:5]])
    
    mi_idx = feature_selection_mi(X_train, y_train)
    print("Top 5 MI features:", [feats[i] for i in mi_idx[:5]])
