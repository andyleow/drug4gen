import pandas as pd
import os
from sklearn.feature_selection import VarianceThreshold
import numpy as np

def load_dataset(data_path):
    """Function to dataset from a specified path 
    
    Args:
        data_path: path containing the data
    
    Returns:
        features_df: dataframe contains gene features for each sample 
        dependency_df: dataframe contains intervention outcome


    """
    features_df = pd.read_csv(os.path.join(data_path,'features.csv'), index_col=0)
    dependency_df = pd.read_csv(os.path.join(data_path,'dependency.csv'), index_col=0)
  
    return features_df, dependency_df

def preprocessing(features_df, dependency_df):
    """Function to preprocess input and create a train dataset
    Specifically, the missing values are impute with the median value,
    and constant/quasi constant features are removed.

    Args:
        features_df: dataframe contains gene features for each sample 
        dependency_df: dataframe contains intervention outcome
    
    Returns:
        output_df = dataset joining the  features and intervention response 
    """
    # impute missing value with median value
    print('Imputing missing value')
    features_df = features_df.fillna(features_df.median())
    dependency_df = dependency_df.fillna(value=dependency_df.median())

    # Remove constant/quasi constant features
    print('Remove constant feature/quasi features')
    sel = VarianceThreshold(threshold=0.01) # constant 99% of the time
    sel.fit(features_df.T)

    selected_features = features_df.index[sel.get_support()]
    features_df = features_df.loc[selected_features,:]
    print(f" # Removed featues: {len(features_df)-len(selected_features)}")

    # Remove highly correlated features
    #features_df = drop_high_correlated_features(features_df,thres=0.9)

    # Join table 
    output_df = features_df.T.join(dependency_df.T)
    print(f"Join table shape : {output_df.shape}")
    
    return output_df

def drop_high_correlated_features(df,thres=0.9):
    corrm_np = np.corrcoef(df)
    upper_np = np.triu(corrm_np, k=1)

    drop_index = []
    for col in range(upper_np.shape[1]):
        if any(upper_np[:,col] > thres):
            drop_index.append(col)

    print(f"dropping {len(drop_index)} features with correlation > {thres}")
    df = df.drop(labels=drop_index, axis=0)

    return df

