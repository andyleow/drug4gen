import argparse
import os 
from src.data import load_dataset,preprocessing
from src.model import classifier

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train classifier model'
    )

    parser.add_argument(
        "--data_path",
        default='./data',
        help='directory to load dataset')
    
    parser.add_argument(
        "--model_path",
        default='./model',
        help='directory to save model')

    parser.add_argument(
        "--train_subset",
        default=False,
        action='store_true',
        help='Train subset for quick test')

    args = parser.parse_args()

    data_path = args.data_path
    model_path = args.model_path

    # Create directory if not exists
    os.makedirs(args.model_path,exist_ok=True)

    # Load table
    features_df, dependency_df = load_dataset(data_path)

    # Preprocessing data
    df = preprocessing(features_df, dependency_df)

    # Store feature and intervention name list
    features_col = [i for i in df.columns if i.startswith('FEAT')]
    
    intervention_col = [i for i in df.columns if i.startswith('INT')]

    if args.train_subset:
        intervention_col = intervention_col[::1000]

    # Training data
    X = df[features_col]
    y = df[intervention_col]

    # Random forest classifier params 
    param = {
    'n_estimators':50,
    'max_depth':5,
    'min_samples_split':10,
    'n_jobs':-1,
    'random_state':0, 
    'verbose':0,
    'class_weight':'balanced' # balanced_subsample
    }

    model = classifier(name='random_forest',param=param)

    a = model.train(
        X,y, 
        feature_name = features_col,
        label_name = intervention_col,
        stratify = True,
        test_split = 0.1,
        save_model = True,
        save_freq = 1,
        model_path = model_path,
        verbose=1
    )

