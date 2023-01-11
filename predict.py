import argparse
import os 
from src.model import classifier
import csv
from tqdm import tqdm
from joblib import Parallel, delayed

def model_prediction(model_path,filename,result_path,remove_redundant):
    model.load(model_path,filename=filename)

    # Perform prediction to generate features importance
    output = model.predict_single_class(
        result_path=result_path,
        save_result=False,
        #type='impurity'
        remove_redundancy=remove_redundant)

    return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Predict feature relevance from classifier model'
    )

    parser.add_argument(
        "--model_path",
        default='./model',
        help='directory to save model')

    parser.add_argument(
        "--result_path",
        default='./results',
        help='directory to save results')
    
    parser.add_argument(
        "--result_filename",
        default='output_full.csv',
        help='result filename')
    
    parser.add_argument(
        "--remove_redundant",
        default=False,
        action='store_true',
        help='Remove redundant feature with zero importance')

    args = parser.parse_args()

    model_path = args.model_path
    result_path = args.result_path
    result_filename = args.result_filename
    remove_redundant = args.remove_redundant

    # Create directory if not exists
    os.makedirs(args.result_path,exist_ok=True)

    # Load classifier
    model = classifier()

    model_name = os.listdir(model_path)
    model_name = [f for f in sorted(model_name) if f.endswith('gz')]

    # Sort the model by intervention number
    INTNum = [int(f.split('.')[0].split('INT')[1]) for f in model_name]
    model_name = [f for _,f in sorted(zip(INTNum,model_name))]

    # feat_list.append([i] + output)
    feat_list = Parallel(n_jobs=-2)(delayed(model_prediction)(model_path,f,result_path,remove_redundant) for f in tqdm(model_name))

    with open(os.path.join(result_path,result_filename),'w') as f:
        writer  = csv.writer(f)
        writer.writerows(feat_list)

