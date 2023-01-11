import joblib
import csv 
import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from copy import deepcopy

class classifier():
    """Tree based classifier
    
    Args:
        name: model name, for eg: random_forest, xgboost
        param: dict contain params for classifier

    """

    def __init__(self,name='random_forest',param=None):
        
        self.name = name
        self.param = param

        if self.name=='random_forest':
            if param:
                self._classifier = RandomForestClassifier(**param)
            else:
                self._classifier = RandomForestClassifier()
    
        else:
            if param:
                self._classifier = XGBClassifier(**param)
            else:
                self._classifier = XGBClassifier()

        self.classifiers = []
        self.metrics = {}
        self.features_name = []
        self.label_name = []

    def train(
        self,
        X,
        y, 
        feature_name = None,
        label_name = None,
        save_model = False,
        save_freq = None,
        stratify=False,
        test_split = 0.2,
        model_path = '.',
        verbose=0):
        """Fit model to input X and output y
        
        Args:
            X: dataframe contain the input features
            y: dataframe contain the output response
            feature_name: input features name
            label_name: output_label
            save_model: save_model by end of training
            save_freq: model checkpoint save frequency
            stratify: stratify data by label
            test_split: Fraction of test data split
            model_path: path to save the model checkpoint
            verbose: whether to print progress message
  
        """
        
        if feature_name:
            self.features_name = feature_name
        
        if label_name:
            self.label_name = label_name

        if save_model:
            if save_freq==None:
                save_freq = len(y_test)
        
        clf = self._classifier
        classifiers_list = []
        labels_list = []
        train_accuracy, test_accuracy =[],[]
        train_roc, test_roc =[],[]

        pbar = tqdm(enumerate(label_name),total=len(label_name))
        for i,label in pbar:#tqdm(enumerate(label_name)):
        
            pbar.set_description(f'Processing {label}')
            
            # Splitting dataset to train and test set
            if stratify:
                if (sum(y[label])>1) and (sum(1- y[label])>1): # take care of extreme output
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,y[label],test_size=test_split,random_state=100, stratify=y[label])
                else:
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,y[label],test_size=test_split,random_state=100)
            else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,y[label],test_size=test_split,random_state=100)

            # Fit classifier
            clf.fit(X_train, y_train)

            # Evalute model
            y_pred = clf.predict(X_train)
            train_accuracy.append(np.float16(accuracy_score(y_train, y_pred)))
            try:
                train_roc.append(np.float16(roc_auc_score(y_train, y_pred)))
            except:
                train_roc.append(np.nan)

            y_pred = clf.predict(X_test)
            test_accuracy.append(np.float16(accuracy_score(y_test, y_pred)))

            try:
                test_roc.append(roc_auc_score(y_test, y_pred))
            except:
                # Extreme case only single label
                test_roc.append(np.nan)


            if (i%100)==0 and verbose:
                # print(f"train time: {train_time:.3}s")
                
                print(f"Training accuracy is {train_accuracy[-1]}")
                print(f"Training roc is {train_roc[-1]}")
            
                print(f"Testing accuracy is {test_accuracy[-1]}")
                print(f"Testing roc is {test_roc[-1]}")

            classifiers_list.append(deepcopy(clf))
            labels_list.append(label)

            self.metrics = {
                'train_accuracy': train_accuracy,
                'train_roc': train_roc,
                'test_accuracy': test_accuracy,
                'test_roc': test_roc
            }

            if save_model:
                if save_freq > 1:
                    if (i>0)&((i%save_freq)==0):
                        filename = f"{self.name}_classifier-{i}.model.gz"
                        self.save(model_path,filename,classifiers_list,labels_list)
                        
                else:
                    filename = f"{self.name}_classifier-{label}.model.gz"
                    self.save(model_path,filename,clf,label)
                
                #Reset classifier and label list
                classifiers_list = [] 
                labels_list = []
                train_accuracy, test_accuracy =[],[]
                train_roc, test_roc =[],[]


                    
                




    def predict(
        self,
        save_result=True,
        result_path=None,
        filename='output.csv',
        #type='impurity',
        remove_redundancy=True):
        """Return the feature importance 

        Args:
            save_result: whether to save the output to a csv
            result_path: path to save the relative importance
            filename: filename of the csv 
            remove_redundancy: whether to remove features with zero importance
        
        Returns:
            output_list: list consist of the relative importance for each intervention
        """

        output_list = []
        
        #pbar = tqdm(enumerate(ziplabel_name),total=len(label_name))
        for i,(clf,label) in enumerate(zip(self.classifiers,self.label_name)):
            print(f'Processing {i} : {label}')
            feat_importance = clf.feature_importances_

            # Sort feature importance and remove irrelevent feature
            # print(feat_importance.shape, self.features_name)
            temp_df = pd.DataFrame(
            {
                'feature': self.features_name,
                'importance':feat_importance
                }
            ).set_index('feature').sort_values(by='importance',ascending=False)
            
            # Filter features not with zero importance
            if remove_redundancy:
                output_list.append([i,label] + temp_df[temp_df['importance']>0].index.to_list())
            else:
                output_list.append([i,label] + temp_df.index.to_list())

        if save_result:
            with open(os.path.join(result_path,filename),'w') as f:
                writer  = csv.writer(f)
                writer.writerows(output_list)

        return output_list

    def predict_single_class(
        self,
        save_result=True,
        result_path=None,
        filename='output.csv',
        #type='impurity',
        remove_redundancy=True):
        """Return the feature importance of a single class 

        Args:
            save_result: whether to save the output to a csv
            result_path: path to save the relative importance
            filename: filename of the csv 
            remove_redundancy: whether to remove features with zero importance
        
        Returns:
            output_list: list consist of the relative importance for each intervention
        """

        clf = self.classifiers
        label = self.label_name

        # print(f'Processing class : {label}')
        feat_importance = clf.feature_importances_

        # Sort feature importance and remove irrelevent feature
        # print(feat_importance.shape, self.features_name)
        temp_df = pd.DataFrame(
        {
            'feature': self.features_name,
            'importance':feat_importance
            }
        ).set_index('feature').sort_values(by='importance',ascending=False)
        
        # Filter features not with zero importance
        if remove_redundancy:
            output_list = [label] + temp_df[temp_df['importance']>0].index.to_list()
        else:
            output_list = [label] + temp_df.index.to_list()

        if save_result:
            with open(os.path.join(result_path,filename),'w') as f:
                writer  = csv.writer(f)
                writer.writerows(output_list)

        return output_list

    def save(
        self, 
        model_path, 
        filename='rf_classifier',
        classifiers = None,
        label_name =None):
        """Save classifers model and metrics
        
        Args: 
            model_path: path to save the model
            filename: model name

        """

        output = {
            'classifiers': classifiers,
            'features': self.features_name,
            'labels': label_name,
            'metrics': self.metrics
        }
        joblib.dump(output, os.path.join(model_path,filename))


    def load(
        self,
        model_path,
        filename='rf_classifier'
        ):
        """Reload classifers model and metrics
        
        Args: 
            model_path: path to save the model
            filename: model name
        """
        # print(f'Loading model {filename}')
        model_dict = joblib.load(os.path.join(model_path,filename))

        self.classifiers = model_dict['classifiers']
        self.features_name = model_dict['features']
        self.label_name = model_dict['labels']
        self.metrics = model_dict['metrics'] 
