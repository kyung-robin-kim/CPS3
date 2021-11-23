#!/usr/bin/env python
# coding: utf-8

# # Watch Feature Extraction: outputs a .csv file of mean (x-y-z), std (x-y-z), median (x-y-z), rms (x-y-z) and Activity for hand wash and non-hand wash smartwatch data

# ## Imports and Setup


#Get useful libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'notebook')


# ## File Management
path_to_v='vib_raw'
path_to_nv='no_vib_raw'
path_to_processed='processed_features'
vibrato_files=os.listdir(path_to_v)
nonvibrato_files=os.listdir(path_to_nv)


# ## Feature Processing
#read_cache=False
read_cache=True #Set to True to read completed feature csvs, false to process new files

def get_features(filename,window_size=1,window_advance=1):
    """Function to read in a watch datafile and output a features dataframe
        Inputs: filename - the .csv file containing the data
        Outputs: feature_data - a pandas.DataFrame object containing per interval mean and std x,y,&z accelerations
    """
    raw_data=pd.read_csv(filename,header=None,skiprows=1)#Pull in data
    raw_data.columns=['timestamp','p','q','ax','ay','az'] #Useful names for data headers
    raw_data['time']=(raw_data['timestamp']-raw_data['timestamp'][0])/1000 # convert time from ms to s
    max_time=np.floor(raw_data['time'].max()) #Figure out our maximum time in the data, throwing out data past largest whole interval
    feature_data=pd.DataFrame(columns=['mean_x','mean_y','mean_z',
                                       'std_x','std_y','std_z',
                                       'med_x','med_y','med_z',
                                       'rms_x','rms_y','rms_z'],index=range(1,int(max_time)+1)) #create DataFrame for our feature data

    for i in range(window_size,int(max_time)+1,window_advance):
        #Grab 1 interval of data , i-1<=t<i
        b=raw_data[raw_data['time']<i]
        interval=b[b['time']>=(i-window_size)][['ax','ay','az']]
        mu=interval.mean() # get our means for this interval
        mu.index=['mean_x','mean_y','mean_z'] #labeling
        sigma=interval.std() # get our stds for this interval
        sigma.index=['std_x','std_y','std_z'] #labeling
        medians=interval.median() # get our medians for this interval
        medians.index=['med_x','med_y','med_z']
        rms=interval.apply(get_rms)
        rms.index=['rms_x','rms_y','rms_z']
        #merge and add to output frame
        feature_row=pd.concat([mu,sigma,medians,rms])
        feature_data.loc[i]=feature_row
        
    return feature_data

def get_rms(x):
    """given array-like x, calculate the root-mean-square of the array"""
    return np.sqrt(np.mean(x**2))

def label_class(features,clas):
    """bulk label a DataFrame. adds label column in place
    Inputs: clas -must be a numeric class identifier"""
    
    features['Activity']=np.ones(features.shape[0],int)*clas

def get_featureset(v_files,nv_files,window_size=1,window_advance=1,
                   cache=True,
                   desc_label_dict=None,
                   v_path=os.path.join(os.getcwd(),r'vib_raw'),
                   nv_path=os.path.join(os.getcwd(),r'no_vib_raw'),
                   outpath=os.path.join(os.getcwd(),r'processed_features')):
    """Given a set of vibrato data files, non vibrato data files, a window size, and a window advance, use get features and create a complete dataset.
        v_files-list of strings containing filenames of csv datasets of vibrato data
        nv_files-list of strings containing filenames of csv datasets of NON-vibrato data
        window_size-Size of window to apply on the data, in seconds
        window_advance-How much to slide the window each time, in seconds
        cache-Boolean: True=>save output DataFrame to csv
        desc_label_dict- Dictionary for alternate class labels. if None, defaults to 0 one. if dict, uses the dict to add alternate class labels
        v_path-path to vibrato files
        nv_path-path to NON-vibrato files
        outpath-path to write out featureset if cache=True"""

    print("Processing Features for Window Size: %d with Step: %d"%(window_size,window_advance))
    vfeatures=pd.DataFrame(columns=['mean_x','mean_y','mean_z',
                                       'std_x','std_y','std_z',
                                       'med_x','med_y','med_z',
                                       'rms_x','rms_y','rms_z'])
    for file in v_files: #loop through all vibrato files, and concatenate the features together
        vfeatures=pd.concat([vfeatures,get_features(os.path.join(v_path,file),window_size=window_size,window_advance=window_advance).dropna()],ignore_index=True)
    label_class(vfeatures,1) #1=vibrato
    
    nvfeatures=pd.DataFrame(columns=['mean_x','mean_y','mean_z',
                                       'std_x','std_y','std_z',
                                       'med_x','med_y','med_z',
                                       'rms_x','rms_y','rms_z'])
    for file in nv_files: #loop through all non-vibrato files, and concatenate the features together
        nvfeatures=pd.concat([nvfeatures,get_features(os.path.join(nv_path,file),window_size=window_size,window_advance=window_advance).dropna()],ignore_index=True)
    label_class(nvfeatures,0) #0=nonvibrato
    
    featureset=pd.concat([vfeatures,nvfeatures],ignore_index=True)

    if desc_label_dict is not None:
        n_labels=len(desc_label_dict.keys())
        n_classes=len(set(featureset['Activity']))
        if n_labels!=n_classes:
            raise Exeception("You provided %d labels, but you have %d classes"%(n_labels,n_classes))
        featureset['Activity']=featureset['Activity'].apply(lambda x: desc_label_dict[x])
        
    if cache==True:
        now=pd.Timestamp.now().strftime('%H%M%S_%m%d%Y')
        featureset.to_csv(os.path.join(outpath,"featureset_w%d_s%d_%s.csv"%(window_size,window_advance,now)))
        
    return featureset

### preprocessing

window_sizes=[1,2,3,4]
feature_sets={}
# labels={0:"non-vibrato",1:"vibrato"}
labels=None
if read_cache==True: #read in previously processed features
    feature_files=os.listdir(path_to_processed)
    for f in feature_files:
        print("Reading In: %s"%f)
        feature_sets[int(f[12])]=pd.read_csv(os.path.join(path_to_processed,f))
else: #Process raw data into features
    for sz in window_sizes:
        feature_sets[sz]=get_featureset(vibrato_files,nonvibrato_files,window_size=sz,window_advance=1,desc_label_dict=labels)


## Applying ML Algorithms

#Switched from WEKA to sklearn

import sklearn.model_selection as ms
import sklearn.metrics as mt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

#List of scorers to calculate: Currently only using F1
scorers={"F1":mt.make_scorer(mt.f1_score),"Precision": mt.make_scorer(mt.precision_score),"Recall": mt.make_scorer(mt.recall_score),"Accuracy": mt.make_scorer(mt.accuracy_score)}

#Types of model to try
models={"Decision Tree":DecisionTreeClassifier(),"Random Forest":RandomForestClassifier(), "Support Vector Classifier":SVC(),"Ada Boost Classifier":AdaBoostClassifier()}
#Hyperparameter space to explore for each model
params={"Decision Tree":{'max_depth':[1,2,3,4,5]},"Random Forest":{'max_depth':[1,2,3,4,5]}, "Support Vector Classifier":{'C': [1, 10,100], 'kernel': ('linear', 'rbf')},"Ada Boost Classifier":{'n_estimators':[25,50,75,100]}}

#Find best models of each type, for features from each window size
overall_best_models={}
for sz in feature_sets.keys():
    print("Training on Features with Window Length %s"%sz)
    data=feature_sets[sz]
    features=data[['mean_x','mean_y','mean_z',
                    'std_x','std_y','std_z',
                    'med_x','med_y','med_z',
                    'rms_x','rms_y','rms_z']]
    target=data['Activity']
    best_models={}
    for model in models:
        est=models[model]
        param_dict=params[model]
        gsearch = GridSearchCV(est, param_dict, scoring = scorers, refit='F1')
        gsearch.fit(features,target)
        best_models[model]=gsearch
    overall_best_models[sz]=best_models

#Save results for plotting; print out hyper params of each top model
df_best_models=pd.DataFrame(columns=models.keys(),index=feature_sets.keys())
for sz in feature_sets.keys():
    print("\n")
    for model in models.keys():
        gsresult=overall_best_models[sz][model]
        df_best_models[model].loc[sz]=gsresult.best_score_
        print("Window Size: %s | Model: %s"%(sz,model))
        print("Best F1 Score: %0.4f"%gsresult.best_score_)
        print("Using Hyper Parameters: %s\n"%gsresult.best_params_)

#Plotting
df_best_models.plot(marker='o')
plt.grid()
plt.xticks([1,2,3,4])
plt.xlabel('Window Size (s)')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Feature Window Length For Best Models from Grid Search')
plt.show()





