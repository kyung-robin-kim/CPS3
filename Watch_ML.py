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
labels={0:"non-vibrato",1:"vibrato"}
#labels=None
if read_cache==True: #read in previously processed features
    feature_files=os.listdir(path_to_processed)
    for f in feature_files:
        print("Reading In: %s"%f)
        feature_sets[int(f[12])]=pd.read_csv(os.path.join(path_to_processed,f))
else: #Process raw data into features
    for sz in window_sizes:
        feature_sets[sz]=get_featureset(vibrato_files,nonvibrato_files,window_size=sz,window_advance=1,desc_label_dict=labels)


## Applying ML Algorithms


#https://github.com/fracpete/sklearn-weka-plugin
#import sklweka.jvm as jvm
#from sklweka.dataset import load_arff, to_nominal_labels
#from sklweka.classifiers import WekaEstimator
#from sklweka.clusters import WekaCluster
#from sklweka.preprocessing import WekaTransformer

import sklearn.model_selection as ms
import sklearn.metrics as mt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

scoring={"F1":mt.f1_score,"Precision":mt.precision_score,"Recall":mt.recall_score,"Accuracy":mt.accuracy_score}
models={"Decision Tree":DecisionTreeClassifier,"Random Forest":RandomForestClassifier, "Support Vector Classifier":SVC,"Ada Boost Classifier":AdaBoostClassifier}
params={"Decision Tree":{'max_depth':[1,2,3,4,5]},"Random Forest":{'max_depth':[1,2,3,4,5]}, "Support Vector Classifier":{'C': [1, 10], 'kernel': ('linear', 'rbf')},"Ada Boost Classifier":{'n_estimators':[25,50,75,100]}}
#jvm.start(packages=True)

for sz in feature_sets.keys():
    print("Training on Features with Window Length %s"%sz)
    data=feature_sets[sz]
    features=data['mean_x','mean_y','mean_z',
                    'std_x','std_y','std_z',
                    'med_x','med_y','med_z',
                    'rms_x','rms_y','rms_z']
    target=data['Activity']

    #Do grid search cv on the data, for each model in models
    for model,param in zip(models.keys(),params.keys()):
        cv_params = GridSearchCV(models[model], params[param], scoring(scoring.keys()))
        cv_params.fit(X,y)
        cv_params.best_params_



'''
###PART 1: 1sec window, adv 1sec, original features
print("\nPart 1: Original Scheme With More Data")
j48 = WekaEstimator(classname="weka.classifiers.trees.J48",options=["-C","0.25","-M", "2"])
p1_data=feature_sets[1]
p1_X=p1_data[['mean_x','std_x','mean_y','std_y','mean_z','std_z']]
p1_y=p1_data['Activity']
p1_accuracy=cross_val_score(j48,p1_X, p1_y, cv=10, scoring='accuracy')
print("1 second window accuracy= %f"%p1_accuracy.mean())

##PART 2: 2,3,4 sec windows with original features
print("\nPart 2: Varying Window Length")
windows=[2,3,4]
p2_dict={1:p1_accuracy}
for window in windows:
    data=feature_sets[window]
    X=data[['mean_x','std_x','mean_y','std_y','mean_z','std_z']]
    y=data['Activity']
    accuracy=cross_val_score(j48,X, y, cv=10, scoring='accuracy')
    p2_dict[window]=accuracy
print("1 second window accuracy= %f"%p2_dict[1].mean())
print("2 second window accuracy= %f"%p2_dict[2].mean())
print("3 second window accuracy= %f"%p2_dict[3].mean())
print("4 second window accuracy= %f"%p2_dict[4].mean())

Get best window/accuracy:
all_vals=list(p2_dict.values())
mean_vals=[np.mean(a) for a in all_vals]
best_acc=max(mean_vals)
inv_dict=dict(zip(mean_vals,p2_dict.keys()))
best_window=inv_dict[best_acc]

best_score=0
best_window=0
for key in p2_dict.keys():
    score=p2_dict[key].mean()
    if score>best_score:
        best_score=score
        best_window=key


##Part 3: 1,2,3,4 sec windows with full features
print("\nPart 3: Added Feature Set")
windows=[1,2,3,4]
p3_dict={}
for window in windows:
    data=feature_sets[window]
    X=data[['mean_x','mean_y','mean_z',
            'std_x','std_y','std_z',
            'med_x','med_y','med_z',
            'rms_x','rms_y','rms_z']]
    y=data['Activity']
    accuracy=cross_val_score(j48,X, y, cv=10, scoring='accuracy')
    p3_dict[window]=accuracy
print("1 second window accuracy= %f"%p3_dict[1].mean())
print("2 second window accuracy= %f"%p3_dict[2].mean())
print("3 second window accuracy= %f"%p3_dict[3].mean())
print("4 second window accuracy= %f"%p3_dict[4].mean())

## Part 4: sequential feature selection
print("\nPart 4: Sequential Feature Selection (Decision Tree)")


def get_best_feature(mlmodel,data,feature_lists,target='Activity'):
    For a given set of lists of features, choose the list with the highest accuracy
    best_score=0
    best_fl=None
    y=data[target]
    for feature_list in feature_lists:
        X=data[feature_list]
        accuracy=cross_val_score(mlmodel,X, y, cv=10, scoring='accuracy').mean()
        if accuracy>best_score:
            best_score=accuracy
            best_fl=feature_list
    return (best_fl, best_score)


def sequential_feature_selector(mlmodel,data,fundamental_features,
                                target='Activity',delta_score_thresh=1):
    find the best feature list, breaking if only a 1% gain is made
    ff=fundamental_features.copy()
    feature_lists=fundamental_features.copy()
    n_features=len(feature_lists)
    ranked_features=[]
    i=0
    delta_score=100
    prev_score=1e-6

    feature_log={}
    while i<len(feature_lists):
        print("finding the %d best features"%(i+1))
        best=get_best_feature(mlmodel,data,feature_lists)
        best_feature_list=best[0]
        next_best_feature=best_feature_list[-1]
        ff.remove([next_best_feature])
        best_score=best[1]
        delta_score=(best_score-prev_score)/prev_score*100
        if delta_score<delta_score_thresh:
            break
        prev_score=best_score
        feature_lists.remove(best_feature_list)
        feature_lists=[best_feature_list+i for i in ff]
        feature_log[i+1]=(best_feature_list,best_score)
        i+=1
    return feature_log


fundamental_features=[[i] for i in['mean_x','mean_y','mean_z',
                'std_x','std_y','std_z',
                'med_x','med_y','med_z',
                'rms_x','rms_y','rms_z']]
print("J48")
DT_sfs=sequential_feature_selector(j48,feature_sets[best_window],fundamental_features)
print ("\nJ48 Sequential Feature Selection")
for key in DT_sfs:
    print("%d Best Features: "%key,DT_sfs[key])

## Part 5: SFS with Random Forest and SVM
print("\nPart 5: SFS with Random Forest and SVM")
print("SVM")
smo=WekaEstimator(classname="weka.classifiers.functions.SMO")
SMO_sfs=sequential_feature_selector(smo,feature_sets[best_window],fundamental_features)
print ("\nSMO (SVC) Sequential Feature Selection")
for key in SMO_sfs:
    print("%d Best Features: "%key,SMO_sfs[key])

print("\n\nRF")
rf=WekaEstimator(classname="weka.classifiers.trees.RandomForest",options=["-I","10"])
RF_sfs=sequential_feature_selector(rf,feature_sets[best_window],fundamental_features)
print ("\nRandom Forest Sequential Feature Selection")
for key in RF_sfs:
    print("%d Best Features: "%key,RF_sfs[key])
'''
