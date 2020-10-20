# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:03:31 2019

@author: Group Q
"""

import pandas as pd
from pandas import read_csv
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer, StandardScaler, Normalizer, MinMaxScaler, Binarizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import model_from_json
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import xgboost as xgb
import matplotlib.pyplot as plt
#import scorers
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix



#load train data and split input features from target feature

filename = 'train_imperson_without4n7_balanced_data.csv'
dataframe = read_csv(filename)

array =  dataframe.values

X = array[:,0:152]

y = array[:,-1]

#load test data and split input features from target feature

filename_test = 'test_imperson_without4n7_balanced_data.csv'
dataframe_test = read_csv(filename_test)
array_test =  dataframe_test.values

X_test = array_test[:,0:152]

y_test = array_test[:,-1]


def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
####BUILD CLASSIFIERS#####

#Xgboost classifier
features = []
features.append(('select_best_chi', SelectKBest(score_func=chi2, k=8)))
features.append(('select_best_f', SelectKBest(score_func=f_classif, k=8)))
feature_union = FeatureUnion(features)

estimators = []
estimators.append(('VarThresh', VarianceThreshold()))
estimators.append(('feature_union', feature_union))
estimators.append(('xgb', xgb.XGBClassifier(max_depth=3, learning_rate=0.25,random_state=42, n_estimators=31)))
pipe1 = Pipeline(estimators)


#Adaboost classifier

features4 = []
features4.append(('select_best_chi', SelectKBest(score_func=chi2, k='all')))
features4.append(('select_best_f', SelectKBest(score_func=f_classif, k='all')))
features4.append(('pca10', PCA(n_components=30)))
feature_union4 = FeatureUnion(features4)

estimators4 = []
estimators4.append(('VarThresh', VarianceThreshold()))
estimators4.append(('feature_union', feature_union4))
estimators4.append(('AdaBoost', AdaBoostClassifier(n_estimators=31)))
pipe4 = Pipeline(estimators4)


#logistic regression classifier
pp_options = []
pp_options.append(('rescale', MinMaxScaler()))
pp_options.append(('select_best_chi', SelectKBest(score_func=chi2, k=23)))
pp_options.append(('select_best_f', SelectKBest(score_func=f_classif, k=23)))
pp_options.append(('LR_lbfgs', LogisticRegression(solver = "lbfgs")))
pipe5 = Pipeline(pp_options)

#Ensemble classifier
ensemble_clf = EnsembleVoteClassifier(clfs=[pipe1, pipe4], voting='hard', weights=[1, 1])

#NN Classifier
##############################################################################
##############################################################################

def NN():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"])
    return loaded_model

estimators_NN = []
estimators_NN.append(("skb_15_chi2", SelectKBest(score_func=chi2, k=15)))
estimators_NN.append(('NN',KerasClassifier(build_fn=NN, verbose=0)))
pipe_NN = Pipeline(estimators_NN)

##############################################################################
##############################################################################


##############################################################################
##############################################################################

models=[]
models.append(('pipe_LR',pipe5))
models.append(('ECHV',ensemble_clf))
models.append(('NN',pipe_NN))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier(n_neighbors=2)))
models.append(('CART',DecisionTreeClassifier()))
models.append(('XGB',xgb.XGBClassifier(max_depth=3, learning_rate=0.2,random_state=42, n_estimators=31)))
models.append(('AB',AdaBoostClassifier(n_estimators=31)))
models.append(('RF',RandomForestClassifier(n_estimators=300, random_state=0)))

# evaluate each model in turn
results = []
names = []

for name, model in models:
 stratifiedKFold = StratifiedKFold(n_splits=10, random_state=42)
 cv_results = cross_val_score(model, X, y, cv=stratifiedKFold, scoring='accuracy')
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

start = time.time()
results = []
combination_names = []
time_to_build = []
time_to_test = []

for name, model in models:
    startTime = time.time()
    model.fit(X, y)
    endTime = time.time()
    elapsed_train = endTime-startTime
    time_to_build.append(elapsed_train)
    #test model
    startTime = time.time()
    result = model.score(X, y)
    endTime = time.time()
    elapsed_test = endTime-startTime
    time_to_test.append(elapsed_test)
    print('for ',name, "train accuracy is:", round(result,4)*100)

start = time.time()
results = []
combination_names = []
time_to_build = []
time_to_test = []

for name, model in models:
    startTime = time.time()
    model.fit(X, y)
    endTime = time.time()
    elapsed_train = endTime-startTime
    time_to_build.append(elapsed_train)
    #test model
    startTime = time.time()
    result = model.score(X_test, y_test)
    endTime = time.time()
    elapsed_test = endTime-startTime
    time_to_test.append(elapsed_test)

    print('for ',name, "test accuracy is:", round(result,4)*100)


# Evaluate all models
models=[]
models.append(pipe1)#XGBOOST
models.append(pipe4)#ADABOOST
models.append(pipe5) # LR
models.append(ensemble_clf) # ensemble
models.append(pipe_NN) #NN


#times
time_to_build = []
time_to_test = []

#Test results
test_accuracy = []
test_dr = []
test_fpr = []

#Train results
train_accuracy = []
train_dr = []
train_fpr = []

#ROC 
roc_fpr=[]
roc_tpr=[]
roc_auc=[]

#no of fn
train_fn = []

for model in models:
    #build model
    startTime = time.time()
    model.fit(X,y)
    endTime = time.time()
    elapsed = endTime-startTime
    time_to_build.append(round(elapsed,2))
    ####TEST RESULTS####
    #test model
    startTime = time.time()
    #Calculate % accuracy
    result = model.score(X_test, y_test)
    elapsed = endTime-startTime
    time_to_test.append(elapsed)
    test_accuracy.append(round(result*100,2))
    
    
    #Calculate %AUC score 
    probs = model.predict_proba(X_test)
    probs = probs[: , 1]
    auc = roc_auc_score(y_test, probs)
    roc_auc.append(auc) 
    
    #Calculate confusion matrix and tn, fp etc. 
    predicted = model.predict(X_test)
    matrix = confusion_matrix(y_test, predicted)
    tn, fp, fn, tp = matrix.ravel()
    train_fn.append(fn)
    print(matrix)
    
    #calculate the roc curve
    fpr2, tpr, thresholds = roc_curve(y_test, probs)
    roc_fpr.append(fpr2)
    roc_tpr.append(tpr)    
    
    dr = round((tp / (tp + fn))*100, 2)
    test_dr.append(dr)
    fpr = round((fp / (fp + tn))*100,2)
    test_fpr.append(fpr)
    
    #print results
    #print("Test accuracy is:", round((result*100),2), "%")
    #print("Test Detection Rate/Recall =", dr, "%")
    #print("Test False Positive Rate=", fpr , "%")
    #print("No of false negatives=", fn)

    
    ####TRAIN RESULTS####
    result_train = model.score(X, y)
    train_accuracy.append(round(result_train*100,2))
    #Calculate %AUC score 
    probs_train = model.predict_proba(X)
    probs_train = probs_train[: , 1]
    auc_train = roc_auc_score(y, probs_train)
    
    #Calculate confusion matrix and tn, fp etc. 
    predicted_train = model.predict(X)
    matrix_train = confusion_matrix(y, predicted_train)
    tn_train, fp_train, fn_train, tp_train = matrix_train.ravel()
    
    #calculate the roc curve
    fpr_train, tpr_train, thresholds_train = roc_curve(y, probs_train)
    
    dr_train = round((tp_train / (tp_train + fn_train))*100, 2)
    train_dr.append(dr_train)
    
    fpr_train = round((fp_train / (fp_train + tn_train))*100,2)
    train_fpr.append(fpr_train)
    
    #print results
#    print("Train accuracy is:", round((result_train*100),2), "%")
#    print("Train Detection Rate/Recall =", dr_train, "%")
#    print("Train False Positive Rate=", fpr_train, "%")
    
###COMPARE TRAIN/TEST ACCURACY###
label = ['XGB', 'ADABST', 'LR', 'ENS XG+ADA', 'NN']
df = pd.DataFrame({'test_accuracy': test_accuracy,'train_accuracy': train_accuracy}, index=label)
axes = df.plot.line(rot=0, subplots=False, markevery=1, marker='o', markerfacecolor='r')
axes.axhline(y=99.918, color='r', label='Benchmark Accuracy 99.97%', ls='--', lw=2)
axes.legend(loc=4)
#axes.set_xticklabels(label)
for x, y in enumerate(test_accuracy):
    axes.text(x, y, y, ha="left")
for x, y in enumerate(train_accuracy):
    axes.text(x, y, y, ha="left")   
axes.set_xlabel('Classifier')
axes.set_ylabel('Accuracy, %')
axes.set_title('Comparison of Train/Test Accuracy')


###COMPARE TRAIN/TEST DR###
df = pd.DataFrame({'test_dr': test_dr,'train_dr': train_dr}, index=label)
axes = df.plot.line(rot=0, subplots=False, markevery=1, marker='o', markerfacecolor='r')
axes.axhline(y=99.918, color='r', label='Benchmark DR 99.918%', ls='--', lw=2)
axes.legend(loc=4)
#axes.set_xticklabels(label)
for x, y in enumerate(test_dr):
    axes.text(x, y, y, ha="left")
for x, y in enumerate(train_dr):
    axes.text(x, y, y, ha="left") 
axes.set_xlabel('Classifier')
axes.set_ylabel('Detection Rate, %')
axes.set_title('Comparison of Train/Test Detection Rate')

###COMPARE TRAIN/TEST FPR###
df = pd.DataFrame({'test_fpr': test_fpr,'train_fpr': train_fpr},index=label)
axes = df.plot.line(rot=0, subplots=False, markevery=1, marker='o', markerfacecolor='r')
axes.axhline(y=0.012, color='r', label='Benchmark FPR 0.012%', ls='--', lw=2)
axes.legend(loc=1)
#axes.set_xticklabels(label)
for x, y in enumerate(test_fpr):
    axes.text(x, y, y, ha="left")
for x, y in enumerate(train_fpr):
    axes.text(x, y, y, ha="left")
axes.set_xlabel('Classifier')
axes.set_ylabel('False Positive Rate, %')
axes.set_title('Comparison of Train/Test False Positive Rate')

###COMPARE BUILD TIMINGS DR###
df = pd.DataFrame({'time_to_build': time_to_build}, index=label)
axes = df.plot.line(rot=0, subplots=False, markevery=1, marker='o', markerfacecolor='r')
#axes.axhline(y=12073, color='r', label='Benchmark TBM 12073s', ls='--', lw=2)
axes.legend(loc=1)
#axes.set_xticklabels(label)
for x, y in enumerate(time_to_build):
    axes.text(x, y, y, ha="left")
axes.set_xlabel('Classifier')
axes.set_ylabel('Time to build (s)')
axes.set_title('Comparison of Classifier Build times')


#PLOT ROC Curve
plt.figure(1)
lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
for x,y,z,a in zip(roc_fpr, roc_tpr, label,roc_auc):
    #markers_on = [1]
    plt.plot(x,y, markevery=100, marker='o', markerfacecolor='r', label='%s,(area = %0.4f)' %(z,a))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


plt.figure(2)
lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
for x,y,z,a in zip(roc_fpr, roc_tpr, label,roc_auc):
    plt.plot(x,y,label='%s,(area = %0.4f)' %(z,a))
plt.xlim([0.0, 0.2])
plt.ylim([0.96, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate/Detection Rate')
plt.title('Comparison of Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#ROC Table
df_roc = pd.DataFrame({'fpr': fpr2,  'tpr': tpr, 'thresholds': thresholds})
df_roc.iloc[199:220].to_html('roc.html')
df_roc.iloc[199:220].to_csv('roc.csv')

#Summary Table
df_table = pd.DataFrame({'test_accuracy, %': test_accuracy,  'test_dr (tpr), %': test_dr, 'test_fpr, %': test_fpr, 'No of FNs': train_fn, 'time_to_build, (s)': time_to_build}, index=label)


fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=df_table.values, colLabels=df_table.columns, rowLabels=label, loc='center')
plt.show()

df_table.to_html('temp.html')
df_table.to_csv('table.csv')


