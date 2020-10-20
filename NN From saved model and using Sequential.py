"""
Created on Tue Nov 26 18:03:31 2019

@author: Group Q
"""
from sklearn.feature_selection import SelectKBest, chi2
from pandas import read_csv
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, matthews_corrcoef, f1_score, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import time
import pandas as pd
import matplotlib.pyplot as plt

#load train data and split input features from target feature
filename = 'train_imperson_without4n7_balanced_data.csv'
dataframe = read_csv(filename)

array =  dataframe.values

X = array[:,0:152]

y = array[:,-1]
skb = SelectKBest(score_func=chi2, k=15)
skb.fit(X,y)

#load test data and split input features from target feature

filename_test = 'test_imperson_without4n7_balanced_data.csv'
dataframe_test = read_csv(filename_test)

array_test =  dataframe_test.values

X_test = array_test[:,0:152]

y_test = array_test[:,-1]

X_test=skb.transform(X_test)
X=skb.transform(X)


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

#########################################################
#########################################################
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
startTime = time.time()
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
endTime = time.time()
elapsed = endTime-startTime
results_accuracy = []

p = model.predict(X_test)
y_pred = [int(t[0]) for t in numpy.around(p)]
#Calculate accuracy using scorer
startTime = time.time()
acc = accuracy_score(y_test, y_pred)
elapsed2 = endTime-startTime
#Calculate %AUC score using roc_auc_score
probs = model.predict_proba(X_test)
probs = probs[: , 0]
auc = roc_auc_score(y_test, probs)
            
#Calculate confusion matrix and tn, fp etc. 
matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = matrix.ravel()

#calculate the roc curve
fpr2, tpr, thresholds = roc_curve(y_test, probs)
       
#Append results 
#results_accuracy.append(result)
time_to_build.append(round(elapsed,2))
time_to_test.append(elapsed2)
test_accuracy.append(round(acc*100,2))
results_accuracy.append(acc)
roc_auc.append(auc)
train_fn.append(fn)
roc_fpr.append(fpr2)
roc_tpr.append(tpr)
dr = round((tp / (tp + fn))*100, 2)
test_dr.append(dr)
fpr = round((fp / (fp + tn))*100,2)
test_fpr.append(fpr)
#####################################################
p = model.predict(X)
y_pred = [int(t[0]) for t in numpy.around(p)]
#Calculate accuracy using scorer
#Should be the same as "model.score" above - just to check
acc = accuracy_score(y, y_pred)
train_accuracy.append(round(acc*100,2))
#Calculate %AUC score using roc_auc_score
probs = model.predict_proba(X)
probs = probs[: , 0]
auc = roc_auc_score(y, probs)
            
matrix = confusion_matrix(y, y_pred)
tn_train, fp_train, fn_train, tp_train = matrix.ravel()

#calculate the roc curve
#fpr2, tpr, thresholds = roc_curve(y, probs)

dr_train = round((tp_train / (tp_train + fn_train))*100, 2)
train_dr.append(dr_train)
    
fpr_train = round((fp_train / (fp_train + tn_train))*100,2)
train_fpr.append(fpr_train)

#####################################################
#build model using sequential
startTime = time.time()
from numpy.random import seed       
seed(7)
from tensorflow import random
random.set_seed(7)
model = Sequential()
model.add(Dense(15, input_dim=15, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(73, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, y, epochs=387, batch_size=220,verbose = 0)
endTime = time.time()
elapsed = endTime-startTime
####TEST RESULTS####
p = model.predict(X_test)
y_pred = [int(t[0]) for t in numpy.around(p)]
#Calculate accuracy using scorer
startTime = time.time()
acc = accuracy_score(y_test, y_pred)
elapsed2 = endTime-startTime
#Calculate %AUC score using roc_auc_score
probs = model.predict_proba(X_test)
probs = probs[: , 0]
auc = roc_auc_score(y_test, probs)
            
#Calculate confusion matrix and tn, fp etc. 
matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = matrix.ravel()

#calculate the roc curve
fpr2, tpr, thresholds = roc_curve(y_test, probs)
       
#Append results 
#results_accuracy.append(result)
time_to_build.append(round(elapsed,2))
time_to_test.append(elapsed2)
test_accuracy.append(round(acc*100,2))
results_accuracy.append(acc)
roc_auc.append(auc)
train_fn.append(fn)
roc_fpr.append(fpr2)
roc_tpr.append(tpr)
dr = round((tp / (tp + fn))*100, 2)
test_dr.append(dr)
fpr = round((fp / (fp + tn))*100,2)
test_fpr.append(fpr)
#####################################################
p = model.predict(X)
y_pred = [int(t[0]) for t in numpy.around(p)]
#Calculate accuracy using scorer
#Should be the same as "model.score" above - just to check
acc = accuracy_score(y, y_pred)
train_accuracy.append(round(acc*100,2))
#Calculate %AUC score using roc_auc_score
probs = model.predict_proba(X)
probs = probs[: , 0]
auc = roc_auc_score(y, probs)
            
matrix = confusion_matrix(y, y_pred)
tn_train, fp_train, fn_train, tp_train = matrix.ravel()

#calculate the roc curve
#fpr2, tpr, thresholds = roc_curve(y, probs)

dr_train = round((tp_train / (tp_train + fn_train))*100, 2)
train_dr.append(dr_train)
    
fpr_train = round((fp_train / (fp_train + tn_train))*100,2)
train_fpr.append(fpr_train)
#####################################################
#####################################################
###COMPARE TRAIN/TEST ACCURACY###
label = ['NN From Saved Model', 'NN Using Sequential']
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


