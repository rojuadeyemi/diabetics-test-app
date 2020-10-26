# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:00:21 2020

@author: TOSHBA
"""
###LOGISTICS REGRESSION USING STATMODEL AND SKLEARNMODEL
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import researchpy as rs
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as pl
import numpy as np

df= pd.read_csv("C:/Users/Toshiba/Desktop/LIBRARY/Document/Data/diabetes.csv")
df.head()   ### show the variable names
### check the distribution of the variables
sns.heatmap(df.corr(),annot = True)
df.describe()
rs.summary_cont(df)
rs.corr_pair(df)
rs.summary_cat(df['Outcome'])
df.boxplot
sns.pairplot(df, diag_kind = 'kde')
x = df.iloc[:,0:8] ### select the columns 0:8, the independent variables
y = df.iloc[:,8] ### select the columns 8, the dependent variables
test_size = 0.5  ### the percentage of test data
seed = 1 ### for reproducibility
### assign the corresponding values to the appropriate variables
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = test_size, random_state = seed)
column = x_train.columns
#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.fit_transform(x_test)

### generate a no skill data
noskill = [0 for _ in range(len(y_test))]

### Fittin the logistic model
model = LogisticRegression(solver ='lbfgs')
model.fit(x_train,y_train)

## Print the coefficients of the model
coef = pd.DataFrame(model.coef_, columns = column)
coef['Intercept'] = model.intercept_
print(coef)

## Determine the scores of the model against the test data
logd = model.score(x_train,y_train)
logd
logm = model.score(x_test,y_test)
logm

y_predict = model.predict(x_test)
ytr = model.predict(x_train)
metrics.accuracy_score(y_test,y_predict)
print(metrics.classification_report(y_test,y_predict), end = "")

r = model.predict_proba(x_test)   ### better
p = r[:,1]
q = r[:,0]
odd = p/q
log_odd = np.sort(np.log(odd))
t = np.sort(p)
pl.plot(log_odd, t)


#calculate ROC curve and plot (Assuming General baseline)
fpr, tpr, threshold = roc_curve(y_test,p)
noskillfpr, noskilltpr, _ = roc_curve(y_test,noskill)
roc_auc = auc(fpr, tpr) #The auc for roc
pl.plot(noskillfpr, noskilltpr, linestyle = '--', label = 'No Skill Model')
pl.plot(fpr, tpr, marker = '.', label = 'Logistic Model with ROC AUC = %.2f'%roc_auc)
pl.xlabel('False Positive Rate') ## 1- specificity
pl.ylabel('True Positive Rate')  ##sensitivity
pl.legend()   ### Shows the legend
pl.show()  ### Shows the plot

print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
print("Precision:",metrics.precision_score(y_test, y_predict))
print("Recall:",metrics.recall_score(y_test, y_predict))

#calculate Precison-Recall and plot (Assuming specific baseline)
precision, recall, _ = precision_recall_curve(y_test,p)
s = f1_score(y_test, y_predict)
area_uc = auc(recall, precision)
no_skill = len(y_test[y_test == 1])/len(y_test)
pl.plot(noskillfpr, [no_skill, no_skill], linestyle = '--', label = 'No skill model')
pl.plot(recall, precision, marker = '.', lw = 2, label = 'Logistic model with AUC = %.3f'%area_uc)
pl.xlabel('Recall')
pl.ylabel('Precision')
### Shows the legend
pl.legend(loc = 'center left')
pl.show()  ### Shows the plot

# confusion matrixs
a = metrics.confusion_matrix(y_test, y_predict) 

class_names=[0,1] # name  of classes
fig, ax = pl.subplots()
tick_marks = np.arange(len(class_names))
pl.xticks(tick_marks, class_names)
pl.yticks(tick_marks, class_names)
sns.heatmap(a,annot = True)
ax.xaxis.set_label_position("top")
pl.tight_layout()
pl.title('Confusion matrix', y=1.1)
pl.ylabel('Actual label')
pl.xlabel('Predicted label')

### USING STATSMODELS
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics

df= pd.read_csv("C:/Users/TOSHBA/Desktop/LIBRARY/Document/Data/diabetes.csv")
x = sm.add_constant(df.iloc[:,0:8]) ### select the columns 0:8, the independent variables
y = df.iloc[:,8] ### select the columns 8, the dependent variables
test_size = 0.3  ### the percentage of test data
seed = 1 ### for reproducibility

### assign the corresponding values to the appropriate variables
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = test_size, random_state = seed)
scaler = StandardScaler()

mod = sm.Logit(y_train,x_train).fit()
mod.summary()
mod.summary2()

### From the model
y_predict = mod.predict(x_test)
yt = (y_predict>=0.5).astype(int)
metrics.accuracy_score(y_test,yt)
metrics.auc(y_test,yt)
metrics.roc_curve(y_test,yt)
print(metrics.classification_report(y_test,yt), end = "")

### From the Data
y_predictd = mod.predict(x_train)
ytd = (y_predictd>=0.5).astype(int)
metrics.accuracy_score(y_train,ytd)
metrics.auc(y_test,ytd)
metrics.roc_curve(y_test,ytd)
print(metrics.classification_report(y_train,ytd), end = "")

# confusion matrix
a = mod.pred_table() 
sns.heatmap(a,annot = True)
