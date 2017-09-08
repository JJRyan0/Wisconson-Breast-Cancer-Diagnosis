
#Import required libaries
import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning,
                       module="pandas, lineno=570")
from __future__ import print_function
import io
import requests
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
%matplotlib inline
from sklearn import preprocessing
from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot

#-------------------Import csv----------------------------

#Load the data to a pandas data frame.
# print the first five rows of the data frame
df = pd.read_csv("C:\\data\\wisc_bc_data.csv")
df.head()

#Describe the dataset 
df.describe()
#-------------------Data preperation----------------------------

#feature response variable (Y = target) M = Malignant B= Benign 
sns.countplot(x="diagnosis", data=df)
del df['id']#delete the ID as not required
df.tail()#view last few rows of the data set

#Missing Value Detection
print("Number of NA values : {0}".format((df.shape[0] * df.shape[1]) - df.count().sum()))
#----------------------------------------------
# set target variable
XE = df
YE = df.diagnosis

#--------------------One hot encoder--------------------------

#One hot encoder that turns the string labels in to a format for the algorithm
for feature in XE.columns:
    if XE[feature].dtype=='object':
        le = LabelEncoder()
        XE[feature] = le.fit_transform(XE[feature])
XE.tail(3)

#------------------Feature Importance & Ranking-----------------------------

#Feature Importance - selecting only highly prdictive features using random forest Model
from sklearn.ensemble import ExtraTreesClassifier
df.shape
# feature extraction
model = ExtraTreesClassifier(n_estimators = 250, max_features = "auto", random_state=0)
model.fit(XE, YE)
print(model.feature_importances_)

#Ranking the most imporatnt predictive variables potentially build model based on top ranked i.e 1 -16
featureimportance = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(featureimportance)[::-1]
#Print top ranked predictive features
print("Feature ranking:")

for feature in range(XE.shape[1]):
    print("%d. feature %d (%f)" % (feature + 1, indices[feature], featureimportance[indices[feature]]))

#Create Y and X matrix with newly selected features outputted from the previous stage
feature_cols = ['concave points_mean','radius_worst', 'area_worst','symmetry_worst','concavity_mean']
X = df[feature_cols]
y = df.diagnosis

#-----------Uni-variate / Bi-variate Analysis Plots------------

#Creates a scatter plot of variables
sns.FacetGrid(df, hue="diagnosis", size=5) \
   .map(plt.scatter, "radius_worst", "symmetry_worst") \
   .add_legend();
#constructs a violin plots to view outliers
sns.violinplot(x="diagnosis", y="concavity_mean", data =df);

#--------------Cross Validation Train/Test Split--------------------------------------

# Split the data into x and y into training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#-------------Machine Learning: Creating A set of Predictive Models-------------------

#Classification model
# during the fitting process the model is learning the relationship between X_train & y_train
from sklearn.linear_model import LogisticRegression
LRmodel = LogisticRegression()
LRmodel.fit(X_train, y_train)

#make class predictions for the test set
#we pass X_test the feature matrix for the testing set to the predict () method 
#for the fitted model
y_predict_class = LRmodel.predict(X_test)
#outputs a class prediction 1 or 0 for every observation in the testing set which
#is then stored in an object called y_pred_class


#-------------------Evaluate Model Performance-------------------

#-------------------Metric 1: Classification Accurracy: % of correct predictions----------

# calculate accurracy pass y_test and y_predict_class to the accurracy score function
#y_test contains true response values
#for the test set the accuracy score function can tell us what percentage of the predictions in 
#y_pred_class are correct.
from sklearn import metrics
print (metrics.accuracy_score(y_test, y_predict_class))

#-----------------Metric 2: Null Accuracy ----------------------------------

#display the class distribution
y_test.value_counts()

##the below y_test only contains ones and zeros.
#Calcualte the percent of 1's by taking the mean()
y_test.mean()

#calcualte percentage of zero's 
1 - y_test.mean()

#code for null accuracy binary problem zero & one
max(y_test.mean(), 1 - y_test.mean())

#Code for null accuracy multiclass problem 
y_test.value_counts().head(1)/len(y_test)

#Print out true and predicted responses to see a snapshot of some of the errors the model 
#is producing for instance.
print ('True:', y_test.values[0:30])
print ('Pred:', y_predict_class[0:30])

#-------------------Metric 3: Confusion Matrix----------------------
#true values i.e y_test needs to be the first argument otherwise we wont see any errors produced
from sklearn.metrics import confusion_matrix
print (metrics.confusion_matrix(y_test, y_predict_class))

#1st way to compute the claasification accuracy for confusion matrix
confusion = metrics.confusion_matrix(y_test,y_predict_class)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]

#classifiaction accuracy score
(TP + TN)/float(TP + TN + FP + FN)

#2nd way to compute the claasification accuracy for confusion matrix
from sklearn import metrics
print (metrics.accuracy_score(y_test, y_predict_class))

#------------------------Metric 4: Mis - Classification Rate---------------------------

# add the FP + FN / float(total)
1 - metrics.accuracy_score(y_test, y_predict_class)

#-----------------------Metric 5: Sensitivity (recall) & Specificity---------------

#Sensitivity (recall) TP / float(TP + FN)
metrics.recall_score(y_test, y_predict_class)

#Specificity need to maximise TN/ total
TN /float(TN+ FP)
#-----------------------Metric 6: False Positive Rate-------------------
#Calculate the False Positive rate
FP / float (TN + FP)

#-----------------------Metric 6: Precision-------------------

#precision TP/float(TP + FP)
metrics.precision_score(y_test, y_predict_class)

#-----------------------Improving Performance: Changing the Classification Threshold--------
#Print first 10 predictions of the logistic regression model
LRmodel.predict(X_test)[0:10]

#--------

#Print the first 10 prediction probabilities for class memebership
#left colunm shows predicted probably of class zero
#Right shows "" "" of class one
LRmodel.predict_proba(X_test)[0:10, :]

#predicted probabilties of class 1
y_predicted_prob = LRmodel.predict_proba(X_test)[:,1]
y_predicted_prob

##Analysis the distribution of class distribution
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.size']=12

plt.hist(y_predict_class, bins= 10)
plt.xlim(0,1)
plt.title('Histogram of Predicted Classes')
plt.xlabel('Predicted Probability of Breast Cancer')
plt.ylabel('Frequency')

#Analysis the distribution of a numerical variable
plt.hist(y_predicted_prob, bins= 10)
plt.xlim(0,1)
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability of Breast Cancer')
plt.ylabel('Frequency')

#Adjusting the Classification to Threshold (0.3) to Improve Model performance

#alter the default predict probabilty to 0.3
from sklearn.preprocessing import binarize
y_pred_class = binarize(y_predicted_prob, 0.3)[0]

#print the first 10 predicted probabilities with lower threshold
y_predicted_prob[0:10]

#print the first 10 predicted classes with lower threshold
y_pred_class[0:10]

#Confusion Matrix at Default Threshold 0.5

#previous confusion matrix before threshold change (default 0.5)
confusion
#new confusion matrix (threshold 0.3)
metrics.confusion_matrix(y_test, y_pred_class)

#Sensitivity (recall) TP / float(TP + FN) has gone from 90% previous to 96%
metrics.recall_score(y_test, y_pred_class)

#Specificity TN /float(TN+ FP) has decreased from 94% to 91% as oberservations moved 
#from left to right colunm meaning FP will increase and TN will decrease. 
51/ float(51 + 5)

#Analysis the distribution of new class distribution
plt.hist(y_pred_class, bins= 10)
plt.xlim(0,1)
plt.title('Histogram of Predicted Classes')
plt.xlabel('Predicted class of Breast Cancer diagnosis')
plt.ylabel('Frequency')

#Area Under AUC Curve
#The ROC curve helps to select a threshold value that balances sensitivity and specificity in a way that makes sense.
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predicted_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC Curve for Cancer Prediction')
plt.xlabel('False Positive Rate(1 - specificity)')
plt.ylabel('True Positive Rate(sensitivity)')
plt.grid(True)