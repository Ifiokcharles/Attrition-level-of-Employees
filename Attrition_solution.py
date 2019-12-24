# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:12:07 2019

@author: charles pc
"""

#iMPORTING THE LIBARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.offline as py

import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# IMPORTING THE DATASET
dataset = pd.read_csv('Employee Test data No. 1.csv')

#REMOVE UNIMPORTANT ATTRIBUTES
dataset = dataset.drop(['Over18','StandardHours','EmployeeNumber','EmployeeCount' ],1,errors='ignore')


# CHECKING FOR ANY MISSING VALUES
dataset.isnull().sum()

#SHOWING THE NUMBER OF CATEGORICAL ATTRIBUTES
number_of_attributes = dataset.select_dtypes(include=['object']).head()
y = dataset.iloc[:, 1]

#GRAPHS

#create dummy variables using onehotencoder
X_heat = dataset.drop(['Over18','StandardHours','EmployeeNumber','EmployeeCount' ],1,errors='ignore')
labelencoder_X_heat = LabelEncoder()
encoder_names_heat = ['Attrition']
for i in encoder_names_heat:
    X_heat[i] = labelencoder_X_heat.fit_transform(X_heat[i])




#CORROLATION MAP
#Correlation Matrix
sns.set(style="white")
corr_heat = X_heat.corr()

mask_heat = np.zeros_like(corr_heat, dtype=np.bool)
mask_heat[np.triu_indices_from(mask_heat)] = True

f, ax = plt.subplots(figsize=(5, 4))

cmap = sns.diverging_palette(15, 220, as_cmap=True)

sns.heatmap(corr_heat, mask=mask_heat, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)



#Attrition to Jobsatisfation
sns.set(style="white")
f, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=X_heat.JobSatisfaction,y=dataset.Attrition,orient="h", ax=ax)



# Set up the matplotlib figure
f, axes = plt.subplots(7, 3, figsize=(9,7))
sns.despine(left=True)

#people that left
leavers = X_heat.loc[X_heat['Attrition'] == 1]

# Plot a simple histogram with binsize determined automatically
sns.distplot(leavers['Age'], kde=False, color="b", ax=axes[0,0])
sns.distplot(leavers['DailyRate'], bins=3, kde=False, color="b", ax=axes[0, 1])
sns.distplot(leavers['DistanceFromHome'], kde=False, color="b", ax=axes[0, 2])
sns.distplot(leavers['HourlyRate'], kde=False, bins=5, color="b", ax=axes[1, 0])
sns.distplot(leavers['MonthlyIncome'], bins=10,kde=False, color="b", ax=axes[1, 1])
sns.distplot(leavers['MonthlyRate'], bins=10,kde=False, color="b", ax=axes[1, 2])
sns.distplot(leavers['NumCompaniesWorked'], bins=10,kde=False, color="b", ax=axes[2, 0])
sns.distplot(leavers['TotalWorkingYears'], bins=10,kde=False, color="b", ax=axes[2, 1])
sns.distplot(leavers['YearsAtCompany'], bins=10,kde=False, color="b", ax=axes[2, 2])
sns.distplot(leavers['Education'], bins=10,kde=False, color="b", ax=axes[3, 0])
sns.distplot(leavers['EnvironmentSatisfaction'], bins=10,kde=False, color="b", ax=axes[3, 1])
sns.distplot(leavers['JobInvolvement'], bins=10,kde=False, color="b", ax=axes[3, 2])
sns.distplot(leavers['JobLevel'], bins=10,kde=False, color="b", ax=axes[4, 0])
sns.distplot(leavers['JobSatisfaction'], bins=10,kde=False, color="b", ax=axes[4, 1])
sns.distplot(leavers['PercentSalaryHike'], bins=10,kde=False, color="b", ax=axes[4, 2])
sns.distplot(leavers['PerformanceRating'], bins=10,kde=False, color="b", ax=axes[5, 0])
sns.distplot(leavers['RelationshipSatisfaction'], bins=10,kde=False, color="b", ax=axes[5, 1])
sns.distplot(leavers['StockOptionLevel'], bins=10,kde=False, color="b", ax=axes[5,2])
sns.distplot(leavers['TrainingTimesLastYear'], bins=10,kde=False, color="b", ax=axes[6, 0])
sns.distplot(leavers['WorkLifeBalance'], bins=10,kde=False, color="b", ax=axes[6, 1])
sns.distplot(leavers['YearsInCurrentRole'], bins=10,kde=False, color="b", ax=axes[6, 2])
sns.distplot(leavers['YearsSinceLastPromotion'], bins=10,kde=False, color="b", ax=axes[7, 0])
sns.distplot(leavers['YearsWithCurrManager'], bins=10,kde=False, color="b", ax=axes[7, 1])

plt.tight_layout()

# Set up the matplotlib figure
f, axes = plt.subplots(figsize=(9,7))
sns.despine(left=True)

#people that left
leavers = X_heat.loc[X_heat['Attrition'] == 1]
sns.distplot(leavers['MonthlyRate'], bins=10,kde=False, color="b")
plt.tight_layout()


# Isolate Data, class labels and column values
#create dummy variables using onehotencoder
X_Important = dataset.drop('Attrition',1,errors='ignore')
look = X_Important.describe()
y = dataset.iloc[:, 1]
labelencoder_X_Important = LabelEncoder()
encoder_names_important = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 
                      'Gender', 'JobInvolvement','JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'OverTime', 
                      'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']
for i in encoder_names_important:
    X_Important[i] = labelencoder_X_Important.fit_transform(X_Important[i])
dataset.Attrition.value_counts()
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y) 

# Build the model
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

# Fit the model
rfc.fit(X_Important, y)

FI_drop = pd.Series(rfc.feature_importances_, index=X_Important.columns)
FI_drop.nlargest(10).plot(kind='barh')
plt.show()

important_features = rfc.feature_importances_
dataframe_coeff = pd.DataFrame(columns=['Feature', 'Coefficient'])
for i in range(30):
    features = X_Important.columns[i]
    coefficient = important_features[i]
    dataframe_coeff.loc[i] = (features, coefficient)
dataframe_coeff.sort_values(by='Coefficient', ascending=False, inplace=True)
dataframe_coeff = dataframe_coeff.reset_index(drop=True)
dataframe_coeff.head(10)



pd.crosstab(dataset.DistanceFromHome,dataset.Attrition).plot(kind='bar')
plt.title('Attrition with respect to DistanceFromHome')
plt.xlabel('DistanceFromHome')
plt.ylabel('Frequency of Attrition')

pd.crosstab(dataset.OverTime,dataset.Attrition).plot(kind='bar')
plt.title('Attrition with respect to Overtime')
plt.xlabel('Overtime')
plt.ylabel('Frequency of Attrition')

pd.crosstab(dataset.TotalWorkingYears,dataset.Attrition).plot(kind='bar')
plt.title('Attrition with respect to TotalWorkingYears')
plt.xlabel('TotalWorkingYears')
plt.ylabel('Frequency of Attrition')


pd.crosstab(dataset.NumCompaniesWorked,dataset.Attrition).plot(kind='bar')
plt.title('Attrition with respect to NumCompaniesWorked')
plt.xlabel('NumCompaniesWorked')
plt.ylabel('Frequency of Attrition')

pd.crosstab(dataset.YearsAtCompany,dataset.Attrition).plot(kind='bar')
plt.title('Attrition with respect to YearsAtCompany')
plt.xlabel('YearsAtCompany')
plt.ylabel('Frequency of Attrition')


#all key employees
key_employees = X_heat.loc[X_heat['MonthlyIncome'] >=2500].loc[X_heat['Age'] >= 30]
Key_show = key_employees.describe()
#lost key employees
lost_key_employees = key_employees.loc[X_heat['Attrition']==1]
lost_key_employees.describe()
print("Number of key employees: ", len(key_employees))
print ("Number of lost key employees: ", len(lost_key_employees))
print ("Percentage of lost key employees: ", round((float(len(lost_key_employees))/float(len(key_employees))*100),2),"%")







#filter out people with a good last evaluation
leaving_Income = leavers.loc[leavers['MonthlyIncome'] >=2500]
leaving_Income = leaving_Income.drop('Attrition',1,errors='ignore')

sns.set(style="white")

# Compute the correlation matrix
corr = leaving_Income.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5, 4))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)








#filter out people with a good last evaluation
leaving_Age = leavers.loc[leavers['Age'] >= 30]
leaving_Age = leaving_Age.drop('Attrition',1,errors='ignore')
sns.set(style="white")

# Compute the correlation matrix
corr = leaving_Age.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(5, 4))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)






X = dataset.drop('Attrition',1,errors='ignore')
labelencoder_X = LabelEncoder()
encoder_names = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 
                      'Gender', 'JobInvolvement','JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'OverTime', 
                      'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']
for i in encoder_names:
    X[i] = labelencoder_X.fit_transform(X[i])
dataset.Attrition.value_counts()


 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X= sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)






#MODELLING

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)


param_grid_DT = {'min_samples_split':[2,4,6,8,10],
              'min_samples_leaf': [1, 2, 3, 4],
              'max_depth': [5, 10, 15, 20, 25]}

grid_DT = GridSearchCV(estimator = classifier_DT,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid_DT,
                        scoring='roc_auc',
                        cv=10)

grid_DT.fit(X_train, y_train)
log_DT = grid_DT.best_estimator_

# Predicting the Test set results
y_pred_DT = grid_DT.predict(X_test)
print('='*20)
print("best params: " + str(grid_DT.best_estimator_))
print("best params: " + str(grid_DT.best_params_))
print('best score:', grid_DT.best_score_)
print('='*20)

# Classification report for the optimised Log Regression
log_DT.fit(X_train, y_train)
print(classification_report(y_test, log_DT.predict(X_test)))

#AUC score
log_DT.fit(X_train, y_train) 
probs_DT = log_DT.predict_proba(X_test) 
probs_DT = probs_DT[:, 1] 
logit_roc_auc_DT = roc_auc_score(y_test, probs_DT) 
print('AUC score: %.3f' % logit_roc_auc_DT)

#making a confusion matrix
from sklearn.metrics import confusion_matrix
CM_DT = confusion_matrix(y_test, y_pred_DT)
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(CM_DT), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')




#RANDOMFORESTCLASSIFIER
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
Important_features_drop = RandomForestClassifier(class_weight = "balanced",
                                       random_state=7)


param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175],
              'min_samples_split':[2,4,6,8,10],
              'min_samples_leaf': [1, 2, 3, 4],
              'max_depth': [5, 10, 15, 20, 25]}

grid_RF = GridSearchCV(estimator = Important_features_drop,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=10)
grid_RF = grid_RF.fit(X_train,y_train)
log_RF = grid_RF.best_estimator_

#predicting test sets results
y_pred_RF = grid_RF.predict(X_test)
print('='*20)
print("best params: " + str(grid_RF.best_estimator_))
print("best params: " + str(grid_RF.best_params_))
print('best score:', grid_RF.best_score_)
print('='*20)

#making a confusion matrix
from sklearn.metrics import confusion_matrix
CM_RF = confusion_matrix(y_test, y_pred_RF)
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(CM_RF), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Classification report for the optimised Log Regression
log_RF.fit(X_train, y_train)
print(classification_report(y_test, log_RF.predict(X_test)))

#AUC score
log_RF.fit(X_train, y_train) 
probs_RF = log_RF.predict_proba(X_test) 
probs_RF = probs_RF[:, 1] 
is_leave = y == 1
logit_roc_auc_RF = roc_auc_score(y_test, probs_RF) 
print('AUC score: %.3f' % logit_roc_auc_RF)

#create a dataframe containing prob values
pred_prob_df = pd.DataFrame(probs_RF)
pred_prob_df.columns = ['prob_leaving']
X_test_prob = pd.DataFrame(data=X_test).reset_index(drop=True)
y_test_prob = pd.DataFrame(data=y_test)
y_test_prob.columns = ['Attrition']

#merge dataframes to get the name of employees
all_employees_pred_prob_df = pd.concat([X_test_prob, y_test_prob, pred_prob_df], axis=1)


#Based on the most important features, we can predict which employees will leave 
High_income_employees_will_leave_df = all_employees_pred_prob_df[(all_employees_pred_prob_df["MonthlyIncome"] >=2500)]


#Based on the most important features, we can predict which employees will leave next.
High_income_employees_still_working_df = all_employees_pred_prob_df[(all_employees_pred_prob_df["Attrition"] == 0  ) & 
                                                            (all_employees_pred_prob_df["MonthlyIncome"] >=2500)]


High_income_employees_still_working_df.sort_values(by='prob_leaving', ascending=False, inplace=True)







#LOGISTICREGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


classifier_LR = LogisticRegression(random_state = 0)
param_grid = {'C': np.arange(1e-03, 2, 0.01)}

grid_LR = GridSearchCV(estimator =classifier_LR,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=10)
grid_LR = grid_LR.fit(X_train, y_train)
log_LR = grid_LR.best_estimator_

#predicting test sets results
y_pred_LR = log_LR.predict(X_test)
print('='*20)
print("best params: " + str(grid_LR.best_estimator_))
print("best params: " + str(grid_LR.best_params_))
print('best score:', grid_LR.best_score_)
print('='*20)

#making a confusion matrix
from sklearn.metrics import confusion_matrix
CM_LR = confusion_matrix(y_test, y_pred_LR)
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(CM_LR), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Classification report for the optimised Log Regression
log_LR.fit(X_train, y_train)
print(classification_report(y_test, log_LR.predict(X_test)))

#AUC score
log_LR.fit(X_train, y_train) 
probs_LR = log_LR.predict_proba(X_test) 
probs_LR = probs_LR[:, 1] 
logit_roc_auc_LR = roc_auc_score(y_test, probs_LR) 
print('AUC score: %.3f' % logit_roc_auc_LR)


# Create ROC Graph
from sklearn.metrics import roc_curve
LR_fpr, LR_tpr, LR_thresholds = roc_curve(y_test, log_LR.predict_proba(X_test)[:,1])
RF_fpr, RF_tpr, RF_thresholds = roc_curve(y_test, log_RF.predict_proba(X_test)[:,1])
DT_fpr, DT_tpr, DT_thresholds = roc_curve(y_test, log_DT.predict_proba(X_test)[:,1])
plt.figure(figsize=(14, 6))

# Plot Logistic Regression ROC
plt.plot(LR_fpr, LR_tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc_LR)
# Plot Random Forest ROC
plt.plot(RF_fpr, RF_tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc_RF)

plt.plot(DT_fpr, DT_tpr, label='Decision Tree (area = %0.2f)' % logit_roc_auc_DT)
# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()



