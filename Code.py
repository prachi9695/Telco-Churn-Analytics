import pandas as pd
import numpy as np
#For visualization
import seaborn as sns
import matplotlib.pyplot as plt


'''Getting the Telco Customer Churn Data'''

telco_df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

'''Data Preprocessing, Cleansing and Exploratory Data Analysis'''
telco_df.columns
telco_df.info()

'''Converting 'TotalCharges' from object to numeric column'''

telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'], errors='coerce')


'''Dropping Unnecesary columns'''

telco_df.drop(['customerID'], axis=1, inplace=True)


'''Checking for Missing Values and handling them'''

telco_df.isnull().sum()

telco_df.dropna(inplace = True)
telco_df.isnull().sum()

'''Checking for unique columns and choosing to drop more redundant columns'''

#Columns to cols
cols = telco_df.columns
cols

# for each column
for col in cols:
    print(col)

    # get a list of unique values
    unique = telco_df[col].unique()
    print(unique, '\n*************************************\n\n')

# get number of unique values if greater than 30
for col in cols:
    print(col)

    # get a list of unique values
    unique = telco_df[col].unique()

    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
    if len(unique)<30:
        print(unique, '\n====================================\n\n')
    else:
        print(str(len(unique)) + ' unique values', '\n====================================\n\n')


'''Summary Statistics for each column'''

#1. All numeric columns
telco_df.describe()

#2. Object columns
telco_df.describe(include='object')

### Frequency distribution for each value of a column

for col in cols:
    print(col)
    value_counts = telco_df[col].value_counts()
    print(value_counts, '\n\n=======================================================')

### Checking for duplicate rows and deleting them

telco_df.duplicated()

telco_df.duplicated().sum()

telco_df.drop_duplicates(inplace=True)

#Rechecking for duplicates
telco_df.duplicated().sum()

### Checking for Outliers
telco_df.boxplot(figsize=(12,6))
plt.show()

#EDA
#Churn count
churncount = telco_df['Churn'].value_counts()
plt.figure(figsize=(3, 5))
my_palette = ['#FF0000', '#008000']
plt.pie(churncount,colors = my_palette, labels = churncount.index, autopct= '%1.1f%%');
plt.title('Percentage Churn')

# Calculate churn counts by gender
gender_churn_count = telco_df.groupby(['gender', 'Churn'])['Churn'].count().unstack()

# Create bar chart
sns.countplot(data=telco_df, x=telco_df['gender'], hue='Churn')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right')
plt.title('Churn counts by gender')
plt.show()

#Contract vs Churn
sns.countplot(data=telco_df, x=telco_df['Contract'], hue='Churn')


#Services vs Churn
#Phone service churn
sns.countplot(data=telco_df, x=telco_df['PhoneService'], hue='Churn')

# Define the services to plot with Phone Service
services = ['PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

fig, axs = plt.subplots(3, 3, figsize=(18, 12))

axs = axs.flatten()

# Loop through the services and create a bar chart for each service
for i, service in enumerate(services):
    # countplot for the service against the Churn column
    sns.countplot(x=service, hue='Churn', data=telco_df, ax=axs[i], palette="deep")
    axs[i].set_xlabel(service)
    axs[i].set_ylabel('Count')
    axs[i].legend(title='Churn', loc='upper right')

fig.suptitle('Bar charts for services against Churn')

fig.tight_layout()
plt.show()

##Cheking for the demographics for users churned with phone service
phoneservice_churn = telco_df[(telco_df['PhoneService']=='Yes') & (telco_df['Churn']=='Yes')]
dems=['gender','SeniorCitizen', 'Partner','Dependents']

fig, axs = plt.subplots(2, 2, figsize=(15, 12))

axs = axs.flatten()

# Loop through the Demographics and create a bar chart for each Demogrphic
for i, dem in enumerate(dems):
    # countplot for the service against the Churn column
    sns.countplot(x=dem, hue='Churn', data=phoneservice_churn, ax=axs[i], palette="deep")
    axs[i].set_xlabel(dem)
    axs[i].set_ylabel('Count')
    axs[i].legend(title='Churn', loc='upper right')

fig.suptitle('Bar charts for Churn of User Demographics with Phone Service')

fig.tight_layout()
plt.show()

'''Getting Dummy Variables, Dealing with them and checking Correlation'''
telco_df['Churn'] = np.where(telco_df['Churn'] == 'Yes', 1 ,0)
telco_df=pd.get_dummies(telco_df,drop_first=False)
telco_df.isnull().sum() #No Null values
telco_df.duplicated().sum() #No Duplicates

#Correlation Matrix
corr_matrix = telco_df.corr()
print(corr_matrix)

corr_matrix.to_excel('correlation_matrix.xlsx', index=True)
telco_df.to_excel('Telco.xlsx', index=True)
#Visualizing it
plt.figure(figsize=(22,14))
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Correlation Matrix', fontsize=20)
plt.show()
 

'''Applying Machine Learning Techniques to Find the best Predictor and
the best suitable Churn Prediction Model'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score

#Splitting Test and Train Data
X = telco_df.drop(['Churn'], axis=1)
y = telco_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
#Random state 0.2 as the churn ratio is 2.76 so taking a 80/20 split

'''~~~~~~~~~~~~~~~~~~~~~~~logistic regression~~~~~~~~~~~~~~~~~~~~~~~~~'''
lr = LogisticRegression()
lr.fit(X_train, y_train)

coef = lr.coef_[0]

# Rank the features based on their absolute coefficient value
importance = sorted(zip(X.columns, abs(coef)), key=lambda x: x[1], reverse=True)

# Print the feature names and their corresponding importance scores
for feat, score in importance:
    print(feat, score)

# make predictions on test data
y_pred = lr.predict(X_test)
y_pred

# evaluate performance of model
LR_AccuracyScore = accuracy_score(y_test, y_pred)
LR_F1Score = f1_score(y_test, y_pred)

print("LR Accuracy: ", LR_AccuracyScore) #0.82
print("LR F1 Score: ", LR_F1Score) #0.59

#Improving the model
# performing feature selection using SelectKBest with f_classif as the scoring function
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=40)
selector.fit(X, y)

# get the indices of the selected features
selected_features_indices = selector.get_support(indices=True)

# select the top 10 features from the original data
X_selected = X.iloc[:, selected_features_indices]
X_selected
# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# train a logistic regression model on the selected features
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# make predictions on test data
y_pred = lr.predict(X_test)

# evaluate performance of model
LR_AccuracyScore = accuracy_score(y_test, y_pred)
LR_F1Score = f1_score(y_test, y_pred)

print("LR K Accuracy: ", LR_AccuracyScore) #0.82
print("LR K F1 Score: ", LR_F1Score) #0.59

# Apply SMOTE oversampling to balance the data
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)
# Fit logistic regression model to the resampled data
lr = LogisticRegression(random_state=42, max_iter=2000)
lr.fit(xr_train, yr_train)

# Predict using the test set
y_pred = lr.predict(X_test)

# Evaluate the model
print("SM OS Accuracy: ", accuracy_score(y_test, y_pred)) #0.8145506419400856
print("SM OS F1 Score: ", f1_score(y_test, y_pred)) #0.6036585365853658

#Applying Hyper Parameter Tuning
from sklearn.model_selection import GridSearchCV

# define the hyperparameter grid to search over
param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'None'],
              'C': [0.01, 0.1, 1, 10, 100],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [100, 500, 1000, 5000]}

# Initiate the logistic regression model
lr = LogisticRegression(max_iter=5000)

# create the GridSearchCV object
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1')

# fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train) #Takes around 3-5 minutes

# get the best parameters and best score from the GridSearchCV object
best_params = grid_search.best_params_
#{'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg'}

best_score = grid_search.best_score_
#0.60

# fit the logistic regression model using the best parameters
best_lr = LogisticRegression(**best_params)
best_lr.fit(X_train, y_train)

# make predictions on test data using the best model
y_pred = best_lr.predict(X_test)

# evaluate performance of best model
best_LR_AccuracyScore = accuracy_score(y_test, y_pred)
best_LR_F1Score = f1_score(y_test, y_pred)

print("Best Parameters: ", best_params) #{'C': 0.1, 'max_iter': 1000, 'penalty': 'l1', 'solver': 'liblinear'}
print("Best F1 Score: ", best_score) #0.604421328116541
print("Best Accuracy: ", best_LR_AccuracyScore) #0.8252496433666191
print("Best F1 Score: ", best_LR_F1Score) #0.5977011494252873

'''Best Parameters:  {'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg'}
Best F1 Score:  0.6042723901956395
Best Accuracy:  0.8245363766048502
Best F1 Score:  0.5980392156862745'''

'''~~~~~~~~~~~~~~~~~~~~~~~Random Forest~~~~~~~~~~~~~~~~~~~~~~~~~'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
x = telco_df.drop(['Churn'], axis=1)
y = telco_df['Churn']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

rfc = RandomForestClassifier(random_state= 1, n_estimators = 200 )
rfc.fit(x_train, y_train)
y_pred_test = rfc.predict(x_test)
f1_score(y_test, y_pred_test)   

accuracy = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print("RF Accuracy: ", accuracy)
print("RF F1 Score: ", f1)

'''RF Accuracy:  0.7817403708987162
RF F1 Score:  0.521875'''

#Improving the model
# perform feature selection using SelectKBest with f_classif as the scoring function
selector = SelectKBest(f_classif, k=30)
selector.fit(x, y)

# get the indices of the selected features
selected_features_indices = selector.get_support(indices=True)

# select the top 10 features from the original data
X_selected = x.iloc[:, selected_features_indices]
X_selected
# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# train a random forest model on the selected features
rfc = RandomForestClassifier(random_state= 42, n_estimators = 200 )
rfc.fit(X_train, y_train)

# make predictions on test data
y_pred = rfc.predict(X_test)

# evaluate performance of model
RF_AccuracyScore = accuracy_score(y_test, y_pred)
RF_F1Score = f1_score(y_test, y_pred)

print("RF K Accuracy: ", RF_AccuracyScore) #0.80
print("RF K F1 Score: ", RF_F1Score) #0.54

'''RF K Accuracy:  0.8017118402282454
RF K F1 Score:  0.5442622950819672'''

# Apply SMOTE oversampling to balance the data
# split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# apply SMOTE oversampling to the training data
smote = SMOTE(random_state=1)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# train a random forest model on the resampled data
rfc = RandomForestClassifier(random_state=1, n_estimators=200)
rfc.fit(x_train_resampled, y_train_resampled)

# make predictions on test data
y_pred_test = rfc.predict(x_test)

# evaluate performance of best model
best_RF_AccuracyScore = accuracy_score(y_test, y_pred)
best_RF_F1Score = f1_score(y_test, y_pred)

print("Best Accuracy: ", best_RF_AccuracyScore) #0.644793152639087
print("Best F1 Score: ", best_RF_F1Score) #0.22670807453416147

'''~~~~~~~~~~~~~~~~~~~~~~~Decision tree~~~~~~~~~~~~~~~~~~~~~~~~~'''

from sklearn.tree import DecisionTreeClassifier
x = telco_df.drop(['Churn'], axis=1)
y = telco_df['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=1)
dt = DecisionTreeClassifier(random_state=1)

dt.fit(x_train, y_train)
y_pred_test = dt.predict(x_test)
f1_score(y_test, y_pred_test)    

accuracy = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print("Accuracy: ", accuracy)
print("F1 Score: ", f1)

'''Accuracy:  0.7289586305278174
F1 Score:  0.48787061994609165'''

#Improving the model
# perform feature selection using SelectKBest with f_classif as the scoring function
selector = SelectKBest(f_classif, k=30)
selector.fit(x, y)

# get the indices of the selected features
selected_features_indices = selector.get_support(indices=True)

# select the top 10 features from the original data
X_selected = x.iloc[:, selected_features_indices]
X_selected
# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# train a Decision tree model on the selected features
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# make predictions on test data
y_pred = dt.predict(X_test)

# evaluate performance of model
DT_AccuracyScore = accuracy_score(y_test, y_pred)
DT_F1Score = f1_score(y_test, y_pred)

print("DT K Accuracy: ", DT_AccuracyScore) #0.74
print("DT K F1 Score: ", DT_F1Score) #0.50

# Apply SMOTE oversampling to balance the data
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
xr_train,xr_test,yr_train,yr_test=train_test_split(X_resampled, y_resampled,test_size=0.2)

# Fit Decision tree model to the resampled data
dt = DecisionTreeClassifier(random_state=42)
dt.fit(xr_train, yr_train)

# Predict using the test set
y_pred = dt.predict(X_test)

# Evaluate the model
print("SM OS Accuracy: ", accuracy_score(y_test, y_pred)) #0.724679029957204
print("SM OS F1 Score: ", f1_score(y_test, y_pred)) #0.4623955431754874

'''Quantifying the importance of each independent variable in predicting 
    Churn using Logistic Regression Model'''
X = telco_df.drop(['Churn'], axis=1)
y = telco_df['Churn']

# fit logistic regression model
lr = LogisticRegression()
lr.fit(X, y)

# create table of coefficients
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': lr.coef_[0]
})

coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1)
print(coefficients)

imp_features = coefficients.sort_values('Coefficient', ascending=False).iloc[:10, :]

# Plot bar chart of top 10 coefficients
top_10 = imp_features[:10]
plt.barh(top_10['Variable'], top_10['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Important Churn Predictor Features')
plt.show()
