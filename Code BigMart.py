# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 02:51:39 2018

@author: Amol
"""

import seaborn as sb
import matplotlib.pyplot as plt
#%matplotlib inlin
import pandas as pd
import numpy as np


# reading the files
# Read files TRAIN AND TEST :
path2="G:/Only Python/Python Project/Project 2/train.csv"
train = pd.read_csv(path2)
path3="G:/Only Python/Python Project/Project 2/test.csv"
test = pd.read_csv(path3)

train.head()
test.head()
# check for shape
print(train.shape)
print(test.shape)

# knowing the statistical values of the data
train.describe()
test.describe()

# datatypes of the features
train.dtypes
test.dtypes

# checking the null values
train.isnull().sum()
test.isnull().sum()

# checking the number of unique values each columns
train.apply(lambda x : len(x.unique()))
test.apply(lambda x : len(x.unique()))

#---------------------------------------------------------------------------------------
# Combine Datasets
train['Train/Test'] = 'train'
test['Train/Test'] = 'test'
data = pd.concat([train,test], ignore_index = True)
data.shape
data.head()
print (train.shape, test.shape, data.shape)

#for getting the mean ,std dev,max
data.describe()

# checking the null values
data.isnull().sum()

#for getting the unique values non repeated values
data.apply(lambda x: len(x.unique()))

# check for NULLS, blanks and zeroes
# -------------------------------
cols = list(data.columns)
type(cols)
cols.remove("Item_Outlet_Sales")
print(cols)

for c in cols:
    if (len(data[c][data[c].isnull()])) > 0:
        print("WARNING: Column '{}' has NULL values".format(c))

    if (len(data[c][data[c] == 0])) > 0:
        print("WARNING: Column '{}' has value = 0".format(c))
        
data.head()

#######################################################################
#######################################################################

#############################################################################
# scatter plot matrix
sb.pairplot(train, hue='Outlet_Size', palette=['black', 'orange', 'blue'])
plt.plot()

# checking the counts of the outlets store with respect to their location
sb.countplot(data=train, x=train.Outlet_Type, hue=train.Outlet_Location_Type)

######
sb.countplot(data=train, x=train.Outlet_Size, hue=train.Outlet_Location_Type)

####Size/fact content
sb.countplot(data=train, x=train.Outlet_Size, hue=train.Item_Fat_Content)


##############################################################################

sb.countplot(data=train, x=train.Outlet_Size)
sb.countplot(data=train, x=train.Item_Fat_Content)
sb.countplot(data=train, x=train.Outlet_Location_Type)
sb.countplot(data=train, x=train.Outlet_Type)

##############################################################################

# let's find the correlation between the features
cor = train.corr()
sb.heatmap(cor, vmax=0.10, square=True, annot=True)

# clearer view. removed the 1st row as it contains same info (total records)
# ------------------------------------------------------------
desc = data.describe()
desc = desc.drop(desc.index[0])
desc

# describe the dataset (R,C)
# --------------------------
data.dtypes
#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object'] #shows the data which is object in nature in x 

#Exclude ID cols and Train/Test:
categorical_columns = [x for x in categorical_columns if x not in 
                       ['Item_Identifier','Outlet_Identifier','Train/Test']]

#Print frequency of categories each column
for col in categorical_columns:
    print ('\nFrequency of Categories for varible %s'%col)
    print (data[col].value_counts())

#Determine the average weight per item:
item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
item_avg_weight  
#Get a boolean variable specifying missing Item_Weight values
miss_Weight = data['Item_Weight'].isnull()  
miss_Weight # print all values (True and False)
print ('Orignal #missing: %d'% sum(miss_Weight))   #2439  missing values
                                                  
data.loc[miss_Weight,'Item_Weight'] = data.loc[miss_Weight,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
print ('Final #missing: %d'% sum(data['Item_Weight'].isnull())) # 0 missing values
     
###############################################################################
       
#Import mode function:
from scipy.stats import mode
#Determing the mode for each
outlet_size_mode = data.pivot_table(values='Outlet_Size',
                                    columns='Outlet_Type',
                                    aggfunc=lambda x:x.mode().iat[0] )
print ('Mode for each Outlet_Type:')
print (outlet_size_mode)
#Get a boolean variable specifying missing size values
miss_Size= data['Outlet_Size'].isnull()  #True/False
miss_Size
#Impute data and check missing  and sum
print ('\nOrignal missing: %d'% sum(miss_Size))
#all replace values put
data.loc[miss_Size,'Outlet_Size'] = data.loc[miss_Size,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
data.loc[miss_Size,'Outlet_Size'] #print(only replace values)
print (sum(data['Outlet_Size'].isnull()))   # missing values


# Consider combining Outlet_Type
data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type')
###############################################################################

#Determine average visibility of a product
visibility_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')
#Impute 0 values with mean visibility of that product:
miss_Visibility = (data['Item_Visibility'] == 0)
miss_Visibility
print ('Number of 0 values initially: %d'%sum(miss_Visibility))
data.loc[miss_Visibility,'Item_Visibility'] = data.loc[miss_Visibility,'Item_Identifier'].apply(lambda x:visibility_avg.loc[x])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))
                 
data.shape    #(14204, 13)
###############################################################################
# Create a broad category of Type of Item
#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2]) # 1st colume in 1st 2 word
data['Item_Type_Combined'] 
#(14204, 14)          # (14204, 14) #New column(Item_Type_Combined)
                                               
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable','DR':'Drinks'})
data['Item_Type_Combined']                                                            

#sum of each type
data['Item_Type_Combined'].value_counts()


###############################################################################
#Change categories of low fat:
print ('Original Categories:')
print (data['Item_Fat_Content'].value_counts())

print ('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print (data['Item_Fat_Content'].value_counts())

#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()
data['Item_Fat_Content'].value_counts()

###############################################################################
#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier']) # 0 to 9 types in Outlet
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
#(14204, 15)    # Outlet (0-9)
for i in var_mod:                               #convert all var_mod in (Numer)
    data[i] = le.fit_transform(data[i])
                                            

#One Hot Coding:getting dummies of the data
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])
 #(14204, 35) 20 columns new create
data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10) 

data.dtypes
 
###############################################################################
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
#(14204, 33)

#Divide into test and train:
train = data.loc[data['Train/Test']=="train"]
test = data.loc[data['Train/Test']=="test"]
 
train.shape         ##(8523, 33)      
test.shape         #(5681, 33)

train.head()
#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','Train/Test'],axis=1,inplace=True)
train.drop(['Train/Test'],axis=1,inplace=True)


#Export files as modified versions:
train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)

train.shape     #(8523, 32)
test.shape      #(5681, 31)
###############################################################################
mean_sales = train['Item_Outlet_Sales'].mean()

baseline_submission = pd.DataFrame({
'Item_Identifier':test['Item_Identifier'],
'Outlet_Identifier':test['Outlet_Identifier'],
'Item_Outlet_Sales': mean_sales
},columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])
print(baseline_submission)



X_train = train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1)
X_train.shape  #(8523, 29)

Y_train = train['Item_Outlet_Sales']
Y_train.shape #(8523,)


X_test = test.drop(['Item_Identifier','Outlet_Identifier'],axis=1).copy()
X_test.shape  #(5681, 29)


############################################################################
###############################################################################

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print (np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ((np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)



##################################################################################################
# Linear Regression Model

from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]

# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')




alg1.fit(X_train, Y_train)
lr_pred = alg1.predict(X_test)
lr_accuracy = round(alg1.score(X_train,Y_train) * 100,2)
print( lr_accuracy)


#Decision treee
###############################################################################
from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

alg3.fit(X_train,Y_train)
tree_pred = alg3.predict(X_test)
tree_accuracy = round(alg3.score(X_train,Y_train)*100,2)
print(tree_accuracy)
###############################################################################
#randomForest
from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

alg5.fit(X_train,Y_train)
rf_pred = alg5.predict(X_test)
rf_accuracy = round(alg5.score(X_train,Y_train)*100,2)
print( rf_accuracy)


