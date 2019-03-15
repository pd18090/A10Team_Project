# -*- coding: utf-8 -*-
"""
@author: bb18115

Majority of the code has been taken from :
http://www.kaggle.com/dgawlik/house-prices-eda and https://www.kaggle.com/sumana65/stacked-regression
Modifications by bb18115    
"""
import csv 
import numpy as np # Numpy is a library used in array computing
import pandas as pd # Pandas is a library which provide a data structure to store and present data in a 
# tabular format
import matplotlib.pyplot as plt # This is a plotting library used for data visualising
import seaborn as sns # seaborn is another data visualising library used in statistical plotting
from scipy.stats import skew # Used to provide functions to compute the skewness of a data set and 
import sklearn.linear_model as linear_model # Linear_model is a library containing various linear models
from sklearn.ensemble import GradientBoostingRegressor #Contains the Gradient Boosting Regressor
from sklearn.kernel_ridge import KernelRidge 
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import RobustScaler

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore warning (from sklearn and seaborn)

pd.options.display.max_rows = 1000 # this limits the number of rows displayed to be at most 1000
pd.options.display.max_columns = 20# this limits the number of columns displayed to be at most 20
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

# Loading the training and testing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Store the number of records in both training and test data
ntrain = train.shape[0]
y = train['SalePrice']

#Combining the training and testing data in order to preprocess the data
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
#Checking for missing data
all_data_na = (train.isnull().sum() / len(train)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data)

#Visualising the missing data using a bar plot
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

# Replaces null values in qualitative features where it most likely means the house does not have this feature 
# with the string 'None' 
cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','PoolQC','MiscFeature','Alley','Fence','BsmtQual', 'BsmtCond', 'BsmtExposure','FireplaceQu','MasVnrType','MSSubClass']
all_data[cols] = all_data[cols].fillna('None')
# Replaces null values in Lot Frontage with the median lot frontage in that neighbourhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
# Replaces null values in quantitative features where it most likely means the house does not have this feature
# with the value 0
cols = ['GarageYrBlt', 'GarageArea', 'BsmtFinType1', 'BsmtFinType2', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea']
all_data[cols] = all_data[cols].fillna(0)

all_data["Functional"] = all_data["Functional"].fillna("Typ")
# Replaces the null values for features in which these details were missed out, with the value most frequently used
for col in ('Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType', 'MSZoning'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
# Utilities has only one differing value therefore it can be dropped   
all_data = all_data.drop(['Utilities'], axis=1)

#A repeat check for missing data to ensure there is no missing data left
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()

# Changing some numerical values into a qualitative value so that it can be treated as a categorical variable
# Applying the string casting function to the MSSubClass variable
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# Implementing a Label Encoder to process categorical features so that it can be inluded in the regression
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# prints the shape of the data        
print('Shape all_data: {}'.format(all_data.shape))

# Creating a new variable to record the total area of space in the house as this is an important feature to consider when purchasing a house
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# numeric_feats holds the numerical features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

# numeric_feats holds the numerical features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

# Retrieves the features which have a skewness larger than 0.75 or less than -0.75
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

# The box cox transformation is then applied to these features in the combined data set
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# Adds dummy columns for categorical variables to indicates its value with a1 in that column
all_data = pd.get_dummies(all_data)
print(all_data.shape)

# Split the data back into training and testing data now that it has been preprocess
train = all_data[:ntrain]
test = all_data[ntrain:]


# Defining a function which returns the root-square mean error of the 2 sets of data passed into the function
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

#Instantiates and initialises the Lasso model with a maximum number of iterations at 10000 and 3 folds for the cross validation technique
lasso = linear_model.LassoLarsCV(max_iter=10000, cv=3)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
ENet = make_pipeline(RobustScaler(), linear_model.ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
# This then fits the model using the house features and house prices provided in the training data
houseFeats = train[train.columns]
housePrices = y
lasso.fit(houseFeats, np.log(housePrices))
pricePred = []
# The model is then used to predict house prices using the training data
pricePred.append(np.exp(lasso.predict(houseFeats)))
# The error is displayed using the error function defined previously
print('LassoCV:',error(housePrices, pricePred[0]))
KRR.fit(houseFeats, np.log(housePrices))
ENet.fit(houseFeats, np.log(housePrices))
pricePred.append(np.exp(KRR.predict(houseFeats)))
print('KRR:', error(housePrices,pricePred[1]))
pricePred.append(np.exp(ENet.predict(houseFeats)))
GBoost.fit(houseFeats, np.log(housePrices))
print('ENet:', error(housePrices,pricePred[2]))
pricePred.append(np.exp(GBoost.predict(houseFeats)))
print('GBoost:', error(housePrices,pricePred[3]))
avgPricePred = np.mean(pricePred)
print('SimpleAvg:',error(housePrices,avgPricePred))
# The house prices are then predicted for the test data using the model and the results are formatted into a csv.
# This csv can then be submitted to the kaggle competition
testHouseFeats = test[test.columns]
pricePred = np.exp(lasso.predict(testHouseFeats))+np.exp(KRR.predict(testHouseFeats))+np.exp(ENet.predict(testHouseFeats))+np.exp(GBoost.predict(testHouseFeats))
pricePred /= 4
with open ('test_prEdictions.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile,delimiter=',',
                        quotechar="'",quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Id', 'SalePrice'])
    for i in range(1461,len(pricePred)+1461):
        writer.writerow([i,pricePred[i-1461]])




