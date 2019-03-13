import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train data shape:", train.shape)
print("Test data shape:", test.shape)

print(train.head())

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

print (train.SalePrice.describe())

#show skew to log transform data
print("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()

#use np.log to bring data to a normal distribution
target =np.log(train.SalePrice)
print ("\n Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


#shows top 5 correlated attributes and bottom 5
numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])

#handle outliers
#this visualizes relationship between garage area and sale price using scatter plot
plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show

#remove outliers
train = train[train['GarageArea'] < 1200]

#display scatter again
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show 

#handling null values
#returning a count of null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

#view non-numeric features
categoricals = train.select_dtypes(exclude=[np.number])
print(categoricals.describe())

print("Original: \n")
print(train.Street.value_counts(), "\n")

#change street to numerical data
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)

print('Encoded: \n')
print(train.enc_street.value_counts())

#look at saleCondition with pivot table
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

#encode partial salecondition as new feature
def encode(x): return 1 if x=='Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

#fill in the missing values called interpolation
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() !=0))


#build a linear regression model to predict Saleprice
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print(model.score(X_test, y_test))

#consider RMSE and plot with scatter
predictions = model.predict(X_test)
print(mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75, color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear regression Model')
plt.show()

#use ridge regularization change influence of attributes
for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()

#create csv file for predictions
submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.select_dtypes(
    include=[np.number]).drop(['Id'], axis=1).interpolate()
predictions = model.predict(feats)
final_predictions = np.exp(predictions)
print("Original predictions are: \n", predictions[:10], "\n")
print("Final predictions are: \n", final_predictions[:10])

submission['SalePrice'] = final_predictions
print(submission.head)

#export to csv
submission.to_csv('submission1.csv', index=False)





