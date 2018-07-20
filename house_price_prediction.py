# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:02:12 2018

@author: MOHMMED SAKET
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
"""missing_numeric = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])
missing_numeric = missing_numeric[(missing_numeric['train']>0) | (missing_numeric['test']>0)]
print(missing_numeric.sort_values(by=['train', 'test'], ascending=False))"""
# Drop the features
feature_drop = ['PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageYrBlt', 'MoSold', 'YrSold', 
                'LowQualFinSF', 'MiscVal', 'PoolArea']
#check and fill data
datasets = [train, test]

for df in datasets:
    df.drop(feature_drop, axis=1, inplace=True)
    df.loc[df['Alley'].isnull(), 'Alley'] = 'NoAlley'

    df.loc[df['GarageCond'].isnull(), 'GarageCond'] = 'NoGarage'
    df.loc[df['GarageQual'].isnull(), 'GarageQual'] = 'NoGarage'
    df.loc[df['GarageType'].isnull(), 'GarageType'] = 'NoGarage'
    df.loc[df['GarageFinish'].isnull(), 'GarageFinish'] = 'NoGarage'
    

    df.loc[df['BsmtExposure'].isnull(), 'BsmtExposure'] = 'NoBsmt'
    df.loc[df['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 'NoBsmt'
    df.loc[df['BsmtCond'].isnull(), 'BsmtCond'] = 'NoBsmt'
    df.loc[df['BsmtQual'].isnull(), 'BsmtQual'] = 'NoBsmt'
    df.loc[df['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'NoBsmt'
    

    df.loc[df['MasVnrType'].isnull(), 'MasVnrType'] = 'None'
    df.loc[df['MasVnrArea'].isnull(), 'MasVnrArea'] = 0
       
train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)

test_numeric_missing = ['BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageArea', 'GarageCars', 'TotalBsmtSF']
test_categorical_missing = ['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType']
for i in test_numeric_missing:
    test[i].fillna(0, inplace=True)
for j in test_categorical_missing:
    test[j].fillna(test[j].mode()[0], inplace=True)

# Check the missing values again 
"""missing_numeric = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])
missing_numeric = missing_numeric[(missing_numeric['train']>0) | (missing_numeric['test']>0)]
print(missing_numeric.sort_values(by=['train', 'test'], ascending=False))
print(train.select_dtypes(exclude=[object]).describe())"""
#plot
"""numeric_data_select = train[['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'FullBath', 'TotalBsmtSF', 
                                    'YearBuilt', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'WoodDeckSF', 'OpenPorchSF',
                                    'HalfBath', 'LotArea']]
corr_select = numeric_data_select.corr()
plt.figure(figsize=(8, 8))
mask = np.zeros_like(corr_select)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_select, vmax=1, square=True, annot=True, mask=mask, cbar=False, linewidths=0.1)
plt.xticks(rotation=45)
sns.pairplot(numeric_data_select, size=2)"""
#Dummy
train_ExterQual_dummy = pd.get_dummies(train['ExterQual'], prefix='ExterQual')
test_ExterQual_dummy = pd.get_dummies(test['ExterQual'], prefix='ExterQual')

train_BsmtQual_dummy = pd.get_dummies(train['BsmtQual'], prefix='BsmtQual')
test_BsmtQual_dummy = pd.get_dummies(test['BsmtQual'], prefix='BsmtQual')

train_BsmtExposure_dummy = pd.get_dummies(train['BsmtExposure'], prefix='BsmtExposure')
test_BsmtExposure_dummy = pd.get_dummies(test['BsmtExposure'], prefix='BsmtExposure')

train_GarageFinish_dummy = pd.get_dummies(train['GarageFinish'], prefix='GarageFinish')
test_GarageFinish_dummy = pd.get_dummies(test['GarageFinish'], prefix='GarageFinish')

train_SaleCondition_dummy = pd.get_dummies(train['SaleCondition'], prefix='SaleCondition')
test_SaleCondition_dummy = pd.get_dummies(test['SaleCondition'], prefix='SaleCondition')

train_CentralAir_dummy = pd.get_dummies(train['CentralAir'], prefix='CentralAir')
test_CentralAir_dummy = pd.get_dummies(test['CentralAir'], prefix='CentralAir')

train_KitchenQual_dummy = pd.get_dummies(train['KitchenQual'], prefix='KitchenQual')
test_KitchenQual_dummy = pd.get_dummies(test['KitchenQual'], prefix='KitchenQual')

data = train.select_dtypes(exclude=[object])
y = np.log1p(data['SalePrice'])
X = data.drop(['Id', 'SalePrice'], axis=1)
X = pd.concat([X, train_ExterQual_dummy, train_BsmtQual_dummy, train_GarageFinish_dummy, train_BsmtExposure_dummy,
              train_SaleCondition_dummy, train_CentralAir_dummy, train_KitchenQual_dummy], axis=1)
#models
lr = LinearRegression()
ri = Ridge(alpha=0.1, normalize=False)
ricv = RidgeCV(cv=5)
gdb = GradientBoostingRegressor(n_estimators=200)

test_id = test['Id']
test = test.select_dtypes(exclude=[object]).drop('Id', axis=1)
test = pd.concat([test, test_ExterQual_dummy, test_BsmtQual_dummy, test_GarageFinish_dummy, test_BsmtExposure_dummy,
              test_SaleCondition_dummy, test_CentralAir_dummy, test_KitchenQual_dummy], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
for i in [lr,ri,ricv,gdb]:
    i.fit(X_train, y_train)

for j in [lr,ri,ricv,gdb]:
    print(j.score(X_test, y_test)*100,'%')
# so we use gdb for prediction
pred = gdb.predict(test)
pred = np.expm1(pred)

prediction = pd.DataFrame({'Id':test_id, 'SalePrice':pred})
prediction.to_csv('Prediction.csv', index=False)
