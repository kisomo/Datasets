import numpy as np
import pandas as pd


#https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/


train = pd.read_csv('/home/terrence/CODING/Python/MODELS/application_train.csv')
#POS_CASH_balance = pd.read_csv('/home/terrence/CODING/Python/MODELS/POS_CASH_balance.csv')
#bureau_balance = pd.read_csv('/home/terrence/CODING/Python/MODELS/bureau_balance.csv')
#prev = pd.read_csv('/home/terrence/CODING/Python/MODELS/previous_application.csv')
#installments_payments = pd.read_csv('/home/terrence/CODING/Python/MODELS/installments_payments.csv')
#credit_card_balance = pd.read_csv('/home/terrence/CODING/Python/MODELS/credit_card_balance.csv')
#bureau = pd.read_csv('/home/terrence/CODING/Python/MODELS/bureau.csv')
test = pd.read_csv('/home/terrence/CODING/Python/MODELS/application_test.csv')

print(train.shape)
print(test.shape)

print(test.head(1))

#train = train.head(2000)
#test = test.head(200)

print(train.shape)
print(test.shape)

from catboost import CatBoostRegressor

#Identify the datatype of variables
print(train.dtypes)

print(train.isnull().sum())

print(test.dtypes)

print(test.isnull().sum())

#Imputing missing values for both train and test
train.fillna(-999, inplace=True)
test.fillna(-999,inplace=True)

#Creating a training set for modeling and validation set to check model performance
X = train.drop(['TARGET'], axis=1)
y = train.TARGET

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)

#Look at the data type of variables
print(X.dtypes)

categorical_features_indices = np.where(X.dtypes != np.float)[0]
'''
#importing library and building model
from catboost import CatBoostRegressor
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)


test['TARGET'] = model.predict(test) #logistic_regression.predict_proba(x_test)[:,1]

test[['SK_ID_CURR', 'TARGET']].to_csv('first_submission.csv', index=False, float_format='%.8f')

'''

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html


import catboost as cb

'''
# read in the train and test data from csv files
colnames = ['age','wc','fnlwgt','ed','ednum','ms','occ','rel','race','sex','cgain','closs','hpw','nc','label']
train_set = pd.read_csv("adult.data.txt",header=None,names=colnames,na_values='?')
test_set = pd.read_csv("adult.test.txt",header=None,names=colnames,na_values='?',skiprows=[0])

print(train_set.shape)
print(test_set.shape)
print(train_set.head(3))

# convert categorical columns to integers
category_cols = ['wc','ed','ms','occ','rel','race','sex','nc','label']
for header in category_cols:
    train_set[header] = train_set[header].astype('category').cat.codes
    test_set[header] = test_set[header].astype('category').cat.codes


# split labels out of data sets    
train_label = train_set['label']
train_set = train_set.drop('label', axis=1) # remove labels
test_label = test_set['label']
test_set = test_set.drop('label', axis=1) # remove labels


# train default classifier    
#clf = cb.CatBoostClassifier()
#cat_dims = [train_set.columns.get_loc(i) for i in category_cols[:-1]] 
#print(cat_dims)

#clf.fit(train_set, np.ravel(train_label), cat_features=cat_dims)
#res = clf.predict(test_set)
#print('error:',1-np.mean(res==np.ravel(test_label)))
'''

'''
import pandas
import numpy as np
import catboost as cb
from sklearn.model_selection import KFold
from paramsearch import paramsearch
from itertools import product,chain


# read in the train and test data from csv files
colnames = ['age','wc','fnlwgt','ed','ednum','ms','occ','rel','race','sex','cgain','closs','hpw','nc','label']
train_set = pandas.read_csv("adult.data.txt",header=None,names=colnames,na_values='?')
test_set = pandas.read_csv("adult.test.txt",header=None,names=colnames,na_values='?',skiprows=[0])

# convert categorical columns to integers
category_cols = ['wc','ed','ms','occ','rel','race','sex','nc','label']
cat_dims = [train_set.columns.get_loc(i) for i in category_cols[:-1]] 
for header in category_cols:
    train_set[header] = train_set[header].astype('category').cat.codes
    test_set[header] = test_set[header].astype('category').cat.codes

# split labels out of data sets    
train_label = train_set['label']
train_set = train_set.drop('label', axis=1)
test_label = test_set['label']
test_set = test_set.drop('label', axis=1)


params = {'depth':[3,1,2,6,4,5,7,8,9,10],
          'iterations':[250,100,500,1000],
          'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
          'l2_leaf_reg':[3,1,5,10,100],
          'border_count':[32,5,10,20,50,100,200],
          'ctr_border_count':[50,5,10,20,100,200],
          'thread_count':-1}



# this function does 3-fold crossvalidation with catboostclassifier          
def crossvaltest(params,train_set,train_label,cat_dims,n_splits=3):
    kf = KFold(n_splits=n_splits,shuffle=True) 
    res = []
    for train_index, test_index in kf.split(train_set):
        train = train_set.iloc[train_index,:]
        test = train_set.iloc[test_index,:]

        labels = train_label.ix[train_index]
        test_labels = train_label.ix[test_index]

        clf = cb.CatBoostClassifier(**params)
        clf.fit(train, np.ravel(labels), cat_features=cat_dims)

        res.append(np.mean(clf.predict(test)==np.ravel(test_labels)))
    return np.mean(res)


###Terrence = crossvaltest(params,train_set,train_label,cat_dims,n_splits=3)
###print(Terrence)


# this function runs grid search on several parameters
def catboost_param_tune(params,train_set,train_label,cat_dims=None,n_splits=3):
    ps = paramsearch(params)
    # search 'border_count', 'l2_leaf_reg' etc. individually 
    #   but 'iterations','learning_rate' together
    for prms in chain(ps.grid_search(['border_count']),
                      ps.grid_search(['ctr_border_count']),
                      ps.grid_search(['l2_leaf_reg']),
                      ps.grid_search(['iterations','learning_rate']),
                      ps.grid_search(['depth'])):
        res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)
        # save the crossvalidation result so that future iterations can reuse the best parameters
        ps.register_result(res,prms)
        #print(res,prms,s'best:',ps.bestscore(),ps.bestparam())
        print(res,prms,'best:',ps.bestscore(),ps.bestparam())
    return ps.bestparam()

bestparams = catboost_param_tune(params,train_set,train_label,cat_dims)



# train classifier with tuned parameters    
clf = cb.CatBoostClassifier(**bestparams)
clf.fit(train_set, np.ravel(train_label), cat_features=cat_dims)
res = clf.predict(test_set)
print('error:',1-np.mean(res==np.ravel(test_label)))

'''





















