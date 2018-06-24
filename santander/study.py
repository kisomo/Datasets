import numpy as np
import pandas as pd


#++++++++++++++++ xgboost lightgbm catboost +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.kaggle.com/samratp/lightgbm-xgboost-catboost

### Import required libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')


# Read train and test files
train_df = pd.read_csv('/home/terrence/CODING/Python/MODELS/Santander_Data/train.csv')
test_df = pd.read_csv('/home/terrence/CODING/Python/MODELS/Datasets/santander/test.csv')


print(train_df.head(2))

print(train_df.info())

print(test_df.head(2))

print(test_df.info())

#### Check if there are any NULL values in Train Data
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
if (train_df.columns[train_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))
    train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)


 #### Check if there are any NULL values in Test Data
print("Total Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))
if (test_df.columns[test_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(test_df.columns[test_df.isnull().sum() != 0])))
    test_df[test_df.columns[test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)


X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)

'''
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


#+++++++++++++++++ LightGBM +++++++++++++++++++++++++

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=10, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result


# Training LGB
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")


# feature importance
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])


#+++++++++++++++++++++++ XGBoost ++++++++++++++++++++++++++++++++++++++==

def run_xgb(train_X, train_y, val_X, val_y, test_X):
    params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.005,
          'max_depth': 10, 
          'subsample': 0.7, 
          'colsample_bytree': 0.5,
          'alpha':0,
          'random_state': 42, 
          'silent': True}
    
    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)
    
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 30, verbose_eval=50)
    
    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    return xgb_pred_y, model_xgb


# Training XGB
pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, X_test)
print("XGB Training Completed...")



#+++++++++++++++++++++++ CatBoost ++++++++++++++++++++++++++++++++

cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.1,
                             depth=7,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 20,
                             od_wait=20)


cb_model.fit(dev_X, dev_y,
             eval_set=(val_X, val_y),
             use_best_model=True,
             verbose=True)


pred_test_cat = np.expm1(cb_model.predict(X_test))


sub = pd.read_csv('../sample_submission.csv')

sub_lgb = pd.DataFrame()
sub_lgb["target"] = pred_test

sub_xgb = pd.DataFrame()
sub_xgb["target"] = pred_test_xgb

sub_cat = pd.DataFrame()
sub_cat["target"] = pred_test_cat

sub["target"] = (sub_lgb["target"] + sub_xgb["target"] + sub_cat["target"])/3

print(sub.head())
sub.to_csv('sub_lgb_xgb_cat.csv', index=False)

'''














#+++++++++++++++++++ CatBoost +++++++++++++++++++++++++++++++++++++++








































#++++++++++++++++++++++++++++++++++++++++++ SVR ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://sadanand-singh.github.io/posts/svmpython/

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

#X_train = train_df.drop(["ID", "target"], axis=1)
#y_train = np.log1p(train_df["target"].values)
#X_test = test_df.drop(["ID"], axis=1)
X = X_train
y = y_train

# shuffle the dataset
#X, y = shuffle(X, y, random_state=0)

# Split the dataset in two equal parts
#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=0)

# Set the parameters by cross-validation
#parameters = [{'kernel': ['rbf'],
#               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
#                'C': [1, 10, 100, 1000]},
#              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


parameters = [{'kernel': ['rbf'],
               'gamma': [ 0.01, 0.1, 0.2],
                'C': [10, 100, 1000]},
              {'kernel': ['linear'], 'C': [10, 100, 1000]}]


print("# Tuning hyper-parameters")
print()

'''
clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()
'''



from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)
y_rbf = svr_rbf.fit(X, y).predict(X_test)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)


sub = pd.read_csv('/home/terrence/CODING/Python/MODELS/Datasets/santander/sample_submission.csv')

sub["target"] = y_rbf 

print(sub.head())
sub.to_csv('SVR1.csv', index=False)


'''
lw = 2
plt.figure(figsize=(12, 7))
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

'''

