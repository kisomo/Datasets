
#++++++++++++++++ xgboost lightgbm catboost +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
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

'''
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

#https://www.kaggle.com/alexpengxiao/preprocessing-model-averaging-by-xgb-lgb-1-39

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("/home/terrence/CODING/Python/MODELS/Santander_Data"))
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('/home/terrence/CODING/Python/MODELS/Santander_Data/train.csv')
test = pd.read_csv('/home/terrence/CODING/Python/MODELS/Santander_Data/test.csv')

print(train.shape)
print(test.shape)


test_ID = test['ID']
y_train = train['target']
y_train = np.log1p(y_train)
train.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)
cols_with_onlyone_val = train.columns[train.nunique() == 1]
train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
test.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
NUM_OF_DECIMALS = 32
train = train.round(NUM_OF_DECIMALS)
test = test.round(NUM_OF_DECIMALS)
colsToRemove = []

columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i + 1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
train.drop(colsToRemove, axis=1, inplace=True) 
test.drop(colsToRemove, axis=1, inplace=True) 
print(train.shape)

from sklearn import model_selection
from sklearn import ensemble
NUM_OF_FEATURES = 1000
def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(y - pred, 2)))

x1, x2, y1, y2 = model_selection.train_test_split(
    train, y_train.values, test_size=0.20, random_state=5)
model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=7)
model.fit(x1, y1)
print(rmsle(y2, model.predict(x2)))

'''
col = pd.DataFrame({'importance': model.feature_importances_, 'feature': train.columns}).sort_values(
    by=['importance'], ascending=[False])[:NUM_OF_FEATURES]['feature'].values
train = train[col]
test = test[col]
train.shape

from scipy.stats import ks_2samp
THRESHOLD_P_VALUE = 0.01 #need tuned
THRESHOLD_STATISTIC = 0.3 #need tuned
diff_cols = []
for col in train.columns:
    statistic, pvalue = ks_2samp(train[col].values, test[col].values)
    if pvalue <= THRESHOLD_P_VALUE and np.abs(statistic) > THRESHOLD_STATISTIC:
        diff_cols.append(col)
for col in diff_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)
train.shape


from sklearn import random_projection
ntrain = len(train)
ntest = len(test)
tmp = pd.concat([train,test])#RandomProjection
weight = ((train != 0).sum()/len(train)).values
tmp_train = train[train!=0]
tmp_test = test[test!=0]
train["weight_count"] = (tmp_train*weight).sum(axis=1)
test["weight_count"] = (tmp_test*weight).sum(axis=1)
train["count_not0"] = (train != 0).sum(axis=1)
test["count_not0"] = (test != 0).sum(axis=1)
train["sum"] = train.sum(axis=1)
test["sum"] = test.sum(axis=1)
train["var"] = tmp_train.var(axis=1)
test["var"] = tmp_test.var(axis=1)
train["median"] = tmp_train.median(axis=1)
test["median"] = tmp_test.median(axis=1)
train["mean"] = tmp_train.mean(axis=1)
test["mean"] = tmp_test.mean(axis=1)
train["std"] = tmp_train.std(axis=1)
test["std"] = tmp_test.std(axis=1)
train["max"] = tmp_train.max(axis=1)
test["max"] = tmp_test.max(axis=1)
train["min"] = tmp_train.min(axis=1)
test["min"] = tmp_test.min(axis=1)
train["skew"] = tmp_train.skew(axis=1)
test["skew"] = tmp_test.skew(axis=1)
train["kurtosis"] = tmp_train.kurtosis(axis=1)
test["kurtosis"] = tmp_test.kurtosis(axis=1)
del(tmp_train)

del(tmp_test)
NUM_OF_COM = 100 #need tuned
transformer = random_projection.SparseRandomProjection(n_components = NUM_OF_COM)
RP = transformer.fit_transform(tmp)
rp = pd.DataFrame(RP)
columns = ["RandomProjection{}".format(i) for i in range(NUM_OF_COM)]
rp.columns = columns

rp_train = rp[:ntrain]
rp_test = rp[ntrain:]
rp_test.index = test.index

#concat RandomProjection and raw data
train = pd.concat([train,rp_train],axis=1)
test = pd.concat([test,rp_test],axis=1)

del(rp_train)
del(rp_test)
train.shape

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#define evaluation method for a given model. we use k-fold cross validation on the training set. 
#the loss function is root mean square logarithm error between target and prediction
#note: train and y_train are feeded as global variables
NUM_FOLDS = 5 #need tuned
def rmsle_cv(model):
    kf = KFold(NUM_FOLDS, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
#ensemble method: model averaging

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    # the reason of clone is avoiding affect the original base models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]  
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([ model.predict(X) for model in self.models_ ])
        return np.mean(predictions, axis=1)


model_xgb = xgb.XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5, 
                             gamma=1.5, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=57, n_estimators=1000, reg_alpha=0, 
                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=144,
                              learning_rate=0.005, n_estimators=720, max_depth=13,
                              metric='rmse',is_training_metric=True,
                              max_bin = 55, bagging_fraction = 0.8,verbose=-1,
                              bagging_freq = 5, feature_fraction = 0.9) 
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
averaged_models = AveragingModels(models = (model_xgb, model_lgb))
score = rmsle_cv(averaged_models)
print("averaged score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

averaged_models.fit(train.values, y_train)
pred = np.expm1(averaged_models.predict(test.values))
ensemble = pred
sub = pd.DataFrame()
sub['ID'] = test_ID
sub['target'] = ensemble
sub.to_csv('submission.csv',index=False)

#Xgboost score: 1.3582 (0.0640)
#LGBM score: 1.3437 (0.0519)
#averaged score: 1.3431 (0.0586)

#Xgboost score: 1.3566 (0.0525)
#LGBM score: 1.3477 (0.0497)
#averaged score: 1.3438 (0.0516)

#Xgboost score: 1.3540 (0.0621)
#LGBM score: 1.3463 (0.0485)
#averaged score: 1.3423 (0.0556)

'''



















