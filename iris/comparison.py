import numpy as np
import pandas as pd

#https://www.kaggle.com/mgabrielkerr/visualizing-knn-svm-and-xgboost-on-iris-dataset

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)

#print(y)

print(X_df.shape)
print(y_df.shape)

print(X_df.head(5))
print(y_df.head(5))


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize the Xs
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
print('After standardizing our features, the first 5 rows of our data now look like this:\n')
print(pd.DataFrame(X_train, columns=X_df.columns).head())

# Print the unique labels of the dataset
print('\n' + 'The unique labels in this data are ' + str(np.unique(y)))



#+++++++++++++++++++++++++++++++++++++++ Visualize ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#https://www.kaggle.com/mgabrielkerr/visualizing-knn-svm-and-xgboost-on-iris-dataset

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               c=cmap(idx), marker=markers[idx], label=cl)
plt.show()


import warnings

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.show()



#+++++++++++++++++++++++++++++++++++++++++++++++ AdaBoost +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.ritchieng.com/machine-learning-ensemble-of-learners-adaboost/

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
#Instantiate
abc = AdaBoostClassifier()

# Fit
abc.fit(X_train, y_train)

# Predict
y_pred = abc.predict(X_test)

# Accuracy
AdaBoost_score = accuracy_score(y_pred, y_test)
print("AdaBoost score = {:.4f}" .format(AdaBoost_score))


#+++++++++++++++++++++++++++++++++++++++++++ xgboost ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.kdnuggets.com/2017/03/simple-xgboost-tutorial-iris-dataset.html

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#from sklearn.datasets import dump_svmlight_file
#dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
#dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
#dtrain_svm = xgb.DMatrix('dtrain.svm')
#dtest_svm = xgb.DMatrix('dtest.svm')

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this dataset
num_round = 500  # the number of training iterations

bst = xgb.train(param, dtrain, num_round)

bst.dump_model('dump.raw.txt')

preds = bst.predict(dtest)
#print(preds)

import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])

from sklearn.metrics import precision_score

xgb_score = precision_score(y_test, best_preds, average='macro')
print("XGB score = {:.4f} " .format(xgb_score))

from sklearn.externals import joblib

joblib.dump(bst, 'bst_model.pkl', compress=True)
# bst = joblib.load('bst_model.pkl') # load it later


#++++++++++++++++++++++++++++++++++++++++++ xgboost ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.kaggle.com/mgabrielkerr/visualizing-knn-svm-and-xgboost-on-iris-dataset

import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, y_train)

#print('The accuracy of the xgb classifier is {:.2f} out of 1 on training data'.format(xgb_clf.score(X_train, y_train)))
#print('The accuracy of the xgb classifier is {:.2f} out of 1 on test data'.format(xgb_clf.score(X_test, y_test)))

xgb_clf_score = xgb_clf.score(X_test, y_test)
print('XGB_CLF score = {:.4f}' .format(xgb_clf_score))

#plot_decision_regions(X_test_std, y_test, xgb_clf)



#++++++++++++++++++++++++++++++++++++++++++++++ lightgbm ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py

import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

try:
    import cPickle as pickle
except BaseException:
    import pickle


num_train, num_feature = X_train.shape
#print(num_train)
#print(num_feature)


# create dataset for lightgbm
# if you want to re-use data, remember to set free_raw_data=False
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)


# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# generate a feature name
feature_name = ['feature_' + str(col) for col in range(num_feature)]

print('Start training...')
# feature_name and categorical_feature
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,  # eval training data
                feature_name=feature_name,
                categorical_feature=[21])

# check feature name
print('Finish first 10 rounds...')
print('4th feature name is:', repr(lgb_train.feature_name[3]))


# save model to file
gbm.save_model('model.txt')

# dump model to json (and save to file)
print('Dump model to JSON...')
model_json = gbm.dump_model()

with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

# feature names
print('Feature names:', gbm.feature_name())

# feature importances
print('Feature importances:', list(gbm.feature_importance()))

# load model to predict
print('Load model to predict')
bst = lgb.Booster(model_file='model.txt')
# can only predict with the best iteration (or the saving iteration)
y_pred = bst.predict(X_test)
# eval with loaded model
print('The rmse of loaded model\'s prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


# dump model with pickle
with open('model.pkl', 'wb') as fout:
    pickle.dump(gbm, fout)
# load model with pickle to predict
with open('model.pkl', 'rb') as fin:
    pkl_bst = pickle.load(fin)
# can predict with any iteration when loaded in pickle way
y_pred = pkl_bst.predict(X_test, num_iteration=7)
# eval with loaded model
print('The rmse of pickled model\'s prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)


# continue training
# init_model accepts:
# 1. model file name
# 2. Booster()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model='model.txt',
                valid_sets=lgb_eval)

print('Finish 10 - 20 rounds with model file...')

# decay learning rates
# learning_rates accepts:
# 1. list/tuple with length = num_boost_round
# 2. function(curr_iter)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                valid_sets=lgb_eval)

print('Finish 20 - 30 rounds with decay learning rates...')

# change other parameters during training
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=lgb_eval,
                callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])

print('Finish 30 - 40 rounds with changing bagging_fraction...')

# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
def loglikelood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess


# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, value: array, is_higher_better: bool
# binary error
def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                fobj=loglikelood,
                feval=binary_error,
                valid_sets=lgb_eval)

print('Finish 40 - 50 rounds with self-defined objective function and eval metric...')

print('Start a new training job...')

# callback
def reset_metrics():
    def callback(env):
        lgb_eval_new = lgb.Dataset(X_test, y_test, reference=lgb_train)
        if env.iteration - env.begin_iteration == 5:
            print('Add a new valid dataset at iteration 5...')
            env.model.add_valid(lgb_eval_new, 'new valid')
    callback.before_iteration = True
    callback.order = 0
    return callback


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,
                callbacks=[reset_metrics()])

print('Finish first 10 rounds with callback function...')

#++++++++++++++++++++++++++++++++++++++++++++++++ lightgbm ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://github.com/Microsoft/LightGBM/issues/991

from sklearn import datasets
import numpy as np
import lightgbm as lgb
param_algo = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class' : 3,
                'metric':{'multi_logloss'},
                'num_leaves': 31,
                'min_data_in_leaf':1,
                'learning_rate':0.1,
                'zero_as_missing':False,
            }

feature_name=['a', 'b', 'c', 'd']

lgb_train = lgb.Dataset(X_train, 
                        y_train,
                        feature_name=feature_name) #,categorical_feature=[0])

gbm = lgb.train(param_algo,
                lgb_train, 
                num_boost_round=6, 
                valid_sets=[lgb_train],
                feature_name=feature_name) #,categorical_feature=[0])


res = gbm.predict(X_test)
#print(res)

best_predss = np.asarray([np.argmax(line) for line in res])

#print(best_predss)
#print(y_test)

correct = [estimate== target for estimate, target in zip(best_predss, y_test)]

#print(correct)

accuracy =  sum(correct) / len(correct)

#lgbm_score = gbm.score(y_test, best_preds, average='macro')
print("LightGBM score = {:.4f} " .format(accuracy))






#++++++++++++++++++++++++++++++++++++++++++++ catboost ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/

from catboost import Pool, CatBoostClassifier

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=30, learning_rate=1, depth=3, loss_function='MultiClass')
# Fit model
model.fit(X_train,y_train)
# Get predicted classes
preds_class = model.predict(X_test)
#print(preds_class)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(X_test)
#print(preds_proba)
# Get predicted RawFormulaVal
preds_raw = model.predict(X_test, prediction_type='RawFormulaVal')  

catboost_score = model.score(X_test, y_test)
print('CatBoost score = {:.4f}' .format(catboost_score))






#+++++++++++++++++++++++++++++++++++++++++++ k-NN +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#https://www.kaggle.com/mgabrielkerr/visualizing-knn-svm-and-xgboost-on-iris-dataset

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)

#print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(X_train, y_train)))
#print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(X_test, y_test)))
kNN_score = knn.score(X_test, y_test)
print('k_NN score = {:.4f}' .format(kNN_score))

#plot_decision_regions(X_test, y_test, knn)




#++++++++++++++++++++++++++++++++++++++++++++ K-Means ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++







print("+++++++++++++++++++++++++++++++++++++++ Hierarchical clustering ++++++++++++++++++++++++++++++++++++++++++++++++++++")

# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

plt.scatter(X_train[:,0], X_train[:,1])
plt.show()

# generate the linkage matrix
Z = linkage(X_train, 'ward')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X_train))
print(c)


print(Z[0])
print(Z[1])
print(Z[:20])

print(X[[33, 68, 62]])

idxs = [33, 68, 62]
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1])  # plot all points
plt.scatter(X[idxs,0], X[idxs,1], c='r')  # plot interesting points in red again
plt.show()

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


print(Z[-8:,2])


plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()


# set cut-off to 50
max_d = 7  # max_d as in max_distance

fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()









print("+++++++++++++++++++++++++++++++++++++++++++++++ svm +++++++++++++++++++++++++++++++++++++++++++++++++++++")

#https://www.kaggle.com/mgabrielkerr/visualizing-knn-svm-and-xgboost-on-iris-dataset

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train, y_train)

#print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train, y_train)))

#print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test, y_test)))

svm_score = svm.score(X_test, y_test)
print('SVM score = {:.4f}' .format(svm_score))

#plot_decision_regions(X_test, y_test, svm)




#+++++++++++++++++++++++++++++++++++++++++ GMM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++






#++++++++++++++++++++++++++++++++++++++ Logistic regression ++++++++++++++++++++++++++++++++++++++++++++++++






#++++++++++++++++++++++++++++++++++++ Nearest Centroid +++++++++++++++++++++++++++++++++++++++++++++++++++






#+++++++++++++++++++++++++++++++++++ Naive Bayes +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++








print("++++++++++++++++++++++++++++++++++++++++ SOM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

#https://dzone.com/articles/self-organizing-maps

from numpy import genfromtxt,array,linalg,zeros,apply_along_axis

# reading the iris dataset in the csv format    
# (downloaded from http://aima.cs.berkeley.edu/data/iris.csv)
#data = genfromtxt('iris.csv', delimiter=',',usecols=(0,1,2,3))
# normalization to unity of each pattern in the data
#data = apply_along_axis(lambda x: x/linalg.norm(x),1,data)
data = X_train
from minisom import MiniSom
### Initialization and training ###
som = MiniSom(7,7,4,sigma=1.0,learning_rate=0.5)
som.random_weights_init(data)
print("Training...")
som.train_random(data,100) # training with 100 iterations
print("\n...ready!")

'''
from pylab import plot,axis,show,pcolor,colorbar,bone
bone()
pcolor(som.distance_map().T) # distance map as background
colorbar()
# loading the labels
#target = genfromtxt('iris.csv', delimiter=',',usecols=(4),dtype=str)
target = y_train

t = zeros(len(target),dtype=int)
print(t)
t[target == 'setosa'] = 0
t[target == 'versicolor'] = 1
t[target == 'virginica'] = 2
print(t)


# use different colors and markers for each label
markers = ['o','s','D']
colors = ['r','g','b']
for cnt,xx in enumerate(data):
 w = som.winner(xx) # getting the winner
 # palce a marker on the winning position for the sample xx
 plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
   markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
axis([0,som.weights.shape[0],0,som.weights.shape[1]])
show() # show the figure
'''

