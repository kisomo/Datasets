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
print(preds)

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







#+++++++++++++++++++++++++++++++++++++++ Hierarchical clustering ++++++++++++++++++++++++++++++++++++++++++++++++++++++







#+++++++++++++++++++++++++++++++++++++++++++++++ svm +++++++++++++++++++++++++++++++++++++++++++++++++++++++
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





#++++++++++++++++++++++++++++++++++++++++ SOM +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

