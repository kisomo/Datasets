# -*- coding: utf-8 -*-
#https://www.kaggle.com/nicapotato/simple-catboost

'''
import numpy as np
import pandas as pd

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
#print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Viz
import seaborn as sns
import matplotlib.pyplot as plt



print("\nData Load Stage")
training = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv', 
index_col = "item_id", parse_dates = ["activation_date"]).sample(250)
traindex = training.index
#print(traindex)
testing = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv', 
index_col = "item_id", parse_dates = ["activation_date"]).sample(50)
testdex = testing.index
#print(testdex)
print(training.shape)
print(testing.shape)


y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
#print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(training.head(1))


# Combine Train and Test
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Remove Dead Variables
df.drop(["activation_date","image"],axis=1,inplace=True)
print(df.shape)
#print(df.head(2))
#print(df.dtypes)


print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name","category_name","item_seq_number","user_type","image_top_1"]
messy_categorical = ["param_1","param_2","param_3","title","description"] # Need to find better technique for these
print("Encoding :",categorical + messy_categorical)


# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical + messy_categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))

print(df.head(2))

print("\nCatboost Modeling Stage")
X = df.loc[traindex,:].copy()
print("Training Set shape",X.shape)
test = df.loc[testdex,:].copy()
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df
gc.collect()


# Training and Validation Set
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=23)

# Prepare Categorical Variables
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
categorical_features_pos = column_index(X,categorical + messy_categorical)
print(categorical_features_pos)

#print(df.head(2))

# Train Model
print("Train CatBoost Decision Tree")
modelstart= time.time()
cb_model = CatBoostRegressor(iterations=20,
                             learning_rate=0.02,
                             depth=7,
                             eval_metric='RMSE',
                             random_seed = 23,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=10)
cb_model.fit(X_train, y_train,
             eval_set=(X_valid,y_valid),
             cat_features=categorical_features_pos,
             use_best_model=True,
             verbose=True)


# # Feature Importance
# fea_imp = pd.DataFrame({'imp': cb_model.feature_importances_, 'col': X.columns})
# fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
# _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
# plt.savefig('catboost_feature_importance.png')   

print("Model Evaluation Stage")
print(cb_model.get_params())

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, cb_model.predict(X_valid))))

catpred = cb_model.predict(test)
catsub = pd.DataFrame(catpred,columns=["deal_probability"],index=testdex)
catsub['deal_probability'].clip(0.0, 1.0, inplace=True)
#catsub.to_csv("catsub_two.csv",index=True,header=True) # Between 0 and 1
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
'''


#++++++++++++++++++++++++++++++++++++++++++++++++++++ FastText +++++++++++++++++++++++++++++++++++++++++++++++++++++")

'''
#https://www.kaggle.com/christofhenkel/fasttext-starter-description-only/code

import pandas as pd
from keras.preprocessing import text, sequence
import numpy as np
from tqdm import tqdm
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os

EMBEDDING_FILE = '/home/terrence/CODING/Python/MODELS/AvitoData/cc.ru.300.vec'
TRAIN_CSV = '/home/terrence/CODING/Python/MODELS/AvitoData/train.csv'
TEST_CSV = '/home/terrence/CODING/Python/MODELS/AvitoData/test.csv'

max_features = 100000
maxlen = 100
embed_size = 300

train = pd.read_csv(TRAIN_CSV, index_col = 0)
print(train.shape)
#print(train.head(2))
labels = train[['deal_probability']].copy()
train = train[['description']].copy()

emb = pd.read_csv(EMBEDDING_FILE, index_col = 0)
print(emb.shape)

#print(emb.head(2))

tokenizer = text.Tokenizer(num_words=max_features)
print('fitting tokenizer')

train['description'] = train['description'].astype(str)
tokenizer.fit_on_texts(list(train['description'].fillna('NA').values))

print('getting embeddings')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMBEDDING_FILE)))

print("+++++++++++++++++++++++++++")
#print(embeddings_index.shape)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tqdm(word_index.items()):
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

del embeddings_index
X_train, X_valid, y_train, y_valid = train_test_split(train['description'].values, labels['deal_probability'].values, test_size = 0.1, random_state = 23)
del train
print('convert to sequences')
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)


print('padding')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def build_model():
    inp = Input(shape = (maxlen, ))
    emb = Embedding(nb_words, embed_size, weights = [embedding_matrix],
                    input_length = maxlen, trainable = False)(inp)
    main = SpatialDropout1D(0.2)(emb)
    main = Bidirectional(CuDNNGRU(32,return_sequences = True))(main)
    main = GlobalAveragePooling1D()(main)
    main = Dropout(0.2)(main)
    out = Dense(1, activation = "sigmoid")(main)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error',
                  metrics =[root_mean_squared_error])
    model.summary()
    return model

EPOCHS = 4

model = build_model()
file_path = "model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)
history = model.fit(X_train, y_train, batch_size = 256, epochs = EPOCHS, validation_data = (X_valid, y_valid),
                verbose = 1, callbacks = [check_point])
model.load_weights(file_path)
prediction = model.predict(X_valid)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, prediction)))


test = pd.read_csv(TEST_CSV, index_col = 0)
test = test[['description']].copy()

test['description'] = test['description'].astype(str)
X_test = test['description'].values
X_test = tokenizer.texts_to_sequences(X_test)

print('padding')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
prediction = model.predict(X_test,batch_size = 128, verbose = 1)

sample_submission = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/sample_submission.csv', index_col = 0)
submission = sample_submission.copy()
submission['deal_probability'] = prediction
submission.to_csv('FastText_one.csv')

'''

#+++++++++++++++++++++++++++++++++++++++++++++ xgboost++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
#https://www.kaggle.com/wolfgangb33r/avito-prediction-xgboost-simple

# Simple first attempt to predict the propability of demand
# Not using the image info so far and only taking simple 
# categorical features into account


import numpy as np
import pandas as pd
import math
import time
import os.path
import gc
import random

import xgboost as xgb


def print_duration (start_time, msg):
    print("[%d] %s" % (int(time.time() - start_time), msg))
    start_time = time.time()
    return start_time


start_time = time.time()
print_duration(start_time, "Just Testing") 

msg = "Just Testing"
# quick way of calculating a numeric has for a string
def n_hash(s):
    random.seed(hash(s))
    return random.random()

print(n_hash(msg))
print(hash(msg))
print(random.seed(hash(msg)))
print(random.random())

# hash a complete column of a pandas dataframe    
def hash_column (row, col):
    if col in row:
        return n_hash(row[col])
    return n_hash('none')

def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    

#df["description"]   = df["description"].apply(lambda x: cleanName(x))


train = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv').sample(2000)
test = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv').sample(200)

print(train.shape)
#print(train.head(2))
#print(train.dtypes)

print(test.shape)
#print(test.head(2))
#print(test.dtypes)

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
stopWords = stopwords.words('russian')

#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#n_features = 3
#countvec = CountVectorizer(max_features=n_features, stop_words = stopWords)
#tfidf = TfidfVectorizer(max_features=n_features, stop_words = stopWords)
#countvec_train = np.array(countvec.fit_transform(X_train['description']).todense(), dtype=np.float16)

#CountVectorizer(charset='koi8r', stop_words=stopWords)

count_vectorizer = CountVectorizer(stop_words = stopWords)

start_time = time.time()
# create a xgboost model
model = xgb.XGBRegressor(n_estimators=2, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=3)

#import re
#import string
#re_tok = re.compile(r'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
#re_tok = re.compile(r'([{string.punctuation}“”¨«»®´·!\:/()<>=+#[]{}|º½¾¿¡§£₤‘’])')
#def tokenize(s): return re_tok.sub(r'\1', s).split()
##train['description'] = train['description'].apply(lambda comment: tokenize(comment))

# calculate consistent numeric hashes for any categorical features 
train['user_id'] = train.apply (lambda row: hash_column (row, 'user_id'),axis=1)
#train['user_id'] = count_vectorizer.fit_transform(train['user_id'])
train['region'] = train.apply (lambda row: hash_column (row, 'region'),axis=1)
#train['region'] = count_vectorizer.fit_transform(train['regions'])
train['city'] = train.apply (lambda row: hash_column (row, 'city'),axis=1)
#train['city'] = count_vectorizer.fit_transform(train['city'])
train['parent_category_name'] = train.apply (lambda row: hash_column (row, 'parent_category_name'),axis=1)
#train['parent_category_name'] = count_vectorizer.fit_transform(train['parent_category_name'])
train['category_name'] = train.apply (lambda row: hash_column (row, 'category_name'),axis=1)
#train['category_name'] = count_vectorizer.fit_transform(train['category_name'])
train['user_type'] = train.apply (lambda row: hash_column (row, 'user_type'),axis=1)
#train['user_type'] = count_vectorizer.fit_transform(train['user_type'])
#train['description'] = train.apply (lambda row: hash_column (row, 'description'),axis=1)
#train['description'].fillna(0)
train['description'].fillna('Unknown')
#train['description'] = count_vectorizer.fit_transform(train['description'].apply(lambda comment: tokenize(comment)))
train['description'] = count_vectorizer.fit_transform(train['description'].apply(lambda comment: cleanName(comment)))
train['price'] = np.log(train['price'] + 0.01)
start_time = print_duration (start_time, "Finished reading")      
cleanName
print(train.shape)
print(train.head(2))
print(train.dtypes)

# start training
train_X = train.as_matrix(columns=['user_id', 'price', 'region', 'city', 'parent_category_name', 'category_name',
 'user_type', 'description'])

from sklearn import preprocessing 
for f in train.columns: 
    if train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(train[f].values)) 
        train[f] = lbl.transform(list(train[f].values))

train.fillna((-999), inplace=True) 

train=np.array(train) 
train = train.astype(float) 

train_data = xgb.DMatrix(train_X, train['deal_probability'])
#valid_data = xgb.DMatrix(X_va, y_va)
#del X_tr
#del X_va
#del y_tr
#del y_va
#gc.collect()
#watchlist = [(trAin_data, 'train'), (valid_data, 'valid')]
#model = xgb.train(params, train_data, 1000, watchlist, maximize=False, early_stopping_rounds = 25, verbose_eval=5)
#X_te = xgb.DMatrix(X_te)
model.fit(train_data)
#y_pred = model.predict(X_te, ntree_limit=model.best_ntree_limit)

#model.fit(train_X, train['deal_probability'])

  
# read test data set
#test['user_id'] = test.apply (lambda row: hash_column (row, 'user_id'),axis=1)
test['user_id'] = test.apply (lambda row: hash_column (row, 'user_id'),axis=1)
#test['region'] = count_vectorizer.fit_transform(test['user_id'])
test['region'] = test.apply (lambda row: hash_column (row, 'region'),axis=1)
#test['city'] = count_vectorizer.fit_transform(test['city'])
test['city'] = test.apply (lambda row: hash_column (row, 'city'),axis=1)
#test['parent_category_name'] = count_vectorizer.fit_transform(test['parent_category_name'])
test['parent_category_name'] = test.apply (lambda row: hash_column (row, 'parent_category_name'),axis=1)
#test['category_name'] = count_vectorizer.fit_transform(test['category_name'])
test['category_name'] = test.apply (lambda row: hash_column (row, 'category_name'),axis=1)
#test['user_type'] = count_vectorizer.fit_transform(test['user_type'])
test['user_type'] = test.apply (lambda row: hash_column (row, 'user_type'),axis=1)
#test['description'].fillna(0)
test['description'].fillna('Unknown')
test['description'] = count_vectorizer.fit_transform(test['description'])
#test['description'] = test.apply (lambda row: hash_column (row, 'description'),axis=1)
test['price'] = np.log(test['price'] + 0.01)

for f in test.columns: 
    if test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(test[f].values)) 
        test[f] = lbl.transform(list(test[f].values))

test.fillna((-999), inplace=True)

test=np.array(test) 
test = test.astype(float)

test_X = test.as_matrix(columns=[ 'user_id', 'price', 'region', 'city', 'parent_category_name', 'category_name', 'user_type', 'description'])

start_time = print_duration (start_time, "Finished training, start prediction")   
    # predict the propabilities for binary classes    
pred = model.predict(test_X)
   
start_time = print_duration (start_time, "Finished prediction, start store results")    
submission = pd.read_csv("/home/terrence/CODING/Python/MODELS/AvitoData/sample_submission.csv")
submission['deal_probability'] = pred
print(submission[submission['deal_probability'] > 0])
submission.to_csv("xgb_three.csv", index=False)
start_time = print_duration(start_time, "Finished to store result")

'''



'''
def main():
    start_time = time.time()
    # create a xgboost model
    model = xgb.XGBRegressor(n_estimators=20, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=3)
    
    # load the training data
    train = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv')
    # calculate consistent numeric hashes for any categorical features 
    train['user_hash'] = train.apply (lambda row: hash_column (row, 'user_id'),axis=1)
    train['region_hash'] = train.apply (lambda row: hash_column (row, 'region'),axis=1)
    train['city_hash'] = train.apply (lambda row: hash_column (row, 'city'),axis=1)
    train['parent_category_name_hash'] = train.apply (lambda row: hash_column (row, 'parent_category_name'),axis=1)
    train['category_name_hash'] = train.apply (lambda row: hash_column (row, 'category_name'),axis=1)
    train['user_type_hash'] = train.apply (lambda row: hash_column (row, 'user_type'),axis=1)
    # for the beginning I use only the information if there is an image or not 
    train['image_exists'] = train['image'].isnull().astype(int)
    # calc log for price to reduce effect of very large price differences
    train['price'] = np.log(train['price'] + 0.01)
    #print(train.groupby(['image_exists']).image_exists.count())
    #print(train['image_exists'])
    start_time = print_duration (start_time, "Finished reading")   

    # start training
    train_X = train.as_matrix(columns=['image_top_1', 'user_hash', 'price', 'region_hash', 'city_hash', 'parent_category_name_hash', 'category_name_hash', 'user_type_hash', 'image_exists'])
    model.fit(train_X, train['deal_probability'])
    
    # read test data set
    test = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv')
    test['user_hash'] = test.apply (lambda row: hash_column (row, 'user_id'),axis=1)
    test['region_hash'] = test.apply (lambda row: hash_column (row, 'region'),axis=1)
    test['city_hash'] = test.apply (lambda row: hash_column (row, 'city'),axis=1)
    test['parent_category_name_hash'] = test.apply (lambda row: hash_column (row, 'parent_category_name'),axis=1)
    test['category_name_hash'] = test.apply (lambda row: hash_column (row, 'category_name'),axis=1)
    test['user_type_hash'] = test.apply (lambda row: hash_column (row, 'user_type'),axis=1)
    test['image_exists'] = test['image'].isnull().astype(int)
    test['price'] = np.log(test['price'])
    test_X = test.as_matrix(columns=['image_top_1', 'user_hash', 'price', 'region_hash', 'city_hash', 'parent_category_name_hash', 'category_name_hash', 'user_type_hash', 'image_exists'])
    start_time = print_duration (start_time, "Finished training, start prediction")   
    # predict the propabilities for binary classes    
    pred = model.predict(test_X)
    
    start_time = print_duration (start_time, "Finished prediction, start store results")    
    submission = pd.read_csv("/home/terrence/CODING/Python/MODELS/AvitoData/sample_submission.csv")
    submission['deal_probability'] = pred
    print(submission[submission['deal_probability'] > 0])
    submission.to_csv("xgb_one.csv", index=False)
    start_time = print_duration(start_time, "Finished to store result")
    
if __name__ == '__main__':
    main()
    
'''


#++++++++++++++++++++++++++++++++++++++++++++++++++ lightGBM ++++++++++++++++++++++++++++++++++++++++++++++++++
'''
#https://www.kaggle.com/him4318/avito-lightgbm-with-ridge-feature-v-2-0/code

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("/home/terrence/CODING/Python/MODELS/AvitoData"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

NFOLDS = 5
SEED = 42
VALID = True
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

##ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
   
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
   
def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    
  
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))


print("\nData Load Stage")
training = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv', index_col = "item_id", 
parse_dates = ["activation_date"]).sample(2500)
traindex = training.index
testing = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv', index_col = "item_id", 
parse_dates = ["activation_date"]).sample(500)
testdex = testing.index

ntrain = training.shape[0]
ntest = testing.shape[0]

print(ntrain)
print(ntest)

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
print(kf)

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(df.price.mean(),inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

print(df.shape)

# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
df.drop(["activation_date","image"],axis=1,inplace=True)

print(df.shape)
#print(df.head(2))
#print(df.dtypes)

print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]
print("Encoding :",categorical)


# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))

df.drop(["user_type","image_top_1","param_1","param_2","param_3","item_seq_number","image_top_1","Weekd of Year","Day of Month"],axis=1,inplace=True) # TERRENCE

print("\nText Features")
print(df.shape)
print(df.head(2))

# Feature Engineering 

# Meta Text Features
textfeats = ["description", "title"]
df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df['title'] = df['title'].apply(lambda x: cleanName(x))
df["description"]   = df["description"].apply(lambda x: cleanName(x))

print(df.shape)
print(df.head(2))

for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

print(df.shape)    
print(df.head(2))

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

#print(russian_stop)

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    "min_df":5,
    "max_df":.9,
    "smooth_idf":False
}


def get_col(col_name): return lambda x: x[col_name]
##I added to the max_features of the description. It did not change my score much but it may be worth investigating

vectorizer = FeatureUnion([
        ('description',TfidfVectorizer( ngram_range=(1, 2),
         max_features=17000, 
         #**tfidf_para, 
         preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 2),
            stop_words = russian_stop,
            max_features=7000,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()

import sys
reload(sys)
#sys.setdefaultencoding('utf-8')
#sys.setdefaultencoding('ascii')

print(df.shape)

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
#vectorizer.fit(df.to_dict('records'))
#vectorizer.fit(df)

ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
textfeats = ["description", "title"]
df.drop(textfeats, axis=1,inplace=True)

print(df.shape)

from sklearn.metrics import mean_squared_error
from math import sqrt


ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}


#Ridge oof method from Faron's kernel
#I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
#It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

df['ridge_preds'] = ridge_preds

# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();


print("\nModeling Stage")

del ridge_preds,vectorizer,ready_df
gc.collect();
    
print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves': 270,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.016,
    'verbose': 0
}  

if VALID == False:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=23)
        
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    del X, X_train; gc.collect()
    
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=20000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    print("Model Evaluation Stage")
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
    del X_valid ; gc.collect()

else:
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X, y,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    del X; gc.collect()
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=1550,
        verbose_eval=100
    )



# Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import.png')

print("Model Evaluation Stage")
lgpred = lgb_clf.predict(testing) 

#Mixing lightgbm with ridge. I haven't really tested if this improves the score or not
#blend = 0.95*lgpred + 0.05*ridge_oof_test[:,0]
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgsub.csv",index=True,header=True)
#print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

'''

#++++++++++++++++++++++++++++++++++++ image using keras VGG16 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
#https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("/home/terrence/CODING/Python/MODELS/AvitoData"))

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3


#https://www.kaggle.com/gaborfodor/keras-pretrained-models/data

print(os.listdir("/home/terrence/CODING/Python/MODELS/keras-pretrained-models/"))

from os import listdir, makedirs
from os.path import join, exists, expanduser

#cache_dir = expanduser(join('~', '.keras'))
#if not exists(cache_dir):
#    makedirs(cache_dir)
#models_dir = join(cache_dir, 'models')
#if not exists(models_dir):
#    makedirs(models_dir)

#!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
#!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
#!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/

#!ls ~/.keras/models


#from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
print("----------------- VGG16 summary -----------")
model = VGG16(weights='imagenet', include_top=False)
model.summary()

#!ls ../input/avito-demand-prediction/
#import zipfile
#myzip = zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip')
#files_in_zip = myzip.namelist()
#for idx, file in enumerate(files_in_zip[:5]):
#    if file.endswith('.jpg'):
#        myzip.extract(file, path=file.split('/')[3])
#myzip.close()

#!ls *.jpg
#!ls 856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2.jpg/data/competition_files/train_jpg/

#img_path = '/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/train_jpg/856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2.jpg/data/competition_files/train_jpg/856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2.jpg'
#img_path = '/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/train_jpg/856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2.jpg'
img_path = '/home/terrence/Desktop/PHONE/Camera/20140218_170332.jpg'
#tes = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/train_jpg/*.jpg')

#print(test.shape)


img = image.load_img(img_path, target_size=(224, 224))

#print(img.shape)

x = image.img_to_array(img)  # 3 dims(3, 224, 224)
x = np.expand_dims(x, axis=0)  # 4 dims(1, 3, 224, 224)
x = preprocess_input(x)
print(x.shape)

features = model.predict(x)
print(features.shape)

feat = features.reshape((25088,))
print(feat.shape)

c = 784
n = 32
k =10

feat2 = feat.reshape(n,c)
print(feat2.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=k)
res = pca.fit_transform(feat2)
print(res.shape)

res2 = res.reshape((-1,n*k))
print(res2.shape)

print(res2)
img.show()
'''

#++++++++++++++++++++++++++++++++++++++++++++++++ image features +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
#https://www.kaggle.com/sukhyun9673/extracting-image-features-test

#Thanks to Nooh, who gave an inspiration of im KP extraction : https://www.kaggle.com/c/avito-demand-prediction/discussion/59414#348151

import os
from zipfile import ZipFile
import cv2
import numpy as np
import pandas as pd
#from dask import bag, threaded
#from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
from keras.preprocessing import image


#image_path = "data/competition_files/test_jpg/"
image_path = "/home/terrence/Desktop/PHONE/Camera/"

def keyp(img):
    try:        
        img = image_path + str(img) + ".jpg"
        exfile = zipped.read(img)
        arr = np.frombuffer(exfile, np.uint8)

        imz = cv2.imdecode(arr, 1)
        fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
        kp = fast.detect(imz,None)
        kp =len(kp)
        return kp
    except:
        return 0

x = '20140218_170332' 
print(keyp(x))

#test = pd.read_csv("../input/test.csv")

img_path = '/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/train_jpg/856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img.show()

#test = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/test_jpg')
#print(test.dtypes)

image_path = '/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/train_jpg'
x = '856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2' 
print(keyp(x))

image_path = '/home/terrence/CODING/Python/MODELS/AvitoData/data/competition_files/test_jpg'
x = '856e74b8c46edcf0c0e23444eab019bfda63687bb70a3481955cc6ab86e39df2' 
print(keyp(x))

test = pd.read_csv("/home/terrence/CODING/Python/MODELS/AvitoData/test.csv").sample(2500)
print(test.shape)
print(test.head(2))
#print(test.dtypes)

#images = test[["image"]].drop_duplicates().dropna()
#print(images.shape)

#zipped = ZipFile('../input/test_jpg.zip')

#images["Image_kp_score"] = images["image"].apply(lambda x: keyp(x))
test["Image_kp_score"] = test["image"].apply(lambda x: keyp(x))
print(test.shape)
print(test.head(2))
print(test.dtypes)
print(np.unique(test["Image_kp_score"]))

images.to_csv("Image_KP_SCORES_test.csv", index = False)
'''

#+++++++++++++++++++++++++++++++++++++++++++++++ Boosting MLP +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.kaggle.com/peterhurford/boosting-mlp-lb-0-2297
'''
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from contextlib import contextmanager
from operator import itemgetter
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.sparse import vstack


#@contextmanager
def timer(name):
    t0 = time.time()
    #yield
    #print(f'[{name}] done in {time.time() - t0:.0f} s')
    print('{} done in {}'.format(name,time.time() - t0))


#with timer('reading data'):
#    train = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv').sample(300)
#    test = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv').sample(50)


print("\nData Load Stage")
train = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv').sample(30)
test = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv').sample(5)


print(train.shape)
print(train.head(2))


#with timer('imputation'):
train['param_1'].fillna('missing', inplace=True)
test['param_1'].fillna('missing', inplace=True)

train['param_2'].fillna('missing', inplace=True)
test['param_2'].fillna('missing', inplace=True)

train['param_3'].fillna('missing', inplace=True)
test['param_3'].fillna('missing', inplace=True)

train['image_top_1'].fillna(0, inplace=True)
test['image_top_1'].fillna(0, inplace=True)

train['price'].fillna(0, inplace=True)
test['price'].fillna(0, inplace=True)

train['price'] = np.log1p(train['price'])
test['price'] = np.log1p(test['price'])

price_mean = train['price'].mean()
price_std = train['price'].std()

train['price'] = (train['price'] - price_mean) / price_std
test['price'] = (test['price'] - price_mean) / price_std

train['description'].fillna('', inplace=True)
test['description'].fillna('', inplace=True)

# City names are duplicated across region, HT: Branden Murray https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
train['city'] = train['city'] + '_' + train['region']
test['city'] = test['city'] + '_' + test['region']


#with timer('add new features'):
cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type']
num_cols = ['price', 'deal_probability']
for c in cat_cols:
    for c2 in num_cols:
        enc = train.groupby(c)[c2].agg(['mean']).astype(np.float32).reset_index()
        enc.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc.columns]
        train = pd.merge(train, enc, how='left', on=c)
        test = pd.merge(test, enc, how='left', on=c)
del(enc)

print(train.shape)
print(train.head(2))


from sklearn.model_selection import train_test_split
##df = train
#def preprocess(df: pd.DataFrame) -> pd.DataFrame:
def preprocess(df):
    ex_col = ['item_id', 'user_id', 'deal_probability', 'title', 'param_1', 'param_2', 'param_3', 'activation_date']
    df['description_len'] = df['description'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['description_wc'] = df['description'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['description'] = (df['parent_category_name'] + ' ' + df['category_name'] + ' ' + df['param_1'] + ' ' + df['param_2'] + ' ' + df['param_3'] + ' ' +
                        df['title'] + ' ' + df['description'].fillna(''))
    df['description'] = df['description'].str.lower().replace(r"[^[:alpha:]]", " ")
    df['description'] = df['description'].str.replace(r"\\s+", " ")
    df['title_len'] = df['title'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['title_wc'] = df['title'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['image'] = df['image'].map(lambda x: 1 if len(str(x))>0 else 0)
    #df['price'] = np.log1p(df['price'].fillna(0))
    df['price'] = df['price'].fillna(0)
    df['wday'] = pd.to_datetime(df['activation_date']).dt.dayofweek
    col = [c for c in df.columns if c not in ex_col]
    return df[col]

#df = preprocess(train)
#print(df.shape)
#print(df.head(2))

#with timer('process train'):
train, valid = train_test_split(train, test_size=0.05, shuffle=True, random_state=37)
y_train = train['deal_probability'].values
X_train = preprocess(train)
#print(f'X_train: {X_train.shape}')
print(X_train.shape)

#with timer('process valid'):
X_valid = preprocess(valid)
#print(f'X_valid: {X_valid.shape}')

#with timer('process test'):
X_test = preprocess(test)
#print(f'X_test: {X_test.shape}')

print(X_train.head(2))
print(X_train.shape)

# Do some normalization

desc_len_mean = X_train['description_len'].mean()
desc_len_std = X_train['description_len'].std()

X_train.is_copy = False
X_valid.is_copy = False
X_test.is_copy = False

X_train['description_len'] =  (X_train['description_len'] - desc_len_mean) / desc_len_std
X_valid['description_len'] = (X_valid['description_len'] - desc_len_mean) / desc_len_std
X_test['description_len'] = (X_test['description_len'] - desc_len_mean) / desc_len_std

desc_wc_mean = X_train['description_wc'].mean()
desc_wc_std = X_train['description_wc'].std()
X_train['description_wc'] = (X_train['description_wc'] - desc_wc_mean) / desc_wc_std
X_valid['description_wc'] = (X_valid['description_wc'] - desc_wc_mean) / desc_wc_std
X_test['description_wc'] = (X_test['description_wc'] - desc_wc_mean) / desc_wc_std

title_len_mean = X_train['title_len'].mean()
title_len_std = X_train['title_len'].std()
X_train['title_len'] = (X_train['title_len'] - title_len_mean) / title_len_std
X_valid['title_len'] = (X_valid['title_len'] - title_len_mean) / title_len_std
X_test['title_len'] = (X_test['title_len'] - title_len_mean) / title_len_std

title_wc_mean = X_train['title_wc'].mean()
title_wc_std = X_train['title_wc'].std()
X_train['title_wc'] = (X_train['title_wc'] - title_wc_mean) / title_wc_std
X_valid['title_wc'] = (X_valid['title_wc'] - title_wc_mean) / title_wc_std
X_test['title_wc'] = (X_test['title_wc'] - title_wc_mean) / title_wc_std

image_top_1_mean = X_train['image_top_1'].mean()
image_top_1_std = X_train['image_top_1'].std()
X_train['image_top_1'] = (X_train['image_top_1'] - image_top_1_mean) / image_top_1_std
X_valid['image_top_1'] = (X_valid['image_top_1'] - image_top_1_mean) / image_top_1_std
X_test['image_top_1'] = (X_test['image_top_1'] - image_top_1_mean) / image_top_1_std


# I don't know why I need to fill NA a second time, but alas here we are...
X_train.fillna(0, inplace=True)
X_valid.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

#print(X_train.columns)
print(X_train.head(2))
print(X_train.shape)


#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

#with timer('CountVec'):
count_vec = CountVectorizer(ngram_range=(1, 2),
                        max_features=100000,
                        token_pattern='\w+',
                        encoding='KOI8-R')
countvec_train = count_vec.fit_transform(X_train['description'])
countvec_valid = count_vec.transform(X_valid['description'])
countvec_test = count_vec.transform(X_test['description'])

print(X_train.columns)


#with timer('TFIDF'):
tfidf = TfidfVectorizer(ngram_range=(1, 2),
                        max_features=100000,
                        token_pattern='\w+',
                        encoding='KOI8-R')
tfidf_train = tfidf.fit_transform(X_train['description'])
tfidf_valid = tfidf.transform(X_valid['description'])
tfidf_test = tfidf.transform(X_test['description'])

#print(X_train.shape)
#print(X_train.head(2))

#with timer('Dummy'):
dummy_cols = ['parent_category_name', 'category_name', 'user_type', 'image_top_1', 'wday', 'region', 'city']
for col in dummy_cols:
    le = LabelEncoder()
    le.fit(X_train[col] + X_valid[col] + X_test[col])
    le.fit(list(X_train[col].values.astype('str')) + list(X_valid[col].values.astype('str')) + list(X_test[col].values.astype('str')))
    X_train[col] = le.transform(list(X_train[col].values.astype('str')))
    X_valid[col] = le.transform(list(X_valid[col].values.astype('str')))
    X_test[col] = le.transform(list(X_test[col].values.astype('str')))

print(X_train.shape)
#print(X_train.head(2))

#with timer('Dropping'):
X_train.drop('description', axis=1, inplace=True)
X_valid.drop('description', axis=1, inplace=True)
X_test.drop('description', axis=1, inplace=True)


#with timer('OHE'):
ohe = OneHotEncoder(categorical_features=[X_train.columns.get_loc(c) for c in dummy_cols])
X_train = ohe.fit_transform(X_train)
#print(f'X_train: {X_train.shape}')
X_valid = ohe.transform(X_valid)
#print(f'X_valid: {X_valid.shape}')
X_test = ohe.transform(X_test)
#print(f'X_test: {X_test.shape}')

#print(X_train.head(2))
print(X_train.shape)


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)



def fit_predict(xs, y_train, loss_fn='mean_squared_error'): # -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=4, use_per_session_threads=4, inter_op_parallelism_threads=4)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss=loss_fn, optimizer=ks.optimizers.Adam(lr=2e-3))
        for i in range(3):
            #with timer(f'epoch {i + 1}'):
            model.fit(x=X_train, y=y_train, batch_size=2**(8 + i), epochs=1, verbose=0)
        return model.predict(X_test, batch_size=2**(8 + i))[:, 0]


X_train = X_train.tocsr()
X_valid = X_valid.tocsr()
X_test = X_test.tocsr()
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size = 0.5, shuffle = False)    


preds_oofs = []
preds_valids = []
preds_tests = []
for r in range(8):
    with timer('Round {}'.format(r)):
        if r % 2 == 0:
            loss_name = 'huber_loss'
            loss = huber_loss
        else:
            loss_name = 'mean_squared_error'
            loss = 'mean_squared_error'
        if r >= 4:
            print('Running loss = {}, binary = True'.format(loss_name))
            xs = [x.astype(np.bool).astype(np.float32) for x in [X_train_1, X_train_2]]
            y_pred1 = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
            xs = [x.astype(np.bool).astype(np.float32) for x in [X_train_2, X_train_1]]
            y_pred2 = fit_predict(xs, y_train=y_train_2, loss_fn=loss)
            xs = [x.astype(np.bool).astype(np.float32) for x in [X_train_1, X_valid]]
            y_predf = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
            xs = [x.astype(np.bool).astype(np.float32) for x in [X_train_1, X_test]]
            y_predt = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
        else:
            print('Running loss = {}, binary = False'.format(loss_name))
            xs = [X_train_1, X_train_2]
            y_pred1 = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
            xs = [X_train_2, X_train_1]
            y_pred2 = fit_predict(xs, y_train=y_train_2, loss_fn=loss)
            xs = [X_train_1, X_valid]
            y_predf = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
            xs = [X_train_1, X_test]
            y_predt = fit_predict(xs, y_train=y_train_1, loss_fn=loss)
        preds_oof = np.concatenate((y_pred2, y_pred1), axis=0)
        preds_valid = y_predf
        preds_test = y_predt
        print('Round {} OOF RMSE: {:.4f}'.format(r, np.sqrt(mean_squared_error(train['deal_probability'], preds_oof))))
        print('Round {} Valid RMSE: {:.4f}'.format(r, np.sqrt(mean_squared_error(valid['deal_probability'], preds_valid))))
        preds_oofs.append(preds_oof)
        preds_valids.append(preds_valid)
        preds_tests.append(preds_test)


preds_oof = np.mean(preds_oofs, axis=0)
print('Overall OOF RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(train['deal_probability'], preds_oof))))
preds_valid = np.mean(preds_valids, axis=0)
print('Overall Valid RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(valid['deal_probability'], preds_valid))))


# As we can see, the individual submodels have very low correlation with each other!
import numpy as np
np.mean(np.corrcoef(preds_oofs), axis=0)

with timer('reading data'):
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    target = train['deal_probability']

with timer('imputation'):
    train['param_1'].fillna('missing', inplace=True)
    test['param_1'].fillna('missing', inplace=True)
    train['param_2'].fillna('missing', inplace=True)
    test['param_2'].fillna('missing', inplace=True)
    train['param_3'].fillna('missing', inplace=True)
    test['param_3'].fillna('missing', inplace=True)
    train['price'].fillna(0, inplace=True)
    test['price'].fillna(0, inplace=True)
    # City names are duplicated across region, HT: Branden Murray https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
    train['city'] = train['city'] + '_' + train['region']
    test['city'] = test['city'] + '_' + test['region']

with timer('FE'):
    trainp = preprocess(train)
    testp = preprocess(test)
    for col in ['param_1', 'param_2', 'param_3']:
        trainp[col] = train[col]
        testp[col] = test[col]

print(train.shape)
train.head()


with timer('drop'):
    trainp.drop(['description', 'image'], axis=1, inplace=True)
    testp.drop(['description', 'image'], axis=1, inplace=True)
print(trainp.shape)
print(testp.shape)


with timer('To cat'):
    trainp['image_top_1'] = trainp['image_top_1'].astype('str').fillna('missing')
    testp['image_top_1'] = testp['image_top_1'].astype('str').fillna('missing') # My pet theory is that image_top_1 is categorical. Fight me.
    cat_cols = ['region', 'city', 'parent_category_name', 'category_name',
                'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1', 'wday']
    for col in trainp.columns:
        print(col)
        if col in cat_cols:
            trainp[col] = trainp[col].astype('category')
            testp[col] = testp[col].astype('category')
        else:
            trainp[col] = trainp[col].astype(np.float64)
            testp[col] = testp[col].astype(np.float64)


print(trainp.shape)
print(trainp.columns)
trainp.head()

trainp.dtypes


with timer('Split'):
    train, valid, y_train, y_valid = train_test_split(trainp, target, test_size=0.05, shuffle=True, random_state=37)
    test = testp
    print(train.shape)
    print(valid.shape)
    print(test.shape)


with timer('Submodels'):
    train_models = pd.DataFrame(np.array(preds_oofs).transpose())
    valid_models = pd.DataFrame(np.array(preds_valids).transpose())
    test_models = pd.DataFrame(np.array(preds_tests).transpose())
    train_models.columns = ['nn_' + str(i + 1) for i in range(train_models.shape[1])]
    valid_models.columns = ['nn_' + str(i + 1) for i in range(train_models.shape[1])]
    test_models.columns = ['nn_' + str(i + 1) for i in range(train_models.shape[1])]
    print(train_models.shape)
    print(valid_models.shape)
    print(test_models.shape)


with timer('Concat'):
    print(train.shape)
    X_train = pd.concat([train.reset_index(), train_models.reset_index()], axis=1)
    print(X_train.shape)
    print('-')
    print(valid.shape)
    X_valid = pd.concat([valid.reset_index(), valid_models.reset_index()], axis=1)
    print(X_valid.shape)
    print('-')
    print(test.shape)
    X_test = pd.concat([test.reset_index(), test_models.reset_index()], axis=1)
    print(X_test.shape)

X_train.head()

X_train.drop('index', axis=1, inplace=True)
X_valid.drop('index', axis=1, inplace=True)
X_test.drop('index', axis=1, inplace=True)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)


#del X_train_1
#del X_train_2
del trainp
del testp
gc.collect()


from pprint import pprint
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid)
watchlist = [d_train, d_valid]
params = {'application': 'regression',
          'metric': 'rmse',
          'nthread': 3,
          'verbosity': -1,
          'data_random_seed': 3,
          'learning_rate': 0.05,
          'num_leaves': 31,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.2,
          'lambda_l1': 3,
          'lambda_l2': 3,
          'min_data_in_leaf': 40}
model = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=1500,
                  valid_sets=watchlist,
                  verbose_eval=100)
pprint(sorted(list(zip(model.feature_importance(), X_train.columns)), reverse=True))
print('Done')


valid_preds = model.predict(X_valid).clip(0, 1)
print('Overall Valid RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(y_valid, valid_preds))))
test_preds = model.predict(X_test).clip(0, 1)
submission = pd.read_csv('../input/test.csv', usecols=["item_id"])
submission["deal_probability"] = test_preds
submission.to_csv("submit_boosting_mlp.csv", index=False, float_format="%.2g")

submission.head()
'''





#++++++++++++++++++++++++++++++++++++ Russian word embedding +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.kaggle.com/gunnvant/russian-word-embeddings-for-fun-and-for-profit


import os
import pandas as pd
import numpy as np
import glob
import nltk
import gensim
'''
train = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv').sample(2500)
test = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv').sample(500)

print(train.head(2))

#from gensim.models import KeyedVectors

print("here we are")

import gensim
#model = gensim.models.Word2Vec.load_word2vec_format('/home/terrence/CODING/Python/MODELS/AvitoData/wiki.ru.vec', binary=False)
model = gensim.models.KeyedVectors.load_word2vec_format('/home/terrence/CODING/Python/MODELS/AvitoData/wiki.ru.vec', binary=False)
#        gensim.models.KeyedVectors.load_word2vec_format
# if you vector file is in binary format, change to binary=True
#sentence = ["London", "is", "the", "capital", "of", "Great", "Britain"]
#vectors = [model[w] for w in sentence]

find_similar_to = 'Автомобили'.lower()

print(model.similar_by_word(find_similar_to))

#ru_model = KeyedVectors.load_word2vec_format('/home/terrence/CODING/Python/MODELS/AvitoData/wiki.ru.vec')
#print("The size of vocabulary for this corpus is {}".format(len(ru_model.vocab)))


# Pick a word 
find_similar_to = 'Автомобили'.lower()
ru_model.similar_by_word(find_similar_to)

import nltk
def tokenize(x):
    #Input: One description
    tok=nltk.tokenize.toktok.ToktokTokenizer()
    return [t.lower() for t in tok.tokenize(x)]
def get_vector(x):
    #Input: Single token #If the word is out of vocab, then return a 300 dim vector filled with zeros
    try:
        return ru_model.get_vector(x)
    except:
        return np.zeros(shape=300)
def vector_sum(x):
    #Input:List of word vectors
    return np.sum(x,axis=0)

    features=[]
for desc in train['description'].values:
    tokens=tokenize(desc)
    if len(tokens)!=0: ## If the description is missing then return a 300 dim vector filled with zeros
        word_vecs=[get_vector(w) for w in tokens]
        features.append(vector_sum(word_vecs))
    else:
        features.append(np.zeros(shape=300))                 


print("Features were extracted from {} rows".format(len(features)))

## Convert into numpy array
train_desc_features=np.array(features)
print("Shape of features extracted from 'Description' column is:")
print(train_desc_features.shape)

'''



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.kaggle.com/paulorzp/tfidf-tensor-starter-lb-0-233

# ref: https://github.com/pjankiewicz/mercari-solution/blob/master/mercari_golf.py

'''
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from contextlib import contextmanager
from operator import itemgetter
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.sparse import vstack
from nltk.corpus import stopwords
sw = stopwords.words('russian')


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    ex_col = ['item_id', 'user_id', 'deal_probability', 'title', 'param_1', 'param_2', 'param_3', 'activation_date']
    df['description_len'] = df['description'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['description_wc'] = df['description'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['description'] = (df['title'] + ' ' + df['description'].fillna('') + ' ' + df['city'] + ' ' + df['param_1'].fillna(''))
    df['description'] = df['description'].str.lower().replace(r"[^[:alpha:]]", " ")
    df['description'] = df['description'].str.replace(r"\\s+", " ")
    df['categ'] = (df['category_name'] + ' ' + df['parent_category_name'] + ' ' + df['param_2'].fillna('') + ' ' + df['param_3'].fillna(''))
    df['title_len'] = df['title'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['title_wc'] = df['title'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['image'] = df['image'].map(lambda x: 1 if len(str(x))>0 else 0)
    df['price'] = np.log1p(df['price'].fillna(0))
    df['wday'] = pd.to_datetime(df['activation_date']).dt.dayofweek
    df['day'] = pd.to_datetime(df['activation_date']).dt.day
    df['week'] = pd.to_datetime(df['activation_date']).dt.week
    col = [c for c in df.columns if c not in ex_col]
    return df[col]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=2e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        return model.predict(X_test, batch_size=2**(11 + i))[:, 0]


def main():
    vectorizer = make_union(
        on_field('description', Tfidf(max_features=3500, stop_words=sw, token_pattern='\w+', norm='l2',
                                    min_df=3, max_df=0.3, sublinear_tf=True, ngram_range=(1, 3))),
        on_field('categ', Tfidf(max_features=2000, stop_words=sw, token_pattern='\w+', norm='l2',
                                    min_df=2, max_df=0.3, sublinear_tf=True, ngram_range=(1, 3))),
        on_field(['region', 'user_type'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=1)
    with timer('reading data '):
        dtypes = {
        'region': 'category',
        'item_seq_number': 'uint32',
        'user_type': 'category',
        'image_top_1': 'float32',
        'price':'float32',
        'deal_probability': 'float32'
        }
        train = pd.read_csv('../input/train.csv', dtype=dtypes)
        test = pd.read_csv('../input/test.csv', dtype=dtypes)
    with timer('add new features'):
        cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type']
        num_cols = ['price', 'image_top_1', 'deal_probability']
        for c in cat_cols:
            for c2 in num_cols:
                enc = train.groupby(c)[c2].agg(['mean']).astype(np.float32).reset_index()
                enc.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc.columns]
                train = pd.merge(train, enc, how='left', on=c)
                test = pd.merge(test, enc, how='left', on=c)
        del(enc)

with timer('process train'):
        cv = KFold(n_splits=20, shuffle=True, random_state=37)
        train_ids, valid_ids = next(cv.split(train))
        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        y_train = train['deal_probability'].values
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        del train
    with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    gc.collect()
    print('train shape',X_train.shape)
    print('valid shape',X_valid.shape)
    with timer('process test'):
        X_test = vectorizer.transform(preprocess(test)).astype(np.float32)
        del test
        gc.collect()
    print('test shape',X_test.shape)
    
    valid_length = X_valid.shape[0]
    X_valid = vstack([X_valid, X_test])
    del(X_test)
    gc.collect()
    xs = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
    del(X_train, X_valid)
    gc.collect()    
    y_pred = fit_predict(xs, y_train=y_train)
    test_pred = y_pred[valid_length:]
    y_pred = y_pred[:valid_length]
    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_error(valid['deal_probability'], y_pred))))
    submission = pd.read_csv(f'../input/test.csv', usecols=["item_id"])
    submission["deal_probability"] = test_pred.clip(0,1)
    submission.to_csv("tensor_starter2.csv", index=False, float_format="%.5g")

if __name__ == '__main__':
    main()

'''





#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#https://www.kaggle.com/dicksonchin93/xgb-with-mean-encode-tfidf-feature-0-232
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("/home/terrence/CODING/Python/MODELS/AvitoData"))

# Any results you write to the current directory are saved as output.


tr = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv')
te = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv')
print('train data shape is :', tr.shape)
print('test data shape is :', te.shape)

data = pd.concat([tr, te], axis=0)

print(tr.shape)
#print(tr.head(2))

print(data.shape)


from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm

data.activation_date = pd.to_datetime(data.activation_date)
tr.activation_date = pd.to_datetime(tr.activation_date)

data['day_of_month'] = data.activation_date.apply(lambda x: x.day)
data['day_of_week'] = data.activation_date.apply(lambda x: x.weekday())

tr['day_of_month'] = tr.activation_date.apply(lambda x: x.day)
tr['day_of_week'] = tr.activation_date.apply(lambda x: x.weekday())

data['char_len_title'] = data.title.apply(lambda x: len(str(x)))
data['char_len_desc'] = data.description.apply(lambda x: len(str(x)))


agg_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'image_top_1', 'user_type','item_seq_number','day_of_month','day_of_week'];
for c in tqdm(agg_cols):
    gp = tr.groupby(c)['deal_probability']
    mean = gp.mean()
    std  = gp.std()
    data[c + '_deal_probability_avg'] = data[c].map(mean)
    data[c + '_deal_probability_std'] = data[c].map(std)

for c in tqdm(agg_cols):
    gp = tr.groupby(c)['price']
    mean = gp.mean()
    data[c + '_price_avg'] = data[c].map(mean)

print(data.shape)
print(data.head(2))


cate_cols = ['city',  'category_name', 'user_type',]

for c in cate_cols:
    data[c] = LabelEncoder().fit_transform(data[c].values)

print(data.shape)

from nltk.corpus import stopwords
stopWords = stopwords.words('russian')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
data['description'] = data['description'].fillna(' ')

tfidf = TfidfVectorizer(max_features=100, stop_words = stopWords)

tfidf_train = np.array(tfidf.fit_transform(data['description']).todense(), dtype=np.float16)
for i in range(100):
    data['tfidf_' + str(i)] = tfidf_train[:, i]


new_data = data.drop(['user_id','description','image','parent_category_name','region',
                      'item_id','param_1','param_2','param_3','title'], axis=1)

print(data.shape)
print(new_data.shape)

import gc
del data
del tr
del te
gc.collect()


from sklearn.model_selection import train_test_split

X = new_data.loc[new_data.activation_date<=pd.to_datetime('2017-04-07')]
X_te = new_data.loc[new_data.activation_date>=pd.to_datetime('2017-04-08')]

y = X['deal_probability']
X = X.drop(['deal_probability','activation_date'],axis=1)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=2018)
X_te = X_te.drop(['deal_probability','activation_date'],axis=1)

print(X_tr.shape, X_va.shape, X_te.shape)


#del X
#del y
#gc.collect()

'''


'''
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBRegressor(
        n_jobs = 1,
        objective = 'regression',
        eval_metric = 'rmse',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
    
'''



'''

import xgboost as xgb

params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'reg:logistic', 
          'eval_metric': 'rmse', 
          'random_state': 99, 
          'silent': True}

tr_data = xgb.DMatrix(X_tr, y_tr)
va_data = xgb.DMatrix(X_va, y_va)
del X_tr
del X_va
del y_tr
del y_va
gc.collect()

watchlist = [(tr_data, 'train'), (va_data, 'valid')]

model = xgb.train(params, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 25, verbose_eval=5)

X_te = xgb.DMatrix(X_te)
y_pred = model.predict(X_te, ntree_limit=model.best_ntree_limit)
sub = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv('xgb_with_mean_encode_and_nlp.csv', index=False)
sub.head()


from xgboost import plot_importance
import matplotlib.pyplot as plt
plot_importance(model)
#plt.gcf().savefig('feature_importance_xgb.png')
'''


#++++++++++++++++++++++++++++++++++++++++++++++++ Mwisho kabisa +++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import pandas as pd
import numpy as np
import xgboost as xgb


print("\nData Load Stage")
training = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv', index_col = "item_id",
 parse_dates = ["activation_date"]) #.sample(100)
traindex = training.index
#print(traindex)
testing = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv', index_col = "item_id",
 parse_dates = ["activation_date"]) #.sample(10)
testdex = testing.index
#print(testdex)
print(training.shape)
print(testing.shape)

88

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
print("+++++++++++++++++++++ step 1 combine train and test ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#print(training.head(1))

# Combine Train and Test
df = pd.concat([training,testing],axis=0)
#del training, testing

print(df.shape)
#print(df.columns)

print("+++++++++++++++++++++++++ step 2 fill missing entries ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

df['param_1'].fillna('missing', inplace=True)
df['param_2'].fillna('missing', inplace=True)
df['param_3'].fillna('missing', inplace=True)
df['image_top_1'].fillna(0, inplace=True)
df['price'].fillna(0, inplace=True)
df['price'] = np.log1p(df['price'])
price_mean = df['price'].mean()
price_std = df['price'].std()
df['price'] = (df['price'] - price_mean) / price_std
df['description'].fillna('', inplace=True)
df['city'] = df['city'] + '_' + df['region']
df.drop("region",axis=1, inplace=True) #TERRENCE

print(df.shape)
#print(df.columns)

print("+++++++++++++++++++++++++++++++++ step 3 break back to train and test ++++++++++++++++++++++++++++++++++++++++++++++")
train = df.loc[traindex,:].copy()
#train = pd.concat([train, y], axis=1) 
#print("Training Set shape",train.shape)
test = df.loc[testdex,:].copy()
#print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
#del df
#gc.collect()

print(train.shape)
print(test.shape)
#print(test.columns)

print("+++++step 4 feature engineering the train and test by the price numerical variables category average +++++++++++++++++++++++++++++++")

#cat_cols = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type']
cat_cols = ['city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3', 'user_type']
num_cols = ['price'] #, 'deal_probability']
for c in cat_cols:
    for c2 in num_cols:
        enc = train.groupby(c)[c2].agg(['mean']).astype(np.float32).reset_index()
        enc2 = test.groupby(c)[c2].agg(['mean']).astype(np.float32).reset_index() #TERRENCE
        enc.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc.columns]
        enc2.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc2.columns] #TERRENCE
        train = pd.merge(train, enc, how='left', on=c)
        test = pd.merge(test, enc2, how='left', on=c)
del(enc)
del(enc2)

print(train.shape)
print(test.shape)
#print(train.columns)


print("+++++++++++++++++++++++++++++ step 5 training and validation split +++++++++++++++++++++++++++++++++++++++++++++++++++++++")
from sklearn.model_selection import train_test_split

# Training and Validation Set

#X_train, X_valid, y_train, y_valid = train_test_split(train,y, test_size=0.010, random_state=23)
X_train = train
y_train = y 

print("++++++++++++++++++++ step 6 feature engineering on categorical variables ++++++++++++++++++++++++++++++++++++++++++++")


#def preprocess(df: pd.DataFrame) -> pd.DataFrame:
def preprocess(df):
    df.is_copy = False
    ex_col = ['item_id', 'user_id', 'deal_probability', 'title', 'param_1', 'param_2', 'param_3', 'activation_date']
    df['description_len'] = df['description'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['description_wc'] = df['description'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['description'] = (df['parent_category_name'] + ' ' + df['category_name'] + ' ' + df['param_1'] + ' ' + df['param_2'] + ' ' + df['param_3'] + ' ' +
                        df['title'] + ' ' + df['description'].fillna(''))
    df['description'] = df['description'].str.lower().replace(r"[^[:alpha:]]", " ")
    df['description'] = df['description'].str.replace(r"\\s+", " ")
    df['title_len'] = df['title'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
    df['title_wc'] = df['title'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
    df['image'] = df['image'].map(lambda x: 1 if len(str(x))>0 else 0)
    #df['price'] = np.log1p(df['price'].fillna(0))
    df['price'] = df['price'].fillna(0)
    df['wday'] = pd.to_datetime(df['activation_date']).dt.dayofweek
    col = [c for c in df.columns if c not in ex_col]
    return df[col]


X_train = preprocess(X_train)
#X_valid = preprocess(X_valid)
X_test = preprocess(test)

print(X_train.shape)
#print(X_valid.shape)
print(X_test.shape)

#print(X_train.columns)
'''


'''
desc_len_mean = X_train['description_len'].mean()
desc_len_std = X_train['description_len'].std()

X_train.is_copy = False
X_valid.is_copy = False
X_test.is_copy = False

X_train['description_len'] =  (X_train['description_len'] - desc_len_mean) / desc_len_std
X_valid['description_len'] = (X_valid['description_len'] - desc_len_mean) / desc_len_std
X_test['description_len'] = (X_test['description_len'] - desc_len_mean) / desc_len_std

desc_wc_mean = X_train['description_wc'].mean()
desc_wc_std = X_train['description_wc'].std()
X_train['description_wc'] = (X_train['description_wc'] - desc_wc_mean) / desc_wc_std
X_valid['description_wc'] = (X_valid['description_wc'] - desc_wc_mean) / desc_wc_std
X_test['description_wc'] = (X_test['description_wc'] - desc_wc_mean) / desc_wc_std

title_len_mean = X_train['title_len'].mean()
title_len_std = X_train['title_len'].std()
X_train['title_len'] = (X_train['title_len'] - title_len_mean) / title_len_std
X_valid['title_len'] = (X_valid['title_len'] - title_len_mean) / title_len_std
X_test['title_len'] = (X_test['title_len'] - title_len_mean) / title_len_std


title_wc_mean = X_train['title_wc'].mean()
title_wc_std = X_train['title_wc'].std()
X_train['title_wc'] = (X_train['title_wc'] - title_wc_mean) / title_wc_std
X_valid['title_wc'] = (X_valid['title_wc'] - title_wc_mean) / title_wc_std
X_test['title_wc'] = (X_test['title_wc'] - title_wc_mean) / title_wc_std

image_top_1_mean = X_train['image_top_1'].mean()
image_top_1_std = X_train['image_top_1'].std()
X_train['image_top_1'] = (X_train['image_top_1'] - image_top_1_mean) / image_top_1_std
X_valid['image_top_1'] = (X_valid['image_top_1'] - image_top_1_mean) / image_top_1_std
X_test['image_top_1'] = (X_test['image_top_1'] - image_top_1_mean) / image_top_1_std
'''

'''

# I don't know why I need to fill NA a second time, but alas here we are...
#X_train.fillna(0, inplace=True)
#X_valid.fillna(0, inplace=True)
#X_test.fillna(0, inplace=True)

#print(X_train.columns)

print("+++++++++++++++++++++++++++++++++ vectorize the datasets ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# CountVec
##count_vec = CountVectorizer(ngram_range=(1, 2),
##                        max_features=100000,
##                        token_pattern='\w+',
##                        encoding='KOI8-R')
##countvec_train = count_vec.fit_transform(X_train['description'])
##countvec_valid = count_vec.transform(X_valid['description'])
##countvec_test = count_vec.transform(X_test['description'])

#print(X_train.columns)

#   TFIDF
##tfidf = TfidfVectorizer(ngram_range=(1, 2),
##                        max_features=100000,
##                        token_pattern='\w+',
##                        encoding='KOI8-R')
##tfidf_train = tfidf.fit_transform(X_train['description'])
##tfidf_valid = tfidf.transform(X_valid['description'])
##tfidf_test = tfidf.transform(X_test['description'])


from nltk.corpus import stopwords
stopWords = stopwords.words('russian')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
n_features = 3

countvec = CountVectorizer(max_features=n_features, stop_words = stopWords)
tfidf = TfidfVectorizer(max_features=n_features, stop_words = stopWords)

countvec_train = np.array(countvec.fit_transform(X_train['description']).todense(), dtype=np.float16)
#countvec_valid = np.array(countvec.fit_transform(X_valid['description']).todense(), dtype=np.float16)
countvec_test = np.array(countvec.fit_transform(X_test['description']).todense(), dtype=np.float16)

#tfidf_train = np.array(tfidf.fit_transform(X_train['description']).todense(), dtype=np.float16)
##tfidf_valid = np.array(tfidf.fit_transform(X_valid['description']).todense(), dtype=np.float16)
#tfidf_test = np.array(tfidf.fit_transform(X_test['description']).todense(), dtype=np.float16)


X_train.is_copy = False

X_test.is_copy = False

for i in range(n_features):
    X_train['countvec_' + str(i)] = countvec_train[:, i]
    #X_valid['countvec_' + str(i)] = countvec_valid[:, i]
    X_test['countvec_' + str(i)] = countvec_test[:, i]

#for i in range(n_features):
#    X_train['tfidf_' + str(i)] = tfidf_train[:, i]
#    #X_valid['tfidf_' + str(i)] = tfidf_valid[:, i]
#    X_test['tfidf_' + str(i)] = tfidf_test[:, i]

#new_data = data.drop(['user_id','description','image','parent_category_name','region',
#                      'item_id','param_1','param_2','param_3','title'], axis=1)

#region, city, parent_category_name, category_name, description, user_type

X_train = X_train.drop(['description','parent_category_name','category_name','city','user_type'],axis =1 ) #,'param_1','param_2','param_3','title'], axis=1,inplace=True)
#X_valid = X_valid.drop(['description','parent_category_name','category_name','city','user_type'],axis =1) #,'param_1','param_2','param_3','title'], axis=1)
X_test = X_test.drop(['description','parent_category_name','category_name','city','user_type'],axis =1) #,'param_1','param_2','param_3','title'], axis=1)

#X_train.drop('description', axis=1, inplace=True)
#X_valid.drop('description', axis=1, inplace=True)
#X_test.drop('description', axis=1, inplace=True)

print(X_train.shape)
#print(X_valid.shape)
print(X_test.shape)

print(X_train.columns)
#print(X_train.head(2))
print("=========================================")
print(X_test.columns)



#---------------- XGBoost ----------------------------

# create a xgboost model
model = xgb.XGBRegressor(n_estimators=5, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=3)

# start training
#train_X = train.as_matrix(columns=['user_id', 'price', 'region', 'city', 'parent_category_name', 'category_name', 'user_type', 'description'])

#train_X = X_train.as_matrix()

#model.fit(train_X, y_train)
model.fit(X_train, y_train)

#test_X = X_test.as_matrix()
#test_X1 = np.array(test_X.reshape(test_X.size), copy=False) #, dtype=np.float32)
pred = model.predict(X_test)

submission = pd.read_csv("/home/terrence/CODING/Python/MODELS/AvitoData/sample_submission.csv")
submission['deal_probability'] = pred
submission['deal_probability'].clip(0.0, 1.0, inplace=True)
print(submission[submission['deal_probability'] > 0])
submission.to_csv("xgb_mwisho_one.csv", index=False)
#submission.to_csv("xgb_mwisho_trash.csv", index=False)

'''

'''
catpred = cb_model.predict(test)
catsub = pd.DataFrame(catpred,columns=["deal_probability"],index=testdex)
catsub['deal_probability'].clip(0.0, 1.0, inplace=True)
catsub.to_csv("catsub_two.csv",index=True,header=True) # Between 0 and 1
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

'''


'''
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test) #, label=y_test)

model.fit(X_train, y_train, eval_set=(X_valid,y_valid))
#model.fit(dtrain, eval_set = dvalid)
   
pred = model.predict(X_test)
#pred = model.predict(dtest)

submission = pd.read_csv("/home/terrence/CODING/Python/MODELS/AvitoData/sample_submission.csv")
submission['deal_probability'] = pred
print(submission[submission['deal_probability'] > 0])
#submission.to_csv("xgb_mwisho_one.csv", index=False)
'''

'''
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

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
preds = bst.predict(dtest)
#print(preds)

import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])

from sklearn.metrics import precision_score

xgb_score = precision_score(y_test, best_preds, average='macro')
print("XGB score = {:.4f} " .format(xgb_score))

'''




#---------------- LightGBM ----------------------------






'''
#---------------- CatBoost ----------------------------

# Train Model
print("Train CatBoost Decision Tree")

cb_model = CatBoostRegressor(iterations=200,
                             learning_rate=0.02,
                             depth=7,
                             eval_metric='RMSE',
                             random_seed = 23,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=10)
cb_model.fit(X_train, y_train,
             eval_set=(X_valid,y_valid),
             #cat_features=categorical_features_pos,
             use_best_model=True,
             verbose=True)

# # Feature Importance
# fea_imp = pd.DataFrame({'imp': cb_model.feature_importances_, 'col': X.columns})
# fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
# _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
# plt.savefig('catboost_feature_importance.png')   


print("Model Evaluation Stage")
print(cb_model.get_params())

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, cb_model.predict(X_valid))))

catpred = cb_model.predict(X_test)
catsub = pd.DataFrame(catpred,columns=["deal_probability"],index=testdex)
catsub['deal_probability'].clip(0.0, 1.0, inplace=True)
catsub.to_csv("catBoost_final_one.csv",index=True,header=True) # Between 0 and 1

'''









