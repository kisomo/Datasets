# -*- coding: utf-8 -*-
#https://www.kaggle.com/nicapotato/simple-catboost

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

'''

print("\nData Load Stage")
training = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
traindex = training.index
#print(traindex)
testing = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
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
print(df.dtypes)


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


# Train Model
print("Train CatBoost Decision Tree")
modelstart= time.time()
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
catsub.to_csv("catsub_two.csv",index=True,header=True) # Between 0 and 1
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

'''




'''
print("++++++++++++++++++++++++++++++++++++++++++++++++++++ FastText +++++++++++++++++++++++++++++++++++++++++++++++++++++")

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
print(train.head(2))
labels = train[['deal_probability']].copy()
train = train[['description']].copy()

#emb = pd.read_csv(EMBEDDING_FILE, index_col = 0)
#print(emb.shape)
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



'''

print("+++++++++++++++++++++++++++++++++++++++++++++ xgboost+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
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

#start_time = time.time()
#print_duration(start_time, "Just Testing") 

msg = "Just Testing"
# quick way of calculating a numeric has for a string
def n_hash(s):
    random.seed(hash(s))
    return random.random()

#print(n_hash(msg))
#print(hash(msg))
#print(random.seed(hash(msg)))
#print(random.random())


# hash a complete column of a pandas dataframe    
def hash_column (row, col):
    if col in row:
        return n_hash(row[col])
    return n_hash('none')

train = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv')
test = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv')

#print(train.shape)
#print(train.head(2))
#print(train.dtypes)

#print(test.shape)
#print(test.head(2))
#print(test.dtypes)

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#CountVectorizer(charset='koi8r', stop_words=stopWords)


#count_vectorizer = CountVectorizer(stop_words='english')
#count_train = count_vectorizer.fit_transform(X_train)
#count_test = count_vectorizer.fit_transform(X_test)
#tfidf_vectorizer = TfidfVectorizer(stop_words = 'english',max_df=0.7)
#tfidf_vectorizer = TfidfTransformer(stop_words = 'english',max_df=0.7)
#tfidf_train = tfidf_vectorizer.fit_transform(X_train)
#tfidf_test = tfidf_vectorizer.fit_transform(X_test)
#CountVectorizer(charset='koi8r', stop_words=stopWords)


count_vectorizer = CountVectorizer()

start_time = time.time()
# create a xgboost model
model = xgb.XGBRegressor(n_estimators=2, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=3)

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
train['description'] = count_vectorizer.fit_transform(train['description'])
train['price'] = np.log(train['price'] + 0.01)
start_time = print_duration (start_time, "Finished reading")      

print(train.shape)
#print(train.head(2))
#print(train.dtypes)

# start training
train_X = train.as_matrix(columns=['user_id', 'price', 'region', 'city', 'parent_category_name', 'category_name', 'user_type', 'description'])
model.fit(train_X, train['deal_probability'])
    
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






print("++++++++++++++++++++++++++++++++++++++++++++++++++ lightGBM +++++++++++++++++++++++++++++++++++++++++++++++++++")

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
training = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/train.csv', index_col = "item_id", parse_dates = ["activation_date"]).sample(2000)
traindex = training.index
testing = pd.read_csv('/home/terrence/CODING/Python/MODELS/AvitoData/test.csv', index_col = "item_id", parse_dates = ["activation_date"]).sample(2000)
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
print(df.head(2))
print(df.dtypes)

print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]
print("Encoding :",categorical)


# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nText Features")
#print(df.shape)
#print(df.head(2))
# Feature Engineering 

# Meta Text Features
textfeats = ["description", "title"]
df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

df['title'] = df['title'].apply(lambda x: cleanName(x))
df["description"]   = df["description"].apply(lambda x: cleanName(x))

print(df.shape)
#print(df.head(2))


for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

print(df.shape)    
#print(df.head(2))

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
    #"min_df":5,
    #"max_df":.9,
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
            #max_features=7000,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#sys.setdefaultencoding('ascii')

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
vectorizer.fit(df.to_dict('records'))

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

'''
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










