import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)

# load dataset
dataframe = pd.read_csv('Postures.csv')

for val in list(dataframe.columns.values):
    dataframe[val] = pd.to_numeric(dataframe[val], errors='coerce')

#fill missing data
dataframe=dataframe.fillna(dataframe.mean())

#keep only first class
dataframe=dataframe.loc[dataframe['Class'] == 1]

dataframe=dataframe.drop(['Class'], axis=1)


#normalize data
listValuesToNormalize=list(dataframe.columns.values)
listValuesToNormalize.remove('User')
listValuesToNormalize
dataframe[listValuesToNormalize] = minmax_scale(dataframe[listValuesToNormalize])

# get dataset (split it in input/output)
dataset = dataframe.values

X = dataset[:,1:].astype(float)
Y = dataset[:,0].astype(int)

# convert integers to  one hot encoded
hot_encoded_y = np_utils.to_categorical(Y)

#split 0.8/0.2
X_train, X_test, y_train, y_test = train_test_split(X, hot_encoded_y, test_size=0.2, random_state=seed)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)

print("the dataset has "+str(X.shape[0])+ "samples that are splitted in:")
print("- "+str(X_train.shape[0])+"samples (training set)" )
print("- "+str(X_val.shape[0])+"samples (validation set)")
print("- "+str(X_test.shape[0])+"samples (test set)")

# create model
model = Sequential()
model.add(Dense(12, input_dim=36, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(15, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=140, batch_size=20)

#test model
loss, acc =  model.evaluate(X_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

#test model
loss, acc =  model.evaluate(X_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
model.save('my_model_class0_predictUser.h5')