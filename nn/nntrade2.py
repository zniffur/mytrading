import os
import pandas as pd
import numpy
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt

CSV_DIR = "csv"
TEST_TICKER_1 = "AAPL"
TEST_TICKER_2 = "F"

filename = os.path.join(CSV_DIR, TEST_TICKER_1 + ".csv")
df = pd.read_csv(filename)

# preparo i dati
df2 = df['high']
len = df2.shape[0] - df2.shape[0] % 10
df2 = df2[:len]
values = df2.values.reshape((-1, 10))  # ca 4 anni di dati
df3 = pd.DataFrame(values)  # creo un nuovo df a 10 colonne
df3['y'] = 0  # aggiungo la colonna y

for i in range(0, df3.shape[0]-2):
    # creo Y per tutte le righe tranne le ultime 2
    if df3.loc[i+1].max() > df3.loc[i].max():
        df3.loc[i, 'y'] = 1
    else:
        df3.loc[i, 'y'] = 0

dataset = df3.values

X_raw = dataset[0:20, 0:10].astype(float)  # 1 anno
X = X_raw
Y = dataset[0:20, 10]

# costruisco modello
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error',
                  # loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# evaluate model with  dataset
encoded_Y = Y

#estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=1, verbose=2)
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# evaluate baseline model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,
                                          epochs=250,
                                          batch_size=5,
                                          verbose=2)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# MANUAL TRAINING TESTING

X_raw = dataset[0:21, 0:10].astype(float)  # ca 1 anno di dati + la riga di test
X = preprocessing.minmax_scale(X_raw, feature_range=(0, 1))
Y = dataset[0:21, 10].astype(float)

numpy.random.seed(seed)
model = create_baseline()
model.fit(X[0:20], Y[0:20], epochs=250, shuffle=False, verbose=2)

X_test = X[20, :]
Y_test = Y[20]

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

model.predict(X_test)
