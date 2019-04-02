import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
# import numpy as np
# import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os

CSV_DIR = "csv"
TEST_TICKER_1 = "AAPL"
TEST_TICKER_2 = "F"

filename = os.path.join(CSV_DIR, TEST_TICKER_1 + ".csv")
df = pd.read_csv(filename)

df2 = df['high']
len = df2.shape[0] - df2.shape[0] % 10
df2 = df2[:len]
values = df2.values.reshape((-1, 10))  # ca 4 anni di dati
df3 = pd.DataFrame(values)  # creo un nuovo df a 10 colonne
df3['y'] = 0  # aggiungo la colonna y

for i in range(0, df3.shape[0]-2):
    if df3.loc[i+1].max() > df3.loc[i].max():
        df3.loc[i, 'y'] = 1
    else:
        df3.loc[i, 'y'] = 0


scaled_data = preprocessing.minmax_scale(df3, feature_range=(0, 1))
training_df = scaled_data[:90]  # ca 4 anni dati
test_df = scaled_data[90:]  # il resto, ca 1 anno

##### MODEL 1

X = training_df[:, 0:10]
Y = training_df[:, 10]

# Define the model
model = Sequential()
model.add(Dense(20, input_dim=10, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')


# Train the model
model.fit(
    X,
    Y,
    epochs=1000,
    shuffle=True,
    verbose=2
)

X_test = test_df[:, 0:10]
Y_test = test_df[:, 10]

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))


model.predict(X_test)
