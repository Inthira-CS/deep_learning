import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

def train_lstm(X_train, y_train):

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    return model   # ✅ return actual trained model


if __name__ == "__main__":
    print("train_lstm.py loaded")


   