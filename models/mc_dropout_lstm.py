from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def build_mc_dropout_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model
