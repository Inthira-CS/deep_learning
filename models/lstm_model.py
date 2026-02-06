from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_lstm(input_shape):
    model = Sequential()
    # input_shape should be a tuple: (timesteps, features)
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model
 