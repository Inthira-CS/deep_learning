from models.mc_dropout_lstm import build_mc_dropout_lstm


def train(X_train, y_train):
    model = build_mc_dropout_lstm((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    return model
