import numpy as np
from scipy.signal import triang
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, SimpleRNN
from keras.models import Sequential
from keras.optimizers import RMSprop

def get_model(batch_shape, num_hidden_units, num_output):
    model = Sequential()
    model.add(SimpleRNN(num_hidden_units, batch_input_shape=batch_shape, stateful=True))
    model.add(Dense(num_output, activation='tanh'))
    return model

def compile_model(model):
    rms = RMSprop()
    model.compile(optimizer=rms, loss='mse')

def train_model(m, X, y, batch_size):
    epochs = 1
    compile_model(m)

    m.fit(X, y, nb_epoch=epochs, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    N = 64
    batch_size = 4
    time_steps = 1
    num_feature = 2 # 2 coordinates
    num_output = num_feature

    x = np.zeros((65, num_feature))
    x[:, 0] = np.concatenate([[0], triang(N-1), [0]]) #Triangle wave
    x[:, 1] = np.sin(np.linspace(0, 4 * np.pi, 65)) # Sine wave

    x_train = x[:-1, :]
    y_train = x[1:, :]
    x_train = x_train.reshape(x_train.shape[0], time_steps, num_feature)
    y_train = y_train.reshape(x_train.shape[0], num_feature)
    batch_shape = (batch_size, time_steps, num_feature)

    m = get_model(batch_shape, 15, num_output)
    m = train_model(m, x_train, y_train, batch_size)





