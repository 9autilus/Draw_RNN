seed = 1337  # for reproducibility
import numpy as np
np.random.seed(seed)        # Seed Numpy
import random               # Seed random
random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)    # Seed Tensor Flow

from scipy.signal import triang
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, SimpleRNN
from keras.models import Sequential, load_model, save_model
from keras.optimizers import RMSprop, SGD
import keras.backend as K

def plot_signals(signals):
    plt.plot(signals[:, 0])
    plt.plot(signals[:, 1])
    plt.grid()
    plt.show()

def get_model(batch_shape, num_hidden_units, num_output):
    model = Sequential()
    model.add(SimpleRNN(num_hidden_units, batch_input_shape=batch_shape, stateful=True))
    model.add(Dense(num_output, activation='tanh'))
    return model

def compile_model(model):
    # The loss fluctuates after certain epochs with rmsprop
    # possibly because our setup doesn't have epochs-setup
    # opt = RMSprop(0.0001)
    opt = SGD(0.01) # The loss monotonically decreases with SGD
    model.compile(optimizer=opt, loss='mse')

def dump_train_history(history, val_set_present, log_file_name):
    f = open(log_file_name, "w")

    accuracy_present = 'acc' in history.keys()

    if accuracy_present:
        train_acc = history['acc']

    train_loss = history['loss']

    if val_set_present:
        if accuracy_present:
            val_acc = history['val_acc']
        val_loss = history['val_loss']
    else:
        val_loss = [-1] * len(train_loss)
        val_acc = [-1] * len(train_loss)

    f.write('Epoch  Train_loss  Train_acc  Val_loss  Val_acc  \n')

    for i in range(len(train_loss)):
        if accuracy_present:
            f.write('{0:d} {1:f} {2:.2f}% {3:f} {4:.2f}%\n'.format(
                i, train_loss[i], 100 * train_acc[i], val_loss[i], 100 * val_acc[i]))
        else:
            f.write('{0:d} {1:f} {2:f}\n'.format(i, train_loss[i], val_loss[i]))

    print('Dumped history to file: {0:s}'.format(log_file_name))

def run_part_a(m, x, y, batch_size):
    train_epochs = 100
    test_trajectories = 10
    train = 1# Whether to train or load a trained model
    model_file = 'model.h5'
    use_states = True
    state_file = 'state.npy'

    # Train
    if train:
        compile_model(m)
        if 1:
            loss = np.zeros(train_epochs)
            for epoch in range(train_epochs):
                for i in range(len(x)):
                    loss[epoch] = m.train_on_batch(x[i].reshape(1, 1, 2), y[i].reshape(1, 2))
                print("Epoch : {0:d} loss: {1:f}".format(epoch, loss[epoch]))
        else:
            hist = m.fit(x, y, nb_epoch=train_epochs, batch_size=batch_size, shuffle=False, verbose=2)
            dump_train_history(hist.history, False, 'history_a.log')

        save_model(m, model_file)

        if use_states:
            # save_state
            states = np.array([K.get_value(s) for s,_ in m.state_updates])
            np.save(state_file, states)

    # Test
    if not train:
        m = load_model(model_file)

    y_pred = np.zeros([test_trajectories, len(y), 2])
    mse = np.zeros(test_trajectories)

    y_prev = np.array([0., 0.])

    if use_states:
    # Load states
        states = np.load(state_file)
        for (d,_), s in zip(m.state_updates, states):
            K.set_value(d, s)

    # Predict() does not use the states generated during training
    for t in range(test_trajectories):
        for i in range(len(x)):
            if 0:
                y_pred[t, i] = m.predict_on_batch(x[i].reshape(batch_size, 1, -1))
            else:
                y_pred[t, i] = m.predict_on_batch(y_prev.reshape(batch_size, 1, -1))
                y_prev = y_pred[t, i]

        mse[t] = np.sum(np.linalg.norm((y_pred[t] - y), axis=1) ** 2)/len(x)

    print('Part (a) MSE: ', mse)

    if 1:
        plt.figure()
        original = y
        predicted = y_pred[0, :, :]
        plt.plot(original[:, 0], original[:, 1]) # Original input
        plt.plot(predicted[:, 0], predicted[:, 1])
        plt.savefig('plot_0.jpg')
        print(np.concatenate([original, predicted], axis=1))
        plt.show()

    return y_pred

if __name__ == '__main__':
    N = 64
    batch_size = 1
    time_steps = 1
    num_feature = 2 # 2 coordinates
    num_output = num_feature
    trajectories = 1 # epochs

    signals = np.zeros((65, num_feature))
    signals[:, 0] = np.concatenate([[0], triang(31), [0], -triang(31), [0]]) #Triangle wave
    signals[:, 1] = np.sin(np.linspace(0, 4 * np.pi, 65)) # Sine wave
    signals = signals[:-1, :] # Remove last sample to create a time-period

    if 0:
        plot_signals(signals)
        exit(0)

    x = signals
    y = np.concatenate([signals[1:], signals[0].reshape(1, -1)])
    x = x.reshape(x.shape[0], time_steps, num_feature)
    y = y.reshape(x.shape[0], num_feature)
    batch_shape = (batch_size, time_steps, num_feature)

    m = get_model(batch_shape, 15, num_output)

    if 1: # part(a)
        run_part_a(m, x, y, batch_size)






