import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# tf.random.set_random_seed(13)
# np.random.seed(13)

def shuffle_together(x, y, seed=10):
    ## Shuffle together two arrays of the exact same dimensions
    idxs = np.arange(x.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idxs)
    x = x[idxs]
    y = y[idxs]
    return x, y


def multivariate_data_balanced(dataset, history_size,
                      target_size, step, 
                      quake_classes, num_copies, 
                      single_step=False):
    data = []
    labels = []
    
    start_index = history_size
    end_index = dataset.shape[1] - target_size
        
    for q_idx in range(len(dataset)):
        quake = dataset[q_idx]
        quake_class = quake_classes[q_idx] #class for quake
        quake_wt = int(num_copies[int(quake_class)])
        for _ in range(quake_wt):
            for i in range(start_index, end_index):
                indices = range(i-history_size, i, step)
                data.append(quake[indices])

                if single_step:
                    labels.append(quake[i+target_size])
                else:
                    labels.append(quake[i:i+target_size])

    return np.array(data), np.array(labels)

def multivariate_data(dataset, history_size,
                      target_size, step, 
                      single_step=False):
    data = []
    labels = []
    
    start_index = history_size
    end_index = dataset.shape[1] - target_size
        
    for q_idx in range(len(dataset)):
        quake = dataset[q_idx]
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(quake[indices])

            if single_step:
                labels.append(quake[i+target_size])
            else:
                labels.append(quake[i:i+target_size])

    return np.array(data), np.array(labels)

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
    
def recover_quake(y_pred, y, means, norms, num_quakes):
    y_pred = y_pred.reshape((num_quakes, -1, 15))
    y = y.reshape((num_quakes, -1, 15))
                  
    y = y*np.expand_dims(norms, 1) + np.expand_dims(means, 1)
    y_pred = y_pred*np.expand_dims(norms, 1) + np.expand_dims(means, 1)

    return y, y_pred, np.abs((y - y_pred).reshape((-1,)))

