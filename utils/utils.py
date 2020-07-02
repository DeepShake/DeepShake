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
                      single_step=False,
                      return_indices = False):
    data = []
    labels = []
    quake_idxs = []
    
    start_index = history_size
    end_index = dataset.shape[1] - target_size
        
    for q_idx in range(len(dataset)):
        quake = dataset[q_idx]
        quake_class = quake_classes[q_idx] #class for quake
        quake_wt = int(num_copies[int(quake_class)])
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.extend([quake[indices]]*quake_wt)
            quake_idxs.extend([q_idx]*quake_wt)
            
            if single_step:
                labels.extend([quake[i+target_size]]*quake_wt)
            else:
                labels.extend([quake[i:i+target_size]]*quake_wt)
    
    if return_indices is True:    
        return np.array(data), np.array(labels), np.array(quake_idxs)
    else:
        return np.array(data), np.array(labels)


"""
STEP = 1
target_size = 10
past_history = 15
single_step=False
"""
def multivariate_data(dataset, history_size,
                      target_size, step, 
                      single_step=False, 
                      return_indices = False):
    data = []
    labels = []
    quake_idxs = []
    
    start_index = history_size
    end_index = dataset.shape[1] - target_size
        
    for q_idx in range(len(dataset)):
        quake = dataset[q_idx]
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(quake[indices])
            quake_idxs.append(q_idx)
        
            if single_step:
                labels.append(quake[i+target_size])
            else:
                labels.append(quake[i:i+target_size])
    if return_indices is True:      
        return np.array(data), np.array(labels), np.array(quake_idxs)
    else:
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

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps