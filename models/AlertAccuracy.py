# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Find Classwise Accuracy

# +
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#tf.enable_eager_execution()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sklearn

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

tf.random.set_seed(13)

import py3nvml
py3nvml.grab_gpus(1)
# -

from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())

multi_step = True
num_predicted_steps = 10
model_path = "trained_models_acc/multistep/LSTM_histnorm.h5"

# # Load the dataset

# +
data_path = "../data/hist15_norm_acc"

x_val = np.load(os.path.join(data_path, "orig_order_X_val.npy"))
y_val = np.load(os.path.join(data_path, "orig_order_y_val.npy"))

richters = np.load(os.path.join(data_path, "orig_order_richters.npy"))
# -

#Normalize the val set
eps = 1e-3
val_mean = np.mean(x_val, axis = 1, keepdims=True)
val_var = np.linalg.norm(x_val - val_mean, axis=1, keepdims=True)
val_var[val_var < eps] = 1
val_mean.shape, val_var.shape

x_norm = (x_val - val_mean)/val_var
y_norm = (y_val - val_mean)/val_var

# # Load and run the model

# +
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128,
                            return_sequences=True,
                            input_shape=x_val.shape[-2:]))
model.add(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'))
model.add(tf.keras.layers.LSTM(64, activation='relu'))
model.add(tf.keras.layers.Dense(150))

model.load_weights(model_path)

model.compile(optimizer='nadam', loss='mae', metrics=['mae', 'mse'])
model.summary()
# -

model.evaluate(x_norm, y_norm.reshape(-1, 150), batch_size = 4096)

y_pred_norm = model.predict(x_norm, batch_size = 4096).reshape(-1, 10, 15)

nums = [42, 114, 5636, 10234, 16352, 42295]
for num in nums:
    plt.figure()
    plt.plot(y_pred_norm[num, :, 0])
    plt.plot(y_norm[num, :, 0])

# # Recover the original quakes

#Unnormalize the quake
y_pred = y_pred_norm*val_var + val_mean

y_pred.mean(), y_pred_norm.mean(), val_mean.mean()

np.sum(y_pred < 0)/np.prod(y_pred.shape)

# +
#Take the last prediction in a multistep model
y_pred = y_pred[:, -1, :]
#y_pred = y_pred[:, 0, :] # Or the first one, for 1 second preds
y_true = y_val[:, -1, :]
    
y_pred.shape, y_true.shape
# -

#Recover the original quakes (before windowing)
num_quakes = len(richters)
y_pred = y_pred.reshape((num_quakes, -1, 15))
y_true = y_true.reshape((num_quakes, -1, 15))
y_pred.shape, y_true.shape


def mae(delta):
    return np.mean(np.abs(delta))
mae(y_true - y_pred)

# +
from sklearn import metrics

y_true_flat = y_true.reshape(-1)
y_pred_flat = y_pred.reshape(-1)
print("Explained Variance Score:", metrics.explained_variance_score(y_true_flat, y_pred_flat)) 
print("Max Error Score:", metrics.max_error(y_true_flat, y_pred_flat)) 
print("Mean Absolute Error Score:", metrics.mean_absolute_error(y_true_flat, y_pred_flat)) 
print("Mean Squared Error Score:", metrics.mean_squared_error(y_true_flat, y_pred_flat)) 
print("Median Absolute Error Score:", metrics.median_absolute_error(y_true_flat, y_pred_flat)) 
print("R2 Score:", metrics.r2_score(y_true_flat, y_pred_flat)) 
# -

# # Alert accuracy

#MMI accel cutoffs (m/s^2) 
#              I,  II/III  IV        V      VI      VII       VIII      IX    X+
MMI_cutoffs = [    0.01667, 0.13729, 0.38246, 0.90221, 1.76520, 3.33426, 6.37432, 12.16025]

MMI_val = np.searchsorted(MMI_cutoffs, y_true.reshape(-1), side = 'right')
MMI_pred = np.searchsorted(MMI_cutoffs, y_pred.reshape(-1), side = 'right')
print(y_true.shape, y_pred.shape, MMI_val.shape, MMI_pred.shape)

np.unique(MMI_val, return_counts = True), np.unique(MMI_pred, return_counts = True)


# +
def print_confusion_matrix(confusion_matrix, class_names = None, figsize = (10,7), fontsize=14, labels=None, title=None):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    sns.set(font_scale=1)
    if class_names == None:
        class_names = list(np.arange(len(confusion_matrix)))
        
    print(class_names)
    
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f')
    
    if labels:
        heatmap.yaxis.set_ticklabels(labels, rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(labels, rotation=45, ha='right', fontsize=fontsize)
    else:
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True MMI')
    plt.xlabel('Predicted MMI')
    if title != None:
        plt.title(title)
    return fig

cmat = metrics.confusion_matrix(MMI_val, MMI_pred)
cmat = cmat / np.expand_dims(np.sum(cmat, axis = 1), 1)
print_confusion_matrix(cmat, labels=["I", "II/III", "IV", "V", "VI", "VII", "VIII", "IX", "X+"], title="MMI Confusion Matrix (10 Second Prediction)");
# -

alert_thresh = 3  #MMI V
fpr = np.mean(og_MMI_pred[og_MMI < alert_thresh] >= alert_thresh)
fnr = np.mean(og_MMI_pred[og_MMI >= alert_thresh] < alert_thresh)
fpr, fnr

# +
# Make a F1 Curve
decision_threshes = np.linspace(0, 0.01, 1001)
alert_thresh = 0.38246
sens = []
spec = []
for decision_thresh in decision_threshes:
    sens.append(np.mean(og_y_pred.reshape(-1)[og_y.reshape(-1) >= alert_thresh] >= decision_thresh))
    spec.append(np.mean(og_y_pred.reshape(-1)[og_y.reshape(-1) < alert_thresh] < decision_thresh))
    
sens = np.array(sens)
spec = np.array(spec)
sens, spec
# -

plt.plot(sens)

# +
plt.title("Earthquake Text Alerts")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#plt.xlim([0, 1])
#plt.ylim([0, 1])
plt.plot(1-spec, sens)

print(sklearn.metrics.auc(1-spec, sens))
# -

#Find EER
temp = spec > sens
print("EER: ", spec[np.argmax(temp)], sens[np.argmax(temp)])
plt.plot(spec)
plt.plot(sens)

# # Molchan time!
# We will draw Molchan curves, one line for every time offset, with the line parameterized by additional margin

# +
#python version of above equation
from scipy.special import comb
from scipy.optimize import fsolve

def func(a, N, h, tau):
    out = 0
    for i in range(h, N+1):
        out += comb(N, i) * (tau**i) * (1-tau)**(N-i)
        
    return out - a

#Solves for one tau
def get_tau(a, N, h):
    sfunc = lambda tau : func(a, N, h, tau)
    
    x = np.arange(0, 1, 0.01)
    y = np.array([sfunc(datum) for datum in x])
    init_guess = np.argmax(y > 0)/100
    
    return fsolve(sfunc, init_guess)

#Gives all the taus
def get_taus(a, N, h):
    taus = np.zeros(h.shape)
    
    for i in range(len(h)):
        taus[i] = get_tau(a, N, i)
        
    return taus
        
#Plots a molchan
def plot_molchan(N):
    h = np.arange(N+1)
    
    plt.figure()
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, N, 10), 'r--')
    for a in [0.001, 0.01, 0.05, 0.25, 0.50]:
        plt.plot(get_taus(a, N, h), h, label=r'$\alpha$ = ' + str(a))
    plt.xlim([0, 1])
    plt.ylim([N, 0])
    plt.legend()
    plt.title("Molchan Diagram")
    plt.show()
    
plot_molchan(15)
# -

time_offsets = np.arange(10)
margins = np.linspace(-100, 40, 101)
time_offsets, margins

# +
past_history = 15
STEP = 1

x, y = multivariate_data(val_dataset,past_history,
                                                   num_predicted_steps, STEP,
                                                   single_step=not multi_step)

y_pred = single_step_model.predict(x, batch_size = 4096).reshape(y.shape)
y.shape, y_pred.shape
# -

molchan_tau = []
molchan_tpr = []
for offset in time_offsets:
    curr_tau = []
    curr_tpr = []

    curr_y, curr_y_pred, delta = recover_quake(y_pred[:, offset, :], 
                                             y[:, offset, :], 
                                             val_data_mean, 
                                             val_data_var, 
                                             len(val_dataset))
    
    curr_MMI = np.searchsorted(MMI_cutoffs, curr_y.reshape(-1), side = 'right')

    for margin in margins:
        curr_MMI_pred = np.searchsorted(MMI_cutoffs, curr_y_pred.reshape(-1) + margin, side = 'right')
        sens = np.mean(curr_MMI_pred[curr_MMI >= alert_thresh] >= alert_thresh)
        tau = np.mean(curr_MMI_pred >= alert_thresh)
        curr_tpr.append(sens)
        curr_tau.append(tau)
        
    molchan_tau.append(curr_tau)
    molchan_tpr.append(curr_tpr)
molchan_tau = np.array(molchan_tau)
molchan_tpr = np.array(molchan_tpr)
molchan_tau = np.concatenate((np.zeros((10, 1)), molchan_tau), axis = 1)
molchan_tpr = np.concatenate((np.zeros((10, 1)), molchan_tpr), axis = 1)
molchan_tau, molchan_tpr

for i in time_offsets:
    plt.plot(molchan_tau[i], molchan_tpr[i], color = 'b', alpha = 0.5 - 0.05*i, label = str(i+1) + " Second Prediction")
plt.legend()
plt.ylim([1, 0])
plt.xlim([0, 1])


# +
#Plots a result molchan
def plot_molchan_result(tau, tpr, time_offsets, N = 15):
    h = np.arange(N+1)
    
    plt.figure()
    
    #Plot significance thresholds
    plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 'r--')
    for a in [0.001, 0.05, 0.25]:
        plt.plot(get_taus(a, N, h), h/N, color = 'k', alpha = 0.8 - a, label=r'$\alpha$ = ' + str(a))
        
    for i in range(len(time_offsets)):
        plt.plot(tau[i], tpr[i], color = 'C' + str(i), label = str(time_offsets[i]+1) + " Second Prediction")
        
    plt.xlim([0, 1])
    plt.ylim([1, 0])
    plt.ylabel("Sensitivity")
    plt.xlabel("Proportion of Time Alarmed")
    plt.legend()
    plt.title("Molchan Diagram")
    plt.show()
    
idxs = [0, 1, 4, 9]
plot_molchan_result(molchan_tau[idxs], molchan_tpr[idxs], time_offsets[idxs])
# -

# # Alert Latency

# Find the time from origin it takes for the model to predict
large_quake_idxs = np.nonzero(np.max(np.max(og_y, axis = -1), axis=-1) > MMI_cutoffs[1])
large_quakes = og_y[large_quake_idxs]
large_quakes_pred = og_y_pred[large_quake_idxs]
large_quake_idxs

for i, quake, quake_pred in zip(range(len(large_quakes)), large_quakes, large_quakes_pred):
    print("Quake", i)
    plt.figure()
    plt.imshow(quake.T > MMI_cutoffs[1], interpolation='none', cmap='gray')
    plt.show()
    plt.figure()
    plt.imshow(quake_pred.T > MMI_cutoffs[1], interpolation='none', cmap='gray')
    plt.show()

# # Peak Deviation

deviation = np.abs(np.max(og_y, axis = (-1, -2)) - np.max(og_y_pred, axis=(-1, -2)))
plt.hist(deviation)
