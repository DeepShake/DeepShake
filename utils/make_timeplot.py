import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from math import radians, cos, sin, asin, sqrt

STATION_COORDS = {
    "CI.WVP2.": (35.94939, -117.81769),
    "CI.WRV2.": (36.00774, -117.8904),
    "CI.WRC2.": (35.9479, -117.65038),
    "CI.WNM.": (35.8422, -117.90616),
    "CI.WMF.": (36.11758, -117.85486),
    "CI.WCS2.": (36.02521, -117.76526),
    "CI.WBM.": (35.60839, -117.89049),
    "CI.TOW2.": (35.80856, -117.76488),
    "CI.SRT.": (35.69235, -117.75051),
    "CI.SLA.": (35.89095, -117.28332),
    "CI.MPM.": (36.05799 ,-117.48901),
    "CI.LRL.": (35.47954, -117.68212),
    "CI.DTP.": (35.26742, -117.84581),
    "CI.CCC.": (35.52495, -117.36453),
    "CI.JRC2.": (35.98249, -117.80885)
}


def CalcDistance(epicenter, station):
    """
    Calculates the distance between two gps points
    Source: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    
    parameters
    ----------
    epicenter: tuple
    tuple of (lat, long) value for the epicenter of your earthquake
    
    station: tuple
    tuple of (lat, long) value for the location of your station
    
    Returns
    -------
    Distance: float 
    Distance between epicenter and station
    
    """
    
    lat1, lon1 = epicenter
    lat2, lon2 = station
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956
    return c * r


def create_time_steps(length):
    """
    Helper function for generating the time data for historical timeseries
    
    parameters
    ----------
    length: int
    Number of time steps in your historical time series
    
    Returns
    -------
    time_steps: List
    List containing the second-by-second time values of your history timeseries
    
    """
    
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps


def multi_step_plot(history, true_future, save_path, epicenter = None, prediction = None, scale = 1, title = None):
    """
    Plots the timeseries data for a given earthquake across the different stations
    based on the distance of each station from the epicenter of the quake,
    along with the predicted future time series for the quake
    
    parameters
    ----------
    history: numpy array, shape = (15, t_hist)
    The historical time series fed into the model
    
    true_future: numpy array, shape = (15, t_future)
    The true time series data of the given quake
    
    save_path: str
    Path under which to save the desired plot
    
    epicenter: tuple(float), default = None
    Latitude/Longitude point of the epicenter of the earthquake
    
    prediction: numpy array, shape = (15, t_future), default = None
    Time series of the predicted future progress of the quake
    
    scale: int
    Quantity by which to scale the plotted graphs
    
    title: str
    Title for the plot
    
    Return
    ------
    None
    
    """
    
    distances = []
    plt.figure(figsize=(15, 10))
    num_in = np.arange(history.shape[1])
    future_out = np.arange(true_future.shape[1]) + history.shape[1]
    
    if epicenter is None:
        # Randomly assign epicenter to be CI.WRV2 if none given
        epicenter = (36.00774, -117.8904)
    
    mean = np.mean(history[:, :15])
    std = np.std(history[:, :15])
    
    for i in tqdm(range(len(STATION_COORDS))):
        station = list(STATION_COORDS.keys())[i]
        dist = CalcDistance(epicenter, STATION_COORDS[station])
        plt.plot(
            num_in,
            scale * (history[i] - mean)/std/2 + dist, 
            'gray', 
            linewidth = 1
        )
        
        plt.plot(
            future_out,
            scale * (true_future[i] - mean)/std/2 + dist, 
            'b--',
            linewidth = 1
        )
        
        if prediction is not None:
            pred_out = np.arange(prediction.shape[1]) + history.shape[1]
            num_preds = prediction.shape[1]
            plt.plot(
                pred_out,
                scale * (prediction[i] - mean)/std/2 + dist, 
                'r--',
                linewidth = 1
            )
            
        if i % 2 == 0:
            plt.text(num_in[0], dist, station)
        else:
            plt.text(num_in[5], dist, station)
    
    plt.ylabel("Distance from Epicenter (km)")
    plt.xlabel("Time (s)")
    
    if title is not None:
        plt.title(title)
    
    plt.savefig(save_path)
    plt.close()