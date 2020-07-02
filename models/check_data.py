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

import pickle
import os
import obspy
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/data/beroza/danjwu/DeepTremor/data/Ridgecrest"

station_name = ["WVP2","WRV2","WRC2","WNM","WMF","WCS2","WBM","TOW2","SRT","SLA","MPM","LRL","DTP","CCC","JRC2","CLC"]

## M6.4 17:33:49 (UTC)
g = 9.8
tstart = (17*3600 + 33*60 + 49)*100
tend = tstart + 6000
with open(os.path.join(data_dir, "2019-07-04.pkl"), "rb") as fp:
  meta = pickle.load(fp)

stations = obspy.read_inventory("/home/zhuwq/danjwu/DeepTremor/data/stations.xml")

station_data = {}
for st in meta["stations"]:
  if st.split(".")[1] in station_name:
    station_data[st.split(".")[1]] = meta["stations"][st]["data"]

station_response = {}
for network in stations: 
    for station in network: 
        if station.code in station_name:
          station_response[station.code] = np.zeros(3)
          for channel in station: 
              if channel.code == "HNE": 
                station_response[station.code][0] = channel.response.instrument_sensitivity.value
              if channel.code == "HNN": 
                station_response[station.code][1] = channel.response.instrument_sensitivity.value
              if channel.code == "HNZ": 
                station_response[station.code][2] = channel.response.instrument_sensitivity.value

station_pga = {}
for st in station_data:
  station_pga[st] = np.max(station_data[st][tstart:tend,:] / station_response[st], axis=0) /g * 100.

station_pga2 = {}
for st in station_data:
  station_pga2[st] = np.max(station_data[st][tstart:tend,:] / station_response[st], axis=0) 

station_pga

station_pga2

station_response

station_data


