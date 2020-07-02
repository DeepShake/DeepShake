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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os

data_path = '../data/raw/'

files = []
with os.scandir(data_path) as it:
    for entry in it:
        print(entry.name)
        files.append(entry)
#         if not entry.name.startswith('.') and entry.is_file():
#             print(entry.name)

# print(files)
stations, events = [], []
for file in files:
#     print(np.load(os.path.join(data_path, file)).shape)
    temp = np.load(file)
    print(temp.shape)
    stations.append(temp.shape[0])
    events.append(temp.shape[1])
    del temp


dates = [file.name for file in files]
stat_evs = zip(stations, events)
shapes = [(s, v, 3, 6000) for s, v in stat_evs]
mapping = dict(zip(dates, shapes))
print(mapping)

import json
with open('./misc/date_to_shape.json', 'w') as fp:
    json.dump(mapping, fp)

# +
plt.bar(dates, events)
plt.ylabel("Events")
plt.show()

plt.bar(dates, stations)
plt.ylabel("Station counts")
plt.show()
# -

print(sum(events))

print(mapping['2019-06-07.npy'])

rev_map = dict(zip(events, dates))

print(rev_map)


