# mar 2024
# analysis script for QZFM crosstalk

import os, glob, time, sys
import matplotlib.pyplot as plt
from labjack import ljm 
from datetime import datetime
import ctypes
import numpy as np
from time import time
import time as time_
import pandas as pd
import pathlib
from scipy import signal
from scipy import fft
from scipy import optimize
from scipy.signal import butter, sosfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y
    
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#%%
sr = 1000

x_77 = []
y_77 = []
z_77 = []
for file in glob.glob('..//data//mar22//77HzConstancy//*.csv'):
    df = pd.read_csv(file)
    Bx = df['x']
    By = df['y']
    Bz = df['z']
    
    
    # calculate periodogram
    fx, px = signal.periodogram(Bx, sr)
    fy, py = signal.periodogram(By, sr)
    fz, pz = signal.periodogram(Bz, sr)
    
    
    # find index of 76.5 and 77.5 Hz
    idx_start = find_nearest_idx(fx, 76.5)
    idx_end = find_nearest_idx(fx, 77.5)
    
    # sum over this 1 Hz band, and append to respective lists
    x_77.append(np.sum(px[idx_start:idx_end+1]))
    y_77.append(np.sum(py[idx_start:idx_end+1]))
    z_77.append(np.sum(pz[idx_start:idx_end+1]))

#%%
plt.figure()
plt.hist(x_77)
plt.hist(y_77)
plt.hist(z_77)
plt.show()

#%%

lowcut = 59
highcut = 61

amplitude_dict = {'x':[],
                  'y':[],
                  'z':[]}


for file in glob.glob('..//data//mar22//77HzConstancy//*.csv'):
    df = pd.read_csv(file)

    for direction in ['x', 'y', 'z']:
        raw_B = np.asarray(df[direction])
        raw_B_bp = butter_bandpass_filter(raw_B, lowcut, highcut, sr, 5)[2500:]
        
        # find index of maximums and minimums
        max_idx = signal.find_peaks(raw_B_bp, distance=int(0.9*sr/np.mean([lowcut, highcut])))
        min_idx = signal.find_peaks(-raw_B_bp, distance=int(0.9*sr/np.mean([lowcut, highcut])))
        
        # get interweaved peak values
        maximas = raw_B_bp[max_idx[0]]
        minimas = raw_B_bp[min_idx[0]]
        extremas = np.asarray([val for pair in zip(maximas, minimas) for val in pair])
        
        for i in range(len(extremas)-1):
            amplitude_dict[direction].append(abs(extremas[i]-extremas[i+1]))
    
#%%
for direction in ['x', 'y', 'z']:
    data = amplitude_dict[direction]
    binwidth = 0.05
    plt.figure()
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    plt.hist(data, bins, density=True)
    # add a 'best fit' line
    sigma = np.std(data)
    mu = np.mean(data)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    plt.plot(bins, y, 'r--')
    # plt.hist(amplitude_dict['y'])
    # plt.hist(amplitude_dict['z'])
    plt.show()

#%%
df = pd.read_csv('..//data//mar22//77HzConstancy//240322T115755.csv')
plt.figure()
plt.plot(df['x'])
plt.plot(butter_bandpass_filter(df['x'], lowcut, highcut, sr, 5))
plt.show()