# mar 2024
# measurement script for QZFM crosstalk

from QZFM import QZFM
import os, glob, time, sys
import matplotlib.pyplot as plt
import queue
from threading import Thread
from labjack import ljm 
from datetime import datetime
import ctypes
import numpy as np
from time import time
import time as time_
import pandas as pd
from labjackmeasure import labjack_measure
import pathlib
from scipy import signal
from scipy import fft


def make_folder(fp, folder_name):
    '''
    Parameters
    ----------
    fp : str
        file path to location where folder is to be created
    axis : str, x|y|z
        axis of rotation.
    sensor : str
        Name of sensor being measured e.g., 'AAY4'.

    Returns
    -------
    folder_name : str
        name of folder
    
    creates folder and then returns folder name
    '''
    # check if folder already exists, if not create it
    pathlib.Path(fp+folder_name).mkdir(parents=True, exist_ok=True) 
    return folder_name


#%%
q = QZFM("COM3")
#%%
q.auto_start(zero_calibrate=False)


#%%
q.set_gain('1x')

#%%
# sample at 1kHz
sr = 1000
zero_time = 10
measure_time = 20
save = True
q.field_zero(True, True, False)
fp = '..//data//mar22//'
folder_name = '77HzConstancy'
folder_name = make_folder(fp, folder_name)

n_measurement = 100
for i in range(n_measurement):
    print('performing measurement no.{} out of {}'.format(i+1, 5))
    # zero
    for i in range(zero_time):
        time_.sleep(1)
        print(i)
    q.field_zero(False)
    # measure
    out = labjack_measure(measure_time, sr, ["AIN0", "AIN1", "AIN2"], [0.0027, 0.0027, 0.0027], 
                                              [1.0, 1.0, 1.0], False)
    out_df = pd.DataFrame(out.T)
    out_df.columns = ['Epoch Time', 'x', 'y', 'z']
    out_df.set_index("Epoch Time", inplace=True)
    
    time = datetime.now().strftime('%y%m%dT%H%M%S')
    # save
    if save:
        out_df.to_csv(fp+folder_name+'//'+time+".csv")

#%%
plt.figure()
for axis in range(3):
    a, b = signal.periodogram(out[axis+1, :], sr)
    plt.semilogy(a, np.sqrt(b), label=axis)
plt.legend()
plt.grid()
plt.show()

