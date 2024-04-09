# apr 2024
# multisensor run

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
from SiglentDevices import DG1032Z
from scipy.signal import butter, sosfilt, sosfreqz

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
plt.rcParams.update({'font.size': 14})
#%%
# Digital Function Generator
awg = DG1032Z(hostname='USB0::0x1AB1::0x0642::DG1ZA232603182::INSTR')
awg.query('*IDN?')

#%%
AAL9 = QZFM("COM3")  

#%%
AAY4 = QZFM("COM4")
#%%
AAL9.auto_start(zero_calibrate=False)
#%%
AAY4.auto_start(zero_calibrate=False)
#%%
gain = '0.33x'
#%%
AAL9.set_gain(gain)
#%%
AAY4.set_gain(gain)

#%%
# sample at 1kHz
sr = 1000
zero_time = 15
measure_time = 30
save = True
gain_dict = {'0.1x':0.27,
             '0.33x':0.9,
             '1x':2.7,
             '3x':8.1}
#%%
if __name__ == '__main__':
    t = 30
    save = False
    # initialize queue object for saving outputs of functions
    output_queue = queue.Queue()
     
    # turn on field zeroing for sensor 1
    t0 = Thread(target=AAL9.field_zero, args=(True, True, False))
    t0.start()
    t0.join()
    
    # turn on field zeroing for sensor 2
    t1 = Thread(target=AAY4.field_zero, args=(True, True, False))
    t1.start()
    t1.join()
    
  
    # initialize thread of QZFM module for coil offset reading
    t2 = Thread(target=AAL9.read_offsets_custom, args=(int(t*7.5), output_queue))
    print('AAL9 thread defined')
    
    t3 = Thread(target=AAY4.read_offsets_custom, args=(int(t*7.5), output_queue))
    print('AAY4 thread defined')
    
    
    # # initialize thread of labjack for cell reading
    # t2 = Thread(target=labjack_measure, args=(t, 1000, ["AIN0", "AIN1", "AIN2"], [0.0027, 0.0027, 0.0027], [10.0, 10.0, 10.0], output_queue))
    # print("Labjack thread defined")
                                                               
    # start the threads
    t2.start()
    print('AAL9 thread started')
    t3.start()
    print('AAY4 thread started')

    # join the threads
    t2.join()
    print("AAL9 joined")
    t3.join()
    print("AAY4 joined")
    
    # stop field zeroing
    t4 = Thread(target=AAL9.field_zero, args=(False, True, False))
    t4.start()
    t4.join()
    print("AAL9 finished")
    
    t5 = Thread(target=AAY4.field_zero, args=(False, True, False))
    t5.start()
    t5.join()
    print("AAY4 finished")
        
    cnt = 0
    out = []
    while cnt < 2:
        try:
            out.append(output_queue.get())
            cnt +=1
        except Exception:
            break
        
    s1_coil = out[0]
    s2_coil = out[1]

    # cell_readings = pd.DataFrame(cell_readings.T)
    # cell_readings.columns = ['Epoch Time', 'x', 'y', 'z']
    # cell_readings.set_index("Epoch Time", inplace=True)
    
    # if save:
    #     file_name = "" #input("Name of the file is?\n")
    #     coil_offsets.to_csv("flip_data//feb20//"+file_name+datetime.now().strftime('%y%m%dT%H%M%S')+"coiloffsets"+".csv")
    #     cell_readings.to_csv("flip_data//feb20//"+file_name+datetime.now().strftime('%y%m%dT%H%M%S')+"cellreadings"+".csv")


#%%
plt.figure()
for axis in ['x', 'y', 'z']: 
    
    
    plt.plot(s1_coil[axis].iloc[:], label="sensor 1 Coil Reading {}".format(axis))
    plt.plot(s2_coil[axis].iloc[:], label="sensor 2 Coil Reading {}".format(axis))
    #plt.plot((coil_offsets.index-cell_readings.index[0])[25*8:], coil_offsets[axis].iloc[25*8:]-coil_offsets[axis].iloc[25*8:].mean(), label="Coil Reading {}".format(axis))
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("B field [pT]")
    plt.title("Simultaneous Zeroing")
#plt.plot((coil_offsets.index-cell_readings.index[0])[25*8:], coil_offsets['temp'].iloc[25*8:], label="temperature voltage")
plt.grid()
plt.show()


#%%

out = labjack_measure(measure_time, sr, ["AIN0", "AIN3"], [gain_dict[gain], gain_dict[gain]], 
                                          [10.0, 10.0], False)

#%%
plt.figure(figsize=(1.2*6.4, 1.2*4.8))
for axis in range(2):

    plt.plot(out[axis+1, :], label='sensor {}'.format(axis+1))
plt.legend()
plt.ylabel('B (nT)')
plt.xlabel('Time (A.U.)')
plt.tight_layout()
plt.show()


#%%
fig, ax = plt.subplots(2, 1, figsize=(1.2*6.4, 1.5*1.2*4.8))


for axis in ['x', 'y', 'z']: 
    ax[0].plot(np.array(s1_coil[axis])/1000, label="sensor 1 Coil Reading {}".format(axis))
    ax[0].plot(np.array(s2_coil[axis])/1000, label="sensor 2 Coil Reading {}".format(axis))
    #plt.plot((coil_offsets.index-cell_readings.index[0])[25*8:], coil_offsets[axis].iloc[25*8:]-coil_offsets[axis].iloc[25*8:].mean(), label="Coil Reading {}".format(axis))
    ax[0].legend()
ax[0].set_xlabel("Time [A.U.]")
ax[0].set_ylabel("B field [nT]")
ax[0].grid()


for axis in range(2):

    ax[1].plot(out[axis+1, :], label='sensor {}'.format(axis+1))
ax[1].legend()
ax[1].set_ylabel('B (nT)')
ax[1].set_xlabel('Time (A.U.)')
ax[1].grid()
plt.tight_layout()
plt.show()













