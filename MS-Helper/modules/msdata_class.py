from sre_constants import SUCCESS
from pybaselines import Baseline
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from time import time 

# Define a class to store mass spectrometry data
class MSData():
    def __init__(self):
        self.original_data = None
        self.working_data = None
        self.baseline = None
        self.baseline_corrected = None
        self.filtered = None
        self.peaks: Dict[int : peak_params] = {}
        self.baseline_toggle = False
        self.baseline_corrected_clipped = None
        self.last_baseline_corrected = time()
    
    def import_csv(self, path:str):
        data = pd.read_csv(path)
        data = data.dropna()
        self.original_data = data.to_numpy()
        self.working_data = self.original_data
        self.correct_baseline(0)
        
        print("Data imported successfully")
        self.filter_data(window_length=50)
    
    def clip_data(self, L_clip:int, R_clip:int):
        self.working_data = self.original_data[(self.original_data[:,0] > L_clip) & (self.original_data[:,0] < R_clip)]
        self.baseline_corrected_clipped = self.baseline_corrected[(self.baseline_corrected[:,0] > L_clip) & (self.baseline_corrected[:,0] < R_clip)]
        

    def filter_data(self, window_length, polyorder=2):
        if window_length % 2 == 0:
            window_length += 1  # Ensure window_length is odd

        self.filtered = np.column_stack((self.working_data[:, 0], savgol_filter(self.working_data[:, 1], window_length=window_length, polyorder=polyorder)))

    def correct_baseline(self, window):
        if not self.baseline_toggle:
            self.baseline_corrected = self.working_data
            self.baseline_corrected_clipped = self.working_data
            self.baseline = np.column_stack((self.working_data[:,0], [0]*len(self.working_data)))
            self.last_baseline_corrected = time()
            return
          
        baseline_fitter = Baseline(x_data=self.working_data[:,0])
        bkg_4, params_4 = baseline_fitter.snip(self.working_data[:,1], max_half_window=window, decreasing=True, smooth_half_window=3)    
        self.baseline = np.column_stack((self.working_data[:,0], bkg_4 ))
        self.baseline_corrected = np.column_stack((self.working_data[:,0], self.working_data[:,1] - bkg_4))
        self.baseline_corrected_clipped = self.baseline_corrected
        self.last_baseline_corrected = time()
    
    def guess_sampling_rate(self):
        sampling_rate = np.mean(np.diff(self.working_data[:,0]))
        return sampling_rate
        

@dataclass
class peak_params:
    A: float
    x0_init: float
    x0_refined: float
    sigma_L: float
    sigma_R: float
    width: np.ndarray
    fitted: bool

if __name__ == "__main__":
    ms = MSData()
    ms.import_csv(rf"D:\MassSpec\Um_2-1_1x.csv")
    #ms.clip_data(10000, 12000)
    ms.baseline_toggle = True
    ms.correct_baseline(40)
    ms.guess_sampling_rate()
    ms.filter_data(50)
    print(ms.filtered[:,0])
    print(ms.working_data[:,0])

