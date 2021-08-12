import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt


class ButterFilter:
    def __init__(self, order, low_f=None, high_f=None, filter_type='lowpass', fs=512, display=False):
        nyq_f = 0.5 * fs
        if filter_type is 'lowpass':
            self.b, self.a = signal.butter(order, high_f/nyq_f, btype=filter_type)
        elif filter_type is 'highpass':
            self.b, self.a = signal.butter(order, low_f/nyq_f, btype=filter_type)
        elif filter_type is 'bandpass' or filter_type is 'bandstop':
            self.b, self.a = signal.butter(order, [low_f/nyq_f, high_f/nyq_f], btype=filter_type)
        else:
            print('[ButterFilter] Wrong filter type!')
        if display is True:
            w, h = signal.freqz(self.b, self.a)
            plt.plot((fs * 0.5 / np.pi) * w, abs(h))
            plt.show()

    def apply_filt(self, data):
        return signal.filtfilt(self.b, self.a, data, padtype='odd', axis=0)

    def compute_filt(self, sig):
        return signal.filtfilt(self.b, self.a, sig, padtype='odd')
