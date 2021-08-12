from scipy.signal import hilbert
import numpy as np


class Hilbert:
    def __init__(self):
        self.data = []

    def apply(self, input):
        self.data = hilbert(input, axis=0)

    def get_real(self):
        return self.data.real

    def get_imag(self):
        return self.data.imag

    def get_envelope(self):
        return np.abs(self.data)

