import math
import numpy as np
import numpy.ma as ma


class ShannonEntropy:
    def __init__(self, nbins=None):
        self.nbins = nbins

    def apply(self, data):
        return self.compute(np.exp(data))

    def compute(self, sig):
        if self.nbins is None:
            nbins = math.log2(np.shape(sig)[0] + 1)
        else:
            nbins = self.nbins
        maximum = np.amax(sig, axis=0)
        minimum = np.amin(sig, axis=0)
        step = (maximum - minimum) / nbins
        sdistr = self.histogram(sig, nbins, minimum, step)
        return (-1) * np.sum(sdistr * ma.log2(sdistr), axis=0)

    def histogram(self, sig, nb, m, stp):
        hist = np.zeros((nb, np.shape(sig)[1]))
        for x in sig:
            idx = np.ceil((x-m)/stp) - 1
            idx[idx < 0] = 0
            hist[idx.astype(int), np.arange(np.shape(sig)[1])] = hist[idx.astype(int), np.arange(np.shape(sig)[1])] + 1
        return hist/np.shape(sig)[0]

