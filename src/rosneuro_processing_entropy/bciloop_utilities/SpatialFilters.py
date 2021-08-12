import numpy as np
import scipy.linalg as la


def car_filter(data, axis=0):
    ref = np.mean(data, axis=axis)
    return data - ref[:, np.newaxis]


class CommonSpatialPatterns:
    def __init__(self, ncsp=None, coeff=[], fullcoeff=[]):
        self.ncsp = ncsp
        self.fullcoeff = fullcoeff
        self.coeff = coeff

    def set_csp_dim(self, ncsp):
        self.ncsp = ncsp

    def set_coeff(self, coeff):
        self.coeff = coeff

    def set_fullcoeff(self, fullcoeff):
        self.fullcoeff = fullcoeff
        if self.ncsp is not None:
            self.coeff = self.extract_filters(self.ncsp)

    def reset_coeff(self):
        self.fullcoeff = []
        self.coeff = []

    def compute_filters(self, x, y):
        x = x.T
        x0 = x[:, y == 0]
        x1 = x[:, y == 1]

        # Compute covariance matrices
        R0 = np.matmul(x0, x0.T)
        R0 = R0 / np.trace(R0)
        R1 = np.matmul(x1, x1.T)
        R1 = R1 / np.trace(R1)
        Rsum = R0 + R1

        # Compute eigenvalues
        eigenval, eigenvec = la.eigh(Rsum)
        idx = np.argsort(eigenval[::-1])
        eigenval[::-1].sort()
        eigenvec[:] = eigenvec[:, idx]

        # Whiten data using whithening transformation matrix
        W = np.matmul(np.sqrt(la.pinv(np.diag(eigenval))), eigenvec.T)
        S0 = np.matmul(np.matmul(W, R0), W.T)
        S1 = np.matmul(np.matmul(W, R1), W.T)

        # Compute the generalized eigenvalues/vectors
        gen_eigenval, gen_eigenvec = la.eigh(S0, S1)
        idx = np.argsort(gen_eigenval)
        gen_eigenvec[:] = gen_eigenvec[:, idx]

        # Compute the csp coefficients/projection matrix
        coeff = np.matmul(gen_eigenvec.T, W)
        self.fullcoeff = coeff.T
        if self.ncsp is not None:
            self.coeff = self.extract_filters(self.ncsp)
        return coeff.T

    def extract_filters(self, ncsp):
        if self.fullcoeff.size == 0:
            print('[CommonSpatialPatterns] No filters to extract!')
            return []
        fullcoeff = self.fullcoeff.T
        filtmat = np.zeros((ncsp, np.shape(fullcoeff)[0]))
        i = 0
        for d in range(1, ncsp+1):
            if (d % 2) == 0:
                i = i + 1
                filtmat[d-1, :] = fullcoeff[-i, :]
            else:
                filtmat[d-1, :] = fullcoeff[i, :]
        return filtmat.T

    def apply(self, data):
        return np.dot(data, self.coeff)

    def apply_full(self, data):
        return np.dot(data, self.fullcoeff)

    # def apply_onesample(self, data):
    #     return np.dot(data, self.coeff)
    #
    # def apply_full_onesample(self, data):
    #     return np.dot(data, self.fullcoeff)

