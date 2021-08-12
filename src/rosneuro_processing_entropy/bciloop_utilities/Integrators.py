import numpy as np


class ExponentialIntegrator:
    def __init__(self, alpha=0.9, threshold=0.5, rejection=0.5, begin=0):
        self.alpha = alpha
        self.threshold = threshold
        self.rejection = rejection
        self.begin = begin
        self.old_pred = []
        self.old_ipp = []
        self.reset()

    def reset(self):
        self.old_pred = self.begin
        self.old_ipp = np.array([0.5, 0.5])
        # if self.old_pred == 0:
        #     self.old_ipp = np.array([1.0, 0.0])
        # else:
        #     self.old_ipp = np.array([0.0, 1.0])

    def apply(self, pp):
        if np.any(pp > self.rejection):
            ipp = self.old_ipp * self.alpha + pp * (1 - self.alpha)
        else:
            ipp = self.old_ipp

        decision = np.nonzero(ipp > self.threshold)[0]
        if decision.size == 0:
            pred = self.old_pred
        else:
            pred = decision
        self.old_ipp = ipp
        self.old_pred = pred
        return pred, ipp

    def apply_array(self, pp_array):
        predictions = np.empty(np.shape(pp_array)[0])
        probabilities = np.empty(np.shape(pp_array))
        for i in np.arange(np.shape(pp_array)[0]):
            predictions[i], probabilities[i, :] = self.apply(np.squeeze(pp_array[i, :]))
        return predictions, probabilities

