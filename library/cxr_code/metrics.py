import numpy as np
from functools import partial

def create_binary_metric(func, name, **kwargs):
    def metric(preds, targets, sigmoid=True, thresh=0.5, **kwargs):
        if sigmoid: preds = 1/(1 + np.exp(-preds))
        preds = (preds >= thresh).astype(np.uint8)
        return func(y_true=targets, y_pred=preds, **kwargs)
    metric = partial(metric, **kwargs)
    setattr(metric, '__name__', name)
    return metric