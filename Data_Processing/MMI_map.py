import numpy as np

def MMI_map(y_true, MMI_cutoffs = [0.01667, 0.13729, 0.38246, 0.90221, 1.76520, 3.33426, 6.37432, 12.16025]):
    MMI_val = np.searchsorted(MMI_cutoffs, y_true.reshape(-1), side = 'right')
    return MMI_val