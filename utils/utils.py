import numpy as np


def check_for_laminarization(sol, threshold):
    sol = sol.loc[:, 4].to_numpy()
    for i in range(100, len(sol)):
        if np.abs(sol[i] - sol[i - 100]) < threshold:
            return 1
    return 0


