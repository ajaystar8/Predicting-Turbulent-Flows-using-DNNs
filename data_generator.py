import pandas as pd
from scipy.integrate import odeint

from utils.utils import check_for_laminarization
from generate_moehlis import *


def generate_datasets(num_of_datasets, threshold=1e-6):
    count = 0
    datasets = []
    init_params = []

    while count < num_of_datasets:
        a4 = 0.1 * np.round(np.random.uniform(0, 1), 3)
        y0 = [1, 0.07066, -0.07076, a4, 0, 0, 0, 0, 0]

        time_points = np.linspace(0, 4000, 4001)

        sol = odeint(MoehlisCoefficientsGenerator, y0, time_points, args=(params,))
        if check_for_laminarization(pd.DataFrame(sol), threshold):
            continue
        else:
            count = count + 1
            datasets.append(sol)
            init_params.append(y0)

    datasets = np.array(datasets)
    init_params = np.array(init_params)
    return datasets, init_params


# Utility Function: Generating sequences
def generate_sequences(A, model):
    seqNoList = range(0, 2)
    seq_len = 10

    saveFilename = "./Sequences/series"

    for seqNo in seqNoList:
        print(seqNo + 1)

        testSeq = A[seqNo:seqNo + 1]
        predSeq = testSeq[:, :seq_len]

        for i in np.arange(testSeq.shape[1] - seq_len):
            nextState = model.predict(predSeq[:, i:i + seq_len], verbose=0)
            predSeq = np.concatenate((predSeq, [nextState]), axis=1)

        testSeq = testSeq.reshape(-1, 9)
        predSeq = predSeq.reshape(-1, 9)

        save_filename = f"{saveFilename}{seqNo + 1}.npz"
        np.savez(save_filename, testSeq=testSeq, predSeq=predSeq)
