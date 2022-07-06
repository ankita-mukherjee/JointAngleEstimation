import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import glob
import os

path = "/workspaces/JointAngleEstimation/Matlab/vicon_csv/"
all_files = glob.glob(path + "*.csv")


for trial in all_files:
    df = pd.read_csv(trial)

    # Lengths shoflex
    shohip_length3 = np.sqrt(
        (df["SHO_Y_mm"] - df["ASI_Y_mm"]) ** 2 + (df["SHO_Z_mm"] - df["ASI_Z_mm"]) ** 2
    )
    shoelb_length3 = np.sqrt(
        (df["SHO_Y_mm"] - df["ELB_Y_mm"]) ** 2 + (df["SHO_Z_mm"] - df["ELB_Z_mm"]) ** 2
    )
    showri_length3 = np.sqrt(
        (df["SHO_Y_mm"] - df["WRI_Y_mm"]) ** 2 + (df["SHO_Z_mm"] - df["WRI_Z_mm"]) ** 2
    )
    elbwri_length3 = np.sqrt(
        (df["ELB_Y_mm"] - df["WRI_Y_mm"]) ** 2 + (df["ELB_Z_mm"] - df["WRI_Z_mm"]) ** 2
    )
    elbhip_length3 = np.sqrt(
        (df["ELB_Y_mm"] - df["ASI_Y_mm"]) ** 2 + (df["ELB_Z_mm"] - df["ASI_Z_mm"]) ** 2
    )
    # Shoulder flex
    cSHOf = (shohip_length3**2 + shoelb_length3**2 - elbhip_length3**2) / (
        2 * shohip_length3 * shoelb_length3
    )
    shoulderflexangle = np.arccos(cSHOf)
    thetashoulderflex = shoulderflexangle * 180 / math.pi

    thetashoulderflex = pd.Series(thetashoulderflex.tolist(), index=df["Frame"].values)
    # save in csv
    # path= "/workspaces/JointAngleEstimation/vicon_csv/vicon_processed/"
    csv_file_path = os.path.split(trial)[0] + "/processed_" + os.path.split(trial)[1]
    thetashoulderflex.to_csv(csv_file_path, header=False)
