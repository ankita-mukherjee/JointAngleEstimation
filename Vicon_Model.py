import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# from sklearn.metrics import matthews_corrcoef
import scipy
from math import sqrt

window_width = 10

# Smoothing
def apply_smoothing(angles, window_width):
    cumsum_vec = np.cumsum(np.insert(angles, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({"font.size": 18})

trials = [
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/python_dl102_elbflexoutput_010elb.csv",
        "model": "/workspaces/JointAngleEstimation/CSV_Output/P_Frames-dl102_elbflex_010.csv",
        "name": "trial 010",
        "skip_lo": 2,
        "skip_hi": 117,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 117,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/python_dl102_elbflexoutput_013elb.csv",
        "model": "/workspaces/JointAngleEstimation/CSV_Output/P_Frames-dl102_elbflex_013.csv",
        "name": "trial 013",
        "skip_lo": 2,
        "skip_hi": 149,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 149,
        "color": "b",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/python_dl102_elbflexoutput_016elb.csv",
        "model": "/workspaces/JointAngleEstimation/CSV_Output/P_Frames-dl102_elbflex_016.csv",
        "name": "trial 016",
        "skip_lo": 2,
        "skip_hi": 130,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 128,
        "color": "g",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/python_dl102_elbflexoutput_002elb.csv",
        "model": "/workspaces/JointAngleEstimation/CSV_Output/P_Frames-dl102_elbflex_002.csv",
        "name": "trial 002",
        "skip_lo": 1,
        "skip_hi": 175,
        "Vicon_skip_lo": 8,
        "Vicon_skip_hi": 172,
        "color": "c",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/python_dl102_elbflexoutput_005elb.csv",
        "model": "/workspaces/JointAngleEstimation/CSV_Output/P_Frames-dl102_elbflex_005.csv",
        "name": "trial 005",
        "skip_lo": 2,
        "skip_hi": 127,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 127,
        "color": "m",
    },
]

ax = None

for trial in trials:
    # from vicon
    dfv = pd.read_csv(
        trial["vicon"],
        header=None,
        usecols=[0, 1],
        skiprows=lambda x: x
        not in range(trial["Vicon_skip_lo"], trial["Vicon_skip_hi"]),
        index_col=0,
    )
    dfv.rename(columns={1: f"vicon angle ({trial['name']})"}, inplace=True)
    dfv.index.names = ["frame number"]
    #print(dfv)

    # from model output
    dfm = pd.read_csv(
        trial["model"],
        usecols=[0, 1],
        skiprows=lambda x: x not in range(trial["skip_lo"], trial["skip_hi"]),
        header=None,
        index_col=0,
    )
    dfm.rename(columns={1: f"model angle ({trial['name']})"}, inplace=True)
    dfm.index.names = ["frame number"]
    #print(dfm)

    newy = apply_smoothing(
        angles=dfm[f"model angle ({trial['name']})"].values, window_width=window_width
    )
    dfs = pd.DataFrame(
        {f"model angle (after smoothing) ({trial['name']})": newy.tolist()}
    )
    dfs.index = dfm.index[window_width - 1 :].values
    dfs.index.names = ["frame number"]
    #print(dfs)

    rms = sqrt(
        mean_squared_error(
            dfv[f"vicon angle ({trial['name']})"].values,
            dfs[f"model angle (after smoothing) ({trial['name']})"].values,
        )
    )

    mae = mean_absolute_error(
        dfv[f"vicon angle ({trial['name']})"].values,
        dfs[f"model angle (after smoothing) ({trial['name']})"].values,
    )

    r = scipy.stats.pearsonr(
        dfv[f"vicon angle ({trial['name']})"].values,
        dfs[f"model angle (after smoothing) ({trial['name']})"].values,
    )
    print(
        f"RMSE        of model vs vicon (elbow flexion {trial['name']}): ",
        round(rms, 2),
    )
    print(
        f"MAE         of model vs vicon (elbow flexion {trial['name']}): ",
        round(mae, 2),
    )
    print(
        f"Correlation of model vs vicon (elbow flexion {trial['name']}): ",
        round(r[0], 3),
    )

    if ax is None:
        ax = dfv.plot(figsize=(25, 10), linestyle="dashed", color=trial["color"])
    else:
        dfv.plot(ax=ax, linestyle="dashed", color=trial["color"])

    plt.title("Elbow_Flexion Angle")
    plt.xlabel("Frame")
    plt.ylabel("Joint Angle (Degrees)")
    dfs.plot(
        ax=ax, color=trial["color"], linewidth=5, label=f"model angle ({trial['name']})"
    )

    ax.legend()

plt.savefig(
    "/workspaces/JointAngleEstimation/Matlab/Plot_vicon_angle_vs_frames/Vicon_Model_Elbow_Flexion.png"
)

plt.show()
