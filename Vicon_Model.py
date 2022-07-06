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
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_001.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_001.csv",
        "name": "trial 001",
        "skip_lo": 2,
        "skip_hi": 208,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 206,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_002.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_002.csv",
        "name": "trial 002",
        "skip_lo": 2,
        "skip_hi": 189,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 187,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_003.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_003.csv",
        "name": "trial 003",
        "skip_lo": 2,
        "skip_hi": 208,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 206,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_004.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_004.csv",
        "name": "trial 004",
        "skip_lo": 2,
        "skip_hi": 174,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 172,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_005.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_005.csv",
        "name": "trial 005",
        "skip_lo": 2,
        "skip_hi": 194,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 192,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_006.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_006.csv",
        "name": "trial 006",
        "skip_lo": 2,
        "skip_hi": 205,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 203,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_007.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_007.csv",
        "name": "trial 007",
        "skip_lo": 2,
        "skip_hi": 185,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 183,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_008.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_008.csv",
        "name": "trial 008",
        "skip_lo": 2,
        "skip_hi": 189,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 187,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_009.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_009.csv",
        "name": "trial 009",
        "skip_lo": 2,
        "skip_hi": 180,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 178,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_010.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_010.csv",
        "name": "trial 010",
        "skip_lo": 2,
        "skip_hi": 185,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 183,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_011.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_011.csv",
        "name": "trial 011",
        "skip_lo": 2,
        "skip_hi": 202,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 200,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_012.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_012.csv",
        "name": "trial 012",
        "skip_lo": 2,
        "skip_hi": 198,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 196,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_013.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_013.csv",
        "name": "trial 013",
        "skip_lo": 2,
        "skip_hi": 195,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 193,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_014.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_014.csv",
        "name": "trial 014",
        "skip_lo": 2,
        "skip_hi": 201,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 199,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_015.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_015.csv",
        "name": "trial 015",
        "skip_lo": 2,
        "skip_hi": 181,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 179,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_016.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_016.csv",
        "name": "trial 016",
        "skip_lo": 2,
        "skip_hi": 185,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 183,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_017.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_017.csv",
        "name": "trial 017",
        "skip_lo": 2,
        "skip_hi": 199,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 197,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_018.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_018.csv",
        "name": "trial 018",
        "skip_lo": 2,
        "skip_hi": 201,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 199,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_019.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_019.csv",
        "name": "trial 019",
        "skip_lo": 2,
        "skip_hi": 194,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 192,
        "color": "r",
    },
    {
        "vicon": "/workspaces/JointAngleEstimation/Matlab/vicon_csv/processed_dl100_shoflex_020.csv",
        "model": "/workspaces/JointAngleEstimation/csv_ouput/P_Frames-dl100_shoflex_020.csv",
        "name": "trial 020",
        "skip_lo": 2,
        "skip_hi": 224,
        "Vicon_skip_lo": 9,
        "Vicon_skip_hi": 222,
        "color": "r",
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
        usecols=[0, 4],
        skiprows=lambda x: x not in range(trial["skip_lo"], trial["skip_hi"]),
        header=None,
        index_col=0,
    )
    dfm.rename(columns={4: f"model angle ({trial['name']})"}, inplace=True)
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
        f"RMSE        of model vs vicon (shoulder_flexion {trial['name']}): ",
        round(rms, 2),
    )
    # print(
    #     f"MAE         of model vs vicon (shoulder_flexion {trial['name']}): ",
    #     round(mae, 2),
    # )
    # print(
    #     f"Correlation of model vs vicon (shoulder_flexion {trial['name']}): ",
    #     round(r[0], 3),
    # )

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
    "/workspaces/JointAngleEstimation/Matlab/Plot_vicon_angle_vs_frames/Vicon_Model_shoulder_flexion.png"
)

plt.show()
