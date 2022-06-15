import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as MAE

# from sklearn.metrics import matthews_corrcoef
import scipy
from math import sqrt


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({"font.size": 22})

# from vicon
df = pd.read_csv(
    "/workspaces/JointAngleEstimation/Matlab/python_dl009_elbflexoutput_020elb.csv",
    header=None,
    usecols=[0, 1],
    # skiprows=lambda x: x not in range(5, 262),
    index_col=0,
)
df.rename(columns={1: "vicon angle"}, inplace=True)
df.index.names = ["frame number"]
print(df)

# from model output
df1 = pd.read_csv(
    "/workspaces/JointAngleEstimation/CSV_Output/P_Frames-Elbflex_020.csv",
    usecols=[0, 1],
    skiprows=lambda x: x not in range(111, 262),
    header=None,
    index_col=0,
)
df1.rename(columns={1: "model angle"}, inplace=True)
df1.index.names = ["frame number"]


# Smoothing
def apply_smoothing(angles, window_width):
    cumsum_vec = np.cumsum(np.insert(angles, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec


window_width = 6
newy = apply_smoothing(angles=df1["model angle"].values, window_width=window_width)

# Check newy values
df3 = pd.DataFrame(newy)
df3.rename(columns={0: "Smooth model angle"}, inplace=True)
df3.index = df1.index[window_width - 1 :].values
df3.index.names = ["frame number"]
path = r"/workspaces/JointAngleEstimation/CSV_Output/Smooth_Model_Data/"
df3.to_csv(path + "Elbow_Flexion.csv")
print(df3)


ax = df.plot(
    figsize=(25, 10),
    kind="line",
    color="g",
    legend=True,
)
plt.title("Elbow_Flexion Angle")
plt.xlabel("Frame")
plt.ylabel("Joint Angle (Degrees)")
df1.plot(ax=ax, kind="line", color="b")
plt.plot(df1.index[window_width - 1 :], newy, color="r")
plt.gca().legend(("Vicon", "Model", "Smooth Model"))
plt.savefig(
    "/workspaces/JointAngleEstimation/Matlab/Plot_vicon_angle_vs_frames/Vicon_Model_Elb_Flexion.png"
)
plt.show()


# RMSE Calculation

rms = sqrt(
    mean_squared_error(df["vicon angle"].values, df3["Smooth model angle"].values)
)
print("RMSE of shoulder is:", rms)
# print("RMSE of kneeflex is:",'%.2f' %rms)

# MAE Calculation
mae = MAE(df["vicon angle"].values, df3["Smooth model angle"].values)
print("MAE of knee is:", mae)

# Correlation Calculation

r = scipy.stats.pearsonr(df["vicon angle"].values, df3["Smooth model angle"].values)
print(" Correlation of knee is:", r)
