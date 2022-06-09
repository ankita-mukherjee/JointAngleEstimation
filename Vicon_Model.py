import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import csv

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({"font.size": 22})

# from vicon
df = pd.read_csv(
    "/workspaces/JointAngleEstimation/Matlab/python_dl009_kneeflexoutput_020knee.csv",
    header=None,
    usecols=[0, 1],
    index_col=0,
)
df.rename(columns={1: "vicon angle"}, inplace=True)
df.index.names = ["frame number"]

# from model output
df1 = pd.read_csv(
    "/workspaces/JointAngleEstimation/CSV_Output/P_Frames-kneeflex_020.csv",
    usecols=[0, 7],
    skiprows=lambda x: x not in range(97, 214),
    header=None,
    index_col=0,
)
df1.rename(columns={7: "model angle"}, inplace=True)
df1.index.names = ["frame number"]

ax = df.plot(
    figsize=(25, 10),
    kind="line",
    color="g",
    legend=True,
)
plt.title("Knee Flexion Angle")
plt.xlabel("Frame")
plt.ylabel("Joint Angle (Degrees)")
df1.plot(ax=ax, kind="line", color="b", legend=True)
plt.savefig(
    "/workspaces/JointAngleEstimation/Matlab/Plot_vicon_angle_vs_frames/Vicon_Model_Knee_Flexion.png"
)
plt.show()
