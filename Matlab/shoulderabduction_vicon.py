import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

df = pd.read_csv("dl009_shoabd_020.csv")
df1 = pd.read_csv("dl009_elbflex_001.csv")
df2 = pd.read_csv("dl009_shoext_001.csv")
df3 = pd.read_csv("dl009_shoflex_001.csv")


# Lengths shoabd
shohip_length = np.sqrt(
    (df["SHO_X_mm"] - df["ASI_X_mm"]) ** 2 + (df["SHO_Z_mm"] - df["ASI_Z_mm"]) ** 2
)
shoelb_length = np.sqrt(
    (df["SHO_X_mm"] - df["ELB_X_mm"]) ** 2 + (df["SHO_Z_mm"] - df["ELB_Z_mm"]) ** 2
)
showri_length = np.sqrt(
    (df["SHO_X_mm"] - df["WRI_X_mm"]) ** 2 + (df["SHO_Z_mm"] - df["WRI_Z_mm"]) ** 2
)
elbwri_length = np.sqrt(
    (df["ELB_X_mm"] - df["WRI_X_mm"]) ** 2 + (df["ELB_Z_mm"] - df["WRI_Z_mm"]) ** 2
)
elbhip_length = np.sqrt(
    (df["ELB_X_mm"] - df["ASI_X_mm"]) ** 2 + (df["ELB_Z_mm"] - df["ASI_Z_mm"]) ** 2
)

 # Lengths elbflex
shohip_length1 = np.sqrt(
    (df1["SHO_X_mm"] - df1["ASI_X_mm"]) ** 2 + (df1["SHO_Z_mm"] - df1["ASI_Z_mm"]) ** 2
)
shoelb_length1 = np.sqrt(
    (df1["SHO_X_mm"] - df1["ELB_X_mm"]) ** 2 + (df1["SHO_Z_mm"] - df1["ELB_Z_mm"]) ** 2
)
showri_length1 = np.sqrt(
    (df1["SHO_X_mm"] - df1["WRI_X_mm"]) ** 2 + (df1["SHO_Z_mm"] - df1["WRI_Z_mm"]) ** 2
)
elbwri_length1 = np.sqrt(
    (df1["ELB_X_mm"] - df1["WRI_X_mm"]) ** 2 + (df1["ELB_Z_mm"] - df1["WRI_Z_mm"]) ** 2
)
elbhip_length1 = np.sqrt(
    (df1["ELB_X_mm"] - df1["ASI_X_mm"]) ** 2 + (df1["ELB_Z_mm"] - df1["ASI_Z_mm"]) ** 2
)

 # Lengths shoext
shohip_length2 = np.sqrt(
    (df2["SHO_X_mm"] - df2["ASI_X_mm"]) ** 2 + (df2["SHO_Z_mm"] - df2["ASI_Z_mm"]) ** 2
)
shoelb_length2 = np.sqrt(
    (df2["SHO_X_mm"] - df2["ELB_X_mm"]) ** 2 + (df2["SHO_Z_mm"] - df2["ELB_Z_mm"]) ** 2
)
showri_length2 = np.sqrt(
    (df2["SHO_X_mm"] - df2["WRI_X_mm"]) ** 2 + (df2["SHO_Z_mm"] - df2["WRI_Z_mm"]) ** 2
)
elbwri_length2 = np.sqrt(
    (df2["ELB_X_mm"] - df2["WRI_X_mm"]) ** 2 + (df2["ELB_Z_mm"] - df2["WRI_Z_mm"]) ** 2
)
elbhip_length2 = np.sqrt(
    (df2["ELB_X_mm"] - df2["ASI_X_mm"]) ** 2 + (df2["ELB_Z_mm"] - df2["ASI_Z_mm"]) ** 2
)

 # Lengths shoflex
shohip_length3 = np.sqrt(
    (df3["SHO_X_mm"] - df3["ASI_X_mm"]) ** 2 + (df3["SHO_Z_mm"] - df3["ASI_Z_mm"]) ** 2
)
shoelb_length3 = np.sqrt(
    (df3["SHO_X_mm"] - df3["ELB_X_mm"]) ** 2 + (df3["SHO_Z_mm"] - df3["ELB_Z_mm"]) ** 2
)
showri_length3 = np.sqrt(
    (df3["SHO_X_mm"] - df3["WRI_X_mm"]) ** 2 + (df3["SHO_Z_mm"] - df3["WRI_Z_mm"]) ** 2
)
elbwri_length3 = np.sqrt(
    (df3["ELB_X_mm"] - df3["WRI_X_mm"]) ** 2 + (df3["ELB_Z_mm"] - df3["WRI_Z_mm"]) ** 2
)
elbhip_length3 = np.sqrt(
    (df3["ELB_X_mm"] - df3["ASI_X_mm"]) ** 2 + (df3["ELB_Z_mm"] - df3["ASI_Z_mm"]) ** 2
)

# Shoulder abd
cSHOa = (shohip_length**2 + shoelb_length**2 - elbhip_length**2) / (
    2 * shohip_length * shoelb_length
)
shoulderabdangle = np.arccos(cSHOa)
thetashoulderabd = shoulderabdangle * 180 / math.pi
# thetashoulder1 = np.real(thetashoulder);

 # Shoulder ext
cSHOe = (shohip_length2**2 + shoelb_length2**2 - elbhip_length2**2) / (
    2 * shohip_length2 * shoelb_length2
)
shoulderextangle = np.arccos(cSHOe)
thetashoulderext = shoulderextangle * 180 / math.pi


 # Shoulder flex
cSHOf = (shohip_length3**2 + shoelb_length3**2 - elbhip_length3**2) / (
    2 * shohip_length3 * shoelb_length3
)
shoulderflexangle = np.arccos(cSHOf)
thetashoulderflex = shoulderflexangle * 180 / math.pi


# # Elbow
cELBa = (shoelb_length1**2 + elbwri_length1**2 - showri_length1**2) / (
    2 * shoelb_length1 * elbwri_length1
)
elbowangle = np.arccos(cELBa)
thetaelbowflex = elbowangle * 180 / math.pi

thetashoulderabd = pd.Series(thetashoulderabd.tolist(), index=df["Frame"].values)
thetaelbowflex = pd.Series(thetaelbowflex.tolist(), index=df1["Frame"].values)
thetashoulderext = pd.Series(thetashoulderext.tolist(), index=df2["Frame"].values)
thetashoulderflex = pd.Series(thetashoulderflex.tolist(), index=df3["Frame"].values)

# np.savetxt(r'python_dl009_shouabdoutput_001sho.csv', thetashoulder, fmt='%.15f')
# np.savetxt(r'python_dl009_shouextoutput_001sho.csv', thetashoulder1, fmt='%.15f')
# np.savetxt(r'python_dl009_shouflexoutput_001sho.csv', thetashoulder2, fmt='%.15f')
# np.savetxt(r'python_dl009_elbflexoutput_001elb.csv', thetaelbow, fmt='%.15f')

# save in csv
thetashoulderabd.to_csv("python_dl009_shouabdoutput_020sho.csv", header=False)
thetaelbowflex.to_csv("python_dl009_elbflexoutput_001elb.csv", header=False)
thetashoulderext.to_csv("python_dl009_shouextoutput_001sho.csv", header=False)
thetashoulderflex.to_csv("python_dl009_shouflexoutput_001sho.csv", header=False)

# plot graph of angle vs frames

plt.figure(figsize=(20, 10))
plt.plot(thetashoulderabd.index, thetashoulderabd.values, color="g")
plt.xlabel("Frame", fontsize=20)
plt.ylabel("Joint Angle (Degrees)", fontsize=20)
plt.title("Shoulder Abduction Angle", fontsize=20)
plt.legend(labels=["Vicon"])
plt.savefig(
    "/workspaces/JointAngleEstimation/Matlab/Plot_vicon_angle_vs_frames/Shoulder_Abduction.png"
)
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(thetaelbowflex.index, thetaelbowflex.values, color="g")
plt.xlabel("Frame", fontsize=20)
plt.ylabel("Joint Angle (Degrees)", fontsize=20)
plt.title("Elbow Flexion Angle", fontsize=20)
plt.legend(labels=["Vicon"])
plt.savefig(
    "/workspaces/JointAngleEstimation/Matlab/Plot_vicon_angle_vs_frames/Elbow_Flexion.png"
)
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(thetashoulderext.index, thetashoulderext.values, color="g")
plt.xlabel("Frame", fontsize=20)
plt.ylabel("Joint Angle (Degrees)", fontsize=20)
plt.title("Shoulder Extension Angle", fontsize=20)
plt.legend(labels=["Vicon"])
plt.savefig(
    "/workspaces/JointAngleEstimation/Matlab/Plot_vicon_angle_vs_frames/Shoulder_Extension.png"
)
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(thetashoulderflex.index, thetashoulderflex.values, color="g")
plt.xlabel("Frame", fontsize=20)
plt.ylabel("Joint Angle (Degrees)", fontsize=20)
plt.title("Shoulder Flexion Angle", fontsize=20)
plt.legend(labels=["Vicon"])
plt.savefig(
    "/workspaces/JointAngleEstimation/Matlab/Plot_vicon_angle_vs_frames/Shoulder_Flexion.png"
)
plt.show()
