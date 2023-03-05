import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Train a Linear Regression Model over all joints:
#   1. Take MOVENET angles CSV and VICON angles CSV for trial 1 of all joints.
#   2. Append all the above rows in X (for MOVENET angles) and Y (for VICON angles).
#   2. Perform 70-30 train-test split, i.e. X_train = random 70% of X, Y_train = random 70% of Y.
#   3. Train Linear Regression model (X_train, Y_train).
# This will give a Linear Regression model which does not overfit to any single joint.
#
# Test this model on other trials of all joints.
# Append MOVENET angles for other 9 trials for all joints to X_test.
# Perform reg.predict(X_test) to get predictions.
# Calculate RMSE, MAE, PearsonR for predictions and Y_test to check post-processing quality.

# We use 70% of hipext_001 for training.
VICON_ANGLES_PREFIX = "/workspaces/JointAngleEstimation/Matlab/vicon_csv_hip/"
MOVENET_ANGLES_PREFIX = "/workspaces/JointAngleEstimation/csv_ouput/"
TRAINING_TRIAL = {
    "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_001.csv",
    "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_001.csv",
}
TEST_TRIALS = [
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_002.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_002.csv",
    },
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_003.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_003.csv",
    },
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_004.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_004.csv",
    },
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_005.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_005.csv",
    },
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_006.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_006.csv",
    },
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_007.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_007.csv",
    },
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_008.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_008.csv",
    },
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_009.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_009.csv",
    },
    {
        "VICON_ANGLES": VICON_ANGLES_PREFIX + "processed_dl1003a_hipext_010.csv",
        "MOVENET_ANGLES": MOVENET_ANGLES_PREFIX + "P_Frames-dl1003a_hipext_010.csv",
    },
]


def read_vicon_angles(vicon_angles_path):
    vicon_output = pd.read_csv(
        vicon_angles_path,
        header=None,
        usecols=[0, 1],
        index_col=0,
    )
    return vicon_output.iloc[:, 0].values


def read_movenet_angles(movenet_angles_path):
    model_output = pd.read_csv(
        movenet_angles_path,
        header=None,
        usecols=[0, 6],
        index_col=0,
        skiprows=1,
    )
    return model_output.values


def rmse(model_angles, vicon_angles):
    return sqrt(
        mean_squared_error(
            model_angles,
            vicon_angles,
        )
    )


Y = read_vicon_angles(vicon_angles_path=TRAINING_TRIAL["VICON_ANGLES"])
X = read_movenet_angles(movenet_angles_path=TRAINING_TRIAL["MOVENET_ANGLES"])

# Randomly split X and Y into training and testing data
# Let us try 70-30 split.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42
)

# How can we use logistic regression here? Because Y is not a categorical variable,
# running this will complain with "ValueError: Unknown label type: 'continuous'"
# clf = LogisticRegression(random_state=0).fit(X, Y)
# print(clf)

# So, we will use linear regression since Y is a continuous variable.
print("-------- training linear regression model --------")
print(
    f"number of angles used for training: movenet = {X_train.shape}, vicon = {Y_train.shape}"
)
reg = LinearRegression().fit(X_train, Y_train)
print("reg.score(X_train, Y_train) is", reg.score(X_train, Y_train))
print("-------- training linear regression model done --------\n")

print("-------- appending data from trials for testing --------")
trial_num = 1
print(
    f"number of angles from trial {trial_num} for testing: movenet = {X_test.shape}, vicon = {X_test.shape}"
)
for trial in TEST_TRIALS:
    trial_num += 1
    trial_vicon_angles = read_vicon_angles(vicon_angles_path=trial["VICON_ANGLES"])
    trial_movenet_angles = read_movenet_angles(
        movenet_angles_path=trial["MOVENET_ANGLES"]
    )
    print(
        f"number of angles from trial {trial_num} for testing: movenet = {trial_movenet_angles.shape}, vicon = {trial_vicon_angles.shape}"
    )
    Y_test = np.append(Y_test, trial_vicon_angles, axis=0)
    X_test = np.append(X_test, trial_movenet_angles, axis=0)
print("-------- appending done --------\n")

print("-------- testing linear regression model --------")
print(
    f"number of angles used for testing: movenet = {X_test.shape}, vicon = {Y_test.shape}"
)
# predictions = reg.predict(X_test)
predictions = reg.predict(trial_movenet_angles)
print(predictions)
print(trial_vicon_angles)
print(
    f"\nBetween movenet & vicon,\n"
    f"  RMSE     = {rmse(model_angles=X_test.flatten(), vicon_angles=Y_test)}\n"
    f"  MAE      = {mean_absolute_error(X_test.flatten(), Y_test)}\n"
    f"  PearsonR = {pearsonr(X_test.flatten(), Y_test)}"
)
print(
    f"\nBetween movenet+regression & vicon,\n"
    # f"  RMSE     = {rmse(model_angles=predictions.flatten(), vicon_angles=Y_test)}\n"
    f"  RMSE     = {rmse(model_angles=predictions, vicon_angles=trial_vicon_angles)}\n"
    #     f"  MAE      = {mean_absolute_error(predictions.flatten(), Y_test)}\n"
    #     f"  PearsonR = {pearsonr(predictions.flatten(), Y_test)}"
)
# print("-------- testing linear regression model done --------")


# score = r2_score(data["Actual Value"], data["Preds"])
# print("The accuracy of our model is {}%".format(round(score, 2) *100))    Pearson (r) ≠ r2
