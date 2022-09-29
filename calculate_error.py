import os
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


WINDOW_WIDTH = 10

# Smoothing
def apply_smoothing(angles, window_width):
    cumsum_vec = np.cumsum(np.insert(angles, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec


def get_joint_name(movement, group):
    joint_name = ""
    if "right" in group.lower():
        joint_name = "right_"
        if "hip" in movement.lower():
            joint_name += "hip"
        elif "knee" in movement.lower():
            joint_name += "knee"
        elif "shoulder" in movement.lower():
            joint_name += "shoulder"
        elif "elbow" in movement.lower():
            joint_name += "elbow"
        else:
            raise ValueError(f"unknown movement {movement}")
    elif "left" in group.lower():
        joint_name = "left_"
        if "hip" in movement.lower():
            joint_name += "hip"
        elif "knee" in movement.lower():
            joint_name += "knee"
        elif "shoulder" in movement.lower():
            joint_name += "shoulder"
        elif "elbow" in movement.lower():
            joint_name += "elbow"
        else:
            raise ValueError(f"unknown movement {movement}")
    else:
        raise ValueError(f"unknown group {group}")
    return joint_name


plt.rcParams["figure.figsize"] = [20.00, 25.50]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({"font.size": 18})

ax = None
colors = "bgrcmykw"

if __name__ == "__main__":
    data_path = "./data/"
    movement_groups = next(os.walk(data_path))[1]
    for group in movement_groups:
        movement_group_path = data_path + group + "/"
        movements = next(os.walk(movement_group_path))[1]
        for movement in movements:
            movement_path = movement_group_path + movement + "/"
            print(f"Evaluating error for movement in {movement_path}")
            model_output_path = movement_path + "model/"
            vicon_output_path = movement_path + "vicon/"
            joint_name = get_joint_name(movement, group)
            # trial_name is like "dl1003a_hipabd_001"
            trial_names = [
                filename.strip(".csv")
                for filename in os.listdir(movement_path)
                if ".csv" in filename
            ]
            for trial_num, trial_name in enumerate(trial_names):
                print(f"\n===== Trial {trial_name} =====")
                # This trial_name should be present in both model and vicon sub-dirs.
                model_csv = model_output_path + f"{trial_name}.csv"
                vicon_csv = vicon_output_path + f"{trial_name}.csv"
                # Frame number for vicon csv files start from 1, and frame number for
                # model csv files start from 0. We need to make adjustments. Frame 0 of
                # model refers to frame 1 of vicon.
                model_df = pd.read_csv(model_csv, index_col=0)
                model_df.index = pd.Index(range(1, len(model_df.index) + 1))
                vicon_df = pd.read_csv(
                    vicon_csv, header=None, index_col=0, names=[joint_name]
                )

                # If frame number x has NaN for model, then we should drop frame number
                # x from both model and vicon dataframes.
                joint_angles_from_model = model_df[joint_name]
                joint_angles_from_model.dropna(
                    inplace=True
                )  # drops rows (frames) with NaN in joint_name column from model dataframe

                # pick index (frames or rows) without NaN in model AND present in vicon csv
                frames_to_consider = joint_angles_from_model.index.intersection(
                    vicon_df.index
                )

                ytrue = vicon_df[joint_name][frames_to_consider].values
                ypred = joint_angles_from_model[frames_to_consider].values

                # calculate rmse
                rmse = sqrt(mean_squared_error(y_true=ytrue, y_pred=ypred))
                print(
                    f"RMSE for trial {trial_name}, movement {movement}, group {group} is {round(rmse, 2)}"
                )

                # calculate mae
                mae = mean_absolute_error(y_true=ytrue, y_pred=ypred)
                print(
                    f"MAE for trial {trial_name}, movement {movement}, group {group} is {round(mae, 2)}"
                )

                # calculate r2 score
                r2 = r2_score(y_true=ytrue, y_pred=ypred)
                print(
                    f"Coefficient of determination for trial {trial_name}, movement {movement}, group {group} is {round(r2, 2)}"
                )

                r = pearsonr(ytrue, ypred)
                print(
                    f"Coefficient of determination for trial {trial_name}, movement {movement}, group {group} is {r}"
                )

                # plot only for trial_num 0, 1, 2, ..., 7.
                if trial_num >= 8:
                    continue

                y_smooth = apply_smoothing(angles=ypred, window_width=WINDOW_WIDTH)
                y_smooth_df = pd.DataFrame(
                    {f"model angle (after smoothing) ({trial_name})": y_smooth.tolist()}
                )
                y_smooth_df.index = frames_to_consider[WINDOW_WIDTH - 1 :].values
                y_smooth_df.index.names = ["frame number"]

                y_true_df = pd.DataFrame(
                    {f"vicon angle ({trial_name})": ytrue[WINDOW_WIDTH - 1 :].tolist()}
                )
                y_true_df.index = frames_to_consider[WINDOW_WIDTH - 1 :].values
                y_true_df.index.names = ["frame number"]

                if ax is None:
                    # for the first plot
                    ax = y_true_df.plot(figsize=(34, 14), linestyle="dashed", color="r")
                else:
                    # for second and further plots
                    y_true_df.plot(ax=ax, linestyle="dashed", color=colors[trial_num])

                y_smooth_df.plot(ax=ax, color=colors[trial_num], linewidth=5)

                ax.legend()

            plt.title(f"{movement} angle")
            plt.ylabel("Joint Angle (Degrees)")
            plt.savefig(movement_path + joint_name)
            plt.show()
