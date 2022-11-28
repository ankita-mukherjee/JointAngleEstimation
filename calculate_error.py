"""
python3 calculate_error.py
python3 calculate_error.py --with-regression
"""
import fire
import os
import shutil
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


WINDOW_WIDTH = 10
GEN_PATH = "./gen"
DATA_PATH = "./data"
MOVEMENTS = (
    "elbflex",
    "shoabd",
    "shoext",
    "shoflex",
    "hipabd",
    "hipext",
    "hipflex",
    "kneeflex",
)

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
        elif "sho" in movement.lower():
            joint_name += "shoulder"
        elif "elb" in movement.lower():
            joint_name += "elbow"
        else:
            raise ValueError(f"unknown movement {movement}")
    elif "left" in group.lower():
        joint_name = "left_"
        if "hip" in movement.lower():
            joint_name += "hip"
        elif "knee" in movement.lower():
            joint_name += "knee"
        elif "sho" in movement.lower():
            joint_name += "shoulder"
        elif "elb" in movement.lower():
            joint_name += "elbow"
        else:
            raise ValueError(f"unknown movement {movement}")
    else:
        raise ValueError(f"unknown group {group}")
    return joint_name


def get_stats(ytrue, ypred, trial_name, movement):
    # print(f"Statistics for trial {trial_name}, movement {movement}")
    # calculate rmse
    rmse = sqrt(mean_squared_error(y_true=ytrue, y_pred=ypred))
    print(f"RMSE for trial {trial_name}, movement {movement} is {round(rmse, 2)}")

    #  # calculate mae
    # mae = mean_absolute_error(y_true=ytrue, y_pred=ypred)
    # print(
    #     f"MAE for trial {trial_name}, movement {movement} is {round(mae, 2)}"
    # )

    # # calculate r2 score
    # corr_matrix = np.corrcoef(y_true=ytrue, y_pred=ypred)
    # corr = corr_matrix[0, 1]
    # R_sq = corr**2
    # print(
    #     f"Coefficient of determination for trial {trial_name}, movement {movement} is {round(R_sq, 2)}"
    # )

    # # calculate r score
    # r = pearsonr(ytrue, ypred)
    # print(
    #     f"Pearson Coefficient {trial_name}, movement {movement} is {round(r[0], 3)}"
    # )
    # # print("\n")


def collect_model_vicon_csvs():
    """
    Collects model csvs and vicon csvs and puts them
    inside gen/{movement}/model/ or gen/{movement}/vicon/
    """
    for movement in MOVEMENTS:
        os.makedirs(f"{GEN_PATH}/{movement}/model/", exist_ok=True)
        os.makedirs(f"{GEN_PATH}/{movement}/vicon/", exist_ok=True)

    movement_groups = next(os.walk(DATA_PATH))[1]
    for group in movement_groups:
        movement_group_path = (
            f"{DATA_PATH}/{group}/"  # movement_group_path = "./data/left_upper"
        )
        participant_movements = next(os.walk(movement_group_path))[1]
        for participant_movement in participant_movements:
            # participant_movement = "./data/left_upper/dl200_left_upper_elbflex/"
            participant_movement_path = f"{movement_group_path}/{participant_movement}"
            movement = [m for m in MOVEMENTS if m in participant_movement][0]
            if not os.path.exists(f"{participant_movement_path}/model/"):
                print(
                    f"Skipping {participant_movement} as model directory does not exist."
                )
                continue
            if not os.path.exists(f"{participant_movement_path}/vicon/"):
                print(
                    f"Skipping {participant_movement} as vicon directory does not exist."
                )
                continue
            model_csv_files = os.listdir(f"{participant_movement_path}/model/")
            vicon_csv_files = os.listdir(f"{participant_movement_path}/vicon/")
            common_csv_files = list(set(model_csv_files) & set(vicon_csv_files))
            for csv_file in common_csv_files:
                model_csv_file = os.path.join(
                    f"{participant_movement_path}/model/{csv_file}"
                )
                vicon_csv_file = os.path.join(
                    f"{participant_movement_path}/vicon/{csv_file}"
                )
                shutil.copy(model_csv_file, f"{GEN_PATH}/{movement}/model/{csv_file}")
                shutil.copy(vicon_csv_file, f"{GEN_PATH}/{movement}/vicon/{csv_file}")


def run(with_regression=False):
    plt.rcParams["figure.figsize"] = [20.00, 25.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({"font.size": 18})
    ax = None
    colors = "bgrcmykw"
    collect_model_vicon_csvs()
    movements = next(os.walk(GEN_PATH))[1]
    for movement in movements:
        assert movement in MOVEMENTS
        movement_path = f"{GEN_PATH}/{movement}/"
        print(f"Evaluating error for {movement}")
        model_output_path = movement_path + "model/"
        vicon_output_path = movement_path + "vicon/"
        assert len(os.listdir(model_output_path)) == len(os.listdir(vicon_output_path))
        trial_names = [
            filename.strip(".csv")
            for filename in os.listdir(model_output_path)
            if ".csv" in filename
        ]
        trial_names.sort()
        num_trials = len(trial_names)
        num_train_trials = round(
            0.80 * num_trials
        )  # assuming 80-20 split for train and test
        xtrain, ytrain = [], []
        reg = None
        for trial_num, trial_name in enumerate(trial_names):
            # print(f"\n===== Trial {trial_name} =====")
            # This trial_name should be present in both model and vicon sub-dirs.
            model_csv = model_output_path + f"{trial_name}.csv"
            vicon_csv = vicon_output_path + f"{trial_name}.csv"
            # Frame number for vicon csv files start from 1, and frame number for
            # model csv files start from 0. We need to make adjustments. Frame 0 of
            # model refers to frame 1 of vicon.
            model_df = pd.read_csv(model_csv, index_col=0)
            model_df.index = pd.Index(range(1, len(model_df.index) + 1))

            # If frame number x has NaN for model, then we should drop frame number
            # x from both model and vicon dataframes.
            joint_name = [col for col in model_df.columns if movement[:3] in col][0]
            joint_angles_from_model = model_df[joint_name]
            joint_angles_from_model.dropna(
                inplace=True
            )  # drops rows (frames) with NaN in joint_name column from model dataframe

            # pick index (frames or rows) without NaN in model AND present in vicon csv
            vicon_df = pd.read_csv(
                vicon_csv, header=None, index_col=0, names=[joint_name]
            )
            vicon_df.dropna(inplace=True)

            if vicon_df.empty:
                print(
                    f"Trial {trial_name} has empty vicon data. Please check. Skipping."
                )
                continue

            frames_to_consider = joint_angles_from_model.index.intersection(
                vicon_df.index
            )

            ytrue = vicon_df[joint_name][frames_to_consider].values
            ypred = joint_angles_from_model[frames_to_consider].values

            if with_regression:
                if trial_num < num_train_trials:
                    ytrain = np.append(ytrain, ytrue)
                    xtrain = np.append(xtrain, ypred)
                else:
                    if reg is None:
                        # Train a linear regression model for post-processing.
                        reg = LinearRegression().fit(xtrain.reshape(-1, 1), ytrain)
                        print(
                            f"Linear Regression score for training {num_train_trials} trials from {movement} is {reg.score(xtrain.reshape(-1, 1), ytrain)}\n"
                        )
                    ypred_reg = reg.predict(ypred.reshape(-1, 1))
                    get_stats(
                        ytrue=ytrue,
                        ypred=ypred_reg,
                        trial_name=trial_name,
                        movement=movement,
                    )
            else:
                get_stats(
                    ytrue=ytrue, ypred=ypred, trial_name=trial_name, movement=movement
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


if __name__ == "__main__":
    fire.Fire(run)
