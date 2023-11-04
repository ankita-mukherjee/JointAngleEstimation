"""
python3 calculate_error.py

The following --with-regression flag will apply regression.
python3 calculate_error.py --with-regression

The following --apply-trimming flag will apply vicon trimming.
python3 calculate_error.py --apply-trimming
"""
import fire
import os
import random
import shutil
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from definitions import ROOT_DIR
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


WINDOW_WIDTH = 10
GEN_PATH = f"{ROOT_DIR}/gen"
DATA_PATH = f"{ROOT_DIR}/data"
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


def get_stats(ytrue, ypred):
    # calculate rmse
    rmse = sqrt(mean_squared_error(y_true=ytrue, y_pred=ypred))

    # calculate mae
    mae = mean_absolute_error(y_true=ytrue, y_pred=ypred)

    # calculate r2 score
    corr_matrix = np.corrcoef(ytrue, ypred)
    corr = corr_matrix[0, 1]
    r2 = corr**2

    # # calculate r score
    # r = pearsonr(ytrue, ypred)
    # print(
    #     f"Pearson Coefficient {trial_name}, movement {movement} is {round(r[0], 3)}"
    # )
    # # print("\n")

    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 2)}


def rmse_summary(data):
    mean = np.mean(data)
    minimum = np.min(data)
    maximum = np.max(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    variance = np.var(data)
    std_deviation = np.std(data)

    print(f"")
    print(f"=== RMSE Quick Stats Begin ===")
    print(f"Mean: {mean}")
    print(f"Minimum: {minimum}")
    print(f"Maximum: {maximum}")
    print(f"1st Quartile (Q1): {q1}")
    print(f"3rd Quartile (Q3): {q3}")
    print(f"Variance: {variance}")
    print(f"Standard Deviation: {std_deviation}")
    print(f"=== RMSE Quick Stats End ===")
    print(f"")


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
                # print(
                #     f"Skipping {participant_movement} as model directory does not exist."
                # )
                continue
            if not os.path.exists(f"{participant_movement_path}/vicon/"):
                # print(
                #     f"Skipping {participant_movement} as vicon directory does not exist."
                # )
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


def apply_vicon_trimming():
    movements = next(os.walk(GEN_PATH))[1]

    for movement in movements:
        assert movement in MOVEMENTS
        movement_path = f"{GEN_PATH}/{movement}/"
        model_output_path = movement_path + "model/"
        vicon_output_path = movement_path + "vicon/"
        assert len(os.listdir(model_output_path)) == len(os.listdir(vicon_output_path))
        trial_names = [
            filename.strip(".csv")
            for filename in os.listdir(model_output_path)
            if ".csv" in filename
        ]
        trial_names.sort()
        for trial_name in trial_names:
            vicon_csv = vicon_output_path + f"{trial_name}.csv"
            vicon_df = pd.read_csv(vicon_csv, header=None, index_col=0, names=["angle"])
            model_csv = model_output_path + f"{trial_name}.csv"
            model_df = pd.read_csv(model_csv, index_col=0)
            model_df.reset_index(inplace=True, drop=True)

            if vicon_df.shape[0] / model_df.shape[0] >= 2.0:
                vicon_df = vicon_df.iloc[::2]
                vicon_df.reset_index(inplace=True, drop=True)

            lower_threshold = vicon_df.angle.quantile(0.10)
            upper_threshold = vicon_df.angle.quantile(0.90)
            if movement in ("shoabd", "shoflex", "shoext", "hipflex"):
                vicon_df = vicon_df[vicon_df.angle >= lower_threshold]
            elif movement in ("elbflex", "kneeflex", "hipabd", "hipext"):
                vicon_df = vicon_df[vicon_df.angle <= upper_threshold]

            vicon_df.to_csv(vicon_csv, header=False)


# Apply bounds analysis using +/- 2 degrees of last value.
# Assumption: joint angles do not change abruptly.
def apply_bounds_analysis(model_csv: str, movement: str):
    model_df = pd.read_csv(model_csv, index_col=0)

    # Apply bounds analysis only to this column.
    joint_col = [col for col in model_df.columns if movement[:3] in col][0]

    last_val = None
    last_val_row = None
    for index, row in model_df.iterrows():
        current_val = row[joint_col]
        if not pd.isna(current_val):
            if last_val is not None:
                lower_bound = last_val - 5 * (index - last_val_row)
                upper_bound = last_val + 5 * (index - last_val_row)
                if current_val < lower_bound:
                    model_df.at[index, joint_col] = lower_bound
                elif current_val > upper_bound:
                    model_df.at[index, joint_col] = upper_bound
        else:
            last_val = current_val
            last_val_row = index

    # trial_name = model_csv.split('/')[-1]
    # if trial_name == "dl202_left_upper_elbflex_001.csv":
    #     model_df.to_csv("CHANGED_dl202_left_upper_elbflex_001.csv")

    return model_df


def get_filtered_trials(movement):
    assert movement in MOVEMENTS
    movement_path = f"{GEN_PATH}/{movement}/"
    model_output_path = movement_path + "model/"
    vicon_output_path = movement_path + "vicon/"
    assert len(os.listdir(model_output_path)) == len(os.listdir(vicon_output_path))
    trial_names = [
        filename.strip(".csv")
        for filename in os.listdir(model_output_path)
        if ".csv" in filename
    ]
    trial_names.sort()
    filtered_trial_names = []
    for trial_name in trial_names:
        # This trial_name should be present in both model and vicon sub-dirs.
        model_csv = model_output_path + f"{trial_name}.csv"
        vicon_csv = vicon_output_path + f"{trial_name}.csv"

        # Apply Bounds Analysis on model angles
        model_df = apply_bounds_analysis(model_csv, movement)

        # Frame number for vicon csv files start from 1, and frame number for
        # model csv files start from 0. We need to make adjustments. Frame 0 of
        # model refers to frame 1 of vicon.
        # model_df.index = pd.Index(range(1, len(model_df.index) + 1))
        model_df.reset_index(inplace=True, drop=True)
        # print(f"Model -------{trial_name}-{len(model_df.index)}")
        # If frame number x has NaN for model, then we should drop frame number
        # x from both model and vicon dataframes.
        joint_names = [col for col in model_df.columns if movement[:3] in col]
        if not joint_names:
            print(f"Skipping trial {trial_name} due to no {movement[:3]} columns.")
            continue
        joint_name = joint_names[0]
        joint_angles_from_model = model_df[joint_name]
        joint_angles_from_model.dropna(
            inplace=True
        )  # drops rows (frames) with NaN in joint_name column from model dataframe
        # pick index (frames or rows) without NaN in model AND present in vicon csv

        vicon_df = pd.read_csv(vicon_csv, header=None, index_col=0, names=[joint_name])
        # print(f"Vicon -------{trial_name}-{len(vicon_df.index)}")
        # Difference = len(vicon_df.index) / len(model_df.index)
        # print(f"Difference between Vicon-Model -------{trial_name}- {Difference}")
        if vicon_df.shape[0] / model_df.shape[0] >= 2.0:
            vicon_df = vicon_df.iloc[::2]
            vicon_df.reset_index(inplace=True, drop=True)

        vicon_df.dropna(inplace=True)

        if vicon_df.empty:
            # print(f"Trial {trial_name} has empty vicon data. Please check. Skipping.")
            continue

        frames_to_consider = joint_angles_from_model.index.intersection(vicon_df.index)

        ytrue = vicon_df[joint_name][frames_to_consider].values
        # print(len(joint_angles_from_model.index))
        ypred = joint_angles_from_model[frames_to_consider].values
        metrics = get_stats(ytrue=ytrue, ypred=ypred)
        # RMSE >= 25 or MAE >= 20 or R2 >= 0.20
        if not (metrics["RMSE"] >= 21 or metrics["MAE"] >= 20 or metrics["R2"] <= 0.20):
            filtered_trial_names.append(trial_name)
    return filtered_trial_names


def run(model_type="linear", with_regression=False, apply_trimming=False, alpha=0.5):
    random_seed = 42
    np.random.seed(random_seed)

    plt.rcParams["figure.figsize"] = [20.00, 25.50]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams.update({"font.size": 18})
    ax = None
    colors = "bgrcmykw"
    collect_model_vicon_csvs()
    if apply_trimming:
        apply_vicon_trimming()
    movements = next(os.walk(GEN_PATH))[1]
    for movement in movements:
        assert movement in MOVEMENTS
        movement_path = f"{GEN_PATH}/{movement}/"
        print(f"Evaluating error for {movement}")
        model_output_path = movement_path + "model/"
        vicon_output_path = movement_path + "vicon/"
        filtered_trial_names = get_filtered_trials(movement=movement)
        num_trials = len(filtered_trial_names)

        if with_regression:
            random.seed(random_seed)
            random.shuffle(filtered_trial_names)
            num_train_trials = round(
                0.80 * num_trials
            )  # assuming 80-20 split for train and test
            train_trials = filtered_trial_names[:num_train_trials]
            test_trials = filtered_trial_names[num_train_trials:]
            train_trials.sort()
            test_trials.sort()
            xtrain, ytrain = [], []
            reg = None
            mlp_reg = None
            ridge_reg = None
            svm_reg = None
            filtered_trial_names = train_trials + test_trials

        summary = {"AvgRMSE": 0, "AvgMAE": 0, "AvgR2": 0}
        max_rmse = 0
        movement_rmse_data = []
        ytrue_list = []
        ypred_list = []
        for trial_num, trial_name in enumerate(filtered_trial_names):
            # print(f"\n===== Trial {trial_name} =====")
            # This trial_name should be present in both model and vicon sub-dirs.
            model_csv = model_output_path + f"{trial_name}.csv"
            vicon_csv = vicon_output_path + f"{trial_name}.csv"
            # Frame number for vicon csv files start from 1, and frame number for
            # model csv files start from 0. We need to make adjustments. Frame 0 of
            # model refers to frame 1 of vicon.
            model_df = apply_bounds_analysis(model_csv, movement)
            # model_df.index = pd.Index(range(1, len(model_df.index) + 1))
            model_df.reset_index(inplace=True, drop=True)
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

            if vicon_df.shape[0] / model_df.shape[0] >= 2.0:
                vicon_df = vicon_df.iloc[::2]
                vicon_df.reset_index(inplace=True, drop=True)

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

            metrics = None
            ypred_reg = None
            if with_regression:
                if model_type == "linear":
                    if trial_name in train_trials:
                        ytrain = np.append(ytrain, ytrue)
                        xtrain = np.append(xtrain, ypred)
                    else:
                        assert trial_name in test_trials
                        assert trial_num >= num_train_trials
                        if reg is None:
                            # Train a linear regression model for post-processing.
                            reg = LinearRegression().fit(xtrain.reshape(-1, 1), ytrain)
                            print(
                                f"\nLinear Regression score after training on {num_train_trials} trials from {movement} is {reg.score(xtrain.reshape(-1, 1), ytrain)}.\n\tTotal trials = {num_trials}.\n\tTraining trials = {len(train_trials)}.\n\tTesting trials = {len(test_trials)}.\n\treg.intercept_ = {reg.intercept_}.\n\treg.coef_ = {reg.coef_[0]}.\n"
                            )
                        ypred_reg = reg.predict(ypred.reshape(-1, 1))
                        metrics = get_stats(ytrue=ytrue, ypred=ypred_reg)

                elif model_type == "non-linear":
                    if trial_name in train_trials:
                        ytrain = np.append(ytrain, ytrue)
                        xtrain = np.append(xtrain, ypred)
                    else:
                        assert trial_name in test_trials
                        assert trial_num >= num_train_trials
                        if reg is None:
                            # Train a linear regression model with polynomial features for post-processing.
                            degree = 3
                            reg = make_pipeline(
                                PolynomialFeatures(degree), LinearRegression()
                            )
                            reg.fit(xtrain.reshape(-1, 1), ytrain)
                            print(
                                f"\nLinear Regression with PolynomialFeatures(degree={degree}) score after training on {num_train_trials} trials from {movement} is {reg.score(xtrain.reshape(-1, 1), ytrain)}.\n\tTotal trials = {num_trials}.\n\tTraining trials = {len(train_trials)}.\n\tTesting trials = {len(test_trials)}.\n"
                            )
                        ypred_reg = reg.predict(ypred.reshape(-1, 1))
                        metrics = get_stats(ytrue=ytrue, ypred=ypred_reg)

                elif model_type == "mlp":
                    if trial_name in train_trials:
                        ytrain = np.append(ytrain, ytrue)
                        xtrain = np.append(xtrain, ypred)
                    else:
                        assert trial_name in test_trials
                        assert trial_num >= num_train_trials
                        if mlp_reg is None:
                            # Create and train an MLPRegressor model
                            mlp_reg = MLPRegressor(
                                hidden_layer_sizes=(32, 64, 128),
                                activation="relu",
                                solver="adam",
                                alpha=0.01,
                                batch_size="auto",
                                learning_rate="invscaling",
                                learning_rate_init=0.01,
                                random_state=42,
                                max_iter=50,  # Increased number of iterations
                                verbose=True,
                            )
                            mlp_reg.fit(xtrain.reshape(-1, 1), ytrain)
                            # Access the weights for each layer
                            # weights = mlp_reg.coefs_
                            # for i, layer_weights in enumerate(weights):
                            #     print(f"Layer {i} weights:\n{layer_weights}")
                            print(
                                f"\nMLP Regression score after training on {num_train_trials} trials from {movement} is {mlp_reg.score(xtrain.reshape(-1, 1), ytrain)}.\n\tTotal trials = {num_trials}.\n\tTraining trials = {len(train_trials)}.\n\tTesting trials = {len(test_trials)}.\n"
                            )

                        # Predict using the trained MLP model
                        ypred_reg = mlp_reg.predict(ypred.reshape(-1, 1))
                        metrics = get_stats(ytrue=ytrue, ypred=ypred_reg)

                elif model_type == "ridge":  # Handle Ridge Regression
                    if trial_name in train_trials:
                        ytrain = np.append(ytrain, ytrue)
                        xtrain = np.append(xtrain, ypred)
                    else:
                        assert trial_name in test_trials
                        assert trial_num >= num_train_trials
                        if ridge_reg is None:
                            ridge_reg = Ridge(
                                alpha=alpha, random_state=random_seed
                            )  # Create Ridge Regression model
                            print(f"value of alpha: {alpha}")
                            ridge_reg.fit(xtrain.reshape(-1, 1), ytrain)
                            print(
                                f"\nRidge Regression score after training on {num_train_trials} trials from {movement} is {ridge_reg.score(xtrain.reshape(-1, 1), ytrain)}.\n\tTotal trials = {num_trials}.\n\tTraining trials = {len(train_trials)}.\n\tTesting trials = {len(test_trials)}.\n"
                            )
                        ypred_reg = ridge_reg.predict(ypred.reshape(-1, 1))
                        metrics = get_stats(ytrue=ytrue, ypred=ypred_reg)

                elif model_type == "svm":  # Handle SVM Regression
                    if trial_name in train_trials:
                        ytrain = np.append(ytrain, ytrue)
                        xtrain = np.append(xtrain, ypred)
                    else:
                        assert trial_name in test_trials
                        assert trial_num >= num_train_trials
                        if svm_reg is None:
                            svm_reg = make_pipeline(
                                StandardScaler(),
                                SVR(
                                    kernel="rbf",
                                    degree=12,
                                    gamma="scale",
                                    coef0=0.0,
                                    tol=0.001,
                                    C=2,
                                    epsilon=0.2,
                                    shrinking=True,
                                    cache_size=200,
                                    verbose=True,
                                    max_iter=-1,
                                ),
                            )

                            svm_reg.fit(xtrain.reshape(-1, 1), ytrain)
                            print(
                                f"\nSVM Regression score after training on {num_train_trials} trials from {movement} is {svm_reg.score(xtrain.reshape(-1, 1), ytrain)}.\n\tTotal trials = {num_trials}.\n\tTraining trials = {len(train_trials)}.\n\tTesting trials = {len(test_trials)}.\n"
                            )
                        ypred_reg = svm_reg.predict(ypred.reshape(-1, 1))
                        metrics = get_stats(ytrue=ytrue, ypred=ypred_reg)

                elif model_type == "grid_search":
                    if trial_name in train_trials:
                        ytrain = np.append(ytrain, ytrue)
                        xtrain = np.append(xtrain, ypred)
                    else:
                        assert trial_name in test_trials
                        assert trial_num >= num_train_trials
                        if reg is None:
                            # Define the hyperparameter grid for SVR
                            param_grid = {
                                "C": [1],
                                "kernel": ["linear", "rbf"],
                                "degree": [3],
                            }
                            # Create a GridSearchCV object
                            grid_search = RandomizedSearchCV(
                                SVR(),
                                param_grid,
                                cv=2,
                                scoring="neg_mean_squared_error",
                                verbose=True,
                                n_jobs=-1,
                            )

                            # Fit the GridSearchCV object to your training data
                            grid_search.fit(xtrain.reshape(-1, 1), ytrain)

                            # Get the best hyperparameters and best model
                            # best_params = grid_search.best_params_
                            reg = grid_search

                            # print(f"Best hyperparameters: {best_params}")

                        ypred_reg = reg.predict(ypred.reshape(-1, 1))
                        print(
                            f"\nSVR with best hyperparameters score after training on {num_train_trials} trials from {movement} is {reg.score(xtrain.reshape(-1, 1), ytrain)}.\n\tTotal trials = {num_trials}.\n\tTraining trials = {len(train_trials)}.\n\tTesting trials = {len(test_trials)}.\n"
                        )
                        metrics = get_stats(ytrue=ytrue, ypred=ypred_reg)

                else:
                    raise ValueError("Invalid model type")

            ytrue_list.append(ytrue)
            ypred_list.append(ypred)

            if metrics is not None:
                summary["AvgRMSE"] += metrics["RMSE"]
                summary["AvgMAE"] += metrics["MAE"]
                summary["AvgR2"] += metrics["R2"]
                # print(
                #     f"RMSE for trial {trial_name}, movement {movement} is {metrics['RMSE']}\n"
                #     f"MAE for trial {trial_name}, movement {movement} is {metrics['MAE']}\n"
                #     f"R2 of trial {trial_name}, movement {movement} is {metrics['R2']}"
                # )
                if metrics["RMSE"] > max_rmse:
                    max_rmse = metrics["RMSE"]
                    max_rmse_trial = trial_name
                movement_rmse_data.append(metrics["RMSE"])

            # plot only for trial_num 0, 1, 2, ..., 7.
            if trial_num >= 3:
                continue

            y_smooth = apply_smoothing(angles=ypred, window_width=WINDOW_WIDTH)
            y_smooth_df = pd.DataFrame(
                {f"model angle (after smoothing) ({trial_name})": y_smooth.tolist()}
            )

            y_smooth_df.index = frames_to_consider[WINDOW_WIDTH - 1 :].values
            # y_smooth_df.to_csv('sample_model.csv')
            y_smooth_df.index.names = ["frame number"]

            y_true_df = pd.DataFrame(
                {f"vicon angle ({trial_name})": ytrue[WINDOW_WIDTH - 1 :].tolist()}
            )

            y_true_df.index = frames_to_consider[WINDOW_WIDTH - 1 :].values
            # y_true_df.to_csv('sample_vicon.csv')
            y_true_df.index.names = ["frame number"]

            if ax is None:
                # for the first plot
                ax = y_true_df.plot(figsize=(34, 14), linestyle="dashed", color="r")
            else:
                # for second and further plots
                y_true_df.plot(ax=ax, linestyle="dashed", color=colors[trial_num])

            y_smooth_df.plot(ax=ax, color=colors[trial_num], linewidth=5)

            ax.legend(loc="lower right")

        if movement in ["elbflex", "shoext"]:
            save_dir = f"plots/{movement}"
            train_save_dir = f"{save_dir}/train"
            test_save_dir = f"{save_dir}/test"
            os.makedirs(train_save_dir, exist_ok=True)
            os.makedirs(test_save_dir, exist_ok=True)
            for trial_name, ytrue, ypred in zip(
                filtered_trial_names, ytrue_list, ypred_list
            ):
                ypred_reg = reg.predict(ypred.reshape(-1, 1))
                metrics = get_stats(ytrue=ytrue, ypred=ypred_reg)
                df = pd.DataFrame(
                    {"vicon": ytrue, "model": ypred, "model+reg": ypred_reg}
                )

                plt.figure(figsize=(8, 6))  # Optional: Set the figure size
                plt.plot(
                    df["vicon"],
                    df["model"],
                    label="model",
                    marker="o",
                    linestyle="-",
                    color="blue",
                )
                plt.plot(
                    df["vicon"],
                    df["model+reg"],
                    label="model+reg",
                    marker="x",
                    linestyle="--",
                    color="red",
                )

                # Add the y = x line
                plt.plot(
                    df["vicon"],
                    df["vicon"],
                    label="vicon=vicon",
                    linestyle="-",
                    color="green",
                )

                plt.xlabel("vicon-axis")
                plt.ylabel("model-axis")
                plt.title(f"model and model+reg vs vicon, rmse: {metrics['RMSE']}")
                plt.legend()
                plt.grid(True)  # Optional: Display grid lines

                # Save the plot as an image file (e.g., PNG) in the specified directory
                if trial_name in train_trials:
                    save_path = os.path.join(train_save_dir, f"plot_{trial_name}.png")
                else:
                    save_path = os.path.join(test_save_dir, f"plot_{trial_name}.png")

                plt.savefig(save_path)
                plt.close()  # Close the current plot to free up memory

        if test_trials:
            summary["AvgRMSE"] = round(summary["AvgRMSE"] / len(test_trials), 2)
            summary["AvgMAE"] = round(summary["AvgMAE"] / len(test_trials), 2)
            summary["AvgR2"] = round(summary["AvgR2"] / len(test_trials), 2)
            print(f"Summary metrics for {movement}: {summary}")
            print(f"Worst RMSE: {max_rmse}, trial: {max_rmse_trial}")
            rmse_summary(movement_rmse_data)

        plt.title(f"{movement} angle")
        plt.ylabel("Joint Angle (Degrees)")
        plt.savefig(movement_path + movement)
        # plt.show()


if __name__ == "__main__":
    fire.Fire(run)
