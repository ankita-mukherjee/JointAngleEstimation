"""
This file calculates joint angles from the vicon csv file.
"""
import pandas as pd
import numpy as np
import math
import glob
import os

# Calculates hip abduction, hip extension and hip flexion joint angles.
def calculate_hip_angles(path_hip):
    all_files_hip = glob.glob(path_hip + "*.csv")
    for trial_hip in all_files_hip:
        df1 = pd.read_csv(
            trial_hip,
            skiprows=5,
            header=None,
            names=[
                "Frame",
                "SubFrame",
                "ASI_X_mm",
                "ASI_Y_mm",
                "ASI_Z_mm",
                "SHO_X_mm",
                "SHO_Y_mm",
                "SHO_Z_mm",
                "KNEE_X_mm",
                "KNEE_Y_mm",
                "KNEE_Z_mm",
                "ANK_X_mm",
                "ANK_Y_mm",
                "ANK_Z_mm",
            ],
        )

        shohip_length = np.sqrt(
            (df1["SHO_X_mm"] - df1["ASI_X_mm"]) ** 2
            + (df1["SHO_Y_mm"] - df1["ASI_Y_mm"]) ** 2
            + (df1["SHO_Z_mm"] - df1["ASI_Z_mm"]) ** 2
        )
        hipknee_length = np.sqrt(
            (df1["ASI_X_mm"] - df1["KNEE_X_mm"]) ** 2
            + (df1["ASI_Y_mm"] - df1["KNEE_Y_mm"]) ** 2
            + (df1["ASI_Z_mm"] - df1["KNEE_Z_mm"]) ** 2
        )
        shoknee_length = np.sqrt(
            (df1["SHO_X_mm"] - df1["KNEE_X_mm"]) ** 2
            + (df1["SHO_Y_mm"] - df1["KNEE_Y_mm"]) ** 2
            + (df1["SHO_Z_mm"] - df1["KNEE_Z_mm"]) ** 2
        )

        cASIa = (shohip_length**2 + hipknee_length**2 - shoknee_length**2) / (
            2 * shohip_length * hipknee_length
        )
        hip_angle_rad = np.arccos(cASIa)
        theta_hip_angle = hip_angle_rad * 180 / math.pi
        theta_hip_angle = pd.Series(theta_hip_angle.tolist(), index=df1["Frame"].values)
        # Example: "./data/dl103-Right-Lower-obj1/hipabd/vicon/"
        processed_folder_path = os.path.split(trial_hip)[0] + "/vicon/"
        os.makedirs(processed_folder_path, exist_ok=True)
        csv_file_path = processed_folder_path + os.path.split(trial_hip)[1]
        theta_hip_angle.to_csv(csv_file_path, header=False)
        print(f"Generated hip joint angles from vicon data to {csv_file_path}")


# Calculates knee flexion joint angle.
def calculate_knee_angles(path_knee):
    all_files_knee = glob.glob(path_knee + "*.csv")

    for trial_knee in all_files_knee:
        df1 = pd.read_csv(
            trial_knee,
            skiprows=5,
            header=None,
            names=[
                "Frame",
                "SubFrame",
                "ASI_X_mm",
                "ASI_Y_mm",
                "ASI_Z_mm",
                "SHO_X_mm",
                "SHO_Y_mm",
                "SHO_Z_mm",
                "KNEE_X_mm",
                "KNEE_Y_mm",
                "KNEE_Z_mm",
                "ANK_X_mm",
                "ANK_Y_mm",
                "ANK_Z_mm",
            ],
        )

        hipknee_length = np.sqrt(
            (df1["ASI_X_mm"] - df1["KNEE_X_mm"]) ** 2
            + (df1["ASI_Y_mm"] - df1["KNEE_Y_mm"]) ** 2
            + (df1["ASI_Z_mm"] - df1["KNEE_Z_mm"]) ** 2
        )
        kneeank_length = np.sqrt(
            (df1["KNEE_X_mm"] - df1["ANK_X_mm"]) ** 2
            + (df1["KNEE_Y_mm"] - df1["ANK_Y_mm"]) ** 2
            + (df1["KNEE_Z_mm"] - df1["ANK_Z_mm"]) ** 2
        )
        hipank_length = np.sqrt(
            (df1["ASI_X_mm"] - df1["ANK_X_mm"]) ** 2
            + (df1["ASI_Y_mm"] - df1["ANK_Y_mm"]) ** 2
            + (df1["ASI_Z_mm"] - df1["ANK_Z_mm"]) ** 2
        )

        cKNEEa = (hipknee_length**2 + kneeank_length**2 - hipank_length**2) / (
            2 * hipknee_length * kneeank_length
        )
        knee_flex_angle_rad = np.arccos(cKNEEa)
        theta_knee_flex_angle = knee_flex_angle_rad * 180 / math.pi
        theta_knee_flex_angle = pd.Series(
            theta_knee_flex_angle.tolist(), index=df1["Frame"].values
        )
        # generate all joint angles inside processed subdirectory
        processed_folder_path = os.path.split(trial_knee)[0] + "/vicon/"
        os.makedirs(processed_folder_path, exist_ok=True)
        csv_file_path = processed_folder_path + os.path.split(trial_knee)[1]
        theta_knee_flex_angle.to_csv(csv_file_path, header=False)
        print(f"Generated knee flexion joint angles from vicon data to {csv_file_path}")


# Calculate shoulder abduction, shoulder extension and shoulder flexion joint angles.
def calculate_shoulder_angles(path_shoulder):
    ...


# Calculate elbox flexion joint angle.
def calculate_elbow_angles(path_elbow):
    ...


if __name__ == "__main__":
    data_path = "./data/"
    # `movement_group` is one of ["Left-Lower", "Left-Upper", "Right-Lower", "Right-Upper"]
    movement_groups = next(os.walk(data_path))[1]
    for group in movement_groups:
        # this can be ["hipabd", "hipext", "hipflex", "kneeflex", "shoabd", "shoext", "shoflex", "elbowflex"]
        # Example of movement_group_path is "./data/dl103-Right-Lower-obj1/"
        movement_group_path = data_path + group + "/"
        movements = next(os.walk(movement_group_path))[1]
        for movement in movements:
            # Example: movement_path could be "./data/dl103-Right-Lower-obj1/hipabd/"
            movement_path = movement_group_path + movement + "/"
            if "hip" in movement.lower():
                calculate_hip_angles(path_hip=movement_path)
            elif "knee" in movement.lower():
                calculate_knee_angles(path_knee=movement_path)
            elif "sho" in movement.lower():
                calculate_shoulder_angles(path_shoulder=movement_path)
            elif "elbow" in movement.lower():
                calculate_elbow_angles(path_elbow=movement_path)
            else:
                raise ValueError(f"unknown movement {movement}")
