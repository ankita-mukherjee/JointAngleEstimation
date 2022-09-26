# This file extract frames from the video
import cv2
import os


def process_video(movement_group_path):
    # this can be ["hipabd", "hipext", "hipflex", "kneeflex", "shoabd", "shoext", "shoflex", "elbowflex"]
    movements = next(os.walk(movement_group_path))[1]
    for movement in movements:
        print("Start processing", movement)
        # Example: movement_path could be "./data/dl103-Right-Lower-obj1/hipabd/"
        movement_path = movement_group_path + movement + "/"
        video_files = list(filter(lambda x: ".avi" in x, os.listdir(movement_path)))
        for video_name in video_files:
            print("Checking", video_name)
            trial_dir_path = movement_path + "frames-" + video_name[:-4] + "/"
            os.makedirs(trial_dir_path, exist_ok=True)
            print("Processing...")
            vidcap = cv2.VideoCapture(movement_path + video_name)
            (
                success,
                image,
            ) = vidcap.read()  # Grabs, decodes and returns the next video frame

            frame_num = 0
            while success:
                cv2.imwrite(
                    trial_dir_path + f"{frame_num}.jpg", image
                )  # save frame as JPEG file
                success, image = vidcap.read()
                frame_num += 1
            print("Processing success", frame_num, "frames saved")
