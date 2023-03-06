# python3 joint_angle_estimation.py --skip-process-video=True // To skip processing video.
# python3 joint_angle_estimation.py                           // For normal usage.
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from xlwt import *
from video_to_frame import process_video

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.patches as patches

import pandas as pd
import numpy as np
import cv2
import os
import math

import fire

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

LEFT_KEYPOINT_NAMES = [
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
]

RIGHT_KEYPOINT_NAMES = [
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
]

# Maps bones to a matplotlib color name.
# left corresponds to magenta.
# right corresponds to cyan.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}


def _zero_score_for_invalid_upper_ratio(
    show_left: bool, kpts_absolute_xy: pd.DataFrame, kpts_scores: pd.Series
):
    side = "left" if show_left else "right"
    wrist = kpts_absolute_xy.loc[f"{side}_wrist"]
    elbow = kpts_absolute_xy.loc[f"{side}_elbow"]
    shoulder = kpts_absolute_xy.loc[f"{side}_shoulder"]
    elbow_to_wrist = math.hypot(elbow[0] - wrist[0], elbow[1] - wrist[1])
    shoulder_to_elbow = math.hypot(shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    ratio = shoulder_to_elbow / elbow_to_wrist
    if not (0.8 < ratio < 1.2):
        print(f"Found (sho-elb) : (elb-wrist) ratio {ratio} outside (0.8, 1.2).")
        kpts_scores.loc[f"{side}_wrist"] = 0.0
        kpts_scores.loc[f"{side}_elbow"] = 0.0
        kpts_scores.loc[f"{side}_shoulder"] = 0.0


def _keypoints_and_edges_for_display(
    show_left, keypoints_with_scores, height, width, keypoint_threshold=0.25
):
    """Returns high confidence keypoints and edges for visualization.

    Args:
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      height: height of the image in pixels.
      width: width of the image in pixels.
      keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
      A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """
    keypoints_all = []  # list[pd.Series]
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]

        kpts_absolute_xy = pd.DataFrame(
            np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1),
            index=KEYPOINT_NAMES,
        )
        kpts_scores = pd.Series(kpts_scores, index=KEYPOINT_NAMES)

        _zero_score_for_invalid_upper_ratio(
            show_left=show_left,
            kpts_absolute_xy=kpts_absolute_xy,
            kpts_scores=kpts_scores,
        )

        if show_left:
            # NOTE: This adjustment is based visual guess by looking at
            # individual processed frames.
            # kpts_absolute_xy.loc["left_hip"][0] += -30
            # kpts_absolute_xy.loc["left_hip"][1] += +10
            kpts_absolute_xy = kpts_absolute_xy.loc[LEFT_KEYPOINT_NAMES]
            kpts_scores = kpts_scores.loc[LEFT_KEYPOINT_NAMES]
            kpts_indices = [KEYPOINT_DICT[kpt_name] for kpt_name in LEFT_KEYPOINT_NAMES]
            indices_to_kpts = {
                KEYPOINT_DICT[kpt_name]: kpt_name for kpt_name in LEFT_KEYPOINT_NAMES
            }
        else:
            # NOTE: This adjustment is based visual guess by looking at
            # individual processed frames.
            # kpts_absolute_xy.loc["right_hip"][0] += -30
            # kpts_absolute_xy.loc["right_hip"][1] += +10
            kpts_absolute_xy = kpts_absolute_xy.loc[RIGHT_KEYPOINT_NAMES]
            kpts_scores = kpts_scores.loc[RIGHT_KEYPOINT_NAMES]
            kpts_indices = [
                KEYPOINT_DICT[kpt_name] for kpt_name in RIGHT_KEYPOINT_NAMES
            ]
            indices_to_kpts = {
                KEYPOINT_DICT[kpt_name]: kpt_name for kpt_name in RIGHT_KEYPOINT_NAMES
            }

        low_confidence_indices = kpts_absolute_xy[
            kpts_scores <= keypoint_threshold
        ].index
        if not low_confidence_indices.empty:
            print(
                f"Found low confidence scores for {[(ind, kpts_scores.loc[ind]) for ind in low_confidence_indices]}."
            )

        kpts_above_thresh_absolute = kpts_absolute_xy[kpts_scores > keypoint_threshold]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (
                edge_pair[0] in kpts_indices
                and edge_pair[1] in kpts_indices
                # Uncomment the following if we do not want to show the edges with low-confidence keypoints on the processed frames.
                # and kpts_scores.loc[indices_to_kpts[edge_pair[0]]] > keypoint_threshold
                # and kpts_scores.loc[indices_to_kpts[edge_pair[1]]] > keypoint_threshold
            ):
                x_start = kpts_absolute_xy.loc[indices_to_kpts[edge_pair[0]]][0]
                y_start = kpts_absolute_xy.loc[indices_to_kpts[edge_pair[0]]][1]

                x_end = kpts_absolute_xy.loc[indices_to_kpts[edge_pair[1]]][0]
                y_end = kpts_absolute_xy.loc[indices_to_kpts[edge_pair[1]]][1]

                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)

    if keypoints_all:
        keypoints_xy = pd.concat(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors, kpts_scores


def draw_prediction_on_image(
    image,
    keypoints_with_scores,
    show_left,
    crop_region=None,
    output_image_height=None,
):
    """Draws the keypoint predictions on image.

    Args:
      image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
      output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
      A numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """
    height, width, _ = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis("off")

    ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle="solid")
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color="#FF1493", zorder=3)

    (
        keypoint_locs,
        keypoint_edges,
        edge_colors,
        keypoint_scores,
    ) = _keypoints_and_edges_for_display(
        show_left, keypoints_with_scores, height, width
    )

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)

    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs.values)

    if crop_region is not None:
        xmin = max(crop_region["x_min"] * width, 0.0)
        ymin = max(crop_region["y_min"] * height, 0.0)
        rec_width = min(crop_region["x_max"], 0.99) * width - xmin
        rec_height = min(crop_region["y_max"], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            rec_width,
            rec_height,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    plt.close(fig)

    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot,
            dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC,
        )

    return image_from_plot, keypoint_locs, keypoint_scores


# Choose DNN models
model_name = "movenet_thunder"  # @param ["movenet_lightning", "movenet_thunder", "movenet_lightning_f16.tflite", "movenet_thunder_f16.tflite", "movenet_lightning_int8.tflite", "movenet_thunder_int8.tflite"]
if "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
else:
    raise ValueError("Unsupported model name: %s" % model_name)


def movenet(input_image):
    model = module.signatures["serving_default"]
    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs["output_0"].numpy()
    return keypoints_with_scores


def joint_angle(joint, keypoint_locs):
    joint_found = False

    for side in ("left", "right"):
        if joint in f"{side}_elbow":
            a = np.array(
                [
                    keypoint_locs.loc[f"{side}_shoulder"][1],
                    keypoint_locs.loc[f"{side}_shoulder"][0],
                ]
            )
            b = np.array(
                [
                    keypoint_locs.loc[f"{side}_elbow"][1],
                    keypoint_locs.loc[f"{side}_elbow"][0],
                ]
            )
            c = np.array(
                [
                    keypoint_locs.loc[f"{side}_wrist"][1],
                    keypoint_locs.loc[f"{side}_wrist"][0],
                ]
            )
            joint_found = True
            break
        elif joint in f"{side}_shoulder":
            a = np.array(
                [
                    keypoint_locs.loc[f"{side}_elbow"][1],
                    keypoint_locs.loc[f"{side}_elbow"][0],
                ]
            )
            b = np.array(
                [
                    keypoint_locs.loc[f"{side}_shoulder"][1],
                    keypoint_locs.loc[f"{side}_shoulder"][0],
                ]
            )
            c = np.array(
                [
                    keypoint_locs.loc[f"{side}_hip"][1],
                    keypoint_locs.loc[f"{side}_hip"][0],
                ]
            )
            joint_found = True
            break
        elif joint in f"{side}_hip":
            a = np.array(
                [
                    keypoint_locs.loc[f"{side}_shoulder"][1],
                    keypoint_locs.loc[f"{side}_shoulder"][0],
                ]
            )
            b = np.array(
                [
                    keypoint_locs.loc[f"{side}_hip"][1],
                    keypoint_locs.loc[f"{side}_hip"][0],
                ]
            )
            c = np.array(
                [
                    keypoint_locs.loc[f"{side}_knee"][1],
                    keypoint_locs.loc[f"{side}_knee"][0],
                ]
            )
            joint_found = True
            break
        elif joint in f"{side}_knee":
            a = np.array(
                [
                    keypoint_locs.loc[f"{side}_hip"][1],
                    keypoint_locs.loc[f"{side}_hip"][0],
                ]
            )
            b = np.array(
                [
                    keypoint_locs.loc[f"{side}_knee"][1],
                    keypoint_locs.loc[f"{side}_knee"][0],
                ]
            )
            c = np.array(
                [
                    keypoint_locs.loc[f"{side}_ankle"][1],
                    keypoint_locs.loc[f"{side}_ankle"][0],
                ]
            )
            joint_found = True
            break

    if not joint_found:
        raise NotImplementedError(f"Unknown joint {joint}!")

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def process_frames_and_generate_csv(data_path="./data/", skip_process_video=False):
    # `movement_group` is one of ["Left-Lower", "Left-Upper", "Right-Lower", "Right-Upper"]
    # `movement` is one of ["hipabd", "hipext", "hipflex", "kneeflex", "shoabd", "shoext", "shoflex", "elbowflex"]
    #  - "Left-Lower" can have only ["hipabd", "hipext", "hipflex", "kneeflex"]
    #  - "Right-Lower" can have only ["hipabd", "hipext", "hipflex", "kneeflex"]
    #  - "Left-Upper" can have only ["shoabd", "shoext", "shoflex", "elbowflex"]
    #  - "Right-Upper" can have only ["shoabd", "shoext", "shoflex", "elbowflex"]
    movement_groups = next(os.walk(data_path))[1]

    #
    # movement_groups = ["left_upper"]
    #

    for group in movement_groups:
        # generate frames for all videos in each movement in the group
        # Example of movement_group_path is "./data/dl103-Right-Lower-obj1/"
        movement_group_path = data_path + group + "/"
        if not skip_process_video:
            process_video(movement_group_path=movement_group_path)
        # this can be ["hipabd", "hipext", "hipflex", "kneeflex", "shoabd", "shoext", "shoflex", "elbowflex"]
        movements = next(os.walk(movement_group_path))[1]

        #
        # movements = ["dl212_leftupper_shoext"]
        #

        for movement in movements:
            # Example: movement_path could be "./data/dl103-Right-Lower-obj1/hipabd/"
            movement_path = movement_group_path + movement + "/"
            print("Start processing", movement_path)

            # Inside movement_path, we will have subdirectories like:
            # "frames-{video_name}/{frame_number}.jpg"
            # trial_dirs is the list of dirs which start with `frame-{video_name}`
            trial_dirs = list(
                filter(lambda x: "frames-" in x, os.listdir(movement_path))
            )
            print("Found frame dirs:", trial_dirs)

            #
            # trial_dirs = ["frames-dl212_left_upper_shoext_001.2104948.20230204165442"]
            #

            # Inside a trial_dir, we will have image files like 0.jpg, 1.jpg, ... & so on.
            for trial_dir in trial_dirs:
                # Example: trial_path could be "./data/dl103-Right-Lower-obj1/hipabd/frames-{video_name}/"
                trial_path = movement_path + trial_dir + "/"
                print(
                    "Start Processing------------------------------------->:",
                    trial_path,
                )
                frames = list(filter(lambda x: ".jpg" in x, os.listdir(trial_path)))

                show_left = "left" in group.lower()
                if show_left:
                    joints = [
                        "left_elbow",
                        "left_shoulder",
                        "left_hip",
                        "left_knee",
                    ]
                else:
                    joints = [
                        "right_elbow",
                        "right_shoulder",
                        "right_hip",
                        "right_knee",
                    ]

                model_outputs_for_trial = dict()

                #
                # frames = ["65.jpg"]
                #

                for frame in frames:
                    # Example: frame = "150.jpg"
                    frame_number = int(frame[:-4])
                    # For each frame_number, initialize a dict for joint_name to joint_angle mapping
                    model_outputs_for_trial[frame_number] = dict()

                    frame_path = trial_path + frame
                    image = tf.io.read_file(frame_path)
                    image = tf.image.decode_jpeg(image)

                    # Resize and pad the image to keep the aspect ratio and fit the expected size.
                    input_image = tf.expand_dims(image, axis=0)
                    input_image = tf.image.resize_with_pad(
                        input_image, input_size, input_size
                    )

                    # Run model inference.
                    keypoint_with_scores = movenet(input_image)

                    # Visualize the predictions with image.
                    display_image = tf.expand_dims(image, axis=0)
                    display_image = tf.cast(
                        tf.image.resize_with_pad(display_image, 1280, 1280),
                        dtype=tf.int32,
                    )

                    output_overlay, keypoint_locs, _ = draw_prediction_on_image(
                        image=np.squeeze(display_image.numpy(), axis=0),
                        keypoints_with_scores=keypoint_with_scores,
                        show_left=show_left,
                        crop_region=None,
                        output_image_height=None,
                    )

                    plt.figure(figsize=(15, 15))
                    plt.imshow(output_overlay)
                    plt.margins(0, 0)
                    plt.axis("off")
                    # Output annotate
                    pix_index = 0

                    # print(f"Frame {frame} has scores: {keypoint_scores.to_dict()}")

                    for joint in joints:
                        try:
                            angle = round(joint_angle(joint, keypoint_locs), 2)
                        except KeyError:
                            print(
                                f"Skipping frame {frame} for joint {joint} because of low confidence."
                            )
                            continue

                        # map from joint_name to joint_angle
                        model_outputs_for_trial[frame_number][joint] = angle

                        infor = joint + ": " + str(angle) + "\N{DEGREE SIGN}"
                        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
                        plt.text(
                            30,
                            30 + pix_index * 30,
                            infor,
                            ha="left",
                            va="center",
                            size=15,
                            bbox=bbox_props,
                        )

                        pix_index = pix_index + 1

                    # Store the processed images
                    processed_frames_path = trial_path + "processed_frames/"
                    os.makedirs(processed_frames_path, exist_ok=True)
                    plt.savefig(
                        processed_frames_path + "P_" + frame,
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    plt.close("all")

                print(
                    "Finished processing all frames in this trial ---------------------------------->:",
                    trial_dir,
                )

                # Save model_outputs for this trial to csv
                # Example for model_outputs_for_trial_csv_path: "./data/dl103-Right-Lower-obj1/hipabd/model/dl1003_hipabd_001.csv"
                # trial_dir looks like "frames-dl1003a_hipabd_001.2104948.20220804175333"
                model_outputs_for_movement_path = movement_path + "model/"
                os.makedirs(model_outputs_for_movement_path, exist_ok=True)
                model_outputs_for_trial_csv_path = (
                    model_outputs_for_movement_path
                    + trial_dir.split(".")[0].strip("frames-")
                    + ".csv"
                )
                df = pd.DataFrame.from_dict(model_outputs_for_trial, orient="index")
                df.sort_index(inplace=True)
                df.to_csv(model_outputs_for_trial_csv_path)


if __name__ == "__main__":
    fire.Fire(process_frames_and_generate_csv)
