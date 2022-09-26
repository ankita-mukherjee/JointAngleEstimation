# JointAngleEstimation
This project is to estimate the joint angles by deep learning

This is [running video dataset](https://www.kaggle.com/datasets/kmader/running-videos).

```
pip install -r requirements.txt
```

## Workflow:

- No need to manually copy or move directories or files - avoid at all costs!
- **Idea**: Just extract `.zip` file (downloaded from box) to `data/` and run python scripts!
- After extraction, `data/` directory will contain something like `dl103-Right-Lower-obj1` subdirectory.
- Movenet model inference (`joint_angle_estimation.py`) will run inference on videos in `./data/model/`.
- Joint angles from vicon (`joint_angle_vicon.py`) will generate joint angles from vicon csv files in `./data/vicon/`.
- `calculate_error.py` calculates the RMSE, MAE and R2 between the model inferred joint angles and the joint angles from the vicon dataset.

### Example directory structure for `./data/`

```
./data/
|
|------dl103-Right-Lower-obj1/
       |
       |------ hipabd/
               |
               |------ model/
                       |------ {trial_name}.csv
               |
               |------ vicon/
                       |------ {trial_name}.csv
               |
               |------ frames-{video_name}/
                       |------ {frame_number}.jpg
               |
               |------ {video_name}.avi
               |------ {trial_name}.csv
        |
        |------ hipext/
```