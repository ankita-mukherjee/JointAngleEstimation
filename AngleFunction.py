import numpy as np

def jointAngle(joint,_keypoints_and_edges_for_display):
    # (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(keypoint_with_scores, 1280, 1280)
    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display
    try:
        if joint in "left_elbow":
            a = np.array([keypoint_locs[5,1],keypoint_locs[5,0]])   #'left_shoulder': 5,
            b = np.array([keypoint_locs[7,1],keypoint_locs[7,0]])   #'left_elbow': 7,
            c = np.array([keypoint_locs[9,1],keypoint_locs[9,0]])  #'left_wrist': 9,
            
        elif joint in "right_elbow":
            a = np.array([keypoint_locs[6,1],keypoint_locs[6,0]])   #'right_shoulder': 6,
            b = np.array([keypoint_locs[8,1],keypoint_locs[8,0]])   #'right_elbow': 8,
            c = np.array([keypoint_locs[10,1],keypoint_locs[10,0]])  #'right_wrist': 10,

        elif joint in "left_shoulder":
            a = np.array([keypoint_locs[7,1],keypoint_locs[7,0]])   #'left_elbow': 7,
            b = np.array([keypoint_locs[5,1],keypoint_locs[5,0]])   #'left_shoulder': 5,
            c = np.array([keypoint_locs[11,1],keypoint_locs[11,0]]) #'left_hip': 11,

        elif joint in "left_hip":
            a = np.array([keypoint_locs[5,1],keypoint_locs[5,0]])   #'left_shoulder': 5, 
            b = np.array([keypoint_locs[11,1],keypoint_locs[11,0]]) #'left_hip': 11,
            c = np.array([keypoint_locs[13,1],keypoint_locs[13,0]]) #'left_knee': 13, 

        elif joint in "left_knee":
            a = np.array([keypoint_locs[11,1],keypoint_locs[11,0]]) #'left_hip': 11,
            b = np.array([keypoint_locs[13,1],keypoint_locs[13,0]]) #'left_knee': 13,
            c = np.array([keypoint_locs[15,1],keypoint_locs[15,0]]) #'left_ankle': 15, 

        elif joint in "right_shoulder":
            a = np.array([keypoint_locs[8,1],keypoint_locs[8,0]])   #'right_elbow': 8,
            b = np.array([keypoint_locs[6,1],keypoint_locs[6,0]])   #'right_shoulder': 6,
            c = np.array([keypoint_locs[12,1],keypoint_locs[12,0]]) #'right_hip': 12, 


        elif joint in "right_hip":
            a = np.array([keypoint_locs[6,1],keypoint_locs[6,0]])   #'right_shoulder': 6,
            b = np.array([keypoint_locs[12,1],keypoint_locs[12,0]]) #'right_hip': 12,
            c = np.array([keypoint_locs[14,1],keypoint_locs[14,0]]) #'right_knee': 14,

        elif joint in "right_knee":
            a = np.array([keypoint_locs[12,1],keypoint_locs[12,0]]) #'right_hip': 12,
            b = np.array([keypoint_locs[14,1],keypoint_locs[14,0]]) #'right_knee': 14,
            c = np.array([keypoint_locs[16,1],keypoint_locs[16,0]]) #'right_ankle': 16, 
    
    except IndexError as e: # catch the error
        print(e)  
        print("Ignore IndexError")
        a=0
        b=0
        c=0
    
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)