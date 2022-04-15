# This file extract frames from the video
import cv2
import os

work_path = "./Videos/Samples/" # work_path = "./content/dltest/"
actions = next(os.walk(work_path))[1] # list folder 
print('Folders contain frames:',actions)

for action in actions:
    print ("Start processing", action)
    path = work_path + action +'/'
    dir_list = os.listdir(path)
    video_files = list(filter(lambda x: '.mp4' in x, dir_list)) # Keep all video files
    print('Video Files \n:', video_files)

    for video_name in video_files:

        print ('Cheking', video_name)
        isFile = os.path.isdir(path+"Frames-"+video_name[:-4])

        if not isFile:
            print('No Frames. Making Dir: \n', path+"Frames-"+video_name[:-4])
            os.mkdir(path+"Frames-"+video_name[:-4])
            print('Processing...')
            vidcap = cv2.VideoCapture(path + video_name)
            success, image = vidcap.read() # Grabs, decodes and returns the next video frame
            
            count = 0
            while success:
                cv2.imwrite(path+"Frames-"+video_name[:-4]+"/%d.jpg" % count, image)     # save frame as JPEG file      
                success, image = vidcap.read()
                count += 1
            print('Processing success', count, "frames saved")
        