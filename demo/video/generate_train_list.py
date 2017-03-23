import os

video_data_root = "/home/wangyang59/Data/ILSVRC2015_256/Data/VID/train/"

video_dirs = os.listdir(video_data_root)

o = open("./videoData/" + "train.list", "w")
for video_dir in video_dirs:
    o.write(video_data_root + video_dir + "\n")
o.close()
