import numpy as np
import data_handler
import sys
num_frames = int(sys.argv[1])

data = np.loadtxt(
    "rank-00000", comments=";").reshape((-1, 64 * 64 * num_frames))
#data[data>0.1] = 1

batch_size = 50
image_size = 64
num_digits = 2
step_length = 0.1

dataHandler = data_handler.BouncingMNISTDataHandler(
    num_frames, batch_size, image_size, num_digits, step_length,
    "./data/mnist.h5")

dataHandler.DisplayData(data, case_id=0, output_file="rec")
