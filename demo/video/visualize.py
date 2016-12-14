import numpy as np
import data_handler
data = np.loadtxt("rank-00000", comments=";").reshape((-1, 64*64*10))

num_frames = 10
batch_size = 50
image_size = 64
num_digits = 2
step_length = 0.1

dataHandler = data_handler.BouncingMNISTDataHandler(num_frames, 
                                                     batch_size, 
                                                     image_size, 
                                                     num_digits, 
                                                     step_length, 
                                                     "./data/mnist.h5")

dataHandler.DisplayData(data, case_id=10, output_file="rec")
