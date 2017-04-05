import numpy as np
import sys
import matplotlib.pyplot as plt

image_size = int(sys.argv[1])
image_channel = int(sys.argv[2])
num_frames = int(sys.argv[3])

data = np.loadtxt(
    "rank-00000", comments=";").reshape(
        (-1, image_size * image_size * image_channel * num_frames))
data = (data[0, :]).reshape(
    (image_size, image_size, image_channel, num_frames),
    order="F").transpose(1, 0, 2, 3)

num_rows = 2
# create figure for original sequence
plt.figure(2, figsize=(num_frames / 2, 2))
plt.clf()
for i in xrange(num_frames / 2):
    plt.subplot(num_rows, num_frames / 2, i + 1)
    plt.imshow(data[:, :, :, i * 2])
    #interpolation="nearest")
    plt.axis('off')

    plt.subplot(num_rows, num_frames / 2, i + 1 + num_frames / 2)
    plt.imshow(data[:, :, :, i * 2 + 1], interpolation="nearest")
    plt.axis('off')
plt.draw()
plt.savefig("images.png", bbox_inches='tight')
