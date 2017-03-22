import io
import os
import sys
import Image
import argparse
import logging
from time import time
import multiprocessing
logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--imagenet_dir', default='/media/yi/DATA/data-orig/ILSVRC2012')
    parser.add_argument(
        '--output_dir', default='/media/yi/DATA/data-orig/ILSVRC2012_256')
    parser.add_argument('--size', type=int, default=256)

    args = parser.parse_args()
    imagenet_dir = args.imagenet_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    size = args.size

    image_dirs = os.listdir(os.path.join(imagenet_dir, 'train'))
    if not os.path.exists(os.path.join(output_dir, 'train')):
        os.makedirs(os.path.join(output_dir, 'train'))

    num_images = 0
    total_time = 0
    for image_dir in image_dirs:
        if not os.path.exists(os.path.join(output_dir, 'train', image_dir)):
            os.makedirs(os.path.join(output_dir, 'train', image_dir))
        image_names = os.listdir(os.path.join(imagenet_dir, 'train', image_dir))
        for image_name in image_names:
            num_images += 1
            start_time = time()
            im = Image.open(
                os.path.join(imagenet_dir, 'train', image_dir, image_name))
            if im.size[0] < im.size[1]:
                new_height = size
                new_width = int(round(float(size) / im.size[0] * im.size[1]))
            else:
                new_width = size
                new_height = int(round(float(size) / im.size[1] * im.size[0]))
            im = im.resize((new_height, new_width), Image.ANTIALIAS)
            new_file_name = os.path.join(output_dir, 'train', image_dir,
                                         image_name)
            im.save(new_file_name, "JPEG")
            end_time = time()
            total_time += end_time - start_time
            avg_time = total_time / num_images
            approximate_finish_time = (
                1280000 - num_images) * avg_time / 60 / 60
            logging.info('resize image save to %s, takes %.2f second',
                         new_file_name, end_time - start_time)
            logging.info('approximate finish time: %.2f hours',
                         approximate_finish_time)
