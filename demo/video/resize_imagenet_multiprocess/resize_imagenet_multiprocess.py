import io
import os
import sys
from PIL import Image
import argparse
import logging
import time
import multiprocessing
logging.getLogger().setLevel(logging.INFO)


# def resize_image(image_file, input_dir, output_dir):
def resize_image(image_file):
    image_name = image_file['image_name']
    input_dir = image_file['input_dir']
    output_dir = image_file['output_dir']
    im = Image.open(os.path.join(input_dir, image_name))
    if im.size[0] < im.size[1]:
        new_height = size
        new_width = int(round(float(size) / im.size[0] * im.size[1]))
    else:
        new_width = size
        new_height = int(round(float(size) / im.size[1] * im.size[0]))
    im = im.resize((new_height, new_width), Image.ANTIALIAS)
    new_file_name = os.path.join(output_dir, image_name)
    im.save(new_file_name, "JPEG")
    im = None
    logging.info('Resize image save to %s, height %d, width %d', new_file_name,
                 new_height, new_width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--imagenet_dir', default='/media/yi/DATA/data-orig/ILSVRC2012')
    parser.add_argument(
        '--output_dir', default='/media/yi/DATA/data-orig/ILSVRC2012_256')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--subset', default='train')

    args = parser.parse_args()
    imagenet_dir = args.imagenet_dir
    output_dir = args.output_dir
    size = args.size
    subset = args.subset

    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_dirs = os.listdir(os.path.join(imagenet_dir, subset))
    if not os.path.exists(os.path.join(output_dir, subset)):
        os.makedirs(os.path.join(output_dir, subset))
    for image_dir in image_dirs:
        if not os.path.exists(os.path.join(output_dir, subset, image_dir)):
            os.makedirs(os.path.join(output_dir, subset, image_dir))

    # Scan image files
    image_files = []
    for image_dir in image_dirs:
        image_names = os.listdir(os.path.join(imagenet_dir, subset, image_dir))
        for image_name in image_names:
            image_file = {
                'image_name': os.path.join(subset, image_dir, image_name),
                'input_dir': imagenet_dir,
                'output_dir': output_dir
            }
            image_files.append(image_file)
        logging.info('Finsih scan image directory: %s', image_dir)

    # Start multiprocessing image resize
    start_time = time.time()
    core_ct = os.sysconf('SC_NPROCESSORS_ONLN')
    pool = multiprocessing.Pool(processes=core_ct)
    pool.map(resize_image, image_files)
    pool.close()
    pool.join()
    end_time = time.time()
    total_time = end_time - start_time
    logging.info('finish in %.2f hours', total_time / 60 / 60)
