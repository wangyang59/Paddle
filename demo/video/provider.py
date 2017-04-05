from paddle.utils.image_util import *
from paddle.trainer.PyDataProvider2 import *
import numpy as np
import random
import DeJpeg


def hook(settings, image_size, crop_size, color, file_list, is_train, **kwargs):
    settings.resize_size = image_size
    settings.crop_size = crop_size
    settings.color = True
    settings.is_train = is_train
    settings.inner_size = crop_size
    settings.is_test = not is_train

    c = 3 if settings.color else 1
    settings.input_size = settings.crop_size * settings.crop_size * c
    settings.file_list = file_list

    settings.mean_value = kwargs.get('mean_value')
    settings.mean_values = np.array(settings.mean_value, dtype=np.float32)
    print settings.mean_values[0], settings.mean_values[1]

    settings.input_types = [
        dense_vector(settings.input_size),  # image feature
        integer_value(1)
    ]  # labels

    settings.dp = DeJpeg.DecodeJpeg(
        12,  # multi-threads
        20480,
        settings.is_test,
        settings.color,
        settings.resize_size,
        settings.inner_size,
        settings.inner_size,
        settings.mean_values)


@provider(init_hook=hook, pool_size=51200)
def process(settings, file_list):
    with open(file_list, 'r') as fdata:
        lines = [line.strip() for line in fdata]
        random.shuffle(lines)
        l = len(lines)
        for i in xrange(0, l, 20480):
            start = i
            end = i + 20480
            if end > l:
                end = l

            data = []
            labels = []

            for i in range(start, end):
                img_path, lab = lines[i].strip().split('\t')
                data.append(open(img_path.strip(), 'rb').read())
                labels.append(int(lab.strip()))

            labels = np.array(labels, np.int32)
            settings.dp.start(data, labels)

            for i in xrange(len(labels)):
                img = np.empty(settings.input_size, dtype=np.float32)
                img_label = np.zeros(1, dtype=np.int32)
                settings.dp.get(img, img_label)
                yield [img, int(img_label[0])]
