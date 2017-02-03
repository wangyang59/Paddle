import numpy
import h5py

__all__ = ['read_from_mnist']


def read_from_mnist(filename):
    f = h5py.File('./data/mnist.h5')

    if 'train' in filename:
        filename2 = 'train_full'
    else:
        filename2 = 'test'

    images = f[filename2].value.reshape(-1, 28, 28)
    labels = f[filename2 + '_labels'].value

    # Define number of samples for train/test
    if "train" in filename:
        n = 60000
    else:
        n = 10000

    if "train" in filename:
        limit = 100
        cnt = {}
        for i in xrange(n):
            lbl = labels[i]
            if lbl not in cnt:
                cnt[lbl] = 1
            else:
                cnt[lbl] += 1
            if cnt[lbl] <= limit:
                yield {"pixel": images[i, :], 'label': labels[i]}
    else:
        for i in xrange(n):
            yield {"pixel": images[i, :], 'label': labels[i]}
