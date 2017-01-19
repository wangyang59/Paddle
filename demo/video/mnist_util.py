import numpy

__all__ = ['read_from_mnist']


def read_from_mnist(filename):
    imgf = filename + "-images-idx3-ubyte"
    labelf = filename + "-labels-idx1-ubyte"
    f = open(imgf, "rb")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)

    # Define number of samples for train/test
    if "train" in filename:
        n = 60000
    else:
        n = 10000

    limit = 100
    images = numpy.fromfile(
        f, 'ubyte', count=n * 28 * 28).reshape((n, 28 * 28)).astype('float32')
    images = images / 255.0
    labels = numpy.fromfile(l, 'ubyte', count=n).astype("int")

    if "train" in filename:
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

    f.close()
    l.close()
