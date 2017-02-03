import numpy
from PIL import Image
import h5py


def crop_dim(old_dim, new_dim):
    if new_dim >= old_dim:
        old_s = 0
        old_e = old_dim
        new_s = (new_dim - old_dim) / 2
        new_e = new_s + old_dim
    else:
        old_s = (old_dim - new_dim) / 2
        old_e = old_s + new_dim
        new_s = 0
        new_e = new_dim
    return old_s, old_e, new_s, new_e


# The input img is 2-D numpy array
# The return img_t is also a 2-D numpy array
def transform_img(img,
                  angle,
                  x_scale,
                  y_scale,
                  val_scale,
                  resample=Image.BICUBIC):
    old_dim = len(img.shape)
    if old_dim == 3:
        img = numpy.squeeze(img)

    height_old, width_old = img.shape

    img = Image.fromarray(img)

    img_r = img.rotate(angle, resample=resample)

    width_new = int(width_old * x_scale)
    height_new = int(height_old * y_scale)

    img_rs = img_r.resize((width_new, height_new), resample=resample)
    img_rs = numpy.array(img_rs.getdata()).reshape((height_new, width_new))

    old_ws, old_we, new_ws, new_we = crop_dim(width_old, width_new)
    old_hs, old_he, new_hs, new_he = crop_dim(height_old, height_new)

    img_t = numpy.zeros((height_old, width_old))
    img_t[old_hs:old_he, old_ws:old_we] = img_rs[new_hs:new_he, new_ws:new_we]

    old_max = numpy.max(img_t)
    img_t = numpy.minimum(img_t * val_scale, old_max)

    if old_dim == 3:
        img_t = img_t.reshape((1, height_old, width_old))

    return img_t


def main():
    i = 10
    f = h5py.File('./data/mnist.h5')
    data_ = f['train_full'].value.reshape(-1, 28, 28)
    img = numpy.squeeze(data_[i, :, :]) * 255.0
    img_t = transform_img(img, -15, 1, 1, 1)
    #     filename = './data2/raw_data/train'
    #      
    #     imgf = filename + "-images-idx3-ubyte"
    #     labelf = filename + "-labels-idx1-ubyte"
    #     f = open(imgf, "rb")
    #     l = open(labelf, "rb")
    #  
    #     f.read(16)
    #     l.read(8)
    #     n = 60000
    #     images = numpy.fromfile(f, 'ubyte', count=n * 28 * 28).reshape((n, 28, 28)).astype('float32')
    #     images = images / 255.0
    #      
    #     img = numpy.squeeze(images[i, :, :]) * 255.0
    #     img_t = transform_img(img, -15, 1, 1, 1)
    #      
    #     labels = numpy.fromfile(l, 'ubyte', count=n).astype("int")
    #     print(labels[i])
    #     f.close()
    #     l.close()
    #     
    im = Image.fromarray(img + 10.0).convert('RGB')
    im.save("original.png")
    im = Image.fromarray(img_t + 10.0).convert('RGB')
    im.save("transform.png")


if __name__ == '__main__':
    main()
