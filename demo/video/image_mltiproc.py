from PIL import Image
import os
import numpy as np
import multiprocessing as mp
import functools
import h5py


def do_one_image_dir(image_size, min_frames, q, image_file_paths_chunk):
    #print(image_file_path)

    #print(q.qsize())
    #print("after q: " + str(do_one_image_dir.q.full()) + str(image_file_path))
    half_image_size = image_size / 2
    #     image_files = os.listdir(os.path.join(file_dir, image_dir))
    #     image_files.sort()
    #     image_files = image_files[start_id : (start_id + 6)]
    #     if len(image_files) < min_frames:
    #         return
    images = np.zeros((6, 256 * 256 * 3), dtype=np.float32)
    for image_file_path in image_file_paths_chunk:
        for cnt, image_file in enumerate(image_file_path):
            #             if cnt == settings.num_frames * settings.jump:
            #                 break
            #             if cnt % settings.jump != 0:
            #                 continue
            img = Image.open(image_file)
            half_the_width = img.size[0] / 2
            half_the_height = img.size[1] / 2
            img = img.crop((half_the_width - half_image_size,
                            half_the_height - half_image_size,
                            half_the_width + half_image_size,
                            half_the_height + half_image_size))
            #images.append(np.array(img.getdata(), dtype=np.float32).reshape((-1), order = 'F') / 255.0)
            images[cnt, :] = np.array(
                img.getdata(), dtype=np.float32).reshape(
                    (-1), order='F') / 255.0
            img.close()

        q.put(images[:, :])


def get_h5py(q, file_paths):
    for file_path in file_paths[0:5]:
        #         f = h5py.File(file_path, 'r')
        #         q.put(f["data"].value)
        #         f.close()
        f = h5py.File(file_path, 'r')
        images = f["data"].value
        n = images.shape[0]
        for i in range(n - 6 + 1):
            q.put(images[i:(i + 6), :])
        del images

    q.put("end")


class MultiProcessImageTransformer(object):
    def __init__(self, procnum=10, image_size=None, min_frame=None):
        """
        Processing image with multi-process. If it is used in PyDataProvider,
        the simple usage for CNN is as follows:
       
        .. code-block:: python

            def hool(settings, is_train,  **kwargs):
                settings.is_train = is_train
                settings.mean_value = np.array([103.939,116.779,123.68], dtype=np.float32)
                settings.input_types = [
                    dense_vector(3 * 224 * 224),
                    integer_value(1)]
                settings.transformer = MultiProcessImageTransformer(
                    procnum=10,
                    resize_size=256,
                    crop_size=224,
                    transpose=(2, 0, 1),
                    mean=settings.mean_values,
                    is_train=settings.is_train)


            @provider(init_hook=hook, pool_size=20480)
            def process(settings, file_list):
                with open(file_list, 'r') as fdata:
                    for line in fdata: 
                        data_dic = np.load(line.strip()) # load the data batch pickled by Pickle.
                        data = data_dic['data']
                        labels = data_dic['label']
                        labels = np.array(labels, dtype=np.float32)
                        for im, lab in settings.dp.run(data, labels):
                            yield [im.astype('float32'), int(lab)]

        :param procnum: processor number.
        :type procnum: int
        :param resize_size: the shorter edge size of image after resizing.
        :type resize_size: int
        :param crop_size: the croping size.
        :type crop_size: int
        :param transpose: the transpose order, Paddle only allow C * H * W order.
        :type transpose: tuple or list
        :param channel_swap: the channel swap order, RGB or BRG.
        :type channel_swap: tuple or list
        :param mean: the mean values of image, per-channel mean or element-wise mean.
        :type mean: array, The dimension is 1 for per-channel mean.
                    The dimension is 3 for element-wise mean. 
        :param is_train: training peroid or testing peroid.
        :type is_train: bool.
        :param is_color: the image is color or gray. 
        :type is_color: bool.
        :param is_img_string: The input can be the file name of image or image string.
        :type is_img_string: bool.
        """

        self.procnum = procnum
        self.q = mp.Queue(1024)
        self.processes = []

        self.image_size = image_size
        self.min_frame = min_frame

    def run(self, image_file_paths):
        #fun = functools.partial(do_one_image_dir, self.image_size, self.min_frame, self.q)
        #         return self.pool.imap_unordered(
        #             fun, zip(range(len(image_file_paths)), image_file_paths), chunksize=10)
        fun = functools.partial(get_h5py, self.q)
        n = len(image_file_paths) / self.procnum
        for image_file_paths_chunk in [
                image_file_paths[i:i + n]
                for i in xrange(0, len(image_file_paths), n)
        ]:
            process = mp.Process(target=fun, args=(image_file_paths_chunk, ))
            self.processes.append(process)
            process.daemon = True
            process.start()
