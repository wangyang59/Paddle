import os
import io
import sys
import pickle

if __name__ == '__main__':
    train_file_list = {}
    val_file_list = []
    dirs = os.listdir('/media/yi/DATA/data-orig/imagenet/train')
    for each_dir in dirs:
        file_list = os.listdir(
            os.path.join('/media/yi/DATA/data-orig/imagenet/train', each_dir))
        train_file_list[each_dir] = file_list

    # dirs = os.listdir('/media/yi/DATA/data-orig/imagenet/val')
    # for each_file in dirs:
    #    val_file_list.append(each_file)

    save_file = io.open('file_list', 'wb')
    pickle.dump(
        {
            'train_file_list': train_file_list,
            'val_file_list': val_file_list
        },
        save_file,
        protocol=pickle.HIGHEST_PROTOCOL)
