import os
import io
import sys
import pickle

if __name__ == '__main__':
    file_list = pickle.load(open('file_list'))
    my_file_list = pickle.load(open('my_file_list'))

    handle = open('diff_file_list', 'wb')

    for key in file_list['train_file_list'].keys():
        if key in my_file_list['train_file_list']:
            files = file_list['train_file_list'][key]
            my_files = my_file_list['train_file_list'][key]
            diff_files = list(set(files) - set(my_files))
            if len(diff_files) > 0:
                print diff_files
                handle.write('%s\n' % key)
        else:
            print key + ' not in my_file_list'
            handle.write('%s\n' % key)
