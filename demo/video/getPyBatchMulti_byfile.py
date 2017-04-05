import sys, os, random
import cPickle
from struct import unpack
from struct import pack
import numpy
import multiprocessing
from PIL import Image


def multi_gen_batch(file, outputDir, numPerBatch):
    file_list = []
    for lines in open(file, 'r'):
        path, label = lines.strip().split('\t')
        file_list.append([path, label])
    random.shuffle(file_list)

    len_file_list = len(file_list)
    cur_id = 0
    file_id = 0
    while cur_id < len_file_list:
        thread = []
        for i in xrange(8):
            end_id = min(len_file_list, cur_id + numPerBatch)
            end_id_temp = end_id
            w = multiprocessing.Process(
                target=gen_batch,
                args=([j[0] for j in file_list[cur_id:end_id_temp]],
                      [j[1] for j in file_list[cur_id:end_id_temp]],
                      outputDir + '/data_batch_' + str(file_id), i))
            w.daemon = True
            thread.append(w)
            cur_id = end_id
            file_id += 1
            if cur_id == len_file_list:
                break

        for i in thread:
            i.start()
        for i in thread:
            i.join()


def gen_batch(file_list, label_list, file_output_name, thread_id):
    assert (len(file_list) == len(label_list))

    data = []
    labelList = []
    filenames = []
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    k = 0
    for i in xrange(len(file_list)):
        file = file_list[i]
        label = label_list[i]
        count[1] += 1
        try:
            data.append(open(file, 'rb').read())
            labelList.append(label)
            filenames.append(file)
            k += 1
        except:
            count[5] += 1
            continue

    try:
        if len(data) != k:
            print >> sys.stderr, "len of data:%d != k:%d" % (len(data), k)
            raise ValueError
        count[6] += 1
        output = {}
        output['label'] = labelList
        output['data'] = data
        output['filenames'] = filenames

        output['batch_label'] = 'train batch ' + file_output_name
        cPickle.dump(
            output,
            open(file_output_name, 'w'),
            protocol=cPickle.HIGHEST_PROTOCOL)

        print >> sys.stderr, "save batch:", file_output_name, "ok"

    except:
        print >> sys.stderr, "save batch error"
    print >> sys.stderr, "count", count


if __name__ == "__main__":
    file = sys.argv[1]
    outputDir = sys.argv[2]
    numPerBatch = int(sys.argv[3])

    if not os.path.isdir(outputDir):
        os.system("mkdir -p %s" % outputDir)
    multi_gen_batch(file, outputDir, numPerBatch)
