import glob
import json
import cv2
import random
import os



path = 'DATASET'
files = []
for path, subdirs, ifiles in os.walk(path):
    for name in ifiles:
        filename = os.path.join(path, name)
        if 'mirror' in filename:
            continue
        print(filename)

        files.append(filename)

splits = [('dev_set.txt', 0.05), ('test_set.txt', 0.05), ('train_set.txt', True)]

random.shuffle(files)

# Some output...
N = len(files)
print('Total:', N, 'samples')

# Main loop. Do the splitting and prepare for data augmentation *of the training set*
for out_filename, split in splits:
    n = int(split*N)
    if split is True:
        n = len(files)
    print(out_filename, n)

    out_file = open(out_filename, 'w')
    for i in range(n):
        line_text = files.pop()+'\n'
        out_file.write(line_text)
        l = line_text.split('/')
        out_file.write(l[0]+'/mirror_'+l[1])
    out_file.close()





