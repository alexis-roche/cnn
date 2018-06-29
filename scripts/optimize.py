"""
python optimize <learning_rate> <batch_size> <init_var> <max_var> <epochs>
"""
import sys
import os
import glob

import numpy as np
import pylab as pl

import vii

import cnn
from cnn.data_generator import PatchSelector
from cnn.optimizer import *


SIZE = 45
PATCHES_PER_IMAGE = 1
RANDOM = True
EXAMPLE_PATH = '/home/alexis/tmp'
EXAMPLE_NAME = 'example'


def load_example(fnpz):
    f = np.load(fnpz)
    return f['patch'], int(f['label']), int(f['food'])


def check_example(fnpz):
    x, y, f = load_example(fnpz)
    pl.imshow(x[...,::-1])
    pl.show(block=False)
    print(fnpz)
    pl.title('Label=%d (food=%d)' % (y, f))
    n = pl.waitforbuttonpress()
    if not n:
        pl.close()
        return
    n = input('Do you really want to correct label? [0/1] ')
    if not n:
        pl.close()
        return
    if y == 0:
        yc = f + 1
    else:
        yc = 0
    np.savez(fnpz, patch=x, label=yc, food=f)
    print('Label changed: %d -> %d' % (y, yc))
    pl.close()


def get_example_files(path, name):
    files = glob.glob(os.path.join(path, '%s*.npz' % name))
    return [os.path.join(path, '%s%d.npz' % (name, i + 1)) for i in range(len(files))]


def load_examples(path, name):
    files = get_example_files(path, name)
    p, _, _ = load_example(files[0])
    x = np.zeros([len(files)] + list(p.shape))
    y = np.zeros(len(files))
    for i in range(len(files)):
        print('Loading NPZ file %d/%d' % (i+1, len(files)))
        x[i, ...], y[i], _ = load_example(files[i])
    return x, y


def normalize(x, y):
    return x / 255, y > 0
    

def run_optimizer(opt, epochs=1):
    for e in range(epochs):
        out = []
        for i in range(opt._num_batches()):
            out.append(opt._update())
            print('Iteration: %d, Epoch = %d, Losses = %s' % (i + 1, e + 1, out[-1]))
    return np.array(out)


def init_classifier(x, y, prop_test):
    classif = cnn.ImageClassifier(x.shape[1:3], y.max() + 1)
    x_train, y_train, x_test, y_test = cnn.split(x, y, prop_test=prop_test)
    classif._configure_training(x_train, y_train, 0, 1e-4, 1e-6, 'glorot_uniform', 'zeros', x_test, y_test)
    return classif
    


################################################
# Main
################################################

x, y = normalize(*load_examples(EXAMPLE_PATH, EXAMPLE_NAME))
class_weight = cnn.balanced_class_weight(y)
prop_test = .1

"""
Parameters of Average EP:
* batch_size
* lr
* init_var
* max_var
* class_weight
"""
lr = 1
batch_size = 32
init_var = 1
max_var = 100
epochs = 2

if len(sys.argv) > 1:
    lr = float(sys.argv[1])
    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
        if len(sys.argv) > 3:
            init_var = float(sys.argv[3])
            if len(sys.argv) > 4:
                max_var = float(sys.argv[4])
                if len(sys.argv) > 5:
                    epochs = int(sys.argv[5])


classif = init_classifier(x, y, prop_test)
opt = AverageEP(classif,
                lr=lr,
                batch_size=batch_size,
                init_var=init_var,
                max_var=max_var,
                class_weight=class_weight)
loss_vals = run_optimizer(opt, epochs=epochs)

"""
Phantom code
"""
npy_file = 'patricia.npy'

if os.path.exists(npy_file):
    rec = np.load(npy_file).item()
else:
    rec = {}

tmp = np.mean(loss_vals)
key = (lr, batch_size, init_var, max_var)
if rec.has_key(key):
    rec[key].append(tmp)
else:
    rec[key] = [tmp]

np.save(npy_file, rec)
