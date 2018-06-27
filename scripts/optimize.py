"""
python make_classifier 0 --> generate examples
python make_classifier 1 --> check examples
python make_classifier 2 --> train classifier

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


IMAGE_PATH = '/home/alexis/artisan_data'
SIZE = 45
PATCHES_PER_IMAGE = 1
RANDOM = True
EXAMPLE_PATH = '/home/alexis/tmp'
EXAMPLE_NAME = 'example'
MODEL_NAME = 'zob'
CHECK_BUFFER = 10

PROP_TEST = .01
BATCH_SIZE = 16
EPOCHS = 1
DROPOUT = .0


def get_food_list():
    return [os.path.split(p)[1] for p in glob.glob(os.path.join(IMAGE_PATH, '*'))]


def random_image():
    # pick a random image
    food_list = get_food_list()
    food = np.random.randint(len(food_list))
    item = 1 + np.random.randint(10)
    con = 1 + np.random.randint(8)
    pic = 1 + np.random.randint(8)
    return os.path.join(IMAGE_PATH, food_list[food], 'item%d' % item, 'con%d' % con, 'pic0%d.png' % pic)


def save_example(ps, food, path, name):
    if ps.label in (0, 2):
        print('Missing label, not saving the patch')
        return
    elif ps.label == 1:
        label = 0
    elif ps.label == 3:
        label = 1 + food
    files = glob.glob(os.path.join(path, '%s*' % name))
    fnpz = os.path.join(path, '%s%d.npz' % (name, len(files) + 1))
    print('Saving example with label %d in: %s' % (label, fnpz))
    np.savez(fnpz, patch=ps.data, label=label, food=food)


def generate_examples(image_path, size, patches_per_image, random, example_path, npz_name):
    food_list = get_food_list()
    stop = False
    while not stop:
        fimg = random_image()
        food = food_list.index(fimg.replace(os.path.normpath(image_path), '').split(os.path.sep)[1])
        for i in range(patches_per_image):
            ps = PatchSelector(fimg, size)
            ps.run(random=random)
            ps.close()
            if ps.label == 2:
                stop = True
                break
            save_example(ps, food, example_path, npz_name)


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
    

################################################
# Main
################################################


x, y = normalize(*load_examples(EXAMPLE_PATH, EXAMPLE_NAME))

classif = cnn.ImageClassifier(x.shape[1:3], y.max() + 1)
x_train, y_train, x_test, y_test = cnn.split(x, y, prop_test=PROP_TEST)
classif._configure_training(x_train, y_train, 0, 1e-4, 1e-6, 'glorot_uniform', 'zeros', x_test, y_test)

#print classif._model.get_weights()[0].mean()

"""
opt = RMSPropagation(classif)
opt.run(10)
"""
opt = AverageEP(classif, lr=1)

#opt = Optimizer(classif)
