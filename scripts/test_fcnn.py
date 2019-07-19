"""
python2.7 test_fcnn <image file> <h5 file> <device> <brute_force>
"""

import sys
import time
import numpy as np

import vii
import sys
import cnn


SAVE_IMAGES = True
GROUPS = 25, 20

fimg = 'pizza.png'
fmod = 'feb2.h5'
device = 0
brute_force = False
if len(sys.argv) > 1:
    fimg = sys.argv[1]
    if len(sys.argv) > 2:
        fmod = sys.argv[2]
        if len(sys.argv) > 3:
            device = int(sys.argv[3])
            if device < 0:
                device = None
            if len(sys.argv) > 4:
                brute_force = bool(int(sys.argv[4]))

img = vii.load_image(fimg)
classif = cnn.load_image_classifier(fmod)
data = img.get_data() / 255.


print('Starting image segmentation using FCNN...')
t0 = time.time()
pm = classif.label_map(data, device=device, groups=GROUPS)
dt = time.time() - t0
print('Time FCNN = %f' % dt)

silver = pm[..., 1]


### Test 1 ###
x = img.dims[0] // 2
y = img.dims[1] // 2
patch = img.get_data().astype(cnn.FLOAT_DTYPE)[x:(x + classif.image_size[0]), y:(y + classif.image_size[1])] / 255
gold_xy = classif.run(patch)[1]
err = silver[x + classif.image_size[0] // 2, y + classif.image_size[1] // 2] - gold_xy
print ('Error1 = %f' % err)


### Test 2 ###
if brute_force:
    print('Starting brute force patch-based image segmentation...')
    t0 = time.time()
    gold = classif.label_map(data, brute_force=True)[..., 1]
    dt_b = time.time() - t0
    print('Time brute-force: %f' % dt_b)
    a, b = classif.fcnn_borders
    diff = np.abs(silver - gold)[a:-b, a:-b]
    print('Error2 = %f' % np.max(diff))
    print('Speedup factor using FCNN = %f' % (dt_b / dt))

### Save mask image
if SAVE_IMAGES:
    vii.save_image(vii.Image(255 * silver), 'silver.png')
    if brute_force:
        vii.save_image(vii.Image(255 * gold), 'gold.png')
