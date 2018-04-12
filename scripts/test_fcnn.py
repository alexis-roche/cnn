import sys
import time
import numpy as np

import vii
import sys
import cnn


GROUPS = 25, 20

fimg = '/home/alexis/artisan_data/pizza/item1/con5/pic01.png'
device = 0
brute_force = False
if len(sys.argv) > 1:
    fimg = sys.argv[1]
    if len(sys.argv) > 2:
        device = int(sys.argv[2])
        if device < 0:
            device = None
        if len(sys.argv) > 3:
            brute_force = bool(int(sys.argv[3]))

img = vii.load_image(fimg)
classif = cnn.load_image_classifier('feb2.h5')  # 'mar6.h5'
data = img.get_data() / 255.

print('Starting image segmentation using FCNN...')
t0 = time.time()
pm = classif.label_map(data, device=device, groups=GROUPS)
dt = time.time() - t0
print('Time FCNN = %f' % dt)

silver_mask = pm[..., 1]

### Test 1
x = img.dims[0] / 2
y = img.dims[1] / 2
patch = img.get_data().astype(cnn.FLOAT_DTYPE)[x:(x + classif.image_size[0]), y:(y + classif.image_size[1])] / 255
gold = classif.run(patch)[1]
err = silver_mask[x + classif.image_size[0] // 2, y + classif.image_size[1] // 2] - gold
print ('Error1 = %f' % err)


### Test 2
if brute_force:
    print('Starting brute force patch-based image segmentation...')
    t0 = time.time()
    gold_mask = classif.label_map(data, brute_force=True)[..., 1]
    dt_b = time.time() - t0
    print('Time brute-force: %f' % dt_b)
    a, b = classif.fcnn_borders
    diff = np.abs(silver_mask - gold_mask)[a:-b, a:-b]
    print('Error2 = %f' % np.max(diff))
    print('Speedup factor using FCNN = %f' % (dt_b / dt))
