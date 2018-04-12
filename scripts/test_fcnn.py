import sys
import time
import numpy as np

import vii
import cnn

GROUPS = 25, 20

device = 0
if len(sys.argv) > 1:
    device = int(sys.argv[1])
    if device < 0:
        device = None 

img = vii.load_image('/home/alexis/artisan_data/pizza/item1/con5/pic01.png')
classif = cnn.load_image_classifier('feb2.h5')  # 'mar6.h5'
data = img.get_data() / 255.

t0 = time.time()
pm = classif.label_map(data, device=device, groups=GROUPS)
print('Time FCNN = %f' % (time.time() - t0))

silver_mask = pm[..., 1]

### Test 1
x = np.random.randint(img.dims[0] - classif.image_size[0] + 1)
y = np.random.randint(img.dims[1] - classif.image_size[1] + 1)
patch = img.get_data().astype(cnn.FLOAT_DTYPE)[x:(x + classif.image_size[0]), y:(y + classif.image_size[1])] / 255
gold = classif.run(patch)
err = silver_mask[x + classif.image_size[0] // 2, y + classif.image_size[1] // 2] - gold[1]
print ('FCNN error = %f' % err)

#gold_mask = np.load('gold_fcnn.npy')
