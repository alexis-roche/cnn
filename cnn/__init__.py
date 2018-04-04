from ._utils import (FLOAT_DTYPE,
                     _convolve_image,
                     _multi_convolve_image,
                     _relu_max_pool_image,
                     _get_opencl_device_info,
                     _opencl_test1d,
                     _opencl_convolve_image,
                     _opencl_multi_convolve_image)
from .image_classifier import *

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

