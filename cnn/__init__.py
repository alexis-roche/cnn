from .image_classifier import *
from ._fastrun import (_convolve_image,
                       _multi_convolve_image,
                       _relu_max_pool_image,
                       _get_device_info,
                       _basic_test1d,
                       _gpu_convolve_image,
                       _gpu_multi_convolve_image)

                       
from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

