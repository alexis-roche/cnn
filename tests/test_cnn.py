from __future__ import absolute_import

import numpy as np

from ..design import *
from ..image_classifier import *

from nose.tools import assert_true, assert_false, assert_raises
from numpy.testing import (assert_array_equal, 
                           assert_array_almost_equal,
                           assert_almost_equal)



def test_dim_after_convolution_empty_kernel_even_dim():
    assert dim_after_convolution(50, 1) == 50

def test_dim_after_convolution_empty_kernel_odd_dim():
    assert dim_after_convolution(51, 1) == 51

def test_dim_after_convolution_even_dim():
    assert dim_after_convolution(50, 3) == 48

def test_dim_after_convolution_odd_dim():
    assert dim_after_convolution(51, 3) == 49

def test_dim_after_pooling_empty_mask():
    assert dim_after_pooling(50, 1) == 50

def test_dim_after_pooling_even_dim():
    assert dim_after_pooling(48, 2) == 24

def test_dim_after_pooling_odd_dim():
    assert dim_after_pooling(49, 2) == 24


def test_image_classifier_parameters():
    C = ImageClassifier([640, 480], 2,
                        conv_filters=[32, 32, 64],
                        kernel_size=3,
                        pool_size=2,
                        dense_units=[64,])
    assert C.image_size == (640, 480)
    assert C.conv_filters == (32, 32, 64)
    assert C.nclasses == 2
    assert C.kernel_size == 3
    assert C.pool_size == 2
    assert C.dense_units == (64,)
    assert C.x_train is None
    assert C.y_train is None
    assert C.x_test is None
    assert C.y_test is None

    
def test_image_classifier_properties():
    C = ImageClassifier((640, 480), 2)
    assert_raises(AttributeError, setattr, C, 'image_size', (256, 256))
    # more...
