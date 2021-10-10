"""Tests for tensorflow.python.client.session.Session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import os
import sys
import threading
import time
import warnings

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as framework_device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
# Import gradients to resolve circular imports
from tensorflow.python.ops import gradients  # pylint: disable=unused-import
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

with ops.Graph().as_default() as g:
  a = constant_op.constant(6.0, shape=[1, 1])
  b = constant_op.constant(7.0, shape=[1, 1])
  c = math_ops.matmul(a, b, name='matmul')

# print(g)
# with session.Session(graph=g):
#   result = c.eval()

# x = ops._NodeDef("liuchaoop", "myop")
# print(type(x))