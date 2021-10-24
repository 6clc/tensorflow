import tensorflow as tf
from tensorflow.python.keras import Input, Model, layers
import numpy as np

# single thread
tf.config.threading.set_inter_op_parallelism_threads(
    1
)
# show op device
tf.debugging.set_log_device_placement(
    True
)
# Set up logging.
logdir = 'logs'
# writer = tf.summary.create_file_writer(logdir)

x = Input(shape=(244, 244, 3))
y = tf.keras.layers.Conv2DTranspose(3, (1, 1))(x)
model = Model(x, y)

def f():
  x = np.random.randn(1, 244, 244, 3)
  kernel = np.random.randn(1, 1, 3, 3)
  # y = tf.nn.conv2d_transpose(x, filters=kernel, output_shape=(1, 244, 244, 3), strides=[1, 1])
  y = tf.nn.conv2d(x, filters=kernel, strides=[1, 1], padding='SAME')
  return y


# graph
# tf.summary.trace_on(graph=True, profiler=True)
# out = f()
# out = model.predict_on_batch(np.random.randn(1, 244, 244, 3))
# with writer.as_default():
#   tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=logdir)
# print(out.shape)
# print(model.summary())

# profile
out = model.predict_on_batch(np.random.randn(1, 244, 244, 3))
tf.profiler.experimental.start(logdir)
out = model.predict_on_batch(np.random.randn(1, 244, 244, 3))
tf.profiler.experimental.stop()