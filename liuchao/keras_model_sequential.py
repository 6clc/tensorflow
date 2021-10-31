import tensorflow as tf
from tensorflow.python import keras

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(4,)))
model.add(keras.layers.Dense(4))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4))
print(model.summary())

session = keras.backend.get_session()
frozen_graph_def = tf.graph_util.convert_variables_to_constants(session, session.graph_def, [node.op.name for node in model.outputs])
# 写入到 pb文件
tf.io.write_graph(frozen_graph_def, "./pb_model", "freeze_eval_graph.pb",as_text=False) # 细化到了运算的节点