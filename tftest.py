import tensorflow as tf

# tf.config.experimental.list_physical_devices()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))