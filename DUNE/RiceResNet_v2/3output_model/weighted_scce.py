import tensorflow as tf
from tensorflow.keras import losses

class WeightedSCCE(tf.keras.losses.Loss):
    def __init__(self, class_weight, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="weighted_scce"):
        super().__init__(reduction=reduction, name=name)  # Ensure valid reduction
        self.class_weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)  # Ensure indices are integers
        weight_mask = tf.gather(self.class_weight, y_true)  # Apply weights

        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weighted_loss = loss * weight_mask  # Apply the weight mask

        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        config = super().get_config()  # Get base class config
        config.update({"class_weight": self.class_weight.numpy()})  # Add custom attributes
        return config
