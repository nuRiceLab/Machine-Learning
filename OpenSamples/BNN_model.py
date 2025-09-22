import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # keep tf.keras (needed with TFP 0.24 + TF 2.17)

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision

tfd  = tfp.distributions
tfpl = tfp.layers

# ─────────────────────────────────────────────────────────────────────────────
# 1) Mixed precision: convs/BN/activations run in float16 on GPU (Tensor Cores),
#    variables stay float32. We’ll keep TFP distribution layers in float32.
# ─────────────────────────────────────────────────────────────────────────────
mixed_precision.set_global_policy("mixed_float16")

# total “num_train_samples” used to scale KL; adjust to your dataset size
NUM_TRAIN_SAMPLES = 27_000


# global scale variable used by kl_div_fn
#kl_scale = tf.Variable(0.0, trainable=False, dtype=tf.float32)

#def kl_div_fn(q, p, _):
#    return kl_scale * tfd.kl_divergence(q, p) / float(NUM_TRAIN_SAMPLES)

## 2) Use analytic KL (no sampling). Faster and fully GPU-accelerated.
def kl_div_fn(q, p, _):
    return tfd.kl_divergence(q, p) / float(NUM_TRAIN_SAMPLES)




# 3) Simple, stable N(0,1) prior (Independent Normal). Works great with reparameterization.
def prior_fn(dtype, shape, name, trainable, add_variable_fn):
    del name, trainable, add_variable_fn
    # shape is a Python tuple here (OK). Reinterpret all batch dims.
    dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=tf.ones(shape, dtype))
    return tfd.Independent(dist, reinterpreted_batch_ndims=len(shape))

def conv2d_reparam(filters, kernel_size, activation=None, strides=1, padding="same", name=None, dtype=None):
    # For numerical stability, keep the *probabilistic* layer in float32 even under mixed precision.
    # That still lets its internal conv run fast on GPU; logits/params computations are FP32.
    return tfpl.Convolution2DReparameterization(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_prior_fn=prior_fn,
        kernel_divergence_fn=kl_div_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_prior_fn=prior_fn,
        bias_divergence_fn=kl_div_fn,
        name=name,
        dtype=dtype or tf.float32,   # <-- keep TFP layer compute in float32
    )

def residual_block(x, filters, name, kernel_size=3, padding="same", strides=(1,1)):
    shortcut = x

    # Main path
    x = conv2d_reparam(filters, kernel_size, activation=None, strides=strides, padding=padding,
                       name=f"{name}_conv1")(x)
    # Fused BN paths are fastest on GPU with channels_last + float16 compute
    x = layers.BatchNormalization(fused=True, momentum=0.9, epsilon=1e-5, name=f"{name}_bn1")(x)
    x = layers.Activation("relu")(x)

    x = conv2d_reparam(filters, kernel_size, activation=None, strides=1, padding=padding,
                       name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(fused=True, momentum=0.9, epsilon=1e-5, name=f"{name}_bn2")(x)

    # Shortcut path if channels change
    if shortcut.shape[-1] != filters:
        shortcut = conv2d_reparam(filters, 1, activation=None, strides=1, padding=padding,
                                  name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(fused=True, momentum=0.9, epsilon=1e-5, name=f"{name}_proj_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def plane_tower(name, inputs):
    x = conv2d_reparam(32, 7, activation="swish", strides=2, name=f"{name}_stem")(inputs)
    x = layers.BatchNormalization(fused=True, momentum=0.9, epsilon=1e-5, name=f"{name}_bn0")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    for i, f in enumerate([32, 64, 128]):
        x = residual_block(x, f, name=f"{name}_res{i}")
    return layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)

def bayes_three_tower(input_shape=(500,500,3), num_classes=3):
    # split channels into 3 inputs (or pre-split before feeding)
    inp = layers.Input(shape=input_shape, name="inputs")        # (H,W,3)
    planes = [layers.Lambda(lambda t: t[...,i:i+1], name=f"split_p{i}")(inp) for i in range(3)]
    # replicate single-channel to 3 for conv kernels, or change first conv to 1-channel
    planes = [layers.Concatenate()([p,p,p]) for p in planes]    # now (H,W,3) each

    f0 = plane_tower("p0", planes[0])
    f1 = plane_tower("p1", planes[1])
    f2 = plane_tower("p2", planes[2])
    x = layers.Concatenate(name="concat_planes")([f0, f1, f2])

    params_units = tfpl.OneHotCategorical.params_size(num_classes)
    x = tfpl.DenseReparameterization(
        units=params_units,
        activation=None,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_divergence_fn=kl_div_fn,
        name="head_dense",
        dtype=tf.float32,
    )(x)
    out = tfpl.OneHotCategorical(event_size=num_classes, name="output", dtype=tf.float32)(x)
    return models.Model(inp, out, name="BNN_three_tower")

def bayes_model(input_shape=(200, 200, 3), num_classes=3, num_components=1):
    inputs = layers.Input(shape=input_shape, name="inputs")

    # Stem (keep TFP conv in FP32; still runs on GPU; mixed precision applies around it)
    x = conv2d_reparam(32, 7, activation="swish", strides=2, name="stem_conv")(inputs)
    x = layers.BatchNormalization(fused=True, momentum=0.9, epsilon=1e-5, name="stem_bn")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Residual stack
    for i, f in enumerate([32, 64, 128]):
        x = residual_block(x, f, name=f"res{i}")

    x = layers.GlobalAveragePooling2D()(x)  # often a bit smoother than MaxPool for BNNs

    # replace the head:
    params_units = tfpl.OneHotCategorical.params_size(num_classes)

    x = tfpl.DenseReparameterization(
        units=params_units,
        activation=None,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_divergence_fn=kl_div_fn,
        name="head_dense",
        dtype=tf.float32,
        )(x)

    out = tfpl.OneHotCategorical(
        event_size=num_classes,
        name="output",
        dtype=tf.float32,
    )(x)

    model = models.Model(inputs, out, name="BNN")
    return model
